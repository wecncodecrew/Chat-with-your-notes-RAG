import chromadb
from chromadb.utils import embedding_functions
from pypdf import PdfReader
from transformers import pipeline


class RAGEngine:
    def __init__(
        self,
        embedding_model: str = "all-MiniLM-L6-v2",
        llm_model: str = "Qwen/Qwen2.5-1.5B-Instruct",
        collection_name: str = "my_notes",
    ):
        self.embed_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=embedding_model
        )
        self.client = chromadb.Client()
        self.collection_name = collection_name
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name, embedding_function=self.embed_fn
        )

        print(f"Loading LLM ({llm_model})... This may take a moment on first run.")
        self.qa_generator = pipeline(
            "text-generation",
            model=llm_model,
            max_new_tokens=250,
            device_map="auto",
        )

    def load_and_chunk_pdf(
        self, pdf_path: str, chunk_size: int = 500, overlap: int = 100
    ) -> list[dict]:
        """Extracts text from a PDF and splits it into overlapping chunks."""
        reader = PdfReader(pdf_path)
        chunks = []
        for page_num, page in enumerate(reader.pages, start=1):
            text = page.extract_text() or ""
            start = 0
            while start < len(text):
                end = start + chunk_size
                chunk_text = text[start:end].strip()
                if chunk_text:
                    chunks.append(
                        {
                            "id": f"p{page_num}_c{start}",
                            "text": chunk_text,
                            "page": page_num,
                        }
                    )
                start += chunk_size - overlap
        return chunks

    def ingest_pdf(self, pdf_path: str) -> str:
        """Clears existing notes and indexes a new PDF into ChromaDB."""
        # Reset collection so previous PDFs don't bleed into new queries
        self.client.delete_collection(name=self.collection_name)
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name, embedding_function=self.embed_fn
        )

        chunks = self.load_and_chunk_pdf(pdf_path)
        if not chunks:
            return "Warning: No extractable text found in the uploaded PDF."

        self.collection.upsert(
            ids=[c["id"] for c in chunks],
            documents=[c["text"] for c in chunks],
            metadatas=[{"page": c["page"]} for c in chunks],
        )
        return f"Successfully indexed {len(chunks)} chunks from the PDF."

    def query(self, question: str, n_results: int = 3, distance_threshold: float = 1.35) -> str:
        """Retrieves relevant chunks and generates a strictly grounded answer."""
        if self.collection.count() == 0:
            return "Please upload and index a PDF document first."

        results = self.collection.query(query_texts=[question], n_results=n_results)
        docs = results["documents"][0]
        metas = results["metadatas"][0]
        distances = results["distances"][0]

        # Guardrail 1: Refuse if closest semantic match exceeds distance threshold
        if not docs or distances[0] > distance_threshold:
            return "I couldn't find anything about that in your uploaded notes."

        # Build open-book context with page numbers
        context_blocks = []
        pages_used = set()
        for doc, meta in zip(docs, metas):
            context_blocks.append(f"[Page {meta['page']}]: {doc}")
            pages_used.add(str(meta["page"]))

        context_str = "\n\n".join(context_blocks)

        messages = [
            {
                "role": "system",
                "content": (
                    "Answer the question using ONLY the context provided below. "
                    "If the answer is not mentioned in the context, reply with "
                    "'Not mentioned in the notes.'\n\n"
                    f"Context:\n{context_str}"
                ),
            },
            {"role": "user", "content": question},
        ]

        response = self.qa_generator(messages)[0]["generated_text"]
        answer = response[-1]["content"].strip()

        citations = ", ".join(sorted(pages_used, key=int))
        return f"{answer}\n\n**Sources:** Page(s) {citations}"
