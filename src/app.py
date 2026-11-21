import chromadb
import gradio as gr
from transformers import pipeline
from src.config import GENERATOR_MODEL
from src.chunker import load_and_chunk_pdf
from src.indexer import get_vector_collection, index_chunks

# Initialize dependencies
client = chromadb.Client()
collection = get_vector_collection(client)

# Pre-index target file if it exists
try:
    chunks = load_and_chunk_pdf("filename.pdf") #add your filename with the pdf extention here
    index_chunks(collection, chunks)
    print(f"Loaded {len(chunks)} searchable chunks into indexing storage!")
except Exception as e:
    print(f"Warning: Pre-loaded PDF not found. Please verify filepath: {e}")

qa_generator = pipeline("text-generation", model=GENERATOR_MODEL, max_new_tokens=200, device_map="auto")

def chat_with_notes(question, history):
    results = collection.query(query_texts=[question], n_results=3)
    docs = results["documents"][0]
    metas = results["metadatas"][0]
    distances = results["distances"][0]

    if not docs or distances[0] > 1.35:
        return "I couldn't find anything about that in your uploaded notes."

    context_blocks = [f"[Page {meta['page']}]: {doc}" for doc, meta in zip(docs, metas)]
    context_str = "\n\n".join(context_blocks)

    messages = [
        {
            "role": "system",
            "content": f"Answer the question using ONLY the context provided below. If the answer is not mentioned in the context, reply with 'Not mentioned in the notes.'\n\nContext:\n{context_str}"
        },
        {
            "role": "user",
            "content": question
        }
    ]

    response = qa_generator(messages)[0]["generated_text"]
    answer = response[-1]["content"].strip()
    citations = ", ".join(sorted(set(str(meta["page"]) for meta in metas)))
    return f"{answer}\n\n**Sources:** Page(s) {citations}"

if __name__ == "__main__":
    demo = gr.ChatInterface(
        fn=chat_with_notes,
        title="Chat With Your Notes (Intro to RAG)",
        description="Ask questions grounded strictly in your uploaded document.",
    )
    demo.launch()
