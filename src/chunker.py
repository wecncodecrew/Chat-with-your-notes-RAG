from pypdf import PdfReader

def load_and_chunk_pdf(pdf_path: str, chunk_size: int = 500, overlap: int = 100) -> list:
    reader = PdfReader(pdf_path)
    chunks = []
    for page_num, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        start = 0
        while start < len(text):
            end = start + chunk_size
            chunk_text = text[start:end].strip()
            if chunk_text:
                chunks.append({
                    "id": f"p{page_num}_c{start}",
                    "text": chunk_text,
                    "page": page_num
                })
            start += chunk_size - overlap
    return chunks
