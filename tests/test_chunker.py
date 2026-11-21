import pytest
from src.chunker import load_and_chunk_pdf

def test_load_and_chunk_pdf_structure():
    try:
        chunks = load_and_chunk_pdf("filename.pdf") #add your filename with the .pdf extension here
        assert isinstance(chunks, list)
        if len(chunks) > 0:
            first_chunk = chunks[0]
            assert "id" in first_chunk
            assert "text" in first_chunk
            assert "page" in first_chunk
            assert isinstance(first_chunk["page"], int)
    except FileNotFoundError:
        pytest.skip("Target test PDF file not available")
