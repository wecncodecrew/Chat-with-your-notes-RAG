import pytest
import chromadb
from src.indexer import get_vector_collection, index_chunks

def test_database_and_upsert():
    client = chromadb.Client()
    collection = get_vector_collection(client)
    
    assert collection is not None
    
    mock_chunks = [
        {"id": "p1_c0", "text": "This is dummy test text.", "page": 1},
        {"id": "p2_c0", "text": "This is another document page.", "page": 2}
    ]
    
    index_chunks(collection, mock_chunks)
    assert collection.count() == 2

    results = collection.query(query_texts=["dummy"], n_results=1)
    assert "dummy" in results["documents"][0][0]
