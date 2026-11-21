import chromadb
from chromadb.utils import embedding_functions
from src.config import EMBEDDING_MODEL, COLLECTION_NAME

def get_vector_collection(client: chromadb.Client):
    embed_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=EMBEDDING_MODEL
    )
    return client.get_or_create_collection(name=COLLECTION_NAME, embedding_function=embed_fn)

def index_chunks(collection, chunks: list):
    collection.upsert(
        ids=[c["id"] for c in chunks],
        documents=[c["text"] for c in chunks],
        metadatas=[{"page": c["page"]} for c in chunks]
    )
