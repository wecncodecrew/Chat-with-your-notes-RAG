import os

DB_PATH = os.getenv("CHROMA_DB_PATH", "./chroma_db")
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
GENERATOR_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
COLLECTION_NAME = "my_notes"
