"""Settings shared by databaseCreate.py, createSnapshot.py and app.py."""

import os
from qdrant_client import QdrantClient

COLLECTION_NAME = "questions"
EMBEDDING_MODEL = "BAAI/bge-base-en-v1.5"
VECTOR_SIZE = 768

# BGE models are trained with this instruction on the *query* side only.
# Documents are embedded without it.
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

QDRANT_URL = os.environ.get("QDRANT_URL", "http://localhost:6333")
# Set QDRANT_PATH to run Qdrant embedded in-process (no Docker needed),
# e.g. QDRANT_PATH=qdrant_local python databaseCreate.py
QDRANT_PATH = os.environ.get("QDRANT_PATH")


def get_client():
    if QDRANT_PATH:
        return QdrantClient(path=QDRANT_PATH)
    return QdrantClient(url=QDRANT_URL)


def embedding_text(course_name, question_text):
    """Text that gets embedded for each question (documents side)."""
    return f"{course_name or ''}: {question_text}".strip(": ")
