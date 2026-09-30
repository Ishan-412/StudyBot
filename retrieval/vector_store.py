"""
retrieval/vector_store.py

Manages one FAISS index per subject.

Key design decisions:
- Each subject gets its own folder under data/<subject>/
- The index is saved to disk after every new batch of documents.
- When new PDFs are added the existing index is loaded and *merged* rather
  than rebuilt from scratch, so previously indexed files are not lost.
- Only LangChain's FAISS wrapper is used; no raw faiss calls are needed.
"""

from pathlib import Path
from langchain_community.vectorstores import FAISS
from langchain.schema import Document
from retrieval.embeddings import get_embedding_model
from utils.config import get_subject_data_dir


FAISS_INDEX_DIR_NAME = "faiss_index"


class VectorStoreError(Exception):
    """Raised when a FAISS operation fails."""


# ── Public API ────────────────────────────────────────────────────────────────

def add_documents(subject: str, documents: list[Document]) -> None:
    """
    Add *documents* to the subject's FAISS index.

    If an index already exists on disk it is loaded first so that new
    documents are *merged* with existing ones rather than overwriting them.

    Args:
        subject:   The subject name (e.g. "Machine Learning").
        documents: LangChain Document objects with page_content + metadata.

    Raises:
        VectorStoreError: if documents list is empty.
    """
    if not documents:
        raise VectorStoreError("No documents provided for indexing.")

    embeddings = get_embedding_model()
    index_path = _index_path(subject)

    if index_path.exists():
        # Load existing index and merge new documents into it.
        existing = FAISS.load_local(
            str(index_path),
            embeddings,
            allow_dangerous_deserialization=True,
        )
        existing.add_documents(documents)
        existing.save_local(str(index_path))
    else:
        # First time: create a new index from scratch.
        vector_store = FAISS.from_documents(documents, embeddings)
        vector_store.save_local(str(index_path))


def load_index(subject: str) -> FAISS:
    """
    Load the FAISS index for *subject* from disk.

    Returns:
        A LangChain FAISS vector store ready for similarity search.

    Raises:
        VectorStoreError: if no index has been built yet for this subject.
    """
    index_path = _index_path(subject)
    if not index_path.exists():
        raise VectorStoreError(
            f"No index found for subject '{subject}'. "
            "Please upload and index documents first."
        )
    embeddings = get_embedding_model()
    return FAISS.load_local(
        str(index_path),
        embeddings,
        allow_dangerous_deserialization=True,
    )


def index_exists(subject: str) -> bool:
    """Return True if a FAISS index already exists for *subject*."""
    return _index_path(subject).exists()


# ── Private helpers ───────────────────────────────────────────────────────────

def _index_path(subject: str) -> Path:
    """Return the directory path where the FAISS index for *subject* lives."""
    return get_subject_data_dir(subject) / FAISS_INDEX_DIR_NAME
