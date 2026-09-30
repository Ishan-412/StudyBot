"""
retrieval/retriever.py

Given a user query and a subject, retrieves the top-K most relevant document
chunks from the subject's FAISS index.

The retriever is the bridge between the vector store and the LLM features.
"""

from langchain.schema import Document
from retrieval.vector_store import load_index, VectorStoreError
from utils.config import TOP_K


class RetrievalError(Exception):
    """Raised when retrieval fails or returns no results."""


def retrieve_chunks(subject: str, query: str, k: int = TOP_K) -> list[Document]:
    """
    Search the FAISS index of *subject* for the most relevant chunks.

    Steps:
    1. Load the FAISS index from disk (fast — index is memory-mapped).
    2. Embed the query using the same model used at indexing time.
    3. Run cosine similarity search and return top-k Documents.

    Args:
        subject: Subject to search within.
        query:   The user's natural-language question.
        k:       Number of chunks to retrieve (default: TOP_K from config).

    Returns:
        List of LangChain Document objects, ranked by similarity.

    Raises:
        RetrievalError: if no index exists or no results are found.
    """
    try:
        vector_store = load_index(subject)
    except VectorStoreError as exc:
        raise RetrievalError(str(exc)) from exc

    results: list[Document] = vector_store.similarity_search(query, k=k)

    if not results:
        raise RetrievalError(
            "No relevant content found in the uploaded study material. "
            "Try rephrasing your question or upload more documents."
        )

    return results


def format_sources(docs: list[Document]) -> list[str]:
    """
    Deduplicate and format source references for display.

    Returns a list of human-readable strings, e.g.:
    ["Data_Preprocessing.pdf — Page 19", "Intro_to_ML.pdf — Page 3"]
    """
    seen: set[tuple] = set()
    sources: list[str] = []

    for doc in docs:
        meta  = doc.metadata
        key   = (meta.get("source", "Unknown"), meta.get("page", "?"))
        if key not in seen:
            seen.add(key)
            sources.append(f"{key[0]}  —  Page {key[1]}")

    return sources
