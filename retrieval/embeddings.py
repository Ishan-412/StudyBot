"""
retrieval/embeddings.py

Wraps sentence-transformers so the rest of the codebase has one place to
call for producing embeddings.  We use a LangChain-compatible class so it
plugs directly into FAISS helpers.
"""

from functools import lru_cache
from langchain_huggingface import HuggingFaceEmbeddings
from utils.config import EMBEDDING_MODEL_NAME


@lru_cache(maxsize=1)
def get_embedding_model() -> HuggingFaceEmbeddings:
    """
    Load and cache the sentence-transformer embedding model.

    The model is downloaded once on first call and cached in memory for the
    lifetime of the process (lru_cache ensures we never load it twice).

    Returns:
        A LangChain HuggingFaceEmbeddings instance compatible with FAISS.
    """
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        model_kwargs={"device": "cpu"},   # change to "cuda" if GPU is available
        encode_kwargs={"normalize_embeddings": True},
    )
