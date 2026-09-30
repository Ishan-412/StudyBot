"""
features/summary.py

Summary generation pipeline:
  Topic / keyword
      ↓
  Retrieve top-K chunks from subject's FAISS index
      ↓
  Build context string
      ↓
  Summary prompt  →  Gemini  →  structured summary
"""

from retrieval.retriever import retrieve_chunks, RetrievalError
from llm.gemini import generate, GeminiError
from llm.prompts import summary_prompt, build_context_string
from utils.config import TOP_K


def generate_summary(subject: str, topic: str) -> dict:
    """
    Retrieve content about *topic* and ask Gemini to summarise it.

    Args:
        subject: The currently selected subject.
        topic:   A keyword or short phrase (e.g. "neural networks",
                 "TCP/IP", or just the document name).

    Returns:
        {
            "summary": str,
            "sources": list[str],
            "error":   str | None,
        }
    """
    result = {"summary": "", "sources": [], "error": None}

    # Retrieve more chunks for summaries (2× normal TOP_K for broader coverage)
    k = min(TOP_K * 2, 20)

    try:
        docs = retrieve_chunks(subject, topic, k=k)
    except RetrievalError as exc:
        result["error"] = str(exc)
        return result

    context = build_context_string(docs)
    prompt  = summary_prompt(context, topic)

    try:
        summary = generate(prompt)
    except GeminiError as exc:
        result["error"] = str(exc)
        return result

    from retrieval.retriever import format_sources
    result["summary"] = summary
    result["sources"] = format_sources(docs)
    return result
