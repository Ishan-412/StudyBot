"""
features/qa.py

Question-Answering pipeline:
  User Question
      ↓
  Embed question (all-MiniLM-L6-v2)
      ↓
  FAISS similarity search  →  top-K chunks
      ↓
  Build context string
      ↓
  QA prompt  →  Gemini  →  answer text
      ↓
  Return (answer, sources)
"""

from langchain.schema import Document
from retrieval.retriever import retrieve_chunks, format_sources, RetrievalError
from llm.gemini import generate, GeminiError
from llm.prompts import qa_prompt, build_context_string


def answer_question(subject: str, question: str) -> dict:
    """
    Full RAG pipeline for a student's question.

    Args:
        subject:  The currently selected subject.
        question: The student's natural-language question.

    Returns:
        {
            "answer":  str,          # Gemini's answer
            "sources": list[str],    # deduplicated source references
            "error":   str | None,   # non-None if something went wrong
        }
    """
    result = {"answer": "", "sources": [], "error": None}

    # ── Step 1: Retrieve relevant chunks ─────────────────────────────────────
    try:
        docs: list[Document] = retrieve_chunks(subject, question)
    except RetrievalError as exc:
        result["error"] = str(exc)
        return result

    # ── Step 2: Build context and prompt ─────────────────────────────────────
    context = build_context_string(docs)
    prompt  = qa_prompt(context, question)

    # ── Step 3: Call Gemini ───────────────────────────────────────────────────
    try:
        answer = generate(prompt)
    except GeminiError as exc:
        result["error"] = str(exc)
        return result

    # ── Step 4: Package result ────────────────────────────────────────────────
    result["answer"]  = answer
    result["sources"] = format_sources(docs)
    return result
