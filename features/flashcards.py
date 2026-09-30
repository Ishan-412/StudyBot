"""
features/flashcards.py

Flashcard generation pipeline:
  Topic
      ↓
  Retrieve top-K chunks
      ↓
  Flashcard prompt  →  Gemini  →  raw text
      ↓
  Parse into list[{"question": ..., "answer": ...}]
"""

import re
from retrieval.retriever import retrieve_chunks, format_sources, RetrievalError
from llm.gemini import generate, GeminiError
from llm.prompts import flashcard_prompt, build_context_string
from utils.config import TOP_K


def generate_flashcards(subject: str, topic: str) -> dict:
    """
    Generate Q&A flashcards about *topic* from the subject's study material.

    Args:
        subject: The currently selected subject.
        topic:   Keyword or short phrase describing what to make cards about.

    Returns:
        {
            "flashcards": list[{"question": str, "answer": str}],
            "sources":    list[str],
            "error":      str | None,
        }
    """
    result = {"flashcards": [], "sources": [], "error": None}

    k = min(TOP_K * 2, 20)

    try:
        docs = retrieve_chunks(subject, topic, k=k)
    except RetrievalError as exc:
        result["error"] = str(exc)
        return result

    context = build_context_string(docs)
    prompt  = flashcard_prompt(context, topic)

    try:
        raw_text = generate(prompt)
    except GeminiError as exc:
        result["error"] = str(exc)
        return result

    result["flashcards"] = _parse_flashcards(raw_text)
    result["sources"]    = format_sources(docs)

    if not result["flashcards"]:
        result["error"] = (
            "Could not parse flashcards from the model response. "
            "Try a different topic or check the raw output below."
        )

    return result


# ── Parser ────────────────────────────────────────────────────────────────────

def _parse_flashcards(raw: str) -> list[dict]:
    """
    Parse the structured flashcard output from Gemini.

    Expected format (repeated):
    ---
    Q: <question text>
    A: <answer text>
    ---

    Returns a list of {"question": ..., "answer": ...} dicts.
    """
    cards: list[dict] = []

    # Split on the --- separator (handles variations in whitespace)
    blocks = re.split(r"-{3,}", raw)

    for block in blocks:
        block = block.strip()
        if not block:
            continue

        q_match = re.search(r"Q:\s*(.+?)(?=\nA:|\Z)", block, re.DOTALL | re.IGNORECASE)
        a_match = re.search(r"A:\s*(.+)",              block, re.DOTALL | re.IGNORECASE)

        if q_match and a_match:
            cards.append(
                {
                    "question": q_match.group(1).strip(),
                    "answer":   a_match.group(1).strip(),
                }
            )

    return cards
