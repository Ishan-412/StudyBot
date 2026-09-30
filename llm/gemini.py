"""
llm/gemini.py

Thin wrapper around Google's Generative AI SDK.

Why a wrapper?
- Centralises API key loading and model initialisation.
- Single place to swap Gemini model version or add retry logic.
- Keeps feature modules (qa.py, summary.py, flashcards.py) free of SDK
  boilerplate.
"""

import google.generativeai as genai
from utils.config import GEMINI_API_KEY, GEMINI_MODEL


class GeminiError(Exception):
    """Raised when the Gemini API call fails."""


def _get_model() -> genai.GenerativeModel:
    """
    Configure the Gemini SDK and return a GenerativeModel instance.

    Raises:
        GeminiError: if the API key is missing.
    """
    if not GEMINI_API_KEY:
        raise GeminiError(
            "GEMINI_API_KEY is not set. "
            "Please add it to your .env file and restart the app."
        )
    genai.configure(api_key=GEMINI_API_KEY)
    return genai.GenerativeModel(GEMINI_MODEL)


def generate(prompt: str) -> str:
    """
    Send *prompt* to Gemini and return the text response.

    Args:
        prompt: The fully constructed prompt string.

    Returns:
        The model's text response as a string.

    Raises:
        GeminiError: on API or configuration failure.
    """
    try:
        model    = _get_model()
        response = model.generate_content(prompt)
        # response.text raises if the model returned no content (e.g. safety block)
        return response.text.strip()
    except GeminiError:
        raise                       # re-raise our own errors unchanged
    except Exception as exc:
        raise GeminiError(
            f"Gemini API call failed: {exc}\n"
            "Check your API key and internet connection."
        ) from exc
