"""
ingestion/pdf_loader.py

Extracts text from PDF files page by page using PyMuPDF (fitz).
Returns a list of page dictionaries with raw text and page metadata.
"""

from pathlib import Path
from typing import Optional
import fitz  # PyMuPDF


class PDFLoadError(Exception):
    """Raised when a PDF cannot be loaded or contains no text."""


def load_pdf(pdf_path: str | Path, subject: str) -> list[dict]:
    """
    Extract text from a PDF file, one dict per page.

    Args:
        pdf_path: Path to the PDF file on disk.
        subject:  The subject this PDF belongs to (stored in metadata).

    Returns:
        A list of dicts, one per page:
        [
            {
                "text":    "<page text>",
                "page":    <1-indexed page number>,
                "source":  "<filename>",
                "subject": "<subject>",
            },
            ...
        ]

    Raises:
        PDFLoadError: if the file doesn't exist, can't be opened,
                      or contains no extractable text.
    """
    pdf_path = Path(pdf_path)

    if not pdf_path.exists():
        raise PDFLoadError(f"File not found: {pdf_path}")

    try:
        doc = fitz.open(str(pdf_path))
    except Exception as exc:
        raise PDFLoadError(f"Could not open PDF '{pdf_path.name}': {exc}") from exc

    if doc.page_count == 0:
        raise PDFLoadError(f"'{pdf_path.name}' has no pages.")

    pages: list[dict] = []
    for page_num in range(doc.page_count):
        page = doc[page_num]
        raw_text = page.get_text("text")          # plain text extraction
        cleaned  = _clean_text(raw_text)

        if cleaned:                                # skip truly blank pages
            pages.append(
                {
                    "text":    cleaned,
                    "page":    page_num + 1,       # 1-indexed for humans
                    "source":  pdf_path.name,
                    "subject": subject,
                }
            )

    doc.close()

    if not pages:
        raise PDFLoadError(
            f"'{pdf_path.name}' contains no extractable text. "
            "It may be a scanned image-only PDF."
        )

    return pages


def _clean_text(text: str) -> str:
    """
    Light-weight text cleaning:
    - Collapse multiple blank lines to one blank line.
    - Strip leading/trailing whitespace from each line.
    - Remove lines that are purely whitespace.
    """
    lines = text.splitlines()
    cleaned_lines: list[str] = []
    blank_streak = 0

    for line in lines:
        stripped = line.strip()
        if stripped:
            blank_streak = 0
            cleaned_lines.append(stripped)
        else:
            blank_streak += 1
            if blank_streak == 1:               # allow at most one blank line
                cleaned_lines.append("")

    return "\n".join(cleaned_lines).strip()
