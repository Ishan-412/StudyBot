"""
ingestion/chunker.py

Splits page-level text into smaller, overlapping chunks using LangChain's
RecursiveCharacterTextSplitter.  Every chunk carries its source page's
metadata so retrieval results always know where a chunk came from.
"""

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from utils.config import CHUNK_SIZE, CHUNK_OVERLAP


def chunk_pages(pages: list[dict]) -> list[Document]:
    """
    Split a list of page dicts (from pdf_loader) into LangChain Documents.

    Args:
        pages: Output of pdf_loader.load_pdf() — list of dicts with keys
               "text", "page", "source", "subject".

    Returns:
        A list of LangChain Document objects where:
        - doc.page_content = chunk text
        - doc.metadata     = {"subject": ..., "source": ..., "page": ...}
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        # Try to split on paragraph, sentence, then word boundaries.
        separators=["\n\n", "\n", ". ", " ", ""],
        length_function=len,
    )

    documents: list[Document] = []

    for page in pages:
        # Each page may produce 1-N chunks depending on its length.
        chunks = splitter.split_text(page["text"])

        for chunk_text in chunks:
            if chunk_text.strip():      # skip empty strings
                documents.append(
                    Document(
                        page_content=chunk_text,
                        metadata={
                            "subject": page["subject"],
                            "source":  page["source"],
                            "page":    page["page"],
                        },
                    )
                )

    return documents
