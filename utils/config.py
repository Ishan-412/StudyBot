"""
utils/config.py

Central configuration for StudyBot.
All tuneable constants live here so every module can import from one place.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# ── Load .env ──────────────────────────────────────────────────────────────
load_dotenv()

# ── Gemini ──────────────────────────────────────────────────────────────────
GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL: str = "gemini-1.5-flash"          # fast & cheap for student use

# ── Embedding model ─────────────────────────────────────────────────────────
# A lightweight local model; no API key required.
EMBEDDING_MODEL_NAME: str = "sentence-transformers/all-MiniLM-L6-v2"

# ── Text chunking ────────────────────────────────────────────────────────────
CHUNK_SIZE: int = 800        # characters per chunk
CHUNK_OVERLAP: int = 150     # overlap between consecutive chunks

# ── Retrieval ────────────────────────────────────────────────────────────────
TOP_K: int = 5               # number of chunks returned per query

# ── Paths ─────────────────────────────────────────────────────────────────
# Root of the project (one level up from utils/)
PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent

# All FAISS indexes and uploaded PDFs are stored here (git-ignored)
DATA_DIR: Path = PROJECT_ROOT / "data"
UPLOADS_DIR: Path = PROJECT_ROOT / "uploads"

DATA_DIR.mkdir(parents=True, exist_ok=True)
UPLOADS_DIR.mkdir(parents=True, exist_ok=True)


def get_subject_data_dir(subject: str) -> Path:
    """Return (and create) the data directory for a given subject."""
    path = DATA_DIR / _sanitize(subject)
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_subject_upload_dir(subject: str) -> Path:
    """Return (and create) the upload directory for a given subject."""
    path = UPLOADS_DIR / _sanitize(subject)
    path.mkdir(parents=True, exist_ok=True)
    return path


def _sanitize(name: str) -> str:
    """Convert a subject name to a safe directory name."""
    return "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in name)
