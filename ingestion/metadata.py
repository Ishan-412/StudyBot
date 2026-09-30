"""
ingestion/metadata.py

Helpers for reading and writing a lightweight JSON "manifest" that tracks
which PDFs have already been indexed for each subject.  This lets us skip
re-indexing a file that has not changed.
"""

import json
from pathlib import Path
from utils.config import get_subject_data_dir


MANIFEST_FILENAME = "indexed_files.json"


def load_manifest(subject: str) -> dict:
    """
    Load the manifest for a subject.

    Returns a dict like:
    {
        "Data_Preprocessing.pdf": {"pages": 22, "chunks": 45},
        ...
    }
    An empty dict is returned if no manifest exists yet.
    """
    manifest_path = _manifest_path(subject)
    if not manifest_path.exists():
        return {}
    try:
        with open(manifest_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return {}


def save_manifest(subject: str, manifest: dict) -> None:
    """Persist the manifest dict to disk."""
    manifest_path = _manifest_path(subject)
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)


def is_already_indexed(subject: str, filename: str) -> bool:
    """Return True if *filename* is already recorded in the manifest."""
    return filename in load_manifest(subject)


def record_indexed_file(
    subject: str,
    filename: str,
    pages: int,
    chunks: int,
) -> None:
    """Add or update a file entry in the subject's manifest."""
    manifest = load_manifest(subject)
    manifest[filename] = {"pages": pages, "chunks": chunks}
    save_manifest(subject, manifest)


def list_indexed_files(subject: str) -> list[str]:
    """Return sorted list of filenames that have been indexed for a subject."""
    return sorted(load_manifest(subject).keys())


def _manifest_path(subject: str) -> Path:
    return get_subject_data_dir(subject) / MANIFEST_FILENAME
