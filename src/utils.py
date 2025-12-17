"""Utility helpers for the document extraction pipeline."""

from __future__ import annotations

import base64
import mimetypes
import os
from typing import Any, Dict, Tuple


def encode_file_to_base64(file_path: str) -> Tuple[str, str]:
    """Read a file and return (base64_string, mime_type)."""
    with open(file_path, "rb") as f:
        raw = f.read()

    mime_type, _ = mimetypes.guess_type(file_path)
    if not mime_type:
        # Reasonable defaults; many of our inputs are PNG or PDF.
        ext = os.path.splitext(file_path)[1].lower()
        if ext == ".pdf":
            mime_type = "application/pdf"
        elif ext in {".jpg", ".jpeg"}:
            mime_type = "image/jpeg"
        elif ext == ".png":
            mime_type = "image/png"
        else:
            mime_type = "application/octet-stream"

    b64 = base64.b64encode(raw).decode("utf-8")
    return b64, mime_type


def is_pdf(file_path: str) -> bool:
    return os.path.splitext(file_path)[1].lower() == ".pdf"


def get_file_info(file_path: str) -> Dict[str, Any]:
    """Return basic file metadata used by the pipeline."""
    size_bytes = os.path.getsize(file_path)
    name = os.path.basename(file_path)
    ext = os.path.splitext(name)[1].lower().lstrip(".")

    return {
        "path": file_path,
        "name": name,
        "extension": ext,
        "size_bytes": size_bytes,
        "size_mb": round(size_bytes / (1024 * 1024), 3),
    }
