"""
document_processor.py — Parse and chunk PDF, DOCX, TXT documents.
"""
from __future__ import annotations
import re
from pathlib import Path
from typing import Any

from config import settings


def _chunk_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    """
    Split text into overlapping chunks by character count
    (proxy for token count at ~4 chars/token).
    """
    char_size = chunk_size * 4
    char_overlap = overlap * 4

    chunks: list[str] = []
    start = 0
    text = re.sub(r"\s+", " ", text).strip()

    while start < len(text):
        end = min(start + char_size, len(text))
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= len(text):
            break
        start += char_size - char_overlap

    return chunks


def _extract_pdf(path: Path) -> str:
    """Extract text from a PDF using PyMuPDF."""
    try:
        import fitz  # PyMuPDF
        doc = fitz.open(str(path))
        pages = [page.get_text("text") for page in doc]
        doc.close()
        return "\n".join(pages)
    except ImportError:
        raise RuntimeError("PyMuPDF not installed. Run: pip install pymupdf")


def _extract_docx(path: Path) -> str:
    """Extract text from a DOCX file."""
    try:
        from docx import Document
        doc = Document(str(path))
        paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
        return "\n".join(paragraphs)
    except ImportError:
        raise RuntimeError("python-docx not installed. Run: pip install python-docx")


def _extract_txt(path: Path) -> str:
    """Read plain text files, trying common encodings."""
    for encoding in ("utf-8", "utf-16", "latin-1"):
        try:
            return path.read_text(encoding=encoding)
        except (UnicodeDecodeError, ValueError):
            continue
    raise ValueError(f"Cannot decode file: {path.name}")


def process_document(file_path: Path) -> list[dict[str, Any]]:
    """
    Parse a document and return a list of chunk dicts:
    [{text, metadata: {filename, file_type, chunk_index}}]
    """
    suffix = file_path.suffix.lower()

    extractors = {
        ".pdf": _extract_pdf,
        ".docx": _extract_docx,
        ".doc": _extract_docx,
        ".txt": _extract_txt,
        ".md": _extract_txt,
        ".csv": _extract_txt,
    }

    if suffix not in extractors:
        raise ValueError(
            f"Unsupported file type: '{suffix}'. "
            f"Supported: {', '.join(extractors.keys())}"
        )

    raw_text = extractors[suffix](file_path)

    if not raw_text.strip():
        raise ValueError(f"Document '{file_path.name}' appears to be empty or unreadable.")

    chunks_text = _chunk_text(raw_text, settings.CHUNK_SIZE, settings.CHUNK_OVERLAP)

    return [
        {
            "text": chunk,
            "metadata": {
                "filename": file_path.name,
                "file_type": suffix.lstrip("."),
                "chunk_index": idx,
                "total_chunks": len(chunks_text),
            },
        }
        for idx, chunk in enumerate(chunks_text)
    ]
