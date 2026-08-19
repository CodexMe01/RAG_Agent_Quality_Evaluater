"""
routers/ingest.py — File ingestion endpoints.
"""
from __future__ import annotations
import shutil
from pathlib import Path

from fastapi import APIRouter, File, HTTPException, UploadFile

from config import settings
from models.schemas import DocumentInfo, DocumentListResponse, IngestResponse
from services.document_processor import process_document
from services.vector_store import VectorStore

router = APIRouter(prefix="/api", tags=["ingest"])

ALLOWED_EXTENSIONS = {".pdf", ".docx", ".doc", ".txt", ".md", ".csv"}


@router.post("/ingest", response_model=IngestResponse)
async def ingest_file(file: UploadFile = File(...)) -> IngestResponse:
    """
    Upload a document (PDF, DOCX, TXT, MD, CSV).
    The file is chunked, embedded, and stored in ChromaDB.
    """
    suffix = Path(file.filename).suffix.lower()
    if suffix not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{suffix}'. Allowed: {', '.join(ALLOWED_EXTENSIONS)}",
        )

    # Save upload to disk
    save_path = settings.UPLOAD_DIR / file.filename
    with save_path.open("wb") as f:
        shutil.copyfileobj(file.file, f)

    # Process → chunks
    try:
        chunks = process_document(save_path)
    except (ValueError, RuntimeError) as e:
        save_path.unlink(missing_ok=True)
        raise HTTPException(status_code=422, detail=str(e))

    # Embed + store
    vs = VectorStore.get()
    added = vs.add_documents(chunks)
    total_chunks = vs.total_chunks()

    return IngestResponse(
        success=True,
        filename=file.filename,
        chunks_added=added,
        total_documents=len(vs.get_all_documents_info()),
        message=f"Successfully ingested '{file.filename}' into {added} chunks.",
    )


@router.get("/documents", response_model=DocumentListResponse)
async def list_documents() -> DocumentListResponse:
    """List all ingested documents with chunk counts."""
    vs = VectorStore.get()
    docs_raw = vs.get_all_documents_info()

    docs = [
        DocumentInfo(
            id=d["id"],
            filename=d["filename"],
            chunk_count=d["chunk_count"],
            file_type=d["file_type"],
            ingested_at=d["ingested_at"],
        )
        for d in docs_raw
    ]

    return DocumentListResponse(
        documents=docs,
        total_chunks=vs.total_chunks(),
    )
