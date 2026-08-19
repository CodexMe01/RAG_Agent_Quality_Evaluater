"""
main.py — FastAPI application entry point for Lyraa RAG Backend.

Start with:
    uvicorn main:app --reload --host 0.0.0.0 --port 8000
"""
from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from config import settings
from routers import analytics, chat, ingest

# ── App creation ────────────────────────────────────────────────────────────

app = FastAPI(
    title="Lyraa RAG API",
    description=(
        "Customer Support RAG Agent with file ingestion, "
        "cross-encoder reranking, and RAGAS-style evaluation."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# ── CORS ─────────────────────────────────────────────────────────────────────

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Routers ──────────────────────────────────────────────────────────────────

app.include_router(ingest.router)
app.include_router(chat.router)
app.include_router(analytics.router)

# ── Health check ─────────────────────────────────────────────────────────────

@app.get("/health", tags=["meta"])
async def health_check():
    return JSONResponse({"status": "ok", "service": "Lyraa RAG API", "version": "1.0.0"})


@app.get("/", tags=["meta"])
async def root():
    return JSONResponse(
        {
            "message": "Lyraa RAG API is running.",
            "docs": "/docs",
            "endpoints": {
                "ingest": "POST /api/ingest",
                "documents": "GET /api/documents",
                "chat": "POST /api/chat",
                "analytics": "GET /api/analytics",
                "health": "GET /health",
            },
        }
    )


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=True,
        log_level="info",
    )
