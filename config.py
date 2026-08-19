"""
config.py — Application Settings
Reads from environment variables / .env file.
"""
import os
from pathlib import Path
from pydantic_settings import BaseSettings

BASE_DIR = Path(__file__).parent


class Settings(BaseSettings):
    # ── LLM ────────────────────────────────────────────────────────────
    GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")
    GEMINI_MODEL: str = "gemini-1.5-flash"

    # ── Embeddings ──────────────────────────────────────────────────────
    EMBED_MODEL: str = "all-MiniLM-L6-v2"

    # ── Reranker ────────────────────────────────────────────────────────
    RERANK_MODEL: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    RERANK_TOP_N: int = 3

    # ── Vector Store ────────────────────────────────────────────────────
    CHROMA_PATH: Path = BASE_DIR / "data" / "chroma_db"
    COLLECTION_NAME: str = "lyraa_kb"
    RETRIEVAL_TOP_K: int = 10

    # ── Document Processing ─────────────────────────────────────────────
    UPLOAD_DIR: Path = BASE_DIR / "data" / "uploads"
    CHUNK_SIZE: int = 500      # tokens (approx characters / 4)
    CHUNK_OVERLAP: int = 50

    # ── Analytics log ───────────────────────────────────────────────────
    EVAL_LOG_PATH: Path = BASE_DIR / "data" / "eval_log.json"

    # ── Server ──────────────────────────────────────────────────────────
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    CORS_ORIGINS: list[str] = ["*"]

    class Config:
        env_file = BASE_DIR / ".env"
        env_file_encoding = "utf-8"


settings = Settings()

# Ensure required directories exist
settings.CHROMA_PATH.mkdir(parents=True, exist_ok=True)
settings.UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
settings.EVAL_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
