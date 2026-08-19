"""
vector_store.py — ChromaDB vector store with sentence-transformer embeddings.
"""
from __future__ import annotations
import uuid
from typing import Any

import chromadb
from chromadb.config import Settings as ChromaSettings
from sentence_transformers import SentenceTransformer

from config import settings


class VectorStore:
    """Singleton wrapper around ChromaDB with embedded sentence-transformer."""

    _instance: "VectorStore | None" = None

    def __init__(self) -> None:
        print(f"[VectorStore] Loading embedding model: {settings.EMBED_MODEL}")
        self.embedder = SentenceTransformer(settings.EMBED_MODEL)

        self.client = chromadb.PersistentClient(
            path=str(settings.CHROMA_PATH),
            settings=ChromaSettings(anonymized_telemetry=False),
        )

        self.collection = self.client.get_or_create_collection(
            name=settings.COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )
        print(
            f"[VectorStore] Collection '{settings.COLLECTION_NAME}' ready. "
            f"Chunks stored: {self.collection.count()}"
        )

    @classmethod
    def get(cls) -> "VectorStore":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    # ── Ingestion ──────────────────────────────────────────────────────────

    def add_documents(self, chunks: list[dict[str, Any]]) -> int:
        """
        Embed and upsert chunks into ChromaDB.
        Returns number of chunks added.
        """
        if not chunks:
            return 0

        texts = [c["text"] for c in chunks]
        embeddings = self.embedder.encode(texts, show_progress_bar=False).tolist()
        ids = [str(uuid.uuid4()) for _ in chunks]
        metadatas = [c["metadata"] for c in chunks]

        self.collection.upsert(
            ids=ids,
            embeddings=embeddings,
            documents=texts,
            metadatas=metadatas,
        )
        return len(chunks)

    # ── Retrieval ──────────────────────────────────────────────────────────

    def search(
        self, query: str, top_k: int | None = None
    ) -> list[dict[str, Any]]:
        """
        Embed query and return top-K chunks with distance scores.
        Each result: {id, text, metadata, distance}
        """
        k = top_k or settings.RETRIEVAL_TOP_K
        query_emb = self.embedder.encode([query], show_progress_bar=False).tolist()

        results = self.collection.query(
            query_embeddings=query_emb,
            n_results=min(k, self.collection.count() or 1),
            include=["documents", "metadatas", "distances"],
        )

        items = []
        for i, doc_id in enumerate(results["ids"][0]):
            items.append(
                {
                    "id": doc_id,
                    "text": results["documents"][0][i],
                    "metadata": results["metadatas"][0][i],
                    "distance": results["distances"][0][i],
                }
            )
        return items

    # ── Metadata ───────────────────────────────────────────────────────────

    def get_all_documents_info(self) -> list[dict[str, Any]]:
        """Return aggregated document metadata (grouped by filename)."""
        total = self.collection.count()
        if total == 0:
            return []

        # Fetch all metadatas (limit large collections gracefully)
        fetch_limit = min(total, 10_000)
        results = self.collection.get(
            limit=fetch_limit,
            include=["metadatas"],
        )

        from collections import defaultdict
        import datetime

        groups: dict[str, dict] = defaultdict(
            lambda: {"chunk_count": 0, "file_type": "", "ingested_at": ""}
        )

        for meta in results["metadatas"]:
            fname = meta.get("filename", "unknown")
            groups[fname]["chunk_count"] += 1
            groups[fname]["file_type"] = meta.get("file_type", "")
            if not groups[fname]["ingested_at"]:
                groups[fname]["ingested_at"] = datetime.datetime.utcnow().isoformat()

        return [
            {
                "id": str(uuid.uuid5(uuid.NAMESPACE_DNS, name)),
                "filename": name,
                **info,
            }
            for name, info in groups.items()
        ]

    def total_chunks(self) -> int:
        return self.collection.count()
