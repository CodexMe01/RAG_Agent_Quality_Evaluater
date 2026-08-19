"""
reranker.py — Cross-encoder reranker using sentence-transformers.
Reorders vector-retrieved candidates by relevance to the query.
"""
from __future__ import annotations

from sentence_transformers import CrossEncoder

from config import settings


class Reranker:
    """Singleton cross-encoder reranker."""

    _instance: "Reranker | None" = None

    def __init__(self) -> None:
        print(f"[Reranker] Loading cross-encoder: {settings.RERANK_MODEL}")
        self.model = CrossEncoder(settings.RERANK_MODEL, max_length=512)

    @classmethod
    def get(cls) -> "Reranker":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def rerank(
        self,
        query: str,
        candidates: list[dict],
        top_n: int | None = None,
    ) -> list[dict]:
        """
        Score each candidate passage against the query using a cross-encoder.
        Returns top-N candidates sorted by rerank_score descending.

        Each candidate must have a 'text' key.
        Each returned item gets a 'rerank_score' (float, higher = more relevant).
        """
        n = top_n or settings.RERANK_TOP_N

        if not candidates:
            return []

        pairs = [(query, c["text"]) for c in candidates]
        scores = self.model.predict(pairs, show_progress_bar=False).tolist()

        for candidate, score in zip(candidates, scores):
            candidate["rerank_score"] = float(score)

        reranked = sorted(candidates, key=lambda x: x["rerank_score"], reverse=True)
        return reranked[:n]
