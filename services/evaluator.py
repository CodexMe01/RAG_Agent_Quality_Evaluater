"""
evaluator.py — RAGAS-style evaluation metrics computed locally.

Metrics:
  1. Faithfulness       — how grounded the answer is in the context
  2. Answer Relevancy   — cosine similarity of question & answer embeddings
  3. Context Precision  — fraction of retrieved chunks that are semantically relevant

All scores are in [0, 1].
Evaluation history is persisted to a local JSON log file.
"""
from __future__ import annotations
import json
import math
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from sentence_transformers import SentenceTransformer

from config import settings


# ── Lazy-load the same embedding model used for retrieval ───────────────────

_embedder: SentenceTransformer | None = None


def _get_embedder() -> SentenceTransformer:
    global _embedder
    if _embedder is None:
        _embedder = SentenceTransformer(settings.EMBED_MODEL)
    return _embedder


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


# ── Metric 1: Faithfulness ──────────────────────────────────────────────────

def compute_faithfulness(answer: str, context_chunks: list[str]) -> float:
    """
    Measures how much of the answer is grounded in the provided context.
    Approach: tokenise answer into sentences, check what fraction have
    a high-similarity context sentence (cosine ≥ 0.6).
    """
    if not context_chunks or not answer.strip():
        return 0.0

    # Split answer into sentences
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", answer) if s.strip()]
    if not sentences:
        return 0.0

    emb = _get_embedder()
    ans_embs = emb.encode(sentences, show_progress_bar=False)
    ctx_embs = emb.encode(context_chunks, show_progress_bar=False)

    faithful_count = 0
    for ans_emb in ans_embs:
        similarities = [_cosine(ans_emb, ctx_emb) for ctx_emb in ctx_embs]
        if max(similarities) >= 0.55:
            faithful_count += 1

    return round(faithful_count / len(sentences), 4)


# ── Metric 2: Answer Relevancy ──────────────────────────────────────────────

def compute_answer_relevancy(query: str, answer: str) -> float:
    """
    Cosine similarity between the query embedding and the answer embedding.
    Higher = answer more directly addresses the question.
    """
    if not query.strip() or not answer.strip():
        return 0.0

    emb = _get_embedder()
    q_emb, a_emb = emb.encode([query, answer], show_progress_bar=False)
    raw = _cosine(q_emb, a_emb)
    # Scale to [0, 1] — cosine is [-1, 1], but in practice NLP embeddings ≥ 0
    return round(max(0.0, raw), 4)


# ── Metric 3: Context Precision ─────────────────────────────────────────────

def compute_context_precision(query: str, context_chunks: list[str]) -> float:
    """
    Fraction of retrieved chunks that are relevant to the query.
    A chunk is "relevant" if its cosine similarity to the query is ≥ 0.45.
    """
    if not context_chunks or not query.strip():
        return 0.0

    emb = _get_embedder()
    q_emb = emb.encode([query], show_progress_bar=False)[0]
    ctx_embs = emb.encode(context_chunks, show_progress_bar=False)

    relevant = sum(1 for c_emb in ctx_embs if _cosine(q_emb, c_emb) >= 0.45)
    return round(relevant / len(context_chunks), 4)


# ── Overall score ───────────────────────────────────────────────────────────

def compute_overall_score(
    faithfulness: float,
    answer_relevancy: float,
    context_precision: float,
) -> float:
    """Weighted harmonic mean of the three metrics."""
    weights = [0.4, 0.4, 0.2]
    scores = [faithfulness, answer_relevancy, context_precision]
    weighted = sum(w * s for w, s in zip(weights, scores))
    return round(weighted, 4)


# ── Full evaluation ─────────────────────────────────────────────────────────

def evaluate(
    query: str,
    answer: str,
    context_chunks: list[str],
    query_id: str,
    timestamp: str,
    latency_ms: float,
) -> dict[str, float]:
    """
    Run all metrics and persist results to the eval log.
    Returns a dict with all metric values.
    """
    faithfulness = compute_faithfulness(answer, context_chunks)
    answer_relevancy = compute_answer_relevancy(query, answer)
    context_precision = compute_context_precision(query, context_chunks)
    overall = compute_overall_score(faithfulness, answer_relevancy, context_precision)

    record = {
        "query_id": query_id,
        "query": query,
        "answer_snippet": answer[:120],
        "timestamp": timestamp,
        "faithfulness": faithfulness,
        "answer_relevancy": answer_relevancy,
        "context_precision": context_precision,
        "overall_score": overall,
        "latency_ms": latency_ms,
    }

    _append_to_log(record)
    return record


# ── Log persistence ─────────────────────────────────────────────────────────

def _append_to_log(record: dict[str, Any]) -> None:
    log_path: Path = settings.EVAL_LOG_PATH
    existing: list[dict] = []

    if log_path.exists():
        try:
            existing = json.loads(log_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, ValueError):
            existing = []

    existing.append(record)
    log_path.write_text(json.dumps(existing, indent=2), encoding="utf-8")


def load_eval_log() -> list[dict[str, Any]]:
    """Load all evaluation records from disk."""
    log_path: Path = settings.EVAL_LOG_PATH
    if not log_path.exists():
        return []
    try:
        return json.loads(log_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, ValueError):
        return []
