"""
routers/analytics.py — Aggregated analytics from the evaluation log.
"""
from __future__ import annotations
from collections import Counter, defaultdict
from datetime import datetime, timezone

from fastapi import APIRouter

from models.schemas import (
    AnalyticsResponse,
    DocTypeStats,
    HourlyVolume,
    KPIStats,
    RecentEval,
    ScoreDataPoint,
)
from services.evaluator import load_eval_log
from services.vector_store import VectorStore

router = APIRouter(prefix="/api", tags=["analytics"])


@router.get("/analytics", response_model=AnalyticsResponse)
async def get_analytics() -> AnalyticsResponse:
    """Return aggregated evaluation metrics for the dashboard."""
    log = load_eval_log()
    vs = VectorStore.get()
    docs_info = vs.get_all_documents_info()

    # ── KPIs ────────────────────────────────────────────────────────────────
    n = len(log)

    def avg(key: str) -> float:
        return round(sum(r[key] for r in log) / n, 4) if n else 0.0

    kpis = KPIStats(
        total_queries=n,
        avg_faithfulness=avg("faithfulness"),
        avg_answer_relevancy=avg("answer_relevancy"),
        avg_context_precision=avg("context_precision"),
        avg_overall_score=avg("overall_score"),
        avg_latency_ms=round(sum(r["latency_ms"] for r in log) / n, 1) if n else 0.0,
        total_documents=len(docs_info),
        total_chunks=vs.total_chunks(),
    )

    # ── Score timeline (last 50 queries) ────────────────────────────────────
    score_timeline = [
        ScoreDataPoint(
            timestamp=r["timestamp"],
            faithfulness=r["faithfulness"],
            answer_relevancy=r["answer_relevancy"],
            context_precision=r["context_precision"],
            overall_score=r["overall_score"],
            latency_ms=r["latency_ms"],
        )
        for r in log[-50:]
    ]

    # ── Hourly query volume (last 24 h) ─────────────────────────────────────
    hour_counts: Counter = Counter()
    for r in log:
        try:
            dt = datetime.fromisoformat(r["timestamp"].replace("Z", "+00:00"))
            hour_key = dt.strftime("%Y-%m-%dT%H:00")
            hour_counts[hour_key] += 1
        except (ValueError, KeyError):
            pass

    # Build last 24 hours slots
    now = datetime.now(timezone.utc)
    hourly_volume = [
        HourlyVolume(
            hour=now.replace(hour=(now.hour - i) % 24, minute=0, second=0, microsecond=0).strftime(
                "%Y-%m-%dT%H:00"
            ),
            count=hour_counts.get(
                now.replace(hour=(now.hour - i) % 24, minute=0, second=0, microsecond=0).strftime(
                    "%Y-%m-%dT%H:00"
                ),
                0,
            ),
        )
        for i in range(23, -1, -1)
    ]

    # ── Recent evaluations (last 20) ─────────────────────────────────────────
    recent_evaluations = [
        RecentEval(
            query_id=r["query_id"],
            query=r["query"][:80],
            timestamp=r["timestamp"],
            faithfulness=r["faithfulness"],
            answer_relevancy=r["answer_relevancy"],
            context_precision=r["context_precision"],
            overall_score=r["overall_score"],
            latency_ms=r["latency_ms"],
        )
        for r in reversed(log[-20:])
    ]

    # ── Document type distribution ───────────────────────────────────────────
    type_counter: Counter = Counter(d["file_type"] for d in docs_info)
    doc_type_distribution = [
        DocTypeStats(file_type=ft, count=cnt) for ft, cnt in type_counter.most_common()
    ]

    return AnalyticsResponse(
        kpis=kpis,
        score_timeline=score_timeline,
        hourly_volume=hourly_volume,
        recent_evaluations=recent_evaluations,
        doc_type_distribution=doc_type_distribution,
    )
