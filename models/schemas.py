"""
schemas.py — Pydantic request/response models
"""
from __future__ import annotations
from typing import Optional
from pydantic import BaseModel, Field


# ── Ingest ──────────────────────────────────────────────────────────────────

class IngestResponse(BaseModel):
    success: bool
    filename: str
    chunks_added: int
    total_documents: int
    message: str


class DocumentInfo(BaseModel):
    id: str
    filename: str
    chunk_count: int
    file_type: str
    ingested_at: str


class DocumentListResponse(BaseModel):
    documents: list[DocumentInfo]
    total_chunks: int


# ── Chat ────────────────────────────────────────────────────────────────────

class ChatRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=2000)
    session_id: Optional[str] = None


class SourceChunk(BaseModel):
    text: str
    filename: str
    chunk_id: str
    rerank_score: float
    vector_distance: float


class EvalMetrics(BaseModel):
    faithfulness: float = Field(..., ge=0.0, le=1.0)
    answer_relevancy: float = Field(..., ge=0.0, le=1.0)
    context_precision: float = Field(..., ge=0.0, le=1.0)
    overall_score: float = Field(..., ge=0.0, le=1.0)


class ChatResponse(BaseModel):
    query: str
    answer: str
    sources: list[SourceChunk]
    eval_metrics: EvalMetrics
    latency_ms: float
    query_id: str
    timestamp: str


# ── Analytics ───────────────────────────────────────────────────────────────

class KPIStats(BaseModel):
    total_queries: int
    avg_faithfulness: float
    avg_answer_relevancy: float
    avg_context_precision: float
    avg_overall_score: float
    avg_latency_ms: float
    total_documents: int
    total_chunks: int


class ScoreDataPoint(BaseModel):
    timestamp: str
    faithfulness: float
    answer_relevancy: float
    context_precision: float
    overall_score: float
    latency_ms: float


class HourlyVolume(BaseModel):
    hour: str
    count: int


class RecentEval(BaseModel):
    query_id: str
    query: str
    timestamp: str
    faithfulness: float
    answer_relevancy: float
    context_precision: float
    overall_score: float
    latency_ms: float


class DocTypeStats(BaseModel):
    file_type: str
    count: int


class AnalyticsResponse(BaseModel):
    kpis: KPIStats
    score_timeline: list[ScoreDataPoint]
    hourly_volume: list[HourlyVolume]
    recent_evaluations: list[RecentEval]
    doc_type_distribution: list[DocTypeStats]
