"""
routers/chat.py — RAG chat endpoint with evaluation.
"""
from __future__ import annotations

from fastapi import APIRouter, HTTPException

from models.schemas import ChatRequest, ChatResponse, EvalMetrics, SourceChunk
from services.evaluator import evaluate
from services.rag_agent import run_rag_pipeline

router = APIRouter(prefix="/api", tags=["chat"])


@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest) -> ChatResponse:
    """
    Run the full RAG pipeline:
    vector search → cross-encoder rerank → Gemini answer → evaluation metrics.
    """
    try:
        result = run_rag_pipeline(request.query)
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"RAG pipeline error: {e}")

    # Run evaluation metrics
    context_texts = [s["text"] for s in result["sources"]]
    eval_record = evaluate(
        query=result["query"],
        answer=result["answer"],
        context_chunks=context_texts,
        query_id=result["query_id"],
        timestamp=result["timestamp"],
        latency_ms=result["latency_ms"],
    )

    sources = [
        SourceChunk(
            text=s["text"],
            filename=s["filename"],
            chunk_id=s["chunk_id"],
            rerank_score=s["rerank_score"],
            vector_distance=s["vector_distance"],
        )
        for s in result["sources"]
    ]

    metrics = EvalMetrics(
        faithfulness=eval_record["faithfulness"],
        answer_relevancy=eval_record["answer_relevancy"],
        context_precision=eval_record["context_precision"],
        overall_score=eval_record["overall_score"],
    )

    return ChatResponse(
        query=result["query"],
        answer=result["answer"],
        sources=sources,
        eval_metrics=metrics,
        latency_ms=result["latency_ms"],
        query_id=result["query_id"],
        timestamp=result["timestamp"],
    )
