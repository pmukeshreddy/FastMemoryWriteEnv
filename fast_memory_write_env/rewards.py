"""Detailed scoring for streaming memory-write evaluation."""

from __future__ import annotations

from pydantic import Field

from fast_memory_write_env.metrics import AggregateMetrics
from fast_memory_write_env.schemas import StrictBaseModel


class ScoreBreakdown(StrictBaseModel):
    """A detailed score decomposition, not just one scalar."""

    score: float = Field(ge=0.0, le=100.0)
    answer_correct: float = Field(ge=0.0, le=1.0)
    evidence_correct: float = Field(ge=0.0, le=1.0)
    memory_recall: float = Field(ge=0.0, le=1.0)
    memory_precision: float = Field(ge=0.0, le=1.0)
    freshness_score: float = Field(ge=0.0, le=1.0)
    latency_penalty: float = Field(ge=0.0, le=1.0)
    storage_penalty: float = Field(ge=0.0, le=1.0)
    stale_memory_penalty: float = Field(ge=0.0, le=1.0)
    duplicate_penalty: float = Field(ge=0.0, le=1.0)


def score_metrics(
    metrics: AggregateMetrics,
    *,
    useful_memory_target_ms: float = 5_000.0,
    storage_budget_tokens: int = 10_000,
) -> ScoreBreakdown:
    """Return a deterministic score breakdown for aggregate metrics."""

    freshness_score = _freshness_score(metrics.time_to_useful_memory, useful_memory_target_ms)
    latency_penalty = _latency_penalty(metrics)
    storage_penalty = min(1.0, metrics.storage_tokens_used / max(1, storage_budget_tokens)) * 0.2
    stale_memory_penalty = metrics.stale_memory_rate * 0.2
    duplicate_penalty = metrics.duplicate_memory_rate * 0.15
    base = (
        (metrics.answer_correct * 0.30)
        + (metrics.evidence_correct * 0.25)
        + (metrics.memory_recall * 0.15)
        + (metrics.memory_precision * 0.10)
        + (freshness_score * 0.20)
    )
    score = max(
        0.0,
        min(
            1.0,
            base
            - latency_penalty
            - storage_penalty
            - stale_memory_penalty
            - duplicate_penalty,
        ),
    )
    return ScoreBreakdown(
        score=round(score * 100.0, 6),
        answer_correct=metrics.answer_correct,
        evidence_correct=metrics.evidence_correct,
        memory_recall=metrics.memory_recall,
        memory_precision=metrics.memory_precision,
        freshness_score=freshness_score,
        latency_penalty=latency_penalty,
        storage_penalty=storage_penalty,
        stale_memory_penalty=stale_memory_penalty,
        duplicate_penalty=duplicate_penalty,
    )


def _freshness_score(time_to_useful_memory: float | None, target_ms: float) -> float:
    if time_to_useful_memory is None:
        return 0.0
    return max(0.0, 1.0 - min(1.0, time_to_useful_memory / max(1.0, target_ms)))


def _latency_penalty(metrics: AggregateMetrics) -> float:
    latencies = [
        value
        for value in [
            metrics.write_latency_p95,
            metrics.index_latency_p95,
            metrics.query_latency_p95,
        ]
        if value is not None
    ]
    if not latencies:
        return 0.0
    return min(0.25, (max(latencies) / 10_000.0) * 0.25)
