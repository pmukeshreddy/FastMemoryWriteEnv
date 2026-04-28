from __future__ import annotations

from fast_memory_write_env.metrics import AggregateMetrics
from fast_memory_write_env.rewards import ScoreBreakdown, score_metrics


def _metrics(**overrides) -> AggregateMetrics:
    payload = {
        "time_to_raw_write": 1.0,
        "time_to_memory_write": 5.0,
        "time_to_indexed_memory": 8.0,
        "time_to_retrieved_memory": 15.0,
        "time_to_useful_memory": 25.0,
        "answer_success": 1.0,
        "answer_correct": 1.0,
        "evidence_correct": 1.0,
        "memory_precision": 1.0,
        "memory_recall": 1.0,
        "stale_memory_rate": 0.0,
        "duplicate_memory_rate": 0.0,
        "storage_tokens_used": 100,
        "useful_memory_per_storage_token": 0.1,
        "write_latency_p50": 5.0,
        "write_latency_p95": 7.0,
        "index_latency_p50": 4.0,
        "index_latency_p95": 6.0,
        "query_latency_p95": 9.0,
        "ignored_useful_fact_rate": 0.0,
        "stored_noise_rate": 0.0,
        "query_count": 3,
    }
    payload.update(overrides)
    return AggregateMetrics(**payload)


def test_score_breakdown_shape_and_high_success_score() -> None:
    breakdown = score_metrics(_metrics(), storage_budget_tokens=10_000)

    assert isinstance(breakdown, ScoreBreakdown)
    assert breakdown.score > 90
    assert breakdown.answer_correct == 1.0
    assert breakdown.evidence_correct == 1.0
    assert breakdown.freshness_score > 0.9


def test_missed_answers_reduce_score() -> None:
    successful = score_metrics(_metrics())
    missed = score_metrics(
        _metrics(
            answer_success=0.0,
            answer_correct=0.0,
            evidence_correct=0.0,
            memory_recall=0.0,
            time_to_useful_memory=None,
        )
    )

    assert missed.score < successful.score
    assert missed.freshness_score == 0.0


def test_stale_duplicate_latency_and_storage_penalties_reduce_score() -> None:
    clean = score_metrics(_metrics(), storage_budget_tokens=10_000)
    penalized = score_metrics(
        _metrics(
            stale_memory_rate=0.5,
            duplicate_memory_rate=0.5,
            storage_tokens_used=10_000,
            write_latency_p95=15_000,
            index_latency_p95=12_000,
            query_latency_p95=20_000,
        ),
        storage_budget_tokens=10_000,
    )

    assert penalized.score < clean.score
    assert penalized.latency_penalty > clean.latency_penalty
    assert penalized.storage_penalty > clean.storage_penalty
    assert penalized.stale_memory_penalty > clean.stale_memory_penalty
    assert penalized.duplicate_penalty > clean.duplicate_penalty
