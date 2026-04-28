from __future__ import annotations

import csv
import json

from fast_memory_write_env.dataset import generate_episode
from fast_memory_write_env.evaluator import StreamingEvaluator, write_evaluation_outputs
from fast_memory_write_env.llm_client import MockLLMClient
from fast_memory_write_env.metrics import (
    AggregateCounterSnapshot,
    FactLifecycle,
    QueryMetricRecord,
    RolloutRecord,
    aggregate_metrics,
    compute_time_breakdown,
    evaluate_query_result,
    extract_run_config,
    percentile,
    read_rollout_jsonl,
    summarize_rollout_records,
    write_metrics_csv,
    write_rollout_jsonl,
)
from fast_memory_write_env.policies import LLMMemoryWritePolicy
from fast_memory_write_env.schemas import DatasetMode, MemoryRecord, Query


def _query_and_gold_event():
    episode = generate_episode(DatasetMode.SMALL, seed=61, episode_index=0)
    query_item = next(item for item in episode.stream if item.item_type == "query")
    query = query_item.query
    gold_event = next(
        item.event
        for item in episode.stream
        if item.item_type == "event" and item.event.event_id in query.gold.supporting_event_ids
    )
    return query, gold_event


def _memory_for_query(query: Query, content: str = "Useful answer memory.") -> MemoryRecord:
    return MemoryRecord(
        memory_id="mem-useful",
        entity_id=query.target_entity_id,
        content=content,
        source_event_ids=list(query.gold.supporting_event_ids),
        fact_ids=list(query.gold.required_fact_ids),
        created_at_ms=100,
        updated_at_ms=100,
        indexed=True,
        estimated_tokens=4,
    )


def test_percentile_interpolates_and_handles_empty() -> None:
    assert percentile([], 95) is None
    assert percentile([1, 2, 3, 4], 50) == 2.5
    assert percentile([1, 2, 3, 4], 95) == 3.8499999999999996


def test_query_scoring_separates_answer_text_from_evidence_coverage() -> None:
    query, event = _query_and_gold_event()
    memory = _memory_for_query(query)
    fact_lifecycles = {
        fact_id: FactLifecycle(
            fact_id=fact_id,
            source_event_id=event.event_id,
            event_timestamp_ms=float(event.timestamp_ms),
            raw_written_at_ms=float(event.timestamp_ms + 1),
            memory_written_at_ms=float(event.timestamp_ms + 5),
            indexed_at_ms=float(event.timestamp_ms + 8),
            retrieved_at_ms=float(query.timestamp_ms + 3),
        )
        for fact_id in query.gold.required_fact_ids
    }

    metric = evaluate_query_result(
        query=query,
        cited_memories=[memory],
        retrieved_memory_ids=[memory.memory_id],
        fact_lifecycles=fact_lifecycles,
        answer_text=query.gold.answer_facts[0],
        answer_completed_at_ms=float(query.timestamp_ms + 6),
    )

    assert metric.answer_correct is True
    assert metric.evidence_correct is True
    assert metric.fact_evidence_coverage is True
    assert metric.answer_success is True
    assert metric.memory_precision == 1.0
    assert metric.memory_recall == 1.0
    assert metric.debug_contains_answer_facts is True
    assert metric.time_to_useful_memory is not None


def test_unrelated_answer_text_does_not_pass_answer_correctness() -> None:
    query, event = _query_and_gold_event()
    memory = _memory_for_query(query)
    fact_lifecycles = {
        fact_id: FactLifecycle(
            fact_id=fact_id,
            source_event_id=event.event_id,
            event_timestamp_ms=float(event.timestamp_ms),
            raw_written_at_ms=float(event.timestamp_ms + 1),
            memory_written_at_ms=float(event.timestamp_ms + 5),
            indexed_at_ms=float(event.timestamp_ms + 8),
            retrieved_at_ms=float(query.timestamp_ms + 3),
        )
        for fact_id in query.gold.required_fact_ids
    }

    metric = evaluate_query_result(
        query=query,
        cited_memories=[memory],
        retrieved_memory_ids=[memory.memory_id],
        fact_lifecycles=fact_lifecycles,
        answer_text="A completely unrelated operational note.",
        answer_completed_at_ms=float(query.timestamp_ms + 6),
    )

    assert metric.fact_evidence_coverage is True
    assert metric.evidence_correct is True
    assert metric.answer_correct is False
    assert metric.answer_success is False
    assert metric.time_to_useful_memory is None


def test_query_scoring_marks_missing_evidence_incorrect() -> None:
    query, event = _query_and_gold_event()
    memory = _memory_for_query(query).model_copy(update={"source_event_ids": ["wrong-event"]})
    fact_lifecycles = {
        fact_id: FactLifecycle(
            fact_id=fact_id,
            source_event_id=event.event_id,
            event_timestamp_ms=float(event.timestamp_ms),
            raw_written_at_ms=float(event.timestamp_ms + 1),
            memory_written_at_ms=float(event.timestamp_ms + 5),
            indexed_at_ms=float(event.timestamp_ms + 8),
            retrieved_at_ms=float(query.timestamp_ms + 3),
        )
        for fact_id in query.gold.required_fact_ids
    }

    metric = evaluate_query_result(
        query=query,
        cited_memories=[memory],
        retrieved_memory_ids=[memory.memory_id],
        fact_lifecycles=fact_lifecycles,
        answer_text=query.gold.answer_facts[0],
        answer_completed_at_ms=float(query.timestamp_ms + 6),
    )

    assert metric.answer_correct is True
    assert metric.evidence_correct is False
    assert metric.answer_success is False
    assert metric.time_to_useful_memory is None


def test_time_breakdown_returns_none_for_missing_stage() -> None:
    query, event = _query_and_gold_event()
    fact_id = query.gold.required_fact_ids[0]
    lifecycle = FactLifecycle(
        fact_id=fact_id,
        source_event_id=event.event_id,
        event_timestamp_ms=float(event.timestamp_ms),
        raw_written_at_ms=float(event.timestamp_ms + 1),
    )

    breakdown = compute_time_breakdown(
        required_fact_ids=[fact_id],
        fact_lifecycles={fact_id: lifecycle},
        answer_completed_at_ms=float(query.timestamp_ms + 10),
        answer_success=True,
    )

    assert breakdown["time_to_raw_write"] == 1.0
    assert breakdown["time_to_memory_write"] is None
    assert breakdown["time_to_useful_memory"] is None

    for missing_stage in ["indexed_at_ms", "retrieved_at_ms"]:
        complete = lifecycle.model_copy(
            update={
                "memory_written_at_ms": float(event.timestamp_ms + 5),
                "indexed_at_ms": float(event.timestamp_ms + 8),
                "retrieved_at_ms": float(query.timestamp_ms + 3),
            }
        )
        breakdown = compute_time_breakdown(
            required_fact_ids=[fact_id],
            fact_lifecycles={fact_id: complete.model_copy(update={missing_stage: None})},
            answer_completed_at_ms=float(query.timestamp_ms + 10),
            answer_success=True,
        )
        assert breakdown["time_to_raw_write"] == 1.0
        assert breakdown["time_to_memory_write"] == 5.0
        assert breakdown["time_to_useful_memory"] is None


def test_aggregate_metrics_rates_and_latencies() -> None:
    records = [
        QueryMetricRecord(
            episode_id="ep",
            query_id="q1",
            query_timestamp_ms=100,
            time_to_raw_write=1,
            time_to_memory_write=5,
            time_to_indexed_memory=8,
            time_to_retrieved_memory=20,
            time_to_useful_memory=25,
            answer_success=True,
            answer_correct=True,
            evidence_correct=True,
            fact_evidence_coverage=True,
            memory_precision=1.0,
            memory_recall=1.0,
        ),
        QueryMetricRecord(
            episode_id="ep",
            query_id="q2",
            query_timestamp_ms=200,
            answer_success=False,
            answer_correct=False,
            evidence_correct=False,
            memory_precision=0.0,
            memory_recall=0.0,
        ),
    ]
    counters = AggregateCounterSnapshot(
        total_memory_count=4,
        stale_memory_count=1,
        duplicate_memory_count=1,
        storage_tokens_used=40,
        useful_memory_count=2,
        useful_event_count=5,
        ignored_useful_event_count=1,
        noise_event_count=4,
        stored_noise_memory_count=1,
        write_latencies_ms=[4, 6],
        index_latencies_ms=[3, 5],
        query_latencies_ms=[10, 20],
    )

    aggregate = aggregate_metrics(records, counters)

    assert aggregate.time_to_useful_memory == 25
    assert aggregate.answer_success == 0.5
    assert aggregate.fact_evidence_coverage == 0.5
    assert aggregate.memory_precision == 0.5
    assert aggregate.stale_memory_rate == 0.25
    assert aggregate.duplicate_memory_rate == 0.25
    assert aggregate.useful_memory_per_storage_token == 0.05
    assert aggregate.write_latency_p50 == 5.0
    assert aggregate.ignored_useful_fact_rate == 0.2
    assert aggregate.stored_noise_rate == 0.25


def test_rollout_and_metrics_serialization_round_trip(tmp_path) -> None:
    metric = QueryMetricRecord(
        episode_id="ep",
        query_id="q1",
        query_timestamp_ms=100,
        answer_success=True,
        answer_correct=True,
        evidence_correct=True,
        memory_precision=1.0,
        memory_recall=1.0,
    )
    counters = AggregateCounterSnapshot(total_memory_count=1, storage_tokens_used=5)
    records = [
        RolloutRecord(
            record_type="query_metric",
            episode_id="ep",
            payload=metric.model_dump(mode="json"),
        ),
        RolloutRecord(
            record_type="aggregate_inputs",
            episode_id="ep",
            payload=counters.model_dump(mode="json"),
        ),
    ]

    raw_path = tmp_path / "raw_rollouts.jsonl"
    csv_path = tmp_path / "metrics.csv"
    write_rollout_jsonl(records, raw_path)
    loaded = read_rollout_jsonl(raw_path)
    query_metrics, aggregate = summarize_rollout_records(loaded)
    write_metrics_csv(query_metrics, aggregate, csv_path)

    assert len(query_metrics) == 1
    assert aggregate.query_count == 1
    with csv_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert rows[0]["query_id"] == "q1"
    assert json.loads(rows[0]["cited_memory_ids"]) == []


def test_streaming_evaluator_writes_required_outputs(tmp_path) -> None:
    episode = generate_episode(DatasetMode.SMALL, seed=71, episode_index=0)
    policy = LLMMemoryWritePolicy(llm_client=MockLLMClient())
    evaluator = StreamingEvaluator.with_local_test_index(policy=policy, work_dir=tmp_path / "db")

    result = evaluator.evaluate_episode(episode)
    result = write_evaluation_outputs(result, tmp_path / "results")

    assert result.query_metrics
    assert result.aggregate_metrics.query_count == len(result.query_metrics)
    assert (tmp_path / "results" / "raw_rollouts.jsonl").exists()
    assert (tmp_path / "results" / "metrics.csv").exists()
    assert (tmp_path / "results" / "eval_summary.json").exists()

    records = read_rollout_jsonl(tmp_path / "results" / "raw_rollouts.jsonl")
    run_config = extract_run_config(records)
    rebuilt_metrics, rebuilt_aggregate = summarize_rollout_records(records)
    summary = json.loads((tmp_path / "results" / "eval_summary.json").read_text(encoding="utf-8"))
    assert len(rebuilt_metrics) == len(result.query_metrics)
    assert rebuilt_aggregate.query_count == result.aggregate_metrics.query_count
    assert run_config is not None
    assert run_config.dataset_mode == DatasetMode.SMALL.value
    assert run_config.backend_type == "in_memory_test"
    assert summary["run_config"]["dataset_mode"] == DatasetMode.SMALL.value
    assert any(record.record_type == "run_config" for record in records)
