"""Streaming evaluator for Phase 4."""

from __future__ import annotations

import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pydantic import Field

from fast_memory_write_env.actions import (
    AnswerAction,
    IndexNowAction,
    SearchMemoryAction,
    StoreRawAction,
)
from fast_memory_write_env.config import load_pinecone_config
from fast_memory_write_env.env import FastMemoryWriteEnv
from fast_memory_write_env.in_memory_index import InMemoryIndex
from fast_memory_write_env.metrics import (
    AggregateCounterSnapshot,
    AggregateMetrics,
    FactLifecycle,
    QueryMetricRecord,
    RunConfig,
    RolloutRecord,
    aggregate_metrics,
    evaluate_query_result,
    write_eval_summary,
    write_metrics_csv,
    write_rollout_jsonl,
)
from fast_memory_write_env.pinecone_index import PineconeIndex
from fast_memory_write_env.policies import MemoryWritePolicy
from fast_memory_write_env.rewards import ScoreBreakdown, score_metrics
from fast_memory_write_env.schemas import EventCategory, MemoryRecord, MemoryStatus, RawEvent, StreamingEpisode, StrictBaseModel
from fast_memory_write_env.stores import MemoryStore, RawEventStore


USEFUL_EVENT_CATEGORIES = {
    EventCategory.USEFUL_FACT,
    EventCategory.CONTRADICTION,
    EventCategory.STALE_UPDATE,
    EventCategory.URGENT_FACT,
}


class EvaluationResult(StrictBaseModel):
    """Complete in-memory evaluation result."""

    episode_id: str
    rollout_records: list[RolloutRecord]
    query_metrics: list[QueryMetricRecord]
    aggregate_metrics: AggregateMetrics
    score_breakdown: ScoreBreakdown
    run_config: RunConfig
    output_paths: dict[str, str] = Field(default_factory=dict)


class StreamingEvaluator:
    """Run a streaming episode through a memory-write policy and score it."""

    def __init__(
        self,
        *,
        env: FastMemoryWriteEnv,
        policy: MemoryWritePolicy,
        latency_budget_ms: int = 250,
        storage_budget_tokens_remaining: int = 10_000,
        indexing_budget_operations_remaining: int = 3,
        run_config: RunConfig | dict[str, Any] | None = None,
    ) -> None:
        self.env = env
        self.policy = policy
        self.latency_budget_ms = latency_budget_ms
        self.storage_budget_tokens_remaining = storage_budget_tokens_remaining
        self.indexing_budget_operations_remaining = indexing_budget_operations_remaining
        self.run_config = RunConfig.model_validate(run_config or {})

    @classmethod
    def with_local_test_index(
        cls,
        *,
        policy: MemoryWritePolicy,
        work_dir: str | Path | None = None,
        run_config: RunConfig | dict[str, Any] | None = None,
    ) -> StreamingEvaluator:
        base = Path(work_dir) if work_dir is not None else Path(tempfile.mkdtemp(prefix="fmwe-eval-"))
        env = FastMemoryWriteEnv(
            raw_event_store=RawEventStore(base / "raw.sqlite"),
            memory_store=MemoryStore(base / "memory.sqlite"),
            retrieval_index=InMemoryIndex(),
        )
        config = _merge_run_config(run_config, {"backend_type": "in_memory_test"})
        return cls(env=env, policy=policy, run_config=config)

    @classmethod
    def with_pinecone(
        cls,
        *,
        policy: MemoryWritePolicy,
        work_dir: str | Path,
        run_config: RunConfig | dict[str, Any] | None = None,
    ) -> StreamingEvaluator:
        config = load_pinecone_config(required=True)
        assert config is not None
        env = FastMemoryWriteEnv(
            raw_event_store=RawEventStore(Path(work_dir) / "raw.sqlite"),
            memory_store=MemoryStore(Path(work_dir) / "memory.sqlite"),
            retrieval_index=PineconeIndex(config),
        )
        run_config_payload = _merge_run_config(
            run_config,
            {
                "backend_type": "pinecone",
                "pinecone_index_name": config.index_name,
            },
        )
        return cls(env=env, policy=policy, run_config=run_config_payload)

    def evaluate_episode(self, episode: StreamingEpisode) -> EvaluationResult:
        records: list[RolloutRecord] = []
        query_metrics: list[QueryMetricRecord] = []
        recent_events: list[RawEvent] = []
        fact_lifecycles: dict[str, FactLifecycle] = {}
        event_categories: dict[str, EventCategory] = {}
        logical_time_ms = 0.0
        counters = AggregateCounterSnapshot()
        run_config = _merge_run_config(
            self.run_config,
            {
                "dataset_mode": episode.mode.value,
                "seed": episode.seed,
                "episode_index": episode.metadata.get("episode_index"),
                "episode_id": episode.episode_id,
                "policy_name": type(self.policy).__name__,
                "latency_budget_ms": self.latency_budget_ms,
                "storage_budget_tokens_remaining": self.storage_budget_tokens_remaining,
                "indexing_budget_operations_remaining": self.indexing_budget_operations_remaining,
                "timestamp_utc": self.run_config.timestamp_utc
                or datetime.now(timezone.utc).isoformat(),
            },
        )
        records.append(
            RolloutRecord(
                record_type="run_config",
                episode_id=episode.episode_id,
                logical_time_ms=logical_time_ms,
                payload=run_config.model_dump(mode="json"),
            )
        )

        for item in episode.stream:
            logical_time_ms = max(logical_time_ms, float(item.timestamp_ms))
            self.env.current_time_ms = item.timestamp_ms
            if item.item_type == "event":
                event = item.event
                event_categories[event.event_id] = event.category
                if event.category in USEFUL_EVENT_CATEGORIES and event.facts:
                    counters.useful_event_count += 1
                if event.category == EventCategory.NOISE:
                    counters.noise_event_count += 1
                for fact in event.facts:
                    fact_lifecycles[fact.fact_id] = FactLifecycle(
                        fact_id=fact.fact_id,
                        source_event_id=event.event_id,
                        event_timestamp_ms=float(event.timestamp_ms),
                    )
                records.append(
                    RolloutRecord(
                        record_type="event",
                        episode_id=episode.episode_id,
                        timestamp_ms=float(event.timestamp_ms),
                        logical_time_ms=logical_time_ms,
                        payload={
                            "event_id": event.event_id,
                            "category": event.category.value,
                            "fact_ids": [fact.fact_id for fact in event.facts],
                        },
                    )
                )

                raw_result = self.env.execute_action(StoreRawAction(event=event))
                logical_time_ms += raw_result.latency_ms
                for fact in event.facts:
                    fact_lifecycles[fact.fact_id].raw_written_at_ms = logical_time_ms
                records.append(_action_record(episode.episode_id, event.timestamp_ms, logical_time_ms, raw_result))

                actions = self.policy.decide(
                    new_event=event,
                    active_memories=self.env.memory_store.list_active(),
                    recent_events=recent_events[-5:],
                    latency_budget_ms=self.latency_budget_ms,
                    storage_budget_tokens_remaining=self.storage_budget_tokens_remaining,
                    indexing_budget_operations_remaining=self.indexing_budget_operations_remaining,
                )
                for action in actions:
                    result = self.env.execute_action(action)
                    logical_time_ms += result.latency_ms
                    records.append(
                        _action_record(
                            episode.episode_id,
                            event.timestamp_ms,
                            logical_time_ms,
                            result,
                            proposed_action=action.model_dump(mode="json"),
                        )
                    )
                    _update_timelines_for_action(
                        action_type=result.action_type.value,
                        action_payload=action.model_dump(mode="json"),
                        result_payload=result.payload,
                        env=self.env,
                        fact_lifecycles=fact_lifecycles,
                        logical_time_ms=logical_time_ms,
                    )
                    _update_latency_counters(result.action_type.value, result.latency_ms, result.payload, counters)
                    if result.action_type.value == "ignore_event" and event.category in USEFUL_EVENT_CATEGORIES and event.facts:
                        counters.ignored_useful_event_count += 1
                recent_events.append(event)
            else:
                query = item.query
                search_result = self.env.execute_action(
                    SearchMemoryAction(query_id=query.query_id, query_text=query.text, top_k=5)
                )
                logical_time_ms += search_result.latency_ms
                records.append(_action_record(episode.episode_id, query.timestamp_ms, logical_time_ms, search_result))
                retrieved_memory_ids = [
                    hit["memory_id"]
                    for hit in search_result.payload.get("results", [])
                    if isinstance(hit, dict) and "memory_id" in hit
                ]
                retrieved_memories = [
                    memory
                    for memory_id in retrieved_memory_ids
                    if (memory := self.env.memory_store.get(memory_id)) is not None
                ]
                for memory in retrieved_memories:
                    for fact_id in memory.fact_ids:
                        if fact_id in fact_lifecycles:
                            fact_lifecycles[fact_id].retrieved_at_ms = logical_time_ms

                answer_result = self.env.execute_action(
                    AnswerAction(
                        query_id=query.query_id,
                        query_text=query.text,
                        retrieved_memory_ids=retrieved_memory_ids,
                    )
                )
                logical_time_ms += answer_result.latency_ms
                records.append(_action_record(episode.episode_id, query.timestamp_ms, logical_time_ms, answer_result))
                counters.query_latencies_ms.append(search_result.latency_ms + answer_result.latency_ms)

                cited_memory_ids = [
                    memory_id
                    for memory_id in answer_result.payload.get("cited_memory_ids", [])
                    if isinstance(memory_id, str)
                ]
                cited_memories = [
                    memory
                    for memory_id in cited_memory_ids
                    if (memory := self.env.memory_store.get(memory_id)) is not None
                ]
                metric = evaluate_query_result(
                    query=query,
                    cited_memories=cited_memories,
                    retrieved_memory_ids=retrieved_memory_ids,
                    fact_lifecycles=fact_lifecycles,
                    answer_text=str(answer_result.payload.get("answer", "")),
                    answer_completed_at_ms=logical_time_ms,
                )
                query_metrics.append(metric)
                records.append(
                    RolloutRecord(
                        record_type="query",
                        episode_id=episode.episode_id,
                        timestamp_ms=float(query.timestamp_ms),
                        logical_time_ms=logical_time_ms,
                        payload={
                            "query": query.model_dump(mode="json"),
                            "retrieved_memory_ids": retrieved_memory_ids,
                            "answer": answer_result.payload,
                        },
                    )
                )
                records.append(
                    RolloutRecord(
                        record_type="query_metric",
                        episode_id=episode.episode_id,
                        timestamp_ms=float(query.timestamp_ms),
                        logical_time_ms=logical_time_ms,
                        payload=metric.model_dump(mode="json"),
                    )
                )

        final_counters = _finalize_counters(self.env.memory_store.list_all(), event_categories, counters)
        records.append(
            RolloutRecord(
                record_type="aggregate_inputs",
                episode_id=episode.episode_id,
                logical_time_ms=logical_time_ms,
                payload=final_counters.model_dump(mode="json"),
            )
        )
        aggregate = aggregate_metrics(query_metrics, final_counters)
        return EvaluationResult(
            episode_id=episode.episode_id,
            rollout_records=records,
            query_metrics=query_metrics,
            aggregate_metrics=aggregate,
            score_breakdown=score_metrics(aggregate),
            run_config=run_config,
        )


def write_evaluation_outputs(result: EvaluationResult, output_dir: str | Path) -> EvaluationResult:
    """Write raw rollout, metrics CSV, and summary JSON."""

    output_path = Path(output_dir)
    raw_path = output_path / "raw_rollouts.jsonl"
    metrics_path = output_path / "metrics.csv"
    summary_path = output_path / "eval_summary.json"
    write_rollout_jsonl(result.rollout_records, raw_path)
    write_metrics_csv(result.query_metrics, result.aggregate_metrics, metrics_path)
    summary = {
        "episode_id": result.episode_id,
        "aggregate_metrics": result.aggregate_metrics.model_dump(mode="json"),
        "score_breakdown": result.score_breakdown.model_dump(mode="json"),
        "run_config": result.run_config.model_dump(mode="json"),
        "counts": {
            "rollout_records": len(result.rollout_records),
            "query_metrics": len(result.query_metrics),
        },
        "output_paths": {
            "raw_rollouts": str(raw_path),
            "metrics_csv": str(metrics_path),
            "eval_summary": str(summary_path),
        },
    }
    write_eval_summary(summary, summary_path)
    return result.model_copy(
        update={
            "output_paths": {
                "raw_rollouts": str(raw_path),
                "metrics_csv": str(metrics_path),
                "eval_summary": str(summary_path),
            }
        }
    )


def _merge_run_config(
    base: RunConfig | dict[str, Any] | None,
    updates: dict[str, Any],
) -> RunConfig:
    config = RunConfig.model_validate(base or {})
    payload = config.model_dump(mode="json")
    metadata = dict(payload.pop("metadata", {}) or {})
    for key, value in updates.items():
        if key == "metadata" and isinstance(value, dict):
            metadata.update(value)
        elif value is not None:
            payload[key] = value
    payload["metadata"] = metadata
    return RunConfig.model_validate(payload)


def _action_record(
    episode_id: str,
    timestamp_ms: int | float,
    logical_time_ms: float,
    result: Any,
    *,
    proposed_action: dict[str, Any] | None = None,
) -> RolloutRecord:
    payload = {
        "result": result.model_dump(mode="json"),
    }
    if proposed_action is not None:
        payload["proposed_action"] = proposed_action
    return RolloutRecord(
        record_type="action",
        episode_id=episode_id,
        timestamp_ms=float(timestamp_ms),
        logical_time_ms=logical_time_ms,
        payload=payload,
    )


def _update_timelines_for_action(
    *,
    action_type: str,
    action_payload: dict[str, Any],
    result_payload: dict[str, Any],
    env: FastMemoryWriteEnv,
    fact_lifecycles: dict[str, FactLifecycle],
    logical_time_ms: float,
) -> None:
    if action_type in {"write_memory", "update_memory", "compress_memory"}:
        for fact_id in action_payload.get("fact_ids", []):
            if fact_id in fact_lifecycles:
                fact_lifecycles[fact_id].memory_written_at_ms = logical_time_ms
    if action_type in {"write_memory", "update_memory"} and result_payload.get("indexed"):
        for fact_id in action_payload.get("fact_ids", []):
            if fact_id in fact_lifecycles:
                fact_lifecycles[fact_id].indexed_at_ms = logical_time_ms
    if action_type == "index_now":
        memory_id = str(action_payload.get("memory_id", ""))
        memory = env.memory_store.get(memory_id)
        if memory is not None:
            for fact_id in memory.fact_ids:
                if fact_id in fact_lifecycles:
                    fact_lifecycles[fact_id].indexed_at_ms = logical_time_ms
    if action_type == "compress_memory":
        target_id = str(action_payload.get("target_memory_id", ""))
        memory = env.memory_store.get(target_id)
        if memory is not None:
            for fact_id in memory.fact_ids:
                if fact_id in fact_lifecycles:
                    fact_lifecycles[fact_id].memory_written_at_ms = logical_time_ms


def _update_latency_counters(
    action_type: str,
    latency_ms: float,
    result_payload: dict[str, Any],
    counters: AggregateCounterSnapshot,
) -> None:
    if action_type in {"write_memory", "update_memory", "compress_memory"}:
        counters.write_latencies_ms.append(latency_ms)
    if action_type == "index_now" or (
        action_type in {"write_memory", "update_memory"} and result_payload.get("indexed")
    ):
        counters.index_latencies_ms.append(latency_ms)


def _finalize_counters(
    memories: list[MemoryRecord],
    event_categories: dict[str, EventCategory],
    counters: AggregateCounterSnapshot,
) -> AggregateCounterSnapshot:
    total = len(memories)
    stale = sum(1 for memory in memories if memory.status == MemoryStatus.STALE)
    duplicate = sum(
        1
        for memory in memories
        if any(event_categories.get(event_id) == EventCategory.DUPLICATE for event_id in memory.source_event_ids)
    )
    stored_noise = sum(
        1
        for memory in memories
        if any(event_categories.get(event_id) == EventCategory.NOISE for event_id in memory.source_event_ids)
    )
    useful = sum(
        1
        for memory in memories
        if memory.fact_ids
        and any(event_categories.get(event_id) in USEFUL_EVENT_CATEGORIES for event_id in memory.source_event_ids)
    )
    return counters.model_copy(
        update={
            "total_memory_count": total,
            "stale_memory_count": stale,
            "duplicate_memory_count": duplicate,
            "stored_noise_memory_count": stored_noise,
            "useful_memory_count": useful,
            "storage_tokens_used": sum(memory.estimated_tokens for memory in memories),
        }
    )
