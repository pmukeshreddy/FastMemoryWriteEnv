"""Streaming evaluator for Phase 4."""

from __future__ import annotations

import json
import tempfile
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pydantic import Field

from fast_memory_write_env.actions import (
    AnswerAction,
    PolicyPlanError,
    SearchMemoryAction,
    StoreRawAction,
    compile_policy_actions,
)
from fast_memory_write_env.config import load_pinecone_config
from fast_memory_write_env.embeddings import (
    EmbeddingClient,
    EmbeddingClientError,
    OpenAIEmbeddingClient,
)
from fast_memory_write_env.env import FastMemoryWriteEnv
from fast_memory_write_env.hybrid_index import HybridRetrievalIndex
from fast_memory_write_env.in_memory_index import InMemoryIndex
from fast_memory_write_env.llm_client import LLMClient, LLMClientError
from fast_memory_write_env.metrics import (
    AggregateCounterSnapshot,
    AggregateMetrics,
    FactLifecycle,
    QueryMetricRecord,
    RunConfig,
    RolloutRecord,
    aggregate_metrics,
    evaluate_query_result,
    headline_metrics,
    write_eval_summary,
    write_metrics_csv,
    write_rollout_jsonl,
)
from fast_memory_write_env.pinecone_index import PineconeIndex
from fast_memory_write_env.policies import MemoryWritePolicy
from fast_memory_write_env.rewards import ScoreBreakdown, score_metrics
from fast_memory_write_env.schemas import EventCategory, MemoryRecord, MemoryStatus, RawEvent, StreamingEpisode, StrictBaseModel
from fast_memory_write_env.state import MemoryWriteQueue
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


class _AsyncMemoryWriteWorker:
    """Background worker that turns queued raw events into memory actions."""

    def __init__(
        self,
        *,
        env: FastMemoryWriteEnv,
        policy: MemoryWritePolicy,
        queue: MemoryWriteQueue,
        episode_id: str,
        latency_budget_ms: int,
        storage_budget_tokens_remaining: int,
        indexing_budget_operations_remaining: int,
        records: list[RolloutRecord],
        recent_events: list[RawEvent],
        fact_lifecycles: dict[str, FactLifecycle],
        counters: AggregateCounterSnapshot,
        indexed_memory_available_at_ms: dict[str, float],
        state_lock: threading.RLock,
    ) -> None:
        self.env = env
        self.policy = policy
        self.queue = queue
        self.episode_id = episode_id
        self.latency_budget_ms = latency_budget_ms
        self.storage_budget_tokens_remaining = storage_budget_tokens_remaining
        self.indexing_budget_operations_remaining = indexing_budget_operations_remaining
        self.records = records
        self.recent_events = recent_events
        self.fact_lifecycles = fact_lifecycles
        self.counters = counters
        self.indexed_memory_available_at_ms = indexed_memory_available_at_ms
        self.state_lock = state_lock
        self.logical_time_ms = 0.0
        self._error: BaseException | None = None
        self._thread = threading.Thread(
            target=self._run,
            name=f"memory-write-worker-{episode_id}",
            daemon=True,
        )

    def start(self) -> None:
        self._thread.start()

    def stop(self, timeout_seconds: float | None = None) -> None:
        self.queue.close()
        self._thread.join(timeout=timeout_seconds)
        if self._thread.is_alive():
            raise TimeoutError("memory-write worker did not stop before timeout")
        if self._error is not None:
            raise RuntimeError("memory-write worker failed") from self._error

    def _run(self) -> None:
        while True:
            item = self.queue.get_next(timeout_seconds=0.1)
            if item is None:
                if self.queue.is_closed_and_drained():
                    return
                continue
            try:
                self._process_item(item)
            except BaseException as exc:  # pragma: no cover - surfaced by stop()
                self._error = exc
                return
            finally:
                self.queue.task_done()

    def _process_item(self, queue_item: Any) -> None:
        started_at_ms = max(self.logical_time_ms, float(queue_item.enqueued_at_ms))
        self.logical_time_ms = started_at_ms
        with self.state_lock:
            budget_snapshot = self.env.budget_snapshot()
            self.records.append(
                RolloutRecord(
                    record_type="queue",
                    episode_id=self.episode_id,
                    timestamp_ms=float(queue_item.event.timestamp_ms),
                    logical_time_ms=self.logical_time_ms,
                    payload={
                        "event_id": queue_item.event.event_id,
                        "queue_item_id": queue_item.queue_item_id,
                        "queue_event": "started",
                        "enqueued_at_ms": queue_item.enqueued_at_ms,
                        "started_at_ms": started_at_ms,
                        "queue_wait_ms": started_at_ms - float(queue_item.enqueued_at_ms),
                        "pending_event_ids": self.queue.pending_event_ids(),
                        "budgets": budget_snapshot,
                    },
                )
            )
            recent_events = list(self.recent_events[-5:])

        budget_snapshot = self.env.budget_snapshot()
        active_memories = self.env.memory_store.list_active()
        active_memory_ids = [memory.memory_id for memory in active_memories]
        try:
            proposals = self.policy.decide(
                new_event=queue_item.event,
                active_memories=active_memories,
                recent_events=recent_events,
                latency_budget_ms=self.latency_budget_ms,
                storage_budget_tokens_remaining=_budget_value(
                    budget_snapshot["storage_budget_tokens_remaining"],
                    fallback=self.storage_budget_tokens_remaining,
                ),
                indexing_budget_operations_remaining=_budget_value(
                    budget_snapshot["indexing_budget_operations_remaining"],
                    fallback=self.indexing_budget_operations_remaining,
                ),
            )
            actions = compile_policy_actions(
                proposals,
                active_memory_ids=active_memory_ids,
            )
        except PolicyPlanError as exc:
            with self.state_lock:
                self.records.append(
                    RolloutRecord(
                        record_type="action",
                        episode_id=self.episode_id,
                        timestamp_ms=float(queue_item.event.timestamp_ms),
                        logical_time_ms=self.logical_time_ms,
                        payload={
                            "policy_plan_error": str(exc),
                            "event_id": queue_item.event.event_id,
                            "active_memory_ids": active_memory_ids,
                        },
                    )
                )
            actions = []
        for action in actions:
            result = self.env.execute_action_at(action, current_time_ms=int(self.logical_time_ms))
            self.logical_time_ms += result.latency_ms
            action_payload = action.model_dump(mode="json")
            with self.state_lock:
                self.records.append(
                    _action_record(
                        self.episode_id,
                        queue_item.event.timestamp_ms,
                        self.logical_time_ms,
                        result,
                        proposed_action=action_payload,
                    )
                )
                if result.success:
                    _update_index_availability(
                        action_type=result.action_type.value,
                        result_payload=result.payload,
                        indexed_memory_available_at_ms=self.indexed_memory_available_at_ms,
                        logical_time_ms=self.logical_time_ms,
                    )
                    _update_timelines_for_action(
                        action_type=result.action_type.value,
                        action_payload=action_payload,
                        result_payload=result.payload,
                        env=self.env,
                        fact_lifecycles=self.fact_lifecycles,
                        logical_time_ms=self.logical_time_ms,
                    )
                    _update_latency_counters(result.action_type.value, result.latency_ms, result.payload, self.counters)
                if (
                    result.action_type.value == "ignore_event"
                    and queue_item.event.category in USEFUL_EVENT_CATEGORIES
                    and queue_item.event.facts
                ):
                    self.counters.ignored_useful_event_count += 1

        with self.state_lock:
            self.recent_events.append(queue_item.event)
            self.records.append(
                RolloutRecord(
                    record_type="queue",
                    episode_id=self.episode_id,
                    timestamp_ms=float(queue_item.event.timestamp_ms),
                    logical_time_ms=self.logical_time_ms,
                    payload={
                        "event_id": queue_item.event.event_id,
                        "queue_item_id": queue_item.queue_item_id,
                        "queue_event": "completed",
                        "completed_at_ms": self.logical_time_ms,
                        "pending_event_ids": self.queue.pending_event_ids(),
                    },
                )
            )


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
        judge_llm_client: LLMClient | None = None,
        answer_llm_client: LLMClient | None = None,
    ) -> None:
        self.env = env
        self.policy = policy
        self.latency_budget_ms = latency_budget_ms
        self.storage_budget_tokens_remaining = storage_budget_tokens_remaining
        self.indexing_budget_operations_remaining = indexing_budget_operations_remaining
        self.run_config = RunConfig.model_validate(run_config or {})
        # Default the answer-correctness judge to the policy's LLM client when
        # one is available (LLMMemoryWritePolicy exposes ``llm_client``).
        # Baselines and rule-based test policies have no client, so scoring
        # falls back to the legacy string-match verifier.
        self.judge_llm_client: LLMClient | None = (
            judge_llm_client
            if judge_llm_client is not None
            else getattr(policy, "llm_client", None)
        )
        # The env composes per-query answers through an LLM. When a caller
        # constructs the env directly (legacy / unit-test path) and the
        # policy carries a usable client, propagate it so the env is fully
        # configured end-to-end. An explicit override always wins.
        resolved_answer_client = (
            answer_llm_client
            if answer_llm_client is not None
            else getattr(policy, "llm_client", None)
        )
        if resolved_answer_client is not None and getattr(env, "answer_llm_client", None) is None:
            env.answer_llm_client = resolved_answer_client

    @classmethod
    def with_local_test_index(
        cls,
        *,
        policy: MemoryWritePolicy,
        work_dir: str | Path | None = None,
        run_config: RunConfig | dict[str, Any] | None = None,
        answer_llm_client: LLMClient | None = None,
    ) -> StreamingEvaluator:
        base = Path(work_dir) if work_dir is not None else Path(tempfile.mkdtemp(prefix="fmwe-eval-"))
        memory_store = MemoryStore(base / "memory.sqlite")
        env = FastMemoryWriteEnv(
            raw_event_store=RawEventStore(base / "raw.sqlite"),
            memory_store=memory_store,
            retrieval_index=HybridRetrievalIndex(
                vector_index=InMemoryIndex(),
                memory_store=memory_store,
            ),
            answer_llm_client=answer_llm_client
            if answer_llm_client is not None
            else getattr(policy, "llm_client", None),
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
        embedding_client: EmbeddingClient | None = None,
        answer_llm_client: LLMClient | None = None,
    ) -> StreamingEvaluator:
        try:
            embedding_client = embedding_client or OpenAIEmbeddingClient.from_env()
        except EmbeddingClientError as exc:
            raise RuntimeError(
                "Pinecone runs require an embedding client; "
                "set OPENAI_API_KEY (and optionally OPENAI_EMBEDDING_MODEL/"
                "OPENAI_EMBEDDING_DIMENSION), or pass embedding_client explicitly."
            ) from exc
        config = load_pinecone_config(
            required=True,
            dimension=embedding_client.dimension,
        )
        assert config is not None
        memory_store = MemoryStore(Path(work_dir) / "memory.sqlite")
        env = FastMemoryWriteEnv(
            raw_event_store=RawEventStore(Path(work_dir) / "raw.sqlite"),
            memory_store=memory_store,
            retrieval_index=HybridRetrievalIndex(
                vector_index=PineconeIndex(config, embedding_client=embedding_client),
                memory_store=memory_store,
            ),
            answer_llm_client=answer_llm_client
            if answer_llm_client is not None
            else getattr(policy, "llm_client", None),
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
        memory_write_queue = MemoryWriteQueue()
        state_lock = threading.RLock()
        fact_lifecycles: dict[str, FactLifecycle] = {}
        event_categories: dict[str, EventCategory] = {}
        indexed_memory_available_at_ms: dict[str, float] = {}
        logical_time_ms = 0.0
        counters = AggregateCounterSnapshot()
        self.env.set_budgets(
            storage_budget_tokens_remaining=self.storage_budget_tokens_remaining,
            indexing_budget_operations_remaining=self.indexing_budget_operations_remaining,
        )
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
        worker = _AsyncMemoryWriteWorker(
            env=self.env,
            policy=self.policy,
            queue=memory_write_queue,
            episode_id=episode.episode_id,
            latency_budget_ms=self.latency_budget_ms,
            storage_budget_tokens_remaining=self.storage_budget_tokens_remaining,
            indexing_budget_operations_remaining=self.indexing_budget_operations_remaining,
            records=records,
            recent_events=recent_events,
            fact_lifecycles=fact_lifecycles,
            counters=counters,
            indexed_memory_available_at_ms=indexed_memory_available_at_ms,
            state_lock=state_lock,
        )
        records.append(
            RolloutRecord(
                record_type="run_config",
                episode_id=episode.episode_id,
                logical_time_ms=logical_time_ms,
                payload=run_config.model_dump(mode="json"),
            )
        )
        worker.start()

        try:
            for item in episode.stream:
                logical_time_ms = max(logical_time_ms, float(item.timestamp_ms))
                if item.item_type == "event":
                    event = item.event
                    with state_lock:
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

                    raw_result = self.env.execute_action_at(
                        StoreRawAction(event=event),
                        current_time_ms=item.timestamp_ms,
                    )
                    logical_time_ms += raw_result.latency_ms
                    with state_lock:
                        for fact in event.facts:
                            fact_lifecycles[fact.fact_id].raw_written_at_ms = logical_time_ms
                        records.append(_action_record(episode.episode_id, event.timestamp_ms, logical_time_ms, raw_result))

                    queue_item = memory_write_queue.enqueue(
                        event=event,
                        enqueued_at_ms=int(logical_time_ms),
                    )
                    with state_lock:
                        records.append(
                            RolloutRecord(
                                record_type="queue",
                                episode_id=episode.episode_id,
                                timestamp_ms=float(event.timestamp_ms),
                                logical_time_ms=logical_time_ms,
                                payload={
                                    "event_id": event.event_id,
                                    "queue_item_id": queue_item.queue_item_id,
                                    "queue_event": "enqueued",
                                    "enqueued_at_ms": queue_item.enqueued_at_ms,
                                    "pending_event_ids": memory_write_queue.pending_event_ids(),
                                },
                            )
                        )
                else:
                    query = item.query
                    if not memory_write_queue.wait_until_no_ready_work(
                        cutoff_enqueued_at_ms=query.timestamp_ms,
                        timeout_seconds=30.0,
                    ):
                        raise TimeoutError("memory-write queue did not process ready work before query evaluation")
                    search_result = self.env.execute_action_at(
                        SearchMemoryAction(
                            query_id=query.query_id,
                            query_text=query.text,
                            top_k=5,
                            as_of_ms=float(query.timestamp_ms),
                        ),
                        current_time_ms=query.timestamp_ms,
                    )
                    logical_time_ms += search_result.latency_ms
                    with state_lock:
                        records.append(_action_record(episode.episode_id, query.timestamp_ms, logical_time_ms, search_result))
                    retrieved_memory_ids = [
                        hit["memory_id"]
                        for hit in search_result.payload.get("results", [])
                        if isinstance(hit, dict) and "memory_id" in hit
                    ]
                    retrieved_memories = _memories_for_ids(
                        retrieved_memory_ids,
                        search_results=self.env.last_search_results,
                    )
                    with state_lock:
                        for memory in retrieved_memories:
                            for fact_id in _fact_ids_for_event_ids(memory.source_event_ids, fact_lifecycles):
                                if fact_id in fact_lifecycles:
                                    fact_lifecycles[fact_id].retrieved_at_ms = logical_time_ms

                    answer_result = self.env.execute_action_at(
                        AnswerAction(
                            query_id=query.query_id,
                            query_text=query.text,
                            retrieved_memory_ids=retrieved_memory_ids,
                        ),
                        current_time_ms=int(logical_time_ms),
                    )
                    logical_time_ms += answer_result.latency_ms
                    with state_lock:
                        records.append(_action_record(episode.episode_id, query.timestamp_ms, logical_time_ms, answer_result))
                        counters.query_latencies_ms.append(search_result.latency_ms + answer_result.latency_ms)

                    cited_memory_ids = [
                        memory_id
                        for memory_id in answer_result.payload.get("cited_memory_ids", [])
                        if isinstance(memory_id, str)
                    ]
                    cited_memories = _memories_for_ids(
                        cited_memory_ids,
                        search_results=self.env.last_search_results,
                    )
                    with state_lock:
                        try:
                            metric = evaluate_query_result(
                                query=query,
                                cited_memories=cited_memories,
                                retrieved_memory_ids=retrieved_memory_ids,
                                fact_lifecycles=fact_lifecycles,
                                answer_text=str(answer_result.payload.get("answer", "")),
                                answer_completed_at_ms=logical_time_ms,
                                llm_client=self.judge_llm_client,
                            )
                        except LLMClientError as exc:
                            # Single judge path failed; record the query as a
                            # structured failure rather than crash the worker
                            # or silently flip the verdict via substring match.
                            metric = _judge_failure_query_metric(
                                query=query,
                                cited_memory_ids=cited_memory_ids,
                                retrieved_memory_ids=retrieved_memory_ids,
                                answer_text=str(answer_result.payload.get("answer", "")),
                                error=str(exc),
                            )
                            records.append(
                                RolloutRecord(
                                    record_type="action",
                                    episode_id=episode.episode_id,
                                    timestamp_ms=float(query.timestamp_ms),
                                    logical_time_ms=logical_time_ms,
                                    payload={
                                        "judge_error": str(exc),
                                        "query_id": query.query_id,
                                    },
                                )
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
        finally:
            worker.stop(timeout_seconds=30.0)

        with state_lock:
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
                rollout_records=list(records),
                query_metrics=list(query_metrics),
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
    predictions_path = output_path / "predictions.jsonl"
    write_rollout_jsonl(result.rollout_records, raw_path)
    write_metrics_csv(result.query_metrics, result.aggregate_metrics, metrics_path)
    write_predictions_jsonl(result.rollout_records, predictions_path)
    diagnostics = build_failure_diagnostics(result.query_metrics, result.rollout_records)
    summary = {
        "episode_id": result.episode_id,
        "metrics": headline_metrics(result.aggregate_metrics),
        "score": result.score_breakdown.score,
        "run_config": result.run_config.model_dump(mode="json"),
        "counts": {
            "rollout_records": len(result.rollout_records),
            "query_metrics": len(result.query_metrics),
        },
        "diagnostics": diagnostics,
        "output_paths": {
            "raw_rollouts": str(raw_path),
            "metrics_csv": str(metrics_path),
            "eval_summary": str(summary_path),
            "predictions_jsonl": str(predictions_path),
        },
    }
    write_eval_summary(summary, summary_path)
    return result.model_copy(
        update={
            "output_paths": {
                "raw_rollouts": str(raw_path),
                "metrics_csv": str(metrics_path),
                "eval_summary": str(summary_path),
                "predictions_jsonl": str(predictions_path),
            }
        }
    )


def build_failure_diagnostics(
    query_metrics: list[QueryMetricRecord],
    rollout_records: list[RolloutRecord],
) -> dict[str, Any]:
    """Build per-query failure-stage diagnostics for ``eval_summary.json``.

    For each query that did not return ``answer_success``, report the earliest
    pipeline stage that failed (raw-write, memory-write, indexing, retrieval,
    citation, answer). Also surface counts of action-execution failures and
    plan-validation failures observed during the rollout.
    """

    failure_stage_counts: dict[str, int] = {
        "raw_written": 0,
        "memory_written": 0,
        "indexed": 0,
        "retrieved": 0,
        "cited": 0,
        "answered": 0,
    }
    query_failures: list[dict[str, Any]] = []
    for metric in query_metrics:
        if metric.answer_success:
            continue
        stage_status = {
            "raw_written": metric.time_to_raw_write is not None,
            "memory_written": metric.time_to_memory_write is not None,
            "indexed": metric.time_to_indexed_memory is not None,
            "retrieved": bool(metric.retrieved_memory_ids),
            "cited": bool(metric.cited_memory_ids),
            "answered": metric.answer_correct,
        }
        first_missing_stage: str | None = None
        for stage in ("raw_written", "memory_written", "indexed", "retrieved", "cited", "answered"):
            if not stage_status[stage]:
                first_missing_stage = stage
                break
        if first_missing_stage is not None:
            failure_stage_counts[first_missing_stage] += 1
        query_failures.append(
            {
                "query_id": metric.query_id,
                "query_timestamp_ms": metric.query_timestamp_ms,
                "stage_status": stage_status,
                "first_missing_stage": first_missing_stage,
                "retrieved_memory_ids": list(metric.retrieved_memory_ids),
                "cited_memory_ids": list(metric.cited_memory_ids),
                "required_fact_ids": list(metric.required_fact_ids),
                "covered_fact_ids": list(metric.covered_fact_ids),
                "answer_text": metric.answer_text,
            }
        )

    action_failures: list[dict[str, Any]] = []
    plan_errors: list[dict[str, Any]] = []
    delayed_unindexed_at_end = 0
    for record in rollout_records:
        if record.record_type != "action":
            continue
        if "policy_plan_error" in record.payload:
            plan_errors.append(
                {
                    "event_id": record.payload.get("event_id"),
                    "logical_time_ms": record.logical_time_ms,
                    "error": record.payload.get("policy_plan_error"),
                    "active_memory_ids": record.payload.get("active_memory_ids", []),
                }
            )
            continue
        result_payload = record.payload.get("result")
        if not isinstance(result_payload, dict):
            continue
        if result_payload.get("success") is False:
            action_failures.append(
                {
                    "action_type": result_payload.get("action_type"),
                    "error": result_payload.get("error"),
                    "logical_time_ms": record.logical_time_ms,
                    "proposed_action": record.payload.get("proposed_action"),
                }
            )
        if result_payload.get("action_type") == "delay_index" and result_payload.get("success"):
            delayed_unindexed_at_end += 1

    return {
        "query_failure_count": len(query_failures),
        "failure_stage_counts": failure_stage_counts,
        "query_failures": query_failures,
        "action_failure_count": len(action_failures),
        "action_failures": action_failures,
        "policy_plan_error_count": len(plan_errors),
        "policy_plan_errors": plan_errors,
        "delay_index_invocations": delayed_unindexed_at_end,
    }


def write_predictions_jsonl(records: list[RolloutRecord], path: str | Path) -> None:
    """Write LongMemEval-compatible prediction rows from query rollout records."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for record in records:
            if record.record_type != "query":
                continue
            query_payload = record.payload.get("query", {})
            answer_payload = record.payload.get("answer", {})
            if not isinstance(query_payload, dict) or not isinstance(answer_payload, dict):
                continue
            metadata = query_payload.get("metadata", {})
            question_id = (
                metadata.get("longmemeval_question_id")
                if isinstance(metadata, dict)
                else None
            ) or query_payload.get("query_id")
            prediction = {
                "question_id": str(question_id),
                "hypothesis": str(answer_payload.get("answer", "")),
            }
            handle.write(json.dumps(prediction, sort_keys=True) + "\n")


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


def _memories_for_ids(
    memory_ids: list[str],
    *,
    search_results: list[Any],
) -> list[MemoryRecord]:
    results_by_id = {result.memory_id: result for result in search_results}
    memories: list[MemoryRecord] = []
    for memory_id in memory_ids:
        result = results_by_id.get(memory_id)
        if result is not None and result.memory is not None:
            memories.append(result.memory)
    return memories


def _update_timelines_for_action(
    *,
    action_type: str,
    action_payload: dict[str, Any],
    result_payload: dict[str, Any],
    env: FastMemoryWriteEnv,
    fact_lifecycles: dict[str, FactLifecycle],
    logical_time_ms: float,
) -> None:
    if action_type in {"write_memory", "update_memory"}:
        source_event_ids = [str(event_id) for event_id in action_payload.get("source_event_ids", [])]
        fact_ids = _fact_ids_for_event_ids(source_event_ids, fact_lifecycles)
        for fact_id in fact_ids:
            fact_lifecycles[fact_id].memory_written_at_ms = logical_time_ms
        if result_payload.get("indexed"):
            for fact_id in fact_ids:
                fact_lifecycles[fact_id].indexed_at_ms = logical_time_ms
    if action_type == "index_now":
        memory_id = str(action_payload.get("memory_id", ""))
        memory = env.memory_store.get(memory_id)
        if memory is not None:
            for fact_id in _fact_ids_for_event_ids(memory.source_event_ids, fact_lifecycles):
                if fact_id in fact_lifecycles:
                    fact_lifecycles[fact_id].indexed_at_ms = logical_time_ms
    if action_type == "compress_memory":
        target_id = str(result_payload.get("target_memory_id", "") or action_payload.get("target_memory_id", ""))
        memory = env.memory_store.get(target_id)
        if memory is not None:
            for fact_id in _fact_ids_for_event_ids(memory.source_event_ids, fact_lifecycles):
                if fact_id in fact_lifecycles:
                    fact_lifecycles[fact_id].memory_written_at_ms = logical_time_ms
                    if result_payload.get("indexed"):
                        fact_lifecycles[fact_id].indexed_at_ms = logical_time_ms


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


def _update_index_availability(
    *,
    action_type: str,
    result_payload: dict[str, Any],
    indexed_memory_available_at_ms: dict[str, float],
    logical_time_ms: float,
) -> None:
    if action_type in {"write_memory", "update_memory", "index_now"} and result_payload.get("indexed"):
        memory_id = result_payload.get("memory_id")
        if isinstance(memory_id, str):
            indexed_memory_available_at_ms[memory_id] = float(
                result_payload.get("available_at_ms") or logical_time_ms
            )
    if action_type == "compress_memory" and result_payload.get("indexed"):
        memory_id = result_payload.get("target_memory_id")
        if isinstance(memory_id, str):
            indexed_memory_available_at_ms[memory_id] = float(
                result_payload.get("available_at_ms") or logical_time_ms
            )
    if action_type in {"mark_stale", "delay_index"}:
        memory_id = result_payload.get("memory_id")
        if isinstance(memory_id, str):
            indexed_memory_available_at_ms.pop(memory_id, None)


def _budget_value(value: int | None, *, fallback: int) -> int:
    return fallback if value is None else value


def _judge_failure_query_metric(
    *,
    query: Any,
    cited_memory_ids: list[str],
    retrieved_memory_ids: list[str],
    answer_text: str,
    error: str,
) -> QueryMetricRecord:
    """Build a query metric that records a judge failure honestly.

    The judge's single path raised after exhausted retries. We do not invent
    an ``answer_correct`` verdict and we do not silently substring-match.
    Both ``answer_correct`` and ``answer_success`` are False; ``answer_text``
    is annotated with the judge error so downstream diagnostics can surface
    the cause.
    """

    return QueryMetricRecord(
        episode_id=query.episode_id,
        query_id=query.query_id,
        query_timestamp_ms=float(query.timestamp_ms),
        answer_success=False,
        answer_correct=False,
        evidence_correct=False,
        fact_evidence_coverage=False,
        memory_precision=0.0,
        memory_recall=0.0,
        cited_memory_ids=list(cited_memory_ids),
        retrieved_memory_ids=list(retrieved_memory_ids),
        required_fact_ids=list(query.gold.required_fact_ids),
        supporting_event_ids=list(query.gold.supporting_event_ids),
        covered_fact_ids=[],
        covered_event_ids=[],
        debug_contains_answer_facts=False,
        answer_text=f"[judge_failure] {error} :: {answer_text}",
    )


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
        if any(event_categories.get(event_id) in USEFUL_EVENT_CATEGORIES for event_id in memory.source_event_ids)
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


def _fact_ids_for_event_ids(
    source_event_ids: list[str],
    fact_lifecycles: dict[str, FactLifecycle],
) -> list[str]:
    source_ids = set(source_event_ids)
    return [
        fact_id
        for fact_id, lifecycle in fact_lifecycles.items()
        if lifecycle.source_event_id in source_ids
    ]
