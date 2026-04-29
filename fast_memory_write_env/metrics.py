"""Metric models and helpers for streaming memory-write evaluation."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Iterable, Literal

from pydantic import Field

from fast_memory_write_env.schemas import MemoryRecord, Query, StrictBaseModel


class RunConfig(StrictBaseModel):
    """Audit metadata describing an evaluation run."""

    dataset_mode: str | None = None
    seed: int | None = None
    episode_index: int | None = None
    episode_id: str | None = None
    policy_name: str | None = None
    llm_client_type: str | None = None
    llm_model: str | None = None
    llm_base_url: str | None = None
    llm_api_key_configured: bool | None = None
    backend_type: str | None = None
    pinecone_index_name: str | None = None
    latency_budget_ms: int | None = None
    storage_budget_tokens_remaining: int | None = None
    indexing_budget_operations_remaining: int | None = None
    timestamp_utc: str | None = None
    command: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class FactLifecycle(StrictBaseModel):
    """Logical timing path for one fact from event arrival to answer use."""

    fact_id: str
    source_event_id: str
    event_timestamp_ms: float
    raw_written_at_ms: float | None = None
    memory_written_at_ms: float | None = None
    indexed_at_ms: float | None = None
    retrieved_at_ms: float | None = None


class QueryMetricRecord(StrictBaseModel):
    """Per-query metric row."""

    episode_id: str
    query_id: str
    query_timestamp_ms: float
    time_to_raw_write: float | None = None
    time_to_memory_write: float | None = None
    time_to_indexed_memory: float | None = None
    time_to_retrieved_memory: float | None = None
    time_to_useful_memory: float | None = None
    answer_success: bool
    answer_correct: bool
    evidence_correct: bool
    fact_evidence_coverage: bool = False
    memory_precision: float = Field(ge=0.0, le=1.0)
    memory_recall: float = Field(ge=0.0, le=1.0)
    cited_memory_ids: list[str] = Field(default_factory=list)
    retrieved_memory_ids: list[str] = Field(default_factory=list)
    required_fact_ids: list[str] = Field(default_factory=list)
    supporting_event_ids: list[str] = Field(default_factory=list)
    covered_fact_ids: list[str] = Field(default_factory=list)
    covered_event_ids: list[str] = Field(default_factory=list)
    debug_contains_answer_facts: bool = False
    answer_text: str = ""


class AggregateCounterSnapshot(StrictBaseModel):
    """Counters needed to aggregate a rollout."""

    total_memory_count: int = Field(default=0, ge=0)
    stale_memory_count: int = Field(default=0, ge=0)
    duplicate_memory_count: int = Field(default=0, ge=0)
    storage_tokens_used: int = Field(default=0, ge=0)
    useful_memory_count: int = Field(default=0, ge=0)
    useful_event_count: int = Field(default=0, ge=0)
    ignored_useful_event_count: int = Field(default=0, ge=0)
    noise_event_count: int = Field(default=0, ge=0)
    stored_noise_memory_count: int = Field(default=0, ge=0)
    write_latencies_ms: list[float] = Field(default_factory=list)
    index_latencies_ms: list[float] = Field(default_factory=list)
    query_latencies_ms: list[float] = Field(default_factory=list)


class AggregateMetrics(StrictBaseModel):
    """Aggregate evaluation metrics."""

    time_to_raw_write: float | None = None
    time_to_memory_write: float | None = None
    time_to_indexed_memory: float | None = None
    time_to_retrieved_memory: float | None = None
    time_to_useful_memory: float | None = None
    answer_success: float = Field(ge=0.0, le=1.0)
    answer_correct: float = Field(ge=0.0, le=1.0)
    evidence_correct: float = Field(ge=0.0, le=1.0)
    fact_evidence_coverage: float = Field(default=0.0, ge=0.0, le=1.0)
    memory_precision: float = Field(ge=0.0, le=1.0)
    memory_recall: float = Field(ge=0.0, le=1.0)
    stale_memory_rate: float = Field(ge=0.0, le=1.0)
    duplicate_memory_rate: float = Field(ge=0.0, le=1.0)
    storage_tokens_used: int = Field(ge=0)
    useful_memory_per_storage_token: float = Field(ge=0.0)
    write_latency_p50: float | None = None
    write_latency_p95: float | None = None
    index_latency_p50: float | None = None
    index_latency_p95: float | None = None
    query_latency_p95: float | None = None
    ignored_useful_fact_rate: float = Field(ge=0.0, le=1.0)
    stored_noise_rate: float = Field(ge=0.0, le=1.0)
    query_count: int = Field(default=0, ge=0)


HEADLINE_METRIC_FIELDS = (
    "time_to_useful_memory",
    "answer_success",
    "memory_precision",
    "memory_recall",
    "storage_tokens_used",
)

METRICS_CSV_FIELDS = (
    "episode_id",
    "query_id",
    "query_timestamp_ms",
    *HEADLINE_METRIC_FIELDS,
)


class RolloutRecord(StrictBaseModel):
    """JSONL-safe rollout record."""

    record_type: Literal["run_config", "event", "queue", "action", "query", "query_metric", "aggregate_inputs"]
    episode_id: str
    timestamp_ms: float | None = None
    logical_time_ms: float | None = None
    payload: dict[str, Any] = Field(default_factory=dict)


def percentile(values: Iterable[float | int], q: float) -> float | None:
    """Return an interpolated percentile, or None for no values."""

    sorted_values = sorted(float(value) for value in values)
    if not sorted_values:
        return None
    if q <= 0:
        return sorted_values[0]
    if q >= 100:
        return sorted_values[-1]
    position = (len(sorted_values) - 1) * (q / 100.0)
    lower = int(position)
    upper = min(lower + 1, len(sorted_values) - 1)
    if lower == upper:
        return sorted_values[lower]
    fraction = position - lower
    return sorted_values[lower] + ((sorted_values[upper] - sorted_values[lower]) * fraction)


def evaluate_query_result(
    *,
    query: Query,
    cited_memories: list[MemoryRecord],
    retrieved_memory_ids: list[str],
    fact_lifecycles: dict[str, FactLifecycle],
    answer_text: str,
    answer_completed_at_ms: float,
) -> QueryMetricRecord:
    """Score one query using fact/evidence coverage instead of exact matching."""

    required_fact_ids = set(query.gold.required_fact_ids)
    supporting_event_ids = set(query.gold.supporting_event_ids)
    event_fact_ids = _event_fact_ids_from_lifecycles(fact_lifecycles)
    cited_fact_ids = {
        fact_id
        for memory in cited_memories
        for event_id in memory.source_event_ids
        for fact_id in event_fact_ids.get(event_id, [])
    }
    cited_event_ids = {event_id for memory in cited_memories for event_id in memory.source_event_ids}
    covered_fact_ids = sorted(required_fact_ids & cited_fact_ids)
    covered_event_ids = sorted(supporting_event_ids & cited_event_ids)

    if query.gold.is_abstention:
        fact_evidence_coverage = not cited_memories
        evidence_correct = not cited_memories
        answer_correct = answer_is_abstention(answer_text)
        answer_success = answer_correct and evidence_correct
        memory_precision = 1.0 if not cited_memories else 0.0
        memory_recall = 1.0
        breakdown = compute_time_breakdown(
            required_fact_ids=query.gold.required_fact_ids,
            fact_lifecycles=fact_lifecycles,
            answer_completed_at_ms=answer_completed_at_ms,
            answer_success=answer_success,
        )
        return QueryMetricRecord(
            episode_id=query.episode_id,
            query_id=query.query_id,
            query_timestamp_ms=float(query.timestamp_ms),
            answer_success=answer_success,
            answer_correct=answer_correct,
            evidence_correct=evidence_correct,
            fact_evidence_coverage=fact_evidence_coverage,
            memory_precision=memory_precision,
            memory_recall=memory_recall,
            cited_memory_ids=[memory.memory_id for memory in cited_memories],
            retrieved_memory_ids=retrieved_memory_ids,
            required_fact_ids=query.gold.required_fact_ids,
            supporting_event_ids=query.gold.supporting_event_ids,
            covered_fact_ids=covered_fact_ids,
            covered_event_ids=covered_event_ids,
            debug_contains_answer_facts=_contains_all_answer_facts(answer_text, query.gold.answer_facts),
            answer_text=answer_text,
            **breakdown,
        )

    fact_evidence_coverage = required_fact_ids.issubset(cited_fact_ids)
    evidence_correct = fact_evidence_coverage and supporting_event_ids.issubset(cited_event_ids)
    answer_correct = answer_satisfies_gold_facts(answer_text, query.gold.answer_facts)
    answer_success = answer_correct and evidence_correct

    useful_retrieved_count = sum(
        1
        for memory in cited_memories
        if required_fact_ids.intersection(
            fact_id
            for event_id in memory.source_event_ids
            for fact_id in event_fact_ids.get(event_id, [])
        )
        or supporting_event_ids.intersection(memory.source_event_ids)
    )
    memory_precision = _safe_div(useful_retrieved_count, len(cited_memories))
    memory_recall = _safe_div(len(covered_fact_ids), len(required_fact_ids))
    breakdown = compute_time_breakdown(
        required_fact_ids=query.gold.required_fact_ids,
        fact_lifecycles=fact_lifecycles,
        answer_completed_at_ms=answer_completed_at_ms,
        answer_success=answer_success,
    )

    return QueryMetricRecord(
        episode_id=query.episode_id,
        query_id=query.query_id,
        query_timestamp_ms=float(query.timestamp_ms),
        answer_success=answer_success,
        answer_correct=answer_correct,
        evidence_correct=evidence_correct,
        fact_evidence_coverage=fact_evidence_coverage,
        memory_precision=memory_precision,
        memory_recall=memory_recall,
        cited_memory_ids=[memory.memory_id for memory in cited_memories],
        retrieved_memory_ids=retrieved_memory_ids,
        required_fact_ids=query.gold.required_fact_ids,
        supporting_event_ids=query.gold.supporting_event_ids,
        covered_fact_ids=covered_fact_ids,
        covered_event_ids=covered_event_ids,
        debug_contains_answer_facts=_contains_all_answer_facts(answer_text, query.gold.answer_facts),
        answer_text=answer_text,
        **breakdown,
    )


def compute_time_breakdown(
    *,
    required_fact_ids: list[str],
    fact_lifecycles: dict[str, FactLifecycle],
    answer_completed_at_ms: float,
    answer_success: bool,
) -> dict[str, float | None]:
    """Compute slowest required-fact timing path."""

    lifecycles = [fact_lifecycles.get(fact_id) for fact_id in required_fact_ids]
    if not lifecycles or any(lifecycle is None for lifecycle in lifecycles):
        return {
            "time_to_raw_write": None,
            "time_to_memory_write": None,
            "time_to_indexed_memory": None,
            "time_to_retrieved_memory": None,
            "time_to_useful_memory": None,
        }

    typed_lifecycles = [lifecycle for lifecycle in lifecycles if lifecycle is not None]
    time_to_raw_write = _max_delta(typed_lifecycles, "raw_written_at_ms")
    time_to_memory_write = _max_delta(typed_lifecycles, "memory_written_at_ms")
    time_to_indexed_memory = _max_delta(typed_lifecycles, "indexed_at_ms")
    time_to_retrieved_memory = _max_delta(typed_lifecycles, "retrieved_at_ms")
    full_chain_present = all(
        value is not None
        for value in [
            time_to_raw_write,
            time_to_memory_write,
            time_to_indexed_memory,
            time_to_retrieved_memory,
        ]
    )
    return {
        "time_to_raw_write": time_to_raw_write,
        "time_to_memory_write": time_to_memory_write,
        "time_to_indexed_memory": time_to_indexed_memory,
        "time_to_retrieved_memory": time_to_retrieved_memory,
        "time_to_useful_memory": (
            max(answer_completed_at_ms - lifecycle.event_timestamp_ms for lifecycle in typed_lifecycles)
            if answer_success and full_chain_present
            else None
        ),
    }


def aggregate_metrics(
    query_metrics: list[QueryMetricRecord],
    counters: AggregateCounterSnapshot,
) -> AggregateMetrics:
    """Aggregate per-query records and rollout counters."""

    return AggregateMetrics(
        time_to_raw_write=_mean_present(metric.time_to_raw_write for metric in query_metrics),
        time_to_memory_write=_mean_present(metric.time_to_memory_write for metric in query_metrics),
        time_to_indexed_memory=_mean_present(metric.time_to_indexed_memory for metric in query_metrics),
        time_to_retrieved_memory=_mean_present(metric.time_to_retrieved_memory for metric in query_metrics),
        time_to_useful_memory=_mean_present(metric.time_to_useful_memory for metric in query_metrics),
        answer_success=_mean_bool(metric.answer_success for metric in query_metrics),
        answer_correct=_mean_bool(metric.answer_correct for metric in query_metrics),
        evidence_correct=_mean_bool(metric.evidence_correct for metric in query_metrics),
        fact_evidence_coverage=_mean_bool(metric.fact_evidence_coverage for metric in query_metrics),
        memory_precision=_mean_present(metric.memory_precision for metric in query_metrics) or 0.0,
        memory_recall=_mean_present(metric.memory_recall for metric in query_metrics) or 0.0,
        stale_memory_rate=_safe_div(counters.stale_memory_count, counters.total_memory_count),
        duplicate_memory_rate=_safe_div(counters.duplicate_memory_count, counters.total_memory_count),
        storage_tokens_used=counters.storage_tokens_used,
        useful_memory_per_storage_token=_safe_div(
            counters.useful_memory_count,
            counters.storage_tokens_used,
        ),
        write_latency_p50=percentile(counters.write_latencies_ms, 50),
        write_latency_p95=percentile(counters.write_latencies_ms, 95),
        index_latency_p50=percentile(counters.index_latencies_ms, 50),
        index_latency_p95=percentile(counters.index_latencies_ms, 95),
        query_latency_p95=percentile(counters.query_latencies_ms, 95),
        ignored_useful_fact_rate=_safe_div(
            counters.ignored_useful_event_count,
            counters.useful_event_count,
        ),
        stored_noise_rate=_safe_div(
            counters.stored_noise_memory_count,
            counters.noise_event_count,
        ),
        query_count=len(query_metrics),
    )


def headline_metrics(metrics: AggregateMetrics) -> dict[str, Any]:
    """Return the small scorecard used for top-level reporting."""

    payload = metrics.model_dump(mode="json")
    return {field: payload[field] for field in HEADLINE_METRIC_FIELDS}


def summarize_rollout_records(records: list[RolloutRecord]) -> tuple[list[QueryMetricRecord], AggregateMetrics]:
    """Rebuild query metrics and aggregate summary from rollout JSONL records."""

    query_metrics = [
        QueryMetricRecord.model_validate(record.payload)
        for record in records
        if record.record_type == "query_metric"
    ]
    counter_records = [record for record in records if record.record_type == "aggregate_inputs"]
    counters = (
        AggregateCounterSnapshot.model_validate(counter_records[-1].payload)
        if counter_records
        else AggregateCounterSnapshot()
    )
    return query_metrics, aggregate_metrics(query_metrics, counters)


def extract_run_config(records: list[RolloutRecord]) -> RunConfig | None:
    """Return the latest run config record from rollout logs."""

    configs = [record for record in records if record.record_type == "run_config"]
    if not configs:
        return None
    return RunConfig.model_validate(configs[-1].payload)


def write_rollout_jsonl(records: list[RolloutRecord], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(record.model_dump_json() + "\n")


def read_rollout_jsonl(path: str | Path) -> list[RolloutRecord]:
    records: list[RolloutRecord] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                records.append(RolloutRecord.model_validate_json(line))
    return records


def write_metrics_csv(
    query_metrics: list[QueryMetricRecord],
    aggregate: AggregateMetrics,
    path: str | Path,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for metric in query_metrics:
        rows.append(
            {
                "episode_id": metric.episode_id,
                "query_id": metric.query_id,
                "query_timestamp_ms": metric.query_timestamp_ms,
                "time_to_useful_memory": metric.time_to_useful_memory,
                "answer_success": metric.answer_success,
                "memory_precision": metric.memory_precision,
                "memory_recall": metric.memory_recall,
                "storage_tokens_used": aggregate.storage_tokens_used,
            }
        )

    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(METRICS_CSV_FIELDS))
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _csv_value(value) for key, value in row.items()})


def write_eval_summary(summary: dict[str, Any], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")


def _max_delta(lifecycles: list[FactLifecycle], attr: str) -> float | None:
    values: list[float] = []
    for lifecycle in lifecycles:
        timestamp = getattr(lifecycle, attr)
        if timestamp is None:
            return None
        values.append(float(timestamp) - lifecycle.event_timestamp_ms)
    return max(values) if values else None


def _mean_present(values: Iterable[float | int | None]) -> float | None:
    present = [float(value) for value in values if value is not None]
    if not present:
        return None
    return sum(present) / len(present)


def _mean_bool(values: Iterable[bool]) -> float:
    items = list(values)
    if not items:
        return 0.0
    return sum(1 for value in items if value) / len(items)


def _safe_div(numerator: int | float, denominator: int | float) -> float:
    if denominator == 0:
        return 0.0
    return float(numerator) / float(denominator)


def _contains_all_answer_facts(answer_text: str, answer_facts: list[str]) -> bool:
    normalized_answer = _normalize_debug_text(answer_text)
    return all(_normalize_debug_text(fact) in normalized_answer for fact in answer_facts)


def answer_satisfies_gold_facts(answer_text: str, answer_facts: list[str]) -> bool:
    """Conservative deterministic answer verifier for gold fact strings."""

    if not answer_facts:
        return False
    normalized_answer = _normalize_debug_text(answer_text)
    answer_tokens = set(_semantic_tokens(answer_text))
    if not answer_tokens:
        return False
    for fact in answer_facts:
        normalized_fact = _normalize_debug_text(fact)
        if normalized_fact and normalized_fact in normalized_answer:
            continue
        fact_tokens = set(_semantic_tokens(fact))
        if not fact_tokens:
            return False
        coverage = len(fact_tokens & answer_tokens) / len(fact_tokens)
        if coverage < 0.65:
            return False
    return True


def answer_is_abstention(answer_text: str) -> bool:
    """Return whether an answer clearly abstains from unsupported memory."""

    normalized = _normalize_debug_text(answer_text)
    markers = [
        "do not know",
        "don't know",
        "not enough information",
        "not enough info",
        "insufficient information",
        "not mentioned",
        "cannot answer",
        "can't answer",
        "no information",
        "unknown",
    ]
    return any(marker in normalized for marker in markers)


def _event_fact_ids_from_lifecycles(
    fact_lifecycles: dict[str, FactLifecycle],
) -> dict[str, list[str]]:
    by_event: dict[str, list[str]] = {}
    for fact_id, lifecycle in fact_lifecycles.items():
        by_event.setdefault(lifecycle.source_event_id, []).append(fact_id)
    return by_event


def _normalize_debug_text(value: str) -> str:
    return " ".join(value.lower().split())


def _semantic_tokens(value: str) -> list[str]:
    stop_words = {
        "a",
        "an",
        "and",
        "are",
        "at",
        "be",
        "by",
        "for",
        "from",
        "has",
        "have",
        "is",
        "it",
        "now",
        "of",
        "on",
        "or",
        "right",
        "should",
        "the",
        "to",
        "was",
        "were",
        "with",
    }
    cleaned = []
    for raw_token in value.lower().replace(":", " ").replace(";", " ").replace(".", " ").split():
        token = "".join(char for char in raw_token if char.isalnum() or char == "-")
        token = _normalize_token(token)
        if token and token not in stop_words:
            cleaned.append(token)
    return cleaned


def _normalize_token(token: str) -> str:
    replacements = {
        "contacted": "contact",
        "notices": "notice",
        "prefers": "prefer",
        "reported": "report",
        "routed": "route",
        "routing": "route",
        "alerts": "alert",
    }
    if token in replacements:
        return replacements[token]
    if token.endswith("s") and len(token) > 4:
        return token[:-1]
    return token


def _csv_value(value: Any) -> Any:
    if isinstance(value, (list, dict)):
        return json.dumps(value, sort_keys=True)
    return value
