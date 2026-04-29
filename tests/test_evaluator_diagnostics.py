"""End-to-end evaluator scenarios that exercise the new diagnostics.

The plan reproduces three named failure modes:

* **Ada** — a policy that emits ``update_memory`` and ``mark_stale`` on the
  same memory in one decision. The plan validator must block the second
  action and the worker must record a structured plan-error rollout instead
  of crashing.
* **Ben** — two delayed-index memories with limited budget; without
  compression they exhaust the indexing budget and queries miss.
* **Dee** — same setup as Ben but the policy compresses the two delayed
  memories into one indexable memory, so the budget covers it.
"""

from __future__ import annotations

import json
from pathlib import Path

from fast_memory_write_env.actions import (
    CompressMemoryProposal,
    MarkStaleAction,
    UpdateMemoryAction,
    WriteMemoryProposal,
    validate_policy_actions,
)
from fast_memory_write_env.env import FastMemoryWriteEnv
from fast_memory_write_env.evaluator import (
    StreamingEvaluator,
    build_failure_diagnostics,
    write_evaluation_outputs,
)
from fast_memory_write_env.hybrid_index import HybridRetrievalIndex
from fast_memory_write_env.in_memory_index import InMemoryIndex
from fast_memory_write_env.index import estimate_tokens
from fast_memory_write_env.schemas import (
    DatasetMode,
    EventCategory,
    EventFact,
    Query,
    QueryGold,
    RawEvent,
    StreamEventItem,
    StreamQueryItem,
    StreamingEpisode,
)
from fast_memory_write_env.stores import MemoryStore, RawEventStore


def _build_env(tmp_path: Path) -> FastMemoryWriteEnv:
    return FastMemoryWriteEnv(
        raw_event_store=RawEventStore(tmp_path / "raw.sqlite"),
        memory_store=MemoryStore(tmp_path / "memory.sqlite"),
        retrieval_index=InMemoryIndex(),
    )


def _build_hybrid_env(tmp_path: Path) -> FastMemoryWriteEnv:
    """Env wired with the production-shaped HybridRetrievalIndex."""

    memory_store = MemoryStore(tmp_path / "memory.sqlite")
    return FastMemoryWriteEnv(
        raw_event_store=RawEventStore(tmp_path / "raw.sqlite"),
        memory_store=memory_store,
        retrieval_index=HybridRetrievalIndex(
            vector_index=InMemoryIndex(),
            memory_store=memory_store,
        ),
    )


class _AdaUpdateAndStalePolicy:
    """Reproduces the Ada failure: update + mark_stale on same memory."""

    def __init__(self) -> None:
        self.first_call = True

    def decide(
        self,
        *,
        new_event,
        active_memories,
        recent_events,
        latency_budget_ms,
        storage_budget_tokens_remaining,
        indexing_budget_operations_remaining,
    ):
        if self.first_call:
            self.first_call = False
            return validate_policy_actions(
                [
                    {
                        "action_type": "write_memory",
                        "entity_id": new_event.entity_id,
                        "content": new_event.content,
                        "source_event_ids": [new_event.event_id],
                        "index_immediately": True,
                    }
                ]
            )
        target = active_memories[0]
        return [
            UpdateMemoryAction(
                memory_id=target.memory_id,
                content=new_event.content,
                source_event_ids=[new_event.event_id],
                reason="bad correction",
            ),
            MarkStaleAction(memory_id=target.memory_id, reason="redundant stale"),
        ]


class _BenDelayedIndexPolicy:
    """Two delayed-index memories. Indexing budget is one operation."""

    def decide(
        self,
        *,
        new_event,
        active_memories,
        recent_events,
        latency_budget_ms,
        storage_budget_tokens_remaining,
        indexing_budget_operations_remaining,
    ):
        return validate_policy_actions(
            [
                {
                    "action_type": "write_memory",
                    "entity_id": new_event.entity_id,
                    "content": new_event.content,
                    "source_event_ids": [new_event.event_id],
                    "index_immediately": False,
                }
            ]
        )


class _DeeCompressionPolicy:
    """Delays the first two memories, then compresses on the third event so the merged
    summary fits the indexing budget."""

    def __init__(self) -> None:
        self.calls = 0

    def decide(
        self,
        *,
        new_event,
        active_memories,
        recent_events,
        latency_budget_ms,
        storage_budget_tokens_remaining,
        indexing_budget_operations_remaining,
    ):
        self.calls += 1
        if self.calls <= 2:
            return validate_policy_actions(
                [
                    {
                        "action_type": "write_memory",
                        "entity_id": new_event.entity_id,
                        "content": new_event.content,
                        "source_event_ids": [new_event.event_id],
                        "index_immediately": False,
                    }
                ]
            )
        if len(active_memories) >= 2:
            return [
                CompressMemoryProposal(
                    source_memory_ids=[memory.memory_id for memory in active_memories[:2]],
                    compressed_content=(
                        "Account ben prefers email and pin acknowledgement on alerts."
                    ),
                    index_immediately=True,
                )
            ]
        return validate_policy_actions(
            [
                {
                    "action_type": "ignore_event",
                    "event_id": new_event.event_id,
                    "reason": "no compression source available",
                }
            ]
        )


def _make_event(
    *,
    event_id: str,
    timestamp_ms: int,
    content: str,
    fact_id: str,
    category: EventCategory = EventCategory.USEFUL_FACT,
) -> RawEvent:
    return RawEvent(
        event_id=event_id,
        episode_id="ep-diagnostics",
        timestamp_ms=timestamp_ms,
        source="test",
        user_id="user-main",
        entity_id="account-test",
        category=category,
        content=content,
        facts=[
            EventFact(
                fact_id=fact_id,
                entity_id="account-test",
                attribute="preference",
                value=content,
                source_event_id=event_id,
                valid_from_ms=timestamp_ms,
            )
        ],
        estimated_tokens=estimate_tokens(content),
    )


def _make_query(
    *,
    query_id: str,
    timestamp_ms: int,
    text: str,
    required_fact_ids: list[str],
    supporting_event_ids: list[str],
    answer_facts: list[str],
) -> Query:
    return Query(
        query_id=query_id,
        episode_id="ep-diagnostics",
        timestamp_ms=timestamp_ms,
        user_id="user-main",
        target_entity_id="account-test",
        text=text,
        gold=QueryGold(
            required_fact_ids=required_fact_ids,
            supporting_event_ids=supporting_event_ids,
            answer_facts=answer_facts,
        ),
    )


def _episode_for_ada() -> StreamingEpisode:
    first = _make_event(
        event_id="event-ada-1",
        timestamp_ms=0,
        content="Account ada prefers SMS for renewal notices.",
        fact_id="fact-ada-1",
    )
    second = _make_event(
        event_id="event-ada-2",
        timestamp_ms=10,
        content="Account ada now prefers email for renewal notices.",
        fact_id="fact-ada-2",
        category=EventCategory.STALE_UPDATE,
    )
    query = _make_query(
        query_id="query-ada",
        timestamp_ms=200,
        text="How should account ada be contacted for renewal notices?",
        required_fact_ids=["fact-ada-1"],
        supporting_event_ids=["event-ada-1"],
        answer_facts=["sms"],
    )
    return StreamingEpisode(
        episode_id="ep-diagnostics",
        mode=DatasetMode.SMALL,
        seed=0,
        stream=[
            StreamEventItem(timestamp_ms=first.timestamp_ms, event=first),
            StreamEventItem(timestamp_ms=second.timestamp_ms, event=second),
            StreamQueryItem(timestamp_ms=query.timestamp_ms, query=query),
        ],
    )


def _episode_for_ben_or_dee() -> StreamingEpisode:
    first = _make_event(
        event_id="event-ben-1",
        timestamp_ms=0,
        content="Account ben prefers email for renewal notices.",
        fact_id="fact-ben-1",
    )
    second = _make_event(
        event_id="event-ben-2",
        timestamp_ms=10,
        content="Account ben prefers pin acknowledgement on alerts.",
        fact_id="fact-ben-2",
    )
    third = _make_event(
        event_id="event-ben-3",
        timestamp_ms=20,
        content="Filler interaction without new account preference.",
        fact_id="fact-ben-3",
        category=EventCategory.NOISE,
    )
    query = _make_query(
        query_id="query-ben",
        timestamp_ms=400,
        text="How should account ben be contacted on alerts?",
        required_fact_ids=["fact-ben-1", "fact-ben-2"],
        supporting_event_ids=["event-ben-1", "event-ben-2"],
        answer_facts=["email", "pin"],
    )
    return StreamingEpisode(
        episode_id="ep-diagnostics",
        mode=DatasetMode.SMALL,
        seed=0,
        stream=[
            StreamEventItem(timestamp_ms=first.timestamp_ms, event=first),
            StreamEventItem(timestamp_ms=second.timestamp_ms, event=second),
            StreamEventItem(timestamp_ms=third.timestamp_ms, event=third),
            StreamQueryItem(timestamp_ms=query.timestamp_ms, query=query),
        ],
    )


def test_ada_failure_is_blocked_by_plan_validation_and_recorded(tmp_path) -> None:
    evaluator = StreamingEvaluator(
        env=_build_env(tmp_path),
        policy=_AdaUpdateAndStalePolicy(),
        storage_budget_tokens_remaining=1000,
        indexing_budget_operations_remaining=2,
    )

    result = evaluator.evaluate_episode(_episode_for_ada())

    plan_error_records = [
        record
        for record in result.rollout_records
        if record.record_type == "action" and "policy_plan_error" in record.payload
    ]
    assert len(plan_error_records) == 1
    plan_error = plan_error_records[0].payload
    assert "marked stale" in plan_error["policy_plan_error"]

    diagnostics = build_failure_diagnostics(result.query_metrics, result.rollout_records)
    assert diagnostics["policy_plan_error_count"] == 1
    assert diagnostics["policy_plan_errors"][0]["event_id"] == "event-ada-2"


def test_ben_delayed_index_leaves_query_unanswerable(tmp_path) -> None:
    evaluator = StreamingEvaluator(
        env=_build_env(tmp_path),
        policy=_BenDelayedIndexPolicy(),
        storage_budget_tokens_remaining=1000,
        indexing_budget_operations_remaining=0,
    )

    result = evaluator.evaluate_episode(_episode_for_ben_or_dee())

    metric = result.query_metrics[0]
    assert metric.answer_success is False
    assert metric.retrieved_memory_ids == []

    diagnostics = build_failure_diagnostics(result.query_metrics, result.rollout_records)
    assert diagnostics["query_failure_count"] == 1
    failure = diagnostics["query_failures"][0]
    assert failure["first_missing_stage"] in {"indexed", "retrieved"}
    assert failure["stage_status"]["raw_written"] is True
    assert failure["stage_status"]["memory_written"] is True


def test_ben_delayed_index_recovered_by_hybrid_retrieval(tmp_path) -> None:
    """Same Ben policy that fails with vector-only retrieval surfaces relevant
    memories when the env is wired with HybridRetrievalIndex, because the
    lexical FTS5 mirror serves the new memories without needing any of the
    indexing budget the policy never had.

    Top-1 answer composition cites only the single best memory, so a query
    that requires two separately-stored facts will only cover one of them
    here. That is the price of a focused answer; multi-fact recall is the
    job of ``compress_memory`` (see the Dee scenario).
    """

    evaluator = StreamingEvaluator(
        env=_build_hybrid_env(tmp_path),
        policy=_BenDelayedIndexPolicy(),
        storage_budget_tokens_remaining=1000,
        indexing_budget_operations_remaining=0,
    )

    result = evaluator.evaluate_episode(_episode_for_ben_or_dee())

    metric = result.query_metrics[0]
    assert metric.retrieved_memory_ids != [], "hybrid should serve the memory lexically"
    assert metric.cited_memory_ids != []
    # At least one required fact is covered by hybrid retrieval, demonstrating
    # the lexical path lifts the policy out of the indexing-budget cliff.
    assert set(metric.covered_fact_ids) & set(metric.required_fact_ids)


def test_dee_compression_makes_query_answerable_under_tight_budget(tmp_path) -> None:
    evaluator = StreamingEvaluator(
        env=_build_env(tmp_path),
        policy=_DeeCompressionPolicy(),
        storage_budget_tokens_remaining=2000,
        indexing_budget_operations_remaining=1,
    )

    result = evaluator.evaluate_episode(_episode_for_ben_or_dee())

    metric = result.query_metrics[0]
    assert metric.retrieved_memory_ids != []
    assert metric.cited_memory_ids != []
    # Either of the source facts being represented in the compressed memory
    # is enough for evidence to be considered correct.
    assert "email" in metric.answer_text or "pin" in metric.answer_text


def test_eval_summary_contains_per_query_failure_diagnostics(tmp_path) -> None:
    evaluator = StreamingEvaluator(
        env=_build_env(tmp_path),
        policy=_BenDelayedIndexPolicy(),
        storage_budget_tokens_remaining=1000,
        indexing_budget_operations_remaining=0,
    )
    result = evaluator.evaluate_episode(_episode_for_ben_or_dee())

    output_dir = tmp_path / "outputs"
    write_evaluation_outputs(result, output_dir)

    summary = json.loads((output_dir / "eval_summary.json").read_text())

    assert "diagnostics" in summary
    diagnostics = summary["diagnostics"]
    assert diagnostics["query_failure_count"] >= 1
    failure_stages = diagnostics["failure_stage_counts"]
    assert sum(failure_stages.values()) == diagnostics["query_failure_count"]
    assert isinstance(diagnostics["query_failures"], list)
    assert all("stage_status" in failure for failure in diagnostics["query_failures"])


def test_proposal_metadata_round_trips_into_executable_action() -> None:
    """Sanity check that compile preserves proposal metadata."""

    proposals = [
        WriteMemoryProposal(
            entity_id="account-test",
            content="alpha",
            source_event_ids=["event-1"],
            metadata={"baseline": "test"},
        )
    ]
    from fast_memory_write_env.actions import compile_policy_actions

    actions = compile_policy_actions(proposals, active_memory_ids=[])
    assert actions[0].metadata == {"baseline": "test"}
