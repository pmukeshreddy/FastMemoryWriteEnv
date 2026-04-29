"""Tests for the new ``update_memory`` semantics.

Updates no longer auto-spend the indexing budget. When the LLM passes
``index_immediately=False`` on a previously indexed memory, the env drops the
stale vector and marks ``needs_reindex=True`` instead of silently re-upserting.
A subsequent ``index_now`` (which the LLM can see is required because
``needs_reindex`` is exposed in active memories) restores semantic retrieval.
"""

from __future__ import annotations

from fast_memory_write_env.actions import (
    IndexNowAction,
    UpdateMemoryAction,
    WriteMemoryAction,
)
from fast_memory_write_env.env import FastMemoryWriteEnv
from fast_memory_write_env.hybrid_index import HybridRetrievalIndex
from fast_memory_write_env.in_memory_index import InMemoryIndex
from fast_memory_write_env.policies import policy_visible_memory
from fast_memory_write_env.schemas import EventCategory, EventFact, RawEvent
from fast_memory_write_env.stores import MemoryStore, RawEventStore


def _build_env(tmp_path, *, indexing_budget: int = 1, hybrid: bool = False) -> FastMemoryWriteEnv:
    raw_store = RawEventStore(tmp_path / "raw.sqlite")
    memory_store = MemoryStore(tmp_path / "memory.sqlite")
    if hybrid:
        retrieval_index = HybridRetrievalIndex(
            vector_index=InMemoryIndex(),
            memory_store=memory_store,
        )
    else:
        retrieval_index = InMemoryIndex()
    return FastMemoryWriteEnv(
        raw_event_store=raw_store,
        memory_store=memory_store,
        retrieval_index=retrieval_index,
        storage_budget_tokens_remaining=10_000,
        indexing_budget_operations_remaining=indexing_budget,
    )


def _store_event(env: FastMemoryWriteEnv, event_id: str, content: str, *, timestamp_ms: int = 0) -> RawEvent:
    event = RawEvent(
        event_id=event_id,
        episode_id="ep-update-semantics",
        timestamp_ms=timestamp_ms,
        source="test",
        user_id="user-1",
        entity_id="entity-1",
        category=EventCategory.USEFUL_FACT,
        content=content,
        facts=[
            EventFact(
                fact_id=f"{event_id}-fact",
                entity_id="entity-1",
                attribute="preference",
                value=content,
                source_event_id=event_id,
                valid_from_ms=timestamp_ms,
            )
        ],
    )
    env.execute_action({"action_type": "store_raw", "event": event.model_dump(mode="json")})
    return event


def test_update_memory_does_not_spend_budget_when_index_immediately_false(tmp_path) -> None:
    """The Ada-style failure: budget exhausted by initial writes, then an
    update should NOT silently fail. It should succeed and mark needs_reindex."""

    env = _build_env(tmp_path, indexing_budget=1)
    initial = _store_event(env, "event-1", "Account ada prefers SMS for renewals.")
    correction = _store_event(env, "event-2", "Account ada now prefers email for renewals.", timestamp_ms=10)

    write_result = env.execute_action(
        WriteMemoryAction(
            memory_id="mem-ada",
            entity_id=initial.entity_id,
            content=initial.content,
            source_event_ids=[initial.event_id],
            index_immediately=True,
        )
    )
    assert write_result.success is True
    assert env.indexing_budget_operations_remaining == 0
    assert env.memory_store.require("mem-ada").indexed is True
    assert "mem-ada" in env.retrieval_index._memories

    update_result = env.execute_action(
        UpdateMemoryAction(
            memory_id="mem-ada",
            content=correction.content,
            source_event_ids=[correction.event_id],
            reason="newer evidence",
            index_immediately=False,
        )
    )
    assert update_result.success is True
    assert update_result.payload["needs_reindex"] is True
    assert update_result.payload["indexed"] is False
    # Budget still untouched.
    assert env.indexing_budget_operations_remaining == 0
    # Stale vector dropped; canonical record is no longer "indexed".
    assert "mem-ada" not in env.retrieval_index._memories
    memory = env.memory_store.require("mem-ada")
    assert memory.indexed is False
    assert memory.metadata.get("needs_reindex") is True
    assert memory.content == correction.content


def test_index_now_clears_needs_reindex_and_restores_vector(tmp_path) -> None:
    env = _build_env(tmp_path, indexing_budget=2)
    initial = _store_event(env, "event-a", "Account ada prefers SMS.")
    correction = _store_event(env, "event-b", "Account ada now prefers email.", timestamp_ms=10)

    env.execute_action(
        WriteMemoryAction(
            memory_id="mem-ada-2",
            entity_id=initial.entity_id,
            content=initial.content,
            source_event_ids=[initial.event_id],
            index_immediately=True,
        )
    )
    env.execute_action(
        UpdateMemoryAction(
            memory_id="mem-ada-2",
            content=correction.content,
            source_event_ids=[correction.event_id],
            reason="correction",
            index_immediately=False,
        )
    )
    pre_reindex = env.memory_store.require("mem-ada-2")
    assert pre_reindex.metadata.get("needs_reindex") is True
    assert pre_reindex.indexed is False

    index_now = env.execute_action(IndexNowAction(memory_id="mem-ada-2"))

    assert index_now.success is True
    post = env.memory_store.require("mem-ada-2")
    assert post.indexed is True
    assert post.metadata.get("needs_reindex") is False
    assert "mem-ada-2" in env.retrieval_index._memories
    indexed_memory = env.retrieval_index._memories["mem-ada-2"]
    assert indexed_memory.content == correction.content


def test_update_memory_with_index_immediately_true_re_upserts(tmp_path) -> None:
    env = _build_env(tmp_path, indexing_budget=2)
    initial = _store_event(env, "event-a2", "Account ada prefers SMS.")
    correction = _store_event(env, "event-b2", "Account ada now prefers email.", timestamp_ms=10)

    env.execute_action(
        WriteMemoryAction(
            memory_id="mem-ada-3",
            entity_id=initial.entity_id,
            content=initial.content,
            source_event_ids=[initial.event_id],
            index_immediately=True,
        )
    )
    update_result = env.execute_action(
        UpdateMemoryAction(
            memory_id="mem-ada-3",
            content=correction.content,
            source_event_ids=[correction.event_id],
            reason="urgent correction",
            index_immediately=True,
        )
    )

    assert update_result.success is True
    assert update_result.payload["indexed"] is True
    assert update_result.payload["needs_reindex"] is False
    assert env.indexing_budget_operations_remaining == 0
    assert env.memory_store.require("mem-ada-3").metadata.get("needs_reindex") is False
    indexed_memory = env.retrieval_index._memories["mem-ada-3"]
    assert indexed_memory.content == correction.content


def test_policy_visible_memory_exposes_needs_reindex(tmp_path) -> None:
    env = _build_env(tmp_path, indexing_budget=1)
    event = _store_event(env, "event-vis", "Account ada prefers SMS.")
    correction = _store_event(env, "event-vis2", "Account ada now prefers email.", timestamp_ms=10)
    env.execute_action(
        WriteMemoryAction(
            memory_id="mem-visible",
            entity_id=event.entity_id,
            content=event.content,
            source_event_ids=[event.event_id],
            index_immediately=True,
        )
    )
    env.execute_action(
        UpdateMemoryAction(
            memory_id="mem-visible",
            content=correction.content,
            source_event_ids=[correction.event_id],
            reason="correction",
            index_immediately=False,
        )
    )

    visible = policy_visible_memory(env.memory_store.require("mem-visible"))

    assert visible["indexed"] is False
    assert visible["needs_reindex"] is True


def test_hybrid_retrieval_serves_updated_content_without_reindexing(tmp_path) -> None:
    """The whole point of solutions 1+2: an updated memory whose vector was
    invalidated remains queryable lexically, with the NEW content.

    The streaming evaluator advances ``current_time_ms`` by the action's
    latency between calls. This unit test simulates that with
    ``execute_action_at`` so the delete tombstone lands strictly after the
    earlier vector upsert in simulated time.
    """

    env = _build_env(tmp_path, indexing_budget=1, hybrid=True)
    initial = _store_event(env, "event-h1", "Account ada prefers SMS for renewals.")
    correction = _store_event(
        env,
        "event-h2",
        "Account ada now prefers email for renewals.",
        timestamp_ms=10,
    )
    write_result = env.execute_action_at(
        WriteMemoryAction(
            memory_id="mem-hybrid",
            entity_id=initial.entity_id,
            content=initial.content,
            source_event_ids=[initial.event_id],
            index_immediately=True,
        ),
        current_time_ms=10,
    )
    next_time_ms = int(10 + write_result.latency_ms) + 1
    update_result = env.execute_action_at(
        UpdateMemoryAction(
            memory_id="mem-hybrid",
            content=correction.content,
            source_event_ids=[correction.event_id],
            reason="now prefers email",
            index_immediately=False,
        ),
        current_time_ms=next_time_ms,
    )
    assert update_result.success is True

    results = env.retrieval_index.search(
        "How is ada contacted for renewals?",
        top_k=3,
        as_of_ms=1000.0,
    )

    assert [hit.memory_id for hit in results] == ["mem-hybrid"]
    assert results[0].content == correction.content
    assert results[0].metadata["retrieval_sources"] == ["lexical"]
