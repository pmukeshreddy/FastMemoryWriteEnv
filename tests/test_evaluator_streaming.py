from __future__ import annotations

import threading
import time

from fast_memory_write_env.actions import validate_memory_actions
from fast_memory_write_env.dataset import generate_episode
from fast_memory_write_env.env import FastMemoryWriteEnv
from fast_memory_write_env.evaluator import _AsyncMemoryWriteWorker, StreamingEvaluator
from fast_memory_write_env.in_memory_index import InMemoryIndex
from fast_memory_write_env.index import estimate_tokens
from fast_memory_write_env.metrics import AggregateCounterSnapshot
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
from fast_memory_write_env.state import MemoryWriteQueue
from fast_memory_write_env.stores import MemoryStore, RawEventStore


class _IgnorePolicy:
    def __init__(self) -> None:
        self.calls = 0

    def decide(self, *, new_event, active_memories, recent_events, latency_budget_ms, storage_budget_tokens_remaining, indexing_budget_operations_remaining):
        self.calls += 1
        return validate_memory_actions(
            [
                {
                    "action_type": "ignore_event",
                    "event_id": new_event.event_id,
                    "reason": "test policy ignores event",
                }
            ]
        )


class _RecordingBudgetPolicy:
    def __init__(self) -> None:
        self.calls: list[tuple[int, int]] = []

    def decide(self, *, new_event, active_memories, recent_events, latency_budget_ms, storage_budget_tokens_remaining, indexing_budget_operations_remaining):
        self.calls.append((storage_budget_tokens_remaining, indexing_budget_operations_remaining))
        if len(self.calls) == 1:
            return validate_memory_actions(
                [
                    {
                        "action_type": "write_memory",
                        "memory_id": "mem-budget-first",
                        "entity_id": new_event.entity_id,
                        "content": "alpha",
                        "source_event_ids": [new_event.event_id],
                        "index_immediately": True,
                    }
                ]
            )
        return validate_memory_actions(
            [
                {
                    "action_type": "ignore_event",
                    "event_id": new_event.event_id,
                    "reason": "second event sees updated budgets",
                }
            ]
        )


class _WriteAndIndexPolicy:
    def decide(self, *, new_event, active_memories, recent_events, latency_budget_ms, storage_budget_tokens_remaining, indexing_budget_operations_remaining):
        return validate_memory_actions(
            [
                {
                    "action_type": "write_memory",
                    "memory_id": f"mem-{new_event.event_id}",
                    "entity_id": new_event.entity_id,
                    "content": new_event.content,
                    "source_event_ids": [new_event.event_id],
                    "index_immediately": True,
                }
            ]
        )


class _ColorUpdatePolicy:
    def decide(self, *, new_event, active_memories, recent_events, latency_budget_ms, storage_budget_tokens_remaining, indexing_budget_operations_remaining):
        if new_event.event_id == "event-blue":
            return validate_memory_actions(
                [
                    {
                        "action_type": "write_memory",
                        "memory_id": "mem-color",
                        "entity_id": new_event.entity_id,
                        "content": new_event.content,
                        "source_event_ids": [new_event.event_id],
                        "index_immediately": True,
                    }
                ]
            )
        return validate_memory_actions(
            [
                {
                    "action_type": "update_memory",
                    "memory_id": "mem-color",
                    "content": new_event.content,
                    "source_event_ids": [new_event.event_id],
                    "reason": "newer color evidence",
                    "index_immediately": True,
                }
            ]
        )


def test_async_worker_survives_temporary_empty_queue(tmp_path) -> None:
    episode = generate_episode(DatasetMode.SMALL, seed=91, episode_index=0)
    event = next(item.event for item in episode.stream if item.item_type == "event")
    env = FastMemoryWriteEnv(
        raw_event_store=RawEventStore(tmp_path / "raw.sqlite"),
        memory_store=MemoryStore(tmp_path / "memory.sqlite"),
        retrieval_index=InMemoryIndex(),
        storage_budget_tokens_remaining=100,
        indexing_budget_operations_remaining=1,
    )
    queue = MemoryWriteQueue()
    policy = _IgnorePolicy()
    worker = _AsyncMemoryWriteWorker(
        env=env,
        policy=policy,
        queue=queue,
        episode_id=episode.episode_id,
        latency_budget_ms=250,
        storage_budget_tokens_remaining=100,
        indexing_budget_operations_remaining=1,
        records=[],
        recent_events=[],
        fact_lifecycles={},
        counters=AggregateCounterSnapshot(),
        indexed_memory_available_at_ms={},
        state_lock=threading.RLock(),
    )

    worker.start()
    time.sleep(0.2)
    queue.enqueue(event=event, enqueued_at_ms=event.timestamp_ms + 1)

    assert queue.wait_until_idle(timeout_seconds=2.0) is True
    worker.stop(timeout_seconds=2.0)
    assert policy.calls == 1


def test_policy_observes_decremented_budgets_between_queued_events(tmp_path) -> None:
    source_episode = generate_episode(DatasetMode.SMALL, seed=92, episode_index=0)
    events = [item.event for item in source_episode.stream if item.item_type == "event"][:2]
    episode = StreamingEpisode(
        episode_id="ep-budget-observation",
        mode=DatasetMode.SMALL,
        seed=92,
        stream=[StreamEventItem(timestamp_ms=event.timestamp_ms, event=event.model_copy(update={"episode_id": "ep-budget-observation"})) for event in events],
    )
    policy = _RecordingBudgetPolicy()
    env = FastMemoryWriteEnv(
        raw_event_store=RawEventStore(tmp_path / "raw.sqlite"),
        memory_store=MemoryStore(tmp_path / "memory.sqlite"),
        retrieval_index=InMemoryIndex(),
    )
    evaluator = StreamingEvaluator(
        env=env,
        policy=policy,
        storage_budget_tokens_remaining=100,
        indexing_budget_operations_remaining=1,
    )

    evaluator.evaluate_episode(episode)

    assert policy.calls[0] == (100, 1)
    assert policy.calls[1][0] == 99
    assert policy.calls[1][1] == 0


def test_query_cannot_retrieve_memory_before_simulated_index_completion(tmp_path) -> None:
    episode = _causal_episode(query_timestamp_ms=2)
    evaluator = _causal_evaluator(tmp_path)

    result = evaluator.evaluate_episode(episode)

    assert evaluator.env.memory_store.require("mem-event-blue").indexed is True
    assert result.query_metrics[0].retrieved_memory_ids == []
    assert result.query_metrics[0].answer_success is False
    assert result.query_metrics[0].time_to_useful_memory is None


def test_query_retrieves_memory_after_simulated_index_completion(tmp_path) -> None:
    episode = _causal_episode(query_timestamp_ms=1000)
    evaluator = _causal_evaluator(tmp_path)

    result = evaluator.evaluate_episode(episode)

    assert result.query_metrics[0].retrieved_memory_ids == ["mem-event-blue"]
    assert result.query_metrics[0].answer_success is True
    assert result.query_metrics[0].time_to_useful_memory is not None


def test_old_memory_version_remains_visible_before_future_update_completion(tmp_path) -> None:
    evaluator = _color_update_evaluator(tmp_path)
    result = evaluator.evaluate_episode(_color_update_episode(first_query_timestamp_ms=102, second_query_timestamp_ms=200))

    early_query = result.query_metrics[0]

    assert early_query.query_id == "query-blue-102"
    assert early_query.retrieved_memory_ids == ["mem-color"]
    assert "blue" in early_query.answer_text
    assert "red" not in early_query.answer_text
    assert early_query.answer_success is True


def test_updated_memory_version_is_visible_after_update_completion(tmp_path) -> None:
    evaluator = _color_update_evaluator(tmp_path)
    result = evaluator.evaluate_episode(_color_update_episode(first_query_timestamp_ms=102, second_query_timestamp_ms=200))

    later_query = result.query_metrics[1]

    assert later_query.query_id == "query-red-200"
    assert later_query.retrieved_memory_ids == ["mem-color"]
    assert "red" in later_query.answer_text
    assert "blue" not in later_query.answer_text
    assert later_query.answer_success is True


def test_future_update_does_not_replace_prior_query_snapshot(tmp_path) -> None:
    evaluator = _color_update_evaluator(tmp_path)
    result = evaluator.evaluate_episode(_color_update_episode(first_query_timestamp_ms=102, second_query_timestamp_ms=200))

    early_record = next(
        record
        for record in result.rollout_records
        if record.record_type == "query" and record.payload["query"]["query_id"] == "query-blue-102"
    )
    later_record = next(
        record
        for record in result.rollout_records
        if record.record_type == "query" and record.payload["query"]["query_id"] == "query-red-200"
    )

    assert "blue" in early_record.payload["answer"]["answer"]
    assert "red" not in early_record.payload["answer"]["answer"]
    assert "red" in later_record.payload["answer"]["answer"]


def _causal_evaluator(tmp_path) -> StreamingEvaluator:
    env = FastMemoryWriteEnv(
        raw_event_store=RawEventStore(tmp_path / "raw.sqlite"),
        memory_store=MemoryStore(tmp_path / "memory.sqlite"),
        retrieval_index=InMemoryIndex(),
    )
    return StreamingEvaluator(
        env=env,
        policy=_WriteAndIndexPolicy(),
        storage_budget_tokens_remaining=100,
        indexing_budget_operations_remaining=1,
    )


def _color_update_evaluator(tmp_path) -> StreamingEvaluator:
    env = FastMemoryWriteEnv(
        raw_event_store=RawEventStore(tmp_path / "raw.sqlite"),
        memory_store=MemoryStore(tmp_path / "memory.sqlite"),
        retrieval_index=InMemoryIndex(),
    )
    return StreamingEvaluator(
        env=env,
        policy=_ColorUpdatePolicy(),
        storage_budget_tokens_remaining=100,
        indexing_budget_operations_remaining=5,
    )


def _causal_episode(*, query_timestamp_ms: int) -> StreamingEpisode:
    fact = EventFact(
        fact_id="fact-blue",
        entity_id="entity-main",
        attribute="favorite_color",
        value="blue",
        source_event_id="event-blue",
        valid_from_ms=0,
    )
    content = "The user's favorite color is blue."
    event = RawEvent(
        event_id="event-blue",
        episode_id="ep-causal",
        timestamp_ms=0,
        source="test",
        user_id="user-main",
        entity_id="entity-main",
        category=EventCategory.USEFUL_FACT,
        content=content,
        facts=[fact],
        estimated_tokens=estimate_tokens(content),
    )
    query = Query(
        query_id=f"query-blue-{query_timestamp_ms}",
        episode_id="ep-causal",
        timestamp_ms=query_timestamp_ms,
        user_id="user-main",
        target_entity_id="entity-main",
        text="What is the user's favorite color blue?",
        gold=QueryGold(
            required_fact_ids=["fact-blue"],
            supporting_event_ids=["event-blue"],
            answer_facts=["blue"],
        ),
    )
    return StreamingEpisode(
        episode_id="ep-causal",
        mode=DatasetMode.SMALL,
        seed=0,
        stream=[
            StreamEventItem(timestamp_ms=event.timestamp_ms, event=event),
            StreamQueryItem(timestamp_ms=query.timestamp_ms, query=query),
        ],
    )


def _color_update_episode(
    *,
    first_query_timestamp_ms: int,
    second_query_timestamp_ms: int,
) -> StreamingEpisode:
    blue_fact = EventFact(
        fact_id="fact-blue",
        entity_id="entity-main",
        attribute="favorite_color",
        value="blue",
        source_event_id="event-blue",
        valid_from_ms=0,
    )
    red_fact = EventFact(
        fact_id="fact-red",
        entity_id="entity-main",
        attribute="favorite_color",
        value="red",
        source_event_id="event-red",
        valid_from_ms=100,
    )
    blue_event = RawEvent(
        event_id="event-blue",
        episode_id="ep-color-update",
        timestamp_ms=0,
        source="test",
        user_id="user-main",
        entity_id="entity-main",
        category=EventCategory.USEFUL_FACT,
        content="The user's favorite color is blue.",
        facts=[blue_fact],
        estimated_tokens=estimate_tokens("The user's favorite color is blue."),
    )
    red_event = RawEvent(
        event_id="event-red",
        episode_id="ep-color-update",
        timestamp_ms=100,
        source="test",
        user_id="user-main",
        entity_id="entity-main",
        category=EventCategory.STALE_UPDATE,
        content="The user's favorite color is red.",
        facts=[red_fact],
        supersedes=["fact-blue"],
        estimated_tokens=estimate_tokens("The user's favorite color is red."),
    )
    blue_query = Query(
        query_id=f"query-blue-{first_query_timestamp_ms}",
        episode_id="ep-color-update",
        timestamp_ms=first_query_timestamp_ms,
        user_id="user-main",
        target_entity_id="entity-main",
        text="What is the user's favorite color blue?",
        gold=QueryGold(
            required_fact_ids=["fact-blue"],
            supporting_event_ids=["event-blue"],
            answer_facts=["blue"],
        ),
    )
    red_query = Query(
        query_id=f"query-red-{second_query_timestamp_ms}",
        episode_id="ep-color-update",
        timestamp_ms=second_query_timestamp_ms,
        user_id="user-main",
        target_entity_id="entity-main",
        text="What is the user's favorite color red?",
        gold=QueryGold(
            required_fact_ids=["fact-red"],
            supporting_event_ids=["event-red"],
            answer_facts=["red"],
        ),
    )
    return StreamingEpisode(
        episode_id="ep-color-update",
        mode=DatasetMode.SMALL,
        seed=0,
        stream=[
            StreamEventItem(timestamp_ms=blue_event.timestamp_ms, event=blue_event),
            StreamEventItem(timestamp_ms=red_event.timestamp_ms, event=red_event),
            StreamQueryItem(timestamp_ms=blue_query.timestamp_ms, query=blue_query),
            StreamQueryItem(timestamp_ms=red_query.timestamp_ms, query=red_query),
        ],
    )
