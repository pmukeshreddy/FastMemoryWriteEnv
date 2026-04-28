from __future__ import annotations

from fast_memory_write_env.dataset import generate_episode
from fast_memory_write_env.env import FastMemoryWriteEnv, deterministic_memory_id
from fast_memory_write_env.in_memory_index import InMemoryIndex
from fast_memory_write_env.schemas import DatasetMode, EventCategory, MemoryStatus
from fast_memory_write_env.stores import MemoryStore, RawEventStore


def _env(tmp_path) -> FastMemoryWriteEnv:
    return FastMemoryWriteEnv(
        raw_event_store=RawEventStore(tmp_path / "raw.sqlite"),
        memory_store=MemoryStore(tmp_path / "memory.sqlite"),
        retrieval_index=InMemoryIndex(),
    )


def _events_by_category():
    episode = generate_episode(DatasetMode.SMALL, seed=31, episode_index=0)
    events = {
        item.event.category: item.event
        for item in episode.stream
        if item.item_type == "event"
    }
    return episode, events


def test_environment_executes_memory_action_transitions(tmp_path) -> None:
    env = _env(tmp_path)
    _, events = _events_by_category()
    useful = events[EventCategory.USEFUL_FACT]
    noise = events[EventCategory.NOISE]
    urgent = events[EventCategory.URGENT_FACT]

    store_raw = env.execute_action({"action_type": "store_raw", "event": useful.model_dump(mode="json")})
    write = env.execute_action(
        {
            "action_type": "write_memory",
            "memory_id": "mem-primary",
            "entity_id": useful.entity_id,
            "content": useful.content,
            "source_event_ids": [useful.event_id],
            "fact_ids": [fact.fact_id for fact in useful.facts],
            "index_immediately": False,
        }
    )
    delay = env.execute_action(
        {
            "action_type": "delay_index",
            "memory_id": "mem-primary",
            "retry_after_ms": 500,
            "reason": "wait for budget",
        }
    )
    index = env.execute_action({"action_type": "index_now", "memory_id": "mem-primary"})
    search = env.execute_action({"action_type": "search_memory", "query_text": useful.content, "top_k": 3})
    answer = env.execute_action(
        {
            "action_type": "answer",
            "query_text": useful.content,
            "retrieved_memory_ids": ["mem-primary"],
        }
    )
    update = env.execute_action(
        {
            "action_type": "update_memory",
            "memory_id": "mem-primary",
            "content": f"{useful.content} Current verified preference.",
            "source_event_ids": [urgent.event_id],
            "fact_ids": [fact.fact_id for fact in urgent.facts],
            "reason": "new urgent evidence",
            "index_immediately": True,
        }
    )
    env.execute_action({"action_type": "store_raw", "event": noise.model_dump(mode="json")})
    ignore = env.execute_action(
        {"action_type": "ignore_event", "event_id": noise.event_id, "reason": "low-value noise"}
    )
    env.execute_action(
        {
            "action_type": "write_memory",
            "memory_id": "mem-secondary",
            "entity_id": urgent.entity_id,
            "content": urgent.content,
            "source_event_ids": [urgent.event_id],
            "fact_ids": [fact.fact_id for fact in urgent.facts],
            "index_immediately": True,
        }
    )
    env.execute_action(
        {
            "action_type": "write_memory",
            "memory_id": "mem-third",
            "entity_id": urgent.entity_id,
            "content": "Outage escalation requires on-call acknowledgement.",
            "source_event_ids": [urgent.event_id],
            "fact_ids": [],
        }
    )
    compress = env.execute_action(
        {
            "action_type": "compress_memory",
            "source_memory_ids": ["mem-secondary", "mem-third"],
            "target_memory_id": "mem-compressed",
            "compressed_content": "Urgent outage alerts go to the on-call phone with acknowledgement.",
        }
    )
    mark_stale = env.execute_action(
        {
            "action_type": "mark_stale",
            "memory_id": "mem-primary",
            "reason": "compressed memory supersedes it",
            "superseded_by_memory_id": "mem-compressed",
        }
    )

    assert all(
        result.success
        for result in [
            store_raw,
            write,
            delay,
            index,
            search,
            answer,
            update,
            ignore,
            compress,
            mark_stale,
        ]
    )
    assert store_raw.action_type == "store_raw"
    assert store_raw.storage_tokens_delta == useful.estimated_tokens
    assert write.storage_tokens_delta > 0
    assert delay.payload["indexed"] is False
    assert search.payload["results"][0]["memory_id"] == "mem-primary"
    assert answer.payload["cited_memory_ids"] == ["mem-primary"]
    assert update.storage_tokens_delta > 0
    assert noise.event_id in env.ignored_event_ids
    assert env.memory_store.require("mem-secondary").status == MemoryStatus.COMPRESSED
    assert env.memory_store.require("mem-third").status == MemoryStatus.COMPRESSED
    assert env.memory_store.require("mem-primary").status == MemoryStatus.STALE
    assert env.retrieval_index.search("Current verified preference") == []
    assert all(result.latency_ms >= 0 for result in [store_raw, write, search, answer])


def test_environment_returns_structured_failure_for_invalid_action(tmp_path) -> None:
    env = _env(tmp_path)

    result = env.execute_action({"action_type": "index_now", "memory_id": "missing-memory"})

    assert result.success is False
    assert result.action_type == "index_now"
    assert "memory not found" in result.error


def test_episode_loop_stores_events_and_runs_injected_actions(tmp_path) -> None:
    env = _env(tmp_path)
    episode = generate_episode(DatasetMode.SMALL, seed=41, episode_index=0)
    useful = next(
        item.event
        for item in episode.stream
        if item.item_type == "event" and item.event.facts
    )
    memory_id = deterministic_memory_id([useful.event_id], useful.content)

    results = env.run_episode(
        episode,
        action_batches={
            useful.event_id: [
                {
                    "action_type": "write_memory",
                    "memory_id": memory_id,
                    "entity_id": useful.entity_id,
                    "content": useful.content,
                    "source_event_ids": [useful.event_id],
                    "fact_ids": [fact.fact_id for fact in useful.facts],
                    "index_immediately": True,
                }
            ]
        },
    )

    event_count = sum(1 for item in episode.stream if item.item_type == "event")
    query_count = sum(1 for item in episode.stream if item.item_type == "query")

    assert env.raw_event_store.count() == event_count
    assert env.memory_store.require(memory_id).indexed is True
    assert sum(result.action_type == "store_raw" for result in results) == event_count
    assert sum(result.action_type == "search_memory" for result in results) == query_count
    assert sum(result.action_type == "answer" for result in results) == query_count
