from __future__ import annotations

import pytest

from fast_memory_write_env.dataset import generate_episode
from fast_memory_write_env.index import estimate_tokens
from fast_memory_write_env.schemas import DatasetMode, MemoryRecord, MemoryStatus
from fast_memory_write_env.stores import MemoryStore, RawEventStore


def _first_event():
    episode = generate_episode(DatasetMode.SMALL, seed=21, episode_index=0)
    return next(item.event for item in episode.stream if item.item_type == "event")


def _memory(memory_id: str = "mem-001", content: str = "Account Ada prefers SMS.") -> MemoryRecord:
    return MemoryRecord(
        memory_id=memory_id,
        entity_id="account-ada",
        content=content,
        source_event_ids=["event-001"],
        fact_ids=["fact-001"],
        created_at_ms=100,
        updated_at_ms=100,
        estimated_tokens=estimate_tokens(content),
    )


def test_raw_event_store_round_trips_and_persists(tmp_path) -> None:
    path = tmp_path / "raw.sqlite"
    event = _first_event()
    store = RawEventStore(path)

    delta = store.store(event)

    assert delta == event.estimated_tokens
    assert store.get(event.event_id) == event
    assert store.list_by_episode(event.episode_id) == [event]
    assert store.count() == 1

    store.close()
    reopened = RawEventStore(path)
    assert reopened.get(event.event_id) == event
    reopened.close()


def test_raw_event_store_rejects_duplicates(tmp_path) -> None:
    event = _first_event()
    store = RawEventStore(tmp_path / "raw.sqlite")
    store.store(event)

    with pytest.raises(ValueError):
        store.store(event)


def test_memory_store_transitions_and_reopen_persistence(tmp_path) -> None:
    path = tmp_path / "memory.sqlite"
    store = MemoryStore(path)
    memory = _memory()

    create_delta = store.create(memory)
    updated, update_delta = store.update_memory(
        memory_id=memory.memory_id,
        content="Account Ada prefers SMS and needs renewal notices by phone.",
        source_event_ids=["event-002"],
        fact_ids=["fact-002"],
        updated_at_ms=200,
        metadata={"reason": "new evidence"},
    )
    indexed = store.set_indexed(memory.memory_id, True, updated_at_ms=250)
    delayed = store.delay_index(
        memory_id=memory.memory_id,
        retry_after_ms=1000,
        reason="budget",
        updated_at_ms=300,
    )
    stale = store.mark_status(
        memory_id=memory.memory_id,
        status=MemoryStatus.STALE,
        updated_at_ms=400,
        metadata={"stale_reason": "superseded"},
    )

    assert create_delta == memory.estimated_tokens
    assert update_delta == updated.estimated_tokens - memory.estimated_tokens
    assert updated.source_event_ids == ["event-001", "event-002"]
    assert updated.fact_ids == ["fact-001", "fact-002"]
    assert indexed.indexed is True
    assert delayed.indexed is False
    assert delayed.metadata["delayed_index_until_ms"] == 1000
    assert stale.status == MemoryStatus.STALE
    assert stale.indexed is False
    assert store.list_active() == []

    store.close()
    reopened = MemoryStore(path)
    persisted = reopened.require(memory.memory_id)
    assert persisted.status == MemoryStatus.STALE
    assert persisted.metadata["stale_reason"] == "superseded"
    reopened.close()


def test_memory_store_rejects_duplicate_create(tmp_path) -> None:
    store = MemoryStore(tmp_path / "memory.sqlite")
    memory = _memory()
    store.create(memory)

    with pytest.raises(ValueError):
        store.create(memory)
