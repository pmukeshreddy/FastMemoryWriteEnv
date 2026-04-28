from __future__ import annotations

from fast_memory_write_env.in_memory_index import InMemoryIndex
from fast_memory_write_env.index import estimate_tokens
from fast_memory_write_env.schemas import MemoryRecord, MemoryStatus


def _memory(
    memory_id: str,
    content: str,
    *,
    entity_id: str = "account-ada",
    status: MemoryStatus = MemoryStatus.ACTIVE,
    indexed: bool = True,
    metadata: dict[str, object] | None = None,
) -> MemoryRecord:
    return MemoryRecord(
        memory_id=memory_id,
        entity_id=entity_id,
        content=content,
        source_event_ids=[f"{memory_id}-event"],
        fact_ids=[f"{memory_id}-fact"],
        created_at_ms=100,
        updated_at_ms=100,
        status=status,
        indexed=indexed,
        estimated_tokens=estimate_tokens(content),
        metadata=metadata or {},
    )


def test_in_memory_index_searches_and_orders_results() -> None:
    index = InMemoryIndex()
    index.upsert(_memory("mem-email", "Account Ada prefers email notices."))
    index.upsert(_memory("mem-sms", "Account Ada prefers SMS renewal notices and SMS alerts."))

    results = index.search("SMS renewal notices", top_k=2)

    assert [result.memory_id for result in results] == ["mem-sms", "mem-email"]
    assert results[0].score > results[1].score


def test_in_memory_index_filters_and_skips_non_active_memories() -> None:
    index = InMemoryIndex()
    index.upsert(_memory("mem-active", "On-call phone receives outage alerts.", metadata={"tier": "gold"}))
    index.upsert(
        _memory(
            "mem-stale",
            "On-call phone receives outage alerts.",
            status=MemoryStatus.STALE,
        )
    )

    assert [hit.memory_id for hit in index.search("outage alerts")] == ["mem-active"]
    assert [hit.memory_id for hit in index.search("outage alerts", filters={"tier": "gold"})] == [
        "mem-active"
    ]
    assert index.search("outage alerts", filters={"tier": "silver"}) == []


def test_in_memory_index_update_replaces_and_delete_removes() -> None:
    index = InMemoryIndex()
    index.upsert(_memory("mem-001", "Account Ada prefers SMS."))
    assert index.search("SMS")

    index.upsert(_memory("mem-001", "Account Ada prefers email."))

    assert index.search("SMS") == []
    assert index.search("email")[0].memory_id == "mem-001"

    index.delete("mem-001")
    assert index.search("email") == []


def test_in_memory_index_searches_latest_available_version_as_of_time() -> None:
    index = InMemoryIndex()
    blue = _memory(
        "mem-color",
        "The user's favorite color is blue.",
        metadata={"available_at_ms": 10.0},
    )
    red = _memory(
        "mem-color",
        "The user's favorite color is red.",
        metadata={"available_at_ms": 106.0},
    )

    index.upsert(blue)
    index.upsert(red)

    assert [hit.content for hit in index.search("favorite color blue", as_of_ms=102.0)] == [
        "The user's favorite color is blue."
    ]
    assert index.search("red", as_of_ms=102.0) == []
    assert [hit.content for hit in index.search("favorite color red", as_of_ms=200.0)] == [
        "The user's favorite color is red."
    ]
    assert index.search("blue", as_of_ms=200.0) == []


def test_in_memory_index_historical_delete_preserves_old_version_before_delete_time() -> None:
    index = InMemoryIndex()
    memory = _memory(
        "mem-color",
        "The user's favorite color is blue.",
        metadata={"available_at_ms": 10.0},
    )
    index.upsert(memory)
    index.delete("mem-color", available_at_ms=106.0)

    assert [hit.memory_id for hit in index.search("favorite color blue", as_of_ms=102.0)] == [
        "mem-color"
    ]
    assert index.search("favorite color blue", as_of_ms=200.0) == []


def test_in_memory_index_stores_but_does_not_retrieve_unindexed_memory() -> None:
    index = InMemoryIndex()
    memory = _memory(
        "mem-delayed",
        "Account Ada prefers delayed indexing.",
        indexed=False,
    )

    index.upsert(memory)

    assert index._memories["mem-delayed"] == memory
    assert index.search("delayed indexing") == []
