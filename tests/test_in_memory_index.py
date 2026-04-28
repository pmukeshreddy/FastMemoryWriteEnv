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
        indexed=True,
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
