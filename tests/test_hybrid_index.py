"""Tests for ``HybridRetrievalIndex`` and the FTS5 lexical mirror."""

from __future__ import annotations

from pathlib import Path

import pytest

from fast_memory_write_env.hybrid_index import HybridRetrievalIndex
from fast_memory_write_env.in_memory_index import InMemoryIndex
from fast_memory_write_env.index import SearchResult, estimate_tokens
from fast_memory_write_env.schemas import MemoryRecord, MemoryStatus
from fast_memory_write_env.stores import MemoryStore


def _make_memory(
    *,
    memory_id: str,
    content: str,
    status: MemoryStatus = MemoryStatus.ACTIVE,
    indexed: bool = False,
    lexical_available_at_ms: float | None = 0.0,
    indexed_at_ms: float | None = None,
    updated_at_ms: int = 0,
) -> MemoryRecord:
    metadata: dict[str, object] = {}
    if lexical_available_at_ms is not None:
        metadata["lexical_available_at_ms"] = lexical_available_at_ms
    if indexed_at_ms is not None:
        metadata["available_at_ms"] = indexed_at_ms
        metadata["indexed_at_ms"] = indexed_at_ms
    return MemoryRecord(
        memory_id=memory_id,
        entity_id="entity-test",
        content=content,
        source_event_ids=[f"{memory_id}-event"],
        fact_ids=[],
        created_at_ms=0,
        updated_at_ms=updated_at_ms,
        status=status,
        indexed=indexed,
        estimated_tokens=estimate_tokens(content),
        metadata=metadata,
    )


def _store_with(memory: MemoryRecord, tmp_path: Path) -> MemoryStore:
    store = MemoryStore(tmp_path / "memory.sqlite")
    store.create(memory)
    return store


def test_lexical_search_finds_memory_via_fts5(tmp_path) -> None:
    store = _store_with(
        _make_memory(memory_id="mem-alpha", content="Account ada prefers SMS for renewals."),
        tmp_path,
    )

    hits = store.lexical_search("How does ada want renewals?", top_k=5)

    assert hits, "expected lexical FTS5 to surface the memory"
    memory, score = hits[0]
    assert memory.memory_id == "mem-alpha"
    assert score >= 0.0


def test_lexical_search_filters_non_active_memories(tmp_path) -> None:
    store = _store_with(
        _make_memory(
            memory_id="mem-stale",
            content="Account ada prefers SMS.",
            status=MemoryStatus.STALE,
        ),
        tmp_path,
    )

    assert store.lexical_search("ada SMS", top_k=5) == []


def test_lexical_search_respects_as_of_ms(tmp_path) -> None:
    store = _store_with(
        _make_memory(
            memory_id="mem-future",
            content="Account ada prefers SMS.",
            lexical_available_at_ms=1000.0,
        ),
        tmp_path,
    )

    assert store.lexical_search("ada SMS", top_k=5, as_of_ms=500.0) == []
    assert store.lexical_search("ada SMS", top_k=5, as_of_ms=1500.0) != []


def test_hybrid_search_surfaces_lexical_only_hit_when_vector_misses(tmp_path) -> None:
    store = _store_with(
        _make_memory(
            memory_id="mem-lex",
            content="Account ada prefers SMS for renewal notices.",
            indexed=False,  # not indexed in vector backend
        ),
        tmp_path,
    )
    hybrid = HybridRetrievalIndex(
        vector_index=InMemoryIndex(),  # empty: vector path returns no hits
        memory_store=store,
    )

    results = hybrid.search("How is ada contacted for renewals?", top_k=3)

    assert [hit.memory_id for hit in results] == ["mem-lex"]
    assert results[0].metadata["retrieval_sources"] == ["lexical"]


def test_hybrid_search_fuses_vector_and_lexical_with_rrf(tmp_path) -> None:
    memory = _make_memory(
        memory_id="mem-shared",
        content="Account ada prefers SMS for renewals.",
        indexed=True,
        lexical_available_at_ms=0.0,
        indexed_at_ms=0.0,
    )
    store = _store_with(memory, tmp_path)
    vector_index = InMemoryIndex()
    vector_index.upsert(memory)
    hybrid = HybridRetrievalIndex(vector_index=vector_index, memory_store=store)

    results = hybrid.search("ada renewals SMS", top_k=3)

    assert [hit.memory_id for hit in results] == ["mem-shared"]
    assert set(results[0].metadata["retrieval_sources"]) == {"vector", "lexical"}


def test_hybrid_upsert_and_delete_pass_through_to_vector_index(tmp_path) -> None:
    memory = _make_memory(memory_id="mem-roundtrip", content="alpha", indexed=True)
    store = _store_with(memory, tmp_path)

    upserts: list[str] = []
    deletes: list[str] = []

    class _RecordingVector:
        def upsert(self, m: MemoryRecord) -> None:
            upserts.append(m.memory_id)

        def delete(self, memory_id: str, *, available_at_ms: float | None = None) -> None:
            deletes.append(memory_id)

        def search(self, query, *, top_k=5, filters=None, as_of_ms=None) -> list[SearchResult]:
            return []

    hybrid = HybridRetrievalIndex(vector_index=_RecordingVector(), memory_store=store)
    hybrid.upsert(memory)
    hybrid.delete("mem-roundtrip", available_at_ms=10.0)

    assert upserts == ["mem-roundtrip"]
    assert deletes == ["mem-roundtrip"]


def test_hybrid_search_filters_apply_to_lexical_path(tmp_path) -> None:
    keep = _make_memory(memory_id="mem-keep", content="ada prefers SMS")
    drop = _make_memory(memory_id="mem-drop", content="ada prefers SMS")
    store = MemoryStore(tmp_path / "memory.sqlite")
    store.create(keep)
    store.create(drop.model_copy(update={"entity_id": "entity-other"}))
    hybrid = HybridRetrievalIndex(vector_index=InMemoryIndex(), memory_store=store)

    results = hybrid.search("ada SMS", top_k=5, filters={"entity_id": "entity-test"})

    assert [hit.memory_id for hit in results] == ["mem-keep"]


def test_hybrid_search_returns_empty_for_unmatchable_query(tmp_path) -> None:
    store = _store_with(_make_memory(memory_id="mem-x", content="alpha"), tmp_path)
    hybrid = HybridRetrievalIndex(vector_index=InMemoryIndex(), memory_store=store)

    assert hybrid.search("zzzzzzzzz", top_k=5) == []


def test_hybrid_index_rejects_invalid_rrf_k(tmp_path) -> None:
    store = MemoryStore(tmp_path / "memory.sqlite")
    with pytest.raises(ValueError):
        HybridRetrievalIndex(vector_index=InMemoryIndex(), memory_store=store, rrf_k=0)
