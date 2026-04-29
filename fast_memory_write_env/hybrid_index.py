"""Hybrid retrieval that fuses lexical (BM25) and semantic (vector) hits.

The motive: ``time_to_useful_memory`` should not be gated by indexing budget
when the answer can be served from the lexical mirror that the memory store
already maintains. Production agent-memory systems (Mem0, Mind-mem, Hermes
dual-layer, Supabase auto-embeddings, the "delta indexing + hybrid search"
pattern) all separate "memory exists / lexically queryable" from "vector
indexed". This module is the local equivalent.

``HybridRetrievalIndex`` implements the same ``RetrievalIndex`` protocol as
``InMemoryIndex`` and ``PineconeIndex`` so it drops in unchanged at the env
level. Writes pass straight through to the underlying vector index; the
lexical path is served by ``MemoryStore.lexical_search`` which reads the
SQLite FTS5 mirror of ``memory.content``. Searches union the two ranked
lists with Reciprocal Rank Fusion and return one ``SearchResult`` per
memory, scored by RRF.
"""

from __future__ import annotations

from typing import Any

from fast_memory_write_env.index import RetrievalIndex, SearchResult
from fast_memory_write_env.schemas import MemoryRecord, MemoryStatus
from fast_memory_write_env.stores import MemoryStore


DEFAULT_RRF_K = 60


class HybridRetrievalIndex:
    """Vector + lexical retrieval with Reciprocal Rank Fusion."""

    def __init__(
        self,
        *,
        vector_index: RetrievalIndex,
        memory_store: MemoryStore,
        rrf_k: int = DEFAULT_RRF_K,
        lexical_top_k_multiplier: int = 2,
    ) -> None:
        if rrf_k < 1:
            raise ValueError("rrf_k must be a positive integer")
        if lexical_top_k_multiplier < 1:
            raise ValueError("lexical_top_k_multiplier must be a positive integer")
        self.vector_index = vector_index
        self.memory_store = memory_store
        self.rrf_k = rrf_k
        self.lexical_top_k_multiplier = lexical_top_k_multiplier

    # ``RetrievalIndex`` protocol --------------------------------------------------

    def upsert(self, memory: MemoryRecord) -> None:
        """Forward vector upserts to the underlying index.

        The lexical mirror is updated by ``MemoryStore`` itself on create/upsert,
        so there is nothing to do here for the BM25 side.
        """

        self.vector_index.upsert(memory)

    def delete(self, memory_id: str, *, available_at_ms: float | None = None) -> None:
        """Drop a memory from the vector index.

        The lexical row stays in SQLite; ``lexical_search`` filters it out as
        soon as the canonical record's status leaves ``ACTIVE`` (mark_stale,
        compress_memory, ...), so a deleted/stale memory is not lexically
        retrievable either.
        """

        self.vector_index.delete(memory_id, available_at_ms=available_at_ms)

    def search(
        self,
        query: str,
        *,
        top_k: int = 5,
        filters: dict[str, Any] | None = None,
        as_of_ms: float | None = None,
    ) -> list[SearchResult]:
        if top_k < 1:
            return []
        oversample_top_k = max(top_k, top_k * self.lexical_top_k_multiplier)
        vector_hits = self.vector_index.search(
            query,
            top_k=oversample_top_k,
            filters=filters,
            as_of_ms=as_of_ms,
        )
        lexical_hits = self.memory_store.lexical_search(
            query,
            top_k=oversample_top_k,
            as_of_ms=as_of_ms,
        )
        if filters:
            lexical_hits = [
                (memory, score)
                for memory, score in lexical_hits
                if _memory_matches_filters(memory, filters)
            ]
        return _reciprocal_rank_fuse(
            vector_hits=vector_hits,
            lexical_hits=lexical_hits,
            top_k=top_k,
            rrf_k=self.rrf_k,
        )


def _reciprocal_rank_fuse(
    *,
    vector_hits: list[SearchResult],
    lexical_hits: list[tuple[MemoryRecord, float]],
    top_k: int,
    rrf_k: int,
) -> list[SearchResult]:
    fused_scores: dict[str, float] = {}
    memories: dict[str, MemoryRecord] = {}
    contents: dict[str, str] = {}
    metadata: dict[str, dict[str, Any]] = {}
    sources: dict[str, set[str]] = {}

    for rank, hit in enumerate(vector_hits, start=1):
        fused_scores[hit.memory_id] = fused_scores.get(hit.memory_id, 0.0) + 1.0 / (rrf_k + rank)
        sources.setdefault(hit.memory_id, set()).add("vector")
        if hit.memory is not None:
            memories.setdefault(hit.memory_id, hit.memory)
        contents.setdefault(hit.memory_id, hit.content)
        metadata.setdefault(hit.memory_id, dict(hit.metadata or {}))

    for rank, (memory, _score) in enumerate(lexical_hits, start=1):
        memory_id = memory.memory_id
        fused_scores[memory_id] = fused_scores.get(memory_id, 0.0) + 1.0 / (rrf_k + rank)
        sources.setdefault(memory_id, set()).add("lexical")
        memories.setdefault(memory_id, memory)
        contents.setdefault(memory_id, memory.content)
        # Mirror the metadata shape that vector backends produce so callers
        # do not have to special-case lexical-only hits.
        metadata.setdefault(
            memory_id,
            {
                "memory_id": memory.memory_id,
                "entity_id": memory.entity_id,
                "status": memory.status.value,
                "indexed": memory.indexed,
                "content": memory.content,
                "source_event_ids": list(memory.source_event_ids),
                "fact_ids": list(memory.fact_ids),
                "estimated_tokens": memory.estimated_tokens,
                "created_at_ms": memory.created_at_ms,
                "updated_at_ms": memory.updated_at_ms,
                **memory.metadata,
            },
        )

    ordered = sorted(fused_scores.items(), key=lambda item: (-item[1], item[0]))[:top_k]
    fused_results: list[SearchResult] = []
    for memory_id, score in ordered:
        meta = dict(metadata.get(memory_id, {}))
        meta["retrieval_sources"] = sorted(sources.get(memory_id, set()))
        fused_results.append(
            SearchResult(
                memory_id=memory_id,
                score=score,
                content=contents.get(memory_id, ""),
                metadata=meta,
                memory=memories.get(memory_id),
            )
        )
    return fused_results


def _memory_matches_filters(memory: MemoryRecord, filters: dict[str, Any]) -> bool:
    """Apply the same flat key matching ``InMemoryIndex`` uses on metadata."""

    for key, expected in filters.items():
        if key == "memory_id":
            actual: Any = memory.memory_id
        elif key == "entity_id":
            actual = memory.entity_id
        elif key == "status":
            actual = memory.status.value
        elif key == "indexed":
            actual = memory.indexed
        elif key.startswith("metadata."):
            actual = memory.metadata.get(key.removeprefix("metadata."))
        else:
            actual = memory.metadata.get(key)
        if isinstance(expected, dict):
            # Skip Pinecone-style operator filters (e.g. ``$lte``) that the
            # lexical path cannot meaningfully apply.
            continue
        if actual != expected:
            return False
    # The ACTIVE filter is enforced inside the SQL of ``lexical_search``;
    # leaving it here as a defense in depth keeps behaviour explicit.
    if memory.status != MemoryStatus.ACTIVE:
        return False
    return True
