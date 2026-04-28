"""In-memory retrieval fake for unit tests."""

from __future__ import annotations

from typing import Any

from fast_memory_write_env.index import SearchResult, text_match_score
from fast_memory_write_env.schemas import MemoryRecord, MemoryStatus


class InMemoryIndex:
    """A deterministic retrieval fake. Use only in tests."""

    def __init__(self) -> None:
        self._memories: dict[str, MemoryRecord] = {}

    def upsert(self, memory: MemoryRecord) -> None:
        self._memories[memory.memory_id] = memory

    def delete(self, memory_id: str) -> None:
        self._memories.pop(memory_id, None)

    def search(
        self,
        query: str,
        *,
        top_k: int = 5,
        filters: dict[str, Any] | None = None,
    ) -> list[SearchResult]:
        filters = filters or {}
        results: list[SearchResult] = []
        for memory in self._memories.values():
            if memory.status != MemoryStatus.ACTIVE:
                continue
            if not _matches_filters(memory, filters):
                continue
            score = text_match_score(query, memory.content)
            if score <= 0.0:
                continue
            results.append(
                SearchResult(
                    memory_id=memory.memory_id,
                    score=score,
                    content=memory.content,
                    metadata={
                        "entity_id": memory.entity_id,
                        "status": memory.status.value,
                        "indexed": memory.indexed,
                        **memory.metadata,
                    },
                    memory=memory,
                )
            )
        return sorted(results, key=lambda result: (-result.score, result.memory_id))[:top_k]


def _matches_filters(memory: MemoryRecord, filters: dict[str, Any]) -> bool:
    for key, expected in filters.items():
        if key == "memory_id":
            actual = memory.memory_id
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
        if actual != expected:
            return False
    return True
