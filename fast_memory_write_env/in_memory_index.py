"""In-memory retrieval fake for unit tests."""

from __future__ import annotations

import threading
from typing import Any

from fast_memory_write_env.index import SearchResult, text_match_score
from fast_memory_write_env.schemas import MemoryRecord, MemoryStatus


class InMemoryIndex:
    """A deterministic retrieval fake. Use only in tests."""

    def __init__(self) -> None:
        self._memories: dict[str, MemoryRecord] = {}
        self._versions: dict[str, list[tuple[float, int, MemoryRecord | None]]] = {}
        self._sequence = 0
        self._lock = threading.RLock()

    def upsert(self, memory: MemoryRecord) -> None:
        with self._lock:
            self._memories[memory.memory_id] = memory
            self._append_version(memory.memory_id, memory, _available_at_ms(memory))

    def delete(self, memory_id: str, *, available_at_ms: float | None = None) -> None:
        with self._lock:
            self._memories.pop(memory_id, None)
            if available_at_ms is not None:
                self._append_version(memory_id, None, available_at_ms)

    def search(
        self,
        query: str,
        *,
        top_k: int = 5,
        filters: dict[str, Any] | None = None,
        as_of_ms: float | None = None,
    ) -> list[SearchResult]:
        with self._lock:
            filters = filters or {}
            results: list[SearchResult] = []
            for memory in self._candidate_memories(as_of_ms):
                if memory.status != MemoryStatus.ACTIVE:
                    continue
                if not memory.indexed:
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

    def _append_version(
        self,
        memory_id: str,
        memory: MemoryRecord | None,
        available_at_ms: float,
    ) -> None:
        self._versions.setdefault(memory_id, []).append((available_at_ms, self._sequence, memory))
        self._sequence += 1

    def _candidate_memories(self, as_of_ms: float | None) -> list[MemoryRecord]:
        if as_of_ms is None:
            return list(self._memories.values())
        candidates: list[MemoryRecord] = []
        for versions in self._versions.values():
            visible_versions = [
                version
                for version in versions
                if version[0] <= as_of_ms
            ]
            if not visible_versions:
                continue
            _, _, memory = max(visible_versions, key=lambda version: (version[0], version[1]))
            if memory is not None:
                candidates.append(memory)
        return candidates


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


def _available_at_ms(memory: MemoryRecord) -> float:
    value = memory.metadata.get("available_at_ms")
    if value is None:
        value = memory.metadata.get("indexed_at_ms")
    if value is None:
        return float(memory.updated_at_ms)
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(memory.updated_at_ms)
