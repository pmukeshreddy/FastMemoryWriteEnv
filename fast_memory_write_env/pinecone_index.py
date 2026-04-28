"""Pinecone retrieval backend.

Imports from the Pinecone SDK are intentionally lazy so unit tests can run
without credentials or the optional external service being available.
"""

from __future__ import annotations

from typing import Any

from fast_memory_write_env.config import PineconeConfig
from fast_memory_write_env.index import (
    SearchResult,
    deterministic_text_vector,
    memory_metadata,
)
from fast_memory_write_env.schemas import MemoryRecord


class PineconeIndex:
    """Real Pinecone-backed retrieval index."""

    def __init__(self, config: PineconeConfig) -> None:
        self.config = config
        self._index = self._connect(config)

    @classmethod
    def from_env(cls, *, create_if_missing: bool = False) -> PineconeIndex:
        return cls(PineconeConfig.from_env(create_if_missing=create_if_missing))

    def upsert(self, memory: MemoryRecord) -> None:
        self._index.upsert(
            vectors=[
                {
                    "id": memory.memory_id,
                    "values": deterministic_text_vector(memory.content, self.config.dimension),
                    "metadata": memory_metadata(memory),
                }
            ],
            namespace=self.config.namespace,
        )

    def delete(self, memory_id: str) -> None:
        self._index.delete(ids=[memory_id], namespace=self.config.namespace)

    def search(
        self,
        query: str,
        *,
        top_k: int = 5,
        filters: dict[str, Any] | None = None,
    ) -> list[SearchResult]:
        response = self._index.query(
            vector=deterministic_text_vector(query, self.config.dimension),
            top_k=top_k,
            include_metadata=True,
            namespace=self.config.namespace,
            filter=filters or None,
        )
        matches = _get_matches(response)
        results: list[SearchResult] = []
        for match in matches:
            metadata = dict(_get_attr(match, "metadata", {}) or {})
            memory_id = str(_get_attr(match, "id", metadata.get("memory_id", "")))
            if not memory_id:
                continue
            results.append(
                SearchResult(
                    memory_id=memory_id,
                    score=float(_get_attr(match, "score", 0.0) or 0.0),
                    content=str(metadata.get("content", "")),
                    metadata=metadata,
                )
            )
        return results

    def _connect(self, config: PineconeConfig) -> Any:
        try:
            from pinecone import Pinecone, ServerlessSpec
        except ImportError as exc:
            raise RuntimeError(
                "The pinecone package is required for PineconeIndex. "
                "Install project dependencies before real runs."
            ) from exc

        client = Pinecone(api_key=config.api_key)
        if config.create_if_missing:
            index_names = _list_index_names(client)
            if config.index_name not in index_names:
                client.create_index(
                    name=config.index_name,
                    dimension=config.dimension,
                    metric=config.metric,
                    spec=ServerlessSpec(cloud=config.cloud, region=config.region),
                )
        return client.Index(config.index_name)


def _list_index_names(client: Any) -> set[str]:
    indexes = client.list_indexes()
    if hasattr(indexes, "names"):
        return set(indexes.names())
    names: set[str] = set()
    for item in indexes:
        if isinstance(item, str):
            names.add(item)
        else:
            name = _get_attr(item, "name", None)
            if name:
                names.add(str(name))
    return names


def _get_matches(response: Any) -> list[Any]:
    if isinstance(response, dict):
        return list(response.get("matches", []))
    return list(getattr(response, "matches", []))


def _get_attr(value: Any, key: str, default: Any) -> Any:
    if isinstance(value, dict):
        return value.get(key, default)
    return getattr(value, key, default)
