"""Pinecone retrieval backend.

Imports from the Pinecone SDK are intentionally lazy so unit tests can run
without credentials or the optional external service being available.

The backend takes an :class:`~fast_memory_write_env.embeddings.EmbeddingClient`
that produces upsert/query vectors. Real runs use
``OpenAIEmbeddingClient``; ``DeterministicEmbeddingClient`` exists only for
the in-process Pinecone smoke test. The embedding client's declared
dimension must match the configured Pinecone index dimension; we fail fast
during construction if they disagree.
"""

from __future__ import annotations

from typing import Any

from fast_memory_write_env.config import PineconeConfig
from fast_memory_write_env.embeddings import (
    DeterministicEmbeddingClient,
    EmbeddingClient,
)
from fast_memory_write_env.index import (
    SearchResult,
    memory_metadata,
)
from fast_memory_write_env.schemas import MemoryRecord, MemoryStatus


CANONICAL_MEMORY_METADATA_KEYS = {
    "memory_id",
    "entity_id",
    "status",
    "indexed",
    "content",
    "source_event_ids",
    "fact_ids",
    "estimated_tokens",
    "created_at_ms",
    "updated_at_ms",
    "available_at_ms",
    "indexed_at_ms",
}


class PineconeDimensionMismatchError(RuntimeError):
    """Raised when an embedding client and Pinecone config disagree on dimension."""


class PineconeIndex:
    """Real Pinecone-backed retrieval index."""

    def __init__(
        self,
        config: PineconeConfig,
        *,
        embedding_client: EmbeddingClient | None = None,
    ) -> None:
        self.config = config
        self.embedding_client = embedding_client or DeterministicEmbeddingClient(
            dimension=config.dimension
        )
        self._validate_dimensions()
        self._index = self._connect(config)

    @classmethod
    def from_env(
        cls,
        *,
        create_if_missing: bool = False,
        embedding_client: EmbeddingClient | None = None,
    ) -> PineconeIndex:
        return cls(
            PineconeConfig.from_env(create_if_missing=create_if_missing),
            embedding_client=embedding_client,
        )

    def upsert(self, memory: MemoryRecord) -> None:
        self._index.upsert(
            vectors=[
                {
                    "id": memory.memory_id,
                    "values": self.embedding_client.embed_one(memory.content),
                    "metadata": memory_metadata(memory),
                }
            ],
            namespace=self.config.namespace,
        )

    def delete(self, memory_id: str, *, available_at_ms: float | None = None) -> None:
        # Pinecone stores the current vector under the memory ID. The in-memory
        # test index keeps historical tombstones; Pinecone deletion remains a
        # current-state operation.
        self._index.delete(ids=[memory_id], namespace=self.config.namespace)

    def cleanup_namespace(self) -> None:
        """Delete every vector in the configured namespace.

        Used between samples so each independent run starts from an empty
        retrieval space, with no cross-talk from earlier samples sharing the
        same physical Pinecone index.
        """

        try:
            self._index.delete(delete_all=True, namespace=self.config.namespace)
        except Exception:  # pragma: no cover - best-effort cleanup
            # Some Pinecone responses raise when the namespace is already
            # empty. Swallow only here so a per-sample teardown is robust.
            pass

    def search(
        self,
        query: str,
        *,
        top_k: int = 5,
        filters: dict[str, Any] | None = None,
        as_of_ms: float | None = None,
    ) -> list[SearchResult]:
        response = self._index.query(
            vector=self.embedding_client.embed_one(query),
            top_k=top_k,
            include_metadata=True,
            namespace=self.config.namespace,
            filter=_search_filter(filters, as_of_ms=as_of_ms),
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
                    memory=_memory_from_metadata(memory_id, metadata),
                )
            )
        return results

    def _validate_dimensions(self) -> None:
        if self.embedding_client.dimension != self.config.dimension:
            raise PineconeDimensionMismatchError(
                "embedding client dimension does not match Pinecone index dimension: "
                f"client={self.embedding_client.dimension}, "
                f"index={self.config.dimension}"
            )

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


def _search_filter(filters: dict[str, Any] | None, *, as_of_ms: float | None = None) -> dict[str, Any]:
    search_filter = {
        "status": MemoryStatus.ACTIVE.value,
        "indexed": True,
        **(filters or {}),
    }
    if as_of_ms is not None:
        search_filter["available_at_ms"] = {"$lte": as_of_ms}
    return search_filter


def _memory_from_metadata(memory_id: str, metadata: dict[str, Any]) -> MemoryRecord | None:
    """Rebuild the indexed memory from Pinecone metadata when possible."""

    content = metadata.get("content")
    entity_id = metadata.get("entity_id")
    created_at_ms = metadata.get("created_at_ms")
    updated_at_ms = metadata.get("updated_at_ms")
    if not content or not entity_id or created_at_ms is None or updated_at_ms is None:
        return None

    try:
        return MemoryRecord(
            memory_id=str(metadata.get("memory_id") or memory_id),
            entity_id=str(entity_id),
            content=str(content),
            source_event_ids=_string_list(metadata.get("source_event_ids")),
            fact_ids=_string_list(metadata.get("fact_ids")),
            created_at_ms=int(created_at_ms),
            updated_at_ms=int(updated_at_ms),
            status=MemoryStatus(str(metadata.get("status", MemoryStatus.ACTIVE.value))),
            indexed=_bool_value(metadata.get("indexed", True)),
            estimated_tokens=int(metadata.get("estimated_tokens", 0) or 0),
            metadata={
                key: value
                for key, value in metadata.items()
                if key not in CANONICAL_MEMORY_METADATA_KEYS
            },
        )
    except (TypeError, ValueError):
        return None


def _string_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item) for item in value]
    return [str(value)]


def _bool_value(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.lower() in {"1", "true", "yes"}
    return bool(value)
