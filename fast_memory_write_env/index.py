"""Retrieval index interfaces and deterministic text helpers."""

from __future__ import annotations

import hashlib
import math
import re
from typing import Any, Protocol

from pydantic import Field

from fast_memory_write_env.schemas import ID_PATTERN, MemoryRecord, StrictBaseModel


TOKEN_RE = re.compile(r"[A-Za-z0-9]+")


class SearchResult(StrictBaseModel):
    """A memory retrieval hit."""

    memory_id: str = Field(pattern=ID_PATTERN)
    score: float = Field(ge=0.0)
    content: str
    metadata: dict[str, Any] = Field(default_factory=dict)
    memory: MemoryRecord | None = None


class RetrievalIndex(Protocol):
    """Interface implemented by real and fake retrieval backends."""

    def upsert(self, memory: MemoryRecord) -> None:
        """Insert or replace one memory in the retrieval index."""

    def delete(self, memory_id: str) -> None:
        """Remove one memory from the retrieval index."""

    def search(
        self,
        query: str,
        *,
        top_k: int = 5,
        filters: dict[str, Any] | None = None,
    ) -> list[SearchResult]:
        """Search indexed memories."""


def estimate_tokens(text: str) -> int:
    """Estimate storage tokens deterministically without external tokenizers."""

    tokens = tokenize_text(text)
    return max(1, len(tokens))


def tokenize_text(text: str) -> list[str]:
    """Tokenize text for deterministic local matching."""

    return [match.group(0).lower() for match in TOKEN_RE.finditer(text)]


def deterministic_text_vector(text: str, dimension: int = 64) -> list[float]:
    """Convert text to a deterministic hashed bag-of-words vector."""

    if dimension < 1:
        raise ValueError("dimension must be positive")
    values = [0.0] * dimension
    for token in tokenize_text(text):
        digest = hashlib.sha256(token.encode("utf-8")).digest()
        index = int.from_bytes(digest[:4], "big") % dimension
        sign = 1.0 if digest[4] % 2 == 0 else -1.0
        values[index] += sign

    norm = math.sqrt(sum(value * value for value in values))
    if norm == 0.0:
        return values
    return [value / norm for value in values]


def text_match_score(query: str, content: str) -> float:
    """Return a deterministic lexical score for local unit-test retrieval."""

    query_tokens = set(tokenize_text(query))
    content_tokens = set(tokenize_text(content))
    if not query_tokens or not content_tokens:
        return 0.0

    overlap = query_tokens & content_tokens
    if not overlap:
        return 0.0

    recall = len(overlap) / len(query_tokens)
    precision = len(overlap) / len(content_tokens)
    return round((2.0 * recall) + precision, 6)


def memory_metadata(memory: MemoryRecord) -> dict[str, Any]:
    """Create backend metadata for a memory record."""

    metadata = dict(memory.metadata)
    metadata.update(
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
        }
    )
    return metadata
