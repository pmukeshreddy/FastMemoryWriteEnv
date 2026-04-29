"""Embedding clients used by retrieval backends.

Two implementations are provided:

* ``DeterministicEmbeddingClient`` — hashed bag-of-words vectors. Suitable
  only for unit tests and for the in-process Pinecone smoke test.
* ``OpenAIEmbeddingClient`` — real OpenAI-compatible embeddings (e.g.
  ``text-embedding-3-small``) via HTTPS. This is the default for real
  Pinecone runs.

The Pinecone backend resolves an ``EmbeddingClient`` and validates that its
declared dimension matches the configured Pinecone index dimension before
the first upsert/query.
"""

from __future__ import annotations

import json
import math
import os
import urllib.error
import urllib.request
from typing import Any, Iterable, Protocol

from fast_memory_write_env.index import deterministic_text_vector


DEFAULT_OPENAI_EMBEDDING_MODEL = "text-embedding-3-small"
DEFAULT_OPENAI_EMBEDDING_DIMENSION = 1536


class EmbeddingClientError(RuntimeError):
    """Raised when an embedding client cannot produce a usable response."""


class EmbeddingClient(Protocol):
    """Internal abstraction for retrieval-backend embedding generation."""

    @property
    def dimension(self) -> int:
        """Vector length each call returns."""

    def embed_one(self, text: str) -> list[float]:
        """Return a single embedding vector."""

    def embed_many(self, texts: Iterable[str]) -> list[list[float]]:
        """Return one embedding vector per input text, in order."""


class DeterministicEmbeddingClient:
    """Hashed bag-of-words embedding for unit tests and local smoke tests."""

    def __init__(self, *, dimension: int) -> None:
        if dimension < 1:
            raise ValueError("dimension must be positive")
        self._dimension = dimension

    @property
    def dimension(self) -> int:
        return self._dimension

    def embed_one(self, text: str) -> list[float]:
        return deterministic_text_vector(text, self._dimension)

    def embed_many(self, texts: Iterable[str]) -> list[list[float]]:
        return [self.embed_one(text) for text in texts]


class OpenAIEmbeddingClient:
    """Minimal OpenAI-compatible embeddings client.

    Uses the standard library to avoid pulling in a provider package. The
    declared ``dimension`` is the value the model is expected to return; the
    response length is checked on every call so a misconfigured deployment
    fails loudly instead of silently writing wrong-shaped vectors.
    """

    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        model: str | None = None,
        dimension: int | None = None,
        timeout_seconds: float = 30.0,
    ) -> None:
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise EmbeddingClientError(
                "OPENAI_API_KEY is required for OpenAIEmbeddingClient"
            )
        self.base_url = (
            base_url
            or os.getenv("OPENAI_EMBEDDING_BASE_URL")
            or os.getenv("OPENAI_BASE_URL")
            or "https://api.openai.com/v1"
        ).rstrip("/")
        self.model = (
            model
            or os.getenv("OPENAI_EMBEDDING_MODEL")
            or DEFAULT_OPENAI_EMBEDDING_MODEL
        )
        if dimension is None:
            env_value = os.getenv("OPENAI_EMBEDDING_DIMENSION")
            if env_value is None:
                dimension = DEFAULT_OPENAI_EMBEDDING_DIMENSION
            else:
                try:
                    dimension = int(env_value)
                except ValueError as exc:
                    raise EmbeddingClientError(
                        "OPENAI_EMBEDDING_DIMENSION must be an integer"
                    ) from exc
        if dimension < 1:
            raise EmbeddingClientError("dimension must be positive")
        self._dimension = dimension
        self.timeout_seconds = timeout_seconds

    @classmethod
    def from_env(cls) -> OpenAIEmbeddingClient:
        """Construct using OPENAI_* environment variables."""

        return cls()

    @property
    def dimension(self) -> int:
        return self._dimension

    def embed_one(self, text: str) -> list[float]:
        vectors = self.embed_many([text])
        return vectors[0]

    def embed_many(self, texts: Iterable[str]) -> list[list[float]]:
        inputs = [_clean_for_embedding(text) for text in texts]
        if not inputs:
            return []
        payload: dict[str, Any] = {
            "model": self.model,
            "input": inputs,
        }
        # `dimensions` is a real OpenAI knob for v3 embedding models. Request
        # exactly the dimension we promised so the response shape matches.
        payload["dimensions"] = self._dimension
        request = urllib.request.Request(
            f"{self.base_url}/embeddings",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                response_payload = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            raise EmbeddingClientError(
                f"OpenAI embeddings request failed: {exc.code} {body}"
            ) from exc
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            raise EmbeddingClientError(f"OpenAI embeddings request failed: {exc}") from exc

        try:
            data = response_payload["data"]
            vectors = [item["embedding"] for item in data]
        except (KeyError, TypeError) as exc:
            raise EmbeddingClientError(
                "OpenAI embeddings response missing data/embedding fields"
            ) from exc

        if len(vectors) != len(inputs):
            raise EmbeddingClientError(
                "OpenAI embeddings response length did not match input length"
            )
        for vector in vectors:
            if not isinstance(vector, list) or len(vector) != self._dimension:
                raise EmbeddingClientError(
                    "OpenAI embeddings response dimension did not match configured dimension"
                )
        return [[float(value) for value in vector] for vector in vectors]


def _clean_for_embedding(text: str) -> str:
    """Embedding APIs reject empty inputs; collapse whitespace defensively."""

    cleaned = text.strip()
    if not cleaned:
        raise EmbeddingClientError("cannot embed empty text")
    return cleaned


def vector_length(vector: list[float]) -> float:
    """Euclidean norm helper for callers that want to sanity-check vectors."""

    return math.sqrt(sum(value * value for value in vector))
