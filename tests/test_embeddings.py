"""Tests for the embedding clients and Pinecone dimension validation."""

from __future__ import annotations

import json
from io import BytesIO
from unittest import mock

import pytest

from fast_memory_write_env.config import PineconeConfig
from fast_memory_write_env.embeddings import (
    DEFAULT_OPENAI_EMBEDDING_DIMENSION,
    DEFAULT_OPENAI_EMBEDDING_MODEL,
    DeterministicEmbeddingClient,
    EmbeddingClientError,
    OpenAIEmbeddingClient,
)
from fast_memory_write_env.pinecone_index import PineconeDimensionMismatchError, PineconeIndex


def test_deterministic_embedding_client_returns_configured_dimension() -> None:
    client = DeterministicEmbeddingClient(dimension=64)

    vector = client.embed_one("alpha beta gamma")

    assert client.dimension == 64
    assert len(vector) == 64
    assert all(isinstance(value, float) for value in vector)


def test_deterministic_embedding_client_rejects_invalid_dimension() -> None:
    with pytest.raises(ValueError):
        DeterministicEmbeddingClient(dimension=0)


def test_openai_embedding_client_uses_env_defaults(monkeypatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.delenv("OPENAI_EMBEDDING_MODEL", raising=False)
    monkeypatch.delenv("OPENAI_EMBEDDING_DIMENSION", raising=False)

    client = OpenAIEmbeddingClient.from_env()

    assert client.model == DEFAULT_OPENAI_EMBEDDING_MODEL
    assert client.dimension == DEFAULT_OPENAI_EMBEDDING_DIMENSION


def test_openai_embedding_client_requires_api_key(monkeypatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    with pytest.raises(EmbeddingClientError):
        OpenAIEmbeddingClient.from_env()


def test_openai_embedding_client_rejects_invalid_dimension_env(monkeypatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("OPENAI_EMBEDDING_DIMENSION", "not-a-number")

    with pytest.raises(EmbeddingClientError):
        OpenAIEmbeddingClient.from_env()


def test_openai_embedding_client_parses_response_vectors(monkeypatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    client = OpenAIEmbeddingClient(dimension=4, model="test-model")

    response_payload = {
        "data": [
            {"embedding": [0.1, 0.2, 0.3, 0.4]},
            {"embedding": [0.5, 0.6, 0.7, 0.8]},
        ]
    }
    response_body = BytesIO(json.dumps(response_payload).encode("utf-8"))

    class _FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, *exc_info):
            return False

        def read(self):
            return response_body.getvalue()

    captured: dict = {}

    def fake_urlopen(request, timeout):  # noqa: ARG001
        captured["url"] = request.full_url
        captured["body"] = json.loads(request.data.decode("utf-8"))
        return _FakeResponse()

    with mock.patch("fast_memory_write_env.embeddings.urllib.request.urlopen", side_effect=fake_urlopen):
        vectors = client.embed_many(["alpha", "beta"])

    assert vectors == [[0.1, 0.2, 0.3, 0.4], [0.5, 0.6, 0.7, 0.8]]
    assert captured["body"]["model"] == "test-model"
    assert captured["body"]["dimensions"] == 4
    assert captured["body"]["input"] == ["alpha", "beta"]


def test_openai_embedding_client_rejects_dimension_mismatch_in_response(monkeypatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    client = OpenAIEmbeddingClient(dimension=4, model="test-model")

    response_payload = {"data": [{"embedding": [0.1, 0.2]}]}
    response_body = json.dumps(response_payload).encode("utf-8")

    class _FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, *exc_info):
            return False

        def read(self):
            return response_body

    with mock.patch(
        "fast_memory_write_env.embeddings.urllib.request.urlopen",
        return_value=_FakeResponse(),
    ):
        with pytest.raises(EmbeddingClientError):
            client.embed_one("alpha")


def test_pinecone_index_rejects_dimension_mismatch(monkeypatch) -> None:
    config = PineconeConfig(
        api_key="test-key",
        index_name="test-index",
        cloud="aws",
        region="us-east-1",
        dimension=64,
    )
    client = DeterministicEmbeddingClient(dimension=128)

    with pytest.raises(PineconeDimensionMismatchError):
        PineconeIndex(config, embedding_client=client)


def test_pinecone_index_validates_dimension_before_connect(monkeypatch) -> None:
    """Dimension check should fail before we try to talk to the SDK."""

    config = PineconeConfig(
        api_key="test-key",
        index_name="test-index",
        cloud="aws",
        region="us-east-1",
        dimension=8,
    )

    with mock.patch.object(PineconeIndex, "_connect") as connect:
        with pytest.raises(PineconeDimensionMismatchError):
            PineconeIndex(config, embedding_client=DeterministicEmbeddingClient(dimension=16))
        connect.assert_not_called()
