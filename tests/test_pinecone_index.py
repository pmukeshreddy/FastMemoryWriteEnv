from __future__ import annotations

import uuid

import pytest

from fast_memory_write_env.config import load_pinecone_config, pinecone_env_present
from fast_memory_write_env.index import estimate_tokens
from fast_memory_write_env.pinecone_index import PineconeIndex
from fast_memory_write_env.schemas import MemoryRecord


@pytest.mark.integration
@pytest.mark.skipif(
    not pinecone_env_present(),
    reason="Pinecone integration requires PINECONE_API_KEY, PINECONE_INDEX_NAME, PINECONE_CLOUD, PINECONE_REGION",
)
def test_pinecone_index_smoke_upsert_search_delete() -> None:
    config = load_pinecone_config(required=True, namespace="fast-memory-write-env-tests")
    assert config is not None
    index = PineconeIndex(config)
    memory_id = f"mem-pinecone-{uuid.uuid4().hex}"
    memory = MemoryRecord(
        memory_id=memory_id,
        entity_id="account-pinecone",
        content="Pinecone integration smoke memory for outage routing.",
        source_event_ids=["event-pinecone"],
        fact_ids=["fact-pinecone"],
        created_at_ms=1,
        updated_at_ms=1,
        indexed=True,
        estimated_tokens=estimate_tokens("Pinecone integration smoke memory for outage routing."),
    )

    try:
        index.upsert(memory)
        hits = index.search("outage routing", top_k=3)
        assert any(hit.memory_id == memory_id for hit in hits)
    finally:
        index.delete(memory_id)
