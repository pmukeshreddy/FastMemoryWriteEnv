from __future__ import annotations

from fast_memory_write_env.dataset import generate_episode
from fast_memory_write_env.env import FastMemoryWriteEnv, deterministic_memory_id
from fast_memory_write_env.in_memory_index import InMemoryIndex
from fast_memory_write_env.index import estimate_tokens
from fast_memory_write_env.index import SearchResult
from fast_memory_write_env.schemas import DatasetMode, EventCategory, MemoryRecord, MemoryStatus
from fast_memory_write_env.stores import MemoryStore, RawEventStore


def _env(tmp_path) -> FastMemoryWriteEnv:
    return FastMemoryWriteEnv(
        raw_event_store=RawEventStore(tmp_path / "raw.sqlite"),
        memory_store=MemoryStore(tmp_path / "memory.sqlite"),
        retrieval_index=InMemoryIndex(),
    )


class _RetrievalContentOnlyIndex:
    def upsert(self, memory) -> None:
        pass

    def delete(self, memory_id: str) -> None:
        pass

    def search(self, query: str, *, top_k: int = 5, filters=None) -> list[SearchResult]:
        return [
            SearchResult(
                memory_id="mem-retrieved-only",
                score=1.0,
                content="Retrieved backend content should answer without a SQLite fetch.",
                metadata={"content": "Retrieved backend content should answer without a SQLite fetch."},
            )
        ]


def _events_by_category():
    episode = generate_episode(DatasetMode.SMALL, seed=31, episode_index=0)
    events = {
        item.event.category: item.event
        for item in episode.stream
        if item.item_type == "event"
    }
    return episode, events


def test_environment_executes_memory_action_transitions(tmp_path) -> None:
    env = _env(tmp_path)
    _, events = _events_by_category()
    useful = events[EventCategory.USEFUL_FACT]
    noise = events[EventCategory.NOISE]
    urgent = events[EventCategory.URGENT_FACT]

    store_raw = env.execute_action({"action_type": "store_raw", "event": useful.model_dump(mode="json")})
    write = env.execute_action(
        {
            "action_type": "write_memory",
            "memory_id": "mem-primary",
            "entity_id": useful.entity_id,
            "content": useful.content,
            "source_event_ids": [useful.event_id],
            "fact_ids": [fact.fact_id for fact in useful.facts],
            "index_immediately": False,
        }
    )
    assert "mem-primary" not in env.retrieval_index._memories
    delay = env.execute_action(
        {
            "action_type": "delay_index",
            "memory_id": "mem-primary",
            "retry_after_ms": 500,
            "reason": "wait for budget",
        }
    )
    assert "mem-primary" not in env.retrieval_index._memories
    index = env.execute_action({"action_type": "index_now", "memory_id": "mem-primary"})
    assert env.retrieval_index._memories["mem-primary"].indexed is True
    search = env.execute_action({"action_type": "search_memory", "query_text": useful.content, "top_k": 3})
    answer = env.execute_action(
        {
            "action_type": "answer",
            "query_text": useful.content,
            "retrieved_memory_ids": ["mem-primary"],
        }
    )
    store_urgent = env.execute_action({"action_type": "store_raw", "event": urgent.model_dump(mode="json")})
    update = env.execute_action(
        {
            "action_type": "update_memory",
            "memory_id": "mem-primary",
            "content": f"{useful.content} Current verified preference.",
            "source_event_ids": [urgent.event_id],
            "fact_ids": [fact.fact_id for fact in urgent.facts],
            "reason": "new urgent evidence",
            "index_immediately": True,
        }
    )
    env.execute_action({"action_type": "store_raw", "event": noise.model_dump(mode="json")})
    ignore = env.execute_action(
        {"action_type": "ignore_event", "event_id": noise.event_id, "reason": "low-value noise"}
    )
    env.execute_action(
        {
            "action_type": "write_memory",
            "memory_id": "mem-secondary",
            "entity_id": urgent.entity_id,
            "content": urgent.content,
            "source_event_ids": [urgent.event_id],
            "fact_ids": [fact.fact_id for fact in urgent.facts],
            "index_immediately": True,
        }
    )
    env.execute_action(
        {
            "action_type": "write_memory",
            "memory_id": "mem-third",
            "entity_id": urgent.entity_id,
            "content": "Outage escalation requires on-call acknowledgement.",
            "source_event_ids": [urgent.event_id],
            "fact_ids": [],
        }
    )
    compress = env.execute_action(
        {
            "action_type": "compress_memory",
            "source_memory_ids": ["mem-secondary", "mem-third"],
            "target_memory_id": "mem-compressed",
            "compressed_content": "Urgent outage alerts go to the on-call phone with acknowledgement.",
        }
    )
    mark_stale = env.execute_action(
        {
            "action_type": "mark_stale",
            "memory_id": "mem-primary",
            "reason": "compressed memory supersedes it",
            "superseded_by_memory_id": "mem-compressed",
        }
    )

    assert all(
        result.success
        for result in [
            store_raw,
            write,
            delay,
            index,
            search,
            answer,
            store_urgent,
            update,
            ignore,
            compress,
            mark_stale,
        ]
    )
    assert store_raw.action_type == "store_raw"
    assert store_raw.storage_tokens_delta == useful.estimated_tokens
    assert write.storage_tokens_delta > 0
    assert delay.payload["indexed"] is False
    assert search.payload["results"][0]["memory_id"] == "mem-primary"
    assert answer.payload["cited_memory_ids"] == ["mem-primary"]
    assert update.storage_tokens_delta > 0
    assert noise.event_id in env.ignored_event_ids
    assert env.memory_store.require("mem-secondary").status == MemoryStatus.COMPRESSED
    assert env.memory_store.require("mem-third").status == MemoryStatus.COMPRESSED
    assert env.memory_store.require("mem-primary").status == MemoryStatus.STALE
    assert env.retrieval_index.search("Current verified preference") == []
    assert all(result.latency_ms >= 0 for result in [store_raw, write, search, answer])


def test_environment_rejects_memory_actions_with_unknown_source_event_ids(tmp_path) -> None:
    env = _env(tmp_path)
    _, events = _events_by_category()
    useful = events[EventCategory.USEFUL_FACT]

    env.execute_action({"action_type": "store_raw", "event": useful.model_dump(mode="json")})
    fake = env.execute_action(
        {
            "action_type": "write_memory",
            "memory_id": "mem-fake-source",
            "entity_id": useful.entity_id,
            "content": useful.content,
            "source_event_ids": [useful.event_id, "event-not-yet-written"],
            "index_immediately": True,
        }
    )
    valid = env.execute_action(
        {
            "action_type": "write_memory",
            "memory_id": "mem-valid-source",
            "entity_id": useful.entity_id,
            "content": useful.content,
            "source_event_ids": [useful.event_id],
            "index_immediately": True,
        }
    )

    assert fake.success is False
    assert "unknown source_event_ids" in str(fake.error)
    assert env.memory_store.get("mem-fake-source") is None
    assert valid.success is True
    assert env.memory_store.require("mem-valid-source").source_event_ids == [useful.event_id]


def test_environment_rejects_second_index_when_index_budget_is_exhausted(tmp_path) -> None:
    env = FastMemoryWriteEnv(
        raw_event_store=RawEventStore(tmp_path / "raw.sqlite"),
        memory_store=MemoryStore(tmp_path / "memory.sqlite"),
        retrieval_index=InMemoryIndex(),
        storage_budget_tokens_remaining=100,
        indexing_budget_operations_remaining=1,
    )
    _, events = _events_by_category()
    useful = events[EventCategory.USEFUL_FACT]
    urgent = events[EventCategory.URGENT_FACT]
    env.execute_action({"action_type": "store_raw", "event": useful.model_dump(mode="json")})
    env.execute_action({"action_type": "store_raw", "event": urgent.model_dump(mode="json")})

    first = env.execute_action(
        {
            "action_type": "write_memory",
            "memory_id": "mem-index-one",
            "entity_id": useful.entity_id,
            "content": useful.content,
            "source_event_ids": [useful.event_id],
            "index_immediately": True,
        }
    )
    second = env.execute_action(
        {
            "action_type": "write_memory",
            "memory_id": "mem-index-two",
            "entity_id": urgent.entity_id,
            "content": urgent.content,
            "source_event_ids": [urgent.event_id],
            "index_immediately": True,
        }
    )

    assert first.success is True
    assert env.indexing_budget_operations_remaining == 0
    assert second.success is False
    assert "indexing budget exceeded" in str(second.error)
    assert env.memory_store.get("mem-index-two") is None


def test_environment_rejects_second_write_when_storage_budget_is_exhausted(tmp_path) -> None:
    env = FastMemoryWriteEnv(
        raw_event_store=RawEventStore(tmp_path / "raw.sqlite"),
        memory_store=MemoryStore(tmp_path / "memory.sqlite"),
        retrieval_index=InMemoryIndex(),
        storage_budget_tokens_remaining=1,
        indexing_budget_operations_remaining=10,
    )
    _, events = _events_by_category()
    useful = events[EventCategory.USEFUL_FACT]
    urgent = events[EventCategory.URGENT_FACT]
    env.execute_action({"action_type": "store_raw", "event": useful.model_dump(mode="json")})
    env.execute_action({"action_type": "store_raw", "event": urgent.model_dump(mode="json")})

    first = env.execute_action(
        {
            "action_type": "write_memory",
            "memory_id": "mem-storage-one",
            "entity_id": useful.entity_id,
            "content": "alpha",
            "source_event_ids": [useful.event_id],
        }
    )
    second = env.execute_action(
        {
            "action_type": "write_memory",
            "memory_id": "mem-storage-two",
            "entity_id": urgent.entity_id,
            "content": "beta",
            "source_event_ids": [urgent.event_id],
        }
    )

    assert first.success is True
    assert env.storage_budget_tokens_remaining == 0
    assert second.success is False
    assert "storage budget exceeded" in str(second.error)
    assert env.memory_store.get("mem-storage-two") is None


def test_environment_returns_structured_failure_for_invalid_action(tmp_path) -> None:
    env = _env(tmp_path)

    result = env.execute_action({"action_type": "index_now", "memory_id": "missing-memory"})

    assert result.success is False
    assert result.action_type == "index_now"
    assert "memory not found" in result.error


def test_episode_loop_stores_events_and_runs_injected_actions(tmp_path) -> None:
    env = _env(tmp_path)
    episode = generate_episode(DatasetMode.SMALL, seed=41, episode_index=0)
    useful = next(
        item.event
        for item in episode.stream
        if item.item_type == "event" and item.event.facts
    )
    memory_id = deterministic_memory_id([useful.event_id], useful.content)

    results = env.run_episode(
        episode,
        action_batches={
            useful.event_id: [
                {
                    "action_type": "write_memory",
                    "memory_id": memory_id,
                    "entity_id": useful.entity_id,
                    "content": useful.content,
                    "source_event_ids": [useful.event_id],
                    "fact_ids": [fact.fact_id for fact in useful.facts],
                    "index_immediately": True,
                }
            ]
        },
    )

    event_count = sum(1 for item in episode.stream if item.item_type == "event")
    query_count = sum(1 for item in episode.stream if item.item_type == "query")

    assert env.raw_event_store.count() == event_count
    assert env.memory_store.require(memory_id).indexed is True
    assert sum(result.action_type == "store_raw" for result in results) == event_count
    assert sum(result.action_type == "search_memory" for result in results) == query_count
    assert sum(result.action_type == "answer" for result in results) == query_count


def test_answer_uses_retrieved_content_before_sqlite_lookup(tmp_path) -> None:
    env = FastMemoryWriteEnv(
        raw_event_store=RawEventStore(tmp_path / "raw.sqlite"),
        memory_store=MemoryStore(tmp_path / "memory.sqlite"),
        retrieval_index=_RetrievalContentOnlyIndex(),
    )

    search = env.execute_action(
        {
            "action_type": "search_memory",
            "query_text": "What did the retrieval backend return?",
        }
    )
    answer = env.execute_action(
        {
            "action_type": "answer",
            "query_text": "What did the retrieval backend return?",
            "retrieved_memory_ids": ["mem-retrieved-only"],
        }
    )

    assert search.success is True
    assert env.memory_store.get("mem-retrieved-only") is None
    assert answer.payload["answer"] == "Retrieved backend content should answer without a SQLite fetch."
    assert answer.payload["cited_memory_ids"] == ["mem-retrieved-only"]


def test_answer_does_not_fallback_to_sqlite_when_retrieval_misses(tmp_path) -> None:
    env = _env(tmp_path)
    env.memory_store.create(
        MemoryRecord(
            memory_id="mem-sqlite-only",
            entity_id="account-ada",
            content="SQLite-only content must not be used as retrieval fallback.",
            source_event_ids=["event-001"],
            fact_ids=["fact-001"],
            created_at_ms=1,
            updated_at_ms=1,
            indexed=True,
            estimated_tokens=estimate_tokens("SQLite-only content must not be used as retrieval fallback."),
        )
    )

    answer = env.execute_action(
        {
            "action_type": "answer",
            "query_text": "What did retrieval return?",
            "retrieved_memory_ids": ["mem-sqlite-only"],
        }
    )

    assert answer.payload["answer"] == "I do not know from indexed memory."
    assert answer.payload["cited_memory_ids"] == []
