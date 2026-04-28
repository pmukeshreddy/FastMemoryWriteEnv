from __future__ import annotations

import pytest

from fast_memory_write_env.actions import WriteMemoryAction
from fast_memory_write_env.dataset import generate_episode
from fast_memory_write_env.env import FastMemoryWriteEnv
from fast_memory_write_env.in_memory_index import InMemoryIndex
from fast_memory_write_env.llm_client import LLMClientError, MockLLMClient
from fast_memory_write_env.policies import (
    LLMMemoryWritePolicy,
    NoMemoryBaseline,
    OraclePolicy,
    StoreEverythingBaseline,
)
from fast_memory_write_env.schemas import DatasetMode, EventCategory
from fast_memory_write_env.stores import MemoryStore, RawEventStore


def _event(category: EventCategory):
    episode = generate_episode(DatasetMode.SMALL, seed=51, episode_index=0)
    return next(
        item.event
        for item in episode.stream
        if item.item_type == "event" and item.event.category == category
    )


def _policy(client: MockLLMClient) -> LLMMemoryWritePolicy:
    return LLMMemoryWritePolicy(llm_client=client, max_retries=2)


def test_mock_policy_writes_and_indexes_useful_event() -> None:
    event = _event(EventCategory.USEFUL_FACT)
    client = MockLLMClient()
    policy = _policy(client)

    actions = policy.decide(
        new_event=event,
        active_memories=[],
        recent_events=[],
        latency_budget_ms=250,
        storage_budget_tokens_remaining=1000,
        indexing_budget_operations_remaining=1,
    )

    assert len(actions) == 1
    assert isinstance(actions[0], WriteMemoryAction)
    assert actions[0].source_event_ids == [event.event_id]
    assert actions[0].index_immediately is True
    assert len(client.calls) == 1


def test_mock_policy_ignores_noise_event() -> None:
    event = _event(EventCategory.NOISE)
    actions = _policy(MockLLMClient()).decide(
        new_event=event,
        active_memories=[],
        recent_events=[],
        latency_budget_ms=250,
        storage_budget_tokens_remaining=1000,
        indexing_budget_operations_remaining=1,
    )

    assert actions[0].action_type == "ignore_event"
    assert actions[0].event_id == event.event_id


def test_policy_repairs_invalid_json_response() -> None:
    event = _event(EventCategory.NOISE)
    client = MockLLMClient(
        responses=[
            "not-json",
            {
                "actions": [
                    {
                        "action_type": "ignore_event",
                        "event_id": event.event_id,
                        "reason": "repaired response",
                    }
                ]
            },
        ]
    )
    policy = LLMMemoryWritePolicy(llm_client=client, max_retries=1)

    actions = policy.decide(
        new_event=event,
        active_memories=[],
        recent_events=[],
        latency_budget_ms=250,
        storage_budget_tokens_remaining=1000,
        indexing_budget_operations_remaining=0,
    )

    assert actions[0].action_type == "ignore_event"
    assert len(client.calls) == 2
    assert "Repair the previous response" in client.calls[1][-1].content


def test_policy_repairs_schema_invalid_response() -> None:
    event = _event(EventCategory.USEFUL_FACT)
    client = MockLLMClient(
        responses=[
            {"actions": [{"action_type": "write_memory"}]},
            {
                "actions": [
                    {
                        "action_type": "write_memory",
                        "memory_id": "mem-repaired",
                        "entity_id": event.entity_id,
                        "content": event.content,
                        "source_event_ids": [event.event_id],
                        "fact_ids": [fact.fact_id for fact in event.facts],
                    }
                ]
            },
        ]
    )

    actions = LLMMemoryWritePolicy(llm_client=client, max_retries=1).decide(
        new_event=event,
        active_memories=[],
        recent_events=[],
        latency_budget_ms=250,
        storage_budget_tokens_remaining=1000,
        indexing_budget_operations_remaining=1,
    )

    assert actions[0].action_type == "write_memory"
    assert actions[0].memory_id == "mem-repaired"


def test_policy_fails_after_retry_budget() -> None:
    event = _event(EventCategory.NOISE)
    client = MockLLMClient(responses=["not-json", "still-not-json"])

    with pytest.raises(LLMClientError):
        LLMMemoryWritePolicy(llm_client=client, max_retries=1).decide(
            new_event=event,
            active_memories=[],
            recent_events=[],
            latency_budget_ms=250,
            storage_budget_tokens_remaining=1000,
            indexing_budget_operations_remaining=1,
        )


def test_policy_outputs_execute_through_environment_without_policy_mutation(tmp_path) -> None:
    event = _event(EventCategory.URGENT_FACT)
    env = FastMemoryWriteEnv(
        raw_event_store=RawEventStore(tmp_path / "raw.sqlite"),
        memory_store=MemoryStore(tmp_path / "memory.sqlite"),
        retrieval_index=InMemoryIndex(),
    )
    actions = _policy(MockLLMClient()).decide(
        new_event=event,
        active_memories=env.memory_store.list_active(),
        recent_events=[],
        latency_budget_ms=250,
        storage_budget_tokens_remaining=1000,
        indexing_budget_operations_remaining=1,
    )

    assert env.memory_store.list_all() == []

    env.execute_action({"action_type": "store_raw", "event": event.model_dump(mode="json")})
    results = env.execute_actions(list(actions))

    assert all(result.success for result in results)
    assert len(env.memory_store.list_all()) == 1
    assert env.memory_store.list_all()[0].indexed is True


def test_minimal_baselines_only_behaviors() -> None:
    event = _event(EventCategory.USEFUL_FACT)
    noise = _event(EventCategory.NOISE)

    no_memory = NoMemoryBaseline().decide(
        new_event=event,
        active_memories=[],
        recent_events=[],
        latency_budget_ms=1,
        storage_budget_tokens_remaining=0,
        indexing_budget_operations_remaining=0,
    )
    store_all = StoreEverythingBaseline().decide(
        new_event=event,
        active_memories=[],
        recent_events=[],
        latency_budget_ms=1,
        storage_budget_tokens_remaining=1000,
        indexing_budget_operations_remaining=1,
    )
    oracle_noise = OraclePolicy().decide(
        new_event=noise,
        active_memories=[],
        recent_events=[],
        latency_budget_ms=1,
        storage_budget_tokens_remaining=1000,
        indexing_budget_operations_remaining=1,
    )

    assert no_memory[0].action_type == "ignore_event"
    assert store_all[0].action_type == "write_memory"
    assert oracle_noise[0].action_type == "ignore_event"
