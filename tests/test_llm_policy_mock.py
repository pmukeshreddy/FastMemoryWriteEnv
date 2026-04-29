from __future__ import annotations

import json

import pytest

from fast_memory_write_env.actions import (
    PolicyPlanError,
    UpdateMemoryAction,
    WriteMemoryAction,
    WriteMemoryProposal,
    compile_policy_actions,
)
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
from fast_memory_write_env.schemas import DatasetMode, EventCategory, MemoryRecord
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

    proposals = policy.decide(
        new_event=event,
        active_memories=[],
        recent_events=[],
        latency_budget_ms=250,
        storage_budget_tokens_remaining=1000,
        indexing_budget_operations_remaining=1,
    )

    assert len(proposals) == 1
    assert isinstance(proposals[0], WriteMemoryProposal)
    assert proposals[0].source_event_ids == [event.event_id]
    assert proposals[0].index_immediately is True
    assert len(client.calls) == 1


def test_policy_proposal_schema_does_not_accept_llm_supplied_memory_id() -> None:
    """The LLM may not author memory IDs; structured-output forbids the field."""

    event = _event(EventCategory.USEFUL_FACT)
    client = MockLLMClient(
        responses=[
            {
                "actions": [
                    {
                        "action_type": "write_memory",
                        "memory_id": "mem-llm-authored",
                        "entity_id": event.entity_id,
                        "content": event.content,
                        "source_event_ids": [event.event_id],
                    }
                ]
            },
        ]
    )

    with pytest.raises(LLMClientError):
        LLMMemoryWritePolicy(llm_client=client, max_retries=0).decide(
            new_event=event,
            active_memories=[],
            recent_events=[],
            latency_budget_ms=250,
            storage_budget_tokens_remaining=1000,
            indexing_budget_operations_remaining=1,
        )


def test_policy_prompt_hides_evaluator_labels_from_events_and_memories() -> None:
    event = _event(EventCategory.CONTRADICTION)
    recent = _event(EventCategory.STALE_UPDATE)
    memory = MemoryRecord(
        memory_id="mem-existing",
        entity_id=event.entity_id,
        content="Existing visible memory.",
        source_event_ids=["event-secret"],
        fact_ids=["fact-secret"],
        created_at_ms=1,
        updated_at_ms=2,
        indexed=True,
        estimated_tokens=3,
        metadata={
            "role": "user",
            "evidence_label_source": "hidden",
            "answer": "hidden",
        },
    )
    client = MockLLMClient()

    _policy(client).decide(
        new_event=event,
        active_memories=[memory],
        recent_events=[recent],
        latency_budget_ms=250,
        storage_budget_tokens_remaining=1000,
        indexing_budget_operations_remaining=1,
    )

    payload = json.loads(client.calls[0][-1].content)
    hidden_event_keys = {"category", "facts", "duplicate_of", "contradicts", "supersedes", "tags", "priority"}
    assert hidden_event_keys.isdisjoint(payload["new_event"])
    assert hidden_event_keys.isdisjoint(payload["recent_events"][0])
    assert "fact_ids" not in payload["active_memories"][0]
    assert payload["active_memories"][0]["metadata"] == {"role": "user"}
    assert "evidence_label_source" not in payload["new_event"]["metadata"]


def test_mock_policy_ignores_noise_event() -> None:
    event = _event(EventCategory.NOISE)
    proposals = _policy(MockLLMClient()).decide(
        new_event=event,
        active_memories=[],
        recent_events=[],
        latency_budget_ms=250,
        storage_budget_tokens_remaining=1000,
        indexing_budget_operations_remaining=1,
    )

    assert proposals[0].action_type == "ignore_event"
    assert proposals[0].event_id == event.event_id


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

    proposals = policy.decide(
        new_event=event,
        active_memories=[],
        recent_events=[],
        latency_budget_ms=250,
        storage_budget_tokens_remaining=1000,
        indexing_budget_operations_remaining=0,
    )

    assert proposals[0].action_type == "ignore_event"
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
                        "entity_id": event.entity_id,
                        "content": event.content,
                        "source_event_ids": [event.event_id],
                    }
                ]
            },
        ]
    )

    proposals = LLMMemoryWritePolicy(llm_client=client, max_retries=1).decide(
        new_event=event,
        active_memories=[],
        recent_events=[],
        latency_budget_ms=250,
        storage_budget_tokens_remaining=1000,
        indexing_budget_operations_remaining=1,
    )

    assert proposals[0].action_type == "write_memory"
    assert isinstance(proposals[0], WriteMemoryProposal)


def test_policy_requests_strict_structured_output_schema() -> None:
    event = _event(EventCategory.USEFUL_FACT)
    client = MockLLMClient()

    LLMMemoryWritePolicy(llm_client=client, max_retries=0).decide(
        new_event=event,
        active_memories=[],
        recent_events=[],
        latency_budget_ms=250,
        storage_budget_tokens_remaining=1000,
        indexing_budget_operations_remaining=1,
    )

    response_format = client.response_formats[0]
    assert response_format is not None
    assert response_format["type"] == "json_schema"
    assert response_format["json_schema"]["strict"] is True
    schema = response_format["json_schema"]["schema"]
    assert schema["type"] == "object"
    assert schema["additionalProperties"] is False
    write_memory_schema = schema["properties"]["actions"]["items"]["anyOf"][0]
    assert "memory_id" not in write_memory_schema["properties"]
    assert "memory_id" not in write_memory_schema["required"]
    compress_memory_schema = schema["properties"]["actions"]["items"]["anyOf"][4]
    assert compress_memory_schema["properties"]["action_type"]["enum"] == ["compress_memory"]
    assert "target_memory_id" not in compress_memory_schema["properties"]


def test_environment_owns_memory_ids_when_compiling_proposals(tmp_path) -> None:
    event = _event(EventCategory.USEFUL_FACT)
    client = MockLLMClient(
        responses=[
            {
                "actions": [
                    {
                        "action_type": "write_memory",
                        "entity_id": event.entity_id,
                        "content": event.content,
                        "source_event_ids": [event.event_id],
                        "index_immediately": True,
                    }
                ]
            }
        ]
    )
    env = FastMemoryWriteEnv(
        raw_event_store=RawEventStore(tmp_path / "raw.sqlite"),
        memory_store=MemoryStore(tmp_path / "memory.sqlite"),
        retrieval_index=InMemoryIndex(),
    )

    proposals = _policy(client).decide(
        new_event=event,
        active_memories=[],
        recent_events=[],
        latency_budget_ms=250,
        storage_budget_tokens_remaining=1000,
        indexing_budget_operations_remaining=1,
    )
    assert isinstance(proposals[0], WriteMemoryProposal)

    env.execute_action({"action_type": "store_raw", "event": event.model_dump(mode="json")})
    actions = compile_policy_actions(proposals, active_memory_ids=[])

    assert isinstance(actions[0], WriteMemoryAction)
    assert actions[0].memory_id.startswith("mem-")
    assert actions[0].fact_ids == []
    result = env.execute_action(actions[0])

    assert result.success is True
    stored = env.memory_store.require(actions[0].memory_id)
    assert stored.fact_ids == []
    assert stored.indexed is True


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
    proposals = _policy(MockLLMClient()).decide(
        new_event=event,
        active_memories=env.memory_store.list_active(),
        recent_events=[],
        latency_budget_ms=250,
        storage_budget_tokens_remaining=1000,
        indexing_budget_operations_remaining=1,
    )

    assert env.memory_store.list_all() == []

    env.execute_action({"action_type": "store_raw", "event": event.model_dump(mode="json")})
    actions = compile_policy_actions(
        proposals,
        active_memory_ids=[memory.memory_id for memory in env.memory_store.list_active()],
    )
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
    assert isinstance(store_all[0], WriteMemoryProposal)
    assert oracle_noise[0].action_type == "ignore_event"


def test_baseline_proposals_compile_to_executable_actions(tmp_path) -> None:
    """StoreEverythingBaseline should produce proposals that the env can run."""

    event = _event(EventCategory.USEFUL_FACT)
    env = FastMemoryWriteEnv(
        raw_event_store=RawEventStore(tmp_path / "raw.sqlite"),
        memory_store=MemoryStore(tmp_path / "memory.sqlite"),
        retrieval_index=InMemoryIndex(),
    )
    proposals = StoreEverythingBaseline().decide(
        new_event=event,
        active_memories=[],
        recent_events=[],
        latency_budget_ms=1,
        storage_budget_tokens_remaining=1000,
        indexing_budget_operations_remaining=1,
    )
    env.execute_action({"action_type": "store_raw", "event": event.model_dump(mode="json")})
    actions = compile_policy_actions(proposals, active_memory_ids=[])

    assert isinstance(actions[0], WriteMemoryAction)
    assert env.execute_action(actions[0]).success is True


def test_oracle_update_targets_existing_active_memory_only() -> None:
    """OraclePolicy must reference an active memory when applying an update."""

    contradiction = _event(EventCategory.CONTRADICTION)
    matching_memory = MemoryRecord(
        memory_id="mem-active",
        entity_id=contradiction.entity_id,
        content="Old preference.",
        source_event_ids=["event-old"],
        created_at_ms=0,
        updated_at_ms=0,
    )
    proposals = OraclePolicy().decide(
        new_event=contradiction,
        active_memories=[matching_memory],
        recent_events=[],
        latency_budget_ms=1,
        storage_budget_tokens_remaining=1000,
        indexing_budget_operations_remaining=1,
    )

    assert isinstance(proposals[0], UpdateMemoryAction)
    assert proposals[0].memory_id == "mem-active"
    actions = compile_policy_actions(proposals, active_memory_ids=["mem-active"])
    assert isinstance(actions[0], UpdateMemoryAction)


def test_compile_rejects_oracle_update_when_target_no_longer_active() -> None:
    contradiction = _event(EventCategory.CONTRADICTION)
    proposals = OraclePolicy().decide(
        new_event=contradiction,
        active_memories=[
            MemoryRecord(
                memory_id="mem-stale-now",
                entity_id=contradiction.entity_id,
                content="Old preference.",
                source_event_ids=["event-old"],
                created_at_ms=0,
                updated_at_ms=0,
            )
        ],
        recent_events=[],
        latency_budget_ms=1,
        storage_budget_tokens_remaining=1000,
        indexing_budget_operations_remaining=1,
    )

    with pytest.raises(PolicyPlanError):
        compile_policy_actions(proposals, active_memory_ids=[])
