"""Tests for the env's LLM-as-composer answer path.

The env composes per-query answers through a single LLM call. There is no
silent fallback to deterministic top-1; if the configured client cannot
produce a valid composition after retries, the answer action fails loudly
with a structured error so the caller cannot mistake a degraded answer for
a real one.
"""

from __future__ import annotations

import json

import pytest

from fast_memory_write_env.actions import (
    ActionType,
    AnswerAction,
    SearchMemoryAction,
    WriteMemoryAction,
)
from fast_memory_write_env.env import FastMemoryWriteEnv
from fast_memory_write_env.in_memory_index import InMemoryIndex
from fast_memory_write_env.llm_client import LLMClientError, LLMMessage, LLMResponse
from tests._test_llm_client import DeterministicTestLLMClient
from fast_memory_write_env.schemas import EventCategory, EventFact, RawEvent
from fast_memory_write_env.stores import MemoryStore, RawEventStore


def _make_event(event_id: str, content: str, *, timestamp_ms: int = 0) -> RawEvent:
    return RawEvent(
        event_id=event_id,
        episode_id="ep-compose",
        timestamp_ms=timestamp_ms,
        source="test",
        user_id="user-1",
        entity_id="entity-1",
        category=EventCategory.USEFUL_FACT,
        content=content,
        facts=[
            EventFact(
                fact_id=f"{event_id}-fact",
                entity_id="entity-1",
                attribute="x",
                value=content,
                source_event_id=event_id,
                valid_from_ms=timestamp_ms,
            )
        ],
    )


def _build_env(tmp_path, *, answer_llm_client=None) -> FastMemoryWriteEnv:
    return FastMemoryWriteEnv(
        raw_event_store=RawEventStore(tmp_path / "raw.sqlite"),
        memory_store=MemoryStore(tmp_path / "memory.sqlite"),
        retrieval_index=InMemoryIndex(),
        answer_llm_client=answer_llm_client,
    )


def _seed_two_memories(env: FastMemoryWriteEnv) -> tuple[str, str]:
    e_dee = _make_event("event-dee", "account-dee reported deployment window 1 at 09:00 UTC.")
    e_ada = _make_event(
        "event-ada",
        "account-ada now prefers SMS instead of email for renewal notices.",
        timestamp_ms=10,
    )
    env.execute_action({"action_type": "store_raw", "event": e_dee.model_dump(mode="json")})
    env.execute_action({"action_type": "store_raw", "event": e_ada.model_dump(mode="json")})
    env.execute_action(
        WriteMemoryAction(
            memory_id="mem-dee",
            entity_id=e_dee.entity_id,
            content=e_dee.content,
            source_event_ids=[e_dee.event_id],
            index_immediately=True,
        )
    )
    env.execute_action(
        WriteMemoryAction(
            memory_id="mem-ada",
            entity_id=e_ada.entity_id,
            content=e_ada.content,
            source_event_ids=[e_ada.event_id],
            index_immediately=True,
        )
    )
    return "mem-dee", "mem-ada"


def test_answer_action_fails_loudly_when_no_llm_client(tmp_path) -> None:
    """No silent top-1 fallback. The action result must surface the failure
    so the caller can fix the configuration."""

    env = _build_env(tmp_path, answer_llm_client=None)
    dee_id, ada_id = _seed_two_memories(env)

    env.execute_action(SearchMemoryAction(query_text="dee deployment window?", top_k=5))
    answer = env.execute_action(
        AnswerAction(
            query_text="dee deployment window?",
            retrieved_memory_ids=[dee_id, ada_id],
        )
    )

    assert answer.success is False
    assert answer.action_type == ActionType.ANSWER
    assert "answer_llm_client" in (answer.error or "")


def test_llm_compose_drops_irrelevant_memories(tmp_path) -> None:
    composed = {
        "answer": "Account-dee's deployment window was reported at 09:00 UTC.",
        "cited_memory_ids": ["mem-dee"],
    }
    client = DeterministicTestLLMClient(responses=[composed])
    env = _build_env(tmp_path, answer_llm_client=client)
    dee_id, ada_id = _seed_two_memories(env)

    # Use a query that matches BOTH memories ("account") so the env passes
    # both candidates to the LLM and we verify it picks only the relevant one.
    env.execute_action(SearchMemoryAction(query_text="account deployment SMS", top_k=5))
    answer = env.execute_action(
        AnswerAction(
            query_text="What deployment window was reported for account-dee?",
            retrieved_memory_ids=[dee_id, ada_id],
        )
    )

    assert answer.success is True
    assert "09:00" in answer.payload["answer"]
    # The unrelated SMS memory must not leak into the answer.
    assert "SMS" not in answer.payload["answer"]
    assert answer.payload["cited_memory_ids"] == [dee_id]
    user_message = json.loads(client.calls[0][-1].content)
    assert {m["memory_id"] for m in user_message["candidate_memories"]} == {dee_id, ada_id}


def test_compose_retries_on_invalid_citations_and_succeeds(tmp_path) -> None:
    """A first response that cites a non-candidate id is rejected; the env
    sends a repair message and accepts the corrected response."""

    bad = {"answer": "answer", "cited_memory_ids": ["mem-fabricated"]}
    good = {
        "answer": "Account-dee's deployment window was reported at 09:00 UTC.",
        "cited_memory_ids": ["mem-dee"],
    }
    client = DeterministicTestLLMClient(responses=[bad, good])
    env = _build_env(tmp_path, answer_llm_client=client)
    dee_id, ada_id = _seed_two_memories(env)

    env.execute_action(SearchMemoryAction(query_text="dee?", top_k=5))
    answer = env.execute_action(
        AnswerAction(query_text="dee?", retrieved_memory_ids=[dee_id, ada_id])
    )

    assert answer.success is True
    assert answer.payload["cited_memory_ids"] == [dee_id]
    # First call + one repair retry.
    assert len(client.calls) == 2
    assert "Repair the previous response" in client.calls[1][-1].content


def test_compose_fails_loudly_after_exhausted_retries(tmp_path) -> None:
    """Three malformed responses (initial + 2 retries) must surface a
    structured failure rather than degrade silently."""

    bad = {"answer": "x", "cited_memory_ids": ["mem-fabricated"]}
    client = DeterministicTestLLMClient(responses=[bad, bad, bad])
    env = _build_env(tmp_path, answer_llm_client=client)
    dee_id, ada_id = _seed_two_memories(env)

    env.execute_action(SearchMemoryAction(query_text="dee?", top_k=5))
    answer = env.execute_action(
        AnswerAction(query_text="dee?", retrieved_memory_ids=[dee_id, ada_id])
    )

    assert answer.success is False
    assert "answer composition failed" in (answer.error or "")
    assert len(client.calls) == 3


def test_compose_rejects_non_abstention_with_no_citations(tmp_path) -> None:
    """A non-abstention answer without citations is treated as invalid; the
    env retries (and ultimately fails if the LLM keeps producing it)."""

    hallucinated = {"answer": "Some claim without any citation.", "cited_memory_ids": []}
    client = DeterministicTestLLMClient(responses=[hallucinated, hallucinated, hallucinated])
    env = _build_env(tmp_path, answer_llm_client=client)
    dee_id, ada_id = _seed_two_memories(env)

    env.execute_action(SearchMemoryAction(query_text="dee?", top_k=5))
    answer = env.execute_action(
        AnswerAction(query_text="dee?", retrieved_memory_ids=[dee_id, ada_id])
    )

    assert answer.success is False
    assert len(client.calls) == 3


def test_compose_passes_through_abstention(tmp_path) -> None:
    composed = {
        "answer": "I do not know from indexed memory.",
        "cited_memory_ids": [],
    }
    client = DeterministicTestLLMClient(responses=[composed])
    env = _build_env(tmp_path, answer_llm_client=client)
    dee_id, ada_id = _seed_two_memories(env)

    env.execute_action(SearchMemoryAction(query_text="dee?", top_k=5))
    answer = env.execute_action(
        AnswerAction(query_text="dee?", retrieved_memory_ids=[dee_id, ada_id])
    )

    assert answer.success is True
    assert answer.payload["answer"] == "I do not know from indexed memory."
    assert answer.payload["cited_memory_ids"] == []


def test_compose_retries_then_fails_on_persistent_client_errors(tmp_path) -> None:
    """LLMClientError is retried up to the budget; persistent errors surface
    as a structured action failure."""

    class _AlwaysErrors:
        def __init__(self) -> None:
            self.calls = 0

        def complete(self, messages, *, temperature=0.0, response_format=None):
            self.calls += 1
            raise LLMClientError("transient network failure")

    client = _AlwaysErrors()
    env = _build_env(tmp_path, answer_llm_client=client)
    dee_id, ada_id = _seed_two_memories(env)

    env.execute_action(SearchMemoryAction(query_text="dee?", top_k=5))
    answer = env.execute_action(
        AnswerAction(query_text="dee?", retrieved_memory_ids=[dee_id, ada_id])
    )

    assert answer.success is False
    assert "transient network failure" in (answer.error or "")
    assert client.calls == 3  # initial + 2 retries


def test_answer_returns_abstention_without_consulting_llm_when_no_retrieval(tmp_path) -> None:
    """Empty retrieval is the one case that does not need the LLM: there is
    nothing to compose from, so abstention is the principled response."""

    class _Boom:
        calls = 0

        def complete(self, messages, *, temperature=0.0, response_format=None):
            type(self).calls += 1
            raise AssertionError("LLM should not be consulted when retrieval is empty")

    env = _build_env(tmp_path, answer_llm_client=_Boom())

    answer = env.execute_action(
        AnswerAction(query_text="anything?", retrieved_memory_ids=[])
    )

    assert answer.success is True
    assert answer.payload["answer"] == "I do not know from indexed memory."
    assert answer.payload["cited_memory_ids"] == []
    assert _Boom.calls == 0


def test_mock_client_default_compose_response_serves_top_candidate(tmp_path) -> None:
    """Default DeterministicTestLLMClient (no queued responses) routes by request shape and
    returns the top candidate as the cited memory. This is the mock's
    deterministic behavior, not a fallback inside the env."""

    env = _build_env(tmp_path, answer_llm_client=DeterministicTestLLMClient())
    dee_id, ada_id = _seed_two_memories(env)

    env.execute_action(SearchMemoryAction(query_text="account-dee deployment", top_k=5))
    answer = env.execute_action(
        AnswerAction(
            query_text="account-dee deployment",
            retrieved_memory_ids=[dee_id, ada_id],
        )
    )

    assert answer.success is True
    assert answer.payload["cited_memory_ids"] == [dee_id]
