"""Tests for the env's LLM-as-composer answer path.

The env's old ``_answer`` concatenated every retrieved memory into one string,
which polluted single-fact answers with unrelated retrieved facts and caused
the LLM judge to (correctly) mark them ``NO``. The new path:

* When ``answer_llm_client`` is configured, asks the LLM to write a focused
  answer and return the exact ``cited_memory_ids`` it actually used. The env
  intersects those with the retrieved set so the LLM cannot invent citations.
* Falls back to a deterministic top-1 answer (single citation) if no client
  is configured, the client raises, returns malformed JSON, or returns a
  non-abstention answer with no citations.
"""

from __future__ import annotations

import json

from fast_memory_write_env.actions import AnswerAction, SearchMemoryAction, WriteMemoryAction
from fast_memory_write_env.env import FastMemoryWriteEnv
from fast_memory_write_env.in_memory_index import InMemoryIndex
from fast_memory_write_env.llm_client import MockLLMClient
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
    e_ada = _make_event("event-ada", "account-ada now prefers SMS instead of email for renewal notices.", timestamp_ms=10)
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


def test_top1_fallback_when_no_llm_client(tmp_path) -> None:
    """Without an LLM client the env answers with the single best memory,
    not a concatenation of every retrieved hit."""

    env = _build_env(tmp_path)
    dee_id, ada_id = _seed_two_memories(env)

    env.execute_action(
        SearchMemoryAction(
            query_text="What deployment window was reported for account-dee?",
            top_k=5,
        )
    )
    answer = env.execute_action(
        AnswerAction(
            query_text="What deployment window was reported for account-dee?",
            retrieved_memory_ids=[dee_id, ada_id],
        )
    )

    assert answer.success is True
    assert answer.payload["answer"] == "account-dee reported deployment window 1 at 09:00 UTC."
    assert answer.payload["cited_memory_ids"] == [dee_id]


def test_llm_compose_drops_irrelevant_memories(tmp_path) -> None:
    """The LLM answers only with the relevant memory and cites only it."""

    composed = {
        "answer": "Account-dee's deployment window was reported at 09:00 UTC.",
        "cited_memory_ids": ["mem-dee"],
    }
    client = MockLLMClient(responses=[composed])
    env = _build_env(tmp_path, answer_llm_client=client)
    dee_id, ada_id = _seed_two_memories(env)

    env.execute_action(
        SearchMemoryAction(
            query_text="What deployment window was reported for account-dee?",
            top_k=5,
        )
    )
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
    # The compose call carried both candidates so the LLM had to pick.
    user_message = json.loads(client.calls[0][-1].content)
    assert {m["memory_id"] for m in user_message["candidate_memories"]} == {dee_id, ada_id}


def test_llm_compose_invented_citations_are_dropped(tmp_path) -> None:
    """Cited IDs not in the retrieved set must not appear in the env's payload;
    the env intersects with the retrieved set defensively."""

    composed = {
        "answer": "Account-dee's deployment window was reported at 09:00 UTC.",
        "cited_memory_ids": ["mem-dee", "mem-fabricated"],
    }
    client = MockLLMClient(responses=[composed])
    env = _build_env(tmp_path, answer_llm_client=client)
    dee_id, ada_id = _seed_two_memories(env)

    env.execute_action(SearchMemoryAction(query_text="dee deployment window?", top_k=5))
    answer = env.execute_action(
        AnswerAction(
            query_text="dee deployment window?",
            retrieved_memory_ids=[dee_id, ada_id],
        )
    )

    assert answer.payload["cited_memory_ids"] == [dee_id]


def test_llm_compose_falls_back_to_top1_when_client_errors(tmp_path) -> None:
    """A bad response from the LLM should not regress the env to abstention or
    silent failure; we fall back to the deterministic top-1 answer."""

    client = MockLLMClient(responses=["this is not json"])
    env = _build_env(tmp_path, answer_llm_client=client)
    dee_id, ada_id = _seed_two_memories(env)

    env.execute_action(SearchMemoryAction(query_text="dee deployment window?", top_k=5))
    answer = env.execute_action(
        AnswerAction(
            query_text="dee deployment window?",
            retrieved_memory_ids=[dee_id, ada_id],
        )
    )

    assert answer.payload["answer"] == "account-dee reported deployment window 1 at 09:00 UTC."
    assert answer.payload["cited_memory_ids"] == [dee_id]


def test_llm_compose_falls_back_to_top1_when_answer_has_no_citations(tmp_path) -> None:
    """A non-abstention answer with empty citations is almost always the LLM
    hallucinating; fall back so evidence_correct stays honest."""

    composed = {"answer": "Some claim without any citation.", "cited_memory_ids": []}
    client = MockLLMClient(responses=[composed])
    env = _build_env(tmp_path, answer_llm_client=client)
    dee_id, ada_id = _seed_two_memories(env)

    env.execute_action(SearchMemoryAction(query_text="dee deployment window?", top_k=5))
    answer = env.execute_action(
        AnswerAction(
            query_text="dee deployment window?",
            retrieved_memory_ids=[dee_id, ada_id],
        )
    )

    # Top-1 fallback engaged: the answer is the best memory's content,
    # cited_memory_ids contains exactly that memory.
    assert answer.payload["cited_memory_ids"] == [dee_id]
    assert answer.payload["answer"] == "account-dee reported deployment window 1 at 09:00 UTC."


def test_llm_compose_passes_through_abstention(tmp_path) -> None:
    """If the LLM abstains, that's a valid answer with empty citations."""

    composed = {
        "answer": "I do not know from indexed memory.",
        "cited_memory_ids": [],
    }
    client = MockLLMClient(responses=[composed])
    env = _build_env(tmp_path, answer_llm_client=client)
    dee_id, ada_id = _seed_two_memories(env)

    env.execute_action(SearchMemoryAction(query_text="dee deployment window?", top_k=5))
    answer = env.execute_action(
        AnswerAction(
            query_text="dee deployment window?",
            retrieved_memory_ids=[dee_id, ada_id],
        )
    )

    assert answer.payload["answer"] == "I do not know from indexed memory."
    assert answer.payload["cited_memory_ids"] == []


def test_answer_returns_abstention_when_no_memories_retrieved(tmp_path) -> None:
    """An empty retrieval set is still abstention; the LLM is not consulted."""

    client = MockLLMClient()  # would raise on a non-policy request
    env = _build_env(tmp_path, answer_llm_client=client)

    answer = env.execute_action(
        AnswerAction(query_text="Anything?", retrieved_memory_ids=[])
    )

    assert answer.payload["answer"] == "I do not know from indexed memory."
    assert answer.payload["cited_memory_ids"] == []
    assert client.calls == []
