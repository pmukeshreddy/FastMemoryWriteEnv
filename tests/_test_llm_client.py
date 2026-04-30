"""Test-only deterministic LLM client.

This module is named with a leading underscore so pytest does not collect it
as a test module, and it lives inside ``tests/`` so it is structurally
unimportable from production code paths in ``fast_memory_write_env``. It
exists for fast, deterministic unit tests of the env / evaluator / metrics
plumbing without burning real API tokens.

This is the only mock-shaped LLM client in the repo. Every production code
path uses :class:`fast_memory_write_env.llm_client.OpenAICompatibleLLMClient`
or a caller-supplied real client; there is no ``--mock`` flag and the
package no longer exports a mock client. If you find yourself importing
this from anywhere outside ``tests/`` you have a bug.
"""

from __future__ import annotations

import json
from typing import Any

from fast_memory_write_env.llm_client import (
    LLMClientError,
    LLMMessage,
    LLMResponse,
)


class DeterministicTestLLMClient:
    """Deterministic LLM client for evaluator/metrics/env unit tests.

    Behaves like a queue of responses when one is provided, and otherwise
    routes by request shape so the same instance can serve as the policy
    LLM, the answer-compose LLM, and the judge LLM in a single test - each
    branch is a fixed deterministic response, never a fallback inside
    production code.
    """

    def __init__(self, responses: list[str | dict[str, Any] | list[Any]] | None = None) -> None:
        self._responses = list(responses or [])
        self.calls: list[list[LLMMessage]] = []
        self.response_formats: list[dict[str, Any] | None] = []

    def complete(
        self,
        messages: list[LLMMessage],
        *,
        temperature: float = 0.0,
        response_format: dict[str, Any] | None = None,
    ) -> LLMResponse:
        self.calls.append(messages)
        self.response_formats.append(response_format)
        if self._responses:
            response = self._responses.pop(0)
            if isinstance(response, str):
                return LLMResponse(
                    content=response,
                    parsed_json=_loads_json_or_none(response),
                    model="test",
                )
            return LLMResponse(content=json.dumps(response), parsed_json=response, model="test")

        request = _try_extract_request_json(messages)
        if request is not None and "new_event" in request:
            plan = _default_plan_from_request(request)
            return LLMResponse(content=json.dumps(plan), parsed_json=plan, model="test")
        if request is not None and "candidate_memories" in request:
            compose = _default_compose_from_request(request)
            return LLMResponse(content=json.dumps(compose), parsed_json=compose, model="test")
        if _is_judge_request(messages):
            return LLMResponse(content="YES", parsed_json=None, model="test")
        raise LLMClientError("DeterministicTestLLMClient could not classify request shape")


def _default_plan_from_request(request: dict[str, Any]) -> dict[str, Any]:
    event = request["new_event"]
    active_memories = list(request.get("active_memories", []))
    indexing_budget = int(request.get("budgets", {}).get("indexing_budget_operations_remaining", 0))
    content = str(event.get("content", ""))
    event_id = str(event.get("event_id", "event"))
    entity_id = str(event.get("entity_id", "entity"))

    if _looks_low_value_or_duplicate(content):
        return {
            "actions": [
                {
                    "action_type": "ignore_event",
                    "event_id": event_id,
                    "reason": "Event appears low-value or duplicate from visible content.",
                }
            ]
        }

    if _looks_like_update(content) and active_memories:
        memory_id = str(active_memories[0]["memory_id"])
        return {
            "actions": [
                {
                    "action_type": "update_memory",
                    "memory_id": memory_id,
                    "content": content,
                    "source_event_ids": [event_id],
                    "reason": "Visible content appears to correct or supersede prior memory.",
                    "index_immediately": indexing_budget > 0,
                }
            ]
        }

    return {
        "actions": [
            {
                "action_type": "write_memory",
                "entity_id": entity_id,
                "content": content,
                "source_event_ids": [event_id],
                "importance": 5 if "urgent" in content.lower() else 3,
                "index_immediately": indexing_budget > 0,
            }
        ]
    }


def _try_extract_request_json(messages: list[LLMMessage]) -> dict[str, Any] | None:
    for message in reversed(messages):
        if message.role != "user":
            continue
        try:
            payload = json.loads(message.content)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            return payload
    return None


def _is_judge_request(messages: list[LLMMessage]) -> bool:
    for message in messages:
        if message.role == "system" and "strict grader" in message.content.lower():
            return True
    return False


def _default_compose_from_request(request: dict[str, Any]) -> dict[str, Any]:
    candidates = request.get("candidate_memories", []) or []
    if not candidates:
        return {"answer": "I do not know from indexed memory.", "cited_memory_ids": []}
    top = candidates[0]
    memory_id = str(top.get("memory_id", ""))
    content = str(top.get("content", ""))
    if not memory_id or not content:
        return {"answer": "I do not know from indexed memory.", "cited_memory_ids": []}
    return {"answer": content, "cited_memory_ids": [memory_id]}


def _looks_low_value_or_duplicate(content: str) -> bool:
    normalized = content.lower()
    markers = [
        "duplicate note",
        "repeated",
        "opened the dashboard",
        "thank-you message",
        "heartbeat check",
        "without changes",
        "unrelated",
    ]
    return any(marker in normalized for marker in markers)


def _looks_like_update(content: str) -> bool:
    normalized = content.lower()
    markers = ["correction:", "instead of", "stale", "supersede", "supersedes", "now prefers"]
    return any(marker in normalized for marker in markers)


def _loads_json_or_none(content: str) -> Any | None:
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        return None
