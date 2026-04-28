"""LLM client abstractions for memory-write policies."""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from typing import Any, Literal, Protocol

from pydantic import Field

from fast_memory_write_env.schemas import EventCategory, RawEvent, StrictBaseModel


class LLMClientError(RuntimeError):
    """Raised when an LLM client cannot produce a usable response."""


class LLMMessage(StrictBaseModel):
    """One chat message sent to an LLM."""

    role: Literal["system", "user", "assistant"]
    content: str = Field(min_length=1)


class LLMResponse(StrictBaseModel):
    """Raw LLM response with optional parsed JSON content."""

    content: str
    parsed_json: Any | None = None
    model: str | None = None
    raw: dict[str, Any] = Field(default_factory=dict)


class LLMClient(Protocol):
    """Internal abstraction used by memory-write policies."""

    def complete(self, messages: list[LLMMessage], *, temperature: float = 0.0) -> LLMResponse:
        """Return one chat completion."""


class OpenAICompatibleLLMClient:
    """Minimal OpenAI-compatible chat completions client.

    The client uses the standard library so Phase 3 does not need a provider
    package. It expects the response content to be JSON and exposes
    ``complete_json`` for callers that want provider-level JSON parsing.
    """

    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        model: str | None = None,
        timeout_seconds: float = 30.0,
    ) -> None:
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise LLMClientError("OPENAI_API_KEY is required for OpenAICompatibleLLMClient")
        self.base_url = (base_url or os.getenv("OPENAI_BASE_URL") or "https://api.openai.com/v1").rstrip("/")
        self.model = model or os.getenv("OPENAI_MODEL") or "gpt-4o-mini"
        self.timeout_seconds = timeout_seconds

    def complete(self, messages: list[LLMMessage], *, temperature: float = 0.0) -> LLMResponse:
        payload = {
            "model": self.model,
            "messages": [message.model_dump() for message in messages],
            "temperature": temperature,
            "response_format": {"type": "json_object"},
        }
        request = urllib.request.Request(
            f"{self.base_url}/chat/completions",
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
            raise LLMClientError(f"OpenAI-compatible request failed: {exc.code} {body}") from exc
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            raise LLMClientError(f"OpenAI-compatible request failed: {exc}") from exc

        try:
            content = response_payload["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise LLMClientError("OpenAI-compatible response did not contain message content") from exc

        parsed_json = _loads_json_or_none(content)
        return LLMResponse(
            content=content,
            parsed_json=parsed_json,
            model=str(response_payload.get("model") or self.model),
            raw=response_payload,
        )

    def complete_json(self, messages: list[LLMMessage], *, temperature: float = 0.0) -> Any:
        """Return parsed JSON from an OpenAI-compatible chat completion."""

        response = self.complete(messages, temperature=temperature)
        if response.parsed_json is None:
            raise LLMClientError("OpenAI-compatible response content was not valid JSON")
        return response.parsed_json


class MockLLMClient:
    """Deterministic LLM client for policy tests."""

    def __init__(self, responses: list[str | dict[str, Any] | list[Any]] | None = None) -> None:
        self._responses = list(responses or [])
        self.calls: list[list[LLMMessage]] = []

    def complete(self, messages: list[LLMMessage], *, temperature: float = 0.0) -> LLMResponse:
        self.calls.append(messages)
        if self._responses:
            response = self._responses.pop(0)
            if isinstance(response, str):
                return LLMResponse(content=response, parsed_json=_loads_json_or_none(response), model="mock")
            return LLMResponse(content=json.dumps(response), parsed_json=response, model="mock")

        plan = _default_mock_plan(messages)
        return LLMResponse(content=json.dumps(plan), parsed_json=plan, model="mock")


def _default_mock_plan(messages: list[LLMMessage]) -> dict[str, Any]:
    request = _extract_request_json(messages)
    event = RawEvent.model_validate(request["new_event"])
    active_memories = list(request.get("active_memories", []))
    indexing_budget = int(request.get("budgets", {}).get("indexing_budget_operations_remaining", 0))

    if event.category in {EventCategory.NOISE, EventCategory.DUPLICATE}:
        return {
            "actions": [
                {
                    "action_type": "ignore_event",
                    "event_id": event.event_id,
                    "reason": f"{event.category.value} is not worth a durable memory.",
                }
            ]
        }

    fact_ids = [fact.fact_id for fact in event.facts]
    if event.category in {EventCategory.CONTRADICTION, EventCategory.STALE_UPDATE} and active_memories:
        memory_id = str(active_memories[0]["memory_id"])
        return {
            "actions": [
                {
                    "action_type": "update_memory",
                    "memory_id": memory_id,
                    "content": event.content,
                    "source_event_ids": [event.event_id],
                    "fact_ids": fact_ids,
                    "reason": "New event corrects or supersedes prior memory.",
                    "index_immediately": indexing_budget > 0,
                }
            ]
        }

    return {
        "actions": [
            {
                "action_type": "write_memory",
                "memory_id": f"mem-{event.event_id}",
                "entity_id": event.entity_id,
                "content": event.content,
                "source_event_ids": [event.event_id],
                "fact_ids": fact_ids,
                "importance": 5 if event.category == EventCategory.URGENT_FACT else 3,
                "index_immediately": indexing_budget > 0,
            }
        ]
    }


def _extract_request_json(messages: list[LLMMessage]) -> dict[str, Any]:
    for message in reversed(messages):
        if message.role != "user":
            continue
        try:
            payload = json.loads(message.content)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict) and "new_event" in payload:
            return payload
    raise LLMClientError("MockLLMClient could not find policy request JSON")


def _loads_json_or_none(content: str) -> Any | None:
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        return None
