"""LLM client abstractions for memory-write policies."""

from __future__ import annotations

import json
import os
import random
import re
import time
import urllib.error
import urllib.request
from typing import Any, Literal, Protocol

from pydantic import Field

from fast_memory_write_env.schemas import StrictBaseModel


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

    def complete(
        self,
        messages: list[LLMMessage],
        *,
        temperature: float = 0.0,
        response_format: dict[str, Any] | None = None,
    ) -> LLMResponse:
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
        max_retries: int = 12,
        retry_initial_seconds: float = 1.0,
        retry_max_seconds: float = 90.0,
    ) -> None:
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise LLMClientError("OPENAI_API_KEY is required for OpenAICompatibleLLMClient")
        self.base_url = (base_url or os.getenv("OPENAI_BASE_URL") or "https://api.openai.com/v1").rstrip("/")
        self.model = model or os.getenv("OPENAI_MODEL") or "gpt-4o-mini"
        self.timeout_seconds = timeout_seconds
        # OpenAI returns 429 (and sometimes 5xx) under load. Production SDKs
        # all retry with exponential backoff and respect the Retry-After
        # header. Without retries, a single rate-limit window kills a long
        # run; with them, the client coasts through TPM bursts.
        if max_retries < 0:
            raise ValueError("max_retries must be >= 0")
        self.max_retries = max_retries
        self.retry_initial_seconds = retry_initial_seconds
        self.retry_max_seconds = retry_max_seconds

    def complete(
        self,
        messages: list[LLMMessage],
        *,
        temperature: float = 0.0,
        response_format: dict[str, Any] | None = None,
    ) -> LLMResponse:
        payload = {
            "model": self.model,
            "messages": [message.model_dump() for message in messages],
            "temperature": temperature,
            "response_format": response_format or {"type": "json_object"},
        }
        body_bytes = json.dumps(payload).encode("utf-8")
        last_error: Exception | None = None
        for attempt in range(self.max_retries + 1):
            request = urllib.request.Request(
                f"{self.base_url}/chat/completions",
                data=body_bytes,
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                method="POST",
            )
            try:
                with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                    response_payload = json.loads(response.read().decode("utf-8"))
                break
            except urllib.error.HTTPError as exc:
                err_body = exc.read().decode("utf-8", errors="replace")
                if exc.code in {429, 500, 502, 503, 504} and attempt < self.max_retries:
                    sleep_seconds = self._retry_delay_seconds(exc, attempt, err_body=err_body)
                    time.sleep(sleep_seconds)
                    last_error = exc
                    continue
                raise LLMClientError(
                    f"OpenAI-compatible request failed: {exc.code} {err_body}"
                ) from exc
            except (urllib.error.URLError, TimeoutError) as exc:
                if attempt < self.max_retries:
                    time.sleep(self._retry_delay_seconds(None, attempt, err_body=None))
                    last_error = exc
                    continue
                raise LLMClientError(f"OpenAI-compatible request failed: {exc}") from exc
            except json.JSONDecodeError as exc:
                raise LLMClientError(f"OpenAI-compatible request failed: {exc}") from exc
        else:  # pragma: no cover - loop fell through without break or raise
            raise LLMClientError(
                f"OpenAI-compatible request failed after {self.max_retries + 1} attempts: {last_error}"
            )

        try:
            message = response_payload["choices"][0]["message"]
            refusal = message.get("refusal")
            if refusal:
                raise LLMClientError(f"OpenAI-compatible response refused: {refusal}")
            content = message["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise LLMClientError("OpenAI-compatible response did not contain message content") from exc

        parsed_json = _loads_json_or_none(content)
        return LLMResponse(
            content=content,
            parsed_json=parsed_json,
            model=str(response_payload.get("model") or self.model),
            raw=response_payload,
        )

    def _retry_delay_seconds(
        self,
        exc: urllib.error.HTTPError | None,
        attempt: int,
        *,
        err_body: str | None = None,
    ) -> float:
        """Wait time honouring server hints, with full-jitter exponential fallback.

        OpenAI's TPM 429s do not always set ``Retry-After``. They reliably set
        ``x-ratelimit-reset-tokens`` (e.g. ``"660ms"``, ``"1.5s"``) and put the
        same hint in the JSON body (``"Please try again in 660ms"``). Honouring
        these is the difference between a single retry succeeding and a thrash
        loop blowing through the retry budget.
        """

        server_hint_seconds = _server_retry_hint_seconds(exc, err_body)
        if server_hint_seconds is not None:
            # Add proportional jitter so concurrent workers do not retry in
            # lockstep and re-hit the bucket together (thundering herd).
            jitter = random.uniform(0.0, max(0.25, server_hint_seconds * 0.5))
            return min(self.retry_max_seconds, max(0.1, server_hint_seconds + jitter))
        backoff = self.retry_initial_seconds * (2 ** attempt)
        # Full jitter: pick uniformly in [0, backoff] then cap.
        return min(self.retry_max_seconds, random.uniform(0.0, backoff))

    def complete_json(
        self,
        messages: list[LLMMessage],
        *,
        temperature: float = 0.0,
        response_format: dict[str, Any] | None = None,
    ) -> Any:
        """Return parsed JSON from an OpenAI-compatible chat completion."""

        response = self.complete(messages, temperature=temperature, response_format=response_format)
        if response.parsed_json is None:
            raise LLMClientError("OpenAI-compatible response content was not valid JSON")
        return response.parsed_json


_DURATION_RE = re.compile(r"^\s*([\d.]+)\s*(ms|s|m)?\s*$", re.IGNORECASE)
_BODY_RETRY_HINT_RE = re.compile(
    r"try again in\s*([\d.]+)\s*(ms|s|m)\b", re.IGNORECASE
)


def _parse_duration_seconds(raw: str | None) -> float | None:
    """Parse OpenAI rate-limit duration strings like ``"660ms"`` / ``"1.5s"`` / ``"30"``."""

    if raw is None:
        return None
    match = _DURATION_RE.match(raw)
    if not match:
        return None
    try:
        value = float(match.group(1))
    except ValueError:
        return None
    unit = (match.group(2) or "s").lower()
    if unit == "ms":
        return value / 1000.0
    if unit == "s":
        return value
    if unit == "m":
        return value * 60.0
    return None


def _server_retry_hint_seconds(
    exc: urllib.error.HTTPError | None,
    err_body: str | None,
) -> float | None:
    """Return the server's preferred wait, in seconds, if it gave one."""

    if exc is not None and exc.headers is not None:
        for header in (
            "Retry-After",
            "retry-after",
            "x-ratelimit-reset-tokens",
            "x-ratelimit-reset-requests",
        ):
            value = exc.headers.get(header)
            parsed = _parse_duration_seconds(value)
            if parsed is not None and parsed >= 0:
                return parsed
    if err_body:
        match = _BODY_RETRY_HINT_RE.search(err_body)
        if match:
            parsed = _parse_duration_seconds(f"{match.group(1)}{match.group(2)}")
            if parsed is not None and parsed >= 0:
                return parsed
    return None


class MockLLMClient:
    """Deterministic LLM client for policy and compose tests."""

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
                return LLMResponse(content=response, parsed_json=_loads_json_or_none(response), model="mock")
            return LLMResponse(content=json.dumps(response), parsed_json=response, model="mock")

        # No queued response: route by request shape so the same MockLLMClient
        # serves LLMMemoryWritePolicy, the env's answer-compose call, and the
        # answer-correctness judge in tests. Each branch is a fixed
        # deterministic response, not a fallback inside production code.
        request = _try_extract_request_json(messages)
        if request is not None and "new_event" in request:
            plan = _default_mock_plan_from_request(request)
            return LLMResponse(content=json.dumps(plan), parsed_json=plan, model="mock")
        if request is not None and "candidate_memories" in request:
            compose = _default_mock_compose_from_request(request)
            return LLMResponse(content=json.dumps(compose), parsed_json=compose, model="mock")
        if _is_judge_request(messages):
            # Tests that need the judge to disagree must queue "NO" explicitly.
            return LLMResponse(content="YES", parsed_json=None, model="mock")
        raise LLMClientError("MockLLMClient could not classify request shape")


def _default_mock_plan(messages: list[LLMMessage]) -> dict[str, Any]:
    return _default_mock_plan_from_request(_extract_request_json(messages))


def _default_mock_plan_from_request(request: dict[str, Any]) -> dict[str, Any]:
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


def _try_extract_request_json(messages: list[LLMMessage]) -> dict[str, Any] | None:
    """Return the latest JSON-shaped user message, or ``None`` if there is none."""

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
    """Recognise the answer-correctness judge by its system prompt."""

    for message in messages:
        if message.role == "system" and "strict grader" in message.content.lower():
            return True
    return False


def _default_mock_compose_from_request(request: dict[str, Any]) -> dict[str, Any]:
    """Deterministic compose response for tests: cite the top-ranked memory.

    The env passes ``candidate_memories`` ordered by retrieval rank, so picking
    the first one mirrors a focused single-memory answer. This is not a
    fallback inside the env — it is the mock client's deterministic behavior
    for the compose request shape.
    """

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
