"""Memory-write policies for Phase 3."""

from __future__ import annotations

import json
from typing import Any, Protocol

from pydantic import ValidationError

from fast_memory_write_env.actions import MemoryAction, validate_memory_actions
from fast_memory_write_env.env import deterministic_memory_id
from fast_memory_write_env.index import estimate_tokens
from fast_memory_write_env.llm_client import LLMClient, LLMClientError, LLMMessage
from fast_memory_write_env.schemas import (
    EventCategory,
    MemoryRecord,
    MemoryWriteBudget,
    RawEvent,
)


POLICY_SAFE_METADATA_KEYS = {
    "dataset",
    "dataset_format",
    "role",
    "session_date",
    "session_id",
    "session_index",
    "source_dataset",
    "speaker",
    "turn_index",
}


class MemoryWritePolicy(Protocol):
    """Policy interface for proposing memory-write actions."""

    def decide(
        self,
        *,
        new_event: RawEvent,
        active_memories: list[MemoryRecord],
        recent_events: list[RawEvent],
        latency_budget_ms: int,
        storage_budget_tokens_remaining: int,
        indexing_budget_operations_remaining: int,
    ) -> list[MemoryAction]:
        """Return proposed actions. The environment executes them."""


class LLMMemoryWritePolicy:
    """Main LLM memory-write policy.

    This class proposes validated actions only. It does not mutate stores and
    does not call retrieval backends.
    """

    def __init__(
        self,
        *,
        llm_client: LLMClient,
        max_retries: int = 2,
        temperature: float = 0.0,
    ) -> None:
        self.llm_client = llm_client
        self.max_retries = max_retries
        self.temperature = temperature

    def decide(
        self,
        *,
        new_event: RawEvent,
        active_memories: list[MemoryRecord],
        recent_events: list[RawEvent],
        latency_budget_ms: int,
        storage_budget_tokens_remaining: int,
        indexing_budget_operations_remaining: int,
    ) -> list[MemoryAction]:
        budget = MemoryWriteBudget(
            latency_budget_ms=latency_budget_ms,
            storage_budget_tokens_remaining=storage_budget_tokens_remaining,
            indexing_budget_operations_remaining=indexing_budget_operations_remaining,
        )
        messages = self._build_messages(new_event, active_memories, recent_events, budget)
        last_error: Exception | None = None
        last_content = ""

        for attempt in range(self.max_retries + 1):
            response = self.llm_client.complete(messages, temperature=self.temperature)
            last_content = response.content
            try:
                payload = response.parsed_json if response.parsed_json is not None else _parse_json_payload(last_content)
                return _strip_llm_hidden_fact_ids(_validate_action_payload(payload))
            except (json.JSONDecodeError, ValidationError, ValueError) as exc:
                last_error = exc
                if attempt >= self.max_retries:
                    break
                messages = [
                    *messages,
                    LLMMessage(role="assistant", content=last_content or "<empty response>"),
                    LLMMessage(
                        role="user",
                        content=(
                            "Repair the previous response. Return JSON only with this shape: "
                            '{"actions": [validated memory action objects]}. '
                            f"Validation error: {exc}"
                        ),
                    ),
                ]

        raise LLMClientError(f"LLM did not return valid memory actions after repair: {last_error}")

    def _build_messages(
        self,
        new_event: RawEvent,
        active_memories: list[MemoryRecord],
        recent_events: list[RawEvent],
        budget: MemoryWriteBudget,
    ) -> list[LLMMessage]:
        system = (
            "You are LLMMemoryWritePolicy for FastMemoryWriteEnv. "
            "Decide what memory actions to propose for the new raw event under the budgets. "
            "Return JSON only. Do not call tools, stores, or indexes. "
            "Allowed action_type values: write_memory, update_memory, mark_stale, "
            "ignore_event, compress_memory, index_now, delay_index."
        )
        user_payload = {
            "new_event": policy_visible_event(new_event),
            "active_memories": [policy_visible_memory(memory) for memory in active_memories],
            "recent_events": [policy_visible_event(event) for event in recent_events],
            "budgets": budget.model_dump(mode="json"),
            "response_contract": {
                "shape": {"actions": ["memory action objects"]},
                "notes": [
                    "Use write_memory for durable new facts.",
                    "Use update_memory for corrections to existing active memories.",
                    "Use mark_stale when old facts are superseded.",
                    "Use ignore_event for noise or low-value duplicates.",
                    "Use index_now or action index_immediately only when indexing budget allows.",
                    "Use delay_index when memory should exist but not be indexed yet.",
                    "Do not include fact_ids; evaluator-only fact labels are hidden and will be ignored.",
                ],
            },
        }
        return [
            LLMMessage(role="system", content=system),
            LLMMessage(role="user", content=json.dumps(user_payload, sort_keys=True)),
        ]


class NoMemoryBaseline:
    """Minimal lower-bound baseline: store no durable memories."""

    def decide(
        self,
        *,
        new_event: RawEvent,
        active_memories: list[MemoryRecord],
        recent_events: list[RawEvent],
        latency_budget_ms: int,
        storage_budget_tokens_remaining: int,
        indexing_budget_operations_remaining: int,
    ) -> list[MemoryAction]:
        return validate_memory_actions(
            [
                {
                    "action_type": "ignore_event",
                    "event_id": new_event.event_id,
                    "reason": "NoMemoryBaseline ignores all events.",
                }
            ]
        )


class StoreEverythingBaseline:
    """Minimal upper-bound baseline: write every event as memory."""

    def decide(
        self,
        *,
        new_event: RawEvent,
        active_memories: list[MemoryRecord],
        recent_events: list[RawEvent],
        latency_budget_ms: int,
        storage_budget_tokens_remaining: int,
        indexing_budget_operations_remaining: int,
    ) -> list[MemoryAction]:
        return validate_memory_actions(
            [
                {
                    "action_type": "write_memory",
                    "memory_id": deterministic_memory_id([new_event.event_id], new_event.content),
                    "entity_id": new_event.entity_id,
                    "content": new_event.content,
                    "source_event_ids": [new_event.event_id],
                    "fact_ids": [fact.fact_id for fact in new_event.facts],
                    "importance": 3,
                    "index_immediately": indexing_budget_operations_remaining > 0,
                    "metadata": {"baseline": "store_everything"},
                }
            ]
        )


class OraclePolicy:
    """Minimal oracle baseline using dataset category labels."""

    def decide(
        self,
        *,
        new_event: RawEvent,
        active_memories: list[MemoryRecord],
        recent_events: list[RawEvent],
        latency_budget_ms: int,
        storage_budget_tokens_remaining: int,
        indexing_budget_operations_remaining: int,
    ) -> list[MemoryAction]:
        if new_event.category in {EventCategory.NOISE, EventCategory.DUPLICATE}:
            return NoMemoryBaseline().decide(
                new_event=new_event,
                active_memories=active_memories,
                recent_events=recent_events,
                latency_budget_ms=latency_budget_ms,
                storage_budget_tokens_remaining=storage_budget_tokens_remaining,
                indexing_budget_operations_remaining=indexing_budget_operations_remaining,
            )

        matching_memory = next(
            (memory for memory in active_memories if memory.entity_id == new_event.entity_id),
            None,
        )
        if matching_memory and new_event.category in {
            EventCategory.CONTRADICTION,
            EventCategory.STALE_UPDATE,
        }:
            return validate_memory_actions(
                [
                    {
                        "action_type": "update_memory",
                        "memory_id": matching_memory.memory_id,
                        "content": new_event.content,
                        "source_event_ids": [new_event.event_id],
                        "fact_ids": [fact.fact_id for fact in new_event.facts],
                        "reason": "OraclePolicy applies labeled correction/update.",
                        "index_immediately": indexing_budget_operations_remaining > 0,
                        "metadata": {"baseline": "oracle"},
                    }
                ]
            )

        return validate_memory_actions(
            [
                {
                    "action_type": "write_memory",
                    "memory_id": deterministic_memory_id([new_event.event_id], new_event.content),
                    "entity_id": new_event.entity_id,
                    "content": new_event.content,
                    "source_event_ids": [new_event.event_id],
                    "fact_ids": [fact.fact_id for fact in new_event.facts],
                    "importance": 5 if new_event.category == EventCategory.URGENT_FACT else 3,
                    "index_immediately": indexing_budget_operations_remaining > 0,
                    "metadata": {"baseline": "oracle"},
                }
            ]
        )


def _parse_json_payload(content: str) -> Any:
    stripped = content.strip()
    if stripped.startswith("```"):
        stripped = _strip_fenced_json(stripped)
    return json.loads(stripped)


def _strip_fenced_json(content: str) -> str:
    lines = content.splitlines()
    if lines and lines[0].startswith("```"):
        lines = lines[1:]
    if lines and lines[-1].startswith("```"):
        lines = lines[:-1]
    return "\n".join(lines).strip()


def _validate_action_payload(payload: Any) -> list[MemoryAction]:
    if isinstance(payload, dict) and "actions" in payload:
        actions = payload["actions"]
    else:
        actions = payload
    return validate_memory_actions(actions)


def policy_visible_event(event: RawEvent) -> dict[str, Any]:
    """Return the event payload visible to the memory-write policy."""

    return {
        "event_id": event.event_id,
        "episode_id": event.episode_id,
        "timestamp_ms": event.timestamp_ms,
        "source": event.source,
        "user_id": event.user_id,
        "entity_id": event.entity_id,
        "content": event.content,
        "estimated_tokens": event.estimated_tokens,
        "metadata": _safe_policy_metadata(event.metadata),
    }


def policy_visible_memory(memory: MemoryRecord) -> dict[str, Any]:
    """Return the active-memory payload visible to the memory-write policy."""

    return {
        "memory_id": memory.memory_id,
        "entity_id": memory.entity_id,
        "content": memory.content,
        "source_event_ids": list(memory.source_event_ids),
        "created_at_ms": memory.created_at_ms,
        "updated_at_ms": memory.updated_at_ms,
        "status": memory.status.value,
        "indexed": memory.indexed,
        "estimated_tokens": memory.estimated_tokens,
        "metadata": _safe_policy_metadata(memory.metadata),
    }


def _strip_llm_hidden_fact_ids(actions: list[MemoryAction]) -> list[MemoryAction]:
    stripped: list[MemoryAction] = []
    for action in actions:
        if hasattr(action, "fact_ids"):
            stripped.append(action.model_copy(update={"fact_ids": []}))
        else:
            stripped.append(action)
    return stripped


def _safe_policy_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    safe: dict[str, Any] = {}
    for key, value in metadata.items():
        if key not in POLICY_SAFE_METADATA_KEYS:
            continue
        if isinstance(value, (str, int, float, bool)) or value is None:
            safe[key] = value
    return safe


def build_memory_context(
    active_memories: list[MemoryRecord],
    *,
    max_storage_tokens: int,
) -> list[MemoryRecord]:
    """Trim active memory context to a storage-token budget."""

    selected: list[MemoryRecord] = []
    used = 0
    for memory in sorted(active_memories, key=lambda item: item.updated_at_ms, reverse=True):
        tokens = memory.estimated_tokens or estimate_tokens(memory.content)
        if used + tokens > max_storage_tokens:
            continue
        selected.append(memory)
        used += tokens
    return selected
