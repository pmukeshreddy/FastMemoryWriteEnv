"""Memory-write policies for Phase 3."""

from __future__ import annotations

import hashlib
import json
from typing import Any, Protocol

from pydantic import ValidationError

from fast_memory_write_env.actions import (
    PolicyAction,
    PolicyPlanError,
    validate_action_plan,
    validate_policy_actions,
)
from fast_memory_write_env.index import estimate_tokens
from fast_memory_write_env.llm_client import LLMClient, LLMClientError, LLMMessage
from fast_memory_write_env.schemas import (
    EventCategory,
    ID_PATTERN,
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


# Per-decision context caps. Without these the policy prompt grows
# unboundedly across a long stream (every active memory + every recent
# event re-serialized into every call), which is the dominant per-event
# latency cost on real LongMemEval workloads. The caps trade a small
# amount of long-tail context for ~3-5x faster decide latency.
#
# On real conversational data (LongMemEval) the previous tight caps
# (2 recent events, 8 active memories) starved the policy of the context
# it needed to recognize novel personal facts vs continuations and to
# update existing memories instead of duplicating or ignoring them.
POLICY_MAX_ACTIVE_MEMORIES = 24
POLICY_MAX_RECENT_EVENTS = 10

# Cap the per-decision action list. Production LongMemEval turns rarely
# need more than two or three coordinated actions per event. A hard upper
# bound also defends against the LLM emitting a long action list, hitting
# one validation error, and pushing the whole decide into a repair retry.
POLICY_MAX_ACTIONS_PER_DECISION = 4


def memory_action_response_format() -> dict[str, Any]:
    """OpenAI Structured Outputs schema for validated policy proposals."""

    id_string = {"type": "string", "pattern": ID_PATTERN}
    source_event_ids = {"type": "array", "items": id_string, "minItems": 1}
    source_memory_ids = {"type": "array", "items": id_string, "minItems": 2}
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "memory_write_actions",
            "strict": True,
            "schema": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "actions": {
                        "type": "array",
                        "minItems": 1,
                        "maxItems": POLICY_MAX_ACTIONS_PER_DECISION,
                        "items": {
                            "anyOf": [
                                _strict_action_schema(
                                    {
                                        "action_type": {"type": "string", "enum": ["write_memory"]},
                                        "entity_id": id_string,
                                        "content": {"type": "string"},
                                        "source_event_ids": source_event_ids,
                                        "importance": {"type": "integer", "minimum": 1, "maximum": 5},
                                        "index_immediately": {"type": "boolean"},
                                    }
                                ),
                                _strict_action_schema(
                                    {
                                        "action_type": {"type": "string", "enum": ["update_memory"]},
                                        "memory_id": {
                                            **id_string,
                                            "description": "Must reference an existing memory_id from active_memories.",
                                        },
                                        "content": {"type": "string"},
                                        "source_event_ids": source_event_ids,
                                        "reason": {"type": "string"},
                                        "index_immediately": {"type": "boolean"},
                                    }
                                ),
                                _strict_action_schema(
                                    {
                                        "action_type": {"type": "string", "enum": ["mark_stale"]},
                                        "memory_id": {
                                            **id_string,
                                            "description": "Must reference an existing active memory_id.",
                                        },
                                        "reason": {"type": "string"},
                                    }
                                ),
                                _strict_action_schema(
                                    {
                                        "action_type": {"type": "string", "enum": ["ignore_event"]},
                                        "event_id": id_string,
                                        "reason": {"type": "string"},
                                    }
                                ),
                                _strict_action_schema(
                                    {
                                        "action_type": {"type": "string", "enum": ["compress_memory"]},
                                        "source_memory_ids": source_memory_ids,
                                        "compressed_content": {"type": "string"},
                                        "index_immediately": {"type": "boolean"},
                                    }
                                ),
                                _strict_action_schema(
                                    {
                                        "action_type": {"type": "string", "enum": ["index_now"]},
                                        "memory_id": {
                                            **id_string,
                                            "description": "Must reference an existing active memory_id.",
                                        },
                                    }
                                ),
                                _strict_action_schema(
                                    {
                                        "action_type": {"type": "string", "enum": ["delay_index"]},
                                        "memory_id": {
                                            **id_string,
                                            "description": "Must reference an existing active memory_id.",
                                        },
                                        "retry_after_ms": {"type": "integer", "minimum": 0},
                                        "reason": {"type": "string"},
                                    }
                                ),
                            ]
                        },
                    }
                },
                "required": ["actions"],
            },
        },
    }


def _strict_action_schema(properties: dict[str, Any]) -> dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": properties,
        "required": list(properties),
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
    ) -> list[PolicyAction]:
        """Return validated policy proposals. The environment executes them."""


class LLMMemoryWritePolicy:
    """Main LLM memory-write policy.

    The policy proposes validated actions only. It does not mutate stores and
    does not call retrieval backends. Memory IDs for new memories are assigned
    by the environment; the LLM may only reference existing memory IDs from
    ``active_memories``.
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
    ) -> list[PolicyAction]:
        budget = MemoryWriteBudget(
            latency_budget_ms=latency_budget_ms,
            storage_budget_tokens_remaining=storage_budget_tokens_remaining,
            indexing_budget_operations_remaining=indexing_budget_operations_remaining,
        )
        active_memory_ids = [memory.memory_id for memory in active_memories]
        messages = self._build_messages(new_event, active_memories, recent_events, budget)
        last_error: Exception | None = None
        last_content = ""

        for attempt in range(self.max_retries + 1):
            response = self.llm_client.complete(
                messages,
                temperature=self.temperature,
                response_format=memory_action_response_format(),
            )
            last_content = response.content
            try:
                payload = response.parsed_json if response.parsed_json is not None else _parse_json_payload(last_content)
                proposals = _validate_action_payload(payload)
                validate_action_plan(proposals, active_memory_ids=active_memory_ids)
                return proposals
            except (json.JSONDecodeError, ValidationError, PolicyPlanError, ValueError) as exc:
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
                            "The active response_format JSON schema is strict; satisfy it exactly. "
                            "Do not include memory_id on write_memory or target_memory_id on compress_memory; "
                            "the environment generates them. update_memory, mark_stale, index_now, "
                            "delay_index, and compress_memory may only reference memory_ids that already "
                            f"appear in active_memories: {active_memory_ids}. "
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
        # Bound the per-decision prompt so the policy call stays fast even
        # late in a long stream. Most-recently-updated memories are the
        # ones the LLM is most likely to reference for update_memory /
        # mark_stale; older active memories remain in the store and become
        # visible again whenever they are touched.
        trimmed_active = sorted(
            active_memories,
            key=lambda memory: memory.updated_at_ms,
            reverse=True,
        )[:POLICY_MAX_ACTIVE_MEMORIES]
        trimmed_recent = list(recent_events)[-POLICY_MAX_RECENT_EVENTS:]
        system = (
            "You are LLMMemoryWritePolicy for FastMemoryWriteEnv, acting as a Personal "
            "Information Organizer. Your job is to build memory that lets a future query "
            "answer questions about this user/entity correctly. Decide what memory "
            "actions to propose for the new raw event under the budgets.\n"
            "\n"
            "PRIORITIZE USER-ASSERTED FACTS. The new_event.content is prefixed with "
            "the speaker role (e.g. \"user: ...\" or \"assistant: ...\"). Treat user "
            "messages as the primary source of facts about the user. Do NOT promote "
            "assistant suggestions, summaries, or speculation into memory unless the "
            "user has confirmed them. If the new event is an assistant message that "
            "merely restates or paraphrases the user, prefer ignore_event.\n"
            "\n"
            "WHAT TO REMEMBER (write_memory or update_memory):\n"
            "- Personal facts and self-disclosures: name, age, education (\"degree in X\"), "
            "  job/role, employer, location, family, relationships, health conditions, "
            "  identities, demographics.\n"
            "- Preferences and opinions the user states as theirs: likes, dislikes, "
            "  favorite X, hates Y, prefers Z over W.\n"
            "- Decisions, plans, goals, commitments: \"I'm going to...\", \"I just "
            "  bought...\", \"I'm starting...\", deadlines, scheduled events.\n"
            "- Specific named entities the user owns or interacts with: pet names, "
            "  product names, project names, places they go, people they know.\n"
            "- Numerical/temporal facts: dates, durations, prices, ages, scores.\n"
            "- Problems the user is solving and constraints they have.\n"
            "When in doubt about long-term value, prefer write_memory over ignore_event. "
            "Losing a fact the future query needs is far worse than storing an extra one. "
            "Always cite the new event's event_id in source_event_ids.\n"
            "\n"
            "WRITE ATOMIC FACTS. If the user's turn states multiple distinct facts, "
            "emit multiple write_memory actions (one per fact) rather than one long "
            "narrative memory. Atomic memories retrieve more reliably.\n"
            "\n"
            "WHAT TO IGNORE (ignore_event):\n"
            "- Pure acknowledgments, greetings, chit-chat with no information "
            "  (\"thanks\", \"ok\", \"sounds good\", \"how's it going\").\n"
            "- Generic assistant replies that just restate or paraphrase the user.\n"
            "- Boilerplate, marketing copy, content the user is reading aloud rather "
            "  than asserting.\n"
            "- Exact duplicates of an existing active_memory that need no update.\n"
            "\n"
            "PREFER update_memory OVER write_memory when an existing active memory "
            "already covers the same subject and the new event refines or supersedes "
            "it (correction, clarification, change of state). Use mark_stale only when "
            "an old memory is fully superseded by a different existing memory. Use "
            "compress_memory to merge several related delayed memories into one.\n"
            "\n"
            "EXAMPLES (illustrative; do not copy these into output):\n"
            "Example 1 - user self-disclosure with two facts:\n"
            "  new_event.content: \"user: I graduated with a degree in Business "
            "Administration last year and just started at a marketing firm in Seattle.\"\n"
            "  good actions: [\n"
            "    {\"action_type\": \"write_memory\", \"content\": \"User holds a degree "
            "in Business Administration (graduated last year).\", \"importance\": 4, ...},\n"
            "    {\"action_type\": \"write_memory\", \"content\": \"User works at a "
            "marketing firm in Seattle (recently started).\", \"importance\": 4, ...}\n"
            "  ]\n"
            "Example 2 - assistant boilerplate restatement:\n"
            "  new_event.content: \"assistant: Congrats on the new role! Marketing in "
            "Seattle sounds exciting. Let me know how I can help.\"\n"
            "  good actions: [{\"action_type\": \"ignore_event\", \"reason\": \"assistant "
            "restatement, no new user-asserted fact\"}]\n"
            "Example 3 - update on existing memory:\n"
            "  active_memories includes mem-abc123 with content \"User's dog is named Max\".\n"
            "  new_event.content: \"user: My dog's name is actually Maximus, not Max - I "
            "always use the full name.\"\n"
            "  good actions: [{\"action_type\": \"update_memory\", \"memory_id\": "
            "\"mem-abc123\", \"content\": \"User's dog is named Maximus (full name; not "
            "the nickname Max).\", \"reason\": \"name correction\", ...}]\n"
            "\n"
            "OUTPUT RULES:\n"
            "- Return JSON only. Do not call tools, stores, or indexes.\n"
            "- Allowed action_type values: write_memory, update_memory, mark_stale, "
            "  ignore_event, compress_memory, index_now, delay_index.\n"
            "- The environment owns memory IDs: never include memory_id on write_memory "
            "  or target_memory_id on compress_memory. Existing-memory actions may only "
            "  reference memory_ids that appear in active_memories.\n"
            f"- Emit at most {POLICY_MAX_ACTIONS_PER_DECISION} actions per response; "
            "prefer one or two focused actions over long action lists."
        )
        user_payload = {
            "new_event": policy_visible_event(new_event),
            "active_memories": [policy_visible_memory(memory) for memory in trimmed_active],
            "recent_events": [policy_visible_event(event) for event in trimmed_recent],
            "budgets": budget.model_dump(mode="json"),
            "response_contract": {
                "shape": {"actions": ["memory action objects"]},
                "required_fields_by_action_type": {
                    "write_memory": [
                        "action_type",
                        "entity_id",
                        "content",
                        "source_event_ids",
                    ],
                    "update_memory": [
                        "action_type",
                        "memory_id",
                        "content",
                        "source_event_ids",
                        "reason",
                    ],
                    "mark_stale": ["action_type", "memory_id", "reason"],
                    "ignore_event": ["action_type", "event_id", "reason"],
                    "compress_memory": [
                        "action_type",
                        "source_memory_ids",
                        "compressed_content",
                    ],
                    "index_now": ["action_type", "memory_id"],
                    "delay_index": ["action_type", "memory_id", "retry_after_ms", "reason"],
                },
                "notes": [
                    "Use write_memory for durable new facts; the environment assigns the memory_id.",
                    "Prefer update_memory over (update_memory + mark_stale) when correcting an existing memory; do not emit both for the same memory_id.",
                    "Use mark_stale only when an old memory is fully superseded by a different existing memory.",
                    "Use ignore_event for noise or low-value duplicates.",
                    "Indexing budget is only spent by write_memory(index_immediately=true), update_memory(index_immediately=true), compress_memory(index_immediately=true), and index_now. Plain update_memory does NOT auto-spend budget.",
                    "Hybrid retrieval is on: a memory is lexically searchable as soon as it is written, even before its vector is indexed. Set index_immediately only when semantic recall is needed urgently.",
                    "update_memory(index_immediately=false) on a previously indexed memory drops the stale vector and sets needs_reindex=true. Lexical search still serves the new content; call index_now later to restore semantic retrieval.",
                    "Memories with indexed=false and needs_reindex=true exist and answer lexical queries; only call index_now on them when indexing budget is available and semantic recall is required.",
                    "Under tight budgets, prioritize urgent or current facts over older low-priority ones; consider compress_memory to merge several delayed useful memories into one indexable summary.",
                    "Use delay_index when memory should exist but indexing must wait for budget; the memory remains queryable lexically and can be indexed later via index_now.",
                    "Memory IDs for write_memory and target IDs for compress_memory are environment-owned. Do not invent them.",
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
    ) -> list[PolicyAction]:
        return validate_policy_actions(
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
    ) -> list[PolicyAction]:
        return validate_policy_actions(
            [
                {
                    "action_type": "write_memory",
                    "entity_id": new_event.entity_id,
                    "content": new_event.content,
                    "source_event_ids": [new_event.event_id],
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
    ) -> list[PolicyAction]:
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
            return validate_policy_actions(
                [
                    {
                        "action_type": "update_memory",
                        "memory_id": matching_memory.memory_id,
                        "content": new_event.content,
                        "source_event_ids": [new_event.event_id],
                        "reason": "OraclePolicy applies labeled correction/update.",
                        "index_immediately": indexing_budget_operations_remaining > 0,
                        "metadata": {"baseline": "oracle"},
                    }
                ]
            )

        return validate_policy_actions(
            [
                {
                    "action_type": "write_memory",
                    "entity_id": new_event.entity_id,
                    "content": new_event.content,
                    "source_event_ids": [new_event.event_id],
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


def _validate_action_payload(payload: Any) -> list[PolicyAction]:
    if isinstance(payload, dict) and "actions" in payload:
        actions = payload["actions"]
    else:
        actions = payload
    return validate_policy_actions(actions)


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
        "needs_reindex": bool(memory.metadata.get("needs_reindex", False)),
        "estimated_tokens": memory.estimated_tokens,
        "metadata": _safe_policy_metadata(memory.metadata),
    }


def _safe_policy_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    safe: dict[str, Any] = {}
    for key, value in metadata.items():
        if key not in POLICY_SAFE_METADATA_KEYS:
            continue
        if isinstance(value, (str, int, float, bool)) or value is None:
            if key == "session_id" and isinstance(value, str):
                # Anonymise the upstream session_id before showing it to the
                # policy. LongMemEval names gold-evidence sessions
                # ``answer_<hash>`` while noise sessions use ``sharegpt``,
                # ``ultrachat``, or random hex prefixes. Exposing that name
                # would leak the gold-evidence label directly. Hash the id
                # deterministically so within-session correlation still works
                # but the prefix tell is gone.
                safe[key] = _anonymise_session_id(value)
            else:
                safe[key] = value
    return safe


def _anonymise_session_id(session_id: str) -> str:
    digest = hashlib.sha256(session_id.encode("utf-8")).hexdigest()[:12]
    return f"session-{digest}"


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
