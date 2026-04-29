"""Validated memory-write action schemas.

This module exposes two layers:

* ``PolicyAction``: what the LLM policy is allowed to propose. New memories are
  proposed without a ``memory_id`` (the environment owns ID generation), and
  references to existing memories must point at active memories. This layer is
  what the structured-output JSON schema and the policy repair loop validate
  against.
* ``MemoryAction`` / ``EnvironmentAction``: typed actions that the environment
  executes. Memory IDs are required and assumed to be valid for the store.

``compile_policy_actions`` validates a plan of policy proposals against the
current set of active memory IDs and converts them into executable
environment actions, generating stable IDs for newly written or compressed
memories. ``validate_action_plan`` is the underlying plan validator and is
also called inside ``LLMMemoryWritePolicy.decide`` so plan errors flow through
the same repair loop as schema errors.
"""

from __future__ import annotations

import hashlib
from enum import Enum
from typing import Annotated, Any, Iterable, Literal

from pydantic import Field, TypeAdapter, model_validator

from fast_memory_write_env.schemas import ID_PATTERN, RawEvent, StrictBaseModel


class ActionType(str, Enum):
    """Supported memory-write action types."""

    STORE_RAW = "store_raw"
    WRITE_MEMORY = "write_memory"
    UPDATE_MEMORY = "update_memory"
    MARK_STALE = "mark_stale"
    IGNORE_EVENT = "ignore_event"
    COMPRESS_MEMORY = "compress_memory"
    INDEX_NOW = "index_now"
    DELAY_INDEX = "delay_index"
    SEARCH_MEMORY = "search_memory"
    ANSWER = "answer"


class StoreRawAction(StrictBaseModel):
    """Persist a raw event before memory-write processing."""

    action_type: Literal["store_raw"] = "store_raw"
    event: RawEvent


class WriteMemoryAction(StrictBaseModel):
    """Create a new memory from one or more raw events."""

    action_type: Literal["write_memory"] = "write_memory"
    memory_id: str = Field(pattern=ID_PATTERN)
    entity_id: str = Field(pattern=ID_PATTERN)
    content: str = Field(min_length=1)
    source_event_ids: list[str] = Field(min_length=1)
    fact_ids: list[str] = Field(default_factory=list)
    importance: int = Field(default=3, ge=1, le=5)
    index_immediately: bool = False
    metadata: dict[str, Any] = Field(default_factory=dict)


class UpdateMemoryAction(StrictBaseModel):
    """Update an existing memory with new or corrected information."""

    action_type: Literal["update_memory"] = "update_memory"
    memory_id: str = Field(pattern=ID_PATTERN)
    content: str = Field(min_length=1)
    source_event_ids: list[str] = Field(min_length=1)
    fact_ids: list[str] = Field(default_factory=list)
    reason: str = Field(min_length=1)
    index_immediately: bool = False
    metadata: dict[str, Any] = Field(default_factory=dict)


class MarkStaleAction(StrictBaseModel):
    """Mark an existing memory stale because newer evidence supersedes it."""

    action_type: Literal["mark_stale"] = "mark_stale"
    memory_id: str = Field(pattern=ID_PATTERN)
    reason: str = Field(min_length=1)
    source_event_id: str | None = Field(default=None, pattern=ID_PATTERN)
    superseded_by_memory_id: str | None = Field(default=None, pattern=ID_PATTERN)
    metadata: dict[str, Any] = Field(default_factory=dict)


class IgnoreEventAction(StrictBaseModel):
    """Ignore a raw event after validating that it has low memory value."""

    action_type: Literal["ignore_event"] = "ignore_event"
    event_id: str = Field(pattern=ID_PATTERN)
    reason: str = Field(min_length=1)
    metadata: dict[str, Any] = Field(default_factory=dict)


class CompressMemoryAction(StrictBaseModel):
    """Compress several memories into a denser target memory."""

    action_type: Literal["compress_memory"] = "compress_memory"
    source_memory_ids: list[str] = Field(min_length=2)
    target_memory_id: str = Field(pattern=ID_PATTERN)
    compressed_content: str = Field(min_length=1)
    source_event_ids: list[str] = Field(default_factory=list)
    fact_ids: list[str] = Field(default_factory=list)
    index_immediately: bool = False
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_target_not_source(self) -> CompressMemoryAction:
        if self.target_memory_id in self.source_memory_ids:
            raise ValueError("target_memory_id cannot also be a source memory")
        return self


class IndexNowAction(StrictBaseModel):
    """Index a memory immediately."""

    action_type: Literal["index_now"] = "index_now"
    memory_id: str = Field(pattern=ID_PATTERN)
    reason: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class DelayIndexAction(StrictBaseModel):
    """Delay indexing a memory until more budget is available."""

    action_type: Literal["delay_index"] = "delay_index"
    memory_id: str = Field(pattern=ID_PATTERN)
    retry_after_ms: int = Field(ge=0)
    reason: str = Field(min_length=1)
    metadata: dict[str, Any] = Field(default_factory=dict)


class SearchMemoryAction(StrictBaseModel):
    """Search indexed memories for a query."""

    action_type: Literal["search_memory"] = "search_memory"
    query_text: str = Field(min_length=1)
    query_id: str | None = Field(default=None, pattern=ID_PATTERN)
    top_k: int = Field(default=5, ge=1)
    filters: dict[str, Any] = Field(default_factory=dict)
    as_of_ms: float | None = Field(default=None, ge=0.0)


class AnswerAction(StrictBaseModel):
    """Produce a deterministic answer from retrieved memories."""

    action_type: Literal["answer"] = "answer"
    query_text: str = Field(min_length=1)
    query_id: str | None = Field(default=None, pattern=ID_PATTERN)
    retrieved_memory_ids: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


MemoryAction = Annotated[
    WriteMemoryAction
    | UpdateMemoryAction
    | MarkStaleAction
    | IgnoreEventAction
    | CompressMemoryAction
    | IndexNowAction
    | DelayIndexAction,
    Field(discriminator="action_type"),
]

MEMORY_ACTION_ADAPTER: TypeAdapter[MemoryAction] = TypeAdapter(MemoryAction)


EnvironmentAction = Annotated[
    StoreRawAction
    | WriteMemoryAction
    | UpdateMemoryAction
    | MarkStaleAction
    | IgnoreEventAction
    | CompressMemoryAction
    | IndexNowAction
    | DelayIndexAction
    | SearchMemoryAction
    | AnswerAction,
    Field(discriminator="action_type"),
]

ENVIRONMENT_ACTION_ADAPTER: TypeAdapter[EnvironmentAction] = TypeAdapter(EnvironmentAction)


def validate_memory_action(data: object) -> MemoryAction:
    """Validate an untrusted action payload into a typed action model."""

    return MEMORY_ACTION_ADAPTER.validate_python(data)


def validate_memory_actions(data: object) -> list[MemoryAction]:
    """Validate a list of untrusted action payloads."""

    return TypeAdapter(list[MemoryAction]).validate_python(data)


def validate_environment_action(data: object) -> EnvironmentAction:
    """Validate an untrusted environment action payload."""

    return ENVIRONMENT_ACTION_ADAPTER.validate_python(data)


def validate_environment_actions(data: object) -> list[EnvironmentAction]:
    """Validate a list of untrusted environment action payloads."""

    return TypeAdapter(list[EnvironmentAction]).validate_python(data)


# ---------------------------------------------------------------------------
# Policy-proposal layer (LLM-facing schemas)
# ---------------------------------------------------------------------------


class WriteMemoryProposal(StrictBaseModel):
    """Policy proposal to create a new memory.

    The environment owns memory ID generation, so proposals must not include
    ``memory_id``. The compile step assigns a stable ID before execution.
    """

    action_type: Literal["write_memory"] = "write_memory"
    entity_id: str = Field(pattern=ID_PATTERN)
    content: str = Field(min_length=1)
    source_event_ids: list[str] = Field(min_length=1)
    importance: int = Field(default=3, ge=1, le=5)
    index_immediately: bool = False
    metadata: dict[str, Any] = Field(default_factory=dict)


class CompressMemoryProposal(StrictBaseModel):
    """Policy proposal to compress two or more existing memories.

    The environment generates the target memory ID; ``source_memory_ids`` must
    reference active memories. Set ``index_immediately`` to merge and index in
    one step when the indexing budget allows.
    """

    action_type: Literal["compress_memory"] = "compress_memory"
    source_memory_ids: list[str] = Field(min_length=2)
    compressed_content: str = Field(min_length=1)
    index_immediately: bool = False
    metadata: dict[str, Any] = Field(default_factory=dict)


PolicyAction = Annotated[
    WriteMemoryProposal
    | UpdateMemoryAction
    | MarkStaleAction
    | IgnoreEventAction
    | CompressMemoryProposal
    | IndexNowAction
    | DelayIndexAction,
    Field(discriminator="action_type"),
]

POLICY_ACTION_ADAPTER: TypeAdapter[PolicyAction] = TypeAdapter(PolicyAction)


def validate_policy_action(data: object) -> PolicyAction:
    """Validate an untrusted policy proposal into a typed proposal model."""

    return POLICY_ACTION_ADAPTER.validate_python(data)


def validate_policy_actions(data: object) -> list[PolicyAction]:
    """Validate a list of untrusted policy proposals."""

    return TypeAdapter(list[PolicyAction]).validate_python(data)


class PolicyPlanError(ValueError):
    """Raised when a batch of policy proposals violates plan-level constraints."""


def validate_action_plan(
    proposals: Iterable[PolicyAction],
    *,
    active_memory_ids: Iterable[str],
) -> None:
    """Reject plans that the environment cannot safely execute.

    Rules enforced here:

    * ``update_memory``, ``mark_stale``, ``index_now``, ``delay_index`` may
      only target memory IDs that already exist as active memories.
    * ``compress_memory`` source memory IDs must all exist as active memories.
    * The same memory cannot be both updated and marked stale in one plan.
    * The same memory cannot be both newly written and referenced by an
      existing-memory action in one plan (the LLM cannot know env-generated
      IDs in advance, so this is symptomatic of a malformed plan).
    """

    active_ids = set(active_memory_ids)
    update_ids: set[str] = set()
    stale_ids: set[str] = set()
    seen_indexed_or_delayed: set[str] = set()
    duplicate_targets: list[str] = []

    for proposal in proposals:
        if isinstance(proposal, UpdateMemoryAction):
            if proposal.memory_id not in active_ids:
                raise PolicyPlanError(
                    f"update_memory references unknown memory_id {proposal.memory_id!r}; "
                    f"allowed active memory IDs: {sorted(active_ids)}"
                )
            if proposal.memory_id in stale_ids:
                raise PolicyPlanError(
                    f"memory_id {proposal.memory_id!r} cannot be updated and marked stale in one plan"
                )
            if proposal.memory_id in update_ids:
                duplicate_targets.append(proposal.memory_id)
            update_ids.add(proposal.memory_id)
        elif isinstance(proposal, MarkStaleAction):
            if proposal.memory_id not in active_ids:
                raise PolicyPlanError(
                    f"mark_stale references unknown memory_id {proposal.memory_id!r}; "
                    f"mark_stale only applies to existing active memories"
                )
            if proposal.memory_id in update_ids:
                raise PolicyPlanError(
                    f"memory_id {proposal.memory_id!r} cannot be updated and marked stale in one plan"
                )
            stale_ids.add(proposal.memory_id)
        elif isinstance(proposal, IndexNowAction):
            if proposal.memory_id not in active_ids:
                raise PolicyPlanError(
                    f"index_now references unknown memory_id {proposal.memory_id!r}; "
                    f"use write_memory with index_immediately=true for new memories"
                )
            seen_indexed_or_delayed.add(proposal.memory_id)
        elif isinstance(proposal, DelayIndexAction):
            if proposal.memory_id not in active_ids:
                raise PolicyPlanError(
                    f"delay_index references unknown memory_id {proposal.memory_id!r}"
                )
            seen_indexed_or_delayed.add(proposal.memory_id)
        elif isinstance(proposal, CompressMemoryProposal):
            for source_id in proposal.source_memory_ids:
                if source_id not in active_ids:
                    raise PolicyPlanError(
                        f"compress_memory source memory_id {source_id!r} is not active"
                    )

    if duplicate_targets:
        raise PolicyPlanError(
            f"duplicate update_memory entries for memory_ids: {sorted(set(duplicate_targets))}"
        )


def compile_policy_actions(
    proposals: Iterable[PolicyAction],
    *,
    active_memory_ids: Iterable[str],
    id_prefix: str = "mem",
) -> list[MemoryAction]:
    """Validate a plan and convert proposals into executable environment actions.

    Generates stable ``memory_id`` / ``target_memory_id`` values for new
    memories so the LLM never authors them. Existing-memory references are
    passed through unchanged once the plan validator accepts them.
    """

    proposals = list(proposals)
    active_set = set(active_memory_ids)
    validate_action_plan(proposals, active_memory_ids=active_set)

    used_ids: set[str] = set(active_set)
    executables: list[MemoryAction] = []
    for index, proposal in enumerate(proposals):
        if isinstance(proposal, WriteMemoryProposal):
            memory_id = _allocate_memory_id(
                seeds=[*sorted(proposal.source_event_ids), proposal.content],
                used_ids=used_ids,
                prefix=id_prefix,
            )
            used_ids.add(memory_id)
            executables.append(
                WriteMemoryAction(
                    memory_id=memory_id,
                    entity_id=proposal.entity_id,
                    content=proposal.content,
                    source_event_ids=list(proposal.source_event_ids),
                    importance=proposal.importance,
                    index_immediately=proposal.index_immediately,
                    metadata=dict(proposal.metadata),
                )
            )
        elif isinstance(proposal, CompressMemoryProposal):
            target_id = _allocate_memory_id(
                seeds=[
                    *sorted(proposal.source_memory_ids),
                    proposal.compressed_content,
                    f"compress-{index}",
                ],
                used_ids=used_ids,
                prefix=f"{id_prefix}-compressed",
            )
            used_ids.add(target_id)
            executables.append(
                CompressMemoryAction(
                    source_memory_ids=list(proposal.source_memory_ids),
                    target_memory_id=target_id,
                    compressed_content=proposal.compressed_content,
                    index_immediately=proposal.index_immediately,
                    metadata=dict(proposal.metadata),
                )
            )
        else:
            executables.append(proposal)
    return executables


def deterministic_memory_id(
    source_event_ids: list[str],
    content: str,
    *,
    prefix: str = "mem",
) -> str:
    """Create a stable memory ID from source event IDs and content."""

    seed = "|".join([*sorted(source_event_ids), content])
    digest = hashlib.sha1(seed.encode("utf-8")).hexdigest()[:12]
    return f"{prefix}-{digest}"


def _allocate_memory_id(
    *,
    seeds: list[str],
    used_ids: set[str],
    prefix: str,
) -> str:
    base = "|".join(seeds)
    digest = hashlib.sha1(base.encode("utf-8")).hexdigest()[:12]
    base_id = f"{prefix}-{digest}"
    if base_id not in used_ids:
        return base_id
    suffix = 1
    while True:
        candidate = f"{base_id}-{suffix}"
        if candidate not in used_ids:
            return candidate
        suffix += 1


class ActionExecutionResult(StrictBaseModel):
    """Structured result returned after the environment executes an action."""

    success: bool
    action_type: ActionType
    latency_ms: float = Field(ge=0.0)
    storage_tokens_delta: int
    error: str | None = None
    payload: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_error_matches_success(self) -> ActionExecutionResult:
        if self.success and self.error is not None:
            raise ValueError("successful action results must not include error")
        if not self.success and not self.error:
            raise ValueError("failed action results must include error")
        return self


ActionResult = ActionExecutionResult
