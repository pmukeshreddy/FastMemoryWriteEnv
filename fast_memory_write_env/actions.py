"""Validated memory-write action schemas."""

from __future__ import annotations

from enum import Enum
from typing import Annotated, Any, Literal

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
