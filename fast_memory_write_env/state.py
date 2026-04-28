"""Lightweight runtime state schemas for Phase 1."""

from __future__ import annotations

from pydantic import Field, model_validator

from fast_memory_write_env.schemas import (
    ID_PATTERN,
    MemoryRecord,
    MemoryWriteBudget,
    RawEvent,
    StrictBaseModel,
)


class EpisodeCursor(StrictBaseModel):
    """Cursor into a streaming episode."""

    episode_id: str = Field(pattern=ID_PATTERN)
    stream_index: int = Field(default=0, ge=0)
    current_time_ms: int = Field(default=0, ge=0)
    completed: bool = False


class MemoryWriteState(StrictBaseModel):
    """State presented around a memory-write decision."""

    current_time_ms: int = Field(ge=0)
    active_memories: list[MemoryRecord] = Field(default_factory=list)
    recent_events: list[RawEvent] = Field(default_factory=list)
    pending_event_ids: list[str] = Field(default_factory=list)
    budget: MemoryWriteBudget

    @model_validator(mode="after")
    def validate_recent_events_not_future(self) -> MemoryWriteState:
        for event in self.recent_events:
            if event.timestamp_ms > self.current_time_ms:
                raise ValueError("recent_events cannot occur after current_time_ms")
        return self
