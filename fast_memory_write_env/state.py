"""Lightweight runtime state schemas for Phase 1."""

from __future__ import annotations

import threading
import time

from pydantic import Field, model_validator

from fast_memory_write_env.schemas import (
    ID_PATTERN,
    EventPriority,
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


QUEUE_PRIORITY = {
    EventPriority.URGENT: 0,
    EventPriority.HIGH: 1,
    EventPriority.NORMAL: 2,
    EventPriority.LOW: 3,
}


class MemoryWriteQueueItem(StrictBaseModel):
    """One raw event waiting for memory-write policy processing."""

    queue_item_id: str = Field(pattern=ID_PATTERN)
    event: RawEvent
    enqueued_at_ms: int = Field(ge=0)
    attempts: int = Field(default=0, ge=0)
    priority: int = Field(default=2, ge=0)

    @model_validator(mode="after")
    def validate_enqueued_after_event_arrival(self) -> MemoryWriteQueueItem:
        if self.enqueued_at_ms < self.event.timestamp_ms:
            raise ValueError("enqueued_at_ms cannot be before event timestamp")
        return self


class MemoryWriteQueue:
    """Priority queue for raw events waiting to become memory actions."""

    def __init__(self) -> None:
        self._items: list[MemoryWriteQueueItem] = []
        self._active_count = 0
        self._closed = False
        self._condition = threading.Condition()

    def enqueue(self, *, event: RawEvent, enqueued_at_ms: int) -> MemoryWriteQueueItem:
        item = MemoryWriteQueueItem(
            queue_item_id=f"mwq-{event.event_id}",
            event=event,
            enqueued_at_ms=enqueued_at_ms,
            priority=QUEUE_PRIORITY[event.priority],
        )
        with self._condition:
            if self._closed:
                raise RuntimeError("cannot enqueue into a closed memory-write queue")
            self._items.append(item)
            self._condition.notify()
        return item

    def pop_next(self) -> MemoryWriteQueueItem:
        with self._condition:
            if not self._items:
                raise IndexError("memory-write queue is empty")
            item = self._pop_next_unlocked()
            self._condition.notify_all()
            return item

    def get_next(self, timeout_seconds: float | None = None) -> MemoryWriteQueueItem | None:
        """Wait for and return the next queued event, or None when closed and empty."""

        with self._condition:
            while not self._items and not self._closed:
                self._condition.wait(timeout=timeout_seconds)
                if timeout_seconds is not None and not self._items:
                    return None
            if not self._items:
                return None
            item = self._pop_next_unlocked()
            self._active_count += 1
            self._condition.notify_all()
            return item

    def task_done(self) -> None:
        with self._condition:
            if self._active_count <= 0:
                raise RuntimeError("memory-write queue task_done called too many times")
            self._active_count -= 1
            self._condition.notify_all()

    def close(self) -> None:
        with self._condition:
            self._closed = True
            self._condition.notify_all()

    def is_closed_and_drained(self) -> bool:
        """Return true only after no future or active work can remain."""

        with self._condition:
            return self._closed and not self._items and self._active_count == 0

    def wait_until_idle(self, timeout_seconds: float | None = None) -> bool:
        """Return true when all queued and active work has completed."""

        deadline = None if timeout_seconds is None else time.monotonic() + timeout_seconds
        with self._condition:
            while self._items or self._active_count:
                if deadline is None:
                    self._condition.wait()
                    continue
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return False
                self._condition.wait(timeout=remaining)
            return True

    def pending_event_ids(self) -> list[str]:
        with self._condition:
            return [item.event.event_id for item in self._items]

    def __len__(self) -> int:
        with self._condition:
            return len(self._items)

    def _pop_next_unlocked(self) -> MemoryWriteQueueItem:
        index, item = min(
            enumerate(self._items),
            key=lambda pair: (
                pair[1].priority,
                pair[1].enqueued_at_ms,
                pair[1].event.timestamp_ms,
                pair[1].queue_item_id,
            ),
        )
        del self._items[index]
        return item
