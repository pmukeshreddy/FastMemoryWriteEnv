from __future__ import annotations

import pytest
from pydantic import ValidationError

from fast_memory_write_env.dataset import generate_episode
from fast_memory_write_env.schemas import DatasetMode, EventCategory, EventPriority
from fast_memory_write_env.state import MemoryWriteQueue, MemoryWriteQueueItem


def _event(category: EventCategory):
    episode = generate_episode(DatasetMode.SMALL, seed=73, episode_index=0)
    return next(
        item.event
        for item in episode.stream
        if item.item_type == "event" and item.event.category == category
    )


def test_memory_write_queue_prioritizes_urgent_events() -> None:
    useful = _event(EventCategory.USEFUL_FACT).model_copy(update={"priority": EventPriority.NORMAL})
    urgent = _event(EventCategory.URGENT_FACT).model_copy(update={"priority": EventPriority.URGENT})
    queue = MemoryWriteQueue()

    queue.enqueue(event=useful, enqueued_at_ms=useful.timestamp_ms + 5)
    queue.enqueue(event=urgent, enqueued_at_ms=urgent.timestamp_ms + 5)

    assert queue.pending_event_ids() == [useful.event_id, urgent.event_id]
    assert queue.pop_next().event.event_id == urgent.event_id
    assert queue.pop_next().event.event_id == useful.event_id
    assert len(queue) == 0


def test_memory_write_queue_rejects_pre_arrival_enqueue_time() -> None:
    event = _event(EventCategory.USEFUL_FACT)

    with pytest.raises(ValidationError):
        MemoryWriteQueueItem(
            queue_item_id=f"mwq-{event.event_id}",
            event=event,
            enqueued_at_ms=event.timestamp_ms - 1,
        )
