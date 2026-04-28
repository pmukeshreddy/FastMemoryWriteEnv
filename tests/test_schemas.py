from __future__ import annotations

import pytest
from pydantic import ValidationError

from fast_memory_write_env.schemas import (
    DatasetMode,
    EventCategory,
    EventFact,
    Query,
    QueryGold,
    RawEvent,
    StreamEventItem,
    StreamQueryItem,
    StreamingEpisode,
)


def _event(timestamp_ms: int = 100) -> RawEvent:
    fact = EventFact(
        fact_id="event-001-fact-001",
        entity_id="account-ada",
        attribute="contact_preference",
        value="sms",
        source_event_id="event-001",
        valid_from_ms=timestamp_ms,
    )
    return RawEvent(
        event_id="event-001",
        episode_id="ep-small-0",
        timestamp_ms=timestamp_ms,
        source="support_chat",
        user_id="user-alpha",
        entity_id="account-ada",
        category=EventCategory.USEFUL_FACT,
        content="Account Ada prefers SMS.",
        facts=[fact],
    )


def _query(timestamp_ms: int = 200) -> Query:
    return Query(
        query_id="query-001",
        episode_id="ep-small-0",
        timestamp_ms=timestamp_ms,
        user_id="user-alpha",
        target_entity_id="account-ada",
        text="How should Ada be contacted?",
        gold=QueryGold(
            required_fact_ids=["event-001-fact-001"],
            supporting_event_ids=["event-001"],
            answer_facts=["Account Ada prefers SMS."],
        ),
    )


def test_episode_accepts_interleaved_event_and_query_stream() -> None:
    event = _event(timestamp_ms=100)
    query = _query(timestamp_ms=200)

    episode = StreamingEpisode(
        episode_id="ep-small-0",
        mode=DatasetMode.SMALL,
        seed=7,
        stream=[
            StreamEventItem(timestamp_ms=100, event=event),
            StreamQueryItem(timestamp_ms=200, query=query),
        ],
    )

    assert episode.stream[0].item_type == "event"
    assert episode.stream[1].item_type == "query"


def test_raw_event_rejects_invalid_id_and_negative_timestamp() -> None:
    with pytest.raises(ValidationError):
        RawEvent(
            event_id="bad id",
            episode_id="ep-small-0",
            timestamp_ms=-1,
            source="support_chat",
            user_id="user-alpha",
            entity_id="account-ada",
            category=EventCategory.NOISE,
            content="Invalid event.",
        )


def test_episode_rejects_non_monotonic_stream() -> None:
    event = _event(timestamp_ms=300)
    query = _query(timestamp_ms=200)

    with pytest.raises(ValidationError):
        StreamingEpisode(
            episode_id="ep-small-0",
            mode=DatasetMode.SMALL,
            seed=7,
            stream=[
                StreamEventItem(timestamp_ms=300, event=event),
                StreamQueryItem(timestamp_ms=200, query=query),
            ],
        )


def test_stream_item_rejects_mismatched_embedded_timestamp() -> None:
    with pytest.raises(ValidationError):
        StreamEventItem(timestamp_ms=999, event=_event(timestamp_ms=100))


def test_query_gold_requires_facts_and_evidence() -> None:
    with pytest.raises(ValidationError):
        QueryGold(required_fact_ids=[], supporting_event_ids=[], answer_facts=[])
