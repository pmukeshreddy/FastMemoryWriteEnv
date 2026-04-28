"""Core Pydantic schemas for streaming memory-write episodes."""

from __future__ import annotations

from enum import Enum
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


ID_PATTERN = r"^[A-Za-z0-9][A-Za-z0-9_.:\-]*$"


class StrictBaseModel(BaseModel):
    """Base model with strict validation defaults for project schemas."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)


class DatasetMode(str, Enum):
    """Supported dataset sizes."""

    SMALL = "small"
    MEDIUM = "medium"
    LONG = "long"


class StreamItemType(str, Enum):
    """Kinds of items that can appear in a streaming episode."""

    EVENT = "event"
    QUERY = "query"


class EventCategory(str, Enum):
    """Categories used to exercise memory-write decisions."""

    USEFUL_FACT = "useful_fact"
    NOISE = "noise"
    DUPLICATE = "duplicate"
    CONTRADICTION = "contradiction"
    STALE_UPDATE = "stale_update"
    URGENT_FACT = "urgent_fact"


class EventPriority(str, Enum):
    """Priority signal attached to raw events."""

    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    URGENT = "urgent"


class MemoryStatus(str, Enum):
    """Lifecycle status for memory records."""

    ACTIVE = "active"
    STALE = "stale"
    COMPRESSED = "compressed"


class EventFact(StrictBaseModel):
    """A structured fact expressed by a raw event."""

    fact_id: str = Field(pattern=ID_PATTERN)
    entity_id: str = Field(pattern=ID_PATTERN)
    attribute: str = Field(min_length=1)
    value: str = Field(min_length=1)
    source_event_id: str = Field(pattern=ID_PATTERN)
    valid_from_ms: int = Field(ge=0)
    valid_to_ms: int | None = Field(default=None, ge=0)
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_validity_window(self) -> EventFact:
        if self.valid_to_ms is not None and self.valid_to_ms < self.valid_from_ms:
            raise ValueError("valid_to_ms must be greater than or equal to valid_from_ms")
        return self


class RawEvent(StrictBaseModel):
    """Incoming stream event before memory-write policy processing."""

    event_id: str = Field(pattern=ID_PATTERN)
    episode_id: str = Field(pattern=ID_PATTERN)
    timestamp_ms: int = Field(ge=0)
    source: str = Field(min_length=1)
    user_id: str = Field(pattern=ID_PATTERN)
    entity_id: str = Field(pattern=ID_PATTERN)
    category: EventCategory
    content: str = Field(min_length=1)
    facts: list[EventFact] = Field(default_factory=list)
    priority: EventPriority = EventPriority.NORMAL
    duplicate_of: str | None = Field(default=None, pattern=ID_PATTERN)
    contradicts: list[str] = Field(default_factory=list)
    supersedes: list[str] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)
    estimated_tokens: int = Field(default=0, ge=0)
    metadata: dict[str, Any] = Field(default_factory=dict)


class QueryGold(StrictBaseModel):
    """Gold facts and evidence expected to support an answer."""

    required_fact_ids: list[str] = Field(min_length=1)
    supporting_event_ids: list[str] = Field(min_length=1)
    answer_facts: list[str] = Field(min_length=1)
    stale_fact_ids: list[str] = Field(default_factory=list)
    notes: str | None = None


class Query(StrictBaseModel):
    """A future information need that arrives while streaming continues."""

    query_id: str = Field(pattern=ID_PATTERN)
    episode_id: str = Field(pattern=ID_PATTERN)
    timestamp_ms: int = Field(ge=0)
    user_id: str = Field(pattern=ID_PATTERN)
    target_entity_id: str = Field(pattern=ID_PATTERN)
    text: str = Field(min_length=1)
    gold: QueryGold
    metadata: dict[str, Any] = Field(default_factory=dict)


class StreamEventItem(StrictBaseModel):
    """An event item in the chronological stream."""

    item_type: Literal["event"] = "event"
    timestamp_ms: int = Field(ge=0)
    event: RawEvent

    @model_validator(mode="after")
    def validate_timestamp_matches_event(self) -> StreamEventItem:
        if self.timestamp_ms != self.event.timestamp_ms:
            raise ValueError("stream item timestamp_ms must match event timestamp_ms")
        return self


class StreamQueryItem(StrictBaseModel):
    """A query item in the chronological stream."""

    item_type: Literal["query"] = "query"
    timestamp_ms: int = Field(ge=0)
    query: Query

    @model_validator(mode="after")
    def validate_timestamp_matches_query(self) -> StreamQueryItem:
        if self.timestamp_ms != self.query.timestamp_ms:
            raise ValueError("stream item timestamp_ms must match query timestamp_ms")
        return self


StreamItem = Annotated[StreamEventItem | StreamQueryItem, Field(discriminator="item_type")]


class StreamingEpisode(StrictBaseModel):
    """A single chronological stream of events and queries."""

    episode_id: str = Field(pattern=ID_PATTERN)
    mode: DatasetMode
    seed: int
    stream: list[StreamItem] = Field(min_length=1)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_stream(self) -> StreamingEpisode:
        previous_timestamp = -1
        event_ids: set[str] = set()
        query_ids: set[str] = set()

        for item in self.stream:
            if item.timestamp_ms < previous_timestamp:
                raise ValueError("stream timestamps must be monotonically nondecreasing")
            previous_timestamp = item.timestamp_ms

            if item.item_type == "event":
                event = item.event
                if event.episode_id != self.episode_id:
                    raise ValueError("event episode_id must match episode episode_id")
                if event.event_id in event_ids:
                    raise ValueError(f"duplicate event_id: {event.event_id}")
                event_ids.add(event.event_id)
            else:
                query = item.query
                if query.episode_id != self.episode_id:
                    raise ValueError("query episode_id must match episode episode_id")
                if query.query_id in query_ids:
                    raise ValueError(f"duplicate query_id: {query.query_id}")
                query_ids.add(query.query_id)

        return self


class GeneratedDataset(StrictBaseModel):
    """A generated dataset containing one or more streaming episodes."""

    dataset_id: str = Field(pattern=ID_PATTERN)
    mode: DatasetMode
    seed: int
    episodes: list[StreamingEpisode] = Field(min_length=1)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_episodes(self) -> GeneratedDataset:
        episode_ids: set[str] = set()
        for episode in self.episodes:
            if episode.mode != self.mode:
                raise ValueError("episode mode must match dataset mode")
            if episode.episode_id in episode_ids:
                raise ValueError(f"duplicate episode_id: {episode.episode_id}")
            episode_ids.add(episode.episode_id)
        return self


class MemoryWriteBudget(StrictBaseModel):
    """Budgets presented to a future memory-write policy."""

    latency_budget_ms: int = Field(gt=0)
    storage_budget_tokens_remaining: int = Field(ge=0)
    indexing_budget_operations_remaining: int = Field(ge=0)
    max_actions: int = Field(default=8, ge=1)


class MemoryRecord(StrictBaseModel):
    """A memory record shape used by later phases."""

    memory_id: str = Field(pattern=ID_PATTERN)
    entity_id: str = Field(pattern=ID_PATTERN)
    content: str = Field(min_length=1)
    source_event_ids: list[str] = Field(default_factory=list)
    fact_ids: list[str] = Field(default_factory=list)
    created_at_ms: int = Field(ge=0)
    updated_at_ms: int = Field(ge=0)
    status: MemoryStatus = MemoryStatus.ACTIVE
    indexed: bool = False
    estimated_tokens: int = Field(default=0, ge=0)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_update_time(self) -> MemoryRecord:
        if self.updated_at_ms < self.created_at_ms:
            raise ValueError("updated_at_ms must be greater than or equal to created_at_ms")
        return self
