"""Deterministic streaming dataset generation for Phase 1."""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any

from fast_memory_write_env.schemas import (
    DatasetMode,
    EventCategory,
    EventFact,
    EventPriority,
    GeneratedDataset,
    Query,
    QueryGold,
    RawEvent,
    StreamEventItem,
    StreamItem,
    StreamQueryItem,
    StreamingEpisode,
)


@dataclass(frozen=True)
class ModeConfig:
    episode_count: int
    background_cycles: int
    noise_events: int


MODE_CONFIGS: dict[DatasetMode, ModeConfig] = {
    DatasetMode.SMALL: ModeConfig(episode_count=2, background_cycles=1, noise_events=2),
    DatasetMode.MEDIUM: ModeConfig(episode_count=4, background_cycles=3, noise_events=5),
    DatasetMode.LONG: ModeConfig(episode_count=8, background_cycles=8, noise_events=12),
}
SYNTHETIC_DATASET_MODES = tuple(MODE_CONFIGS.keys())


USERS = ["user-alpha", "user-bravo", "user-charlie", "user-delta", "user-echo"]
ENTITIES = ["account-ada", "account-ben", "account-cy", "account-dee", "account-eli"]
SOURCES = ["support_chat", "billing_webhook", "ops_note", "calendar_sync"]


def generate_dataset(mode: DatasetMode | str = DatasetMode.SMALL, seed: int = 0) -> GeneratedDataset:
    """Generate a deterministic dataset with streaming episodes."""

    dataset_mode = DatasetMode(mode)
    if dataset_mode not in MODE_CONFIGS:
        raise ValueError(f"unsupported synthetic dataset mode: {dataset_mode.value}")
    config = MODE_CONFIGS[dataset_mode]
    episodes = [
        generate_episode(dataset_mode, seed=seed, episode_index=episode_index)
        for episode_index in range(config.episode_count)
    ]
    return GeneratedDataset(
        dataset_id=f"dataset-{dataset_mode.value}-{seed}",
        mode=dataset_mode,
        seed=seed,
        episodes=episodes,
        metadata={
            "episode_count": config.episode_count,
            "background_cycles": config.background_cycles,
            "noise_events": config.noise_events,
        },
    )


def generate_episode(
    mode: DatasetMode | str = DatasetMode.SMALL,
    seed: int = 0,
    episode_index: int = 0,
) -> StreamingEpisode:
    """Generate one deterministic streaming episode with interleaved queries."""

    dataset_mode = DatasetMode(mode)
    if dataset_mode not in MODE_CONFIGS:
        raise ValueError(f"unsupported synthetic dataset mode: {dataset_mode.value}")
    mode_offset = {DatasetMode.SMALL: 11, DatasetMode.MEDIUM: 23, DatasetMode.LONG: 37}[dataset_mode]
    rng = random.Random(seed * 1009 + episode_index * 97 + mode_offset)
    builder = _EpisodeBuilder(dataset_mode, seed, episode_index, rng)
    config = MODE_CONFIGS[dataset_mode]

    primary, secondary, urgent_entity, *remaining_entities = builder.entities
    requesting_user = builder.users[0]
    secondary_user = builder.users[1]

    initial_contact = builder.add_event(
        category=EventCategory.USEFUL_FACT,
        entity_id=primary,
        user_id=requesting_user,
        content=f"{primary} prefers email for renewal notices.",
        fact_specs=[("contact_preference", "email")],
        priority=EventPriority.NORMAL,
        tags=["profile", "contact"],
    )
    builder.add_noise_event()
    builder.add_event(
        category=EventCategory.DUPLICATE,
        entity_id=primary,
        user_id=secondary_user,
        content=f"Duplicate note: {primary} prefers email for renewal notices.",
        duplicate_of=initial_contact.event_id,
        priority=EventPriority.LOW,
        tags=["duplicate"],
    )

    urgent = builder.add_event(
        category=EventCategory.URGENT_FACT,
        entity_id=urgent_entity,
        user_id=requesting_user,
        content=f"Urgent: {urgent_entity} needs outage alerts routed to the on-call phone.",
        fact_specs=[("alert_route", "on-call phone")],
        priority=EventPriority.URGENT,
        tags=["urgent", "routing"],
    )
    builder.add_query(
        user_id=requesting_user,
        target_entity_id=urgent_entity,
        text=f"Where should outage alerts for {urgent_entity} go right now?",
        required_facts=urgent.facts,
        supporting_events=[urgent.event_id],
        answer_facts=[f"{urgent_entity} outage alerts should route to the on-call phone."],
    )

    contradiction = builder.add_event(
        category=EventCategory.CONTRADICTION,
        entity_id=primary,
        user_id=secondary_user,
        content=f"Correction: {primary} now prefers SMS instead of email for renewal notices.",
        fact_specs=[("contact_preference", "sms")],
        priority=EventPriority.HIGH,
        contradicts=[initial_contact.event_id],
        tags=["profile", "correction"],
    )
    stale_update = builder.add_event(
        category=EventCategory.STALE_UPDATE,
        entity_id=primary,
        user_id=requesting_user,
        content=f"Mark the old email preference for {primary} stale; SMS is current.",
        fact_specs=[("contact_preference_status", "email stale; sms current")],
        priority=EventPriority.HIGH,
        supersedes=[initial_contact.event_id],
        tags=["stale_update"],
    )
    builder.add_noise_event()
    builder.add_query(
        user_id=secondary_user,
        target_entity_id=primary,
        text=f"What is the current renewal contact method for {primary}?",
        required_facts=[*contradiction.facts, *stale_update.facts],
        supporting_events=[contradiction.event_id, stale_update.event_id],
        answer_facts=[f"{primary} should be contacted by SMS; the email preference is stale."],
        stale_fact_ids=[fact.fact_id for fact in initial_contact.facts],
    )

    far_fact = builder.add_event(
        category=EventCategory.USEFUL_FACT,
        entity_id=secondary,
        user_id=secondary_user,
        content=f"{secondary} has a contract renewal deadline on Friday.",
        fact_specs=[("renewal_deadline", "Friday")],
        priority=EventPriority.NORMAL,
        tags=["contract"],
    )

    for _ in range(config.noise_events):
        builder.add_noise_event()

    for cycle in range(config.background_cycles):
        entity_id = remaining_entities[cycle % len(remaining_entities)] if remaining_entities else primary
        user_id = builder.users[(cycle + 2) % len(builder.users)]
        useful = builder.add_event(
            category=EventCategory.USEFUL_FACT,
            entity_id=entity_id,
            user_id=user_id,
            content=f"{entity_id} reported deployment window {cycle + 1} at 09:00 UTC.",
            fact_specs=[(f"deployment_window_{cycle + 1}", "09:00 UTC")],
            priority=EventPriority.NORMAL,
            tags=["deployment"],
        )
        if cycle % 2 == 0:
            builder.add_noise_event()
        if cycle == 0 or dataset_mode != DatasetMode.SMALL:
            builder.add_query(
                user_id=user_id,
                target_entity_id=entity_id,
                text=f"What deployment window was reported for {entity_id}?",
                required_facts=useful.facts,
                supporting_events=[useful.event_id],
                answer_facts=[f"{entity_id} reported a deployment window at 09:00 UTC."],
            )

    builder.add_query(
        user_id=requesting_user,
        target_entity_id=secondary,
        text=f"What deadline matters for {secondary}?",
        required_facts=far_fact.facts,
        supporting_events=[far_fact.event_id],
        answer_facts=[f"{secondary} has a contract renewal deadline on Friday."],
    )
    builder.add_event(
        category=EventCategory.DUPLICATE,
        entity_id=urgent_entity,
        user_id=secondary_user,
        content=f"Repeated urgent routing note for {urgent_entity}: use the on-call phone.",
        duplicate_of=urgent.event_id,
        priority=EventPriority.LOW,
        tags=["duplicate", "urgent"],
    )
    builder.add_noise_event()

    return StreamingEpisode(
        episode_id=builder.episode_id,
        mode=dataset_mode,
        seed=seed,
        stream=builder.stream,
        metadata={
            "episode_index": episode_index,
            "generator": "phase_1_deterministic",
        },
    )


class _EpisodeBuilder:
    def __init__(self, mode: DatasetMode, seed: int, episode_index: int, rng: random.Random) -> None:
        self.mode = mode
        self.seed = seed
        self.episode_index = episode_index
        self.rng = rng
        self.episode_id = f"ep-{mode.value}-{episode_index}"
        self.timestamp_ms = 0
        self.event_counter = 0
        self.query_counter = 0
        self.stream: list[StreamItem] = []
        self.users = _rotated(USERS, rng)
        self.entities = _rotated(ENTITIES, rng)

    def add_event(
        self,
        *,
        category: EventCategory,
        entity_id: str,
        user_id: str,
        content: str,
        fact_specs: list[tuple[str, str]] | None = None,
        priority: EventPriority = EventPriority.NORMAL,
        duplicate_of: str | None = None,
        contradicts: list[str] | None = None,
        supersedes: list[str] | None = None,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> RawEvent:
        self._advance_time()
        self.event_counter += 1
        event_id = f"{self.episode_id}-event-{self.event_counter:03d}"
        facts = [
            EventFact(
                fact_id=f"{event_id}-fact-{fact_index:02d}",
                entity_id=entity_id,
                attribute=attribute,
                value=value,
                source_event_id=event_id,
                valid_from_ms=self.timestamp_ms,
            )
            for fact_index, (attribute, value) in enumerate(fact_specs or [], start=1)
        ]
        event = RawEvent(
            event_id=event_id,
            episode_id=self.episode_id,
            timestamp_ms=self.timestamp_ms,
            source=self.rng.choice(SOURCES),
            user_id=user_id,
            entity_id=entity_id,
            category=category,
            content=content,
            facts=facts,
            priority=priority,
            duplicate_of=duplicate_of,
            contradicts=contradicts or [],
            supersedes=supersedes or [],
            tags=tags or [],
            estimated_tokens=max(1, len(content.split())),
            metadata=metadata or {},
        )
        self.stream.append(StreamEventItem(timestamp_ms=event.timestamp_ms, event=event))
        return event

    def add_noise_event(self) -> RawEvent:
        entity_id = self.rng.choice(self.entities)
        user_id = self.rng.choice(self.users)
        noise_texts = [
            f"{entity_id} opened the dashboard and closed it without changes.",
            f"{user_id} sent a thank-you message about yesterday's call.",
            f"Heartbeat check for {entity_id} completed with no actionable change.",
            f"{entity_id} viewed a help article unrelated to account state.",
        ]
        return self.add_event(
            category=EventCategory.NOISE,
            entity_id=entity_id,
            user_id=user_id,
            content=self.rng.choice(noise_texts),
            priority=EventPriority.LOW,
            tags=["noise"],
        )

    def add_query(
        self,
        *,
        user_id: str,
        target_entity_id: str,
        text: str,
        required_facts: list[EventFact],
        supporting_events: list[str],
        answer_facts: list[str],
        stale_fact_ids: list[str] | None = None,
    ) -> Query:
        self._advance_time()
        self.query_counter += 1
        query = Query(
            query_id=f"{self.episode_id}-query-{self.query_counter:03d}",
            episode_id=self.episode_id,
            timestamp_ms=self.timestamp_ms,
            user_id=user_id,
            target_entity_id=target_entity_id,
            text=text,
            gold=QueryGold(
                required_fact_ids=[fact.fact_id for fact in required_facts],
                supporting_event_ids=supporting_events,
                answer_facts=answer_facts,
                stale_fact_ids=stale_fact_ids or [],
            ),
        )
        self.stream.append(StreamQueryItem(timestamp_ms=query.timestamp_ms, query=query))
        return query

    def _advance_time(self) -> None:
        self.timestamp_ms += self.rng.randint(250, 1750)


def _rotated(values: list[str], rng: random.Random) -> list[str]:
    copied = list(values)
    rng.shuffle(copied)
    return copied
