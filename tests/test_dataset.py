from __future__ import annotations

from fast_memory_write_env.dataset import generate_dataset, generate_episode
from fast_memory_write_env.schemas import DatasetMode, EventCategory


def test_dataset_generation_is_deterministic_for_same_seed() -> None:
    first = generate_dataset(DatasetMode.SMALL, seed=42)
    second = generate_dataset(DatasetMode.SMALL, seed=42)

    assert first.model_dump(mode="json") == second.model_dump(mode="json")


def test_dataset_generation_changes_with_seed() -> None:
    first = generate_dataset(DatasetMode.SMALL, seed=42)
    second = generate_dataset(DatasetMode.SMALL, seed=43)

    assert first.model_dump(mode="json") != second.model_dump(mode="json")


def test_modes_increase_dataset_size() -> None:
    small = generate_dataset(DatasetMode.SMALL, seed=1)
    medium = generate_dataset(DatasetMode.MEDIUM, seed=1)
    long = generate_dataset(DatasetMode.LONG, seed=1)

    small_items = sum(len(episode.stream) for episode in small.episodes)
    medium_items = sum(len(episode.stream) for episode in medium.episodes)
    long_items = sum(len(episode.stream) for episode in long.episodes)

    assert small_items < medium_items < long_items


def test_episode_contains_interleaved_queries_and_later_events() -> None:
    episode = generate_episode(DatasetMode.SMALL, seed=5, episode_index=0)

    item_types = [item.item_type for item in episode.stream]
    first_query_index = item_types.index("query")

    assert "event" in item_types
    assert "query" in item_types
    assert "event" in item_types[first_query_index + 1 :]


def test_episode_contains_required_event_categories() -> None:
    episode = generate_episode(DatasetMode.SMALL, seed=5, episode_index=0)
    categories = {
        item.event.category
        for item in episode.stream
        if item.item_type == "event"
    }

    assert {
        EventCategory.USEFUL_FACT,
        EventCategory.NOISE,
        EventCategory.DUPLICATE,
        EventCategory.CONTRADICTION,
        EventCategory.STALE_UPDATE,
        EventCategory.URGENT_FACT,
    }.issubset(categories)


def test_queries_reference_only_earlier_facts_and_events() -> None:
    episode = generate_episode(DatasetMode.MEDIUM, seed=9, episode_index=0)
    seen_events: set[str] = set()
    seen_facts: set[str] = set()

    for item in episode.stream:
        if item.item_type == "event":
            seen_events.add(item.event.event_id)
            seen_facts.update(fact.fact_id for fact in item.event.facts)
            continue

        gold = item.query.gold
        assert set(gold.supporting_event_ids).issubset(seen_events)
        assert set(gold.required_fact_ids).issubset(seen_facts)
        assert set(gold.stale_fact_ids).issubset(seen_facts)


def test_episode_includes_multiple_entities_and_users() -> None:
    episode = generate_episode(DatasetMode.SMALL, seed=11, episode_index=0)
    entity_ids = set()
    user_ids = set()

    for item in episode.stream:
        if item.item_type == "event":
            entity_ids.add(item.event.entity_id)
            user_ids.add(item.event.user_id)
        else:
            entity_ids.add(item.query.target_entity_id)
            user_ids.add(item.query.user_id)

    assert len(entity_ids) >= 3
    assert len(user_ids) >= 2
