from __future__ import annotations

import json

from fast_memory_write_env.longmemeval import load_longmemeval_episodes
from fast_memory_write_env.schemas import DatasetMode, EventCategory, EventPriority


def test_longmemeval_loader_converts_normal_and_abstention_items(tmp_path) -> None:
    dataset_path = tmp_path / "longmemeval_fixture.json"
    dataset_path.write_text(
        json.dumps(
            [
                {
                    "question_id": "q-normal",
                    "question_type": "single-session-user",
                    "question": "What is the name of my cat?",
                    "answer": "Luna",
                    "question_date": "2023/05/29 (Mon) 00:00",
                    "haystack_session_ids": ["s1", "s2"],
                    "haystack_dates": ["2023/05/28 (Sun) 06:27", "2023/05/28 (Sun) 07:00"],
                    "answer_session_ids": ["s1"],
                    "haystack_sessions": [
                        [
                            {"role": "user", "content": "My cat is named Luna.", "has_answer": True},
                            {"role": "assistant", "content": "Luna is a lovely name."},
                        ],
                        [
                            {"role": "user", "content": "I also need help choosing socks."},
                        ],
                    ],
                },
                {
                    "question_id": "q-abs_abs",
                    "question_type": "knowledge-update",
                    "question": "What is the name of my dog?",
                    "answer": "The information provided is not enough.",
                    "is_abstention": True,
                    "haystack_session_ids": ["s3"],
                    "haystack_dates": ["2023/05/28 (Sun) 08:00"],
                    "haystack_sessions": [
                        [
                            {"role": "user", "content": "My cat is named Luna."},
                        ]
                    ],
                },
            ]
        ),
        encoding="utf-8",
    )

    normal, abstention = load_longmemeval_episodes(dataset_path)

    assert normal.mode == DatasetMode.LONGMEMEVAL
    assert normal.metadata["longmemeval_question_id"] == "q-normal"
    assert normal.metadata["evidence_label_source"] == "turn_has_answer"
    assert normal.stream[-1].item_type == "query"
    assert all(
        normal.stream[index].timestamp_ms <= normal.stream[index + 1].timestamp_ms
        for index in range(len(normal.stream) - 1)
    )

    evidence_events = [
        item.event
        for item in normal.stream
        if item.item_type == "event" and item.event.category == EventCategory.USEFUL_FACT
    ]
    all_events = [item.event for item in normal.stream if item.item_type == "event"]
    query = normal.stream[-1].query
    assert len(evidence_events) == 1
    assert evidence_events[0].facts
    assert {event.priority for event in all_events} == {EventPriority.NORMAL}
    assert query.gold.required_fact_ids == [evidence_events[0].facts[0].fact_id]
    assert query.gold.supporting_event_ids == [evidence_events[0].event_id]
    assert query.gold.answer_facts == ["Luna"]
    assert query.gold.is_abstention is False

    abstention_query = abstention.stream[-1].query
    assert abstention_query.gold.is_abstention is True
    assert abstention_query.gold.required_fact_ids == []
    assert abstention_query.gold.supporting_event_ids == []
    assert abstention_query.gold.answer_facts == []


def test_longmemeval_loader_uses_answer_session_fallback(tmp_path) -> None:
    dataset_path = tmp_path / "longmemeval_fallback.json"
    dataset_path.write_text(
        json.dumps(
            [
                {
                    "question_id": "q-fallback",
                    "question_type": "multi-session",
                    "question": "What cake did I bake?",
                    "answer": "a chocolate cake",
                    "haystack_session_ids": ["s1", "s2"],
                    "haystack_dates": ["2022/03/21 (Mon) 14:57", "2022/03/22 (Tue) 14:57"],
                    "answer_session_ids": ["s2"],
                    "haystack_sessions": [
                        [{"role": "user", "content": "I bought flour."}],
                        [{"role": "user", "content": "I baked a chocolate cake for my friend."}],
                    ],
                }
            ]
        ),
        encoding="utf-8",
    )

    episode = load_longmemeval_episodes(dataset_path)[0]

    assert episode.metadata["evidence_label_source"] == "answer_session_ids"
    evidence_events = [
        item.event
        for item in episode.stream
        if item.item_type == "event" and item.event.category == EventCategory.USEFUL_FACT
    ]
    assert [event.metadata["session_id"] for event in evidence_events] == ["s2"]
