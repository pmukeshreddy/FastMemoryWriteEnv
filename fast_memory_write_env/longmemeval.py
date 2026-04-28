"""Local-file adapter for LongMemEval streaming episodes."""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fast_memory_write_env.index import estimate_tokens
from fast_memory_write_env.schemas import (
    DatasetMode,
    EventCategory,
    EventFact,
    EventPriority,
    Query,
    QueryGold,
    RawEvent,
    StreamEventItem,
    StreamQueryItem,
    StreamingEpisode,
)


def load_longmemeval_episodes(path: str | Path, *, limit: int | None = None) -> list[StreamingEpisode]:
    """Load LongMemEval JSON from a local file and convert rows to episodes."""

    source_path = Path(path)
    payload = json.loads(source_path.read_text(encoding="utf-8"))
    rows = _extract_rows(payload)
    if limit is not None:
        rows = rows[:limit]
    return [
        longmemeval_item_to_episode(item, item_index=index, source_path=source_path)
        for index, item in enumerate(rows)
    ]


def longmemeval_item_to_episode(
    item: dict[str, Any],
    *,
    item_index: int,
    source_path: str | Path | None = None,
) -> StreamingEpisode:
    """Convert one LongMemEval item into one chronological streaming episode."""

    question_id = str(item.get("question_id") or item.get("id") or f"item-{item_index}")
    safe_question_id = _safe_id(question_id)
    episode_id = f"lme-{safe_question_id}"
    entity_id = f"lme-entity-{safe_question_id}"
    user_id = f"lme-user-{safe_question_id}"
    question_type = str(item.get("question_type") or "unknown")
    answer = str(item.get("answer") or "").strip()
    is_abstention = bool(item.get("is_abstention")) or question_id.endswith("_abs")
    sessions, source_format = _sessions_from_item(item)
    answer_session_ids = {str(value) for value in item.get("answer_session_ids", [])}
    turn_labels_present = any(
        isinstance(turn, dict) and bool(turn.get("has_answer"))
        for session in sessions
        for turn in session["turns"]
    )
    evidence_label_source = (
        "turn_has_answer"
        if turn_labels_present
        else "answer_session_ids"
        if answer_session_ids
        else "documents_field"
        if source_format == "documents"
        else "none"
    )

    events: list[RawEvent] = []
    fact_ids: list[str] = []
    supporting_event_ids: list[str] = []
    event_counter = 0
    for session_index, session in enumerate(sessions):
        session_id = str(session["session_id"])
        base_timestamp_ms = _timestamp_ms(session.get("date"), fallback_ms=session_index * 1_000_000)
        for turn_index, turn in enumerate(session["turns"]):
            event_counter += 1
            role, content, has_answer = _turn_parts(turn)
            event_id = f"{episode_id}-event-{event_counter:04d}"
            event_timestamp_ms = base_timestamp_ms + (turn_index * 1_000)
            is_evidence = _is_evidence_turn(
                source_format=source_format,
                turn_labels_present=turn_labels_present,
                has_answer=has_answer,
                session_id=session_id,
                answer_session_ids=answer_session_ids,
            )
            facts = []
            if is_evidence and not is_abstention:
                fact_id = f"{event_id}-fact-001"
                fact_ids.append(fact_id)
                supporting_event_ids.append(event_id)
                facts = [
                    EventFact(
                        fact_id=fact_id,
                        entity_id=entity_id,
                        attribute="longmemeval_answer_evidence",
                        value=answer or "LongMemEval answer evidence",
                        source_event_id=event_id,
                        valid_from_ms=event_timestamp_ms,
                    )
                ]
            event = RawEvent(
                event_id=event_id,
                episode_id=episode_id,
                timestamp_ms=event_timestamp_ms,
                source="longmemeval_chat",
                user_id=user_id,
                entity_id=entity_id,
                category=EventCategory.USEFUL_FACT if facts else EventCategory.NOISE,
                content=f"{role}: {content}" if role else content,
                facts=facts,
                priority=EventPriority.NORMAL,
                estimated_tokens=estimate_tokens(content),
                metadata={
                    "source_dataset": "longmemeval",
                    "dataset_format": source_format,
                    "longmemeval_question_id": question_id,
                    "question_type": question_type,
                    "session_id": session_id,
                    "session_index": session_index,
                    "session_date": session.get("date"),
                    "turn_index": turn_index,
                    "role": role,
                    "evidence_label_source": evidence_label_source,
                    "has_answer": has_answer,
                },
                tags=["longmemeval", "evidence"] if facts else ["longmemeval", "distractor"],
            )
            events.append(event)

    if not is_abstention and not fact_ids:
        raise ValueError(f"LongMemEval item has no evidence labels: {question_id}")

    query_timestamp_ms = (max(event.timestamp_ms for event in events) + 1_000) if events else 1_000
    query = Query(
        query_id=f"{episode_id}-query-001",
        episode_id=episode_id,
        timestamp_ms=query_timestamp_ms,
        user_id=user_id,
        target_entity_id=entity_id,
        text=str(item.get("question") or ""),
        gold=QueryGold(
            required_fact_ids=fact_ids,
            supporting_event_ids=supporting_event_ids,
            answer_facts=[] if is_abstention else [answer],
            is_abstention=is_abstention,
            notes="LongMemEval abstention item." if is_abstention else None,
        ),
        metadata={
            "dataset_format": "longmemeval",
            "longmemeval_question_id": question_id,
            "question_type": question_type,
            "answer": answer,
            "evidence_label_source": evidence_label_source,
        },
    )
    stream = [StreamEventItem(timestamp_ms=event.timestamp_ms, event=event) for event in sorted(events, key=lambda value: (value.timestamp_ms, value.event_id))]
    stream.append(StreamQueryItem(timestamp_ms=query.timestamp_ms, query=query))
    return StreamingEpisode(
        episode_id=episode_id,
        mode=DatasetMode.LONGMEMEVAL,
        seed=0,
        stream=stream,
        metadata={
            "dataset_format": "longmemeval",
            "source_path": str(source_path) if source_path is not None else None,
            "item_index": item_index,
            "longmemeval_question_id": question_id,
            "question_type": question_type,
            "evidence_label_source": evidence_label_source,
        },
    )


def _extract_rows(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        return [dict(item) for item in payload]
    if isinstance(payload, dict):
        for key in ["data", "items", "examples", "rows"]:
            value = payload.get(key)
            if isinstance(value, list):
                return [dict(item) for item in value]
    raise ValueError("LongMemEval JSON must be a list or contain a data/items/examples/rows list")


def _sessions_from_item(item: dict[str, Any]) -> tuple[list[dict[str, Any]], str]:
    if isinstance(item.get("haystack_sessions"), list):
        session_ids = list(item.get("haystack_session_ids") or [])
        dates = list(item.get("haystack_dates") or [])
        sessions = []
        for index, turns in enumerate(item["haystack_sessions"]):
            session_id = str(session_ids[index]) if index < len(session_ids) else f"session-{index:04d}"
            date = dates[index] if index < len(dates) else None
            sessions.append({"session_id": session_id, "date": date, "turns": _normalize_turns(turns)})
        return sessions, "haystack_sessions"
    if isinstance(item.get("documents"), list):
        sessions = []
        for index, document in enumerate(item["documents"]):
            content = str(document)
            sessions.append(
                {
                    "session_id": f"document-{index:04d}",
                    "date": _date_from_document(content),
                    "turns": [{"role": "document", "content": content, "has_answer": True}],
                }
            )
        return sessions, "documents"
    raise ValueError("LongMemEval item must contain haystack_sessions or documents")


def _normalize_turns(turns: Any) -> list[Any]:
    if isinstance(turns, list):
        return turns
    return [{"role": "session", "content": str(turns)}]


def _turn_parts(turn: Any) -> tuple[str, str, bool]:
    if isinstance(turn, dict):
        role = str(turn.get("role") or turn.get("speaker") or "turn")
        content = str(turn.get("content") or turn.get("text") or "")
        return role, content, bool(turn.get("has_answer"))
    return "turn", str(turn), False


def _is_evidence_turn(
    *,
    source_format: str,
    turn_labels_present: bool,
    has_answer: bool,
    session_id: str,
    answer_session_ids: set[str],
) -> bool:
    if source_format == "documents" and not turn_labels_present and not answer_session_ids:
        return True
    if turn_labels_present:
        return has_answer
    return session_id in answer_session_ids


def _timestamp_ms(value: Any, *, fallback_ms: int) -> int:
    if value is None:
        return fallback_ms
    text = str(value).strip()
    text = re.sub(r"\s*\([^)]*\)", "", text)
    text = text.removeprefix("[Date:").removesuffix("]").strip()
    for pattern in [
        "%Y/%m/%d %H:%M",
        "%Y-%m-%d %H:%M",
        "%Y/%m/%d",
        "%Y-%m-%d",
        "%d %B, %Y",
        "%B %d, %Y",
    ]:
        try:
            parsed = datetime.strptime(text, pattern)
            return int(parsed.replace(tzinfo=timezone.utc).timestamp() * 1000)
        except ValueError:
            continue
    return fallback_ms


def _date_from_document(content: str) -> str | None:
    match = re.search(r"\[Date:\s*([^\]]+)\]", content)
    return match.group(1).strip() if match else None


def _safe_id(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.:\-]+", "-", value).strip("-")
    if not cleaned or not cleaned[0].isalnum():
        cleaned = f"item-{cleaned}"
    return cleaned
