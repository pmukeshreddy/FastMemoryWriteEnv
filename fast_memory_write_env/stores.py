"""SQLite-backed raw event and memory stores."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

from fast_memory_write_env.index import estimate_tokens
from fast_memory_write_env.schemas import MemoryRecord, MemoryStatus, RawEvent


class RawEventStore:
    """SQLite-backed store for fast raw event writes."""

    def __init__(self, sqlite_path: str | Path) -> None:
        self.sqlite_path = Path(sqlite_path)
        self.sqlite_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self.sqlite_path)
        self._conn.row_factory = sqlite3.Row
        self._init_schema()

    def close(self) -> None:
        self._conn.close()

    def store(self, event: RawEvent) -> int:
        """Store one raw event and return storage token delta."""

        tokens = event.estimated_tokens or estimate_tokens(event.content)
        try:
            with self._conn:
                self._conn.execute(
                    """
                    INSERT INTO raw_events (
                        event_id, episode_id, timestamp_ms, user_id, entity_id,
                        category, estimated_tokens, payload_json
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        event.event_id,
                        event.episode_id,
                        event.timestamp_ms,
                        event.user_id,
                        event.entity_id,
                        event.category.value,
                        tokens,
                        event.model_dump_json(),
                    ),
                )
        except sqlite3.IntegrityError as exc:
            raise ValueError(f"raw event already exists: {event.event_id}") from exc
        return tokens

    def get(self, event_id: str) -> RawEvent | None:
        row = self._conn.execute(
            "SELECT payload_json FROM raw_events WHERE event_id = ?",
            (event_id,),
        ).fetchone()
        if row is None:
            return None
        return RawEvent.model_validate_json(row["payload_json"])

    def list_by_episode(self, episode_id: str) -> list[RawEvent]:
        rows = self._conn.execute(
            """
            SELECT payload_json FROM raw_events
            WHERE episode_id = ?
            ORDER BY timestamp_ms, event_id
            """,
            (episode_id,),
        ).fetchall()
        return [RawEvent.model_validate_json(row["payload_json"]) for row in rows]

    def count(self) -> int:
        row = self._conn.execute("SELECT COUNT(*) AS count FROM raw_events").fetchone()
        return int(row["count"])

    def _init_schema(self) -> None:
        with self._conn:
            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS raw_events (
                    event_id TEXT PRIMARY KEY,
                    episode_id TEXT NOT NULL,
                    timestamp_ms INTEGER NOT NULL,
                    user_id TEXT NOT NULL,
                    entity_id TEXT NOT NULL,
                    category TEXT NOT NULL,
                    estimated_tokens INTEGER NOT NULL,
                    payload_json TEXT NOT NULL
                )
                """
            )
            self._conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_raw_events_episode_time "
                "ON raw_events (episode_id, timestamp_ms)"
            )


class MemoryStore:
    """SQLite-backed memory store."""

    def __init__(self, sqlite_path: str | Path) -> None:
        self.sqlite_path = Path(sqlite_path)
        self.sqlite_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self.sqlite_path)
        self._conn.row_factory = sqlite3.Row
        self._init_schema()

    def close(self) -> None:
        self._conn.close()

    def create(self, memory: MemoryRecord) -> int:
        tokens = memory.estimated_tokens or estimate_tokens(memory.content)
        memory = memory.model_copy(update={"estimated_tokens": tokens})
        try:
            with self._conn:
                self._insert_or_replace(memory, replace=False)
        except sqlite3.IntegrityError as exc:
            raise ValueError(f"memory already exists: {memory.memory_id}") from exc
        return tokens

    def upsert(self, memory: MemoryRecord) -> int:
        existing = self.get(memory.memory_id)
        tokens = memory.estimated_tokens or estimate_tokens(memory.content)
        memory = memory.model_copy(update={"estimated_tokens": tokens})
        with self._conn:
            self._insert_or_replace(memory, replace=True)
        previous_tokens = existing.estimated_tokens if existing else 0
        return tokens - previous_tokens

    def get(self, memory_id: str) -> MemoryRecord | None:
        row = self._conn.execute(
            "SELECT payload_json FROM memories WHERE memory_id = ?",
            (memory_id,),
        ).fetchone()
        if row is None:
            return None
        return MemoryRecord.model_validate_json(row["payload_json"])

    def require(self, memory_id: str) -> MemoryRecord:
        memory = self.get(memory_id)
        if memory is None:
            raise ValueError(f"memory not found: {memory_id}")
        return memory

    def list_all(self) -> list[MemoryRecord]:
        rows = self._conn.execute(
            "SELECT payload_json FROM memories ORDER BY created_at_ms, memory_id"
        ).fetchall()
        return [MemoryRecord.model_validate_json(row["payload_json"]) for row in rows]

    def list_active(self) -> list[MemoryRecord]:
        rows = self._conn.execute(
            """
            SELECT payload_json FROM memories
            WHERE status = ?
            ORDER BY created_at_ms, memory_id
            """,
            (MemoryStatus.ACTIVE.value,),
        ).fetchall()
        return [MemoryRecord.model_validate_json(row["payload_json"]) for row in rows]

    def update_memory(
        self,
        *,
        memory_id: str,
        content: str,
        source_event_ids: list[str],
        fact_ids: list[str],
        updated_at_ms: int,
        metadata: dict[str, Any] | None = None,
    ) -> tuple[MemoryRecord, int]:
        existing = self.require(memory_id)
        merged_metadata = dict(existing.metadata)
        if metadata:
            merged_metadata.update(metadata)
        updated = existing.model_copy(
            update={
                "content": content,
                "source_event_ids": _merge_unique(existing.source_event_ids, source_event_ids),
                "fact_ids": _merge_unique(existing.fact_ids, fact_ids),
                "updated_at_ms": max(updated_at_ms, existing.updated_at_ms),
                "estimated_tokens": estimate_tokens(content),
                "metadata": merged_metadata,
            }
        )
        delta = self.upsert(updated)
        return updated, delta

    def mark_status(
        self,
        *,
        memory_id: str,
        status: MemoryStatus,
        updated_at_ms: int,
        metadata: dict[str, Any] | None = None,
    ) -> MemoryRecord:
        existing = self.require(memory_id)
        merged_metadata = dict(existing.metadata)
        if metadata:
            merged_metadata.update(metadata)
        indexed = existing.indexed if status == MemoryStatus.ACTIVE else False
        updated = existing.model_copy(
            update={
                "status": status,
                "indexed": indexed,
                "updated_at_ms": max(updated_at_ms, existing.updated_at_ms),
                "metadata": merged_metadata,
            }
        )
        self.upsert(updated)
        return updated

    def set_indexed(self, memory_id: str, indexed: bool, updated_at_ms: int) -> MemoryRecord:
        existing = self.require(memory_id)
        updated = existing.model_copy(
            update={
                "indexed": indexed,
                "updated_at_ms": max(updated_at_ms, existing.updated_at_ms),
            }
        )
        self.upsert(updated)
        return updated

    def delay_index(
        self,
        *,
        memory_id: str,
        retry_after_ms: int,
        reason: str,
        updated_at_ms: int,
    ) -> MemoryRecord:
        existing = self.require(memory_id)
        metadata = dict(existing.metadata)
        metadata.update({"delayed_index_until_ms": retry_after_ms, "delay_index_reason": reason})
        updated = existing.model_copy(
            update={
                "indexed": False,
                "updated_at_ms": max(updated_at_ms, existing.updated_at_ms),
                "metadata": metadata,
            }
        )
        self.upsert(updated)
        return updated

    def _insert_or_replace(self, memory: MemoryRecord, *, replace: bool) -> None:
        statement = "INSERT OR REPLACE" if replace else "INSERT"
        self._conn.execute(
            f"""
            {statement} INTO memories (
                memory_id, entity_id, status, indexed, estimated_tokens,
                created_at_ms, updated_at_ms, payload_json
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                memory.memory_id,
                memory.entity_id,
                memory.status.value,
                int(memory.indexed),
                memory.estimated_tokens,
                memory.created_at_ms,
                memory.updated_at_ms,
                memory.model_dump_json(),
            ),
        )

    def _init_schema(self) -> None:
        with self._conn:
            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS memories (
                    memory_id TEXT PRIMARY KEY,
                    entity_id TEXT NOT NULL,
                    status TEXT NOT NULL,
                    indexed INTEGER NOT NULL,
                    estimated_tokens INTEGER NOT NULL,
                    created_at_ms INTEGER NOT NULL,
                    updated_at_ms INTEGER NOT NULL,
                    payload_json TEXT NOT NULL
                )
                """
            )
            self._conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_memories_status ON memories (status)"
            )
            self._conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_memories_entity ON memories (entity_id)"
            )


def _merge_unique(left: list[str], right: list[str]) -> list[str]:
    seen: set[str] = set()
    merged: list[str] = []
    for value in [*left, *right]:
        if value not in seen:
            merged.append(value)
            seen.add(value)
    return merged
