"""Environment execution loop for Phase 2 memory actions."""

from __future__ import annotations

import hashlib
from typing import Mapping

from pydantic import ValidationError

from fast_memory_write_env.actions import (
    ActionExecutionResult,
    ActionResult,
    AnswerAction,
    CompressMemoryAction,
    DelayIndexAction,
    EnvironmentAction,
    IgnoreEventAction,
    IndexNowAction,
    MarkStaleAction,
    SearchMemoryAction,
    StoreRawAction,
    UpdateMemoryAction,
    WriteMemoryAction,
    validate_environment_action,
)
from fast_memory_write_env.index import RetrievalIndex, SearchResult, estimate_tokens
from fast_memory_write_env.schemas import MemoryRecord, MemoryStatus, StreamingEpisode
from fast_memory_write_env.stores import MemoryStore, RawEventStore


BASE_LATENCY_MS = {
    "store_raw": 1.0,
    "write_memory": 6.0,
    "update_memory": 5.0,
    "mark_stale": 2.0,
    "ignore_event": 0.5,
    "compress_memory": 8.0,
    "index_now": 4.0,
    "delay_index": 1.0,
    "search_memory": 3.0,
    "answer": 2.0,
}


class FastMemoryWriteEnv:
    """Executes validated actions against stores and retrieval index."""

    def __init__(
        self,
        *,
        raw_event_store: RawEventStore,
        memory_store: MemoryStore,
        retrieval_index: RetrievalIndex,
        current_time_ms: int = 0,
    ) -> None:
        self.raw_event_store = raw_event_store
        self.memory_store = memory_store
        self.retrieval_index = retrieval_index
        self.current_time_ms = current_time_ms
        self.ignored_event_ids: list[str] = []
        self.last_search_results: list[SearchResult] = []

    def execute_action(self, action_data: EnvironmentAction | dict[str, object]) -> ActionResult:
        """Validate and execute one environment action."""

        try:
            action = validate_environment_action(action_data)
            if isinstance(action, StoreRawAction):
                return self._store_raw(action)
            if isinstance(action, WriteMemoryAction):
                return self._write_memory(action)
            if isinstance(action, UpdateMemoryAction):
                return self._update_memory(action)
            if isinstance(action, MarkStaleAction):
                return self._mark_stale(action)
            if isinstance(action, IgnoreEventAction):
                return self._ignore_event(action)
            if isinstance(action, CompressMemoryAction):
                return self._compress_memory(action)
            if isinstance(action, IndexNowAction):
                return self._index_now(action)
            if isinstance(action, DelayIndexAction):
                return self._delay_index(action)
            if isinstance(action, SearchMemoryAction):
                return self._search_memory(action)
            if isinstance(action, AnswerAction):
                return self._answer(action)
            raise TypeError(f"unsupported action: {type(action).__name__}")
        except (ValueError, ValidationError, TypeError) as exc:
            action_type = _extract_action_type(action_data)
            return ActionExecutionResult(
                success=False,
                action_type=action_type,
                latency_ms=_latency_ms(action_type),
                storage_tokens_delta=0,
                error=str(exc),
            )

    def execute_actions(self, actions: list[EnvironmentAction | dict[str, object]]) -> list[ActionResult]:
        """Execute actions sequentially."""

        return [self.execute_action(action) for action in actions]

    def run_episode(
        self,
        episode: StreamingEpisode,
        *,
        action_batches: Mapping[str, list[EnvironmentAction | dict[str, object]]] | None = None,
    ) -> list[ActionResult]:
        """Run an episode, storing raw events and executing injected action batches."""

        results: list[ActionResult] = []
        batches = action_batches or {}
        for item in episode.stream:
            self.current_time_ms = max(self.current_time_ms, item.timestamp_ms)
            if item.item_type == "event":
                results.append(self.execute_action(StoreRawAction(event=item.event)))
                results.extend(self.execute_actions(list(batches.get(item.event.event_id, []))))
            else:
                search_result = self.execute_action(
                    SearchMemoryAction(
                        query_id=item.query.query_id,
                        query_text=item.query.text,
                        top_k=5,
                    )
                )
                results.append(search_result)
                retrieved_ids = [
                    hit["memory_id"]
                    for hit in search_result.payload.get("results", [])
                    if isinstance(hit, dict) and "memory_id" in hit
                ]
                results.append(
                    self.execute_action(
                        AnswerAction(
                            query_id=item.query.query_id,
                            query_text=item.query.text,
                            retrieved_memory_ids=retrieved_ids,
                        )
                    )
                )
        return results

    def _store_raw(self, action: StoreRawAction) -> ActionResult:
        self.current_time_ms = max(self.current_time_ms, action.event.timestamp_ms)
        delta = self.raw_event_store.store(action.event)
        return _success(
            "store_raw",
            token_count=delta,
            storage_tokens_delta=delta,
            payload={"event_id": action.event.event_id},
        )

    def _write_memory(self, action: WriteMemoryAction) -> ActionResult:
        tokens = estimate_tokens(action.content)
        memory = MemoryRecord(
            memory_id=action.memory_id,
            entity_id=action.entity_id,
            content=action.content,
            source_event_ids=action.source_event_ids,
            fact_ids=action.fact_ids,
            created_at_ms=self.current_time_ms,
            updated_at_ms=self.current_time_ms,
            estimated_tokens=tokens,
            metadata={"importance": action.importance, **action.metadata},
        )
        delta = self.memory_store.create(memory)
        indexed = False
        if action.index_immediately:
            indexed_memory = self.memory_store.set_indexed(
                memory.memory_id,
                True,
                self.current_time_ms,
            )
            self.retrieval_index.upsert(indexed_memory)
            indexed = True
        return _success(
            "write_memory",
            token_count=tokens,
            storage_tokens_delta=delta,
            payload={"memory_id": memory.memory_id, "indexed": indexed},
        )

    def _update_memory(self, action: UpdateMemoryAction) -> ActionResult:
        previous = self.memory_store.require(action.memory_id)
        updated, delta = self.memory_store.update_memory(
            memory_id=action.memory_id,
            content=action.content,
            source_event_ids=action.source_event_ids,
            fact_ids=action.fact_ids,
            updated_at_ms=self.current_time_ms,
            metadata={"last_update_reason": action.reason, **action.metadata},
        )
        should_index = action.index_immediately or previous.indexed
        if should_index:
            updated = self.memory_store.set_indexed(action.memory_id, True, self.current_time_ms)
            self.retrieval_index.upsert(updated)
        return _success(
            "update_memory",
            token_count=estimate_tokens(action.content),
            storage_tokens_delta=delta,
            payload={"memory_id": action.memory_id, "indexed": should_index},
        )

    def _mark_stale(self, action: MarkStaleAction) -> ActionResult:
        updated = self.memory_store.mark_status(
            memory_id=action.memory_id,
            status=MemoryStatus.STALE,
            updated_at_ms=self.current_time_ms,
            metadata={
                "stale_reason": action.reason,
                "stale_source_event_id": action.source_event_id,
                "superseded_by_memory_id": action.superseded_by_memory_id,
                **action.metadata,
            },
        )
        self.retrieval_index.delete(action.memory_id)
        return _success(
            "mark_stale",
            payload={"memory_id": action.memory_id, "status": updated.status.value},
        )

    def _ignore_event(self, action: IgnoreEventAction) -> ActionResult:
        self.ignored_event_ids.append(action.event_id)
        return _success(
            "ignore_event",
            payload={"event_id": action.event_id, "reason": action.reason},
        )

    def _compress_memory(self, action: CompressMemoryAction) -> ActionResult:
        source_memories = [self.memory_store.require(memory_id) for memory_id in action.source_memory_ids]
        target_existing = self.memory_store.get(action.target_memory_id)
        source_event_ids = _merge_unique(
            [event_id for memory in source_memories for event_id in memory.source_event_ids],
            action.source_event_ids,
        )
        fact_ids = _merge_unique(
            [fact_id for memory in source_memories for fact_id in memory.fact_ids],
            action.fact_ids,
        )
        target = MemoryRecord(
            memory_id=action.target_memory_id,
            entity_id=source_memories[0].entity_id,
            content=action.compressed_content,
            source_event_ids=source_event_ids,
            fact_ids=fact_ids,
            created_at_ms=target_existing.created_at_ms if target_existing else self.current_time_ms,
            updated_at_ms=self.current_time_ms,
            estimated_tokens=estimate_tokens(action.compressed_content),
            metadata={"compressed_from": action.source_memory_ids, **action.metadata},
        )
        delta = self.memory_store.upsert(target)
        for memory_id in action.source_memory_ids:
            self.memory_store.mark_status(
                memory_id=memory_id,
                status=MemoryStatus.COMPRESSED,
                updated_at_ms=self.current_time_ms,
                metadata={"compressed_into": action.target_memory_id},
            )
            self.retrieval_index.delete(memory_id)
        return _success(
            "compress_memory",
            token_count=target.estimated_tokens,
            storage_tokens_delta=delta,
            payload={
                "target_memory_id": action.target_memory_id,
                "source_memory_ids": action.source_memory_ids,
            },
        )

    def _index_now(self, action: IndexNowAction) -> ActionResult:
        memory = self._index_memory(action.memory_id)
        return _success(
            "index_now",
            token_count=memory.estimated_tokens,
            payload={"memory_id": memory.memory_id, "indexed": True},
        )

    def _delay_index(self, action: DelayIndexAction) -> ActionResult:
        retry_at_ms = self.current_time_ms + action.retry_after_ms
        memory = self.memory_store.delay_index(
            memory_id=action.memory_id,
            retry_after_ms=retry_at_ms,
            reason=action.reason,
            updated_at_ms=self.current_time_ms,
        )
        self.retrieval_index.delete(action.memory_id)
        return _success(
            "delay_index",
            payload={
                "memory_id": action.memory_id,
                "retry_at_ms": retry_at_ms,
                "indexed": memory.indexed,
            },
        )

    def _search_memory(self, action: SearchMemoryAction) -> ActionResult:
        results = self.retrieval_index.search(
            action.query_text,
            top_k=action.top_k,
            filters=action.filters,
        )
        self.last_search_results = results
        return _success(
            "search_memory",
            token_count=estimate_tokens(action.query_text),
            count=len(results),
            payload={
                "query_id": action.query_id,
                "results": [
                    {
                        "memory_id": result.memory_id,
                        "score": result.score,
                        "content": result.content,
                        "metadata": result.metadata,
                    }
                    for result in results
                ],
            },
        )

    def _answer(self, action: AnswerAction) -> ActionResult:
        memory_ids = action.retrieved_memory_ids or [result.memory_id for result in self.last_search_results]
        memories = [memory for memory_id in memory_ids if (memory := self.memory_store.get(memory_id))]
        if memories:
            answer = " ".join(memory.content for memory in memories)
        else:
            answer = "I do not know from indexed memory."
        return _success(
            "answer",
            token_count=estimate_tokens(action.query_text) + estimate_tokens(answer),
            count=len(memories),
            payload={
                "query_id": action.query_id,
                "answer": answer,
                "cited_memory_ids": [memory.memory_id for memory in memories],
            },
        )

    def _index_memory(self, memory_id: str) -> MemoryRecord:
        memory = self.memory_store.require(memory_id)
        if memory.status != MemoryStatus.ACTIVE:
            raise ValueError(f"cannot index non-active memory: {memory_id}")
        indexed_memory = self.memory_store.set_indexed(memory_id, True, self.current_time_ms)
        self.retrieval_index.upsert(indexed_memory)
        return indexed_memory


def deterministic_memory_id(source_event_ids: list[str], content: str, *, prefix: str = "mem") -> str:
    """Create a stable memory ID from source event IDs and content."""

    seed = "|".join([*sorted(source_event_ids), content])
    digest = hashlib.sha1(seed.encode("utf-8")).hexdigest()[:12]
    return f"{prefix}-{digest}"


def _success(
    action_type: str,
    *,
    token_count: int = 0,
    count: int = 0,
    storage_tokens_delta: int = 0,
    payload: dict[str, object] | None = None,
) -> ActionExecutionResult:
    return ActionExecutionResult(
        success=True,
        action_type=action_type,
        latency_ms=_latency_ms(action_type, token_count=token_count, count=count),
        storage_tokens_delta=storage_tokens_delta,
        payload=payload or {},
    )


def _latency_ms(action_type: str, *, token_count: int = 0, count: int = 0) -> float:
    base = BASE_LATENCY_MS.get(action_type, 1.0)
    return round(base + (token_count * 0.05) + (count * 0.2), 3)


def _extract_action_type(action_data: object) -> str:
    if isinstance(action_data, dict):
        raw = action_data.get("action_type", "ignore_event")
    else:
        raw = getattr(action_data, "action_type", "ignore_event")
    if hasattr(raw, "value"):
        return str(raw.value)
    if isinstance(raw, str) and raw in BASE_LATENCY_MS:
        return raw
    return "ignore_event"


def _merge_unique(left: list[str], right: list[str]) -> list[str]:
    seen: set[str] = set()
    merged: list[str] = []
    for value in [*left, *right]:
        if value not in seen:
            merged.append(value)
            seen.add(value)
    return merged
