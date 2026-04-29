"""Environment execution loop for Phase 2 memory actions."""

from __future__ import annotations

import threading
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
    deterministic_memory_id,
    validate_environment_action,
)
from fast_memory_write_env.index import RetrievalIndex, SearchResult, estimate_tokens
from fast_memory_write_env.llm_client import LLMClient, LLMClientError, LLMMessage
from fast_memory_write_env.schemas import MemoryRecord, MemoryStatus, StreamingEpisode
from fast_memory_write_env.stores import MemoryStore, RawEventStore


_ANSWER_COMPOSE_SYSTEM_PROMPT = (
    "You are a focused question-answering assistant for a streaming memory "
    "system. You will be given a user question and a list of candidate memory "
    "passages retrieved from the system's memory store. Each passage has a "
    "memory_id and content.\n\n"
    "Rules:\n"
    "- Answer ONLY the question that was asked. Do not append unrelated facts.\n"
    "- Use ONLY the memories that are directly relevant. Ignore the rest.\n"
    "- If no provided memory is relevant, abstain with the exact text "
    "\"I do not know from indexed memory.\"\n"
    "- Keep the answer to one short, focused sentence (or two if multi-fact).\n"
    "- Quote facts faithfully; do not invent details.\n\n"
    "Reply in JSON only with this exact shape:\n"
    "{\"answer\": str, \"cited_memory_ids\": [str]}\n"
    "where cited_memory_ids lists exactly the memory_ids you actually used in "
    "the answer (empty list when abstaining)."
)
_ANSWER_COMPOSE_MAX_PASSAGE_CHARS = 800
_ANSWER_ABSTENTION_TEXT = "I do not know from indexed memory."


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
        storage_budget_tokens_remaining: int | None = None,
        indexing_budget_operations_remaining: int | None = None,
        answer_llm_client: LLMClient | None = None,
    ) -> None:
        self.raw_event_store = raw_event_store
        self.memory_store = memory_store
        self.retrieval_index = retrieval_index
        self.current_time_ms = current_time_ms
        self.storage_budget_tokens_remaining: int | None = None
        self.indexing_budget_operations_remaining: int | None = None
        self.set_budgets(
            storage_budget_tokens_remaining=storage_budget_tokens_remaining,
            indexing_budget_operations_remaining=indexing_budget_operations_remaining,
        )
        self.ignored_event_ids: list[str] = []
        self.last_search_results: list[SearchResult] = []
        # Optional LLM that composes the per-query answer from the retrieved
        # memories. When ``None`` the env falls back to a deterministic
        # top-1-memory answer that avoids the multi-memory-concat noise that
        # the dumb concatenation path used to produce.
        self.answer_llm_client: LLMClient | None = answer_llm_client
        self._lock = threading.RLock()

    def set_budgets(
        self,
        *,
        storage_budget_tokens_remaining: int | None = None,
        indexing_budget_operations_remaining: int | None = None,
    ) -> None:
        """Configure strict write/index budgets.

        ``None`` keeps a budget unenforced for low-level transition tests and
        direct environment use. The streaming evaluator always sets both.
        """

        if storage_budget_tokens_remaining is not None and storage_budget_tokens_remaining < 0:
            raise ValueError("storage_budget_tokens_remaining must be nonnegative")
        if indexing_budget_operations_remaining is not None and indexing_budget_operations_remaining < 0:
            raise ValueError("indexing_budget_operations_remaining must be nonnegative")
        self.storage_budget_tokens_remaining = storage_budget_tokens_remaining
        self.indexing_budget_operations_remaining = indexing_budget_operations_remaining

    def budget_snapshot(self) -> dict[str, int | None]:
        """Return the currently enforceable write/index budgets."""

        return {
            "storage_budget_tokens_remaining": self.storage_budget_tokens_remaining,
            "indexing_budget_operations_remaining": self.indexing_budget_operations_remaining,
        }

    def execute_action(self, action_data: EnvironmentAction | dict[str, object]) -> ActionResult:
        """Validate and execute one environment action."""

        with self._lock:
            return self._execute_action_unlocked(action_data)

    def execute_action_at(
        self,
        action_data: EnvironmentAction | dict[str, object],
        *,
        current_time_ms: int,
    ) -> ActionResult:
        """Set logical time and execute one action under the environment lock."""

        with self._lock:
            self.current_time_ms = current_time_ms
            return self._execute_action_unlocked(action_data)

    def _execute_action_unlocked(self, action_data: EnvironmentAction | dict[str, object]) -> ActionResult:
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
        self._validate_source_event_ids(action.source_event_ids)
        tokens = estimate_tokens(action.content)
        self._require_storage_budget(tokens)
        self._require_indexing_budget(1 if action.index_immediately else 0)
        lexical_available_at_ms = _available_at_ms(
            self.current_time_ms,
            "write_memory",
            token_count=tokens,
        )
        memory = MemoryRecord(
            memory_id=action.memory_id,
            entity_id=action.entity_id,
            content=action.content,
            source_event_ids=action.source_event_ids,
            fact_ids=action.fact_ids,
            created_at_ms=self.current_time_ms,
            updated_at_ms=self.current_time_ms,
            estimated_tokens=tokens,
            metadata={
                "importance": action.importance,
                "lexical_available_at_ms": lexical_available_at_ms,
                "needs_reindex": False,
                **action.metadata,
            },
        )
        delta = self.memory_store.create(memory)
        indexed = False
        available_at_ms = None
        if action.index_immediately:
            memory = self.memory_store.set_indexed(
                memory.memory_id,
                True,
                self.current_time_ms,
                metadata_updates={"needs_reindex": False},
            )
            indexed = True
            available_at_ms = lexical_available_at_ms
            self.retrieval_index.upsert(_memory_available_at(memory, available_at_ms))
        self._consume_storage_budget(delta)
        self._consume_indexing_budget(1 if indexed else 0)
        return _success(
            "write_memory",
            token_count=tokens,
            storage_tokens_delta=delta,
            payload={
                "memory_id": memory.memory_id,
                "indexed": indexed,
                "lexical_available_at_ms": lexical_available_at_ms,
                "available_at_ms": available_at_ms,
            },
        )

    def _update_memory(self, action: UpdateMemoryAction) -> ActionResult:
        """Apply a content update without auto-spending the indexing budget.

        Semantics:

        * ``index_immediately=True`` re-upserts the vector and costs 1 indexing op.
        * ``index_immediately=False`` on a previously indexed memory drops the
          stale vector and marks ``needs_reindex=True`` instead of silently
          burning budget. The LLM sees ``needs_reindex`` on ``active_memories``
          and can call ``index_now`` later when budget allows.
        * The lexical (BM25) mirror is refreshed on every update so the
          memory remains hybrid-queryable with the new content immediately.
        """

        self._validate_source_event_ids(action.source_event_ids)
        previous = self.memory_store.require(action.memory_id)
        projected_delta = estimate_tokens(action.content) - previous.estimated_tokens
        will_index_now = action.index_immediately
        self._require_storage_budget(projected_delta)
        self._require_indexing_budget(1 if will_index_now else 0)

        token_count = estimate_tokens(action.content)
        new_lexical_available_at_ms = _available_at_ms(
            self.current_time_ms,
            "update_memory",
            token_count=token_count,
        )
        needs_reindex_after = previous.indexed and not will_index_now
        metadata_updates: dict[str, object] = {
            "last_update_reason": action.reason,
            "lexical_available_at_ms": new_lexical_available_at_ms,
            "needs_reindex": needs_reindex_after,
            **action.metadata,
        }
        updated, delta = self.memory_store.update_memory(
            memory_id=action.memory_id,
            content=action.content,
            source_event_ids=action.source_event_ids,
            fact_ids=action.fact_ids,
            updated_at_ms=self.current_time_ms,
            metadata=metadata_updates,
        )
        available_at_ms: float | None = None
        if will_index_now:
            updated = self.memory_store.set_indexed(
                action.memory_id,
                True,
                self.current_time_ms,
                metadata_updates={"needs_reindex": False},
            )
            available_at_ms = new_lexical_available_at_ms
            self.retrieval_index.upsert(_memory_available_at(updated, available_at_ms))
        elif previous.indexed:
            # Drop the stale vector. Lexical search continues to serve the
            # current content via the FTS5 mirror; a future index_now (which
            # the LLM can see is required because needs_reindex=True is on
            # the memory) restores semantic retrieval.
            self.memory_store.set_indexed(
                action.memory_id,
                False,
                self.current_time_ms,
            )
            self.retrieval_index.delete(
                action.memory_id,
                available_at_ms=_available_at_ms(self.current_time_ms, "update_memory"),
            )

        self._consume_storage_budget(delta)
        self._consume_indexing_budget(1 if will_index_now else 0)
        return _success(
            "update_memory",
            token_count=token_count,
            storage_tokens_delta=delta,
            payload={
                "memory_id": action.memory_id,
                "indexed": will_index_now,
                "needs_reindex": needs_reindex_after,
                "lexical_available_at_ms": new_lexical_available_at_ms,
                "available_at_ms": available_at_ms,
            },
        )

    def _mark_stale(self, action: MarkStaleAction) -> ActionResult:
        if action.source_event_id is not None:
            self._validate_source_event_ids([action.source_event_id])
        previous = self.memory_store.require(action.memory_id)
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
        if previous.indexed:
            self.retrieval_index.delete(
                updated.memory_id,
                available_at_ms=_available_at_ms(self.current_time_ms, "mark_stale"),
            )
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
        self._validate_source_event_ids(action.source_event_ids)
        source_memories = [self.memory_store.require(memory_id) for memory_id in action.source_memory_ids]
        indexed_source_ids = {memory.memory_id for memory in source_memories if memory.indexed}
        target_existing = self.memory_store.get(action.target_memory_id)
        source_event_ids = _merge_unique(
            [event_id for memory in source_memories for event_id in memory.source_event_ids],
            action.source_event_ids,
        )
        self._validate_source_event_ids(source_event_ids)
        fact_ids = _merge_unique(
            [fact_id for memory in source_memories for fact_id in memory.fact_ids],
            action.fact_ids,
        )
        compressed_lexical_available_at_ms = _available_at_ms(
            self.current_time_ms,
            "compress_memory",
            token_count=estimate_tokens(action.compressed_content),
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
            metadata={
                "compressed_from": action.source_memory_ids,
                "lexical_available_at_ms": compressed_lexical_available_at_ms,
                "needs_reindex": False,
                **action.metadata,
            },
        )
        previous_target_tokens = target_existing.estimated_tokens if target_existing else 0
        self._require_storage_budget(target.estimated_tokens - previous_target_tokens)
        self._require_indexing_budget(1 if action.index_immediately else 0)
        delta = self.memory_store.upsert(target)
        for memory_id in action.source_memory_ids:
            compressed = self.memory_store.mark_status(
                memory_id=memory_id,
                status=MemoryStatus.COMPRESSED,
                updated_at_ms=self.current_time_ms,
                metadata={"compressed_into": action.target_memory_id},
            )
            if memory_id in indexed_source_ids:
                self.retrieval_index.delete(
                    compressed.memory_id,
                    available_at_ms=_available_at_ms(
                        self.current_time_ms,
                        "compress_memory",
                        token_count=target.estimated_tokens,
                    ),
                )
        indexed = False
        available_at_ms = None
        if action.index_immediately:
            available_at_ms = _available_at_ms(
                self.current_time_ms,
                "compress_memory",
                token_count=target.estimated_tokens,
            )
            indexed_target = self.memory_store.set_indexed(
                target.memory_id,
                True,
                self.current_time_ms,
            )
            self.retrieval_index.upsert(_memory_available_at(indexed_target, available_at_ms))
            indexed = True
        self._consume_storage_budget(delta)
        self._consume_indexing_budget(1 if indexed else 0)
        return _success(
            "compress_memory",
            token_count=target.estimated_tokens,
            storage_tokens_delta=delta,
            payload={
                "target_memory_id": action.target_memory_id,
                "source_memory_ids": action.source_memory_ids,
                "indexed": indexed,
                "available_at_ms": available_at_ms,
            },
        )

    def _index_now(self, action: IndexNowAction) -> ActionResult:
        self._require_indexing_budget(1)
        memory = self.memory_store.require(action.memory_id)
        memory = self._index_memory(
            action.memory_id,
            available_at_ms=_available_at_ms(
                self.current_time_ms,
                "index_now",
                token_count=memory.estimated_tokens,
            ),
        )
        self._consume_indexing_budget(1)
        return _success(
            "index_now",
            token_count=memory.estimated_tokens,
            payload={
                "memory_id": memory.memory_id,
                "indexed": True,
                "available_at_ms": memory.metadata.get("available_at_ms"),
            },
        )

    def _delay_index(self, action: DelayIndexAction) -> ActionResult:
        retry_at_ms = self.current_time_ms + action.retry_after_ms
        previous = self.memory_store.require(action.memory_id)
        memory = self.memory_store.delay_index(
            memory_id=action.memory_id,
            retry_after_ms=retry_at_ms,
            reason=action.reason,
            updated_at_ms=self.current_time_ms,
        )
        if previous.indexed:
            self.retrieval_index.delete(
                memory.memory_id,
                available_at_ms=_available_at_ms(self.current_time_ms, "delay_index"),
            )
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
            as_of_ms=action.as_of_ms,
        )
        self.last_search_results = results
        return _success(
            "search_memory",
            token_count=estimate_tokens(action.query_text),
            count=len(results),
            payload={
                "query_id": action.query_id,
                "as_of_ms": action.as_of_ms,
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
        search_results_by_id = {result.memory_id: result for result in self.last_search_results}
        candidates: list[tuple[str, str]] = []
        for memory_id in memory_ids:
            result = search_results_by_id.get(memory_id)
            content = _retrieved_content(result)
            if content:
                candidates.append((memory_id, content))

        if not candidates:
            answer = _ANSWER_ABSTENTION_TEXT
            cited_memory_ids: list[str] = []
        else:
            answer, cited_memory_ids = self._compose_answer(action.query_text, candidates)

        return _success(
            "answer",
            token_count=estimate_tokens(action.query_text) + estimate_tokens(answer),
            count=len(cited_memory_ids),
            payload={
                "query_id": action.query_id,
                "answer": answer,
                "cited_memory_ids": cited_memory_ids,
            },
        )

    def _compose_answer(
        self,
        query_text: str,
        candidates: list[tuple[str, str]],
    ) -> tuple[str, list[str]]:
        """Compose a focused answer from retrieved memories.

        Uses the configured ``answer_llm_client`` when available so the
        candidate answer reflects only the relevant memories, and cites
        exactly the memories used. Falls back to the top-1 retrieved
        memory deterministically on any LLM failure.
        """

        if self.answer_llm_client is not None:
            composed = _llm_compose_answer(
                client=self.answer_llm_client,
                query_text=query_text,
                candidates=candidates,
            )
            if composed is not None:
                answer, llm_cited = composed
                allowed_ids = {memory_id for memory_id, _ in candidates}
                cited = [mid for mid in llm_cited if mid in allowed_ids]
                # Remove duplicate ids while keeping order.
                seen: set[str] = set()
                ordered_cited = [mid for mid in cited if not (mid in seen or seen.add(mid))]
                if not ordered_cited and answer.strip() and answer.strip() != _ANSWER_ABSTENTION_TEXT:
                    # Defensive: a non-abstention answer with no citations is
                    # almost always the LLM hallucinating. Fall back to top-1
                    # so evidence_correct stays honest.
                    return self._top1_answer(candidates)
                return answer.strip(), ordered_cited
        return self._top1_answer(candidates)

    @staticmethod
    def _top1_answer(candidates: list[tuple[str, str]]) -> tuple[str, list[str]]:
        memory_id, content = candidates[0]
        return content, [memory_id]

    def _index_memory(self, memory_id: str, *, available_at_ms: float) -> MemoryRecord:
        memory = self.memory_store.require(memory_id)
        if memory.status != MemoryStatus.ACTIVE:
            raise ValueError(f"cannot index non-active memory: {memory_id}")
        self._validate_source_event_ids(memory.source_event_ids)
        indexed_memory = self.memory_store.set_indexed(
            memory_id,
            True,
            self.current_time_ms,
            metadata_updates={"needs_reindex": False},
        )
        visible_memory = _memory_available_at(indexed_memory, available_at_ms)
        self.retrieval_index.upsert(visible_memory)
        return visible_memory

    def _validate_source_event_ids(self, source_event_ids: list[str]) -> None:
        missing = [
            event_id
            for event_id in dict.fromkeys(source_event_ids)
            if self.raw_event_store.get(event_id) is None
        ]
        if missing:
            raise ValueError(f"unknown source_event_ids: {', '.join(missing)}")

    def _require_storage_budget(self, delta_tokens: int) -> None:
        required = max(0, delta_tokens)
        if required == 0 or self.storage_budget_tokens_remaining is None:
            return
        if required > self.storage_budget_tokens_remaining:
            raise ValueError(
                "storage budget exceeded: "
                f"requires {required} tokens, remaining {self.storage_budget_tokens_remaining}"
            )

    def _consume_storage_budget(self, delta_tokens: int) -> None:
        used = max(0, delta_tokens)
        if used == 0 or self.storage_budget_tokens_remaining is None:
            return
        self.storage_budget_tokens_remaining = max(0, self.storage_budget_tokens_remaining - used)

    def _require_indexing_budget(self, operations: int) -> None:
        if operations <= 0 or self.indexing_budget_operations_remaining is None:
            return
        if operations > self.indexing_budget_operations_remaining:
            raise ValueError(
                "indexing budget exceeded: "
                f"requires {operations} operation(s), remaining {self.indexing_budget_operations_remaining}"
            )

    def _consume_indexing_budget(self, operations: int) -> None:
        if operations <= 0 or self.indexing_budget_operations_remaining is None:
            return
        self.indexing_budget_operations_remaining = max(0, self.indexing_budget_operations_remaining - operations)


__all__ = [
    "BASE_LATENCY_MS",
    "FastMemoryWriteEnv",
    "deterministic_memory_id",
]


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


def _available_at_ms(
    current_time_ms: int,
    action_type: str,
    *,
    token_count: int = 0,
    count: int = 0,
) -> float:
    return float(current_time_ms) + _latency_ms(action_type, token_count=token_count, count=count)


def _memory_available_at(memory: MemoryRecord, available_at_ms: float) -> MemoryRecord:
    metadata = dict(memory.metadata)
    metadata.update(
        {
            "available_at_ms": available_at_ms,
            "indexed_at_ms": available_at_ms,
        }
    )
    return memory.model_copy(update={"metadata": metadata})


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


def _retrieved_content(result: SearchResult | None) -> str | None:
    if result is None:
        return None
    if result.memory is not None:
        return result.memory.content
    return result.content or None


def _llm_compose_answer(
    *,
    client: LLMClient,
    query_text: str,
    candidates: list[tuple[str, str]],
) -> tuple[str, list[str]] | None:
    """Ask the LLM to write a focused answer + cite only the memories it used.

    Returns ``(answer, cited_memory_ids)`` on success, ``None`` if the client
    errors, refuses, or returns a malformed payload. The caller treats
    ``None`` as "fall back to top-1" so a flaky LLM cannot silently degrade
    or invent citations.
    """

    import json

    user_payload: dict[str, object] = {
        "question": query_text,
        "candidate_memories": [
            {
                "memory_id": memory_id,
                "content": content[:_ANSWER_COMPOSE_MAX_PASSAGE_CHARS],
            }
            for memory_id, content in candidates
        ],
    }
    messages = [
        LLMMessage(role="system", content=_ANSWER_COMPOSE_SYSTEM_PROMPT),
        LLMMessage(role="user", content=json.dumps(user_payload, sort_keys=True)),
    ]
    try:
        response = client.complete(
            messages,
            temperature=0.0,
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "compose_answer",
                    "strict": True,
                    "schema": {
                        "type": "object",
                        "additionalProperties": False,
                        "properties": {
                            "answer": {"type": "string"},
                            "cited_memory_ids": {
                                "type": "array",
                                "items": {"type": "string"},
                            },
                        },
                        "required": ["answer", "cited_memory_ids"],
                    },
                },
            },
        )
    except LLMClientError:
        return None
    except Exception:  # pragma: no cover - defensive
        return None

    payload: object = response.parsed_json
    if payload is None:
        try:
            payload = json.loads(response.content or "")
        except (json.JSONDecodeError, TypeError):
            return None
    if not isinstance(payload, dict):
        return None
    answer = payload.get("answer")
    cited_raw = payload.get("cited_memory_ids", [])
    if not isinstance(answer, str):
        return None
    if not isinstance(cited_raw, list):
        return None
    cited: list[str] = []
    for value in cited_raw:
        if isinstance(value, str):
            cited.append(value)
    return answer, cited
