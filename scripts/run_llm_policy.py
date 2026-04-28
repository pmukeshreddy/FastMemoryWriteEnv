#!/usr/bin/env python3
"""Run LLMMemoryWritePolicy over a small streaming episode."""

from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fast_memory_write_env.actions import AnswerAction, SearchMemoryAction, StoreRawAction
from fast_memory_write_env.config import load_pinecone_config
from fast_memory_write_env.dataset import generate_episode
from fast_memory_write_env.env import FastMemoryWriteEnv
from fast_memory_write_env.in_memory_index import InMemoryIndex
from fast_memory_write_env.llm_client import MockLLMClient, OpenAICompatibleLLMClient
from fast_memory_write_env.pinecone_index import PineconeIndex
from fast_memory_write_env.policies import LLMMemoryWritePolicy
from fast_memory_write_env.schemas import DatasetMode, RawEvent
from fast_memory_write_env.stores import MemoryStore, RawEventStore


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Phase 3 LLM memory-write policy.")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--mock", action="store_true", help="Use deterministic MockLLMClient.")
    parser.add_argument(
        "--use-test-index",
        action="store_true",
        help="Use InMemoryIndex for a local smoke run. Real runs should use Pinecone.",
    )
    args = parser.parse_args()

    with tempfile.TemporaryDirectory(prefix="fmwe-policy-") as tmpdir:
        env = FastMemoryWriteEnv(
            raw_event_store=RawEventStore(Path(tmpdir) / "raw.sqlite"),
            memory_store=MemoryStore(Path(tmpdir) / "memory.sqlite"),
            retrieval_index=_build_index(use_test_index=args.use_test_index),
        )
        policy = LLMMemoryWritePolicy(
            llm_client=MockLLMClient() if args.mock else OpenAICompatibleLLMClient(),
        )
        episode = generate_episode(DatasetMode.SMALL, seed=args.seed, episode_index=0)
        recent_events: list[RawEvent] = []
        action_count = 0
        query_count = 0

        print(f"episode={episode.episode_id} items={len(episode.stream)}")
        for item in episode.stream:
            env.current_time_ms = item.timestamp_ms
            if item.item_type == "event":
                event = item.event
                raw_result = env.execute_action(StoreRawAction(event=event))
                actions = policy.decide(
                    new_event=event,
                    active_memories=env.memory_store.list_active(),
                    recent_events=recent_events[-5:],
                    latency_budget_ms=250,
                    storage_budget_tokens_remaining=10_000,
                    indexing_budget_operations_remaining=3,
                )
                results = env.execute_actions(list(actions))
                recent_events.append(event)
                action_count += len(results)
                print(
                    f"event {event.event_id} {event.category.value}: "
                    f"raw={raw_result.success} actions={[result.action_type.value for result in results]}"
                )
            else:
                query = item.query
                search = env.execute_action(
                    SearchMemoryAction(query_id=query.query_id, query_text=query.text, top_k=3)
                )
                retrieved_ids = [
                    hit["memory_id"]
                    for hit in search.payload.get("results", [])
                    if isinstance(hit, dict) and "memory_id" in hit
                ]
                answer = env.execute_action(
                    AnswerAction(
                        query_id=query.query_id,
                        query_text=query.text,
                        retrieved_memory_ids=retrieved_ids,
                    )
                )
                query_count += 1
                print(
                    f"query {query.query_id}: hits={len(retrieved_ids)} "
                    f"cites={answer.payload.get('cited_memory_ids', [])}"
                )

        print(
            "summary: "
            f"raw_events={env.raw_event_store.count()} "
            f"memories={len(env.memory_store.list_all())} "
            f"policy_actions={action_count} "
            f"queries={query_count}"
        )


def _build_index(*, use_test_index: bool):
    if use_test_index:
        return InMemoryIndex()
    config = load_pinecone_config(required=True)
    assert config is not None
    return PineconeIndex(config)


if __name__ == "__main__":
    main()
