#!/usr/bin/env python3
"""Run LLMMemoryWritePolicy over a small streaming episode."""

from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fast_memory_write_env.dataset import generate_episode
from fast_memory_write_env.evaluator import StreamingEvaluator
from fast_memory_write_env.llm_client import MockLLMClient, OpenAICompatibleLLMClient
from fast_memory_write_env.policies import LLMMemoryWritePolicy
from fast_memory_write_env.schemas import DatasetMode


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
        policy = LLMMemoryWritePolicy(
            llm_client=MockLLMClient() if args.mock else OpenAICompatibleLLMClient(),
        )
        episode = generate_episode(DatasetMode.SMALL, seed=args.seed, episode_index=0)
        evaluator = (
            StreamingEvaluator.with_local_test_index(policy=policy, work_dir=tmpdir)
            if args.use_test_index
            else StreamingEvaluator.with_pinecone(policy=policy, work_dir=tmpdir)
        )
        result = evaluator.evaluate_episode(episode)

        print(f"episode={episode.episode_id} items={len(episode.stream)}")
        for record in result.rollout_records:
            if record.record_type == "queue" and record.payload.get("queue_event") == "completed":
                print(f"queue completed {record.payload['queue_item_id']}")
            if record.record_type == "query":
                answer = record.payload.get("answer", {})
                retrieved = record.payload.get("retrieved_memory_ids", [])
                print(
                    f"query {record.payload['query']['query_id']}: "
                    f"hits={len(retrieved)} cites={answer.get('cited_memory_ids', [])}"
                )

        print(
            "summary: "
            f"raw_events={evaluator.env.raw_event_store.count()} "
            f"memories={len(evaluator.env.memory_store.list_all())} "
            f"policy_actions={sum(1 for record in result.rollout_records if record.record_type == 'action' and 'proposed_action' in record.payload)} "
            f"queries={len(result.query_metrics)}"
        )


if __name__ == "__main__":
    main()
