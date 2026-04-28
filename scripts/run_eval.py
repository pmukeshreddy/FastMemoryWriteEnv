#!/usr/bin/env python3
"""Run a Phase 4 streaming evaluation."""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fast_memory_write_env.dataset import SYNTHETIC_DATASET_MODES, generate_episode
from fast_memory_write_env.evaluator import StreamingEvaluator, write_evaluation_outputs
from fast_memory_write_env.llm_client import MockLLMClient, OpenAICompatibleLLMClient
from fast_memory_write_env.longmemeval import load_longmemeval_episodes
from fast_memory_write_env.metrics import RunConfig
from fast_memory_write_env.policies import LLMMemoryWritePolicy
from fast_memory_write_env.schemas import DatasetMode


def main() -> None:
    parser = argparse.ArgumentParser(description="Run FastMemoryWriteEnv evaluation.")
    parser.add_argument("--dataset-format", choices=["synthetic", "longmemeval"], default="synthetic")
    parser.add_argument("--dataset-path", help="Local LongMemEval JSON path when --dataset-format=longmemeval.")
    parser.add_argument("--mode", choices=[mode.value for mode in SYNTHETIC_DATASET_MODES], default=DatasetMode.SMALL.value)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--episode-index", type=int, default=0)
    parser.add_argument("--mock", action="store_true", help="Use MockLLMClient.")
    parser.add_argument("--use-test-index", action="store_true", help="Use InMemoryIndex instead of Pinecone.")
    parser.add_argument("--output-dir", default="results")
    args = parser.parse_args()

    if args.dataset_format == "longmemeval":
        if not args.dataset_path:
            raise SystemExit("--dataset-path is required when --dataset-format=longmemeval")
        episodes = load_longmemeval_episodes(args.dataset_path)
        if args.episode_index < 0 or args.episode_index >= len(episodes):
            raise SystemExit(f"--episode-index must be between 0 and {len(episodes) - 1}")
        episode = episodes[args.episode_index]
    else:
        episode = generate_episode(DatasetMode(args.mode), seed=args.seed, episode_index=args.episode_index)
    llm_client = MockLLMClient() if args.mock else OpenAICompatibleLLMClient()
    policy = LLMMemoryWritePolicy(
        llm_client=llm_client,
    )
    use_test_index = bool(args.use_test_index or args.mock)
    run_config = RunConfig(
        dataset_mode=episode.mode.value,
        seed=episode.seed,
        episode_index=episode.metadata.get("item_index", args.episode_index),
        episode_id=episode.episode_id,
        policy_name=type(policy).__name__,
        llm_client_type=type(llm_client).__name__,
        llm_model=getattr(llm_client, "model", "mock" if args.mock else None),
        llm_base_url=getattr(llm_client, "base_url", None),
        llm_api_key_configured=bool(getattr(llm_client, "api_key", None)),
        backend_type="in_memory_test" if use_test_index else "pinecone",
        pinecone_index_name=None if use_test_index else os.getenv("PINECONE_INDEX_NAME"),
        latency_budget_ms=250,
        storage_budget_tokens_remaining=10_000,
        indexing_budget_operations_remaining=3,
        timestamp_utc=datetime.now(timezone.utc).isoformat(),
        command="python3 " + " ".join(sys.argv),
        metadata={
            "mock": args.mock,
            "use_test_index": args.use_test_index,
            "output_dir": args.output_dir,
            "dataset_format": args.dataset_format,
            "dataset_path": args.dataset_path,
            "longmemeval_question_id": episode.metadata.get("longmemeval_question_id"),
            "evidence_label_source": episode.metadata.get("evidence_label_source"),
        },
    )
    with tempfile.TemporaryDirectory(prefix="fmwe-eval-") as tmpdir:
        evaluator = (
            StreamingEvaluator.with_local_test_index(policy=policy, work_dir=tmpdir, run_config=run_config)
            if use_test_index
            else StreamingEvaluator.with_pinecone(policy=policy, work_dir=tmpdir, run_config=run_config)
        )
        result = evaluator.evaluate_episode(episode)
        result = write_evaluation_outputs(result, Path(args.output_dir))

    print(f"episode={result.episode_id}")
    print(f"queries={len(result.query_metrics)}")
    print(f"score={result.score_breakdown.score:.3f}")
    print(f"answer_success={result.aggregate_metrics.answer_success:.3f}")
    print(f"time_to_useful_memory={result.aggregate_metrics.time_to_useful_memory}")
    print(f"raw_rollouts={result.output_paths['raw_rollouts']}")
    print(f"metrics_csv={result.output_paths['metrics_csv']}")
    print(f"eval_summary={result.output_paths['eval_summary']}")
    print(f"predictions_jsonl={result.output_paths['predictions_jsonl']}")


if __name__ == "__main__":
    main()
