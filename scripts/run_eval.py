#!/usr/bin/env python3
"""Run a FastMemoryWriteEnv evaluation against LongMemEval data.

LongMemEval is the only supported dataset. The synthetic generator and the
``--mock`` LLM client have been removed; every run uses the real
``OpenAICompatibleLLMClient`` and (by default) the real Pinecone retrieval
backend.
"""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


from fast_memory_write_env.evaluator import StreamingEvaluator, write_evaluation_outputs
from fast_memory_write_env.llm_client import OpenAICompatibleLLMClient
from fast_memory_write_env.longmemeval import load_longmemeval_episodes
from fast_memory_write_env.metrics import RunConfig, subtask_accuracy_breakdown
from fast_memory_write_env.policies import LLMMemoryWritePolicy
from fast_memory_write_env.schemas import DatasetMode


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run FastMemoryWriteEnv evaluation against LongMemEval data."
    )
    parser.add_argument(
        "--longmemeval-path",
        required=True,
        help="Local LongMemEval JSON path.",
    )
    parser.add_argument("--episode-index", type=int, default=0)
    parser.add_argument(
        "--use-test-index",
        action="store_true",
        help=(
            "Use InMemoryIndex instead of Pinecone. The in-memory index is a "
            "real implementation of the retrieval interface, kept only for "
            "local checks when Pinecone credentials are not configured. Real "
            "performance numbers must use Pinecone."
        ),
    )
    parser.add_argument("--output-dir", default="results")
    args = parser.parse_args()

    episodes = load_longmemeval_episodes(args.longmemeval_path)
    if args.episode_index < 0 or args.episode_index >= len(episodes):
        raise SystemExit(f"--episode-index must be between 0 and {len(episodes) - 1}")
    episode = episodes[args.episode_index]

    llm_client = OpenAICompatibleLLMClient()
    policy = LLMMemoryWritePolicy(llm_client=llm_client)
    run_config = RunConfig(
        dataset_mode=episode.mode.value,
        seed=episode.seed,
        episode_index=episode.metadata.get("item_index", args.episode_index),
        episode_id=episode.episode_id,
        policy_name=type(policy).__name__,
        llm_client_type=type(llm_client).__name__,
        llm_model=getattr(llm_client, "model", None),
        llm_base_url=getattr(llm_client, "base_url", None),
        llm_api_key_configured=bool(getattr(llm_client, "api_key", None)),
        backend_type="in_memory_test" if args.use_test_index else "pinecone",
        pinecone_index_name=None if args.use_test_index else os.getenv("PINECONE_INDEX_NAME"),
        latency_budget_ms=250,
        storage_budget_tokens_remaining=10_000,
        indexing_budget_operations_remaining=3,
        timestamp_utc=datetime.now(timezone.utc).isoformat(),
        command="python3 " + " ".join(sys.argv),
        metadata={
            "use_test_index": args.use_test_index,
            "output_dir": args.output_dir,
            "longmemeval_path": args.longmemeval_path,
            "longmemeval_question_id": episode.metadata.get("longmemeval_question_id"),
            "evidence_label_source": episode.metadata.get("evidence_label_source"),
        },
    )
    with tempfile.TemporaryDirectory(prefix="fmwe-eval-") as tmpdir:
        evaluator = (
            StreamingEvaluator.with_local_test_index(policy=policy, work_dir=tmpdir, run_config=run_config)
            if args.use_test_index
            else StreamingEvaluator.with_pinecone(policy=policy, work_dir=tmpdir, run_config=run_config)
        )
        result = evaluator.evaluate_episode(episode)
        result = write_evaluation_outputs(result, Path(args.output_dir))

    print(f"episode={result.episode_id}")
    print(f"queries={len(result.query_metrics)}")
    print(f"score={result.score_breakdown.score:.3f}")
    print(
        f"answer_success={result.aggregate_metrics.answer_success:.3f} "
        f"({result.aggregate_metrics.answer_success:.1%})"
    )
    print("subtask_accuracy:")
    for question_type, payload in subtask_accuracy_breakdown(result.query_metrics).items():
        print(
            f"  {question_type}: {payload['accuracy']:.3f} "
            f"({payload['correct']}/{payload['total']})"
        )
    print(f"memory_precision={result.aggregate_metrics.memory_precision:.3f}")
    print(f"memory_recall={result.aggregate_metrics.memory_recall:.3f}")
    print(f"storage_tokens_used={result.aggregate_metrics.storage_tokens_used}")
    print(f"total_memory_count={result.aggregate_metrics.total_memory_count}")
    print(f"stale_memory_rate={result.aggregate_metrics.stale_memory_rate:.3f}")
    print(f"raw_rollouts={result.output_paths['raw_rollouts']}")
    print(f"metrics_csv={result.output_paths['metrics_csv']}")
    print(f"eval_summary={result.output_paths['eval_summary']}")
    print(f"predictions_jsonl={result.output_paths['predictions_jsonl']}")


if __name__ == "__main__":
    main()
