#!/usr/bin/env python3
"""Regenerate metrics and summary from raw rollout JSONL."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fast_memory_write_env.metrics import (
    extract_run_config,
    headline_metrics,
    read_rollout_jsonl,
    subtask_accuracy_breakdown,
    summarize_rollout_records,
    write_eval_summary,
    write_metrics_csv,
)
from fast_memory_write_env.rewards import score_metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate existing raw rollout records.")
    parser.add_argument("raw_rollouts", nargs="?", default="results/raw_rollouts.jsonl")
    parser.add_argument("--output-dir", default="results")
    args = parser.parse_args()

    records = read_rollout_jsonl(args.raw_rollouts)
    run_config = extract_run_config(records)
    query_metrics, aggregate = summarize_rollout_records(records)
    dataset_mode = run_config.dataset_mode if run_config is not None else None
    subtasks = (
        subtask_accuracy_breakdown(query_metrics, rollout_records=records)
        if dataset_mode == "longmemeval"
        else None
    )
    score = score_metrics(aggregate, subtask_accuracies=subtasks)
    output_dir = Path(args.output_dir)
    metrics_path = output_dir / "metrics.csv"
    summary_path = output_dir / "eval_summary.json"
    write_metrics_csv(query_metrics, aggregate, metrics_path)
    write_eval_summary(
        {
            "primary_metric": {
                "name": "answer_success",
                "value": aggregate.answer_success,
            },
            "metrics": headline_metrics(
                aggregate,
                dataset_mode=dataset_mode,
                query_metrics=query_metrics,
                rollout_records=records,
            ),
            "score": score.score,
            "run_config": run_config.model_dump(mode="json") if run_config is not None else None,
            "counts": {
                "rollout_records": len(records),
                "query_metrics": len(query_metrics),
            },
            "output_paths": {
                "raw_rollouts": str(args.raw_rollouts),
                "metrics_csv": str(metrics_path),
                "eval_summary": str(summary_path),
            },
        },
        summary_path,
    )
    print(f"query_metrics={len(query_metrics)}")
    print(f"score={score.score:.3f}")
    print(f"answer_success={aggregate.answer_success:.3f} ({aggregate.answer_success:.1%})")
    if dataset_mode == "longmemeval":
        print("subtask_accuracy:")
        for question_type, payload in (subtasks or {}).items():
            print(
                f"  {question_type}: {payload['accuracy']:.3f} "
                f"({payload['correct']}/{payload['total']})"
            )
    else:
        print(f"time_to_useful_memory={aggregate.time_to_useful_memory}")
    print(f"memory_precision={aggregate.memory_precision:.3f}")
    print(f"memory_recall={aggregate.memory_recall:.3f}")
    print(f"storage_tokens_used={aggregate.storage_tokens_used}")
    print(f"total_memory_count={aggregate.total_memory_count}")
    print(f"stale_memory_rate={aggregate.stale_memory_rate:.3f}")
    print(f"metrics_csv={metrics_path}")
    print(f"eval_summary={summary_path}")


if __name__ == "__main__":
    main()
