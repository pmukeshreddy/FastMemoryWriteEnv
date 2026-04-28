#!/usr/bin/env python3
"""Regenerate metrics and summary from raw rollout JSONL."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fast_memory_write_env.metrics import (
    extract_run_config,
    read_rollout_jsonl,
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
    score = score_metrics(aggregate)
    output_dir = Path(args.output_dir)
    metrics_path = output_dir / "metrics.csv"
    summary_path = output_dir / "eval_summary.json"
    write_metrics_csv(query_metrics, aggregate, metrics_path)
    write_eval_summary(
        {
            "aggregate_metrics": aggregate.model_dump(mode="json"),
            "score_breakdown": score.model_dump(mode="json"),
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
    print(f"metrics_csv={metrics_path}")
    print(f"eval_summary={summary_path}")


if __name__ == "__main__":
    main()
