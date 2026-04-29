#!/usr/bin/env python3
"""Run a multi-episode streaming evaluation and aggregate metrics.

Each ``sample`` is one full streaming episode: the LLM policy decides
write/update/index/etc actions for the whole event stream, then every
query in the episode is composed and judged independently. ``--samples 20``
runs 20 distinct episodes (different seeds and/or episode_indices, never
repeated), producing roughly 4*20 = 80 queries on the ``small`` mode.

Per-episode artifacts land under ``<output_dir>/sample_<NN>/`` and a
combined aggregate is written at ``<output_dir>/aggregate_summary.json``.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fast_memory_write_env.dataset import (
    SYNTHETIC_DATASET_MODES,
    generate_dataset,
)
from fast_memory_write_env.evaluator import (
    StreamingEvaluator,
    build_failure_diagnostics,
    write_evaluation_outputs,
)
from fast_memory_write_env.llm_client import MockLLMClient, OpenAICompatibleLLMClient
from fast_memory_write_env.metrics import (
    AggregateCounterSnapshot,
    RunConfig,
    aggregate_metrics,
    headline_metrics,
)
from fast_memory_write_env.policies import LLMMemoryWritePolicy
from fast_memory_write_env.rewards import score_metrics
from fast_memory_write_env.schemas import DatasetMode, StreamingEpisode


def _episode_iter(mode: DatasetMode, *, start_seed: int):
    """Yield ``(seed, episode_index, episode)`` tuples without repeats.

    Walks seeds incrementally; for each seed yields every episode in the
    generated dataset (``episode_count`` per mode). Each tuple is unique
    because every (seed, episode_index) pair is hit at most once.
    """

    seed = start_seed
    while True:
        dataset = generate_dataset(mode=mode, seed=seed)
        for episode in dataset.episodes:
            yield seed, episode.metadata.get("episode_index"), episode
        seed += 1


def _merge_counters(parts: list[AggregateCounterSnapshot]) -> AggregateCounterSnapshot:
    """Sum scalar counters and concatenate latency lists across episodes."""

    write_lat: list[float] = []
    index_lat: list[float] = []
    query_lat: list[float] = []
    total_memory = 0
    stale = 0
    duplicate = 0
    storage = 0
    useful = 0
    useful_event = 0
    ignored_useful = 0
    noise_event = 0
    stored_noise = 0
    for c in parts:
        total_memory += c.total_memory_count
        stale += c.stale_memory_count
        duplicate += c.duplicate_memory_count
        storage += c.storage_tokens_used
        useful += c.useful_memory_count
        useful_event += c.useful_event_count
        ignored_useful += c.ignored_useful_event_count
        noise_event += c.noise_event_count
        stored_noise += c.stored_noise_memory_count
        write_lat.extend(c.write_latencies_ms)
        index_lat.extend(c.index_latencies_ms)
        query_lat.extend(c.query_latencies_ms)
    return AggregateCounterSnapshot(
        total_memory_count=total_memory,
        stale_memory_count=stale,
        duplicate_memory_count=duplicate,
        storage_tokens_used=storage,
        useful_memory_count=useful,
        useful_event_count=useful_event,
        ignored_useful_event_count=ignored_useful,
        noise_event_count=noise_event,
        stored_noise_memory_count=stored_noise,
        write_latencies_ms=write_lat,
        index_latencies_ms=index_lat,
        query_latencies_ms=query_lat,
    )


def _extract_counters_from_records(rollout_records) -> AggregateCounterSnapshot:
    counter_records = [r for r in rollout_records if r.record_type == "aggregate_inputs"]
    if not counter_records:
        return AggregateCounterSnapshot()
    return AggregateCounterSnapshot.model_validate(counter_records[-1].payload)


def _evaluate_one(
    *,
    episode: StreamingEpisode,
    use_test_index: bool,
    output_dir: Path,
    sample_slot: int,
    args,
):
    llm_client = MockLLMClient() if args.mock else OpenAICompatibleLLMClient()
    policy = LLMMemoryWritePolicy(llm_client=llm_client)
    run_config = RunConfig(
        dataset_mode=episode.mode.value,
        seed=episode.seed,
        episode_index=episode.metadata.get("episode_index"),
        episode_id=episode.episode_id,
        policy_name=type(policy).__name__,
        llm_client_type=type(llm_client).__name__,
        llm_model=getattr(llm_client, "model", "mock" if args.mock else None),
        llm_base_url=getattr(llm_client, "base_url", None),
        llm_api_key_configured=bool(getattr(llm_client, "api_key", None)),
        backend_type="in_memory_test" if use_test_index else "pinecone",
        pinecone_index_name=None if use_test_index else os.getenv("PINECONE_INDEX_NAME"),
        latency_budget_ms=args.latency_budget_ms,
        storage_budget_tokens_remaining=args.storage_budget,
        indexing_budget_operations_remaining=args.indexing_budget,
        timestamp_utc=datetime.now(timezone.utc).isoformat(),
        command="python3 " + " ".join(sys.argv),
        metadata={
            "mock": args.mock,
            "use_test_index": args.use_test_index,
            "sample_slot": sample_slot,
        },
    )
    episode_output = output_dir / f"sample_{sample_slot:02d}"
    with tempfile.TemporaryDirectory(prefix="fmwe-eval-") as tmpdir:
        evaluator = (
            StreamingEvaluator.with_local_test_index(policy=policy, work_dir=tmpdir, run_config=run_config)
            if use_test_index
            else StreamingEvaluator.with_pinecone(policy=policy, work_dir=tmpdir, run_config=run_config)
        )
        evaluator.latency_budget_ms = args.latency_budget_ms
        evaluator.storage_budget_tokens_remaining = args.storage_budget
        evaluator.indexing_budget_operations_remaining = args.indexing_budget
        result = evaluator.evaluate_episode(episode)
        result = write_evaluation_outputs(result, episode_output)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a multi-episode FastMemoryWriteEnv evaluation.")
    parser.add_argument("--mode", choices=[m.value for m in SYNTHETIC_DATASET_MODES], default=DatasetMode.SMALL.value)
    parser.add_argument(
        "--samples",
        type=int,
        default=20,
        help="Number of distinct episodes to run. Each sample is one full streaming "
        "episode; the runner never repeats a (seed, episode_index) pair.",
    )
    parser.add_argument("--start-seed", type=int, default=7)
    parser.add_argument("--mock", action="store_true")
    parser.add_argument("--use-test-index", action="store_true")
    parser.add_argument("--output-dir", default="results/multi")
    parser.add_argument("--latency-budget-ms", type=int, default=250)
    parser.add_argument("--storage-budget", type=int, default=10_000)
    parser.add_argument("--indexing-budget", type=int, default=3)
    args = parser.parse_args()

    if args.samples < 1:
        raise SystemExit("--samples must be >= 1")

    use_test_index = bool(args.use_test_index or args.mock)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    mode = DatasetMode(args.mode)
    iterator = _episode_iter(mode, start_seed=args.start_seed)

    per_episode_summaries: list[dict] = []
    all_query_metrics = []
    all_counters: list[AggregateCounterSnapshot] = []
    all_diagnostics: list[dict] = []
    total_queries = 0
    seen_keys: set[tuple[int, int | None]] = set()

    for sample_slot in range(args.samples):
        seed, episode_index, episode = next(iterator)
        key = (seed, episode_index)
        if key in seen_keys:
            raise RuntimeError(
                f"refusing to repeat sample (seed={seed}, episode_index={episode_index})"
            )
        seen_keys.add(key)
        result = _evaluate_one(
            episode=episode,
            use_test_index=use_test_index,
            output_dir=output_dir,
            sample_slot=sample_slot,
            args=args,
        )
        ep_queries = len(result.query_metrics)
        total_queries += ep_queries
        all_query_metrics.extend(result.query_metrics)
        all_counters.append(_extract_counters_from_records(result.rollout_records))
        all_diagnostics.append(
            {
                "sample_slot": sample_slot,
                "seed": seed,
                "episode_index": episode_index,
                "episode_id": result.episode_id,
                "diagnostics": build_failure_diagnostics(result.query_metrics, result.rollout_records),
            }
        )
        per_episode_summaries.append(
            {
                "sample_slot": sample_slot,
                "seed": seed,
                "episode_index": episode_index,
                "episode_id": result.episode_id,
                "queries": ep_queries,
                "score": result.score_breakdown.score,
                "metrics": headline_metrics(result.aggregate_metrics),
            }
        )
        print(
            f"[{sample_slot + 1:02d}/{args.samples:02d}] "
            f"seed={seed} idx={episode_index} episode={result.episode_id} "
            f"queries={ep_queries} score={result.score_breakdown.score:.3f} "
            f"answer_success={result.aggregate_metrics.answer_success:.3f}"
        )

    episodes_run = args.samples

    merged_counters = _merge_counters(all_counters)
    merged_aggregate = aggregate_metrics(all_query_metrics, merged_counters)
    merged_score = score_metrics(merged_aggregate)

    failure_stage_totals = {
        "raw_written": 0,
        "memory_written": 0,
        "indexed": 0,
        "retrieved": 0,
        "cited": 0,
        "answered": 0,
    }
    total_failures = 0
    total_action_failures = 0
    total_plan_errors = 0
    for entry in all_diagnostics:
        d = entry["diagnostics"]
        for stage, count in d.get("failure_stage_counts", {}).items():
            failure_stage_totals[stage] = failure_stage_totals.get(stage, 0) + count
        total_failures += d.get("query_failure_count", 0)
        total_action_failures += d.get("action_failure_count", 0)
        total_plan_errors += d.get("policy_plan_error_count", 0)

    summary = {
        "mode": mode.value,
        "samples": args.samples,
        "episodes_run": episodes_run,
        "total_queries": total_queries,
        "unique_sample_keys": [
            {"seed": seed, "episode_index": idx} for seed, idx in sorted(seen_keys)
        ],
        "aggregate": {
            "score": merged_score.score,
            "metrics": headline_metrics(merged_aggregate),
            "answer_success": merged_aggregate.answer_success,
            "answer_correct": merged_aggregate.answer_correct,
            "evidence_correct": merged_aggregate.evidence_correct,
            "memory_precision": merged_aggregate.memory_precision,
            "memory_recall": merged_aggregate.memory_recall,
            "stale_memory_rate": merged_aggregate.stale_memory_rate,
            "stored_noise_rate": merged_aggregate.stored_noise_rate,
            "ignored_useful_fact_rate": merged_aggregate.ignored_useful_fact_rate,
            "storage_tokens_used": merged_aggregate.storage_tokens_used,
        },
        "failure_stage_totals": failure_stage_totals,
        "totals": {
            "query_failure_count": total_failures,
            "action_failure_count": total_action_failures,
            "policy_plan_error_count": total_plan_errors,
        },
        "per_episode": per_episode_summaries,
        "diagnostics_by_episode": all_diagnostics,
    }
    summary_path = output_dir / "aggregate_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    print()
    print("=" * 72)
    print(f"AGGREGATE: samples={args.samples} (distinct episodes) queries={total_queries}")
    print(f"  score                 = {merged_score.score:.3f}")
    print(f"  answer_success        = {merged_aggregate.answer_success:.3f}")
    print(f"  answer_correct        = {merged_aggregate.answer_correct:.3f}")
    print(f"  evidence_correct      = {merged_aggregate.evidence_correct:.3f}")
    print(f"  memory_precision      = {merged_aggregate.memory_precision:.3f}")
    print(f"  memory_recall         = {merged_aggregate.memory_recall:.3f}")
    print(f"  storage_tokens_used   = {merged_aggregate.storage_tokens_used}")
    print(f"  stale_memory_rate     = {merged_aggregate.stale_memory_rate:.3f}")
    print(f"  ignored_useful_fact_rate = {merged_aggregate.ignored_useful_fact_rate:.3f}")
    print(f"  stored_noise_rate     = {merged_aggregate.stored_noise_rate:.3f}")
    print(f"FAILURE STAGES (first-missing across {total_failures} failed queries):")
    for stage, count in failure_stage_totals.items():
        print(f"  {stage:<16} {count}")
    print(f"  action_failures       = {total_action_failures}")
    print(f"  policy_plan_errors    = {total_plan_errors}")
    print(f"summary written to {summary_path}")


if __name__ == "__main__":
    main()
