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
import concurrent.futures
import json
import os
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tqdm.auto import tqdm

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
from fast_memory_write_env.longmemeval import load_longmemeval_episodes
from fast_memory_write_env.metrics import (
    AggregateCounterSnapshot,
    RunConfig,
    aggregate_metrics,
    headline_metrics,
)
from fast_memory_write_env.policies import LLMMemoryWritePolicy
from fast_memory_write_env.rewards import score_metrics
from fast_memory_write_env.schemas import DatasetMode, StreamingEpisode


def _truncate_episode_events(episode: StreamingEpisode, *, max_events: int) -> StreamingEpisode:
    """Cap the number of event items in an episode while preserving every query.

    LongMemEval rows expand to thousands of turn-events because every chat
    turn becomes one streaming event. For honest scaled evaluation that
    finishes in finite time, callers can cap the per-sample event count.
    Queries are always preserved (skipping them would invalidate scoring);
    if any required ``supporting_event_id`` is dropped, retrieval simply
    cannot find it and the failure surfaces honestly in the metrics.
    """

    if max_events <= 0:
        return episode
    new_stream = []
    kept_events = 0
    for item in episode.stream:
        if item.item_type == "event":
            if kept_events >= max_events:
                continue
            kept_events += 1
        new_stream.append(item)
    if len(new_stream) == len(episode.stream):
        return episode
    return episode.model_copy(update={"stream": new_stream})


def _synthetic_episode_iter(mode: DatasetMode, *, start_seed: int):
    """Yield ``(seed, episode_index, episode)`` synthetic tuples without repeats.

    Note: the synthetic generator uses one fixed story template per episode
    and only varies entity names / noise / ordering across seeds. For genuine
    question diversity use ``--dataset longmemeval`` instead.
    """

    seed = start_seed
    while True:
        dataset = generate_dataset(mode=mode, seed=seed)
        for episode in dataset.episodes:
            yield seed, episode.metadata.get("episode_index"), episode
        seed += 1


def _longmemeval_episode_iter(path: str, *, start_index: int):
    """Yield ``(seed, episode_index, episode)`` from a LongMemEval JSON file.

    Each LongMemEval row is its own real conversation with its own diverse
    question. ``seed`` is mirrored from the row index so the per-sample
    print stays consistent with the synthetic path; uniqueness is enforced
    by the index. The iterator stops once the file is exhausted, so the
    runner will fail loudly if you ask for more samples than the file has
    rather than silently repeat.
    """

    episodes = load_longmemeval_episodes(path)
    if start_index < 0 or start_index >= len(episodes):
        raise SystemExit(
            f"--start-seed {start_index} out of range for LongMemEval file "
            f"(loaded {len(episodes)} rows)"
        )
    for index in range(start_index, len(episodes)):
        episode = episodes[index]
        yield index, episode.metadata.get("longmemeval_question_id") or index, episode


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
    show_inner = args.concurrent_samples == 1
    namespace = f"fmwe-multi-sample-{sample_slot:04d}"
    with tempfile.TemporaryDirectory(prefix="fmwe-eval-") as tmpdir:
        if use_test_index:
            evaluator = StreamingEvaluator.with_local_test_index(
                policy=policy, work_dir=tmpdir, run_config=run_config
            )
        else:
            evaluator = StreamingEvaluator.with_pinecone(
                policy=policy,
                work_dir=tmpdir,
                run_config=run_config,
                namespace=namespace,
            )
        evaluator.latency_budget_ms = args.latency_budget_ms
        evaluator.storage_budget_tokens_remaining = args.storage_budget
        evaluator.indexing_budget_operations_remaining = args.indexing_budget
        evaluator.queue_drain_timeout_seconds = args.queue_drain_timeout_seconds
        evaluator.worker_stop_timeout_seconds = args.worker_stop_timeout_seconds
        evaluator.show_inner_progress = show_inner
        evaluator.debug_timing = args.debug_timing
        evaluator.write_worker_concurrency = args.write_worker_concurrency
        try:
            result = evaluator.evaluate_episode(episode)
            result = write_evaluation_outputs(result, episode_output)
        finally:
            # Clean up the per-sample namespace so concurrent samples never
            # see each other's vectors and a sequential run does not pile
            # up state in Pinecone between iterations.
            cleanup = getattr(evaluator.env.retrieval_index, "vector_index", None)
            cleanup_target = cleanup if cleanup is not None else evaluator.env.retrieval_index
            cleanup_fn = getattr(cleanup_target, "cleanup_namespace", None)
            if callable(cleanup_fn):
                try:
                    cleanup_fn()
                except Exception:
                    pass
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a multi-episode FastMemoryWriteEnv evaluation.")
    parser.add_argument(
        "--dataset",
        choices=["synthetic", "longmemeval"],
        default="synthetic",
        help="synthetic: use the project's templated generator (test fixture; "
        "different seeds only swap entity names). longmemeval: use real "
        "human-authored conversations from a LongMemEval JSON file - the "
        "honest test for diverse questions.",
    )
    parser.add_argument(
        "--longmemeval-path",
        help="Path to a LongMemEval JSON file (required when --dataset=longmemeval).",
    )
    parser.add_argument("--mode", choices=[m.value for m in SYNTHETIC_DATASET_MODES], default=DatasetMode.SMALL.value)
    parser.add_argument(
        "--samples",
        type=int,
        default=20,
        help="Number of distinct episodes to run. Each sample is one full streaming "
        "episode; the runner never repeats a sample key.",
    )
    parser.add_argument(
        "--start-seed",
        type=int,
        default=7,
        help="Starting seed (synthetic) or starting row index (longmemeval).",
    )
    parser.add_argument("--mock", action="store_true")
    parser.add_argument("--use-test-index", action="store_true")
    parser.add_argument("--output-dir", default="results/multi")
    parser.add_argument("--latency-budget-ms", type=int, default=250)
    parser.add_argument("--storage-budget", type=int, default=10_000)
    parser.add_argument("--indexing-budget", type=int, default=3)
    parser.add_argument(
        "--queue-drain-timeout-seconds",
        type=float,
        default=1800.0,
        help="How long the evaluator waits for the memory-write worker to "
        "drain queued events before each query. LongMemEval rows can have "
        "thousands of turn-events; raise this if a sample aborts.",
    )
    parser.add_argument(
        "--worker-stop-timeout-seconds",
        type=float,
        default=1800.0,
        help="How long the evaluator waits for the worker thread to exit "
        "between samples. Should usually match --queue-drain-timeout-seconds.",
    )
    parser.add_argument(
        "--max-events-per-sample",
        type=int,
        default=0,
        help="If >0, cap the number of event items streamed per sample. "
        "Queries are always preserved. Use this to keep wall-clock and "
        "OpenAI cost bounded on LongMemEval (~50-100 events is enough for "
        "a fast 20-sample sanity test; 0 means use the full episode).",
    )
    parser.add_argument(
        "--concurrent-samples",
        type=int,
        default=4,
        help="Number of samples to run in parallel via a thread pool. "
        "Samples are independent (own env, own SQLite tmpdir, own Pinecone "
        "namespace) so this is real parallelism, bounded by your OpenAI "
        "RPM/TPM. Set to 1 to disable parallelism.",
    )
    parser.add_argument(
        "--debug-timing",
        action="store_true",
        help="Print per-event and per-query wall-clock timings (decide / "
        "compose / judge) to surface which step is the bottleneck.",
    )
    parser.add_argument(
        "--write-worker-concurrency",
        type=int,
        default=4,
        help="Number of memory-write worker threads inside each sample. "
        "Each thread pulls events from the queue and runs the policy "
        "LLM call concurrently (commits serialise through env._lock). "
        "Set to 1 for strict ordering; >1 trades exact ordering of the "
        "active_memories snapshot for parallel LLM throughput.",
    )
    args = parser.parse_args()

    if args.samples < 1:
        raise SystemExit("--samples must be >= 1")
    if args.concurrent_samples < 1:
        raise SystemExit("--concurrent-samples must be >= 1")
    if args.dataset == "longmemeval" and not args.longmemeval_path:
        raise SystemExit("--longmemeval-path is required when --dataset=longmemeval")

    use_test_index = bool(args.use_test_index or args.mock)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    mode = DatasetMode(args.mode)
    if args.dataset == "longmemeval":
        iterator = _longmemeval_episode_iter(args.longmemeval_path, start_index=args.start_seed)
    else:
        iterator = _synthetic_episode_iter(mode, start_seed=args.start_seed)

    per_episode_summaries: list[dict] = []
    all_query_metrics = []
    all_counters: list[AggregateCounterSnapshot] = []
    all_diagnostics: list[dict] = []
    total_queries = 0
    seen_keys: set[tuple[int, int | None]] = set()

    # Materialize the sample plan up front so seed/index pairs stay
    # deterministic regardless of completion order under concurrency.
    plan: list[tuple[int, Any, StreamingEpisode, int]] = []
    for sample_slot in range(args.samples):
        seed, episode_index, episode = next(iterator)
        key = (seed, episode_index)
        if key in seen_keys:
            raise RuntimeError(
                f"refusing to repeat sample (seed={seed}, episode_index={episode_index})"
            )
        seen_keys.add(key)
        if args.max_events_per_sample > 0:
            episode = _truncate_episode_events(
                episode, max_events=args.max_events_per_sample
            )
        plan.append((seed, episode_index, episode, sample_slot))

    samples_bar = tqdm(
        total=args.samples,
        desc="samples",
        unit="sample",
        leave=True,
        dynamic_ncols=True,
        position=0,
    )

    def _run_one(plan_entry):
        seed, episode_index, episode, sample_slot = plan_entry
        result = _evaluate_one(
            episode=episode,
            use_test_index=use_test_index,
            output_dir=output_dir,
            sample_slot=sample_slot,
            args=args,
        )
        return seed, episode_index, sample_slot, result

    if args.concurrent_samples == 1:
        completed = (_run_one(entry) for entry in plan)
    else:
        executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=args.concurrent_samples,
            thread_name_prefix="fmwe-sample",
        )
        futures = [executor.submit(_run_one, entry) for entry in plan]
        completed = (f.result() for f in concurrent.futures.as_completed(futures))

    completion_index = 0
    for seed, episode_index, sample_slot, result in completed:
        completion_index += 1
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
        samples_bar.set_postfix(
            slot=sample_slot,
            seed=seed,
            idx=episode_index,
            score=f"{result.score_breakdown.score:.2f}",
            success=f"{result.aggregate_metrics.answer_success:.2f}",
        )
        samples_bar.update(1)
        tqdm.write(
            f"[{completion_index:02d}/{args.samples:02d}] slot={sample_slot:02d} "
            f"seed={seed} idx={episode_index} episode={result.episode_id} "
            f"queries={ep_queries} score={result.score_breakdown.score:.3f} "
            f"answer_success={result.aggregate_metrics.answer_success:.3f}"
        )

    samples_bar.close()
    if args.concurrent_samples > 1:
        executor.shutdown(wait=True)
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
