# Example Failure Modes

FastMemoryWriteEnv is designed to expose failures in the memory write path while continuous data keeps arriving.

Representative failure modes include:

- A useful fact is ignored, causing `ignored_useful_fact_rate` to rise and later queries to miss required facts.
- A memory is written but indexing is delayed too long, so `time_to_indexed_memory`, `time_to_retrieved_memory`, and `time_to_useful_memory` are missing or slow.
- A stale memory remains active after a contradiction, increasing `stale_memory_rate` and risking incorrect answers.
- Duplicate or noisy events are stored as durable memories, increasing `duplicate_memory_rate`, `stored_noise_rate`, and storage cost.
- An answer contains plausible text but cites memories that do not support the query gold evidence, making `evidence_correct` false.

These failures are visible in:

- `results/mock_run/raw_rollouts.jsonl` for action-level traces.
- `results/mock_run/metrics.csv` for per-query timing and correctness.
- `results/mock_run/eval_summary.json` for aggregate metrics and score breakdown.

The point of the environment is to make these write-path failures measurable, not to hide them behind a final answer string.
