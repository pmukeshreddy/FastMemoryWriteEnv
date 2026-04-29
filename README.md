# FastMemoryWriteEnv

FastMemoryWriteEnv is a production-shaped streaming memory-write environment for evaluating how quickly and accurately an LLM policy can turn continuous incoming data into useful searchable memory while data keeps arriving.

The project centers on the write path of memory systems: deciding what should become durable memory while new events continue to arrive.

## 1. Project motive

Modern memory systems often focus on retrieval after data has already been stored. FastMemoryWriteEnv focuses on the earlier and harder moment: incoming data is continuous, budgets are limited, and the system must decide what to write, update, ignore, compress, mark stale, or index now.

The environment measures whether those decisions make later queries answerable quickly and correctly. The core metric is not just whether information can eventually be found, but how long it takes for an incoming event to become useful searchable memory.

## 2. Why this is not generic RAG

This project is not a generic RAG benchmark, a static document QA setup, or a comparison of many hand-written memory strategies.

The main system is one LLM memory-write policy. Retrieval and answering are present because they test whether write decisions were useful, but the central object of study is the memory write path under latency, storage, indexing, duplicate, noise, and freshness pressure.

The minimal baselines, `NoMemoryBaseline`, `StoreEverythingBaseline`, and `OraclePolicy`, exist only as sanity checks and bounds.

## 3. Main pipeline

```text
Continuous data stream
    -> fast raw write
    -> memory-write queue
    -> LLMMemoryWritePolicy
    -> validate memory actions
    -> execute actions against SQLite/Pinecone
    -> future queries arrive while data is still coming
    -> retrieve memory + answer
    -> score speed, usefulness, freshness, and storage cost
```

The environment validates and executes actions, records timings, and applies the action side effects. The LLM policy proposes actions only.

The memory-write queue is processed by a background worker during evaluation. Raw event ingestion writes to SQLite and enqueues work; the worker independently pulls queued events, calls `LLMMemoryWritePolicy`, and executes validated memory actions while later stream items can continue to arrive.

## 4. LLMMemoryWritePolicy

`LLMMemoryWritePolicy` receives:

- new raw event
- current active memories
- recent event context
- latency budget
- storage budget
- indexing budget

It calls an internal `LLMClient` abstraction and expects structured JSON actions. All actions are validated with Pydantic before the environment can execute them. Invalid JSON or invalid action payloads trigger retry/repair.

The policy prompt uses a label-hidden event view. Evaluator-only labels such as event category, priority, gold facts, duplicate/contradiction links, stale labels, and memory `fact_ids` are not shown to `LLMMemoryWritePolicy`. If an LLM response includes `fact_ids`, the policy strips them before execution; scoring derives hidden evidence from source event IDs.

Implemented clients:

- `MockLLMClient`: deterministic local test client with no API key.
- `OpenAICompatibleLLMClient`: calls OpenAI-compatible chat completions.

The policy does not mutate SQLite stores and does not call Pinecone directly.

## 5. Memory actions

The memory-write policy can propose:

- `write_memory`
- `update_memory`
- `mark_stale`
- `ignore_event`
- `compress_memory`
- `index_now`
- `delay_index`

The environment also supports execution actions for raw writes, search, and deterministic answer generation:

- `store_raw`
- `search_memory`
- `answer`

Every executed action returns a structured result with success state, action type, simulated latency, storage token delta, error, and payload. Memory actions that cite `source_event_ids` are rejected unless every cited event has already been raw-written, and evaluator runs enforce remaining storage and indexing budgets.

## 6. Pinecone backend

Pinecone is the real retrieval backend for real runs.

Required Pinecone environment variables:

- `PINECONE_API_KEY`
- `PINECONE_INDEX_NAME`
- `PINECONE_CLOUD`
- `PINECONE_REGION`

`InMemoryIndex` exists only as a unit-test and mock-run fake. It is useful for reproducible local checks, but it is not the production retrieval path.

Memory payloads are written to the retrieval backend only when they are indexed or when an already-indexed memory is updated. Delayed or unindexed memories stay in SQLite until an indexing action succeeds; they are not preloaded into Pinecone as hidden future search state. The deterministic test index keeps versioned entries with `available_at_ms`, so query-time search can return the latest memory version available at the query timestamp instead of the latest mutation that happened later in the rollout.

The current index vectorization helper is deterministic and prototype-oriented so local tests and rollouts are reproducible. Pinecone is the real backend, but production retrieval semantics require configuring or extending the embedding path for the target deployment; do not treat the deterministic hashed vectors as a production embedding model.

## 7. Dataset modes and real benchmarks

The built-in synthetic dataset simulates streaming episodes, not static QA pairs. It is for tests, local smoke runs, and controlled debugging.

Synthetic modes:

- `small`: fast tests and local smoke runs.
- `medium`: larger local checks.
- `long`: longer synthetic stress checks.

Generated streams include useful facts, low-value noise, duplicates, contradictions, stale updates, urgent facts, multiple users/entities, far-apart evidence, and storage/indexing pressure.

Generate a dataset JSON file:

```bash
python3 scripts/generate_dataset.py --mode small --seed 7 --output results/dataset_small.json
```

Serious benchmark runs should use imported labeled streams. The first supported external adapter is LongMemEval from a local JSON file; the repo does not auto-download benchmark data. Imported LongMemEval events use neutral queue priority so evidence labels do not affect processing order.

```bash
python3 scripts/run_eval.py \
  --dataset-format longmemeval \
  --dataset-path data/longmemeval_s_cleaned.json \
  --episode-index 0 \
  --output-dir results/lme_run
```

Use `--mock --use-test-index` only for local adapter checks, not for real performance claims.

## 8. Metrics

For synthetic streaming episodes, the write-path timing diagnostic is
`time_to_useful_memory`:

```text
event arrives
-> raw event written
-> useful memory written/updated
-> memory indexed
-> memory retrieved by a future query
-> answer uses it correctly
```

`time_to_useful_memory` is `null` unless every required fact completes the full chain: event arrival -> raw write -> memory write/update -> index -> retrieval -> successful answer. Query retrieval is causally filtered: a query at time `T` can only retrieve memory/index versions whose simulated index availability time is at or before `T`. If a memory update completes after `T`, the query still sees the older indexed version that was available at `T`.

For LongMemEval runs, `time_to_useful_memory` is not a headline metric because
it depends heavily on where the labeled evidence appears in the stream. The
LongMemEval headline scorecard follows the usual benchmark framing:

- `answer_success`
- sub-task accuracy by `question_type`
- `memory_precision`
- `memory_recall`
- `storage_tokens_used`
- `total_memory_count`
- `stale_memory_rate`

Answer success is evidence-based:

```text
answer_correct = answer text satisfies required gold fact strings
evidence_correct = cited memory evidence supports required gold facts/events
answer_success = answer_correct AND evidence_correct
```

Evidence correctness uses hidden evaluator labels keyed by memory `source_event_ids`; it does not require stored memories to carry `fact_ids`. Exact/contains matching is only debug information. The deterministic verifier normalizes the required fact strings and answer text to make the scoring transparent; it is intentionally conservative and not a claim of full semantic NLU. For LongMemEval, `predictions.jsonl` is also written so the official evaluator can be run separately.

## 9. How to run with mock local test mode

Install dependencies, then run tests:

```bash
python3 -m pip install -e ".[dev]"
```

```bash
pytest -q
```

Run a reproducible mock evaluation using the test fake index:

```bash
python3 scripts/run_eval.py --mock --use-test-index --output-dir results/mock_run
```

Regenerate metrics and summary from the raw rollout log:

```bash
python3 scripts/evaluate_results.py results/mock_run/raw_rollouts.jsonl --output-dir results/mock_summary
```

For a smaller policy/action smoke run:

```bash
python3 scripts/run_llm_policy.py --mock --use-test-index
```

## 10. How to run with OpenAI-compatible LLM

Set OpenAI-compatible environment variables:

```bash
export OPENAI_API_KEY=...
export OPENAI_MODEL=gpt-4o-mini
# Optional:
export OPENAI_BASE_URL=https://api.openai.com/v1
```

Then run:

```bash
python3 scripts/run_eval.py --output-dir results/openai_run
```

This uses `OpenAICompatibleLLMClient` and expects the model to return structured JSON actions. Real runs also require Pinecone environment variables because Pinecone is the retrieval backend when `--use-test-index` is omitted.

For a local non-Pinecone integration check with a real OpenAI-compatible policy, pass the test fake index explicitly:

```bash
python3 scripts/run_eval.py --use-test-index --output-dir results/openai_local_check
```

## 11. How to run with Pinecone

Set Pinecone environment variables:

```bash
export PINECONE_API_KEY=...
export PINECONE_INDEX_NAME=...
export PINECONE_CLOUD=aws
export PINECONE_REGION=us-east-1
```

For real retrieval runs, omit `--use-test-index`:

```bash
python3 scripts/run_eval.py --output-dir results/pinecone_run
```

Pinecone real retrieval runs require Pinecone environment variables. OpenAI-compatible LLM runs require OpenAI-compatible environment variables. A real Pinecone + real LLM run needs both sets of environment variables unless you are running the local mock path or explicitly using `--use-test-index`.

The configured Pinecone index must match the vector dimension expected by the current deterministic vectorization helper unless you have added a production embedding path.

## 12. Output artifacts

Every evaluation run writes:

- `raw_rollouts.jsonl`: one JSONL record per run config/event/action/query/metric trace.
- `metrics.csv`: compact per-query rows for reproducible local analysis.
- `eval_summary.json`: the headline scorecard, scalar score, run config metadata, counts, diagnostics, and output paths. LongMemEval summaries put `answer_success` and sub-task accuracies first.
- `predictions.jsonl`: LongMemEval-compatible rows with `question_id` and `hypothesis`.

`run_config` is included in `raw_rollouts.jsonl` and `eval_summary.json`. It records dataset mode, seed, episode id, policy/client/backend names, Pinecone index name when applicable, budgets, timestamp, and the command/config used. API keys are never written; only configured/not configured booleans are recorded.

The reproducible mock artifacts are generated under:

- `results/mock_run/`
- `results/mock_summary/`

## 13. Results

Current reproducible local mock run:

```text
queries=4
score=36.426
answer_success=0.250
time_to_useful_memory=1335.75
```

These are `MockLLMClient + InMemoryIndex` test fake results after label hiding, strict source-event validation, strict budgets, and query-time causality filtering. These are not Pinecone/OpenAI performance numbers.

## 14. Limitations / future work

- The current answer step is deterministic and uses retrieved memory content; future work can add a separate answer model while preserving evidence checks.
- The current vectorization helper is deterministic for testing and auditability; real retrieval uses Pinecone as the backend, but production embedding configuration should be expanded.
- Historical query snapshots are exact in `InMemoryIndex`, which is the deterministic evaluator/test path. The current Pinecone adapter stores the current vector under each memory ID and filters by `available_at_ms`, but Pinecone runs do not yet preserve full historical replaced-vector snapshots for older `as_of_ms` queries.
- Local answer metrics are transparent and reproducible; LongMemEval claims should be checked with the exported predictions and the official evaluator.
- The project does not include RL training.
- Baselines are intentionally minimal sanity checks, not the main result.
