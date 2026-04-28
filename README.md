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
    -> validated memory actions
    -> memory store + Pinecone index
    -> future queries arrive while data is still coming
    -> retrieve memory + answer
    -> score speed, usefulness, freshness, and storage cost
```

The environment executes actions and records timings. The LLM policy proposes actions only.

## 4. LLMMemoryWritePolicy

`LLMMemoryWritePolicy` receives:

- new raw event
- current active memories
- recent event context
- latency budget
- storage budget
- indexing budget

It calls an internal `LLMClient` abstraction and expects structured JSON actions. All actions are validated with Pydantic before the environment can execute them. Invalid JSON or invalid action payloads trigger retry/repair.

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

Every executed action returns a structured result with success state, action type, simulated latency, storage token delta, error, and payload.

## 6. Pinecone backend

Pinecone is the real retrieval backend for real runs.

Required Pinecone environment variables:

- `PINECONE_API_KEY`
- `PINECONE_INDEX_NAME`
- `PINECONE_CLOUD`
- `PINECONE_REGION`

`InMemoryIndex` exists only as a unit-test and mock-run fake. It is useful for reproducible local checks, but it is not the production retrieval path.

The current index vectorization helper is deterministic and prototype-oriented so local tests and rollouts are reproducible. Pinecone is the real backend, but production retrieval semantics require configuring or extending the embedding path for the target deployment; do not treat the deterministic hashed vectors as a production embedding model.

## 7. Dataset modes

The dataset simulates streaming episodes, not static QA pairs. Each episode interleaves events and queries over time.

Modes:

- `small`: fast tests and local smoke runs.
- `medium`: larger evaluation runs.
- `long`: serious longer streaming runs.

Generated streams include useful facts, low-value noise, duplicates, contradictions, stale updates, urgent facts, multiple users/entities, far-apart evidence, and storage/indexing pressure.

Generate a dataset JSON file:

```bash
python3 scripts/generate_dataset.py --mode small --seed 7 --output results/dataset_small.json
```

## 8. Metrics, especially `time_to_useful_memory`

The primary metric is `time_to_useful_memory`:

```text
event arrives
-> raw event written
-> useful memory written/updated
-> memory indexed
-> memory retrieved by a future query
-> answer uses it correctly
```

`time_to_useful_memory` is `null` unless every required fact completes the full chain: event arrival -> raw write -> memory write/update -> index -> retrieval -> successful answer. The breakdown fields remain useful when only part of the path completed.

The environment also reports:

- `time_to_raw_write`
- `time_to_memory_write`
- `time_to_indexed_memory`
- `time_to_retrieved_memory`
- `answer_success`
- `answer_correct`
- `evidence_correct`
- `memory_precision`
- `memory_recall`
- `stale_memory_rate`
- `duplicate_memory_rate`
- `storage_tokens_used`
- `useful_memory_per_storage_token`
- write, index, and query latency percentiles
- `ignored_useful_fact_rate`
- `stored_noise_rate`

Answer success is evidence-based:

```text
answer_correct = answer text satisfies required gold fact strings
evidence_correct = cited memory evidence supports required gold facts/events
answer_success = answer_correct AND evidence_correct
```

Exact/contains matching is only debug information. The deterministic verifier normalizes the required fact strings and answer text to make the scoring transparent; it is intentionally conservative and not a claim of full semantic NLU.

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
- `metrics.csv`: per-query metrics with aggregate fields.
- `eval_summary.json`: aggregate metrics, score breakdown, run config metadata, counts, and output paths.

`run_config` is included in `raw_rollouts.jsonl` and `eval_summary.json`. It records dataset mode, seed, episode id, policy/client/backend names, Pinecone index name when applicable, budgets, timestamp, and the command/config used. API keys are never written; only configured/not configured booleans are recorded.

The reproducible mock artifacts are generated under:

- `results/mock_run/`
- `results/mock_summary/`

## 13. Results

Current reproducible local mock run:

```text
queries=4
score=78.77455
answer_success=1.0
answer_correct=1.0
evidence_correct=1.0
fact_evidence_coverage=1.0
time_to_useful_memory=3716.4125
```

These are `MockLLMClient + InMemoryIndex` test fake results. These are not Pinecone/OpenAI performance numbers.

## 14. Limitations / future work

- The current answer step is deterministic and uses retrieved memory content; future work can add a separate answer model while preserving evidence checks.
- The current vectorization helper is deterministic for testing and auditability; real retrieval uses Pinecone as the backend, but production embedding configuration should be expanded.
- Metrics are intentionally transparent and reproducible; future work can add richer rollout analysis and dashboards.
- The project does not include RL training.
- Baselines are intentionally minimal sanity checks, not the main result.
