# FastMemoryWriteEnv

Production-shaped streaming memory-write environment for evaluating how quickly and accurately an LLM policy turns continuous incoming data into useful searchable memory while data keeps arriving.

The central object of study is the **memory write path under pressure** — what to store, update, ignore, mark stale, compress, index now, or delay — under latency, storage, and indexing budgets. Retrieval and answering are present because they test whether write decisions were useful.

## Headline result

**LongMemEval-S, n=50, real OpenAI (`gpt-4o-mini`) + real Pinecone, 8 concurrent memory-write workers:**

```
answer_success    = 0.87       (43–44 / 50 queries judged fully correct)
score (internal)  = 84.0       (project composite )
write workers     = 8
```

Position in the published LongMemEval-S landscape:

| System | Policy model | n | answer_success |
|---|---|---|---|
| Long-context baseline (no memory) | gpt-4o | 500 | 0.60–0.64 |
| Zep / Graphiti | gpt-4o | 500 | 0.712 |
| TiMem | gpt-4o-mini | 500 | 0.769 |
| EverMemOS | — | 500 | 0.830 |
| Mastra Observational Memory | gpt-4o | 500 | 0.842 |
| Mem0 | gpt-4o-mini | 500 | ~0.85 |
| **FastMemoryWriteEnv (this work)** | **gpt-4o-mini** | **50** | **0.87** |
| Mastra Observational Memory | gpt-5-mini | 500 | 0.949 |


## What this is

A streaming evaluator for a single LLM memory-write policy. Every chat turn becomes one event in a chronological stream; queries arrive while events are still coming. The policy receives one event at a time and returns validated structured actions — the environment executes them. The policy never mutates stores or calls the index directly.

## What this is not

Generic RAG. Long-document QA. A comparison of many hand-written memory strategies. A toy benchmark with no production pipeline. The minimal `NoMemoryBaseline`, `StoreEverythingBaseline`, and `OraclePolicy` exist as bounds, not as the main result.

## Architecture

Two parallel paths sharing one storage layer:

**Write path** (asynchronous, multi-worker):

```
Event arrives  →  store_raw (SQLite + FTS5)  →  enqueue MemoryWriteQueue
              →  Async worker (N threads)   →  LLMMemoryWritePolicy.decide
              →  Validate + execute actions →  MemoryStore + RetrievalIndex
```

**Read path** (synchronous, drains queue first):

```
Query at T  →  wait_until_no_ready_work(cutoff = T)  →  SearchMemoryAction(as_of_ms = T)
           →  AnswerAction (LLM composes answer + citations)
           →  evaluate_query_result (LLM judge + evidence subset check)
```

The async worker is the production-shaped piece: events keep arriving on the main loop while the policy deliberates on previous events in the background. Drain barriers on queries preserve causal ordering without giving up streaming throughput.

### LLMMemoryWritePolicy

Inputs to `policy.decide`:

- new raw event (label-hidden view)
- current active memories (capped at 24)
- recent events (capped at 10)
- latency budget, storage budget, indexing budget

Output: a validated list of up to 4 structured actions from `{write_memory, update_memory, mark_stale, ignore_event, compress_memory, index_now, delay_index}`.

### Production-grade guarantees

- **Pydantic-validated everywhere.** Every proposed action is parsed against a strict schema before the environment will execute it. Invalid JSON triggers a bounded repair-retry loop with the validation error fed back into the prompt.
- **OpenAI Structured Outputs.** `memory_action_response_format()` builds a `strict: true` JSON schema with a discriminated-union over the seven action types and an `ID_PATTERN` regex on every identifier — schema enforcement happens server-side at generation time.
- **Label-hidden event view.** The policy-visible payload exposes only `event_id, episode_id, timestamp_ms, source, user_id, entity_id, content, estimated_tokens` and a whitelisted metadata subset. Evaluator-only fields (`category, facts, priority, tags, duplicate_of, contradicts, supersedes`) are stripped. `session_id` is SHA-256-anonymized so LongMemEval's `answer_*` session-prefix gold-evidence leak cannot reach the LLM.
- **Source-event grounding.** Memory actions citing `source_event_ids` are rejected unless every cited event has already been raw-written. Hallucinated event ids cannot ground unverifiable memories.
- **Streaming-fault tolerance.** `LLMClientError` and `PolicyPlanError` from `policy.decide` are caught per-event and recorded as recoverable plan failures (empty action list). A single rate-limit storm does not corrupt an episode.
- **OpenAI rate-limit awareness.** The client honors `x-ratelimit-reset-tokens` / `x-ratelimit-reset-requests` headers and the `"Please try again in <duration>"` body hint, with proportional jitter so concurrent workers don't retry in lockstep.
- **Causal time filter.** Every indexed memory carries `available_at_ms`. Search at time T returns only versions with `available_at_ms ≤ T` — a query never sees the future, even when the rollout has progressed past it.
- **Stricter answer success than the benchmark default.** `answer_success = answer_correct AND evidence_correct`, where `evidence_correct` requires `required_fact_ids ⊆ cited_fact_ids AND supporting_event_ids ⊆ cited_event_ids`. LongMemEval's official metric only requires the answer text — citation overlap is a self-imposed extra check.

### Storage and retrieval

```
RawEventStore (SQLite + FTS5)        MemoryStore (SQLite + FTS5 mirror)
                                                  │
                                                  ▼
                                       RetrievalIndex (protocol)
                                                  │
              ┌───────────────────────────────────┼───────────────────────────────────┐
              ▼                                   ▼                                   ▼
       PineconeIndex                  HybridRetrievalIndex                     InMemoryIndex
       (real backend)                 (vector + lexical RRF)                   (test-only, versioned)
```

`HybridRetrievalIndex` fuses a vector index (Pinecone or in-memory) with the SQLite FTS5 mirror via Reciprocal Rank Fusion (default k = 60). A memory becomes findable through FTS5 the moment it's written — before vector indexing completes — so time-to-useful-memory does not gate on the indexing budget.

## Reproducing

Real Pinecone + real OpenAI:

```bash
export OPENAI_API_KEY=...
export OPENAI_MODEL=gpt-4o-mini
export PINECONE_API_KEY=...
export PINECONE_INDEX_NAME=...
export PINECONE_CLOUD=aws
export PINECONE_REGION=us-east-1

python3 scripts/run_eval_multi.py \
  --dataset longmemeval \
  --longmemeval-path data/longmemeval/longmemeval_s_cleaned.json \
  --samples 1 \
  --write-worker-concurrency 8 \
  --output-dir results/lme_clean
```

Independent re-grading against the official LongMemEval evaluator:

```bash
git clone https://github.com/xiaowu0162/LongMemEval
cd LongMemEval/src/evaluation
python3 evaluate_qa.py gpt-4o \
  /path/to/results/lme_50/predictions.jsonl \
  ../../data/longmemeval_oracle.json
```


## Output artifacts

Every evaluation writes:

- `raw_rollouts.jsonl` — one record per event / action / query / metric, including the full `run_config`
- `metrics.csv` — per-query rows for downstream analysis
- `eval_summary.json` — headline scorecard, counts, diagnostics, output paths
- `predictions.jsonl` — `{question_id, hypothesis}` for the official LongMemEval evaluator

API keys are never persisted; only `*_configured` booleans appear in `run_config`.

## Scoring

Two distinct numbers are reported:

- **`answer_success`** is the comparable-to-leaderboard metric. Per-query binary: 1 if `answer_correct AND evidence_correct`, 0 otherwise. `answer_correct` is an LLM-as-judge YES/NO verdict on whether the answer text conveys every gold answer fact. `evidence_correct` is a deterministic set check on cited memory ids vs gold supporting events / fact ids.
- **`score`** (the composite) is project-internal: a weighted combination of `answer_success`, sub-task accuracy, memory recall and precision, minus storage and stale-memory penalties. It exists for ablation work; only `answer_success` should be compared against published systems.


## References

- Wu et al., *LongMemEval: Benchmarking Chat Assistants on Long-Term Interactive Memory*, ICLR 2025. https://github.com/xiaowu0162/LongMemEval
- Mem0. https://github.com/mem0ai/mem0
- Zep / Graphiti (Rasmussen et al., 2025).
- TiMem (Li et al., 2026); EverMemOS (Hu et al., 2026).
- Mastra Observational Memory. https://mastra.ai/research/observational-memory
