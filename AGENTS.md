# AGENTS.md

## Main Instruction

You are building `FastMemoryWriteEnv`.

This project is important. Do not turn it into a toy benchmark.

The motive is:

> Create a production-shaped streaming memory-write environment that helps improve how fast incoming data becomes useful memory while data keeps coming continuously.

The core is the memory write path.

## What This Project Is

This project tests whether an LLM can decide:

- what to store
- what to update
- what to ignore
- what to mark stale
- what to compress
- what to index now
- what to delay

under latency, storage, and indexing constraints.

The environment scores whether those decisions make future queries answerable quickly and correctly.

## What This Project Is Not

Do not build this as:

- generic RAG
- long-document QA
- static document retrieval
- a comparison of many hand-written rule policies
- a toy local-only demo
- a pure benchmark with no production-shaped pipeline

Do not add many rule-based strategy policies.

The main system is one LLM memory-write policy.

## Main System

The main policy is:

```text
LLMMemoryWritePolicy
```

It receives:

- new raw event
- current active memories
- recent event context
- latency budget
- storage budget
- indexing budget

It outputs validated structured actions:

- write_memory
- update_memory
- mark_stale
- ignore_event
- compress_memory
- index_now
- delay_index

The environment executes the actions.

The LLM policy must not directly mutate stores or call Pinecone.

## Minimal Baselines Only

Allowed baselines:

- NoMemoryBaseline
- StoreEverythingBaseline
- OraclePolicy

Do not create multiple hand-written memory strategies as the main result.

Rules are only sanity checks or bounds.

## Core Pipeline

```text
Continuous data stream
    ↓
Fast raw write
    ↓
Memory-write queue
    ↓
LLM memory-write policy
    ↓
Validated memory actions
    ↓
Memory store + Pinecone index
    ↓
Future queries arrive while data is still coming
    ↓
Retrieve memory + answer
    ↓
Environment scores speed + usefulness + freshness + storage cost
```

## Production-Shaped Design

Build the architecture like a real system.

Use:

- typed Pydantic schemas
- clean module boundaries
- SQLite raw/memory stores
- Pinecone as the real retrieval backend
- in-memory fake only for unit tests
- structured logging
- latency accounting
- raw rollout logs
- reproducible evaluation scripts

## Pinecone

Pinecone is the real backend.

Use environment variables:

- PINECONE_API_KEY
- PINECONE_INDEX_NAME
- PINECONE_CLOUD
- PINECONE_REGION

InMemoryIndex exists only for unit tests.

Do not make the real path local-only.

## LLM Client

Use an internal abstraction:

```text
LLMClient
```

Implement:

```text
MockLLMClient
OpenAICompatibleLLMClient
```

The LLM policy should call the abstraction, not hardcode provider logic everywhere.

The LLM must return structured JSON.
Validate all outputs with Pydantic action schemas.
Add retry/repair for invalid JSON.

## Dataset Requirements

The dataset must simulate streaming data.

Events should arrive over time.

Queries should arrive while events are still coming.

Include:

- useful facts
- low-value noise
- duplicates
- contradictions
- stale updates
- urgent facts
- multiple entities/users
- far-apart evidence
- storage/indexing pressure

Dataset modes:

- small
- medium
- long

Small is for tests.
Long is for serious runs.

## Main Metric

The primary metric is:

```text
time_to_useful_memory
```

It means:

```text
event arrives
-> raw event written
-> useful memory written/updated
-> memory indexed
-> memory retrieved by future query
-> answer uses it correctly
```

Track the breakdown:

- time_to_raw_write
- time_to_memory_write
- time_to_indexed_memory
- time_to_retrieved_memory
- time_to_useful_memory

## Answer Success

Use evidence + facts.

```text
answer_correct = answer satisfies required gold facts
evidence_correct = cited evidence supports answer
answer_success = answer_correct AND evidence_correct
```

Do not use exact string match as primary scoring.

## Files To Build

Use this structure:

```text
fast_memory_write_env/
  __init__.py
  actions.py
  schemas.py
  state.py
  env.py
  stores.py
  index.py
  pinecone_index.py
  in_memory_index.py
  llm_client.py
  policies.py
  rewards.py
  metrics.py
  dataset.py
  evaluator.py
  config.py
```

Scripts:

```text
scripts/
  generate_dataset.py
  run_llm_policy.py
  run_eval.py
  evaluate_results.py
```

Tests:

```text
tests/
  test_actions.py
  test_dataset.py
  test_env_transitions.py
  test_stores.py
  test_in_memory_index.py
  test_llm_policy_mock.py
  test_metrics.py
  test_rewards.py
```

## Phase Instructions

### Phase 1

Implement:

- schemas.py
- actions.py
- state.py
- dataset.py
- tests for schemas/dataset

Do not implement Pinecone yet.

### Phase 2

Implement:

- stores.py
- index.py
- pinecone_index.py
- in_memory_index.py
- env.py
- config.py
- transition tests

Pinecone is the real backend.
InMemoryIndex is test fake only.

### Phase 3

Implement:

- llm_client.py
- LLMMemoryWritePolicy
- MockLLMClient
- OpenAICompatibleLLMClient
- run_llm_policy.py
- tests using mock client

Do not add multiple rule policies.

### Phase 4

Implement:

- rewards.py
- metrics.py
- evaluator.py
- run_eval.py
- evaluate_results.py

Focus on time-to-useful-memory.

### Phase 5

Implement:

- README.md
- examples
- raw rollout logs
- metrics output
- eval summary

## Quality Bar

Every phase must keep tests passing.

Required command:

```bash
pytest -q
```

Every real run should produce:

```text
results/raw_rollouts.jsonl
results/metrics.csv
results/eval_summary.json
```

No hidden state.
No unvalidated LLM actions.
No vague policy behavior.
No toy framing.

## Coding Style

Use:

- type hints
- Pydantic models
- dataclasses only when appropriate
- small clear modules
- deterministic seeds for dataset generation
- structured logs
- explicit error handling

Do not silently swallow failures.

Action execution should return structured results:

```text
success
action_type
latency_ms
storage_tokens_delta
error
payload
```

## Important Design Rule

The environment is not smart by itself.

The LLM policy decides memory actions.

The environment:

- provides events
- validates actions
- updates stores
- manages index
- serves queries
- records timings
- scores whether memory became useful fast enough

## Final Framing

The final README should describe the project as:

> A production-shaped streaming memory-write environment for evaluating how quickly and accurately an LLM policy can turn continuous incoming data into useful searchable memory.

Do not describe it as:

> A RAG benchmark.

Do not describe it as:

> A comparison of hand-written memory strategies.

Do not describe it as:

> A simple document QA environment.
