# PLAN.md

# FastMemoryWriteEnv

## Project Motive

Build a production-shaped streaming memory-write environment.

The goal is to test and improve the storing process when data keeps arriving continuously.

The core question:

> When new data keeps coming, can an LLM decide what should be stored, updated, ignored, compressed, marked stale, or indexed now so that useful information becomes searchable fast without storing everything?

This is not generic RAG.
This is not long-document QA.
This is not a rule-policy benchmark.

The project is about the write path of memory systems.

## Core Problem

Memory systems face a tradeoff:

- Store everything → slow, noisy, expensive, poor retrieval
- Store too little → fast, but misses important context
- Store stale facts → future answers become wrong
- Index everything immediately → expensive and slow
- Delay too much → important data is not useful in time

FastMemoryWriteEnv creates an environment to measure this tradeoff.

## Main Idea

Incoming events arrive over time.

The environment first saves raw events quickly.

Then an LLM memory-write policy decides what to do with each event:

- write a new memory
- update an existing memory
- mark an old memory stale
- ignore low-value information
- compress information into an existing memory
- index now
- delay indexing

Later, future queries arrive while the stream continues.

The environment checks whether the important information became useful memory fast enough.

## Main Pipeline

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
Environment scores write speed + usefulness + freshness + storage cost