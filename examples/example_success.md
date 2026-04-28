# Example Success Case

This example describes the reproducible mock run generated with:

```bash
python3 scripts/run_eval.py --mock --use-test-index --output-dir results/mock_run
```

In the stream, useful and urgent facts arrive before later queries. The mock `LLMMemoryWritePolicy` proposes validated `write_memory` actions for those facts and marks them for immediate indexing when indexing budget is available. `FastMemoryWriteEnv` then executes the actions, writes memories to SQLite, updates the test index, and records action latencies.

A query later asks where outage alerts should go. The retrieval step returns the memory written from the urgent fact event, and the answer cites the memory. The metric layer checks that the cited memory covers the required gold fact IDs and supporting event IDs, so `answer_correct`, `evidence_correct`, and `answer_success` are true for that query.

The relevant artifacts are:

- `results/mock_run/raw_rollouts.jsonl`
- `results/mock_run/metrics.csv`
- `results/mock_run/eval_summary.json`

The success is not based on exact answer text. It is based on whether retrieved/cited memory covers the required facts and evidence.
