#!/usr/bin/env python3
"""Generate a deterministic streaming dataset as JSON."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fast_memory_write_env.dataset import generate_dataset
from fast_memory_write_env.schemas import DatasetMode


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate FastMemoryWriteEnv streaming dataset JSON.")
    parser.add_argument("--mode", choices=[mode.value for mode in DatasetMode], default=DatasetMode.SMALL.value)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", default="results/dataset.json")
    args = parser.parse_args()

    dataset = generate_dataset(DatasetMode(args.mode), seed=args.seed)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(dataset.model_dump_json(indent=2) + "\n", encoding="utf-8")

    stream_items = sum(len(episode.stream) for episode in dataset.episodes)
    print(f"dataset_id={dataset.dataset_id}")
    print(f"episodes={len(dataset.episodes)}")
    print(f"stream_items={stream_items}")
    print(f"output={output_path}")


if __name__ == "__main__":
    main()
