#!/usr/bin/env python3
"""Run a repeatable local latency and concurrent-load benchmark."""

from __future__ import annotations

import argparse
import json
import statistics
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from driftflow import GraphState, TorchScriptDriftModel


def main() -> None:
    """Measure single-request and concurrent adapter latency."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-dir", type=Path, default=Path("artifacts"))
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    manifest = args.artifact_dir / "driftflow-reference-graph-drift.manifest.json"
    golden = json.loads(
        (args.artifact_dir / "driftflow-reference-graph-drift.golden.json").read_text(
            encoding="utf-8"
        )
    )
    model = TorchScriptDriftModel.from_manifest(manifest)
    state = GraphState.from_lists(golden["features"], golden["adjacency"])
    for _ in range(10):
        model.evaluate(state, golden["time"])
    samples = []
    for _ in range(args.iterations):
        started = time.perf_counter_ns()
        model.evaluate(state, golden["time"])
        samples.append((time.perf_counter_ns() - started) / 1_000_000)
    started = time.perf_counter()
    with ThreadPoolExecutor(max_workers=8) as executor:
        list(executor.map(lambda _: model.evaluate(state, golden["time"]), range(args.iterations)))
    concurrent_seconds = time.perf_counter() - started
    samples.sort()
    result = {
        "iterations": args.iterations,
        "artifact_bytes": model.checkpoint.stat().st_size,
        "latency_ms": {
            "p50": statistics.median(samples),
            "p95": samples[int(len(samples) * 0.95) - 1],
            "max": max(samples),
        },
        "concurrent": {
            "workers": 8,
            "requests": args.iterations,
            "seconds": concurrent_seconds,
            "requests_per_second": args.iterations / concurrent_seconds,
        },
        "note": "macOS local baseline; rerun on the target production host before setting SLOs",
    }
    if args.output:
        args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
