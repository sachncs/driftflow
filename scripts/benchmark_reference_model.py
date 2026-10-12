#!/usr/bin/env python3
"""Run a repeatable local latency and concurrent-load benchmark."""

from __future__ import annotations

import argparse
import json
import platform
import resource
import statistics
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import torch

from driftflow import GraphState, TorchScriptDriftModel


def _teacher(features, adjacency, time_value):
    """Independent reference function for the synthetic drift task."""
    neighbours = adjacency @ features
    feature_target = 0.12 * features + 0.06 * neighbours + 0.02 * torch.sin(time_value) * features
    edge_target = 0.04 * (adjacency - adjacency.transpose(0, 1)) + 0.01 * torch.cos(time_value)
    return feature_target, edge_target


def main() -> None:
    """Measure single-request and concurrent adapter latency."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-dir", type=Path, default=Path("artifacts"))
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    started_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    manifest = args.artifact_dir / "driftflow-reference-graph-drift.manifest.json"
    manifest_data = json.loads(manifest.read_text(encoding="utf-8"))
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
    generator = torch.Generator().manual_seed(456)
    quality_errors = []
    with torch.inference_mode():
        for _ in range(64):
            features = torch.randn((4, 3), generator=generator)
            adjacency = torch.randn((4, 4), generator=generator) * 0.15
            time_value = torch.rand((), generator=generator)
            expected_features, expected_adjacency = _teacher(features, adjacency, time_value)
            actual = model.evaluate(
                GraphState.from_lists(features.tolist(), adjacency.tolist()), float(time_value)
            )
            quality_errors.append(
                float(torch.mean((torch.tensor(actual.features) - expected_features) ** 2))
                + float(torch.mean((torch.tensor(actual.adjacency) - expected_adjacency) ** 2))
            )
    quality_mse = sum(quality_errors) / len(quality_errors)
    if quality_mse > 0.01:
        raise SystemExit(f"quality gate failed: validation MSE {quality_mse}")
    ended_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    rss_unit = 1 if platform.system() == "Darwin" else 1024
    samples.sort()
    result = {
        "iterations": args.iterations,
        "artifact_bytes": model.checkpoint.stat().st_size,
        "parameter_count": manifest_data["parameter_count"],
        "quality": {
            "reference": "independent deterministic synthetic graph-drift teacher",
            "samples": 64,
            "mse": quality_mse,
            "max_mse": 0.01,
        },
        "memory": {
            "max_rss_delta_bytes": max(0, int((ended_rss - started_rss) * rss_unit)),
            "rss_unit": "bytes" if rss_unit == 1 else "kilobytes converted to bytes",
        },
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
