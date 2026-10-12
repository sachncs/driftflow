#!/usr/bin/env python3
"""Validate the exported reference artifact through the public adapter."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from driftflow import GraphState, TorchScriptDriftModel


def main() -> None:
    """Validate the checkpoint against its manifest and golden fixture."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-dir", type=Path, default=Path("artifacts"))
    args = parser.parse_args()
    manifest_path = args.artifact_dir / "driftflow-reference-graph-drift.manifest.json"
    golden_path = args.artifact_dir / "driftflow-reference-graph-drift.golden.json"
    golden = json.loads(golden_path.read_text(encoding="utf-8"))
    model = TorchScriptDriftModel.from_manifest(manifest_path)
    result = model.evaluate(
        GraphState.from_lists(golden["features"], golden["adjacency"]), golden["time"]
    )
    actual = {"feature_drift": result.features, "adjacency_drift": result.adjacency}
    expected = {
        "feature_drift": golden["feature_drift"],
        "adjacency_drift": golden["adjacency_drift"],
    }
    max_error = max(
        abs(actual[key][row][column] - expected[key][row][column])
        for key in actual
        for row in range(len(actual[key]))
        for column in range(len(actual[key][row]))
    )
    if max_error > 1e-5:
        raise SystemExit(f"golden output mismatch: max error {max_error}")
    if model.checkpoint.stat().st_size >= 10_000_000:
        raise SystemExit("checkpoint exceeds the 10MB deployment budget")
    print(
        json.dumps(
            {"status": "ok", "max_error": max_error, "bytes": model.checkpoint.stat().st_size}
        )
    )


if __name__ == "__main__":
    main()
