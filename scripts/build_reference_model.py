#!/usr/bin/env python3
"""Train and export the deterministic, sub-10MB reference model.

This artifact is a deployable contract fixture, not a claim that the package
contains the paper's GruM/GDSS checkpoints.  It proves the full export,
manifest, golden-output, and runtime-validation path end to end.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch


class TinyGraphDriftNet(torch.nn.Module):
    """Small graph-conditioned drift network with stable tensor signatures."""

    def __init__(self, feature_dim: int = 3, hidden_dim: int = 16) -> None:
        """Initialize the compact node and edge heads."""
        super().__init__()
        self.node_input = torch.nn.Linear(feature_dim * 2 + 1, hidden_dim)
        self.node_output = torch.nn.Linear(hidden_dim, feature_dim)
        self.edge_input = torch.nn.Linear(hidden_dim * 2 + 1, hidden_dim)
        self.edge_output = torch.nn.Linear(hidden_dim, 1)

    def forward(
        self, features: torch.Tensor, adjacency: torch.Tensor, time: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return feature and adjacency drift for one graph at one time."""
        node_count = features.shape[0]
        time_column = time.reshape(1, 1).expand(node_count, 1)
        neighbours = adjacency @ features
        node_hidden = torch.tanh(
            self.node_input(torch.cat((features, neighbours, time_column), dim=1))
        )
        feature_drift = self.node_output(node_hidden)
        left = node_hidden.unsqueeze(1).expand(node_count, node_count, -1)
        right = node_hidden.unsqueeze(0).expand(node_count, node_count, -1)
        edge_time = time.reshape(1, 1, 1).expand(node_count, node_count, 1)
        edge_hidden = torch.tanh(self.edge_input(torch.cat((left, right, edge_time), dim=2)))
        adjacency_drift = self.edge_output(edge_hidden).squeeze(-1)
        return feature_drift, adjacency_drift


def _teacher(
    features: torch.Tensor, adjacency: torch.Tensor, time: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    neighbours = adjacency @ features
    time_shape = (*time.shape, 1, 1)
    feature_target = (
        0.12 * features + 0.06 * neighbours + 0.02 * torch.sin(time).reshape(time_shape) * features
    )
    edge_target = 0.04 * (adjacency - adjacency.transpose(-1, -2)) + 0.01 * torch.cos(time).reshape(
        time_shape
    )
    return feature_target, edge_target


def build(output_dir: Path) -> dict[str, object]:
    """Train, export, and write the checkpoint contract artifacts."""
    torch.manual_seed(123)
    output_dir.mkdir(parents=True, exist_ok=True)
    model = TinyGraphDriftNet().float()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.03)
    train_features = torch.randn(256, 4, 3)
    train_adjacency = torch.randn(256, 4, 4) * 0.15
    train_time = torch.rand(256)
    target_features, target_adjacency = _teacher(train_features, train_adjacency, train_time)
    for step in range(512):
        sample = step % train_features.shape[0]
        optimizer.zero_grad()
        predicted_features, predicted_adjacency = model(
            train_features[sample], train_adjacency[sample], train_time[sample]
        )
        loss = torch.nn.functional.mse_loss(
            predicted_features, target_features[sample]
        ) + torch.nn.functional.mse_loss(predicted_adjacency, target_adjacency[sample])
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.inference_mode():
        validation_losses = []
        for _sample in range(32):
            validation_features = torch.randn(4, 3)
            validation_adjacency = torch.randn(4, 4) * 0.15
            validation_time = torch.rand(())
            expected_features, expected_adjacency = _teacher(
                validation_features, validation_adjacency, validation_time
            )
            actual_features, actual_adjacency = model(
                validation_features, validation_adjacency, validation_time
            )
            validation_losses.append(
                float(torch.nn.functional.mse_loss(actual_features, expected_features))
                + float(torch.nn.functional.mse_loss(actual_adjacency, expected_adjacency))
            )
    features = torch.tensor(
        [[0.1, -0.2, 0.3], [0.4, 0.2, -0.1], [-0.3, 0.5, 0.2], [0.0, -0.4, 0.6]]
    )
    adjacency = torch.tensor(
        [[0.0, 0.2, 0.0, 0.1], [0.2, 0.0, 0.3, 0.0], [0.0, 0.3, 0.0, 0.4], [0.1, 0.0, 0.4, 0.0]]
    )
    time = torch.tensor(0.35)
    traced = torch.jit.trace(model, (features, adjacency, time), strict=True)
    checkpoint = output_dir / "driftflow-reference-graph-drift.ts"
    torch.jit.save(traced, checkpoint)
    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    with torch.inference_mode():
        eager_features, eager_adjacency = model(features, adjacency, time)
        traced_features, traced_adjacency = traced(features, adjacency, time)
    max_error = max(
        float(torch.max(torch.abs(eager_features - traced_features))),
        float(torch.max(torch.abs(eager_adjacency - traced_adjacency))),
    )
    if max_error > 1e-6:
        raise RuntimeError(f"TorchScript export changed outputs: max error {max_error}")
    manifest = {
        "name": "driftflow-reference-graph-drift",
        "version": "1.0.0",
        "checkpoint": checkpoint.name,
        "sha256": digest,
        "bytes": checkpoint.stat().st_size,
        "max_artifact_bytes": 10_000_000,
        "num_nodes": 4,
        "feature_dim": 3,
        "feature_order": ["feature_0", "feature_1", "feature_2"],
        "preprocessing": {"name": "identity", "version": "1"},
        "input_dtypes": {"features": "float32", "adjacency": "float32", "time": "float32"},
        "output_shapes": {"feature_drift": [4, 3], "adjacency_drift": [4, 4]},
        "training": {
            "seed": 123,
            "samples": 256,
            "steps": 512,
            "validation_samples": 32,
            "validation_loss": sum(validation_losses) / len(validation_losses),
            "teacher": "deterministic synthetic graph drift",
        },
        "export_max_abs_error": max_error,
    }
    (output_dir / "driftflow-reference-graph-drift.manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    golden = {
        "features": features.tolist(),
        "adjacency": adjacency.tolist(),
        "time": float(time),
        "feature_drift": eager_features.tolist(),
        "adjacency_drift": eager_adjacency.tolist(),
    }
    (output_dir / "driftflow-reference-graph-drift.golden.json").write_text(
        json.dumps(golden, indent=2) + "\n", encoding="utf-8"
    )
    return manifest


def main() -> None:
    """Build the reference artifact from the command line."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts"))
    args = parser.parse_args()
    manifest = build(args.output_dir)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
