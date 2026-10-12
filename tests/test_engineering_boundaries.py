"""Failure-path tests for the engineering contracts."""

from __future__ import annotations

import json
import random
from pathlib import Path

import pytest

from driftflow import (
    ConfigurationError,
    DriftState,
    GraphState,
    NumericalStabilityError,
    ShapeError,
    SolverRegistry,
    compute_timestep,
)
from driftflow.domain import AdaptiveDecision, SamplerDiagnostics, StepDiagnostics
from driftflow.models import matmul, matvec, xavier_uniform
from driftflow.sampler import compute_drift_variation_score, update_ema, validate_solver_parameters


@pytest.mark.parametrize(
    ("features", "adjacency", "message"),
    [
        ([], [[0.0]], "must not be empty"),
        ([[]], [[0.0]], "rows must not be empty"),
        ([[1.0], [2.0, 3.0]], [[0.0, 1.0], [1.0, 0.0]], "rectangular"),
        ([[1.0]], [[0.0, 1.0]], "square"),
    ],
)
def test_graph_boundary_rejects_malformed_shapes(
    features: list[list[float]], adjacency: list[list[float]], message: str
) -> None:
    with pytest.raises(ShapeError, match=message):
        GraphState.from_lists(features, adjacency)


@pytest.mark.parametrize(
    "record",
    [
        (0, 0.0, 0.1, 0.0, 0.0, 0.0, 0.0),
        (1, 2.0, 0.1, 0.0, 0.0, 0.0, 0.0),
        (1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        (1, 0.0, 0.1, float("inf"), 0.0, 0.0, 0.0),
    ],
)
def test_diagnostic_boundary_rejects_invalid_records(record: tuple[object, ...]) -> None:
    with pytest.raises((ShapeError, NumericalStabilityError)):
        StepDiagnostics(*record)  # type: ignore[arg-type]


def test_decision_and_aggregate_diagnostics_reject_invalid_values() -> None:
    with pytest.raises(NumericalStabilityError):
        AdaptiveDecision(0.0, 0.0, 0.0, 0.0, 0.0)
    step = StepDiagnostics(1, 0.0, 0.1, 0.0, 0.0, 0.0, 0.0)
    with pytest.raises(ShapeError):
        SamplerDiagnostics((step,), 2, 0.1)
    with pytest.raises(NumericalStabilityError):
        SamplerDiagnostics((), 0, 2.0)


def test_matrix_and_weight_helpers_reject_invalid_inputs() -> None:
    with pytest.raises(ConfigurationError):
        xavier_uniform(0, 2, random.Random(1))
    with pytest.raises(ShapeError):
        matmul([[1.0]], [[1.0], [2.0, 3.0]])
    with pytest.raises(ShapeError):
        matmul([[1.0]], [])
    with pytest.raises(ShapeError):
        matvec([[1.0], [2.0, 3.0]], [1.0])


@pytest.mark.parametrize(
    ("args", "message"),
    [
        ((-1.0, 1.0, 0.1, 0.01, 0.2, 0.5, 1e-12), "non-negative"),
        ((0.0, 0.0, 0.1, 0.01, 0.2, 0.5, 1e-12), "kappa_ref"),
        ((0.0, 1.0, 0.2, 0.1, 0.05, 0.5, 1e-12), "dt_min"),
        ((0.0, 1.0, 0.1, 0.01, 0.2, 0.5, 0.0), "kappa_ref"),
    ],
)
def test_timestep_boundary_rejects_invalid_configuration(
    args: tuple[float, ...], message: str
) -> None:
    with pytest.raises((ConfigurationError, NumericalStabilityError), match=message):
        compute_timestep(*args)


def test_dvs_and_ema_boundaries_reject_invalid_values() -> None:
    drift = DriftState.from_lists([[0.0]], [[0.0]])
    with pytest.raises(ConfigurationError, match="real numbers"):
        compute_drift_variation_score(drift, drift, True, 1e-12)  # type: ignore[arg-type]
    with pytest.raises(ConfigurationError, match="eps_num"):
        compute_drift_variation_score(drift, drift, 0.1, 0.0)
    with pytest.raises(ConfigurationError, match="alpha"):
        update_ema(0.0, 0.0, 0.0, 0.0, 2.0)
    with pytest.raises(NumericalStabilityError, match="non-negative"):
        update_ema(-1.0, 0.0, 0.0, 0.0, 0.5)


def test_solver_parameter_boundary_rejects_nonfinite_and_negative_values() -> None:
    with pytest.raises(ConfigurationError, match="real numbers"):
        validate_solver_parameters(True, 0.1)  # type: ignore[arg-type]
    with pytest.raises(NumericalStabilityError, match="finite"):
        validate_solver_parameters(float("nan"), 0.1)
    with pytest.raises(NumericalStabilityError, match="positive"):
        validate_solver_parameters(0.0, 0.1)


def test_registry_rejects_invalid_solver_objects() -> None:
    registry = SolverRegistry(())
    with pytest.raises(ConfigurationError, match="implement"):
        registry.register(object())  # type: ignore[arg-type]


def test_manifest_boundary_rejects_non_object_and_missing_fields(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps([]), encoding="utf-8")
    from driftflow import TorchScriptDriftModel

    with pytest.raises(ConfigurationError, match="JSON object"):
        TorchScriptDriftModel.from_manifest(manifest)
    manifest.write_text(json.dumps({"checkpoint": "model.ts"}), encoding="utf-8")
    with pytest.raises(ConfigurationError, match="missing fields"):
        TorchScriptDriftModel.from_manifest(manifest)


def test_drift_state_rejects_bad_adjacency() -> None:
    with pytest.raises(ShapeError, match="square"):
        DriftState.from_lists([[1.0]], [[0.0, 1.0]])
