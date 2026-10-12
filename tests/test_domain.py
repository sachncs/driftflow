"""Tests for immutable domain objects and polymorphic contracts."""

import pytest

from driftflow.domain import (
    AdaptiveDecision,
    DriftState,
    GraphState,
    SamplerDiagnostics,
    StepDiagnostics,
)
from driftflow.errors import NumericalStabilityError, ShapeError


def test_graph_state_freezes_nested_matrices() -> None:
    state = GraphState.from_lists([[1.0]], [[0.0]])

    assert state.features == ((1.0,),)
    assert state.adjacency == ((0.0,),)
    assert state.to_lists() == ([[1.0]], [[0.0]])


def test_graph_state_rejects_incompatible_shapes() -> None:
    with pytest.raises(ShapeError, match="same node count"):
        GraphState.from_lists([[1.0], [2.0]], [[0.0]])


def test_graph_state_rejects_non_finite_values() -> None:
    with pytest.raises(NumericalStabilityError, match="finite"):
        GraphState.from_lists([[float("nan")]], [[0.0]])


def test_drift_state_uses_the_same_graph_contract() -> None:
    drift = DriftState.from_lists([[1.0]], [[0.0]])

    assert drift.features == ((1.0,),)


@pytest.mark.parametrize("value", [["not-a-number"]], ids=["text"])
def test_graph_state_rejects_non_numeric_values(value: list[list[object]]) -> None:
    with pytest.raises(ShapeError, match="numeric"):
        GraphState.from_lists(value, [[0.0]])  # type: ignore[arg-type]


def test_graph_state_rejects_boolean_values() -> None:
    with pytest.raises(ShapeError, match="numeric"):
        GraphState.from_lists([[True]], [[0.0]])  # type: ignore[list-item]


def test_diagnostics_are_validated_and_immutable() -> None:
    step = StepDiagnostics(1, 0.0, 0.1, 0.0, 0.0, 0.0, 0.0)
    diagnostics = SamplerDiagnostics((step,), 1, 0.1)
    decision = AdaptiveDecision(0.1, 0.0, 0.0, 0.0, 0.0)

    assert diagnostics.total_steps == 1
    assert decision.timestep == 0.1
    with pytest.raises(NumericalStabilityError):
        StepDiagnostics(1, 0.0, 0.1, -1.0, 0.0, 0.0, 0.0)
    with pytest.raises(ShapeError):
        SamplerDiagnostics((step,), 0, 0.1)
