"""Tests for immutable domain objects and polymorphic contracts."""

import pytest

from driftflow.domain import DriftState, GraphState
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
