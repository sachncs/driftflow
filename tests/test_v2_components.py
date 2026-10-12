"""Contract and numerical tests for v2 component implementations."""

import math

import pytest

from driftflow import (
    ConfigurationError,
    CosineSchedule,
    GDSSApproximation,
    GraphState,
    GruMApproximation,
    LinearSchedule,
    NumericalStabilityError,
    PolynomialSchedule,
    SigmoidGraphDecoder,
    ThresholdGraphDecoder,
    constant_schedule,
    decode_adjacency,
    sigmoid_decode_adjacency,
)
from driftflow.models import matmul, matvec, time_embedding
from driftflow.utils import stable_sigmoid


def test_all_schedules_are_monotonic_and_callable() -> None:
    schedules = (
        LinearSchedule(0.01, 0.5),
        CosineSchedule(),
        PolynomialSchedule(2.0, 0.0, 1.0),
        constant_schedule(0.1),
    )

    for schedule in schedules:
        values = [schedule.value_at(time) for time in (0.0, 0.5, 1.0)]
        assert all(math.isfinite(value) and value >= 0.0 for value in values)
        assert schedule(0.5) == values[1]
        assert values[0] <= values[-1]


def test_model_evaluate_returns_matching_immutable_drift() -> None:
    state = GraphState.from_lists([[0.1, 0.2], [0.3, 0.4]], [[0.0, 0.1], [0.1, 0.0]])

    for model in (GruMApproximation(2, 2, seed=1), GDSSApproximation(2, 2, seed=1)):
        drift = model.evaluate(state, 0.5)
        assert len(drift.features) == 2
        assert len(drift.features[0]) == 2
        assert len(drift.adjacency) == 2
        assert len(drift.adjacency[0]) == 2


def test_model_rejects_wrong_graph_shape() -> None:
    model = GruMApproximation(2, 2)

    with pytest.raises(ValueError, match="adjacency"):
        model.predict([[0.0, 0.0], [0.0, 0.0]], [[0.0]], 0.0)


def test_matrix_helpers_reject_dimension_mismatch() -> None:
    with pytest.raises(ValueError, match="matmul"):
        matmul([[1.0, 2.0]], [[1.0]])
    with pytest.raises(ValueError, match="matmul"):
        matmul([], [[1.0]])
    assert matvec([[1.0, 2.0]], [3.0, 4.0]) == [11.0]
    with pytest.raises(ValueError, match="matvec"):
        matvec([[1.0]], [1.0, 2.0])
    with pytest.raises(ValueError, match="matvec"):
        matvec([], [1.0])


def test_time_embedding_rejects_negative_dimensions() -> None:
    with pytest.raises(ConfigurationError, match="dimension"):
        time_embedding(0.0, -1)
    with pytest.raises(ConfigurationError, match="within"):
        time_embedding(2.0, 2)
    with pytest.raises(ConfigurationError, match="finite"):
        time_embedding(float("nan"), 2)


def test_approximation_weights_are_immutable_and_inputs_are_validated() -> None:
    model = GruMApproximation(2, 2, seed=1)
    assert isinstance(model.weight1, tuple)
    assert not hasattr(model, "rng")
    with pytest.raises(AttributeError, match="immutable"):
        model.hidden_dim = 99  # type: ignore[misc]
    with pytest.raises(NumericalStabilityError, match="finite"):
        model.predict([[float("nan"), 0.0], [0.0, 0.0]], [[0.0, 0.0], [0.0, 0.0]], 0.0)
    with pytest.raises(ConfigurationError, match="seed"):
        GruMApproximation(2, 2, seed=True)  # type: ignore[arg-type]


def test_decoders_are_polymorphic_and_stable_for_extreme_logits() -> None:
    adjacency = [[-1_000_000.0, 0.0, 1_000_000.0]]

    assert sigmoid_decode_adjacency(adjacency) == [[0.0, 1.0, 1.0]]
    assert SigmoidGraphDecoder().decode(adjacency) == ((0.0, 1.0, 1.0),)
    assert ThresholdGraphDecoder(0.5).decode([[0.4, 0.5]]) == ((0.0, 1.0),)
    assert stable_sigmoid(-1_000_000.0) == 0.0
    assert decode_adjacency([[0.5]]) == [[1.0]]


def test_decoder_rejects_invalid_threshold() -> None:
    with pytest.raises(ConfigurationError, match="threshold"):
        ThresholdGraphDecoder(2.0)
