"""Immutable graph state and typed sampling diagnostics."""

from dataclasses import dataclass
from math import isfinite
from typing import TypeAlias

from .errors import NumericalStabilityError, ShapeError

Matrix: TypeAlias = tuple[tuple[float, ...], ...]


def freeze_matrix(values: list[list[float]] | tuple[tuple[float, ...], ...], name: str) -> Matrix:
    """Validate and convert a nested numeric matrix to an immutable matrix."""
    if not values:
        raise ShapeError(f"{name} must not be empty")
    rows = tuple(tuple(float(value) for value in row) for row in values)
    width = len(rows[0])
    if width == 0:
        raise ShapeError(f"{name} rows must not be empty")
    if any(len(row) != width for row in rows):
        raise ShapeError(f"{name} must be rectangular")
    for row_index, row in enumerate(rows):
        for column_index, value in enumerate(row):
            if not isfinite(value):
                raise NumericalStabilityError(
                    f"{name}[{row_index}][{column_index}] must be finite, got {value!r}"
                )
    return rows


@dataclass(frozen=True, slots=True)
class GraphState:
    """Immutable node-feature and adjacency state for one graph trajectory."""

    features: Matrix
    adjacency: Matrix

    def __post_init__(self) -> None:
        """Validate graph dimensions and finite values."""
        features = freeze_matrix(self.features, "features")
        adjacency = freeze_matrix(self.adjacency, "adjacency")
        if len(adjacency) != len(adjacency[0]):
            raise ShapeError("adjacency must be square")
        if len(features) != len(adjacency):
            raise ShapeError("features and adjacency must have the same node count")
        object.__setattr__(self, "features", features)
        object.__setattr__(self, "adjacency", adjacency)

    @classmethod
    def from_lists(
        cls,
        features: list[list[float]],
        adjacency: list[list[float]],
    ) -> "GraphState":
        """Create an immutable graph state from ordinary nested lists."""
        return cls(tuple(tuple(row) for row in features), tuple(tuple(row) for row in adjacency))

    def to_lists(self) -> tuple[list[list[float]], list[list[float]]]:
        """Return defensive mutable copies for legacy numerical adapters."""
        return [list(row) for row in self.features], [list(row) for row in self.adjacency]


@dataclass(frozen=True, slots=True)
class DriftState:
    """Immutable drift for both graph modalities."""

    features: Matrix
    adjacency: Matrix

    def __post_init__(self) -> None:
        """Validate drift dimensions and finite values."""
        state = GraphState(self.features, self.adjacency)
        object.__setattr__(self, "features", state.features)
        object.__setattr__(self, "adjacency", state.adjacency)

    @classmethod
    def from_lists(
        cls,
        features: list[list[float]],
        adjacency: list[list[float]],
    ) -> "DriftState":
        """Create immutable drift from ordinary nested lists."""
        return cls(tuple(tuple(row) for row in features), tuple(tuple(row) for row in adjacency))

    def to_lists(self) -> tuple[list[list[float]], list[list[float]]]:
        """Return defensive mutable copies for numerical adapters."""
        return [list(row) for row in self.features], [list(row) for row in self.adjacency]


@dataclass(frozen=True, slots=True)
class StepDiagnostics:
    """Diagnostics captured for one integration step."""

    step: int
    time: float
    timestep: float
    feature_variation: float
    adjacency_variation: float
    smoothed_feature_variation: float
    smoothed_adjacency_variation: float


@dataclass(frozen=True, slots=True)
class SamplerDiagnostics:
    """Aggregate diagnostics for a completed sampling run."""

    steps: tuple[StepDiagnostics, ...]
    total_steps: int
    final_time: float


@dataclass(frozen=True, slots=True)
class SampleResult:
    """Typed result returned by the v2 sampler."""

    state: GraphState
    diagnostics: SamplerDiagnostics
