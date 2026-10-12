"""Immutable graph state and typed sampling diagnostics."""

from dataclasses import dataclass
from math import isfinite
from typing import TypeAlias

from .errors import NumericalStabilityError, ShapeError

Matrix: TypeAlias = tuple[tuple[float, ...], ...]


def _is_finite_real(value: object) -> bool:
    """Return whether ``value`` is a finite non-boolean number."""
    return isinstance(value, (int, float)) and not isinstance(value, bool) and isfinite(value)


def freeze_matrix(values: list[list[float]] | tuple[tuple[float, ...], ...], name: str) -> Matrix:
    """Validate and convert a nested numeric matrix to an immutable matrix."""
    if not values:
        raise ShapeError(f"{name} must not be empty")
    try:
        rows: list[tuple[float, ...]] = []
        for row in values:
            converted_row = []
            for value in row:
                if isinstance(value, bool):
                    raise TypeError("boolean values are not numeric graph values")
                converted_row.append(float(value))
            rows.append(tuple(converted_row))
    except (TypeError, ValueError) as error:
        raise ShapeError(f"{name} must contain only numeric values") from error
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
    return tuple(rows)


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

    def __post_init__(self) -> None:
        """Reject invalid diagnostic records at construction time."""
        if isinstance(self.step, bool) or not isinstance(self.step, int) or self.step <= 0:
            raise ShapeError("diagnostic step must be a positive integer")
        if not _is_finite_real(self.time) or not 0.0 <= self.time <= 1.0:
            raise NumericalStabilityError("diagnostic time must be finite and within [0, 1]")
        if not _is_finite_real(self.timestep) or self.timestep <= 0.0:
            raise NumericalStabilityError("diagnostic time and timestep are out of bounds")
        for name, value in (
            ("feature_variation", self.feature_variation),
            ("adjacency_variation", self.adjacency_variation),
            ("smoothed_feature_variation", self.smoothed_feature_variation),
            ("smoothed_adjacency_variation", self.smoothed_adjacency_variation),
        ):
            if not _is_finite_real(value) or value < 0.0:
                raise NumericalStabilityError(f"{name} must be finite and non-negative")


@dataclass(frozen=True, slots=True)
class SamplerDiagnostics:
    """Aggregate diagnostics for a completed sampling run."""

    steps: tuple[StepDiagnostics, ...]
    total_steps: int
    final_time: float

    def __post_init__(self) -> None:
        """Validate aggregate trajectory diagnostics."""
        if not isinstance(self.steps, tuple):
            object.__setattr__(self, "steps", tuple(self.steps))
        if any(not isinstance(step, StepDiagnostics) for step in self.steps):
            raise ShapeError("steps must contain StepDiagnostics values")
        if isinstance(self.total_steps, bool) or not isinstance(self.total_steps, int):
            raise ShapeError("total_steps must be an integer")
        if self.total_steps < 0 or self.total_steps != len(self.steps):
            raise ShapeError("total_steps must match the diagnostic record count")
        if not _is_finite_real(self.final_time) or not 0.0 <= self.final_time <= 1.0:
            raise NumericalStabilityError("final_time must be finite and within [0, 1]")


@dataclass(frozen=True, slots=True)
class AdaptiveDecision:
    """Controller decision and variation metrics for one step."""

    timestep: float
    feature_variation: float
    adjacency_variation: float
    smoothed_feature_variation: float
    smoothed_adjacency_variation: float

    def __post_init__(self) -> None:
        """Validate a controller decision before it reaches a solver."""
        if not _is_finite_real(self.timestep) or self.timestep <= 0.0:
            raise NumericalStabilityError("timestep must be finite and positive")
        for name, value in (
            ("feature_variation", self.feature_variation),
            ("adjacency_variation", self.adjacency_variation),
            ("smoothed_feature_variation", self.smoothed_feature_variation),
            ("smoothed_adjacency_variation", self.smoothed_adjacency_variation),
        ):
            if not _is_finite_real(value) or value < 0.0:
                raise NumericalStabilityError(f"{name} must be finite and non-negative")


@dataclass(frozen=True, slots=True)
class SampleResult:
    """Typed result returned by the v2 sampler."""

    state: GraphState
    diagnostics: SamplerDiagnostics

    def __post_init__(self) -> None:
        """Ensure result boundaries contain the declared domain types."""
        if not isinstance(self.state, GraphState):
            raise ShapeError("sample result state must be a GraphState")
        if not isinstance(self.diagnostics, SamplerDiagnostics):
            raise ShapeError("sample result diagnostics must be SamplerDiagnostics")
