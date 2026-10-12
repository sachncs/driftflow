"""Polymorphic, validated adaptive SDE sampling."""

from collections.abc import Mapping
from math import isfinite, sqrt
from random import Random
from threading import RLock
from types import MappingProxyType

from .config import CommonConfig, DatasetConfig
from .contracts import AdaptiveController, DriftModel, NoiseSchedule, Solver
from .domain import (
    AdaptiveDecision,
    DriftState,
    GraphState,
    SamplerDiagnostics,
    SampleResult,
    StepDiagnostics,
)
from .errors import (
    ConfigurationError,
    NumericalStabilityError,
    SamplingLimitError,
    ShapeError,
    SolverContractError,
)


def squared_l2_difference(
    current: tuple[tuple[float, ...], ...], previous: tuple[tuple[float, ...], ...]
) -> float:
    """Return the squared L2 difference between two equally shaped matrices."""
    if len(current) != len(previous):
        raise ShapeError("variation matrices must have the same row count")
    total = 0.0
    for current_row, previous_row in zip(current, previous, strict=True):
        if len(current_row) != len(previous_row):
            raise ShapeError("variation matrices must have matching row widths")
        total += sum((a - b) ** 2 for a, b in zip(current_row, previous_row, strict=True))
    return total


def compute_drift_variation_score(
    current: DriftState,
    previous: DriftState,
    noise_scale: float,
    eps_num: float,
) -> tuple[float, float]:
    """Compute feature and adjacency DVS values with a stable denominator."""
    if not isinstance(current, DriftState) or not isinstance(previous, DriftState):
        raise ShapeError("DVS inputs must be DriftState values")
    if any(
        isinstance(value, bool) or not isinstance(value, (int, float))
        for value in (noise_scale, eps_num)
    ):
        raise ConfigurationError("noise scale and eps_num must be real numbers")
    if not isfinite(noise_scale) or noise_scale < 0.0:
        raise NumericalStabilityError(
            f"noise_scale must be finite and non-negative, got {noise_scale!r}"
        )
    if not isfinite(eps_num) or eps_num <= 0.0:
        raise ConfigurationError("eps_num must be finite and positive")
    denominator = noise_scale * noise_scale + eps_num
    return (
        squared_l2_difference(current.features, previous.features) / denominator,
        squared_l2_difference(current.adjacency, previous.adjacency) / denominator,
    )


def update_ema(
    current_feature: float,
    current_adjacency: float,
    previous_feature: float,
    previous_adjacency: float,
    alpha: float,
) -> tuple[float, float]:
    """Apply the configured exponential moving average."""
    values = (current_feature, current_adjacency, previous_feature, previous_adjacency, alpha)
    if any(isinstance(value, bool) or not isinstance(value, (int, float)) for value in values):
        raise ConfigurationError("EMA inputs must be real numbers")
    if not all(isfinite(value) for value in values):
        raise NumericalStabilityError("EMA inputs must be finite")
    if not 0.0 <= alpha <= 1.0:
        raise ConfigurationError("alpha must be within [0, 1]")
    if any(value < 0.0 for value in values[:4]):
        raise NumericalStabilityError("EMA scores must be non-negative")
    return (
        alpha * current_feature + (1.0 - alpha) * previous_feature,
        alpha * current_adjacency + (1.0 - alpha) * previous_adjacency,
    )


def compute_timestep(
    smoothed_score: float,
    kappa_ref: float,
    dt_base: float,
    dt_min: float,
    dt_max: float,
    beta: float,
    eps_num: float,
) -> float:
    """Compute and bound the power-law adaptive timestep."""
    values = (smoothed_score, kappa_ref, dt_base, dt_min, dt_max, beta, eps_num)
    if any(isinstance(value, bool) or not isinstance(value, (int, float)) for value in values):
        raise ConfigurationError("timestep inputs must be real numbers")
    if not all(isfinite(value) for value in values):
        raise NumericalStabilityError("timestep inputs must be finite")
    if smoothed_score < 0.0:
        raise NumericalStabilityError(
            f"smoothed_score must be finite and non-negative, got {smoothed_score!r}"
        )
    if kappa_ref <= 0.0 or beta < 0.0 or eps_num <= 0.0:
        raise ConfigurationError("kappa_ref, eps_num, and beta have invalid values")
    if dt_min <= 0.0 or not dt_min <= dt_base <= dt_max:
        raise ConfigurationError("dt_min <= dt_base <= dt_max must hold with positive values")
    raw_timestep = dt_base * float((kappa_ref / (smoothed_score + eps_num)) ** beta)
    return max(dt_min, min(dt_max, raw_timestep))


def validate_drift_matches(state: GraphState, drift: DriftState) -> None:
    """Validate that a model output exactly matches the current state shape."""
    if len(state.features) != len(drift.features) or len(state.adjacency) != len(drift.adjacency):
        raise ShapeError("drift matrices must match state row counts")
    for state_row, drift_row in zip(state.features, drift.features, strict=True):
        if len(state_row) != len(drift_row):
            raise ShapeError("drift features must match state shape")
    for state_row, drift_row in zip(state.adjacency, drift.adjacency, strict=True):
        if len(state_row) != len(drift_row):
            raise ShapeError("drift adjacency must match state shape")


class DvsAdaptiveController(AdaptiveController):
    """Adaptive controller implementing the Drift Variation Score policy."""

    def __init__(self, common: CommonConfig, dataset: DatasetConfig, gamma: float) -> None:
        """Initialize a controller with validated configuration."""
        if isinstance(gamma, bool) or not isinstance(gamma, (int, float)):
            raise ConfigurationError("gamma must be a real number")
        if not isfinite(gamma) or gamma < 0.0:
            raise ConfigurationError("gamma must be finite and non-negative")
        self.common = common
        self.dataset = dataset
        self.gamma = gamma
        self.smoothed_feature_variation = 0.0
        self.smoothed_adjacency_variation = 0.0

    def reset(self) -> None:
        """Reset state accumulated from a previous trajectory."""
        self.smoothed_feature_variation = 0.0
        self.smoothed_adjacency_variation = 0.0

    @property
    def eps_bound(self) -> float:
        """Return the configured termination tolerance."""
        return self.common.eps_bound

    @property
    def max_steps(self) -> int:
        """Return the configured safety step limit."""
        return self.common.max_steps

    def choose_timestep(
        self,
        current_drift: DriftState,
        previous_drift: DriftState | None,
        noise_scale: float,
        time: float,
        terminal_time: float,
    ) -> AdaptiveDecision:
        """Choose a timestep and return all controller diagnostics."""
        if isinstance(time, bool) or isinstance(terminal_time, bool):
            raise ConfigurationError("time values must be real numbers")
        if not all(
            isinstance(value, (int, float)) and isfinite(value) for value in (time, terminal_time)
        ):
            raise ConfigurationError("time values must be finite")
        if not 0.0 <= time <= terminal_time <= 1.0:
            raise ConfigurationError(
                "time and terminal_time must satisfy 0 <= time <= terminal_time <= 1"
            )
        if previous_drift is None or not self.dataset.is_active(time):
            return AdaptiveDecision(
                self.common.dt_base,
                0.0,
                0.0,
                self.smoothed_feature_variation,
                self.smoothed_adjacency_variation,
            )
        feature_variation, adjacency_variation = compute_drift_variation_score(
            current_drift,
            previous_drift,
            noise_scale,
            self.common.eps_num,
        )
        self.smoothed_feature_variation, self.smoothed_adjacency_variation = update_ema(
            feature_variation,
            adjacency_variation,
            self.smoothed_feature_variation,
            self.smoothed_adjacency_variation,
            self.common.alpha,
        )
        feature_timestep = compute_timestep(
            self.smoothed_feature_variation,
            self.dataset.kappa_ref,
            self.common.dt_base,
            self.common.dt_min,
            self.common.dt_max,
            self.common.beta,
            self.common.eps_num,
        )
        adjacency_timestep = compute_timestep(
            self.smoothed_adjacency_variation,
            self.dataset.kappa_ref,
            self.common.dt_base,
            self.common.dt_min,
            self.common.dt_max,
            self.common.beta,
            self.common.eps_num,
        )
        timestep = min(feature_timestep, adjacency_timestep, terminal_time - time)
        self.smoothed_feature_variation = self.gamma * (
            self.smoothed_feature_variation + self.smoothed_adjacency_variation
        )
        self.smoothed_adjacency_variation = self.smoothed_feature_variation
        return AdaptiveDecision(
            timestep,
            feature_variation,
            adjacency_variation,
            self.smoothed_feature_variation,
            self.smoothed_adjacency_variation,
        )


class EulerMaruyamaSolver(Solver):
    """First-order Euler-Maruyama solver."""

    @property
    def name(self) -> str:
        """Return the solver name."""
        return "Euler-Maruyama"

    def step(
        self,
        state: GraphState,
        drift: DriftState,
        timestep: float,
        noise_scale: float,
        random_generator: Random,
        drift_model: DriftModel,
        time: float,
    ) -> GraphState:
        """Advance both graph modalities by one Euler step."""
        del drift_model, time
        validate_solver_parameters(timestep, noise_scale)
        validate_drift_matches(state, drift)
        noise = draw_noise(state, noise_scale, random_generator)
        return advance_with_noise(state, drift, timestep, noise)


class HeunSolver(Solver):
    """Second-order predictor-corrector solver."""

    @property
    def name(self) -> str:
        """Return the solver name."""
        return "Heun"

    def step(
        self,
        state: GraphState,
        drift: DriftState,
        timestep: float,
        noise_scale: float,
        random_generator: Random,
        drift_model: DriftModel,
        time: float,
    ) -> GraphState:
        """Advance the state using a shared-noise Heun correction."""
        validate_solver_parameters(timestep, noise_scale)
        validate_drift_matches(state, drift)
        noise = draw_noise(state, noise_scale, random_generator)
        predicted = advance_with_noise(state, drift, timestep, noise)
        second_drift = drift_model.evaluate(predicted, time + timestep)
        validate_drift_matches(predicted, second_drift)
        return average_drift_and_noise(state, drift, second_drift, timestep, noise)


def validate_solver_parameters(timestep: float, noise_scale: float) -> None:
    """Validate numerical parameters before a solver consumes them."""
    if any(
        isinstance(value, bool) or not isinstance(value, (int, float))
        for value in (timestep, noise_scale)
    ):
        raise ConfigurationError("solver timestep and noise scale must be real numbers")
    if not all(isfinite(value) for value in (timestep, noise_scale)):
        raise NumericalStabilityError("solver timestep and noise scale must be finite")
    if timestep <= 0.0 or noise_scale < 0.0:
        raise NumericalStabilityError("solver timestep must be positive and noise non-negative")


def draw_noise(
    state: GraphState,
    noise_scale: float,
    random_generator: Random,
) -> tuple[tuple[tuple[float, ...], ...], tuple[tuple[float, ...], ...]]:
    """Draw one noise matrix per graph modality."""
    validate_solver_parameters(1.0, noise_scale)
    return (
        tuple(
            tuple(noise_scale * random_generator.gauss(0.0, 1.0) for _ in row)
            for row in state.features
        ),
        tuple(
            tuple(noise_scale * random_generator.gauss(0.0, 1.0) for _ in row)
            for row in state.adjacency
        ),
    )


def advance_with_noise(
    state: GraphState,
    drift: DriftState,
    timestep: float,
    noise: tuple[tuple[tuple[float, ...], ...], tuple[tuple[float, ...], ...]],
) -> GraphState:
    """Advance a state using a caller-provided noise realization."""

    def update(
        state_matrix: tuple[tuple[float, ...], ...],
        drift_matrix: tuple[tuple[float, ...], ...],
        noise_matrix: tuple[tuple[float, ...], ...],
    ) -> tuple[tuple[float, ...], ...]:
        return tuple(
            tuple(
                state_value + drift_value * timestep + noise_value
                for state_value, drift_value, noise_value in zip(
                    state_row, drift_row, noise_row, strict=True
                )
            )
            for state_row, drift_row, noise_row in zip(
                state_matrix, drift_matrix, noise_matrix, strict=True
            )
        )

    return GraphState(
        update(state.features, drift.features, noise[0]),
        update(state.adjacency, drift.adjacency, noise[1]),
    )


def average_drift_and_noise(
    state: GraphState,
    first_drift: DriftState,
    second_drift: DriftState,
    timestep: float,
    noise: tuple[tuple[tuple[float, ...], ...], tuple[tuple[float, ...], ...]],
) -> GraphState:
    """Apply a Heun correction using one shared noise realization."""

    def update(
        state_matrix: tuple[tuple[float, ...], ...],
        first_matrix: tuple[tuple[float, ...], ...],
        second_matrix: tuple[tuple[float, ...], ...],
        noise_matrix: tuple[tuple[float, ...], ...],
    ) -> tuple[tuple[float, ...], ...]:
        return tuple(
            tuple(
                state_value + 0.5 * (first_value + second_value) * timestep + noise_value
                for state_value, first_value, second_value, noise_value in zip(
                    state_row, first_row, second_row, noise_row, strict=True
                )
            )
            for state_row, first_row, second_row, noise_row in zip(
                state_matrix, first_matrix, second_matrix, noise_matrix, strict=True
            )
        )

    return GraphState(
        update(state.features, first_drift.features, second_drift.features, noise[0]),
        update(state.adjacency, first_drift.adjacency, second_drift.adjacency, noise[1]),
    )


class SolverRegistry:
    """Isolated registry of solver implementations."""

    def __init__(self, solvers: tuple[Solver, ...] | None = None) -> None:
        """Create a registry with fresh Euler and Heun defaults."""
        self._solvers: dict[str, Solver] = {}
        for solver in (EulerMaruyamaSolver(), HeunSolver()) if solvers is None else solvers:
            self.register(solver)

    @property
    def solvers(self) -> Mapping[str, Solver]:
        """Return a read-only view of registered solvers."""
        return MappingProxyType(self._solvers)

    def register(self, solver: Solver) -> None:
        """Register a solver under its unique name."""
        if not isinstance(solver, Solver):
            raise ConfigurationError("solver must implement the Solver contract")
        if not isinstance(solver.name, str) or not solver.name.strip():
            raise ConfigurationError("solver name must be a non-empty string")
        if solver.name in self._solvers:
            raise ConfigurationError(f"solver already registered: {solver.name}")
        self._solvers[solver.name] = solver

    def get(self, name: str) -> Solver:
        """Return a solver or raise a descriptive configuration error."""
        try:
            return self._solvers[name]
        except KeyError as error:
            raise ConfigurationError(
                f"unknown solver {name!r}; available: {sorted(self._solvers)}"
            ) from error


class DVSSampler:
    """Stateless, typed sampler orchestrating drift, control, and solving."""

    def __init__(
        self,
        drift_model: DriftModel,
        noise_schedule: NoiseSchedule,
        controller: AdaptiveController,
        solver_registry: SolverRegistry | None = None,
        solver_name: str = "Euler-Maruyama",
        production: bool = False,
    ) -> None:
        """Initialize a sampler with explicit polymorphic collaborators."""
        if not isinstance(drift_model, DriftModel):
            raise ConfigurationError("drift_model must implement the DriftModel contract")
        if not isinstance(noise_schedule, NoiseSchedule):
            raise ConfigurationError("noise_schedule must implement the NoiseSchedule contract")
        if not isinstance(controller, AdaptiveController):
            raise ConfigurationError("controller must implement the AdaptiveController contract")
        self.drift_model = drift_model
        self.noise_schedule = noise_schedule
        self.controller = controller
        self.solver_registry = solver_registry or SolverRegistry()
        self.solver = self.solver_registry.get(solver_name)
        if production and not self.drift_model.production_ready:
            raise ConfigurationError(
                "production mode requires a validated production-ready drift model; "
                "approximation models are restricted to research and smoke tests"
            )
        self.production = production
        self._sampling_lock = RLock()

    def sample(
        self, state: GraphState, terminal_time: float = 1.0, seed: int | None = None
    ) -> SampleResult:
        """Run one trajectory while serializing shared controller state."""
        with self._sampling_lock:
            return self._sample_unlocked(state, terminal_time, seed)

    def _sample_unlocked(
        self, state: GraphState, terminal_time: float = 1.0, seed: int | None = None
    ) -> SampleResult:
        """Sample from ``state`` to ``terminal_time`` using a fresh RNG."""
        if (
            isinstance(terminal_time, bool)
            or not isinstance(terminal_time, (int, float))
            or not isfinite(terminal_time)
            or terminal_time < 0.0
            or terminal_time > 1.0
        ):
            raise ConfigurationError("terminal_time must be finite and within [0, 1]")
        if not isinstance(state, GraphState):
            raise ConfigurationError("state must be a GraphState")
        if seed is not None and (isinstance(seed, bool) or not isinstance(seed, int)):
            raise ConfigurationError("seed must be an integer or None")
        self.controller.reset()
        current_state = state
        previous_drift: DriftState | None = None
        random_generator = Random(seed)
        time = 0.0
        step_number = 0
        diagnostics: list[StepDiagnostics] = []

        while time < terminal_time - self.controller.eps_bound:
            step_number += 1
            if step_number > self.controller.max_steps:
                raise SamplingLimitError("sampling exceeded max_steps")
            drift = self.drift_model.evaluate(current_state, time)
            validate_drift_matches(current_state, drift)
            noise_scale = self.noise_schedule.value_at(time)
            if (
                isinstance(noise_scale, bool)
                or not isinstance(noise_scale, (int, float))
                or not isfinite(noise_scale)
                or noise_scale < 0.0
            ):
                raise NumericalStabilityError("noise schedule returned an invalid noise scale")
            decision = self.controller.choose_timestep(
                drift, previous_drift, noise_scale, time, terminal_time
            )
            metrics = (
                decision.timestep,
                decision.feature_variation,
                decision.adjacency_variation,
                decision.smoothed_feature_variation,
                decision.smoothed_adjacency_variation,
            )
            if any(not isfinite(metric) for metric in metrics):
                raise NumericalStabilityError("controller returned a non-finite decision")
            timestep = min(decision.timestep, terminal_time - time)
            if not isfinite(timestep) or timestep <= 0.0:
                raise SamplingLimitError("controller returned a non-positive timestep")
            next_time = min(terminal_time, time + timestep)
            timestep = next_time - time
            next_state = self.solver.step(
                current_state,
                drift,
                timestep,
                noise_scale * sqrt(timestep),
                random_generator,
                self.drift_model,
                time,
            )
            if not isinstance(next_state, GraphState):
                raise SolverContractError("solver must return a GraphState")
            diagnostics.append(
                StepDiagnostics(
                    step_number,
                    time,
                    timestep,
                    decision.feature_variation,
                    decision.adjacency_variation,
                    decision.smoothed_feature_variation,
                    decision.smoothed_adjacency_variation,
                )
            )
            previous_drift = drift
            current_state = next_state
            time = next_time

        if terminal_time - time <= self.controller.eps_bound:
            time = terminal_time
        return SampleResult(
            current_state,
            SamplerDiagnostics(tuple(diagnostics), step_number, time),
        )
