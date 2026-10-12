"""Behavioral tests for the typed polymorphic sampler pipeline."""

from dataclasses import dataclass

import pytest

from driftflow import (
    CommonConfig,
    ConfigurationError,
    DriftModel,
    DriftState,
    DvsAdaptiveController,
    DVSSampler,
    EulerMaruyamaSolver,
    GraphState,
    LinearSchedule,
    SamplingLimitError,
    ShapeError,
    Solver,
    SolverRegistry,
    get_dataset_config,
)


class ConstantDrift(DriftModel):
    """Small deterministic drift model for contract tests."""

    def evaluate(self, state: GraphState, time: float) -> DriftState:
        """Return a constant drift with the same shape as ``state``."""
        del time
        return DriftState(
            tuple(tuple(1.0 for _ in row) for row in state.features),
            tuple(tuple(1.0 for _ in row) for row in state.adjacency),
        )


class InvalidDrift(DriftModel):
    """Model that violates the finite-value contract."""

    def evaluate(self, state: GraphState, time: float) -> DriftState:
        """Return an invalid non-finite drift."""
        del time
        return DriftState(
            tuple(tuple(float("nan") for _ in row) for row in state.features),
            state.adjacency,
        )


class MismatchedDrift(DriftModel):
    """Model that returns a finite but incompatible drift shape."""

    def evaluate(self, state: GraphState, time: float) -> DriftState:
        """Return a drift with a missing feature column."""
        del state, time
        return DriftState(((1.0,),), ((1.0,),))


class InvalidSolver(Solver):
    """Solver that violates the return-type contract."""

    @property
    def name(self) -> str:
        """Return the invalid solver name."""
        return "Invalid"

    def step(self, state, drift, timestep, noise_scale, random_generator, drift_model, time):
        """Return an invalid value instead of a graph state."""
        del state, drift, timestep, noise_scale, random_generator, drift_model, time
        return []


@dataclass
class IdentitySolver(Solver):
    """Custom solver proving registry polymorphism and isolation."""

    @property
    def name(self) -> str:
        """Return the custom solver name."""
        return "Identity"

    def step(self, state, drift, timestep, noise_scale, random_generator, drift_model, time):
        """Return the original state without advancing it."""
        del drift, timestep, noise_scale, random_generator, drift_model, time
        return state


def build_sampler(model: DriftModel | None = None, max_steps: int = 1_000) -> DVSSampler:
    """Build a sampler with the standard validated collaborators."""
    controller = DvsAdaptiveController(
        CommonConfig(max_steps=max_steps), get_dataset_config("GruM", "QM9"), 0.22
    )
    return DVSSampler(model or ConstantDrift(), LinearSchedule(0.01, 0.1), controller)


def initial_state() -> GraphState:
    """Return a two-node graph for tests."""
    return GraphState.from_lists([[0.0], [0.0]], [[0.0, 0.0], [0.0, 0.0]])


def test_sampler_returns_typed_result_and_exact_terminal_time() -> None:
    result = build_sampler().sample(initial_state(), terminal_time=0.01, seed=7)

    assert result.diagnostics.total_steps >= 2
    assert result.diagnostics.final_time == 0.01
    assert result.state.features[0][0] != 0.0
    assert len(result.diagnostics.steps) == result.diagnostics.total_steps


def test_sampler_is_reproducible_per_call_seed() -> None:
    sampler = build_sampler()

    first = sampler.sample(initial_state(), terminal_time=0.01, seed=42)
    second = sampler.sample(initial_state(), terminal_time=0.01, seed=42)

    assert first == second


def test_heun_and_euler_match_for_constant_drift() -> None:
    state = initial_state()
    euler = build_sampler()
    heun = DVSSampler(
        ConstantDrift(),
        LinearSchedule(0.01, 0.1),
        DvsAdaptiveController(CommonConfig(), get_dataset_config("GruM", "QM9"), 0.22),
        solver_name="Heun",
    )

    assert euler.sample(state, 0.01, seed=9).state == heun.sample(state, 0.01, seed=9).state


def test_registry_isolated_and_custom_solver_is_polymorphic() -> None:
    registry = SolverRegistry((EulerMaruyamaSolver(),))
    registry.register(IdentitySolver())

    assert registry.get("Identity").name == "Identity"
    with pytest.raises(ConfigurationError, match="unknown solver"):
        SolverRegistry().get("Identity")


def test_invalid_drift_fails_before_integration() -> None:
    with pytest.raises(ValueError, match="finite"):
        build_sampler(InvalidDrift()).sample(initial_state(), terminal_time=0.01, seed=1)


def test_mismatched_drift_fails_with_shape_error() -> None:
    with pytest.raises(ShapeError, match="drift"):
        build_sampler(MismatchedDrift()).sample(initial_state(), terminal_time=0.01, seed=1)


def test_solver_return_type_is_checked() -> None:
    registry = SolverRegistry((InvalidSolver(),))
    sampler = DVSSampler(
        ConstantDrift(),
        LinearSchedule(0.01, 0.1),
        DvsAdaptiveController(CommonConfig(), get_dataset_config("GruM", "QM9"), 0.22),
        solver_registry=registry,
        solver_name="Invalid",
    )
    with pytest.raises(TypeError, match="GraphState"):
        sampler.sample(initial_state(), terminal_time=0.01, seed=1)


def test_sampling_limit_prevents_runaway_trajectories() -> None:
    with pytest.raises(SamplingLimitError, match="max_steps"):
        build_sampler(max_steps=1).sample(initial_state(), terminal_time=0.01, seed=1)
