# API Reference

This document describes the clean-break v2 API. Runtime APIs are typed,
validated, and composed through abstract contracts rather than global
registries or unstructured dictionaries.

## Core domain types

### `GraphState`

```python
GraphState(features: Matrix, adjacency: Matrix)
```

Immutable graph state. Features must be rectangular, adjacency must be square,
and both matrices must have the same node count and finite values.

Use `GraphState.from_lists(features, adjacency)` to construct one from nested
lists and `to_lists()` to obtain defensive mutable copies.

### `DriftState`

Immutable drift with the same shape and finiteness guarantees as `GraphState`.

### `SampleResult`

Contains `state: GraphState` and `diagnostics: SamplerDiagnostics`.

`SamplerDiagnostics` contains a tuple of `StepDiagnostics`, `total_steps`, and
`final_time`.

## Contracts

All extension interfaces are abstract classes in `driftflow.contracts`:

```python
class DriftModel(ABC):
    def evaluate(self, state: GraphState, time: float) -> DriftState: ...

class NoiseSchedule(ABC):
    def value_at(self, time: float) -> float: ...

class Solver(ABC):
    @property
    def name(self) -> str: ...

    def step(
        self, state, drift, timestep, noise_scale,
        random_generator, drift_model, time,
    ) -> GraphState: ...

class AdaptiveController(ABC):
    def choose_timestep(
        self, current_drift, previous_drift, noise_scale, time, terminal_time,
    ) -> AdaptiveDecision: ...

class GraphDecoder(ABC):
    def decode(self, adjacency) -> tuple[tuple[float, ...], ...]: ...
```

## Configuration

`CommonConfig` and `DatasetConfig` are frozen, slotted dataclasses. They reject
non-finite values, invalid timestep ordering, invalid gamma values, and invalid
active ranges during construction.

`get_dataset_config(model, dataset)` returns a validated table entry and raises
`ConfigurationError` for unknown pairs.

## Schedules

`LinearSchedule`, `CosineSchedule`, `PolynomialSchedule`, and
`ConstantSchedule` implement `NoiseSchedule`. They validate time in `[0, 1]`
and return finite non-negative scales. `constant_schedule(value)` is a factory
for `ConstantSchedule`.

## Controllers and samplers

`DvsAdaptiveController` implements the DVS, EMA, timestep scaling, active-range
gating, and global refresh policy.

`DVSSampler` composes a model, schedule, controller, and solver registry:

```python
sampler = DVSSampler(
    drift_model=model,
    noise_schedule=LinearSchedule(0.01, 0.5),
    controller=controller,
    solver_registry=SolverRegistry(),
    solver_name="Euler-Maruyama",
)
result = sampler.sample(state, terminal_time=1.0, seed=42)
```

Each call creates a fresh random generator. The sampler validates terminal time,
noise scales, solver outputs, and step limits.

Built-in solvers:

- `EulerMaruyamaSolver`
- `HeunSolver`

`SolverRegistry` is instance-scoped. Registering a custom solver affects only
the registry instance that receives it.

## Models and decoders

`SimpleGraphDenoiser`, `GruMApproximation`, and `GDSSApproximation` implement
`DriftModel`. The approximation classes are educational stand-ins, not the
published production networks.

`ThresholdGraphDecoder` and `SigmoidGraphDecoder` implement `GraphDecoder`.
The functional `decode_adjacency` and `sigmoid_decode_adjacency` helpers remain
available for simple post-processing.

## Exceptions

- `ConfigurationError`: invalid configuration or unknown registry entry.
- `ShapeError`: incompatible state, drift, or matrix shapes.
- `ScheduleError`: invalid schedule input or time.
- `NumericalStabilityError`: non-finite or invalid numerical values.
- `SolverContractError`: reserved for solver contract violations.
- `SamplingLimitError`: a sampling safety limit was exceeded.
