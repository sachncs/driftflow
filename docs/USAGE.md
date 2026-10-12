# Usage Guide

## Minimal typed integration

```python
from driftflow import (
    CommonConfig,
    DvsAdaptiveController,
    DVSSampler,
    DriftModel,
    DriftState,
    GraphState,
    LinearSchedule,
    get_dataset_config,
)


class DampingDrift(DriftModel):
    """Move every graph value toward zero."""

    def evaluate(self, state: GraphState, time: float) -> DriftState:
        """Return a deterministic drift for ``state``."""
        del time
        return DriftState(
            tuple(tuple(-0.5 * value for value in row) for row in state.features),
            tuple(tuple(-0.5 * value for value in row) for row in state.adjacency),
        )


state = GraphState.from_lists([[1.0, 0.0], [0.0, 1.0]], [[0.5, 0.3], [0.3, 0.5]])
dataset = get_dataset_config("GruM", "QM9")
controller = DvsAdaptiveController(CommonConfig(), dataset, dataset.gamma_euler or 0.0)
sampler = DVSSampler(
    drift_model=DampingDrift(),
    noise_schedule=LinearSchedule(0.01, 0.5),
    controller=controller,
)
result = sampler.sample(state, terminal_time=1.0, seed=42)
print(result.diagnostics.final_time)
```

## Choosing solvers

Use `solver_name="Euler-Maruyama"` for the first-order solver or
`solver_name="Heun"` for the predictor-corrector solver. Heun evaluates the
drift one additional time per step.

## Custom model

Implement `DriftModel.evaluate`. The method must return a `DriftState` with the
same node and feature dimensions as the input. Non-finite values are rejected.

## Custom schedule

Subclass `NoiseSchedule` and implement `value_at`. Return a finite,
non-negative scale for every time in `[0, 1]`.

## Custom solver

Subclass `Solver`, then register it on an isolated registry:

```python
class IdentitySolver(Solver):
    """Example solver that leaves the state unchanged."""

    @property
    def name(self) -> str:
        """Return the registry name."""
        return "Identity"

    def step(self, state, drift, timestep, noise_scale, random_generator, drift_model, time):
        """Return the state unchanged."""
        del drift, timestep, noise_scale, random_generator, drift_model, time
        return state


registry = SolverRegistry()
registry.register(IdentitySolver())
sampler = DVSSampler(..., solver_registry=registry, solver_name="Identity")
```

## Reproducibility

Pass `seed` to each `sample()` call. The sampler has no hidden RNG state, so
the same state, model, configuration, and seed produce the same result even
when the sampler object is reused.

## CLI demo

```bash
python3 examples/demo.py --model GruM --dataset QM9 \
  --solver Euler-Maruyama --use-approximation --seed 42
```

The approximation is for smoke tests and education, not production inference.
