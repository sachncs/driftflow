# Architecture

## Design goals

- Immutable state at component boundaries.
- Explicit polymorphism for models, schedules, controllers, solvers, and decoders.
- Fail-fast validation before numerical integration.
- No mutable package-level solver registry.
- Runtime dependency-free core with an optional PyTorch inference adapter.

## Data flow

```text
GraphState
    |
    v
DriftModel.evaluate(state, time) -> DriftState
    |
    +--> NoiseSchedule.value_at(time) -> g(t)
    |
    +--> AdaptiveController.choose_timestep(...) -> AdaptiveDecision
    |
    v
Solver.step(state, drift, dt, g(t)*sqrt(dt), rng, model, time)
    |
    v
GraphState + StepDiagnostics
```

`DVSSampler` orchestrates these collaborators. It owns no persistent random
state; every `sample()` call creates a fresh generator from its seed.

## Modules

- `domain.py`: immutable graph/drift states and typed results.
- `contracts.py`: abstract polymorphic interfaces.
- `config.py`: validated immutable configuration.
- `sampler.py`: DVS controller, solver implementations, registry, orchestration.
- `schedule.py`: validated noise schedules.
- `models.py`: immutable educational MLP approximations.
- `inference.py`: integrity-checked optional TorchScript model adapter.
- `utils.py`: decoders and scalar helpers.
- `errors.py`: explicit failure taxonomy.

## Invariants

- Graph features are rectangular and adjacency is square.
- Feature and adjacency node counts match.
- Drift and solver outputs preserve state shapes.
- Numerical values crossing component boundaries are finite.
- Timestep is positive, bounded, and cannot overshoot terminal time.
- Sampling stops at the configured tolerance (reporting the exact terminal
  time) or safety step limit.
- Active-range boundaries are inclusive.
