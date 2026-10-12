# Extension Guide

driftflow extensions are explicit polymorphic collaborators. Add behavior by
implementing a contract; do not mutate package-level registries or branch on
class names inside the sampler.

## New drift model

Implement `DriftModel.evaluate(state, time)` and return `DriftState`. Validate
any external tensor-to-matrix conversion before constructing the result.

## New noise schedule

Implement `NoiseSchedule.value_at(time)`. Reject invalid configuration in the
constructor and return finite, non-negative values.

## New controller

Implement `AdaptiveController.choose_timestep`. The result must be an
`AdaptiveDecision` with a positive timestep and finite diagnostics. Expose
`eps_bound` and `max_steps` so the sampler can enforce safety limits.

## New solver

Implement `Solver.name` and `Solver.step`. Solvers must preserve graph shapes,
return finite values, and use the supplied random generator. Register instances
on a local `SolverRegistry`.

## New decoder

Implement `GraphDecoder.decode` and return an immutable adjacency matrix.

## Performance backends

The public contracts intentionally separate state/control from numerical
storage. A future NumPy or tensor backend can implement the same contracts as
an optional package without changing the adaptive controller or sampler.
