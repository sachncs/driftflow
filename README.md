<div align="center">

# driftflow

**Typed adaptive SDE sampling for graph diffusion.**

[![Python](https://img.shields.io/badge/python-3.10%20%E2%80%933.14-blue)](#installation)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![CI](https://img.shields.io/github/actions/workflow/status/sachncs/driftflow/ci.yml?branch=master)](https://github.com/sachncs/driftflow/actions)

</div>

driftflow is a pure-Python research implementation of Drift Variation Score
(DVS) adaptive sampling for graph-diffusion SDEs. The v2 API uses immutable
graph states and explicit polymorphic contracts for drift models, schedules,
controllers, solvers, and decoders.

The built-in approximation models are educational stand-ins. They are not the
GruM or GDSS neural networks and must not be treated as production denoisers.

## Installation

```bash
python3 -m pip install -e .
python3 -m pip install -e ".[dev]"
```

Runtime code has no third-party dependencies. Development checks use pytest,
coverage, Ruff, mypy, build, and pip-audit.

Production inference uses the optional PyTorch adapter:

```bash
python3 -m pip install "driftflow[production]"
```

The repository does not ship trained GruM or GDSS weights. Use a versioned,
validated TorchScript checkpoint with `TorchScriptDriftModel` and enable
`production=True` on `DVSSampler`; the built-in approximation models are
rejected in that mode.

## Quick start

```python
from driftflow import (
    CommonConfig,
    DvsAdaptiveController,
    DVSSampler,
    GraphState,
    GruMApproximation,
    LinearSchedule,
    get_dataset_config,
)

state = GraphState.from_lists(
    features=[[0.1, 0.2], [0.3, 0.4]],
    adjacency=[[0.0, 0.1], [0.1, 0.0]],
)
model = GruMApproximation(num_nodes=2, feature_dim=2, seed=42)
dataset = get_dataset_config("GruM", "QM9")
controller = DvsAdaptiveController(CommonConfig(), dataset, dataset.gamma_euler or 0.0)

sampler = DVSSampler(
    drift_model=model,
    noise_schedule=LinearSchedule(sigma_min=0.01, sigma_max=0.5),
    controller=controller,
)
result = sampler.sample(state, terminal_time=1.0, seed=42)
print(result.diagnostics.total_steps)
```

`SampleResult` contains an immutable final `GraphState` and typed
`SamplerDiagnostics`. Sampling receives its seed per call, so reusing a sampler
does not consume hidden random state.

## Polymorphic extension points

Implement the abstract contracts in `driftflow.contracts`:

- `DriftModel.evaluate(state, time) -> DriftState`
- `NoiseSchedule.value_at(time) -> float`
- `AdaptiveController.choose_timestep(...) -> AdaptiveDecision`
- `Solver.step(...) -> GraphState`
- `GraphDecoder.decode(adjacency) -> immutable adjacency`

Register solvers in an isolated `SolverRegistry` instance. There is no global
mutable solver registry.

## Validation and failures

Configuration, state, drift, schedule, and solver boundaries are validated.
Expected failures use typed exceptions such as `ConfigurationError`,
`ShapeError`, `ScheduleError`, `NumericalStabilityError`, and
`SamplingLimitError`.

## Development checks

```bash
./scripts/check.sh
```

The demo can be run with:

```bash
python3 examples/demo.py --model GruM --dataset QM9 \
  --solver Euler-Maruyama --use-approximation --seed 42
```

See [docs/USAGE.md](docs/USAGE.md), [docs/API_REFERENCE.md](docs/API_REFERENCE.md),
and [docs/API_MIGRATION.md](docs/API_MIGRATION.md) for the full v2 surface.

## Production boundary

The sampler and validation contracts are suitable for integrating a validated
production model, but this repository is not a trained model distribution.
Numerical results must be validated against the reference paper, the target
dataset, and the deployed checkpoint before production rollout. See
[docs/PRODUCTION.md](docs/PRODUCTION.md) for the checkpoint contract,
integrity checks, rollout gates, and operational checklist.

## License

MIT. See [LICENSE](LICENSE).
