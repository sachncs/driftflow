# Developer Guide

This guide describes the v2 development workflow and extension boundaries.

## Development setup

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
```

Run the complete local gate with:

```bash
PYTHON=.venv/bin/python ./scripts/check.sh
PYTHON=.venv/bin/python ./scripts/ci.sh
```

The gate runs Ruff, Google-style formatting checks, mypy, pytest with the 90%
coverage floor, the demo, the Astro build, package building, and pip-audit.

## Engineering rules

- Public extension points are explicit abstract classes in `contracts.py`.
- Public data crosses boundaries as frozen, validated domain objects.
- Use Google Python style, complete type annotations, and descriptive errors.
- Do not use underscore-prefixed members as a semi-private API convention.
- Keep mutable trajectory state inside an explicit collaborator, not globals.
- Every numerical boundary validates shape, finiteness, and bounds.
- Add a regression test for every bug fix and failure contract.

## Adding a solver

Implement `Solver` and register it on an instance-local `SolverRegistry`:

```python
from random import Random

from driftflow import GraphState, Solver, SolverRegistry


class MySolver(Solver):
    @property
    def name(self) -> str:
        return "My solver"

    def step(
        self,
        state: GraphState,
        drift,
        timestep: float,
        noise_scale: float,
        random_generator: Random,
        drift_model,
        time: float,
    ) -> GraphState:
        # Return a new validated GraphState.
        ...


registry = SolverRegistry()
registry.register(MySolver())
```

Do not modify `DVSSampler` or introduce a process-global registry. Add tests
for deterministic behavior, malformed outputs, and reproducibility.

## Adding a schedule

Subclass `NoiseSchedule`, validate constructor parameters, validate the unit
interval in `value_at`, and return a finite non-negative scale.

## Adding a model or decoder

Subclass `DriftModel` or `GraphDecoder`. Return `DriftState` from models and a
tuple-of-tuples from decoders. Preserve shape validation and test extreme
values, empty inputs where supported, and incompatible dimensions.

## Test layout

The suite is contract-focused:

- `tests/test_domain.py`: immutability and finite-value boundaries.
- `tests/test_v2_config.py`: configuration validation.
- `tests/test_v2_components.py`: schedules, models, matrix helpers, decoders.
- `tests/test_v2_sampler.py`: integration, polymorphism, reproducibility,
  solver failures, and safety limits.

Tests should be runnable through pytest; standalone test-file execution is not
part of the supported workflow.

## Site development

```bash
cd site
npm ci
npm run dev
npm run build
```

The site is a static Astro project with its own CSS system. Keep copy honest
about approximation models, preserve keyboard-accessible tabs and mobile
navigation, and run `npm audit --omit=dev --audit-level=high` before release.
