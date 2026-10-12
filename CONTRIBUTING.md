# Contributing to driftflow

Thank you for contributing. Changes should preserve the typed v2 API, the
Google Python style guide, and the repository's explicit reliability contracts.

## Development setup

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
```

Run the same checks used locally and in CI:

```bash
PYTHON=.venv/bin/python ./scripts/check.sh
PYTHON=.venv/bin/python ./scripts/ci.sh
```

These commands lint and format-check Python, run mypy, run pytest with the
90% coverage floor, execute the demo, build the Astro site and Python package,
and audit Python and site dependencies.

## Coding standards

- Follow the [Google Python Style Guide](https://google.github.io/styleguide/pyguide.html).
- Use complete type annotations and Google-style docstrings for public APIs.
- Use `snake_case` for functions and variables, `PascalCase` for classes, and
  `UPPER_CASE` for module constants.
- Do not use underscore-prefixed members as a semi-private API convention.
- Keep public extension points in `driftflow/contracts.py` and public data in
  frozen domain objects.
- Validate shape, finiteness, and bounds at numerical boundaries.
- Do not add global mutable registries or hidden process state.

## Tests

Tests live in `tests/` and should cover both successful behavior and failure
contracts. The current layout is:

- `test_domain.py`: immutable graph and drift states.
- `test_v2_config.py`: configuration validation.
- `test_v2_components.py`: schedules, models, matrix helpers, and decoders.
- `test_v2_sampler.py`: integration, polymorphism, reproducibility, and safety.

Add a regression test for every bug fix. Use pytest node IDs from the current
files when running a focused test, for example:

```bash
python -m pytest tests/test_v2_sampler.py -q
python -m pytest tests/test_v2_sampler.py::test_sampler_is_reproducible_per_call_seed -q
```

## Extension workflow

Implement `DriftModel`, `NoiseSchedule`, `AdaptiveController`, `Solver`, or
`GraphDecoder` rather than adding special cases to `DVSSampler`. Register
custom solvers on an instance-local `SolverRegistry`; never modify a global
registry. Return `GraphState` and `DriftState` objects from model and solver
boundaries.

## Documentation and site

Update the relevant file in `docs/` and add a migration note when changing a
public contract. Keep examples executable against the current API. For the
site:

```bash
cd site
npm ci
npm run build
npm audit --omit=dev --audit-level=high
```

The site is a static Astro project deployed from `site/dist` under the
`/driftflow` project path.

## Commits and pull requests

Use Conventional Commit messages and keep commits atomic. Before opening a
pull request, update `CHANGELOG.md` under `Unreleased`, explain behavior and
test changes, and confirm the complete local gate passes.
