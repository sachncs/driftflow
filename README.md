<div align="center">

# driftflow

**Adaptive SDE sampling for graph diffusion, driven by the Drift Variation Score (DVS).**

A pure-Python reproduction of the information-geometric adaptive sampler.

[![Python](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.13-blue)](#installation)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![CI](https://img.shields.io/github/actions/workflow/status/sachncs/driftflow/ci.yml?branch=master)](https://github.com/sachncs/driftflow/actions)
[![Stars](https://img.shields.io/github/stars/sachncs/driftflow)](https://github.com/sachncs/driftflow/stargazers)
[![Checked with mypy](https://img.shields.io/badge/mypy-strict-green.svg)](https://mypy-lang.org/)

</div>

**driftflow** is a pure-Python implementation of the **Drift Variation Score
(DVS) adaptive sampler** for graph diffusion models. It reproduces the
training-free algorithms from the paper
[**Information-Geometric Adaptive Sampling for Graph Diffusion**](https://arxiv.org/abs/2605.00250)
(arXiv:2605.00250), wrapping any graph-diffusion drift function with an
SDE solver whose timestep is adapted from the local Fisher-Rao curvature
of the transition manifold.

The name *driftflow* evokes the two things the library is built around —
the **drift** of the diffusion SDE and the adaptive **flow** of the
sampling trajectory it produces. It supersedes the former package name
``igasgd`` while keeping the same public API.

The package targets three algorithms from the paper:

- **Algorithm 1** — DVS-driven adaptive sampler (meta-algorithm).
- **Algorithm 2** — DVS-Euler-Maruyama adaptive sampler.
- **Algorithm 3** — DVS-Heun adaptive sampler (predictor-corrector).

---

## Features

- **Drift Variation Score (DVS)** — Equations 13-15 implemented as
  composable Python functions for use in any graph-diffusion pipeline.
- **Euler-Maruyama and Heun solvers** — Both first- and second-order
  SDE solvers with a shared diffusion noise scale ``g(t) * sqrt(dt)``.
- **Hyperparameter catalogue** — Common (Table 6) and dataset-specific
  (Table 7) configurations for GruM and GDSS models, fully validated.
- **Active-range logic** — DVS is computed only inside configured time
  intervals (e.g. ``[0, 0.2] U [0.95, 1.0]`` for GDSS/QM9).
- **Noise schedules** — Linear, cosine, and polynomial parameterisations
  of ``g(t)``; users can supply their own callable.
- **Simplified denoiser approximations** — Educational stand-ins for
  GruM and GDSS with a documented drift interface contract.
- **Zero runtime dependencies** — Pure Python standard library
  implementation, no NumPy or PyTorch required.
- **Pluggable solver registry** — Add new SDE integrators via the
  ``SOLVERS`` / ``register_solver`` extension point without editing
  the sampler loop.
- **Fail-fast input validation** — Ragged/empty matrices and invalid
  terminal times raise descriptive ``ValueError``\ s before the loop.
- **Comprehensive test suite** — 165 tests covering mathematical
  correctness, edge cases, numerical stability, determinism, and
  input validation, at 100% line coverage.

---

## Installation

### From source

```bash
git clone https://github.com/sachncs/driftflow.git
cd driftflow
pip install -e .
```

### With dev dependencies

```bash
pip install -e ".[dev]"
```

**Requirements**: Python >= 3.10. No third-party runtime dependencies.

---

## Quick Start

### Python API

```python
from driftflow import (
    CommonConfig,
    DVSSampler,
    GruMApproximation,
    LinearSchedule,
    get_dataset_config,
    make_drift_function,
)

# Use the simplified approximation as a stand-in for a real denoising network
approx = GruMApproximation(num_nodes=9, feature_dim=4, seed=42)
drift = make_drift_function(approx)

# Load configuration from the paper
common = CommonConfig()
dataset = get_dataset_config("GruM", "QM9")

# Build and run the sampler
sampler = DVSSampler(
    drift_function=drift,
    noise_schedule=LinearSchedule(sigma_min=0.01, sigma_max=0.5),
    common_config=common,
    dataset_config=dataset,
    solver="Euler",
    seed=42,
)

# Initial noise
features_0 = [[0.1 for _ in range(4)] for _ in range(9)]
adjacency_0 = [[0.1 for _ in range(9)] for _ in range(9)]

# Sample a graph
features_t, adjacency_t, info = sampler.sample(
    initial_features=features_0,
    initial_adjacency=adjacency_0,
    terminal_time=1.0,
)

print(f"Steps: {info['total_steps'][0]}, Final time: {info['final_time'][0]}")
```

### CLI Demo

```bash
# Euler solver with GruM approximation
python examples/demo.py --model GruM --dataset QM9 --solver Euler --use-approximation

# Heun solver with GDSS approximation
python examples/demo.py --model GDSS --dataset QM9 --solver Heun --use-approximation

# Reproducible run with a custom seed
python examples/demo.py --model GruM --dataset QM9 --solver Euler --use-approximation --seed 42
```

### Solvers

```python
# First-order Euler-Maruyama (Algorithm 2)
sampler = DVSSampler(..., solver="Euler")

# Second-order Heun predictor-corrector (Algorithm 3)
sampler = DVSSampler(..., solver="Heun")
```

### Ablation Studies

```python
# Disable the adaptive timestep (always use dt_base)
# Construct a DatasetConfig with an empty active_range, e.g.
dataset = get_dataset_config("GruM", "QM9")
# ... and pass active_range=[] to a custom DatasetConfig instead.

# Use a constant noise schedule for deterministic testing
from driftflow import constant_schedule
schedule = constant_schedule(0.1)
```

---

## Configuration

### Common Hyperparameters (Table 6)

| Setting | Env Variable | Default | Description |
|---------|--------------|---------|-------------|
| `alpha` | — | `0.2` | EMA smoothing coefficient for the raw DVS (Equation 14). |
| `beta` | — | `0.5` | Sensitivity exponent in the power-law timestep rule. |
| `dt_base` | — | `1e-3` | Reference timestep; also used when DVS is inactive. |
| `dt_min` | — | `2e-4` | Lower bound on the adapted timestep. |
| `dt_max` | — | `5e-3` | Upper bound on the adapted timestep. |
| `eps_num` | — | `1e-12` | Numerical stabiliser added to denominators. |
| `eps_bound` | — | `1e-6` | Boundary tolerance for sampling-loop termination. |

### Dataset-Specific Hyperparameters (Table 7)

| Model | Dataset           | `kappa_ref` | `gamma_euler` | `gamma_heun` | Active Range              |
|-------|-------------------|-------------|---------------|--------------|---------------------------|
| GruM  | QM9               | `1.0`       | `0.22`        | `0.23`       | `[0, 1]`                  |
| GruM  | ZINC250k          | `5.0`       | `0.02`        | `0.04`       | `[0, 1]`                  |
| GruM  | Planar            | `10.0`      | `0.31`        | `0.30`       | `[0.5, 1.0]`              |
| GruM  | SBM               | `10.0`      | `0.26`        | `0.26`       | `[0.4, 1.0]`              |
| GDSS  | QM9               | `1.0`       | `0.68`        | --           | `[0, 0.2] U [0.95, 1.0]`  |
| GDSS  | Ego-small         | `0.2`       | --            | --           | `[0, 1]`                  |
| GDSS  | Grid              | `0.1`       | --            | --           | `[0, 1]`                  |
| GDSS  | Community-small   | `0.1`       | --            | --           | `[0, 1]`                  |

Retrieve any entry via `get_dataset_config(model, dataset)`.

### Noise Schedules

```python
from driftflow import LinearSchedule, CosineSchedule, PolynomialSchedule, constant_schedule

# Linear interpolation
schedule = LinearSchedule(sigma_min=0.01, sigma_max=0.5)

# Cosine schedule (approximate Improved-DDPM form)
schedule = CosineSchedule(offset=0.008)

# Polynomial schedule g(t) = sigma_max * t**exponent + sigma_min
schedule = PolynomialSchedule(exponent=2.0, sigma_min=0.0, sigma_max=1.0)

# Constant schedule (useful for deterministic testing)
schedule = constant_schedule(0.1)
```

See [`docs/MATH.md`](docs/MATH.md) for the exact equations and
[`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) for the data flow.

---

## API

| Symbol | Type | Description |
|--------|------|-------------|
| `DVSSampler` | class | Core DVS-driven adaptive sampler (Algorithms 1–3) |
| `SOLVERS` | dict | Registry mapping solver names to step functions |
| `register_solver(name, fn)` | function | Register a custom solver step (extension point) |
| `CommonConfig` | dataclass | Hyperparameters from Table 6 |
| `DatasetConfig` | dataclass | Hyperparameters from Table 7 |
| `get_dataset_config(model, dataset)` | function | Look up configuration for a (model, dataset) pair |
| `make_drift_function(approx)` | function | Adapt a denoiser approximation to the drift interface |
| `LinearSchedule(sigma_min, sigma_max)` | class | Linear noise schedule `g(t)` |
| `CosineSchedule(offset)` | class | Cosine-form noise schedule |
| `PolynomialSchedule(exponent, sigma_min, sigma_max)` | class | Power-law noise schedule |
| `constant_schedule(sigma)` | function | Constant `g(t) = sigma` schedule |
| `GruMApproximation(num_nodes, feature_dim, seed)` | class | Simplified GruM stand-in denoiser |
| `GDSSApproximation(num_nodes, feature_dim, seed)` | class | Simplified GDSS stand-in denoiser |
| `__version__` | str | Package version (single source in `version.py`) |

---

## Project Structure

```
driftflow/
├── driftflow/              # Package source code
│   ├── __init__.py         # Public API exports
│   ├── version.py          # Single source of version
│   ├── config.py           # Hyperparameter dataclasses (Tables 6 & 7)
│   ├── sampler.py          # Core DVS sampler (Algorithms 1-3), solver registry
│   ├── models.py           # Simplified denoiser approximations
│   ├── schedule.py         # Noise schedule callables
│   ├── utils.py            # Clipping, active ranges, decoding
│   └── py.typed            # PEP 561 type marker
├── tests/                  # Test suite (165 tests)
│   ├── test_config.py      # 15 tests
│   ├── test_models.py      # 23 tests
│   ├── test_sampler.py     # 82 tests
│   ├── test_schedule.py    # 18 tests
│   └── test_utils.py       # 27 tests
├── examples/               # Runnable demos with CLI
│   └── demo.py             # Synthetic drift + approximation demo
├── docs/                   # Extended documentation
│   ├── API_REFERENCE.md    # Complete public API docs
│   ├── ARCHITECTURE.md     # System design and data flow
│   ├── DEPLOYMENT.md       # Production deployment guide (in docs/)
│   ├── DEVELOPER_GUIDE.md  # Dev workflow, testing, linting
│   ├── EXTENSIONS.md       # Optional enhancements roadmap
│   ├── INDEX.md            # Documentation index
│   ├── MATH.md             # Equation-by-equation math docs
│   └── USAGE.md            # Step-by-step usage guide
├── pyproject.toml          # Build configuration
├── LICENSE                 # MIT License
├── CHANGELOG.md            # Version history
├── CONTRIBUTING.md         # Contribution guidelines
├── CODE_OF_CONDUCT.md      # Community standards
└── SECURITY.md             # Security policy
```

---

## Testing

```bash
pytest tests/ -v
pytest tests/ --cov=driftflow --cov-report=term-missing
```

---

## Build

```bash
python -m build
```

The version is read from a single source
(`driftflow/version.py`) by `[tool.setuptools.dynamic]`, so bumping
the version in one place updates the runtime attribute, the sdist, and
the wheel consistently.

---

## Release

Versions follow [Semantic Versioning](https://semver.org/). Releases are
tagged via `version:X.Y.Z` commits in [CHANGELOG.md](CHANGELOG.md).
Pushing a `v*` tag triggers the CI **publish** job, which builds the
sdist and wheel and uploads them to PyPI.

---

## Development

```bash
# Install with dev dependencies
pip install -e ".[dev]"

# Run the full local CI suite (lint + typecheck + tests + build + artifact verify)
bash scripts/ci.sh

# Run tests
pytest tests/ -v

# Run tests with coverage
pytest tests/ --cov=driftflow --cov-report=term-missing

# Lint
ruff check driftflow/ tests/ examples/

# Format
ruff format driftflow/ tests/ examples/

# Type check
mypy driftflow

# All checks
pytest && ruff check driftflow/ tests/ examples/ && mypy driftflow
```

### Code Style

- Line length: 100
- Quotes: double (`"`)
- Formatting: ruff (auto-format with `ruff format`)
- Type hints: required on all public signatures
- Docstrings: Google-style with "what" and "why"
- Pure Python: no third-party runtime dependencies

### Commit Conventions

We use [Conventional Commits](https://www.conventionalcommits.org/):

```
feat: add adaptive epsilon scheduling
fix: handle NaN in drift variation score
docs: add comprehensive module docstrings
refactor: extract helper for drift caching
test: add parity tests for Euler and Heun
chore: update ruff config
```

---

## Architecture

The sampler separates the **adaptive controller** (DVS computation,
EMA smoothing, power-law step-size scaling) from the **SDE solver**
(Euler-Maruyama or Heun) and the **denoiser network** (any callable
satisfying ``(X, A, t) -> (f_X, f_A)``).

```
+-----------------+      +------------------+      +---------------------+
|   User code     |      |    config.py     |      |    schedule.py      |
|  (drift net)    |      | CommonConfig /   |      |   NoiseSchedule     |
|                 |      | DatasetConfig    |      |     g(t)            |
+--------+--------+      +------------------+      +---------------------+
         |                        |                          |
         | drift_function         | common_config            | noise_schedule
         v                        v                          v
+--------------------------------------------------------------------+
|                          DVSSampler                                |
|  (sampler.py -- Algorithms 1-3)                                   |
|                                                                    |
|  while t < T:                                                      |
|    drift = drift_function(X, A, t)                                 |
|    if active:                                                      |
|      v_x, v_a = compute_drift_variation_score(...)                |
|      s_x, s_a = update_ema(v_x, v_a, ...)                         |
|      dt_x  = compute_timestep(s_x, ...)                           |
|      dt_a  = compute_timestep(s_a, ...)                           |
|      dt    = min(dt_x, dt_a)   # bottleneck                       |
|      s_x, s_a = global_refresh(s_x, s_a, gamma)                  |
|    else:                                                           |
|      dt = dt_base                                                  |
|    dt = clip(dt, 0, T - t)                                         |
|    X, A = SOLVERS[solver](...)   # dispatch via solver registry    |
+--------------------------------------------------------------------+
         |
         v
+--------------------------------------------------------------------+
|  Output: X_T, A_T, info_dict                                      |
|  Optional: decode_adjacency(A_T)                                  |
+--------------------------------------------------------------------+
```

### Mathematical Guarantees

1. **Shape preservation** — every solver step returns matrices with
   identical shapes to the inputs.
2. **Time monotonicity** — ``t`` never decreases; ``dt_k`` is clipped
   to ``[dt_min, dt_max]`` and then to ``[0, T - t]``.
3. **Determinism** — fixing the ``seed`` makes the entire trajectory
   bit-for-bit reproducible, including the per-step history in
   ``info_dict`` (verified by the test suite).
4. **Fail-fast validation** — empty, ragged, or otherwise malformed
   input matrices and non-finite terminal times raise descriptive
   ``ValueError``\ s before any work is done.
5. **Bottleneck synchrony** — both modalities share the same ``dt_k``,
   so the X and A timelines remain aligned at every step.
6. **Active-range inclusivity** — interval endpoints are inclusive.

See [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) for the full design
rationale, error-handling strategy, and extensibility points.

### Known Limitations

1. **Denoising networks** — the actual GruM/GDSS architectures and
   pretrained weights are not publicly released. The included
   approximations satisfy the drift interface but do not perform real
   message passing or graph attention.
2. **Noise schedule g(t)** — the exact functional form used in the
   paper is not stated. Common parameterisations are provided; users
   can supply custom callables.
3. **Evaluation metrics** — FCD and NSPDK metrics require RDKit and
   graph-kernel libraries, which are out of scope.
4. **Training code** — out of scope; this is a sampler-only reproduction.

---

## Tech Stack

| Category       | Technology                                          |
|----------------|-----------------------------------------------------|
| Language       | Python 3.10+                                        |
| Numerical      | Pure Python standard library (no NumPy/SciPy)       |
| Lint/Format    | [ruff](https://docs.astral.sh/ruff/)                |
| Type Check     | [mypy](https://mypy-lang.org/) (strict)              |
| Testing        | [pytest](https://docs.pytest.org/) + pytest-cov      |
| Build          | setuptools (via `pyproject.toml`)                   |
| Docs           | Markdown in `docs/`                                 |
| CI/CD          | GitHub Actions (lint, type-check, test, build, publish) |

---

## Roadmap

Planned enhancements tracked in [`docs/EXTENSIONS.md`](docs/EXTENSIONS.md):

- [x] Pluggable solver registry (`SOLVERS` / `register_solver`)
- [ ] Adaptive epsilon scheduling
- [ ] Soft clipping with smooth transitions
- [ ] NumPy vectorized operations for performance
- [ ] Numba JIT compilation support
- [ ] Structured logging with configurable verbosity
- [ ] Matplotlib visualization utilities
- [ ] Synthetic drift test suites
- [ ] Regression test harness
- [ ] Additional model architecture approximations

---

## Contributing

We welcome contributions! See [CONTRIBUTING.md](CONTRIBUTING.md) for:

- Development setup
- Pull request process
- Coding standards
- Test expectations

## Code of Conduct

This project follows the [Contributor Covenant v2.1](CODE_OF_CONDUCT.md).
By participating you agree to abide by its terms.

## Security

Report vulnerabilities to **sachncs@gmail.com** — see [SECURITY.md](SECURITY.md).

## License

[MIT](LICENSE) © 2026 Sachin

## Citation

If you use this reproduction in academic work, please cite the original paper:

```bibtex
@article{driftflow2026,
  title={Information-Geometric Adaptive Sampling for Graph Diffusion},
  journal={arXiv preprint arXiv:2605.00250},
  year={2026}
}
```