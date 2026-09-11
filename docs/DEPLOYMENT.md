# Deployment Guide

This document covers deploying `driftflow` in various environments.

## Installation

### From Source (Recommended)

```bash
git clone https://github.com/sachncs/driftflow.git
cd driftflow
pip install -e .
```

### As a Dependency

Add to your project's `pyproject.toml`:

```toml
[project]
dependencies = [
    "driftflow",
]

# Or pin to a version
dependencies = [
    "driftflow>=1.0.0",
]
```

### In a Requirements File

```
driftflow
```

## Environment Considerations

### Python Version

Requires Python 3.10 or later. Verify with:

```bash
python --version  # Must be >= 3.10
```

### No Runtime Dependencies

`driftflow` uses only the Python standard library. No additional system packages are required.

### Virtual Environments

Always install in a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate  # Linux/macOS
# or
.venv\Scripts\activate     # Windows

pip install driftflow
```

## Production Usage

### Basic Integration

```python
from driftflow import (
    CommonConfig,
    get_dataset_config,
    DVSSampler,
    LinearSchedule,
    make_drift_function,
)

# Initialize once at module level
common_config = CommonConfig()
dataset_configs = {
    ("GruM", "QM9"): get_dataset_config("GruM", "QM9"),
    ("GruM", "ZINC250k"): get_dataset_config("GruM", "ZINC250k"),
}

def generate_graph(drift_function, model, dataset, seed=None):
    """Generate a graph using DVS adaptive sampling."""
    config = dataset_configs[(model, dataset)]
    schedule = LinearSchedule(sigma_min=0.01, sigma_max=0.5)

    sampler = DVSSampler(
        drift_function=drift_function,
        noise_schedule=schedule,
        common_config=common_config,
        dataset_config=config,
        solver="Euler",
        seed=seed,
    )

    # Initial noise (shapes must match your graph dimensions)
    features_0 = [[0.0] * 4 for _ in range(9)]
    adjacency_0 = [[0.0] * 9 for _ in range(9)]

    features_t, adjacency_t, info = sampler.sample(
        initial_features=features_0,
        initial_adjacency=adjacency_0,
        terminal_time=1.0,
    )

    return features_t, adjacency_t, info
```

### Performance Considerations

- **Numerical stability:** Use the default `eps_num=1e-12` unless you have specific requirements.
- **Solver choice:** Euler is the fast first-order integrator; Heun is a second-order predictor-corrector that requires an extra drift evaluation per step. With zero noise the two reduce to Euler-equivalent; for non-zero noise Heun's accuracy advantage depends on the corrector sharing the Brownian path with the predictor, which the current implementation enforces (see `docs/MATH.md §9`).
- **Timestep bounds:** Adjust `dt_min` and `dt_max` if you observe convergence issues or excessive step counts.

### Resource Usage

| Resource | Typical Usage |
|----------|---------------|
| Memory | Proportional to graph size (nodes x features) |
| CPU | Pure Python; single-threaded. Consider NumPy vectorization for large graphs (see `docs/EXTENSIONS.md`) |
| Network | None (no external calls) |
| Disk | None (no caching) |

## Docker Deployment

### Dockerfile

```dockerfile
FROM python:3.12-slim

WORKDIR /app

# Copy and install package
COPY . .
RUN pip install --no-cache-dir .

# Copy application code
COPY your_app/ /app/your_app/

CMD ["python", "-m", "your_app.main"]
```

### Build and Run

```bash
docker build -t your-app .
docker run your-app
```

## Testing in Production

### Smoke Test

After deployment, verify the package works:

```python
from driftflow import (
    DVSSampler,
    CommonConfig,
    DatasetConfig,
    LinearSchedule,
)

config = CommonConfig()
dataset_config = DatasetConfig(
    model="Test",
    dataset="Test",
    kappa_ref=1.0,
    gamma_euler=0.5,
    gamma_heun=0.5,
    active_range=[(0.0, 1.0)],
)
schedule = LinearSchedule(sigma_min=0.01, sigma_max=0.5)

# Minimal test
def dummy_drift(X, A, t):
    return [[0.0] * len(X[0]) for _ in X], [[0.0] * len(A[0]) for _ in A]

sampler = DVSSampler(
    drift_function=dummy_drift,
    noise_schedule=schedule,
    common_config=config,
    dataset_config=dataset_config,
    solver="Euler",
    seed=42,
)

X, A, info = sampler.sample(
    initial_features=[[0.1]],
    initial_adjacency=[[0.1]],
    terminal_time=1.0,
)

assert info["total_steps"][0] > 0
print("Smoke test passed")
```

## Monitoring

Key metrics to monitor in production:

| Metric | Key | Healthy Range |
|--------|-----|---------------|
| Total steps | `info["total_steps"][0]` | 10-1000 (varies by problem) |
| Final time | `info["final_time"][0]` | Close to `terminal_time` |
| Feature DVS | `info["v_x"]`, `info["smoothed_x"]` | Should converge toward zero |
| Adjacency DVS | `info["v_a"]`, `info["smoothed_a"]` | Should converge toward zero |
| Timestep sizes | `info["dt"]` | Between `dt_min` and `dt_max` |

## Troubleshooting

### "Maximum iterations exceeded"

Increase `terminal_time` or adjust `dt_max` to allow larger steps.

### Numerical instability (NaN/Inf in output)

- Ensure `eps_num` is not too small
- Check that initial noise values are reasonable
- Verify drift function output is bounded

### Slow performance

- The sampler is pure Python. For large graphs, consider the NumPy vectorization extensions in `docs/EXTENSIONS.md`.
- Profile with `python -m cProfile your_script.py` to identify bottlenecks.

## Version Pinning

For reproducible deployments, pin the exact version:

```
driftflow==1.0.0
```

Or in `pyproject.toml`:

```toml
dependencies = [
    "driftflow==1.0.0",
]
```
