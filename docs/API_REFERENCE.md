# API Reference

Complete reference for every public module, class, function, and constant.

---

## Table of Contents

- [driftflow.config](#driftflowconfig)
- [driftflow.sampler](#driftflowsampler)
- [driftflow.models](#driftflowmodels)
- [driftflow.schedule](#driftflowschedule)
- [driftflow.utils](#driftflowutils)

---

## `driftflow.config`

### `CommonConfig`

```python
@dataclass(frozen=True)
class CommonConfig:
    alpha: float = 0.2
    beta: float = 0.5
    dt_base: float = 1e-3
    dt_min: float = 2e-4
    dt_max: float = 5e-3
    eps_num: float = 1e-12
    eps_bound: float = 1e-6
```

Common hyperparameters from Table 6.  The dataclass is frozen so instances are immutable and hashable.

### `DatasetConfig`

```python
@dataclass(frozen=True)
class DatasetConfig:
    model: str
    dataset: str
    kappa_ref: float
    gamma_euler: Optional[float] = None
    gamma_heun: Optional[float] = None
    active_range: List[Tuple[float, float]] = field(default_factory=list)
```

Dataset-specific hyperparameters from Table 7.

**Methods:**
- `is_active(time: float) -> bool`  
  Returns `True` if `time` lies inside any of the configured intervals (inclusive).  An empty `active_range` means always active.

### `DATASET_CONFIGS`

```python
DATASET_CONFIGS: dict[tuple[str, str], DatasetConfig]
```

Global dictionary containing all eight configurations from Table 7.

### `get_dataset_config(model, dataset)`

```python
def get_dataset_config(model: str, dataset: str) -> DatasetConfig:
```

Lookup helper.  Raises `KeyError` with a list of available keys if the pair is not found.

---

## `driftflow.sampler`

### Type Aliases

```python
DriftFunction = Callable[[List[List[float]], List[List[float]], float],
                         Tuple[List[List[float]], List[List[float]]]]
NoiseSchedule = Callable[[float], float]
```

### `compute_drift_variation_score(...)`

```python
def compute_drift_variation_score(
    current_drift_x: List[List[float]],
    previous_drift_x: List[List[float]],
    current_drift_a: List[List[float]],
    previous_drift_a: List[List[float]],
    noise_scale: float,
    eps_num: float,
) -> Tuple[float, float]
```

Implements Equation 13: Drift Variation Score.

Computes the squared L2 norm of the drift difference for both modalities and divides by `noise_scale^2 + eps_num`.

**Raises:**
- `ValueError` if any drift matrix pair has mismatched row or column counts.

### `update_ema(...)`

```python
def update_ema(
    v_x: float,
    v_a: float,
    smoothed_x: float,
    smoothed_a: float,
    alpha: float,
) -> Tuple[float, float]
```

Implements Equation 14: EMA smoothing.

Returns `(alpha * v_x + (1 - alpha) * smoothed_x, alpha * v_a + (1 - alpha) * smoothed_a)`.

### `compute_timestep(...)`

```python
def compute_timestep(
    smoothed_score: float,
    kappa_ref: float,
    dt_base: float,
    dt_min: float,
    dt_max: float,
    beta: float,
    eps_num: float,
) -> float
```

Implements Equation 15: power-law step-size scaling with clipping.

Returns `clip(dt_base * (kappa_ref / (smoothed_score + eps_num))^beta, dt_min, dt_max)`.

### `global_refresh(...)`

```python
def global_refresh(
    smoothed_x: float,
    smoothed_a: float,
    gamma: float,
) -> Tuple[float, float]
```

Implements the global variation refresh.

Returns `(gamma * (smoothed_x + smoothed_a), gamma * (smoothed_x + smoothed_a))`.

### `euler_step(...)`

```python
def euler_step(
    features: List[List[float]],
    adjacency: List[List[float]],
    drift_features: List[List[float]],
    drift_adjacency: List[List[float]],
    timestep: float,
    noise_scale: float,
    rng: random.Random,
) -> Tuple[List[List[float]], List[List[float]]]
```

Implements Equation 2 / Algorithm 2: Euler-Maruyama update.

`noise_scale` should already be `g_t * sqrt(timestep)`.

### `heun_step(...)`

```python
def heun_step(
    features: List[List[float]],
    adjacency: List[List[float]],
    first_drift_features: List[List[float]],
    first_drift_adjacency: List[List[float]],
    timestep: float,
    noise_scale: float,
    drift_function: DriftFunction,
    time: float,
    rng: random.Random,
) -> Tuple[List[List[float]], List[List[float]]]
```

Implements Algorithm 3: Heun predictor-corrector update.

Evaluates the corrector drift at `time + timestep` using the provided `drift_function`.

### `SOLVERS`

```python
SOLVERS: dict[str, SolverStep]
```

Registry mapping solver names (the `solver=` values accepted by `DVSSampler`)
to step functions.  Populated at import time with `"Euler"` and `"Heun"`.

### `register_solver(name, step_function)`

```python
def register_solver(name: str, step_function: SolverStep) -> None:
```

Registers a custom solver step under `name`, making it usable as a `solver=`
value for `DVSSampler`.  The step function must have the uniform signature:

```python
(X, A, f_X, f_A, dt, noise, rng, drift_function, time) -> (X_next, A_next)
```

A registered solver still resolves its aggregation factor `gamma` from the
dataset configuration (`gamma_euler` for `"Euler"`, `gamma_heun` for all
other solvers); subclasses may override `DVSSampler.resolve_gamma`.

### `DVSSampler`

```python
class DVSSampler:
    def __init__(
        self,
        drift_function: DriftFunction,
        noise_schedule: NoiseSchedule,
        common_config: CommonConfig,
        dataset_config: DatasetConfig,
        solver: str = "Euler",
        seed: Optional[int] = None,
    ) -> None
```

**Constructor arguments:**
- `drift_function`: Callable `(X, A, t) -> (f_X, f_A)`.
- `noise_schedule`: Callable `t -> g_t`.
- `common_config`: Table 6 hyperparameters.
- `dataset_config`: Table 7 hyperparameters.
- `solver`: Registered solver name (`"Euler"`, `"Heun"`, or any solver
  registered via `register_solver`).
- `seed`: Optional integer for reproducible noise.

**Raises:**
- `ValueError` if `solver` is not registered in `SOLVERS`.
- `ValueError` if the dataset config does not provide a `gamma` for the requested solver.

**Properties:**
- `gamma: float` — The aggregation factor used by this sampler instance.

**Methods:**
- `sample(initial_features, initial_adjacency, terminal_time=1.0, verbose=False) -> Tuple[List[List[float]], List[List[float]], Dict[str, List[float]]]`  
  Runs the adaptive sampling loop from `t = 0` to `terminal_time`.  Returns `(X_T, A_T, info)` where `info` contains step histories.
  Validates that both matrices are non-empty and rectangular and that
  `terminal_time` is finite and non-negative, raising `ValueError` otherwise.

**`info` dictionary keys:**

| Key | Shape | Description |
|-----|-------|-------------|
| `steps` | `[N]` | Step index at each iteration |
| `dt` | `[N]` | Actual timestep used at each iteration |
| `time` | `[N]` | Diffusion time *before* taking the step |
| `v_x` | `[N]` | Raw DVS for features (0.0 when inactive) |
| `v_a` | `[N]` | Raw DVS for adjacency (0.0 when inactive) |
| `smoothed_x` | `[N]` | EMA-smoothed DVS for features |
| `smoothed_a` | `[N]` | EMA-smoothed DVS for adjacency |
| `total_steps` | `[1]` | Total number of steps taken |
| `final_time` | `[1]` | Final diffusion time reached |

---

## `driftflow.models`

### `SimpleGraphDenoiser`

```python
class SimpleGraphDenoiser:
    def __init__(
        self,
        num_nodes: int,
        feature_dim: int,
        hidden_dim: int = 16,
        seed: int = 42,
    ) -> None
```

Simplified time-conditioned MLP denoising network.  Satisfies the drift interface.

**Methods:**
- `__call__(features, adjacency, time) -> Tuple[List[List[float]], List[List[float]]]`  
  Predicts the drift for every node.

### `GruMApproximation`

```python
class GruMApproximation(SimpleGraphDenoiser):
    def __init__(self, num_nodes: int, feature_dim: int, seed: int = 42) -> None
```

Stand-in for the GruM model with `hidden_dim = 32`.

### `GDSSApproximation`

```python
class GDSSApproximation(SimpleGraphDenoiser):
    def __init__(self, num_nodes: int, feature_dim: int, seed: int = 42) -> None
```

Stand-in for the GDSS model with `hidden_dim = 24`.

### `make_drift_function(model)`

```python
def make_drift_function(model: SimpleGraphDenoiser) -> callable
```

Wraps a model instance into the standard drift interface `(X, A, t) -> (f_X, f_A)`.

---

## `driftflow.schedule`

### `NoiseSchedule`

Type alias: `Callable[[float], float]`

### `LinearSchedule`

```python
class LinearSchedule:
    def __init__(self, sigma_min: float = 0.01, sigma_max: float = 1.0) -> None
    def __call__(self, time: float) -> float
```

`g(t) = sigma_max * t + sigma_min * (1 - t)`

### `CosineSchedule`

```python
class CosineSchedule:
    def __init__(self, offset: float = 0.008) -> None
    def __call__(self, time: float) -> float
```

Approximate cosine schedule: `g(t) = tan((t + offset) / (1 + offset) * pi / 2)`.

### `PolynomialSchedule`

```python
class PolynomialSchedule:
    def __init__(self, exponent: float = 1.0, sigma_min: float = 0.01,
                 sigma_max: float = 1.0) -> None
    def __call__(self, time: float) -> float
```

`g(t) = sigma_max * t^exponent + sigma_min`

### `constant_schedule(value)`

```python
def constant_schedule(value: float = 1.0) -> NoiseSchedule
```

Returns a schedule that always returns `value`, useful for deterministic testing.

---

## `driftflow.utils`

### `clip_value(...)`

```python
def clip_value(value: float, lower_bound: float, upper_bound: float) -> float
```

Clamps `value` to `[lower_bound, upper_bound]`.  If `value` is `NaN`, returns `NaN`.

### `in_active_range(...)`

```python
def in_active_range(time: float, ranges: List[Tuple[float, float]]) -> bool
```

Returns `True` if `time` lies inside any interval (inclusive).  An empty list means always active.

### `decode_adjacency(...)`

```python
def decode_adjacency(adjacency: List[List[float]],
                     threshold: float = 0.5) -> List[List[float]]
```

Binarizes a continuous adjacency matrix: `1.0` if `value >= threshold`, else `0.0`.

### `sigmoid_decode_adjacency(...)`

```python
def sigmoid_decode_adjacency(adjacency: List[List[float]],
                             threshold: float = 0.5) -> List[List[float]]
```

Applies `sigmoid(x)` to each entry, then thresholds at `threshold`.
