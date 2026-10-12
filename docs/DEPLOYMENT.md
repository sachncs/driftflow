# Deployment

driftflow 2.0.0 is a typed library release and a static documentation site.
The runtime has no network or service dependency; deploy the package and site
independently.

## Package build

```bash
python3 -m pip install -e ".[dev]"
PYTHON=.venv/bin/python ./scripts/ci.sh
```

The CI script builds the sdist and wheel and runs pip-audit with the editable
package excluded from the vulnerability scan. Install the wheel in a clean
environment before publishing it.

## Library integration

Models, schedules, controllers, and solvers are explicit collaborators:

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
    features=[[0.0, 0.0], [0.0, 0.0]],
    adjacency=[[0.0, 0.0], [0.0, 0.0]],
)
model = GruMApproximation(num_nodes=2, feature_dim=2, seed=7)
dataset = get_dataset_config("GruM", "QM9")
controller = DvsAdaptiveController(CommonConfig(), dataset, gamma=0.22)
sampler = DVSSampler(model, LinearSchedule(0.01, 0.5), controller)

result = sampler.sample(state, terminal_time=1.0, seed=42)
print(result.diagnostics.total_steps, result.diagnostics.final_time)
```

For a custom denoiser, implement `DriftModel.evaluate` and return a validated
`DriftState`. The approximation models included in this repository are
educational stand-ins, not trained production denoisers.

## Container smoke test

```dockerfile
FROM python:3.12-slim
WORKDIR /app
COPY . .
RUN python -m pip install --no-cache-dir .
CMD ["python", "examples/demo.py", "--model", "GruM", "--dataset", "QM9", "--solver", "Euler-Maruyama", "--use-approximation"]
```

## Static site

```bash
cd site
npm ci
npm run build
```

The Pages workflow publishes `site/dist` and copies the generated index to
`404.html` when Astro has not produced one. The site is configured for the
project path `/driftflow`.

## Operational checks

Monitor `SampleResult.diagnostics` rather than parsing logs. In particular,
track total steps, final time, timestep bounds, and failure counts by
`ConfigurationError`, `ShapeError`, `NumericalStabilityError`, and
`SamplingLimitError`. Set an application-level timeout around sampling and
keep `max_steps` finite for every deployment.
