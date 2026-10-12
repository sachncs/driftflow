# Production inference

`driftflow` provides the validated sampling runtime and model contract. It
does not ship trained GruM or GDSS weights. Production inference therefore
requires a separately trained and validated model exported as TorchScript.

## Checkpoint contract

The scripted module must accept three inputs:

1. `features`: `float32` tensor with shape `(num_nodes, feature_dim)`.
2. `adjacency`: `float32` tensor with shape `(num_nodes, num_nodes)`.
3. `time`: scalar `float32` diffusion time in `[0, 1]`.

It must return exactly two tensors:

1. feature drift with shape `(num_nodes, feature_dim)`;
2. adjacency drift with shape `(num_nodes, num_nodes)`.

The adapter rejects missing checkpoints, digest mismatches, wrong shapes,
non-finite outputs, failed inference, and non-production models before the
solver can advance a trajectory.

## Loading a checkpoint

```python
from driftflow import (
    CommonConfig,
    DvsAdaptiveController,
    DVSSampler,
    GraphState,
    LinearSchedule,
    TorchScriptDriftModel,
    get_dataset_config,
)

model = TorchScriptDriftModel(
    "artifacts/grum-qm9-2.0.0.ts",
    num_nodes=9,
    feature_dim=5,
    device="cpu",
    expected_sha256="<recorded-release-digest>",
)
dataset = get_dataset_config("GruM", "QM9")
sampler = DVSSampler(
    drift_model=model,
    noise_schedule=LinearSchedule(0.01, 0.5),
    controller=DvsAdaptiveController(
        CommonConfig(), dataset, dataset.gamma_euler or 0.0
    ),
    production=True,
)
result = sampler.sample(GraphState.from_lists(features, adjacency), seed=42)
```

`production=True` deliberately rejects `SimpleGraphDenoiser`,
`GruMApproximation`, and `GDSSApproximation`. Those classes remain available
for smoke tests and educational examples only.

## Reproducible reference checkpoint

The repository includes a small deployable reference artifact under
`artifacts/`. It is a deterministic synthetic graph-drift model used to prove
the production path, not a replacement for the paper's trained GruM/GDSS
weights. Rebuild it with:

```bash
pip install -e ".[production]"
python scripts/build_reference_model.py --output-dir artifacts
python scripts/validate_reference_model.py --artifact-dir artifacts
python scripts/benchmark_reference_model.py --artifact-dir artifacts --iterations 100
```

The generated manifest records the checkpoint digest, 4-node/3-feature input
contract, canonical feature order, identity preprocessing, training seed,
validation loss, and eager-versus-TorchScript export error. The checked-in
checkpoint is below the 10 MB deployment budget. `from_manifest()` rejects
missing metadata, non-canonical feature order, and unsupported preprocessing
before loading the artifact.

The local Apple Silicon baseline is recorded in
`artifacts/reference-benchmark.json`; it is evidence for regression detection,
not a production SLO. Re-run the benchmark on the target host and set service
limits from that environment.

## Release gates

Before promoting a checkpoint, record and review:

- exact model and dataset versions, preprocessing, feature ordering, and
  graph-size limits;
- checkpoint SHA-256 and Torch/Python/runtime versions;
- numerical agreement with the eager reference implementation on a fixed golden
  corpus;
- quality metrics on held-out data and adversarial malformed inputs;
- latency, memory, throughput, and maximum-step behavior under representative
  load;
- deterministic replay for a fixed seed and reproducible environment;
- rollback artifact, monitoring thresholds, and an owner for incidents.

The CI suite verifies package contracts, type checks, coverage, dependency
auditing, artifact builds, and the approximation smoke test. It cannot prove
model quality or replace checkpoint-specific validation.
