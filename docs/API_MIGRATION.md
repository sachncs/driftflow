# v1 to v2 Migration

v2 is a clean-break API redesign.

| v1 | v2 |
| --- | --- |
| Nested list state arguments | Immutable `GraphState` |
| Callable drift function | `DriftModel.evaluate` |
| Callable schedule | `NoiseSchedule.value_at` |
| `common_config` + `dataset_config` inside sampler | `DvsAdaptiveController` |
| String solver plus global `SOLVERS` | `SolverRegistry` plus `Solver` objects |
| `(X, A, info)` tuple | `SampleResult` |
| `info` dictionary | `SamplerDiagnostics` and `StepDiagnostics` |
| Sampler-owned persistent seed | Per-call `sample(..., seed=...)` |

The v1 global registry and old solver callback signature are removed. Update
integrations by introducing the corresponding contract implementation and
constructing an explicit controller and registry.
