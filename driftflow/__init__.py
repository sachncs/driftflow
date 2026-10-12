"""Typed adaptive SDE sampling for graph diffusion."""

from .config import DATASET_CONFIGS, CommonConfig, DatasetConfig, get_dataset_config
from .contracts import AdaptiveController, DriftModel, GraphDecoder, NoiseSchedule, Solver
from .domain import (
    AdaptiveDecision,
    DriftState,
    GraphState,
    SamplerDiagnostics,
    SampleResult,
    StepDiagnostics,
)
from .errors import (
    ConfigurationError,
    DriftflowError,
    NumericalStabilityError,
    SamplingLimitError,
    ScheduleError,
    ShapeError,
    SolverContractError,
)
from .models import GDSSApproximation, GruMApproximation, SimpleGraphDenoiser
from .sampler import (
    DvsAdaptiveController,
    DVSSampler,
    EulerMaruyamaSolver,
    HeunSolver,
    SolverRegistry,
    compute_drift_variation_score,
    compute_timestep,
    update_ema,
)
from .schedule import (
    ConstantSchedule,
    CosineSchedule,
    LinearSchedule,
    PolynomialSchedule,
    constant_schedule,
)
from .utils import (
    SigmoidGraphDecoder,
    ThresholdGraphDecoder,
    decode_adjacency,
    sigmoid_decode_adjacency,
)
from .version import __version__

__all__ = [
    "__version__",
    "AdaptiveController",
    "AdaptiveDecision",
    "CommonConfig",
    "ConfigurationError",
    "ConstantSchedule",
    "CosineSchedule",
    "DATASET_CONFIGS",
    "DatasetConfig",
    "DVSSampler",
    "DriftModel",
    "DriftState",
    "DriftflowError",
    "DvsAdaptiveController",
    "EulerMaruyamaSolver",
    "GDSSApproximation",
    "GraphDecoder",
    "GraphState",
    "GruMApproximation",
    "HeunSolver",
    "LinearSchedule",
    "NoiseSchedule",
    "NumericalStabilityError",
    "PolynomialSchedule",
    "SampleResult",
    "SamplerDiagnostics",
    "SamplingLimitError",
    "ScheduleError",
    "ShapeError",
    "SigmoidGraphDecoder",
    "SimpleGraphDenoiser",
    "Solver",
    "SolverContractError",
    "SolverRegistry",
    "StepDiagnostics",
    "ThresholdGraphDecoder",
    "compute_drift_variation_score",
    "compute_timestep",
    "constant_schedule",
    "decode_adjacency",
    "get_dataset_config",
    "sigmoid_decode_adjacency",
    "update_ema",
]
