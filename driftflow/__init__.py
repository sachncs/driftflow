"""Top-level package for the **driftflow** library.

``driftflow`` (Information-Geometric Adaptive Sampling for Graph Diffusion) is a
pure-Python implementation of the Drift Variation Score (DVS) adaptive sampler
introduced in the paper:

    Information-Geometric Adaptive Sampling for Graph Diffusion
    arXiv:2605.00250

The library provides a training-free, plug-in sampler that wraps an existing
graph-diffusion denoising network and adaptively chooses its SDE-solver
timestep based on the local Fisher-Rao curvature of the transition manifold.

Package layout
--------------
::

    driftflow/
        config    # Hyperparameter dataclasses (Tables 6 & 7 of the paper)
        sampler   # DVS sampler core (Algorithms 1-3) and solver steps
        models    # Simplified stand-in denoising networks for GruM / GDSS
        schedule  # Noise schedule g(t) parameterisations
        utils     # Small stateless helpers: clipping, ranges, decoding

Design rationale
----------------
The package is intentionally minimal: no NumPy, PyTorch, or other third-party
runtime dependencies.  All numerical primitives operate on nested
``list[float]`` containers so that the code can be embedded in environments
that restrict heavy scientific stacks (e.g. educational tools, lightweight
inference servers).

Every name exported here is part of the stable public API.  Submodules also
expose internal helpers (such as ``driftflow.sampler.squared_l2_difference``)
directly; these are public but intentionally not re-exported at the package
level, so the surface documented in ``docs/API_REFERENCE.md`` stays the
authoritative reference.

References:
----------
* Paper: https://arxiv.org/abs/2605.00250
* Extended math: ``docs/MATH.md``
* Architecture overview: ``docs/ARCHITECTURE.md``
"""

from .config import DATASET_CONFIGS, CommonConfig, DatasetConfig, get_dataset_config
from .models import GDSSApproximation, GruMApproximation, SimpleGraphDenoiser, make_drift_function
from .sampler import (
    SOLVERS,
    DVSSampler,
    compute_drift_variation_score,
    compute_timestep,
    euler_step,
    global_refresh,
    heun_step,
    register_solver,
    update_ema,
)
from .schedule import (
    CosineSchedule,
    LinearSchedule,
    NoiseSchedule,
    PolynomialSchedule,
    constant_schedule,
)
from .utils import clip_value, decode_adjacency, in_active_range, sigmoid_decode_adjacency
from .version import __version__

__all__ = [
    "__version__",
    # Config
    "CommonConfig",
    "DatasetConfig",
    "DATASET_CONFIGS",
    "get_dataset_config",
    # Models
    "GDSSApproximation",
    "GruMApproximation",
    "SimpleGraphDenoiser",
    "make_drift_function",
    # Sampler
    "DVSSampler",
    "SOLVERS",
    "register_solver",
    "compute_drift_variation_score",
    "update_ema",
    "compute_timestep",
    "global_refresh",
    "euler_step",
    "heun_step",
    # Schedule
    "CosineSchedule",
    "LinearSchedule",
    "NoiseSchedule",
    "PolynomialSchedule",
    "constant_schedule",
    # Utils
    "clip_value",
    "decode_adjacency",
    "in_active_range",
    "sigmoid_decode_adjacency",
]
