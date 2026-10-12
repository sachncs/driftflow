"""Validated immutable sampler configuration objects."""

from dataclasses import dataclass
from math import isfinite

from .errors import ConfigurationError


def validate_finite(value: float, name: str) -> float:
    """Validate and return a finite floating-point value."""
    if not isfinite(value):
        raise ConfigurationError(f"{name} must be finite, got {value!r}")
    return value


@dataclass(frozen=True, slots=True)
class CommonConfig:
    """Shared validated parameters for adaptive sampling."""

    alpha: float = 0.2
    beta: float = 0.5
    dt_base: float = 1e-3
    dt_min: float = 2e-4
    dt_max: float = 5e-3
    eps_num: float = 1e-12
    eps_bound: float = 1e-6
    max_steps: int = 1_000_000

    def __post_init__(self) -> None:
        """Validate controller and integration limits."""
        for name, value in (
            ("alpha", self.alpha),
            ("beta", self.beta),
            ("dt_base", self.dt_base),
            ("dt_min", self.dt_min),
            ("dt_max", self.dt_max),
            ("eps_num", self.eps_num),
            ("eps_bound", self.eps_bound),
        ):
            validate_finite(value, name)
        if not 0.0 <= self.alpha <= 1.0:
            raise ConfigurationError("alpha must be within [0, 1]")
        if self.beta < 0.0:
            raise ConfigurationError("beta must be non-negative")
        if self.dt_min <= 0.0 or self.dt_min > self.dt_base or self.dt_base > self.dt_max:
            raise ConfigurationError("dt_min <= dt_base <= dt_max must hold with positive values")
        if self.eps_num <= 0.0 or self.eps_bound < 0.0:
            raise ConfigurationError("eps_num must be positive and eps_bound non-negative")
        if self.max_steps <= 0:
            raise ConfigurationError("max_steps must be positive")


@dataclass(frozen=True, slots=True)
class DatasetConfig:
    """Validated model/dataset-specific controller parameters."""

    model: str
    dataset: str
    kappa_ref: float
    gamma_euler: float | None = None
    gamma_heun: float | None = None
    active_range: tuple[tuple[float, float], ...] = ()

    def __post_init__(self) -> None:
        """Validate dataset metadata, controller coefficients, and ranges."""
        if not self.model or not self.dataset:
            raise ConfigurationError("model and dataset must be non-empty")
        validate_finite(self.kappa_ref, "kappa_ref")
        if self.kappa_ref <= 0.0:
            raise ConfigurationError("kappa_ref must be positive")
        for name, value in (("gamma_euler", self.gamma_euler), ("gamma_heun", self.gamma_heun)):
            if value is not None:
                validate_finite(value, name)
                if value < 0.0:
                    raise ConfigurationError(f"{name} must be non-negative")
        normalized_ranges = tuple(tuple(interval) for interval in self.active_range)
        for interval in normalized_ranges:
            if len(interval) != 2:
                raise ConfigurationError("active ranges must contain (start, end) pairs")
            start, end = interval
            validate_finite(start, "active range start")
            validate_finite(end, "active range end")
            if not 0.0 <= start <= end <= 1.0:
                raise ConfigurationError("active ranges must satisfy 0 <= start <= end <= 1")
        object.__setattr__(self, "active_range", normalized_ranges)

    def is_active(self, time: float) -> bool:
        """Return whether DVS adaptation is enabled at ``time``."""
        if not self.active_range:
            return True
        return any(start <= time <= end for start, end in self.active_range)


DATASET_CONFIGS: dict[tuple[str, str], DatasetConfig] = {
    ("GruM", "QM9"): DatasetConfig("GruM", "QM9", 1.0, 0.22, 0.23, ((0.0, 1.0),)),
    ("GruM", "ZINC250k"): DatasetConfig("GruM", "ZINC250k", 5.0, 0.02, 0.04, ((0.0, 1.0),)),
    ("GruM", "Planar"): DatasetConfig("GruM", "Planar", 10.0, 0.31, 0.30, ((0.5, 1.0),)),
    ("GruM", "SBM"): DatasetConfig("GruM", "SBM", 10.0, 0.26, 0.26, ((0.4, 1.0),)),
    ("GDSS", "QM9"): DatasetConfig("GDSS", "QM9", 1.0, 0.68, None, ((0.0, 0.2), (0.95, 1.0))),
    ("GDSS", "Ego-small"): DatasetConfig("GDSS", "Ego-small", 0.2, None, None, ((0.0, 1.0),)),
    ("GDSS", "Grid"): DatasetConfig("GDSS", "Grid", 0.1, None, None, ((0.0, 1.0),)),
    ("GDSS", "Community-small"): DatasetConfig(
        "GDSS", "Community-small", 0.1, None, None, ((0.0, 1.0),)
    ),
}


def get_dataset_config(model: str, dataset: str) -> DatasetConfig:
    """Return the validated configuration for a model/dataset pair."""
    try:
        return DATASET_CONFIGS[(model, dataset)]
    except KeyError as error:
        available = ", ".join(f"{item[0]}/{item[1]}" for item in DATASET_CONFIGS)
        raise ConfigurationError(
            f"No config for {model}/{dataset}. Available configurations: {available}"
        ) from error
