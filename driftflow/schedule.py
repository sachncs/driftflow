"""Validated polymorphic noise schedules for diffusion sampling."""

from math import cos, isfinite, pi

from .contracts import NoiseSchedule
from .errors import ConfigurationError, ScheduleError


def validate_time(time: float) -> float:
    """Validate and return a finite diffusion time."""
    if isinstance(time, bool) or not isinstance(time, (int, float)) or not isfinite(time):
        raise ScheduleError(f"time must be finite, got {time!r}")
    if time < 0.0 or time > 1.0:
        raise ScheduleError(f"time must be within [0, 1], got {time!r}")
    return time


def validate_noise_scale(value: float, name: str) -> float:
    """Validate and return a finite non-negative noise scale."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ScheduleError(f"{name} must be a real number")
    if not isfinite(value) or value < 0.0:
        raise ScheduleError(f"{name} must be finite and non-negative, got {value!r}")
    return value


class LinearSchedule(NoiseSchedule):
    """Linearly interpolate between two non-negative noise scales."""

    def __init__(self, sigma_min: float = 0.01, sigma_max: float = 1.0) -> None:
        """Initialize the schedule.

        Args:
            sigma_min: Noise scale at diffusion time zero.
            sigma_max: Noise scale at diffusion time one.

        Raises:
            ScheduleError: If either scale is invalid.
        """
        self.sigma_min = validate_noise_scale(sigma_min, "sigma_min")
        self.sigma_max = validate_noise_scale(sigma_max, "sigma_max")
        if self.sigma_min > self.sigma_max:
            raise ScheduleError("sigma_min must not exceed sigma_max")

    def value_at(self, time: float) -> float:
        """Return the interpolated noise scale at ``time``."""
        time = validate_time(time)
        return self.sigma_min + (self.sigma_max - self.sigma_min) * time


class CosineSchedule(NoiseSchedule):
    """Monotonic cosine-ramp noise schedule on the unit diffusion interval."""

    def __init__(
        self,
        offset: float = 0.008,
        sigma_min: float = 0.0,
        sigma_max: float = 1.0,
    ) -> None:
        """Initialize the schedule.

        Args:
            offset: Positive endpoint offset used by the cosine ramp.
            sigma_min: Noise scale at the beginning of the ramp.
            sigma_max: Noise scale at the end of the ramp.

        Raises:
            ConfigurationError: If the parameters are invalid.
        """
        if isinstance(offset, bool) or not isinstance(offset, (int, float)):
            raise ConfigurationError("offset must be a real number")
        if not isfinite(offset) or offset < 0.0 or offset >= 1.0:
            raise ConfigurationError("offset must be finite and within [0, 1)")
        self.offset = offset
        self.sigma_min = validate_noise_scale(sigma_min, "sigma_min")
        self.sigma_max = validate_noise_scale(sigma_max, "sigma_max")
        if self.sigma_min > self.sigma_max:
            raise ScheduleError("sigma_min must not exceed sigma_max")

    def value_at(self, time: float) -> float:
        """Return the cosine-ramp noise scale at ``time``."""
        time = validate_time(time)
        normalized_time = (time + self.offset) / (1.0 + self.offset)
        angle = min(pi, max(0.0, normalized_time * pi))
        fraction = (1.0 - cos(angle)) / 2.0
        return self.sigma_min + (self.sigma_max - self.sigma_min) * fraction


class PolynomialSchedule(NoiseSchedule):
    """Power-law noise schedule with validated non-negative parameters."""

    def __init__(
        self,
        exponent: float = 1.0,
        sigma_min: float = 0.01,
        sigma_max: float = 1.0,
    ) -> None:
        """Initialize the schedule.

        Args:
            exponent: Non-negative power applied to normalized time.
            sigma_min: Noise scale floor.
            sigma_max: Noise scale contribution at time one.

        Raises:
            ConfigurationError: If the parameters are invalid.
        """
        if isinstance(exponent, bool) or not isinstance(exponent, (int, float)):
            raise ConfigurationError("exponent must be a real number")
        if not isfinite(exponent) or exponent < 0.0:
            raise ConfigurationError("exponent must be finite and non-negative")
        self.exponent = exponent
        self.sigma_min = validate_noise_scale(sigma_min, "sigma_min")
        self.sigma_max = validate_noise_scale(sigma_max, "sigma_max")
        if self.sigma_min > self.sigma_max:
            raise ScheduleError("sigma_min must not exceed sigma_max")

    def value_at(self, time: float) -> float:
        """Return the power-law noise scale at ``time``."""
        time = validate_time(time)
        powered_time = float(time**self.exponent)
        return self.sigma_min + (self.sigma_max - self.sigma_min) * powered_time


class ConstantSchedule(NoiseSchedule):
    """Noise schedule returning one fixed scale for every valid time."""

    def __init__(self, value: float = 1.0) -> None:
        """Initialize the constant schedule."""
        self.value = validate_noise_scale(value, "value")

    def value_at(self, time: float) -> float:
        """Return the configured scale after validating ``time``."""
        validate_time(time)
        return self.value


def constant_schedule(value: float = 1.0) -> ConstantSchedule:
    """Create a validated constant noise schedule."""
    return ConstantSchedule(value)
