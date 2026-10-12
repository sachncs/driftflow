"""Exception hierarchy for driftflow's validated runtime contracts."""


class DriftflowError(Exception):
    """Base class for all expected driftflow errors."""


class ConfigurationError(DriftflowError, ValueError):
    """Raised when a sampler, schedule, or solver configuration is invalid."""


class ShapeError(DriftflowError, ValueError):
    """Raised when a graph state or drift has an incompatible shape."""


class NumericalStabilityError(DriftflowError, ValueError):
    """Raised when a non-finite value would enter a numerical operation."""


class SolverContractError(DriftflowError, TypeError):
    """Raised when a solver violates the state transition contract."""


class ScheduleError(DriftflowError, ValueError):
    """Raised when a noise schedule produces an invalid value."""


class SamplingLimitError(DriftflowError, RuntimeError):
    """Raised when sampling exceeds its configured safety limit."""
