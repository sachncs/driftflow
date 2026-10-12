"""Polymorphic contracts shared by driftflow's sampling components."""

from abc import ABC, abstractmethod
from collections.abc import Sequence
from random import Random

from .domain import AdaptiveDecision, DriftState, GraphState


class DriftModel(ABC):
    """Abstract graph drift model evaluated at a diffusion time."""

    @abstractmethod
    def evaluate(self, state: GraphState, time: float) -> DriftState:
        """Return the drift for ``state`` at ``time``."""


class NoiseSchedule(ABC):
    """Abstract diffusion noise schedule."""

    def __call__(self, time: float) -> float:
        """Evaluate the schedule using its canonical object interface."""
        return self.value_at(time)

    @abstractmethod
    def value_at(self, time: float) -> float:
        """Return the scalar noise scale at ``time``."""


class Solver(ABC):
    """Abstract one-step SDE solver."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Return the stable human-readable solver name."""

    @abstractmethod
    def step(
        self,
        state: GraphState,
        drift: DriftState,
        timestep: float,
        noise_scale: float,
        random_generator: Random,
        drift_model: DriftModel,
        time: float,
    ) -> GraphState:
        """Advance ``state`` by one validated solver step."""


class AdaptiveController(ABC):
    """Abstract controller that chooses a synchronized timestep."""

    @property
    @abstractmethod
    def eps_bound(self) -> float:
        """Return the controller's termination tolerance."""

    @property
    @abstractmethod
    def max_steps(self) -> int:
        """Return the controller's safety step limit."""

    def reset(self) -> None:
        """Reset trajectory-local state before a new sampling run."""
        return None

    @abstractmethod
    def choose_timestep(
        self,
        current_drift: DriftState,
        previous_drift: DriftState | None,
        noise_scale: float,
        time: float,
        terminal_time: float,
    ) -> AdaptiveDecision:
        """Return the next timestep and diagnostics for the trajectory."""


class GraphDecoder(ABC):
    """Abstract conversion from continuous adjacency values to graph edges."""

    @abstractmethod
    def decode(self, adjacency: Sequence[Sequence[float]]) -> tuple[tuple[float, ...], ...]:
        """Return a decoded adjacency matrix."""
