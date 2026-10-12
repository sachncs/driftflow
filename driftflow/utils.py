"""Validated graph-adjacency decoders."""

import math
from collections.abc import Sequence

from .contracts import GraphDecoder
from .errors import ConfigurationError


def validate_threshold(threshold: float) -> float:
    """Validate and return an inclusive probability threshold."""
    if not math.isfinite(threshold) or not 0.0 <= threshold <= 1.0:
        raise ConfigurationError("threshold must be finite and within [0, 1]")
    return threshold


def stable_sigmoid(value: float) -> float:
    """Evaluate sigmoid without overflowing for extreme logits."""
    if value >= 0.0:
        exponent = math.exp(-value)
        return 1.0 / (1.0 + exponent)
    exponent = math.exp(value)
    return exponent / (1.0 + exponent)


def decode_adjacency(
    adjacency: Sequence[Sequence[float]], threshold: float = 0.5
) -> list[list[float]]:
    """Binarize continuous adjacency values using an inclusive threshold."""
    threshold = validate_threshold(threshold)
    return [[1.0 if value >= threshold else 0.0 for value in row] for row in adjacency]


def sigmoid_decode_adjacency(
    adjacency: Sequence[Sequence[float]], threshold: float = 0.5
) -> list[list[float]]:
    """Apply stable sigmoid activation and then binarize adjacency logits."""
    threshold = validate_threshold(threshold)
    return [
        [1.0 if stable_sigmoid(value) >= threshold else 0.0 for value in row] for row in adjacency
    ]


class ThresholdGraphDecoder(GraphDecoder):
    """Decode adjacency probabilities using an inclusive threshold."""

    def __init__(self, threshold: float = 0.5) -> None:
        """Initialize the decoder."""
        self.threshold = validate_threshold(threshold)

    def decode(self, adjacency: Sequence[Sequence[float]]) -> tuple[tuple[float, ...], ...]:
        """Return a threshold-decoded immutable adjacency matrix."""
        return tuple(tuple(row) for row in decode_adjacency(adjacency, self.threshold))


class SigmoidGraphDecoder(GraphDecoder):
    """Decode adjacency logits using stable sigmoid thresholding."""

    def __init__(self, threshold: float = 0.5) -> None:
        """Initialize the decoder."""
        self.threshold = validate_threshold(threshold)

    def decode(self, adjacency: Sequence[Sequence[float]]) -> tuple[tuple[float, ...], ...]:
        """Return a sigmoid-decoded immutable adjacency matrix."""
        return tuple(tuple(row) for row in sigmoid_decode_adjacency(adjacency, self.threshold))
