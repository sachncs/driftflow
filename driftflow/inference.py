"""Production model adapters with strict checkpoint and tensor contracts."""

from __future__ import annotations

import hashlib
import json
import re
import threading
from os import PathLike
from pathlib import Path

from .contracts import DriftModel
from .domain import DriftState, GraphState
from .errors import ConfigurationError, NumericalStabilityError, ShapeError, SolverContractError


def sha256_file(path: str | PathLike[str]) -> str:
    """Return the SHA-256 digest of a checkpoint without loading it."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as checkpoint:
        for chunk in iter(lambda: checkpoint.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class TorchScriptDriftModel(DriftModel):
    """Evaluate a versioned TorchScript graph denoiser.

    The exported module must accept three inputs—feature matrix, adjacency
    matrix, and scalar diffusion time—and return ``(feature_drift,
    adjacency_drift)`` tensors with the same shapes as the inputs. The
    adapter intentionally rejects ambiguous output formats and non-finite
    tensors before they can enter the numerical solver.

    ``torch`` is optional at install time and imported lazily so the core
    contracts remain dependency-free. Install ``driftflow[production]`` for
    this adapter.
    """

    @property
    def production_ready(self) -> bool:
        """TorchScript checkpoints are eligible for production mode."""
        return True

    def __init__(
        self,
        checkpoint: str | PathLike[str],
        num_nodes: int,
        feature_dim: int,
        *,
        device: str = "cpu",
        expected_sha256: str,
    ) -> None:
        """Load and validate a scripted model checkpoint.

        Args:
            checkpoint: Path to a TorchScript ``.pt``/``.ts`` artifact.
            num_nodes: Fixed graph node count expected by the model.
            feature_dim: Fixed node feature width expected by the model.
            device: Torch device, normally ``cpu`` or a configured CUDA device.
            expected_sha256: Exact 64-character artifact digest for
                supply-chain and deployment integrity checks.
        """
        path = Path(checkpoint)
        if not path.is_file():
            raise ConfigurationError(f"model checkpoint does not exist: {path}")
        if num_nodes <= 0 or feature_dim <= 0:
            raise ConfigurationError("num_nodes and feature_dim must be positive")
        if not re.fullmatch(r"[0-9a-fA-F]{64}", expected_sha256):
            raise ConfigurationError("expected_sha256 must be a 64-character hexadecimal digest")
        actual_sha256 = sha256_file(path)
        if actual_sha256.lower() != expected_sha256.lower():
            raise ConfigurationError(
                f"checkpoint SHA-256 mismatch: expected {expected_sha256}, got {actual_sha256}"
            )
        try:
            import torch  # type: ignore[import-not-found]
        except ImportError as error:
            raise ConfigurationError(
                "TorchScriptDriftModel requires PyTorch; install driftflow[production]"
            ) from error

        try:
            self._device = torch.device(device)
            self._module = torch.jit.load(str(path), map_location=self._device)
            self._module.eval()
        except Exception as error:
            raise ConfigurationError(f"unable to load TorchScript checkpoint {path}") from error
        self._torch = torch
        self.checkpoint = path
        self.checkpoint_sha256 = sha256_file(path)
        self.num_nodes = num_nodes
        self.feature_dim = feature_dim
        self.device = str(self._device)
        self._lock = threading.RLock()

    @classmethod
    def from_manifest(
        cls,
        manifest: str | PathLike[str],
        *,
        checkpoint: str | PathLike[str] | None = None,
        device: str = "cpu",
    ) -> TorchScriptDriftModel:
        """Load a checkpoint using its signed deployment manifest.

        The manifest is intentionally explicit: feature order and preprocessing
        are deployment metadata, while the adapter enforces the tensor shapes
        and digest.  Callers can therefore fail closed when a model artifact is
        paired with the wrong input pipeline.
        """
        manifest_path = Path(manifest)
        try:
            metadata = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise ConfigurationError(f"unable to read model manifest {manifest_path}") from error
        required = {
            "checkpoint",
            "sha256",
            "num_nodes",
            "feature_dim",
            "feature_order",
            "preprocessing",
        }
        if not required.issubset(metadata):
            missing = sorted(required - set(metadata))
            raise ConfigurationError(f"model manifest is missing fields: {', '.join(missing)}")
        if metadata["feature_order"] != [
            f"feature_{index}" for index in range(metadata["feature_dim"])
        ]:
            raise ConfigurationError(
                "model manifest feature_order is not the declared canonical order"
            )
        if metadata["preprocessing"] != {"name": "identity", "version": "1"}:
            raise ConfigurationError("unsupported model preprocessing contract")
        checkpoint_path = (
            Path(checkpoint)
            if checkpoint is not None
            else manifest_path.parent / metadata["checkpoint"]
        )
        return cls(
            checkpoint_path,
            num_nodes=metadata["num_nodes"],
            feature_dim=metadata["feature_dim"],
            device=device,
            expected_sha256=metadata["sha256"],
        )

    def evaluate(self, state: GraphState, time: float) -> DriftState:
        """Run inference and convert a validated tensor result to ``DriftState``."""
        if state.features and len(state.features[0]) != self.feature_dim:
            raise ShapeError(
                f"model expects feature width {self.feature_dim}, got {len(state.features[0])}"
            )
        if len(state.features) != self.num_nodes or len(state.adjacency) != self.num_nodes:
            raise ShapeError(f"model expects {self.num_nodes} graph nodes")
        if not isinstance(time, (int, float)) or not 0.0 <= float(time) <= 1.0:
            raise ConfigurationError(f"time must be within [0, 1], got {time!r}")

        features = self._torch.tensor(
            state.features, dtype=self._torch.float32, device=self._device
        )
        adjacency = self._torch.tensor(
            state.adjacency, dtype=self._torch.float32, device=self._device
        )
        diffusion_time = self._torch.tensor(
            float(time), dtype=self._torch.float32, device=self._device
        )
        with self._lock, self._torch.inference_mode():
            try:
                output = self._module(features, adjacency, diffusion_time)
            except Exception as error:
                raise SolverContractError("production drift model inference failed") from error

        if not isinstance(output, (tuple, list)) or len(output) != 2:
            raise SolverContractError(
                "production drift model must return (feature_drift, adjacency_drift)"
            )
        feature_drift, adjacency_drift = output
        expected_feature_shape = (self.num_nodes, self.feature_dim)
        expected_adjacency_shape = (self.num_nodes, self.num_nodes)
        if tuple(feature_drift.shape) != expected_feature_shape:
            raise ShapeError(
                "feature drift must have shape "
                f"{expected_feature_shape}, got {tuple(feature_drift.shape)}"
            )
        if tuple(adjacency_drift.shape) != expected_adjacency_shape:
            raise ShapeError(
                "adjacency drift must have shape "
                f"{expected_adjacency_shape}, got {tuple(adjacency_drift.shape)}"
            )
        if not bool(self._torch.isfinite(feature_drift).all()) or not bool(
            self._torch.isfinite(adjacency_drift).all()
        ):
            raise NumericalStabilityError("production drift model returned non-finite values")

        return DriftState.from_lists(
            feature_drift.detach().to("cpu").tolist(),
            adjacency_drift.detach().to("cpu").tolist(),
        )
