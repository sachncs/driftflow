"""Tests for the optional production inference adapter."""

from __future__ import annotations

import hashlib
import json
import sys
from contextlib import contextmanager
from pathlib import Path
from types import ModuleType

import pytest

from driftflow import (
    ConfigurationError,
    GraphState,
    SolverContractError,
    TorchScriptDriftModel,
    sha256_file,
)


class FakeTensor:
    """Small tensor double covering the adapter's required tensor surface."""

    def __init__(self, value: object) -> None:
        self.value = value
        if isinstance(value, (list, tuple)) and value and isinstance(value[0], (list, tuple)):
            self.shape = (len(value), len(value[0]))
        elif isinstance(value, (list, tuple)):
            self.shape = (len(value),)
        else:
            self.shape = ()

    def detach(self) -> FakeTensor:
        return self

    def to(self, device: str) -> FakeTensor:
        del device
        return self

    def tolist(self) -> object:
        return self.value


class FakeModule:
    """Scripted-model double returning drifts with input-compatible shapes."""

    def __init__(self, invalid_output: bool = False) -> None:
        self.invalid_output = invalid_output

    def eval(self) -> FakeModule:
        return self

    def __call__(self, features: FakeTensor, adjacency: FakeTensor, time: FakeTensor) -> object:
        del time
        if self.invalid_output:
            return features
        return features, adjacency


class FakeTorch(ModuleType):
    """Minimal PyTorch module double used without an ML dependency in CI."""

    float32 = object()

    class jit:
        module = FakeModule()

        @classmethod
        def load(cls, path: str, map_location: str) -> FakeModule:
            del path, map_location
            return cls.module

    @staticmethod
    def device(value: str) -> str:
        return value

    @staticmethod
    def tensor(value: object, dtype: object, device: str) -> FakeTensor:
        del dtype, device
        return FakeTensor(value)

    @staticmethod
    @contextmanager
    def inference_mode():
        yield

    @staticmethod
    def isfinite(value: FakeTensor) -> FiniteResult:
        del value
        return FiniteResult()


class FiniteResult:
    """Boolean-like result for ``torch.isfinite(...).all()``."""

    def all(self) -> bool:
        return True


def test_sha256_file_and_torchscript_adapter_contract(tmp_path: Path, monkeypatch) -> None:
    checkpoint = tmp_path / "model.ts"
    checkpoint.write_bytes(b"checkpoint-v1")
    expected_digest = hashlib.sha256(b"checkpoint-v1").hexdigest()
    fake_torch = FakeTorch("torch")
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    model = TorchScriptDriftModel(
        checkpoint,
        num_nodes=2,
        feature_dim=1,
        expected_sha256=expected_digest,
    )
    result = model.evaluate(GraphState.from_lists([[0.0], [1.0]], [[0.0, 1.0], [1.0, 0.0]]), 0.5)

    assert model.production_ready
    assert model.checkpoint_sha256 == sha256_file(checkpoint)
    assert result.features == ((0.0,), (1.0,))
    assert result.adjacency == ((0.0, 1.0), (1.0, 0.0))


def test_torchscript_adapter_rejects_bad_integrity_and_outputs(tmp_path: Path, monkeypatch) -> None:
    checkpoint = tmp_path / "model.ts"
    checkpoint.write_bytes(b"checkpoint-v1")
    monkeypatch.setitem(sys.modules, "torch", FakeTorch("torch"))

    with pytest.raises(ConfigurationError, match="SHA-256 mismatch"):
        TorchScriptDriftModel(checkpoint, 2, 1, expected_sha256="0" * 64)

    FakeTorch.jit.module = FakeModule(invalid_output=True)
    model = TorchScriptDriftModel(checkpoint, 2, 1, expected_sha256=sha256_file(checkpoint))
    with pytest.raises(SolverContractError, match="must return"):
        model.evaluate(GraphState.from_lists([[0.0], [1.0]], [[0.0, 1.0], [1.0, 0.0]]), 0.5)


def test_torchscript_adapter_reports_missing_dependency(tmp_path: Path, monkeypatch) -> None:
    checkpoint = tmp_path / "model.ts"
    checkpoint.write_bytes(b"checkpoint-v1")
    monkeypatch.setitem(sys.modules, "torch", None)

    with pytest.raises(ConfigurationError, match="requires PyTorch"):
        TorchScriptDriftModel(checkpoint, 2, 1, expected_sha256=sha256_file(checkpoint))


def test_torchscript_adapter_loads_and_validates_manifest(tmp_path: Path, monkeypatch) -> None:
    checkpoint = tmp_path / "model.ts"
    checkpoint.write_bytes(b"checkpoint-v1")
    manifest = tmp_path / "model.json"
    manifest.write_text(
        json.dumps(
            {
                "checkpoint": checkpoint.name,
                "sha256": sha256_file(checkpoint),
                "num_nodes": 2,
                "feature_dim": 1,
                "feature_order": ["feature_0"],
                "preprocessing": {"name": "identity", "version": "1"},
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setitem(sys.modules, "torch", FakeTorch("torch"))
    model = TorchScriptDriftModel.from_manifest(manifest)
    assert model.num_nodes == 2
    assert model.feature_dim == 1

    invalid = json.loads(manifest.read_text(encoding="utf-8"))
    invalid["feature_order"] = ["wrong"]
    manifest.write_text(json.dumps(invalid), encoding="utf-8")
    with pytest.raises(ConfigurationError, match="canonical order"):
        TorchScriptDriftModel.from_manifest(manifest)
