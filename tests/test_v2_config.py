"""Behavioral tests for validated v2 configuration and schedules."""

import pytest

from driftflow import CommonConfig, ConfigurationError, DatasetConfig, LinearSchedule
from driftflow.errors import ScheduleError


def test_common_config_is_immutable_and_validated() -> None:
    config = CommonConfig()

    with pytest.raises(AttributeError):
        config.alpha = 0.5  # type: ignore[misc]
    with pytest.raises(ConfigurationError, match="dt_min"):
        CommonConfig(dt_min=0.2, dt_base=0.1)


def test_dataset_ranges_are_immutable_tuples() -> None:
    config = DatasetConfig("model", "dataset", 1.0, active_range=[(0.0, 0.5)])

    assert config.active_range == ((0.0, 0.5),)
    assert config.is_active(0.5)
    assert not config.is_active(0.75)


def test_schedule_rejects_invalid_time_and_parameters() -> None:
    with pytest.raises(ScheduleError, match="non-negative"):
        LinearSchedule(-1.0)
    with pytest.raises(ScheduleError, match=r"within \[0, 1\]"):
        LinearSchedule().value_at(2.0)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("alpha", 1.1, "alpha"),
        ("beta", -1.0, "beta"),
        ("eps_num", 0.0, "eps_num"),
        ("eps_bound", -1.0, "eps_bound"),
        ("max_steps", 0, "max_steps"),
    ],
)
def test_common_config_rejects_invalid_limits(field: str, value: object, message: str) -> None:
    with pytest.raises(ConfigurationError, match=message):
        CommonConfig(**{field: value})


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"model": "", "dataset": "dataset"}, "non-empty"),
        ({"kappa_ref": 0.0}, "kappa_ref"),
        ({"gamma_euler": -1.0}, "gamma_euler"),
        ({"active_range": [(0.8, 0.2)]}, "active ranges"),
    ],
)
def test_dataset_config_rejects_invalid_metadata(kwargs: dict[str, object], message: str) -> None:
    values: dict[str, object] = {
        "model": "model",
        "dataset": "dataset",
        "kappa_ref": 1.0,
    }
    values.update(kwargs)
    with pytest.raises(ConfigurationError, match=message):
        DatasetConfig(**values)  # type: ignore[arg-type]


def test_unknown_dataset_config_has_actionable_error() -> None:
    from driftflow import get_dataset_config

    with pytest.raises(ConfigurationError, match="No config"):
        get_dataset_config("unknown", "dataset")
