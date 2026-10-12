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
