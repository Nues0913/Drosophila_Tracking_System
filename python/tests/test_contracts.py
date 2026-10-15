import numpy as np
import pytest
from flytrack_core.contracts import ActionLimits, SensorContract, SensorObservation


def test_contract_hash_detects_sensor_geometry_change():
    baseline = SensorContract()
    assert baseline.sha256() == SensorContract.from_dict(baseline.to_dict()).sha256()
    assert baseline.sha256() != SensorContract(odor_baseline_m=0.2).sha256()
    assert baseline.sha256() != SensorContract(eye_yaw_deg=40).sha256()


def test_noninteger_neural_control_ratio_rejected():
    with pytest.raises(ValueError, match="integer multiple"):
        SensorContract(control_dt=0.055)


def test_actions_obey_both_velocity_and_acceleration_limits():
    limits = ActionLimits()
    action = limits.clip(np.array([100, -100]), previous=np.array([0.2, 0.0]), dt=0.05)
    assert np.allclose(action, [0.25, -0.15])
    with pytest.raises(ValueError, match="finite"):
        limits.clip(np.array([np.nan, 0]))


def test_sensor_contract_rejects_invalid_shape_and_ranges():
    contract = SensorContract()
    valid = SensorObservation(np.zeros((2, 721)), np.zeros(2), np.ones(3), 1.0)
    valid.validate(contract)
    with pytest.raises(ValueError, match="vision"):
        SensorObservation(np.zeros((2, 720)), np.zeros(2), np.ones(3), 1.0).validate(contract)
    with pytest.raises(ValueError, match="nonnegative"):
        SensorObservation(np.zeros((2, 721)), np.array([-1, 0]), np.ones(3), 1.0).validate(contract)
