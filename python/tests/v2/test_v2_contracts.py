"""v2 sensor contract and observation checks (plan 5.3)."""

import json
from pathlib import Path

import pytest
import torch
from flytrack_core.contracts import SensorContract
from flytrack_v2.contracts import (
    OBS_KEYS,
    SUCCESS_RULE,
    VALID_CHANNELS,
    SensorContractV2,
    obs_shapes,
    validate_obs,
)

ROOT = Path(__file__).resolve().parents[3]
CONTRACT = ROOT / "configs/v2/sensor_contract.json"


@pytest.fixture
def contract():
    return SensorContractV2.from_json(CONTRACT)


def _obs(contract, batch=3):
    obs = {name: torch.zeros(shape) for name, shape in obs_shapes(contract, batch).items()}
    obs["valid"][:] = 1
    return obs


def test_contract_keeps_v1_geometry_and_adds_wind(contract):
    v1 = SensorContract.from_json(ROOT / "configs/sensor_contract.json")
    v2_fields = json.loads(CONTRACT.read_text(encoding="utf-8"))
    assert {k: v2_fields[k] for k in v1.to_dict()} == v1.to_dict()
    assert contract.n_ommatidia == 721 and contract.neural_substeps == 5
    assert (contract.wind_hz, contract.antenna_angle_deg) == (50.0, 45.0)
    assert contract.sha256() != v1.sha256()


def test_contract_rejects_unknown_fields_and_bad_antenna_angle():
    values = json.loads(CONTRACT.read_text(encoding="utf-8"))
    with pytest.raises(ValueError, match="Unknown"):
        SensorContractV2.from_dict({**values, "target_pose": 1})
    with pytest.raises(ValueError, match="antenna_angle_deg"):
        SensorContractV2.from_dict({**values, "antenna_angle_deg": 90.0})


def test_validate_obs_accepts_the_contract_layout(contract):
    assert validate_obs(_obs(contract), contract, check_values=True) == 3
    assert OBS_KEYS == ("vision", "odor", "wind", "valid") and len(VALID_CHANNELS) == 4


def test_validate_obs_refuses_privileged_keys_and_wrong_shapes(contract):
    obs = _obs(contract)
    with pytest.raises(ValueError, match="Privileged"):
        validate_obs({**obs, "target_pose": torch.zeros(3, 3)}, contract)
    with pytest.raises(ValueError, match="valid must have shape"):
        validate_obs({**obs, "valid": torch.zeros(3, 3)}, contract)
    with pytest.raises(TypeError, match="float32"):
        validate_obs({**obs, "odor": torch.zeros(3, 2, dtype=torch.float64)}, contract)


def test_value_checks_are_opt_in(contract):
    obs = _obs(contract)
    obs["valid"][0, 0] = 0.5
    assert validate_obs(obs, contract) == 3  # shape-only: no host sync in hot loops
    with pytest.raises(ValueError, match="binary"):
        validate_obs(obs, contract, check_values=True)
    obs = _obs(contract)
    obs["vision"][1, 0, 0] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        validate_obs(obs, contract, check_values=True)


def test_success_rule_is_v1s():
    assert (SUCCESS_RULE.distance_m, SUCCESS_RULE.bearing_deg, SUCCESS_RULE.fraction) == (
        2.0,
        45.0,
        0.8,
    )
    assert SUCCESS_RULE.collision_invalidates
