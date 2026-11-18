"""The v2 world audit runs v1's teacher and blind set on v1compat (WP2a, #132)."""

from pathlib import Path

import pytest
from flytrack_train.audit import FIXED_ACTIONS
from flytrack_v2.contracts import SensorContractV2
from flytrack_v2.world.audit import audit_world

ROOT = Path(__file__).resolve().parents[3]


@pytest.mark.slow  # about 6 s on CPU
def test_audit_reports_teacher_and_blind_policies():
    contract = SensorContractV2.from_json(ROOT / "configs/v2/sensor_contract.json")
    # 10 s episodes keep this short; the real audit (scripts/v2/audit_world.py) runs 60 s.
    # A stopped agent starts 1.5 m behind and loses a 0.2 m/s target after about 2.5 s.
    report = audit_world(contract, "normal", {"episode_seconds": 10}, episodes=4)
    assert set(report["blind_policies"]) == {*FIXED_ACTIONS, "teacher_mean"}
    assert report["teacher"]["success_rate"] == 1.0
    assert report["blind_policies"]["stop"]["success_rate"] == 0.0
    assert len(report["teacher_mean_action"]) == 2 and report["teacher_mean_action"][0] > 0
    assert report["passed"] == (not report["failures"])
