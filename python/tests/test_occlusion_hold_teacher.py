"""#35 M1: while occluded, the hold teacher repeats the last unoccluded label (no privileged input)."""

import numpy as np
import pytest
from flytrack_core.contracts import ActionLimits, SensorContract
from flytrack_sim import FlyTrackEnv

OPTS = dict(
    scenario="occlusion",
    episode_seconds=1.0,
    scenario_config={"occlusion_start_s": 0.3, "occlusion_duration_s": 0.5},
    sensing=False,
)


def test_unoccluded_labels_are_unchanged_and_occluded_labels_hold():
    contract = SensorContract(eye_radius=1)
    plain = FlyTrackEnv(contract, **OPTS)
    hold = FlyTrackEnv(contract, **OPTS, occlusion_hold_teacher=True)
    limits = ActionLimits.from_contract(contract)
    (_, a), (_, b) = plain.reset(seed=35), hold.reset(seed=35)
    last_unoccluded, held_steps = None, 0
    while True:
        if not hold.scenario_spec.occluded_at(hold.time_s):
            assert a["teacher_action"].tobytes() == b["teacher_action"].tobytes()
            last_unoccluded = b["teacher_action"].copy()
        else:
            expected = limits.clip(last_unoccluded, hold.previous_action, contract.control_dt)
            np.testing.assert_array_equal(b["teacher_action"], expected)
            held_steps += 1
        # Drive both worlds with the same fixed command so the poses stay identical.
        command = np.array([0.2, 0.3], np.float32)
        _, _, done_a, trunc_a, a = plain.step(command)
        _, _, done_b, trunc_b, b = hold.step(command)
        if done_b or trunc_b:
            break
    assert held_steps > 0


def test_hold_teacher_starting_occluded_falls_back_to_the_privileged_label():
    contract = SensorContract(eye_radius=1)
    opts = dict(OPTS, scenario_config={"occlusion_start_s": 0.0, "occlusion_duration_s": 0.5})
    plain = FlyTrackEnv(contract, **opts)
    hold = FlyTrackEnv(contract, **opts, occlusion_hold_teacher=True)
    (_, a), (_, b) = plain.reset(seed=7), hold.reset(seed=7)
    assert a["teacher_action"].tobytes() == b["teacher_action"].tobytes()


def test_hold_teacher_is_off_by_default_and_exclusive_with_odor_steering():
    assert FlyTrackEnv(SensorContract(eye_radius=1)).occlusion_hold_teacher is False
    with pytest.raises(ValueError, match="exclusive"):
        FlyTrackEnv(occlusion_hold_teacher=True, occlusion_odor_steering={"gain": 8})
    with pytest.raises(ValueError, match="bool"):
        FlyTrackEnv(occlusion_hold_teacher="yes")
