"""The batched privileged teacher reproduces v1's teacher_action (WP2a, #132)."""

from pathlib import Path

import numpy as np
import pytest
import torch
from flytrack_core.contracts import SensorContract
from flytrack_sim.teacher import teacher_action
from flytrack_v2.interfaces import Teacher
from flytrack_v2.world.teacher import PrivilegedTeacher

ROOT = Path(__file__).resolve().parents[3]


@pytest.fixture(scope="module")
def contract():
    return SensorContract.from_json(ROOT / "configs/sensor_contract.json")


def _states(count: int, seed: int = 0):
    """Random states; a third put the target heading back at the agent (evasion)."""
    rng = np.random.default_rng(seed)
    agent = np.c_[rng.uniform(-4, 4, (count, 2)), rng.uniform(-np.pi, np.pi, count)]
    distance = rng.uniform(0.3, 4.0, count)
    angle = rng.uniform(-np.pi, np.pi, count)
    target = np.c_[agent[:, :2] + distance[:, None] * np.c_[np.cos(angle), np.sin(angle)],
                   rng.uniform(-np.pi, np.pi, count)]  # fmt: skip
    heading = rng.uniform(-np.pi, np.pi, count)
    back = np.arange(count) % 3 == 0  # target velocity pointing at the agent
    heading[back] = angle[back] + np.pi + rng.uniform(-0.3, 0.3, back.sum())
    speed = np.where(np.arange(count) % 7 == 0, 0.0, 0.2)
    velocity = speed[:, None] * np.c_[np.cos(heading), np.sin(heading)]
    previous = np.c_[rng.uniform(0, 0.5, count), rng.uniform(-1.5, 1.5, count)].astype(np.float32)
    odor = rng.uniform(0, 1, (count, 2)).astype(np.float32)
    odor[::5] = 0  # no odor at all
    valid = np.ones((count, 4), np.float32)
    valid[1::6, 2] = 0  # stale odor reading
    occluded = rng.random(count) < 0.5
    return agent, target, velocity, previous, odor, valid, occluded


def test_is_a_teacher(contract):
    teacher = PrivilegedTeacher(contract)
    assert isinstance(teacher, Teacher)
    assert teacher.initial_state(4, "cpu").shape == (4, 0)


@pytest.mark.parametrize("gain", [None, 8.0])
def test_matches_v1_teacher_action(contract, gain):
    agent, target, velocity, previous, odor, valid, occluded = _states(400)
    teacher = PrivilegedTeacher(contract, occlusion_odor_gain=gain)
    privileged = {
        "agent_pose": torch.as_tensor(agent),
        "target_pose": torch.as_tensor(target),
        "target_velocity": torch.as_tensor(velocity),
        "occluded": torch.as_tensor(occluded),
    }
    obs = {"odor": torch.as_tensor(odor), "valid": torch.as_tensor(valid)}
    action, state = teacher.act(
        privileged, obs, torch.as_tensor(previous), teacher.initial_state(400, "cpu")
    )
    assert action.dtype == torch.float32 and state.shape == (400, 0)
    options = {} if gain is None else {"occlusion_odor_gain": gain}
    for b in range(400):
        extra = {}
        if gain is not None:
            extra = {
                "occluded": bool(occluded[b]),
                "observation": {"odor": odor[b], "valid": valid[b, :3]},
            }
        expected = teacher_action(
            agent[b],
            target[b],
            previous[b],
            contract,
            target_velocity=velocity[b],
            **options,
            **extra,
        )
        np.testing.assert_allclose(action[b].numpy(), expected, rtol=0, atol=1e-6)
