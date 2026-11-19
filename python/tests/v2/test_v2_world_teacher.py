"""The batched privileged teacher reproduces v1's teacher_action (WP2a, #132)."""

from pathlib import Path

import numpy as np
import pytest
import torch
from flytrack_core.contracts import SensorContract
from flytrack_sim.teacher import teacher_action
from flytrack_v2.contracts import SensorContractV2
from flytrack_v2.interfaces import Teacher
from flytrack_v2.world.teacher import OcclusionTeacher, PrivilegedTeacher

ROOT = Path(__file__).resolve().parents[3]


@pytest.fixture(scope="module")
def contract():
    return SensorContract.from_json(ROOT / "configs/sensor_contract.json")


@pytest.fixture(scope="module")
def contract_v2():
    return SensorContractV2.from_json(ROOT / "configs/v2/sensor_contract.json")


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


def _occlusion_inputs(count=64, seed=3, occluded=True):
    rng = np.random.default_rng(seed)
    privileged = {
        "agent_pose": torch.as_tensor(
            np.c_[rng.uniform(-4, 4, (count, 2)), rng.uniform(-3, 3, count)]
        ),
        "target_pose": torch.as_tensor(
            np.c_[rng.uniform(-4, 4, (count, 2)), rng.uniform(-3, 3, count)]
        ),
        "target_velocity": torch.as_tensor(rng.normal(0, 0.2, (count, 2))),
        "occluded": torch.full((count,), occluded),
    }
    odor = rng.uniform(0, 0.3, (count, 2)).astype(np.float32)
    odor[::4] = 0  # odor lost
    obs = {
        "odor": torch.as_tensor(odor),
        "wind": torch.as_tensor(rng.normal(0, 0.5, (count, 2)).astype(np.float32)),
        "valid": torch.ones(count, 4),
    }
    previous = torch.as_tensor(
        np.c_[rng.uniform(0, 0.5, count), rng.uniform(-1, 1, count)].astype(np.float32)
    )
    return privileged, obs, previous


def test_occlusion_teacher_never_reads_the_hidden_target(contract_v2):
    contract = contract_v2
    """#161 C3: same observable history and previous action, different hidden target ->
    the same labels during occlusion, step after step."""
    teacher = OcclusionTeacher(PrivilegedTeacher(contract), walls=(-5.0, -5.0, 5.0, 5.0))
    privileged, obs, previous = _occlusion_inputs()
    other = dict(privileged)
    other["target_pose"] = privileged["target_pose"] + torch.tensor(
        [3.0, -2.0, 1.0], dtype=torch.float64
    )
    other["target_velocity"] = -privileged["target_velocity"]
    a_state = b_state = teacher.initial_state(64, "cpu")
    for step in range(10):
        obs["odor"] = obs["odor"].roll(step, 0)
        a, a_state = teacher.act(privileged, obs, previous, a_state)
        b, b_state = teacher.act(other, obs, previous, b_state)
        torch.testing.assert_close(a, b, rtol=0, atol=0)
        torch.testing.assert_close(a_state, b_state, rtol=0, atol=0)
        previous = a
    # Outside occlusion it is the privileged teacher, which does follow the target.
    privileged["occluded"][:] = False
    other["occluded"] = privileged["occluded"]
    a, _ = teacher.act(privileged, obs, previous, teacher.initial_state(64, "cpu"))
    b, _ = teacher.act(other, obs, previous, teacher.initial_state(64, "cpu"))
    assert not torch.equal(a, b)
    expected, _ = PrivilegedTeacher(contract).act(privileged, obs, previous, a_state[:, :0])
    torch.testing.assert_close(a, expected)


def test_occlusion_teacher_holds_speed_and_uses_odor_and_wind(contract_v2):
    contract = contract_v2
    teacher = OcclusionTeacher(PrivilegedTeacher(contract))
    privileged, obs, previous = _occlusion_inputs(count=8)
    state = teacher.initial_state(8, "cpu")
    privileged["occluded"][:] = False
    _, state = teacher.act(privileged, obs, previous, state)  # holds speed and turn
    torch.testing.assert_close(state[:, 0], previous[:, 0])
    privileged["occluded"][:] = True
    obs["odor"] = torch.tensor([[0.2, 0.0]] * 8)  # odor on the left only
    obs["wind"] = torch.tensor([[0.0, 0.0]] * 8)  # no wind angle information
    straight = torch.stack((state[:, 0], torch.zeros(8)), -1).float()  # within the rate limit
    action, _ = teacher.act(privileged, obs, straight, state)
    assert bool((action[:, 1] > 0).all())  # turns left, toward the odor
    # Without a valid wind reading (v1compat) only the odor steers: the same here.
    obs["valid"][:, 3] = 0
    no_wind, _ = teacher.act(privileged, obs, straight, state)
    torch.testing.assert_close(no_wind, action)
