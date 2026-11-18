"""The batched evaluation runner on a scripted world (#139)."""

import math

import pytest
import torch
from flytrack_v2.eval.runner import EPISODE_KEYS, SpeedClamp, run_episodes, to_records
from flytrack_v2.interfaces import StepResult
from flytrack_v2.metrics import validate_episode_record


class _World:
    """Row i ends its first episode after seed % 4 + 3 steps; distance grows 0.1 m per step."""

    def __init__(self, batch):
        self.num_envs, self.device, self.contract = batch, torch.device("cpu"), None

    def reset(self, mask=None, episode_seeds=None):
        self.seeds = episode_seeds.clone()
        self.t = torch.zeros(self.num_envs)
        self.length = (self.seeds % 4 + 3).to(torch.float32)
        return self._obs()

    def _obs(self):
        return {"vision": torch.zeros(self.num_envs, 2, 721), "valid": torch.ones(self.num_envs, 4)}

    def step(self, action):
        self.t += 1
        ended = self.t >= self.length
        b = self.num_envs
        info = {
            "distance_m": 0.1 * self.t,
            "tracked": torch.ones(b, dtype=torch.bool),
            "collision": torch.zeros(b, dtype=torch.bool),
            "bearing_rad": torch.zeros(b),
            "occluded": torch.zeros(b, dtype=torch.bool),
        }
        for key in EPISODE_KEYS:
            info[key] = torch.zeros(b)
        info["episode_steps"] = self.t.clone()
        info["episode_seed"] = self.seeds.to(torch.float32)
        info["episode_success"] = torch.ones(b)
        info["episode_tracking_fraction"] = torch.full((b,), 0.9)
        info["episode_within_distance_fraction"] = torch.full((b,), 0.95)
        info["episode_occlusion_tracking_fraction"] = torch.full((b,), math.nan)
        info["episode_reacquisition_time_s"] = torch.full((b,), math.nan)
        self.t = torch.where(ended, torch.zeros_like(self.t), self.t)  # automatic reset
        flags = torch.zeros(b, dtype=torch.bool)
        return StepResult(self._obs(), torch.zeros(b), flags, ended, info, self._obs(), {})


class _Actor:
    """Speed rises with the distance the world reports through the step count."""

    def reset(self, batch, device):
        self.t = torch.zeros(batch)
        self.resets = 0

    def act(self, obs, reset_mask):
        if reset_mask is not None:
            self.resets += int(reset_mask.sum())
            self.t = torch.where(reset_mask, torch.zeros_like(self.t), self.t)
        self.t += 1
        return torch.stack((0.05 * self.t, torch.zeros_like(self.t)), dim=-1)


def test_each_row_keeps_only_its_first_episode():
    world = _World(4)
    result = run_episodes(world, _Actor(), [0, 1, 2, 5], max_steps=20, sync_every=1)
    assert result.values["episode_steps"] == [3.0, 4.0, 5.0, 4.0]
    assert result.values["episode_seed"] == [0.0, 1.0, 2.0, 5.0]
    # speed = 0.05 t and distance = 0.1 t: slope 0.5 m/s per m, over the first episode only
    assert result.speed_distance["slope_m_s_per_m"] == pytest.approx([0.5] * 4)
    assert result.speed_distance["n"] == [3.0, 4.0, 5.0, 4.0]


def test_rows_that_never_finish_raise():
    with pytest.raises(RuntimeError, match="did not finish"):
        run_episodes(_World(2), _Actor(), [0, 1], max_steps=2)


def test_speed_clamp_holds_speed_and_flattens_the_slope():
    result = run_episodes(_World(2), SpeedClamp(_Actor(), 0.2), [0, 1], max_steps=20)
    assert result.speed_distance["mean_speed_m_s"] == pytest.approx([0.2, 0.2])
    assert result.speed_distance["slope_m_s_per_m"] == pytest.approx([0.0, 0.0], abs=1e-9)


def test_records_from_a_run_pass_the_episode_schema():
    result = run_episodes(_World(2), _Actor(), [3, 4], max_steps=20)
    rows = [{"scenario": "normal"}, {"scenario": "noise", "noise_scale": 0.75}]
    records = to_records(
        result,
        rows,
        run_id="t",
        variant="v1:VO",
        train_seed=0,
        world="v1compat",
        motion="curved",
        artifact_sha256="0" * 64,
        commit="abc",
    )
    for record in records:
        validate_episode_record(record)
    assert records[1]["noise_scale"] == 0.75 and records[0]["noise_scale"] is None
