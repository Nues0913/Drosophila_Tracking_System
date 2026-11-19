"""Batched rollouts and CPU replay with globally aligned recurrent snapshots."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import NamedTuple

import numpy as np
import torch
from torch import Tensor

from flytrack_v2.interfaces import reset_rows
from flytrack_v2.model.sensory import SensoryState
from flytrack_v2.world.teacher import OcclusionTeacher, PrivilegedTeacher
from flytrack_v2.world.v1compat import V1CompatWorld
from flytrack_v2.world.wind import WindConfig

from .config import (
    MOTION_NAMES,
    ROOT,
    SCENARIO_NAMES,
    STATE_STRIDE,
    WORLD_NAMES,
    RoundPlan,
    read_yaml,
    scenario_rows,
)

EPISODE_SEED_BIT = 1 << 40
_TRANSFER_TICKS = 32
_READOUT_SAMPLE_STRIDE = 4
_TEACHER_KEYS = ("agent_pose", "agent_velocity", "target_pose", "target_velocity", "occluded")


def mix_actions(teacher: Tensor, student: Tensor, beta: float, generator: torch.Generator):
    choose_teacher = torch.rand(len(teacher), device=teacher.device, generator=generator) < beta
    return torch.where(choose_teacher[:, None], teacher, student)


class StepData(NamedTuple):
    code: Tensor
    action: Tensor
    reset: Tensor
    occluded: Tensor


@dataclass
class ReplayShard:
    steps: StepData
    states: Tensor  # [snapshots, envs, state_size]
    tags: Tensor  # [envs, 3], indices into WORLD_NAMES, MOTION_NAMES, SCENARIO_NAMES
    held_out: Tensor  # [envs], fixed throughout the run
    first_snapshot: int
    readout_rates: Tensor | None = None  # [samples, DNs], only for the first fit

    def checkpoint(self) -> dict:
        saved = {**vars(self), "steps": self.steps._asdict()}
        if self.readout_rates is None:
            del saved["readout_rates"]
        return saved

    @classmethod
    def restore(cls, saved: dict) -> ReplayShard:
        return cls(**{**saved, "steps": StepData(**saved["steps"])})


class SequenceBatch(NamedTuple):
    code: Tensor
    action: Tensor
    reset: Tensor
    initial: Tensor
    tags: Tensor
    occluded: Tensor

    def to(self, device: torch.device) -> SequenceBatch:
        return SequenceBatch(*(value.to(device) for value in self))


def _allocate_steps(ticks: int, rows: int, width: int, device: torch.device | str) -> StepData:
    return StepData(
        torch.empty(ticks, rows, width, dtype=torch.float32, device=device),
        torch.empty(ticks, rows, 2, dtype=torch.float32, device=device),
        torch.empty(ticks, rows, dtype=torch.bool, device=device),
        torch.empty(ticks, rows, dtype=torch.bool, device=device),
    )


def _concatenate(parts: list[dict[str, Tensor]], keys) -> dict[str, Tensor]:
    return {name: torch.cat([part[name] for part in parts]) for name in keys}


class RolloutCollector:
    def __init__(self, policy, config: dict, seed: int, device: torch.device, *, restore=None):
        self.policy, self.device = policy, device
        rows = config["envs_per_instance"]
        self.rows = rows * len(WORLD_NAMES) * len(MOTION_NAMES)
        self.mix_rng = torch.Generator(device=device).manual_seed(seed)
        split_rng = torch.Generator().manual_seed(seed ^ 0x5EED)
        reserve = max(1, round(self.rows * config["held_out"]["fraction"]))
        self.held_out = torch.zeros(self.rows, dtype=torch.bool)
        self.held_out[
            torch.randperm(self.rows, generator=split_rng)[: min(reserve, self.rows - 1)]
        ] = True
        names = scenario_rows(config["scenarios"], rows)
        scenario_configs = {
            name: read_yaml(ROOT / f"configs/scenarios/{name}.yaml") for name in SCENARIO_NAMES
        }
        self.worlds, tags = [], []
        for world_id, kind in enumerate(WORLD_NAMES):
            for motion_id, motion in enumerate(MOTION_NAMES):
                world = V1CompatWorld(
                    policy.contract,
                    rows,
                    scenarios=names,
                    options={
                        "target_motion": motion,
                        "episode_seconds": config["episode_seconds"],
                        "plume_model": config["plume_model"],
                    },
                    seed=seed,
                    device=device,
                    scenario_configs=scenario_configs,
                    wind=WindConfig(speed_m_s=config["wind_speed_m_s"]) if kind == "wind" else None,
                    env_index_offset=len(self.worlds) * rows,
                    episode_seed_namespace=EPISODE_SEED_BIT,
                )
                self.worlds.append(world)
                tags.extend((world_id, motion_id, SCENARIO_NAMES.index(name)) for name in names)
        self.tags = torch.tensor(tags, dtype=torch.uint8)
        walls = self.worlds[0].walls
        if any(world.walls != walls for world in self.worlds):
            raise ValueError("The batched DAgger teacher requires shared arena walls")
        self.teacher = OcclusionTeacher(PrivilegedTeacher(policy.contract), walls=walls)
        self.teacher_initial = self.teacher.initial_state(self.rows, device)
        self._first_reset_steps: Tensor | None = None
        if restore is None:
            self.obs = self._observations([world.reset() for world in self.worlds])
            self.sensory = policy.initial_sensory_state(self.rows, device)
            self.central = policy.initial_central_state(self.rows, device)
            self.teacher_state = self.teacher_initial.clone()
            if config.get("desync_first_episode", False):
                rng = torch.Generator().manual_seed(seed ^ 0xD35A)
                steps = torch.randint(
                    self.worlds[0].max_steps, (len(self.worlds), rows), generator=rng
                )
                self._set_first_resets(steps, torch.zeros_like(steps, dtype=torch.bool), 0)
        else:
            self.restore(restore)

    def _observations(self, parts: list[dict[str, Tensor]]) -> dict[str, Tensor]:
        return _concatenate(parts, parts[0])

    def _set_first_resets(self, steps: Tensor, forced: Tensor, ticks: int) -> None:
        # Only check masks on ticks with scheduled resets for that world.
        self._first_reset_schedule: dict[int, list[int]] = {}
        for world_id, offsets in enumerate(steps.tolist()):
            for tick in set(offsets) - {0}:
                self._first_reset_schedule.setdefault(tick, []).append(world_id)
        self._first_reset_steps = steps.to(self.device)
        self._forced_first_resets = forced.to(self.device)
        self._rollout_ticks = ticks

    def checkpoint(self) -> dict:
        # Checkpoint/restore are the only users of world internals because V1CompatWorld has
        # no serialization API. On CUDA the graphs are re-captured against the restored state.
        saved = {
            "worlds": [
                {
                    "state": {name: value.cpu() for name, value in world._state.items()},
                    "episode_index": world.episode_index.cpu(),
                }
                for world in self.worlds
            ],
            "sensory": tuple(value.cpu() for value in self.sensory),
            "central": self.central.cpu(),
            "teacher": self.teacher_state.cpu(),
            "mix_rng": self.mix_rng.get_state(),
        }
        if self._first_reset_steps is not None:
            saved["first_episode_resets"] = {
                "steps": self._first_reset_steps.cpu(),
                "forced": self._forced_first_resets.cpu(),
                "ticks": self._rollout_ticks,
            }
        return saved

    def restore(self, saved: dict) -> None:
        """Restore a checkpoint and discard CUDA graphs before replacing world state tensors.

        The next rollout step must recapture the graphs against the new buffers.
        """
        for world in self.worlds:
            world.release_rollout_graphs()
        for world, values in zip(self.worlds, saved["worlds"], strict=True):
            world._state = {name: value.to(self.device) for name, value in values["state"].items()}
            world.episode_index.copy_(values["episode_index"])
        # Rebuild the observation from the restored state without advancing the world.
        self.obs = self._observations([world._observation(world._state) for world in self.worlds])
        self.sensory = SensoryState(*(value.to(self.device) for value in saved["sensory"]))
        self.central = saved["central"].to(self.device)
        self.teacher_state = saved["teacher"].to(self.device)
        self.mix_rng.set_state(saved["mix_rng"])
        self._first_reset_steps = None
        if "first_episode_resets" in saved:
            self._set_first_resets(**saved["first_episode_resets"])

    @torch.no_grad()
    def step(
        self,
        beta: float,
        initial: Tensor,
        *,
        episode_steps: Tensor | None = None,
        readout_sample: tuple[Tensor, Tensor] | None = None,
    ) -> StepData:
        code, self.sensory = self.policy.sense(self.obs, self.sensory)
        student = self.policy.forward_code(code, self.central)
        if readout_sample is not None:
            rows, output = readout_sample
            torch.index_select(student.aux["dn_rates"], 0, rows, out=output)
        privileged = _concatenate([world.privileged() for world in self.worlds], _TEACHER_KEYS)
        label, self.teacher_state = self.teacher.act(
            privileged, self.obs, privileged["agent_velocity"], self.teacher_state
        )
        action = mix_actions(label, student.dist.mode(), beta, self.mix_rng)
        results = [
            world.step(rows)
            for world, rows in zip(self.worlds, action.split(self.worlds[0].num_envs), strict=True)
        ]
        if episode_steps is not None:
            torch.cat([result.info["episode_steps"] for result in results], out=episode_steps)
        resets = [result.terminated | result.truncated for result in results]
        observations = [result.obs for result in results]
        if self._first_reset_steps is not None:
            self._rollout_ticks += 1
            for world_id in self._first_reset_schedule.get(self._rollout_ticks, ()):
                world = self.worlds[world_id]
                forced = self._forced_first_resets[world_id]
                mask = (
                    (self._first_reset_steps[world_id] == self._rollout_ticks)
                    & (world.episode_index == 0)
                    & ~forced
                )
                if bool(mask.any()):
                    # The public reset commits into the same buffers used by the CUDA graphs.
                    observations[world_id] = world.reset(mask)
                    resets[world_id] |= mask
                    forced |= mask
        reset = torch.cat(resets)
        self.obs = self._observations(observations)
        self.sensory = self.policy.reset_sensory_state(self.sensory, reset)
        self.central = reset_rows(student.central_state, initial, reset)
        self.teacher_state = reset_rows(self.teacher_state, self.teacher_initial, reset)
        return StepData(code, label, reset, privileged["occluded"])

    @torch.no_grad()
    def collect(
        self, plan: RoundPlan, *, sample_readout_rates: bool = False
    ) -> tuple[ReplayShard, dict[str, float]]:
        start = time.perf_counter()
        steps = _allocate_steps(plan.ticks, self.rows, self.policy.code_size, "cpu")
        episode_steps = torch.empty(plan.ticks, self.rows, dtype=torch.int64, device=self.device)
        snapshots = range(plan.first_snapshot, plan.ticks, STATE_STRIDE)
        states = torch.empty(
            len(snapshots), self.rows, self.policy.central_state_size, dtype=torch.float32
        )
        initial = self.policy.initial_central_state(self.rows, self.device)
        is_cuda = self.device.type == "cuda"
        readout_rates = None
        if sample_readout_rates:
            train_rows = torch.where(~self.held_out)[0]
            if self.tags[train_rows, 0].unique().numel() != len(WORLD_NAMES):
                raise ValueError("DN readout normalization requires training rows from both worlds")
            train_rows = train_rows.to(self.device)
            readout_rates = self.policy.readout.rate_mean.new_empty(
                math.ceil(plan.ticks / _READOUT_SAMPLE_STRIDE),
                len(train_rows),
                self.policy.readout.rate_mean.numel(),
            )
        if is_cuda:
            staging = _allocate_steps(
                _TRANSFER_TICKS, self.rows, self.policy.code_size, self.device
            )
            staged_states = torch.empty_like(
                states[: math.ceil(_TRANSFER_TICKS / STATE_STRIDE)], device=self.device
            )
        else:
            staging, staged_states = steps, states
        next_snapshot = 0
        for start_tick in range(0, plan.ticks, _TRANSFER_TICKS):
            end = min(start_tick + _TRANSFER_TICKS, plan.ticks)
            first_saved = next_snapshot
            for tick in range(start_tick, end):
                if tick in snapshots:
                    slot = next_snapshot - first_saved if is_cuda else next_snapshot
                    staged_states[slot].copy_(self.central)
                    next_snapshot += 1
                slot = tick - start_tick if is_cuda else tick
                sample = None
                if readout_rates is not None and tick % _READOUT_SAMPLE_STRIDE == 0:
                    sample = train_rows, readout_rates[tick // _READOUT_SAMPLE_STRIDE]
                data = self.step(
                    plan.beta, initial, episode_steps=episode_steps[tick], readout_sample=sample
                )
                for buffer, value in zip(staging, data, strict=True):
                    buffer[slot].copy_(value)
            if is_cuda:
                for dest, source in zip(steps, staging, strict=True):
                    dest[start_tick:end].copy_(source[: end - start_tick])
                states[first_saved:next_snapshot].copy_(
                    staged_states[: next_snapshot - first_saved]
                )
        rate = plan.ticks * self.rows / (time.perf_counter() - start)
        if is_cuda:
            for world in self.worlds:
                world.release_rollout_graphs()
        # Code and labels use the observation before the action, one step before the info count.
        std, mean = torch.std_mean(episode_steps.to(torch.float64), correction=0)
        dt = self.policy.contract.control_dt
        metrics = {
            "collect_env_steps_per_s": rate,
            "episode_time_mean_s": (mean.item() - 1) * dt,
            "episode_time_std_s": std.item() * dt,
        }
        samples = readout_rates.flatten(0, 1).cpu() if readout_rates is not None else None
        return (
            ReplayShard(steps, states, self.tags, self.held_out, plan.first_snapshot, samples),
            metrics,
        )


class Replay:
    def __init__(self, burn_in: int, length: int):
        self.window = burn_in + length
        self.shards: list[ReplayShard] = []

    def sample(
        self, size: int, generator: torch.Generator, *, held_out: bool = False
    ) -> SequenceBatch:
        rows = [torch.where(shard.held_out == held_out)[0] for shard in self.shards]
        counts = [
            len(envs)
            * (1 + (len(shard.steps.code) - shard.first_snapshot - self.window) // STATE_STRIDE)
            for shard, envs in zip(self.shards, rows, strict=True)
        ]
        total = sum(counts)
        if not total:
            raise ValueError("Replay has no complete sequences for this split")
        if held_out:
            draws = torch.randperm(total, generator=generator)[:size]
        else:
            draws = torch.randint(total, (size,), generator=generator)
        first = self.shards[0]
        steps = _allocate_steps(self.window, len(draws), first.steps.code.size(-1), "cpu")
        states = first.states.new_empty((len(draws), first.states.size(-1)))
        tags = first.tags.new_empty((len(draws), first.tags.size(-1)))
        time_offsets = torch.arange(self.window)[:, None]
        offset = 0
        for shard, envs, count in zip(self.shards, rows, counts, strict=True):
            positions = torch.where((draws >= offset) & (draws < offset + count))[0]
            if len(positions):
                local = draws[positions] - offset
                snaps, env = local // len(envs), envs[local % len(envs)]
                ticks = shard.first_snapshot + STATE_STRIDE * snaps + time_offsets
                for dest, source in zip(steps, shard.steps, strict=True):
                    dest[:, positions] = source[ticks, env]
                states[positions] = shard.states[snaps, env]
                tags[positions] = shard.tags[env]
            offset += count
        return SequenceBatch(steps.code, steps.action, steps.reset, states, tags, steps.occluded)


def normalization_stats(shard: ReplayShard, norm) -> dict[str, float]:
    """Exact per-modality extrema and quantiles over valid values, one modality in memory."""
    width = sum(norm.layout.widths)
    codes = shard.steps.code.reshape(-1, shard.steps.code.size(-1)).numpy()
    mean = norm.mean.cpu().numpy()
    std = norm.std.clamp_min(norm.std_floor).cpu().numpy()
    values, first = {}, 0
    for index, (name, count) in enumerate(
        zip(("left_eye", "right_eye", "odor", "wind"), norm.layout.widths, strict=True)
    ):
        active = codes[:, width + index] > 0.5
        observations = np.empty(int(active.sum()) * count, dtype=np.float32)
        channels = slice(first, first + count)
        first += count
        cursor = 0
        for start in range(0, len(codes), 8192):
            block = slice(start, start + 8192)
            data = codes[block, channels][active[block]]
            data -= mean[channels]
            data /= std[channels]
            np.abs(data, out=data)
            observations[cursor : cursor + data.size] = data.reshape(-1)
            cursor += data.size
        maximum, quantile = 0.0, 0.0
        if cursor:
            maximum = float(observations.max())
            rank = (cursor - 1) * 0.999
            lo, hi = math.floor(rank), math.ceil(rank)
            observations.partition((lo, hi))
            quantile = float(observations[lo]) + (rank - lo) * (
                float(observations[hi]) - float(observations[lo])
            )
        values[f"z_{name}_max"] = maximum
        values[f"z_{name}_p999"] = quantile
        del observations
    return values
