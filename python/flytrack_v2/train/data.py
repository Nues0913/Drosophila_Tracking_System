"""Batched rollouts and CPU replay with globally aligned recurrent snapshots."""

from __future__ import annotations

import ctypes
import math
import mmap
import os
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from typing import NamedTuple

import numpy as np
import torch
from torch import Tensor

from flytrack_v2.eval.runner import rotate_wind, swap_odor
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
_PACK_ROWS = 64
_REPLAY_LAYOUT = "env_major_f16"
_PAIRED_LAYOUT = "env_major_f16_paired"  # #138 A2: with chem_pair and action_pair
_READOUT_SAMPLE_STRIDE = 4
_TEACHER_KEYS = ("agent_pose", "agent_velocity", "target_pose", "target_velocity", "occluded")
_MADVISE_WARNING_LOCK = Lock()
_madvise_warned = False


def mix_actions(teacher: Tensor, student: Tensor, beta: float, generator: torch.Generator):
    choose_teacher = torch.rand(len(teacher), device=teacher.device, generator=generator) < beta
    return torch.where(choose_teacher[:, None], teacher, student)


class StepData(NamedTuple):
    code: Tensor
    action: Tensor
    reset: Tensor
    occluded: Tensor
    # #138 A2, only when pair_weight > 0: the chemical columns of the step's code for the
    # transformed observation (even rows: odor swapped; odd rows: wind rotated by 90 degrees),
    # and the teacher's label for it from the same teacher state.
    chem_pair: Tensor | None = None
    action_pair: Tensor | None = None


def pair_transform(obs: dict[str, Tensor], antenna_angle_deg: float) -> dict[str, Tensor]:
    """E1's two ablations (flytrack_v2.eval.runner), split by row: even rows see the odor
    swapped, odd rows the wind rotated by 90 degrees."""
    odd = (torch.arange(len(obs["odor"]), device=obs["odor"].device) % 2 == 1)[:, None]
    swapped, rotated = swap_odor(obs), rotate_wind(90.0, antenna_angle_deg)(obs)
    return {
        **obs,
        "odor": torch.where(odd, obs["odor"], swapped["odor"]),
        "wind": torch.where(odd, rotated["wind"], obs["wind"]),
    }


@dataclass
class ReplayShard:
    steps: StepData  # [envs, ticks, ...], code is float16
    states: Tensor  # [envs, snapshots, state_size], float16
    tags: Tensor  # [envs, 3], indices into WORLD_NAMES, MOTION_NAMES, SCENARIO_NAMES
    held_out: Tensor  # [envs], fixed throughout the run
    first_snapshot: int
    readout_rates: Tensor | None = None  # [samples, DNs], only for the first fit

    def checkpoint(self) -> dict:
        steps = {key: value for key, value in self.steps._asdict().items() if value is not None}
        layout = _REPLAY_LAYOUT if self.steps.chem_pair is None else _PAIRED_LAYOUT
        saved = {**vars(self), "layout": layout, "steps": steps}
        if self.readout_rates is None:
            del saved["readout_rates"]
        return saved

    @classmethod
    def restore(cls, saved: dict) -> ReplayShard:
        values = saved.copy()
        layout = values.pop("layout", "legacy time-major float32")
        if layout not in (_REPLAY_LAYOUT, _PAIRED_LAYOUT):
            raise ValueError(
                f"Unsupported replay layout {layout!r}; expected {_REPLAY_LAYOUT!r}. "
                "Start a new DAgger run in a new output directory."
            )
        values["steps"] = StepData(**values["steps"])
        return cls(**values)

    @classmethod
    def load(cls, path: Path) -> ReplayShard:
        global _madvise_warned
        shard = cls.restore(torch.load(path, map_location="cpu", weights_only=True, mmap=True))
        # PyTorch exposes the mapped storages, but keeps the mmap handle internal.
        madvise = ctypes.CDLL(None, use_errno=True).madvise
        madvise.argtypes = (ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int)
        madvise.restype = ctypes.c_int
        for tensor in (*shard.steps, shard.states, shard.tags, shard.held_out):
            if tensor is None:
                continue
            storage = tensor.untyped_storage()
            size = storage.nbytes()
            if size:
                address = storage.data_ptr()
                page_offset = address % mmap.PAGESIZE
                if madvise(address - page_offset, size + page_offset, mmap.MADV_RANDOM) != 0:
                    errno = ctypes.get_errno()
                    with _MADVISE_WARNING_LOCK:
                        if _madvise_warned:
                            continue
                        _madvise_warned = True
                    warnings.warn(
                        f"madvise(MADV_RANDOM) failed for {path}: "
                        f"[Errno {errno}] {os.strerror(errno)}",
                        stacklevel=2,
                    )
        return shard


class SequenceBatch(NamedTuple):
    code: Tensor
    action: Tensor
    reset: Tensor
    initial: Tensor
    tags: Tensor
    occluded: Tensor
    chem_pair: Tensor | None = None  # #138 A2, see StepData
    action_pair: Tensor | None = None

    def to(self, device: torch.device) -> SequenceBatch:
        return SequenceBatch(*(None if value is None else value.to(device) for value in self))


def _allocate_steps(
    ticks: int, rows: int, width: int, device: torch.device | str, chem_width: int = 0
) -> StepData:
    """chem_width > 0 also allocates the #138 A2 pair fields."""
    pair = (
        (
            torch.empty(ticks, rows, chem_width, dtype=torch.float32, device=device),
            torch.empty(ticks, rows, 2, dtype=torch.float32, device=device),
        )
        if chem_width
        else (None, None)
    )
    return StepData(
        torch.empty(ticks, rows, width, dtype=torch.float32, device=device),
        torch.empty(ticks, rows, 2, dtype=torch.float32, device=device),
        torch.empty(ticks, rows, dtype=torch.bool, device=device),
        torch.empty(ticks, rows, dtype=torch.bool, device=device),
        *pair,
    )


def _env_major(time_major: Tensor, dtype: torch.dtype) -> Tensor:
    rows = time_major.transpose(0, 1)
    packed = torch.empty(rows.shape, dtype=dtype, device=rows.device)
    for start in range(0, len(rows), _PACK_ROWS):
        block = slice(start, start + _PACK_ROWS)
        packed[block].copy_(rows[block])
    return packed


def _concatenate(parts: list[dict[str, Tensor]], keys) -> dict[str, Tensor]:
    return {name: torch.cat([part[name] for part in parts]) for name in keys}


class RolloutCollector:
    def __init__(self, policy, config: dict, seed: int, device: torch.device, *, restore=None):
        self.policy, self.device = policy, device
        rows = config["envs_per_instance"]
        self.rows = rows * len(WORLD_NAMES) * len(MOTION_NAMES)
        self.mix_rng = torch.Generator(device=device).manual_seed(seed)
        self.limit_acceleration = config.get("teacher_limit_acceleration", True)
        self.chem_width = 0
        if config.get("pair_weight", 0.0) > 0:  # #138 A2
            columns = policy.encoder.chemical_columns
            self.chem_width = columns.stop - columns.start
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

    def _first_episode_resets(self, observations: list, resets: list, before_reset=None) -> None:
        """End the scheduled first episodes early (desync_first_episode), in place.

        ``before_reset(world_id, mask)`` runs before a world resets those rows, while the
        world still holds their last state (PPO bootstraps from it, #138).
        """
        if self._first_reset_steps is None:
            return
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
                if before_reset is not None:
                    before_reset(world_id, mask)
                # The public reset commits into the same buffers used by the CUDA graphs.
                observations[world_id] = world.reset(mask)
                resets[world_id] |= mask
                forced |= mask

    @torch.no_grad()
    def step(
        self,
        beta: float,
        initial: Tensor,
        *,
        episode_steps: Tensor | None = None,
        readout_sample: tuple[Tensor, Tensor] | None = None,
    ) -> StepData:
        sensory_before = self.sensory
        code, self.sensory = self.policy.sense(self.obs, self.sensory)
        student = self.policy.forward_code(code, self.central)
        if readout_sample is not None:
            rows, output = readout_sample
            torch.index_select(student.aux["dn_rates"], 0, rows, out=output)
        privileged = _concatenate([world.privileged() for world in self.worlds], _TEACHER_KEYS)
        teacher_before = self.teacher_state
        label, self.teacher_state = self.teacher.act(
            privileged,
            self.obs,
            privileged["agent_velocity"],
            self.teacher_state,
            limit_acceleration=self.limit_acceleration,
        )
        pair = (None, None)
        if self.chem_width:
            # The same sensory and teacher state, one transformed observation; neither state
            # advances from it (#138 A2).
            shown = pair_transform(self.obs, self.policy.contract.antenna_angle_deg)
            chem = self.policy.encoder.chemical_code(shown, sensory_before)
            label_pair, _ = self.teacher.act(
                privileged,
                shown,
                privileged["agent_velocity"],
                teacher_before,
                limit_acceleration=self.limit_acceleration,
            )
            pair = (chem, label_pair)
        action = mix_actions(label, student.dist.mode(), beta, self.mix_rng)
        results = [
            world.step(rows)
            for world, rows in zip(self.worlds, action.split(self.worlds[0].num_envs), strict=True)
        ]
        if episode_steps is not None:
            torch.cat([result.info["episode_steps"] for result in results], out=episode_steps)
        resets = [result.terminated | result.truncated for result in results]
        observations = [result.obs for result in results]
        self._first_episode_resets(observations, resets)
        reset = torch.cat(resets)
        self.obs = self._observations(observations)
        self.sensory = self.policy.reset_sensory_state(self.sensory, reset)
        self.central = reset_rows(student.central_state, initial, reset)
        self.teacher_state = reset_rows(self.teacher_state, self.teacher_initial, reset)
        return StepData(code, label, reset, privileged["occluded"], *pair)

    @torch.no_grad()
    def collect(
        self, plan: RoundPlan, *, sample_readout_rates: bool = False
    ) -> tuple[ReplayShard, dict[str, float], dict[str, float]]:
        start = time.perf_counter()
        steps = _allocate_steps(
            plan.ticks, self.rows, self.policy.code_size, "cpu", self.chem_width
        )
        episode_steps = torch.empty(plan.ticks, self.rows, dtype=torch.int64, device=self.device)
        snapshots = range(plan.first_snapshot, plan.ticks, STATE_STRIDE)
        states = torch.empty(
            self.rows, len(snapshots), self.policy.central_state_size, dtype=torch.float16
        )
        staged_states = torch.empty(
            min(len(snapshots), math.ceil(_TRANSFER_TICKS / STATE_STRIDE)),
            self.rows,
            self.policy.central_state_size,
            dtype=torch.float32,
            device=self.device,
        )
        initial = self.policy.initial_central_state(self.rows, self.device)
        is_cuda = self.device.type == "cuda"
        cpu_states = torch.empty_like(staged_states, device="cpu") if is_cuda else staged_states
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
        staging = (
            _allocate_steps(
                _TRANSFER_TICKS, self.rows, self.policy.code_size, self.device, self.chem_width
            )
            if is_cuda
            else steps
        )
        next_snapshot = 0
        for start_tick in range(0, plan.ticks, _TRANSFER_TICKS):
            end = min(start_tick + _TRANSFER_TICKS, plan.ticks)
            first_saved = next_snapshot
            for tick in range(start_tick, end):
                if tick in snapshots:
                    staged_states[next_snapshot - first_saved].copy_(self.central)
                    next_snapshot += 1
                slot = tick - start_tick if is_cuda else tick
                sample = None
                if readout_rates is not None and tick % _READOUT_SAMPLE_STRIDE == 0:
                    sample = train_rows, readout_rates[tick // _READOUT_SAMPLE_STRIDE]
                data = self.step(
                    plan.beta, initial, episode_steps=episode_steps[tick], readout_sample=sample
                )
                for buffer, value in zip(staging, data, strict=True):
                    if buffer is not None:
                        buffer[slot].copy_(value)
            saved_count = next_snapshot - first_saved
            if is_cuda:
                for dest, source in zip(steps, staging, strict=True):
                    if dest is not None:
                        dest[start_tick:end].copy_(source[: end - start_tick])
                cpu_states[:saved_count].copy_(staged_states[:saved_count])
            # Pack into the shard as we collect, keeping the float16 cast on CPU.
            states[:, first_saved:next_snapshot].copy_(cpu_states[:saved_count].transpose(0, 1))
        del staged_states, cpu_states
        rate = plan.ticks * self.rows / (time.perf_counter() - start)
        if is_cuda:
            for world in self.worlds:
                world.release_rollout_graphs()
        # Code and labels use the observation before the action, one step before the info count.
        std, mean = torch.std_mean(episode_steps.to(torch.float64), correction=0)
        del episode_steps
        dt = self.policy.contract.control_dt
        metrics = {
            "collect_env_steps_per_s": rate,
            "episode_time_mean_s": (mean.item() - 1) * dt,
            "episode_time_std_s": std.item() * dt,
        }
        samples = readout_rates.flatten(0, 1).cpu() if readout_rates is not None else None
        code, action, reset, occluded, chem_pair, action_pair = steps
        del steps, staging
        stats = normalization_stats(code, self.policy.input_norm)
        code = _env_major(code, torch.float16)
        steps = StepData(
            code,
            _env_major(action, action.dtype),
            _env_major(reset, reset.dtype),
            _env_major(occluded, occluded.dtype),
            *(
                (None, None)
                if chem_pair is None
                # float16 like the code, so an unchanged column compares equal to the code's
                else (_env_major(chem_pair, torch.float16), _env_major(action_pair, torch.float32))
            ),
        )
        return (
            ReplayShard(steps, states, self.tags, self.held_out, plan.first_snapshot, samples),
            metrics,
            stats,
        )


class Replay:
    def __init__(self, burn_in: int, length: int):
        self.window = burn_in + length
        self.shards: list[ReplayShard] = []

    def sample(
        self, size: int, generator: torch.Generator, *, held_out: bool = False
    ) -> SequenceBatch:
        rows = [torch.where(shard.held_out == held_out)[0] for shard in self.shards]
        starts = [
            max(
                0,
                1 + (shard.steps.code.size(1) - shard.first_snapshot - self.window) // STATE_STRIDE,
            )
            for shard in self.shards
        ]
        counts = [len(envs) * n_starts for envs, n_starts in zip(rows, starts, strict=True)]
        total = sum(counts)
        if not total:
            raise ValueError("Replay has no complete sequences for this split")
        if held_out:
            draws = torch.randperm(total, generator=generator)[:size]
        else:
            draws = torch.randint(total, (size,), generator=generator)
        first = self.shards[0]
        chem = first.steps.chem_pair
        steps = _allocate_steps(
            self.window,
            len(draws),
            first.steps.code.size(-1),
            "cpu",
            0 if chem is None else chem.size(-1),
        )
        states = first.states.new_empty((len(draws), first.states.size(-1)), dtype=torch.float32)
        tags = first.tags.new_empty((len(draws), first.tags.size(-1)))
        offset = 0
        for shard, envs, n_starts, count in zip(self.shards, rows, starts, counts, strict=True):
            positions = torch.where((draws >= offset) & (draws < offset + count))[0]
            if len(positions):
                local = draws[positions] - offset
                snaps, env = local // len(envs), envs[local % len(envs)]
                order = (env * n_starts + snaps).argsort()
                positions, env, snaps = positions[order], env[order], snaps[order]
                reads = list(zip(positions.tolist(), env.tolist(), snaps.tolist(), strict=True))
                for dest, source in zip(steps, shard.steps, strict=True):
                    if dest is None:
                        continue
                    for position, row, snap in reads:
                        tick = shard.first_snapshot + STATE_STRIDE * snap
                        dest[:, position].copy_(source[row, tick : tick + self.window])
                for position, row, snap in reads:
                    states[position].copy_(shard.states[row, snap])
                tags[positions] = shard.tags[env]
            offset += count
        return SequenceBatch(
            steps.code, steps.action, steps.reset, states, tags, steps.occluded,
            steps.chem_pair, steps.action_pair,
        )  # fmt: skip


def normalization_stats(code: Tensor, norm) -> dict[str, float]:
    """Exact per-modality extrema and quantiles over valid values, one modality in memory."""
    width = sum(norm.layout.widths)
    codes = code.reshape(-1, code.size(-1)).numpy()
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
