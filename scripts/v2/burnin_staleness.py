"""Measure recurrent snapshot staleness in saved env-major DAgger replay.

Pass an exported policy directory (usually RUN/final) to --weights. For a DAgger
checkpoint file, RUN/final supplies the matching graph, frontend, contract and
input normalization. The checkpoint must match RUN/config.json and supplies the
policy state dict (including DN normalization) and the saved reset state, when
present. The collection-weight control uses the same procedure with the previous
round's checkpoint, or RUN/initial.pt for round 1 when it exists.

For a pre-sense snapshot at tick t, measure after consuming code[t:t+n]. Require
a complete reset-free window of config.sequence.burn_in + max(--steps) ticks.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections.abc import Iterator
from pathlib import Path
from typing import NamedTuple

import numpy as np
import torch
from flytrack_v2.model import load_policy
from flytrack_v2.model.policy import ConnectomePolicyV2
from flytrack_v2.train.config import STATE_STRIDE
from flytrack_v2.train.data import ReplayShard
from torch import Tensor

AGE_LIMITS = (16, 64, 256, 1024)
AGE_LABELS = ("0–16", "17–64", "65–256", "257–1024", ">1024")
BATCH_SIZE = 64
CODE_BLOCK_TICKS = 64
PM_STEP = 16
PM_THRESHOLD = 0.3
METRICS = (
    "state_rmse",
    "dn_rmse_sigma",
    "dn_mae_sigma",
    "speed_mode_mae_m_s",
    "turn_mode_mae_rad_s",
)


class Snapshot(NamedTuple):
    row: int
    tick: int
    age: int


def snapshot_bins(
    shard: ReplayShard, window: int, *, first_round: bool
) -> tuple[list[list[Snapshot]], dict[str, int]]:
    resets_by_row = shard.steps.reset.numpy()
    rows, ticks = resets_by_row.shape
    snapshots = shard.first_snapshot + np.arange(shard.states.size(1)) * STATE_STRIDE
    complete = snapshots[snapshots + window <= ticks]
    counts = {
        "snapshots": rows * len(snapshots),
        "excluded_short_window": rows * (len(snapshots) - len(complete)),
        "excluded_reset": 0,
        "excluded_cross_round": 0,
    }
    bins: list[list[Snapshot]] = [[] for _ in AGE_LABELS]
    for row, resets in enumerate(resets_by_row):
        # Flag r describes the reset after code[r], so an episode starts at r + 1.
        ends = np.flatnonzero(resets)
        next_end = np.searchsorted(ends, complete, side="left")
        clear = np.append(ends, ticks)[next_end] >= complete + window
        counts["excluded_reset"] += int(np.count_nonzero(~clear))
        known_start = (next_end > 0) | first_round
        counts["excluded_cross_round"] += int(np.count_nonzero(clear & ~known_start))
        keep = clear & known_start
        starts = np.insert(ends + 1, 0, 0)[next_end[keep]]
        eligible = complete[keep]
        ages = eligible - starts
        groups = np.searchsorted(AGE_LIMITS, ages, side="left")
        for tick, age, group in zip(eligible.tolist(), ages.tolist(), groups.tolist(), strict=True):
            bins[group].append(Snapshot(row, tick, age))
    return bins, counts


def load_dn_policy(artifact: Path, device: torch.device, variant: str) -> ConnectomePolicyV2:
    policy = load_policy(artifact, device=device)
    if not isinstance(policy, ConnectomePolicyV2) or not policy.readout.node_indices.numel():
        raise ValueError("The burn-in probe requires a connectome policy with DN readout features")
    if policy.variant != variant:
        raise ValueError(f"{artifact} has variant {policy.variant}, expected {variant}")
    return policy


def restore_checkpoint(policy: ConnectomePolicyV2, path: Path, config_sha256: str) -> None:
    saved = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    if saved["config_sha256"] != config_sha256:
        raise ValueError(f"{path} does not match run/config.json")
    policy.load_state_dict(saved["policy"], strict=True)
    if saved["rest_state"] is not None:
        # On CPU, the cached reset state must not keep the checkpoint mmap alive.
        policy.restore_rest_state(saved["rest_state"].clone())


def replay_codes(
    shard: ReplayShard,
    batch: list[Snapshot],
    first: int,
    stop: int,
    device: torch.device,
) -> Iterator[tuple[Tensor, int]]:
    """Yield code and active row count by tick relative to each snapshot.

    The batch is sorted oldest first. Before offset zero, each row comes into
    scope at its episode start. Pack and transfer at most one block of code.
    """
    active = 0
    for left in range(first, stop, CODE_BLOCK_TICKS):
        right = min(left + CODE_BLOCK_TICKS, stop)
        block = shard.steps.code.new_zeros((right - left, len(batch), shard.steps.code.size(2)))
        for column, (row, tick, age) in enumerate(batch):
            start = max(left, -age)
            if start < right:
                block[start - left :, column].copy_(
                    shard.steps.code[row, tick + start : tick + right]
                )
        block = block.to(device, dtype=torch.float32)
        for offset, code in enumerate(block, start=left):
            while active < len(batch) and offset >= -batch[active].age:
                active += 1
            yield code[:active], active


def readout_errors(policy: ConnectomePolicyV2, pairs: Tensor, size: int) -> Tensor:
    rates = policy.core.rates(pairs.index_select(1, policy.readout.node_indices))
    dn_reference, dn_stale = policy.readout.normalize_rates(rates).split(size)
    action_reference, action_stale = policy._distribution(policy.readout(rates)).mode().split(size)
    reference, stale = pairs.split(size)
    dn_delta = dn_stale - dn_reference
    return torch.column_stack(
        (
            (stale - reference).square().mean(dim=1).sqrt(),
            dn_delta.square().mean(dim=1).sqrt(),
            dn_delta.abs().mean(dim=1),
            (action_stale - action_reference).abs(),
        )
    )


@torch.no_grad()
def measure_cell(
    shard: ReplayShard,
    samples: list[Snapshot],
    policy: ConnectomePolicyV2,
    collection_policy: ConnectomePolicyV2 | None,
    steps: list[int],
    device: torch.device,
) -> tuple[dict, float | None]:
    if not samples:
        return {
            step: {"samples": 0, **dict.fromkeys(METRICS), "state_rmse_fraction_of_step_0": None}
            for step in (0, *steps)
        }, None

    ordered = sorted(samples, key=lambda sample: sample.age, reverse=True)
    sums = torch.zeros((len(steps) + 1, len(METRICS)), dtype=torch.float64, device=device)
    collection_sum = sums.new_zeros(())
    for left in range(0, len(ordered), BATCH_SIZE):
        batch = ordered[left : left + BATCH_SIZE]
        size = len(batch)
        reference = policy.initial_central_state(size, device)
        collection = (
            collection_policy.initial_central_state(size, device)
            if collection_policy is not None
            else None
        )
        for code, active in replay_codes(shard, batch, -batch[0].age, 0, device):
            reference[:active] = policy.forward_code(code, reference[:active]).central_state
            if collection is not None:
                collection[:active] = collection_policy.forward_code(
                    code, collection[:active]
                ).central_state
        rows, ticks, _ = zip(*batch, strict=True)
        slots = [(tick - shard.first_snapshot) // STATE_STRIDE for tick in ticks]
        stale = shard.states[list(rows), slots].to(device, dtype=torch.float32)
        if collection is not None:
            collection_sum += (stale - collection).square().mean(dim=1).sqrt().sum(dtype=sums.dtype)

        pairs = torch.cat((reference, stale))
        sums[0] += readout_errors(policy, pairs, size).sum(dim=0, dtype=sums.dtype)
        next_step = 0
        future = replay_codes(shard, batch, 0, steps[-1], device)
        for n, (code, _) in enumerate(future, start=1):
            pairs = policy.forward_code(code.repeat(2, 1), pairs).central_state
            if n == steps[next_step]:
                next_step += 1
                sums[next_step] += readout_errors(policy, pairs, size).sum(dim=0, dtype=sums.dtype)

    means = (sums / len(samples)).tolist()
    initial_rmse = means[0][0]
    measurements = {
        step: {
            "samples": len(samples),
            **dict(zip(METRICS, values, strict=True)),
            "state_rmse_fraction_of_step_0": values[0] / initial_rmse if initial_rmse else None,
        }
        for step, values in zip((0, *steps), means, strict=True)
    }
    control = collection_sum.item() / len(samples) if collection_policy is not None else None
    return measurements, control


def replay_rounds(run: Path) -> list[tuple[int, Path]]:
    rounds = []
    for directory in run.iterdir():
        match = re.fullmatch(r"round_([1-9][0-9]*)", directory.name)
        path = directory / "replay.pt"
        if match and path.is_file():
            rounds.append((int(match[1]), path))
    if not rounds:
        raise ValueError(f"No round_<k>/replay.pt files found in {run}")
    return sorted(rounds)


def probe(args: argparse.Namespace) -> dict:
    config_bytes = (args.run / "config.json").read_bytes()
    config = json.loads(config_bytes)
    config_sha256 = hashlib.sha256(config_bytes).hexdigest()
    burn_in = config["sequence"]["burn_in"]
    window = burn_in + max(args.steps)
    rounds = replay_rounds(args.run)
    device = torch.device(args.device)
    artifact = args.weights if args.weights.is_dir() else args.run / "final"
    policy = load_dn_policy(artifact, device, config["variant"])
    if args.weights.is_file():
        restore_checkpoint(policy, args.weights, config_sha256)
    collection_policy = None
    rng = np.random.default_rng(args.seed)
    results = []
    pm_total = 0.0
    pm_count = 0
    for number, path in rounds:
        shard = ReplayShard.load(path)
        if (
            shard.steps.code.size(2) != policy.code_size
            or shard.states.size(2) != policy.central_state_size
        ):
            raise ValueError(f"{path} has code or state dimensions incompatible with the policy")
        bins, counts = snapshot_bins(shard, window, first_round=number == 1)
        collection_path = (
            args.run / "initial.pt"
            if number == 1
            else args.run / f"round_{number - 1}" / "checkpoint.pt"
        )
        control = None
        if collection_path.is_file() and any(bins):
            if collection_policy is None:
                collection_policy = load_dn_policy(args.run / "final", device, config["variant"])
            restore_checkpoint(collection_policy, collection_path, config_sha256)
            control = collection_policy

        cells = []
        for age_bin, candidates in zip(AGE_LABELS, bins, strict=True):
            samples = candidates
            if len(candidates) > args.per_cell:
                indices = rng.choice(len(candidates), size=args.per_cell, replace=False)
                samples = [candidates[index] for index in indices]
            measurements, collection_rmse = measure_cell(
                shard, samples, policy, control, args.steps, device
            )
            cells.append(
                {
                    "age_bin": age_bin,
                    "eligible": len(candidates),
                    "samples": len(samples),
                    "collection_state_rmse": collection_rmse,
                    "measurements": measurements,
                }
            )
            if PM_STEP in measurements and samples:
                pm_total += measurements[PM_STEP]["dn_rmse_sigma"] * len(samples)
                pm_count += len(samples)
        results.append(
            {
                "round": number,
                "replay": str(path),
                **counts,
                "collection_checkpoint": str(collection_path)
                if collection_path.is_file()
                else None,
                "cells": cells,
            }
        )
        del shard
    pm_mean = pm_total / pm_count if pm_count else None
    return {
        "run": str(args.run),
        "weights": str(args.weights),
        "device": str(device),
        "seed": args.seed,
        "per_cell": args.per_cell,
        "steps": args.steps,
        "burn_in": burn_in,
        "required_reset_free_steps": window,
        "dn_normalization_fitted": policy.readout.rate_norm_fitted.item(),
        "pm_trigger": {
            "step": PM_STEP,
            "threshold_sigma": PM_THRESHOLD,
            "samples": pm_count,
            "mean_dn_rmse_sigma": pm_mean,
            "exceeds_threshold": pm_mean > PM_THRESHOLD if pm_mean is not None else None,
        },
        "rounds": results,
    }


def markdown_number(value: float | None) -> str:
    return "—" if value is None else f"{value:.6g}"


def render_markdown(report: dict) -> str:
    pm = report["pm_trigger"]
    threshold = f"> {pm['threshold_sigma']} σ"
    if pm["exceeds_threshold"] is None:
        summary = f"unavailable (no 16-step measurements); {threshold} trigger undetermined"
    else:
        answer = "yes" if pm["exceeds_threshold"] else "no"
        summary = f"{pm['mean_dn_rmse_sigma']:.6g} σ (n={pm['samples']}), {threshold}: {answer}"
    lines = [
        f"**Mean DN readout RMSE after 16 steps across all samples: {summary}.**",
        "",
        f"Required reset-free window: {report['required_reset_free_steps']} ticks.",
        "Each RMSE is computed per snapshot over features and then averaged over snapshots.",
        "State fraction divides each cell's mean state RMSE by its step-0 mean.",
        "Collection RMSE compares the saved snapshot against replay from the episode start "
        "under the collection weights and includes rounding of both code and snapshots.",
        "",
        "| Round | Snapshots | Short window | Reset in window | Cross-round episode | "
        "Collection checkpoint |",
        "| ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in report["rounds"]:
        checkpoint = row["collection_checkpoint"]
        collection = "—" if checkpoint is None else Path(checkpoint).relative_to(report["run"])
        lines.append(
            f"| {row['round']} | {row['snapshots']} | {row['excluded_short_window']} | "
            f"{row['excluded_reset']} | {row['excluded_cross_round']} | {collection} |"
        )
    lines.extend(
        [
            "",
            "| Round | Age (ticks) | Steps | Eligible | Samples | State RMSE | State fraction | "
            "DN RMSE (σ) | DN MAE (σ) | Speed MAE (m/s) | Turn MAE (rad/s) | Collection RMSE |",
            "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in report["rounds"]:
        for cell in row["cells"]:
            for step, values in cell["measurements"].items():
                metrics = [
                    values["state_rmse"],
                    values["state_rmse_fraction_of_step_0"],
                    *(values[key] for key in METRICS[1:]),
                    cell["collection_state_rmse"] if step == 0 else None,
                ]
                columns = " | ".join(map(markdown_number, metrics))
                lines.append(
                    f"| {row['round']} | {cell['age_bin']} | {step} | {cell['eligible']} | "
                    f"{values['samples']} | {columns} |"
                )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--weights", required=True, type=Path)
    parser.add_argument("--per-cell", type=int, default=200)
    parser.add_argument("--steps", nargs="+", type=int, default=[16, 32, 48, 64])
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    if args.per_cell < 1 or min(args.steps) < 1:
        parser.error("--per-cell and --steps must be positive")
    if args.seed < 0:
        parser.error("--seed must be nonnegative")
    if not args.weights.exists():
        parser.error(f"--weights does not exist: {args.weights}")
    args.steps = sorted(set(args.steps))
    report = probe(args)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "burnin.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    (args.out / "burnin.md").write_text(render_markdown(report), encoding="utf-8")


if __name__ == "__main__":
    main()
