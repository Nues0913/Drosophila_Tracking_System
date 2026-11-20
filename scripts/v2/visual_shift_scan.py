"""Scan the evaluation-time visual working point of a v2 connectome artifact.

In code mode the InputNorm output starts with the left-eye and right-eye features.
InputNorm.layout.features_per_eye gives their width. The last len(VALID_CHANNELS)
columns are the flags, starting with left and right eye. Add delta to an eye's
normalized features only when its flag is finite and greater than 0.5. The odor,
wind and flag columns pass through unchanged. The hook also applies during the
policy's neutral rest-state refresh, as a learned offset does.

Use one output directory per artifact and mode. Each completed shift has a
cells.json alongside episodes.jsonl so its visual-drive statistics survive a resume.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import time
from contextlib import nullcontext
from functools import partial
from pathlib import Path
from tempfile import NamedTemporaryFile, TemporaryDirectory

import numpy as np
import torch
import yaml
from evaluate_v2 import (
    ROOT,
    SCENARIOS,
    build_world,
    episode_seeds,
    load_artifact,
    scenario_configs,
    train_seed_of,
)
from flytrack_v2.contracts import VALID_CHANNELS, SensorContractV2
from flytrack_v2.eval.runner import V2Actor, json_line, run_episodes, to_records
from flytrack_v2.metrics import validate_episode_record


def shift_visual_code(norm, _inputs, code, *, delta):
    width = norm.layout.features_per_eye
    vision = code[:, : 2 * width].reshape(len(code), 2, width)
    flags = code[:, -len(VALID_CHANNELS) :][:, :2]
    valid = (flags > 0.5) & flags.isfinite()
    shifted = torch.where(valid.unsqueeze(-1), vision + delta, vision)
    return torch.cat((shifted.flatten(1), code[:, 2 * width :]), dim=1)


def offset_policy(artifact, delta, expected_digest):
    # Reload a separate policy, including the sparse buffers reconstructed from the artifact.
    policy, _, digest = load_artifact(artifact)
    if digest != expected_digest:
        raise ValueError(f"Artifact changed during the scan: {artifact}")
    with torch.no_grad():
        policy.projection.visual_offset.add_(delta)
    return policy


def visual_groups(policy):
    projection = policy.projection
    # Every seed with the same visual source receives the same drive when its eye is valid.
    # Select one seed per RF group in source order to avoid weighting by seed multiplicity.
    sources = projection.source_index.numpy()
    visual_rows = np.flatnonzero(sources < len(projection.rf_eye))
    _, first = np.unique(sources[visual_rows], return_index=True)
    seeds = projection.seed_indices[visual_rows[first]]
    names = policy.graph.type_names[policy.graph.type_index[seeds.numpy()]].tolist()
    sums = {name: {"left": None, "right": None} for name in names}
    weights = torch.nn.functional.softplus(projection.lc_rf.detach()).sum(dim=1).tolist()
    for name, eye, weight in zip(names, projection.rf_eye.tolist(), weights, strict=True):
        sums[name][("left", "right")[eye]] = weight
    return seeds, names, sums


class DriveActor(V2Actor):
    """Measure one visual drive per RF group on valid eye-steps in the first episode."""

    def __init__(self, policy, group_seeds):
        super().__init__(policy)
        self.group_seeds = group_seeds
        self.live = None

    def reset(self, batch, device):
        self.live = None
        super().reset(batch, device)  # This runs the projection during rest-state refresh.
        self.live = torch.ones(batch, dtype=torch.bool, device=device)
        shape = (batch, len(self.group_seeds))
        self.totals = torch.zeros(shape, dtype=torch.float64, device=device)
        self.counts = torch.zeros(shape, dtype=torch.int64, device=device)

    def act(self, obs, reset_mask):
        if reset_mask is not None:
            self.live &= ~reset_mask
        return super().act(obs, reset_mask)

    def observe_drive(self, projection, inputs, output):
        if self.live is None:
            return
        flags = inputs[0][:, -len(VALID_CHANNELS) :].index_select(1, projection.rf_eye)
        valid = (flags > 0.5) & flags.isfinite() & self.live[:, None]
        # On valid rows the projected seed current has no missing bias. It is exactly
        # the pooled visual drive plus the learned offset.
        drive = output.index_select(1, self.group_seeds)
        self.totals.add_(torch.where(valid, drive, 0.0))
        self.counts.add_(valid)

    def by_scenario(self, scenarios, episodes, group_names):
        groups = len(group_names)
        totals = self.totals.reshape(len(scenarios), episodes, groups).sum(dim=1).tolist()
        counts = self.counts.reshape(len(scenarios), episodes, groups).sum(dim=1).tolist()
        for scenario, values, samples in zip(scenarios, totals, counts, strict=True):
            by_type = {}
            for name, value, count in zip(group_names, values, samples, strict=True):
                total, n = by_type.get(name, (0.0, 0))
                by_type[name] = (total + value, n + count)
            yield scenario, {name: total / n if n else None for name, (total, n) in by_type.items()}


def write_summary(output, settings, rf_sums, shifts, cells_by_shift):
    data = {
        **settings,
        "rf_weight_sums": rf_sums,
        "cells": [
            {"shift": shift, **cell} for shift in shifts for cell in cells_by_shift.get(shift, [])
        ],
    }
    columns = [
        (world, motion, scenario)
        for world in settings["worlds"]
        for motion in settings["motions"]
        for scenario in settings["scenarios"]
    ]
    lines = [
        "# Visual shift scan",
        "",
        f"Mode: {settings['mode']}. Cells show success (collision).",
        "",
        "| δ | " + " | ".join(" / ".join(column) for column in columns) + " |",
        "| --- |" + " --- |" * len(columns),
    ]
    for shift in shifts:
        if shift not in cells_by_shift:
            continue
        cells = {
            (cell["world"], cell["motion"], cell["scenario"]): cell
            for cell in cells_by_shift[shift]
        }
        rates = [
            f"{cells[key]['success_rate']:.1%} ({cells[key]['collision_rate']:.1%})"
            for key in columns
        ]
        lines.append(f"| {shift_name(shift)} | " + " | ".join(rates) + " |")
    lines += [
        "",
        "## RF weight sums (softplus)",
        "",
        "| LC type | Left | Right |",
        "| --- | --- | --- |",
    ]
    for name, sides in rf_sums.items():
        weights = [f"{weight:.6f}" if weight is not None else "—" for weight in sides.values()]
        lines.append(f"| {name} | " + " | ".join(weights) + " |")

    for name, content in (
        ("scan.json", json.dumps(data, indent=2, allow_nan=False)),
        ("scan.md", "\n".join(lines)),
    ):
        with NamedTemporaryFile(mode="w", dir=output, encoding="utf-8", delete=False) as stream:
            temporary = Path(stream.name)
            try:
                stream.write(content + "\n")
                stream.close()
                temporary.replace(output / name)
            finally:
                temporary.unlink(missing_ok=True)


def shift_name(shift):
    return "0" if shift == 0 else str(shift).removesuffix(".0")


def evaluate_shift(actor, protocol, contract, settings, seeds, names, run_id, output):
    scenarios, episodes = settings["scenarios"], settings["episodes_per_cell"]
    max_steps = math.ceil(protocol["episode_seconds"] / contract.control_dt) + 5
    cells = []
    with actor.policy.projection.register_forward_hook(actor.observe_drive):
        for world_kind in settings["worlds"]:
            for motion in settings["motions"]:
                started = time.perf_counter()
                world, rows = build_world(
                    contract,
                    world_kind,
                    motion,
                    scenarios,
                    settings["noise_scale"],
                    protocol,
                    episodes,
                    settings["device"],
                )
                result = run_episodes(world, actor, seeds, max_steps=max_steps)
                del world
                records = to_records(
                    result,
                    rows,
                    run_id=run_id,
                    variant=actor.policy.variant,
                    train_seed=settings["train_seed"],
                    world=world_kind,
                    motion=motion,
                    artifact_sha256=settings["artifact_sha256"],
                    commit=settings["commit"],
                )
                for record in records:
                    validate_episode_record(record)
                    output.write(json_line(record))
                for i, (scenario, drive) in enumerate(
                    actor.by_scenario(scenarios, episodes, names)
                ):
                    subset = records[i * episodes : (i + 1) * episodes]
                    cells.append(
                        {
                            "world": world_kind,
                            "motion": motion,
                            "scenario": scenario,
                            "success_rate": sum(record["success"] for record in subset) / episodes,
                            "collision_rate": sum(record["collision"] for record in subset)
                            / episodes,
                            "tracking_fraction": sum(
                                record["tracking_fraction"] for record in subset
                            )
                            / episodes,
                            "visual_drive_mean": drive,
                        }
                    )
                elapsed = time.perf_counter() - started
                print(f"{world_kind} {motion}: {elapsed:.1f} s", flush=True)
    return cells


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--shifts", nargs="+", type=float, required=True)
    parser.add_argument("--mode", choices=("code", "offset"), default="code")
    parser.add_argument("--scenarios", nargs="+", choices=SCENARIOS, default=SCENARIOS)
    parser.add_argument("--seeds", choices=("formal_evaluation", "pilot_held_out"), required=True)
    parser.add_argument("--episodes", type=int)
    parser.add_argument(
        "--protocol", type=Path, default=ROOT / "configs/v2/evaluation_protocol.yaml"
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not all(math.isfinite(shift) for shift in args.shifts):
        parser.error("--shifts requires finite values")
    if len(set(args.scenarios)) != len(args.scenarios):
        parser.error("--scenarios requires distinct scenarios")

    protocol_bytes = args.protocol.read_bytes()
    protocol = yaml.safe_load(protocol_bytes)
    contract = SensorContractV2.from_json(ROOT / "configs/v2/sensor_contract.json")
    episodes = protocol["episodes_per_cell"] if args.episodes is None else args.episodes
    if episodes < 1:
        parser.error("--episodes must be positive")
    scale = float(protocol["scenarios"]["noise"]["scale"])
    policy, card, digest = load_artifact(args.artifact)
    projection = getattr(policy, "projection", None)
    if projection is None or projection.lc_rf is None or not projection.enable_vision:
        parser.error("the artifact must have an enabled visual SeedProjection")
    if args.mode == "offset" and projection.visual_offset is None:
        parser.error("offset mode requires a learned projection.visual_offset")
    train_seed = train_seed_of(args.artifact, card)
    seeds = episode_seeds(protocol, args.seeds, train_seed, episodes, len(args.scenarios))
    group_seeds, names, rf_sums = visual_groups(policy)
    configs = scenario_configs(scale, protocol)
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    settings = {
        "mode": args.mode,
        "artifact_sha256": digest,
        "train_seed": train_seed,
        "seeds": args.seeds,
        "episodes_per_cell": episodes,
        "device": args.device,
        "commit": commit,
        "protocol_sha256": hashlib.sha256(protocol_bytes).hexdigest(),
        "noise_scale": scale,
        "worlds": protocol["worlds"],
        "motions": [protocol["motions"]["primary"], protocol["motions"]["secondary"]],
        "scenarios": list(args.scenarios),
        "scenario_configs": {name: configs[name] for name in args.scenarios},
    }

    args.output.mkdir(parents=True, exist_ok=True)
    shifts = dict.fromkeys(0.0 if shift == 0 else shift for shift in args.shifts)
    cells_by_shift = {}
    for shift in shifts:
        label = f"{args.mode}_{shift_name(shift)}"
        shift_dir = args.output / label
        if (shift_dir / "episodes.jsonl").is_file():
            saved = json.loads((shift_dir / "cells.json").read_text(encoding="utf-8"))
            if saved["settings"] != settings:
                parser.error(f"Completed shift {label} has different settings. Use another output.")
            cells_by_shift[shift] = saved["cells"]
            print(f"{label}: already complete", flush=True)
    for shift in shifts:
        if (args.output / "STOP").exists():
            print("STOP found. Ending the scan before the next shift.", flush=True)
            break
        if shift in cells_by_shift:
            continue
        label = f"{args.mode}_{shift_name(shift)}"
        print(label, flush=True)
        shifted = (
            offset_policy(args.artifact, shift, digest)
            if args.mode == "offset" and shift
            else policy
        )
        actor = DriveActor(shifted.to(args.device), group_seeds.to(args.device))
        hook = nullcontext()
        if args.mode == "code" and shift:
            hook = shifted.input_norm.register_forward_hook(partial(shift_visual_code, delta=shift))
        shift_dir = args.output / label
        shift_dir.mkdir(exist_ok=True)
        with hook, TemporaryDirectory(dir=shift_dir) as temporary:
            stage = Path(temporary)
            with (stage / "episodes.jsonl").open("w", encoding="utf-8") as stream:
                cells = evaluate_shift(
                    actor,
                    protocol,
                    contract,
                    settings,
                    seeds,
                    names,
                    f"{args.output.name}:{label}",
                    stream,
                )
            (stage / "cells.json").write_text(
                json.dumps({"settings": settings, "cells": cells}, allow_nan=False) + "\n",
                encoding="utf-8",
            )
            (stage / "cells.json").replace(shift_dir / "cells.json")
            (stage / "episodes.jsonl").replace(shift_dir / "episodes.jsonl")
        shifted.to("cpu")
        cells_by_shift[shift] = cells
        write_summary(args.output, settings, rf_sums, shifts, cells_by_shift)
    write_summary(args.output, settings, rf_sums, shifts, cells_by_shift)


if __name__ == "__main__":
    main()
