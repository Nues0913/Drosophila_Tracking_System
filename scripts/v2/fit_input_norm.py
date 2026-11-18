"""Fit the fixed v2 input normalization from privileged-teacher episodes (plan 3.4, #137).

The teacher drives; the policy only senses, so the code statistics describe the states a good
tracker visits. Rows split evenly over the four scenarios and the two target motions, and every
row contributes its first 60 s episode. Each code column is averaged only over steps where its
channel is valid, as InputNorm normalizes only valid channels. Columns that are never valid
(the wind channels in v1compat) keep mean 0 and std 1 and are listed as unfitted.

Usage: python scripts/v2/fit_input_norm.py --episodes 100 \
           --output artifacts/input_norm_v2/input_norm.npz
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np
import torch
import yaml
from flytrack_v2.contracts import SensorContractV2
from flytrack_v2.model import ModelConfigV2, build_policy
from flytrack_v2.model.sensory import InputNorm
from flytrack_v2.world.teacher import PrivilegedTeacher
from flytrack_v2.world.v1compat import V1CompatWorld

ROOT = Path(__file__).resolve().parents[2]
SCENARIOS = ("normal", "occlusion", "noise", "conflict")
EPISODE_SEED_BASE = 1_900_000_000  # disjoint from training, v1 evaluation and the v1 audit


@torch.no_grad()
def accumulate(motion: str, rows: int, policy, contract, device, seed_offset: int):
    config = yaml.safe_load((ROOT / "configs/train/bc.yaml").read_text(encoding="utf-8"))
    scenario_configs = {
        name: yaml.safe_load((ROOT / path).read_text(encoding="utf-8"))
        for name, path in config["scenario_configs"].items()
    }
    names = [SCENARIOS[i % len(SCENARIOS)] for i in range(rows)]
    options = {"episode_seconds": 60, "target_motion": motion, "plume_model": "puff"}
    world = V1CompatWorld(
        contract,
        rows,
        scenarios=names,
        options=options,
        scenario_configs=scenario_configs,
        device=device,
    )
    teacher = PrivilegedTeacher(contract)
    seeds = torch.arange(rows, device=device) + EPISODE_SEED_BASE + seed_offset
    obs = world.reset(None, seeds)
    sensory = policy.initial_sensory_state(rows, device)
    teacher_state = teacher.initial_state(rows, device)
    layout = policy.input_norm.layout
    width = sum(layout.widths)
    index = policy.input_norm.validity_index.to(device)
    total = torch.zeros(width, dtype=torch.float64, device=device)
    squares = torch.zeros(width, dtype=torch.float64, device=device)
    count = torch.zeros(width, dtype=torch.float64, device=device)
    live = torch.ones(rows, dtype=torch.bool, device=device)
    samples = 0
    steps = int(np.ceil(60 / contract.control_dt))
    for _ in range(steps):
        code, sensory = policy.sense(obs, sensory)
        raw, flags = code.split((width, code.shape[1] - width), dim=1)
        valid = (flags > 0.5).index_select(1, index) & live[:, None]
        x = torch.where(valid, raw, 0.0).to(torch.float64)
        total += x.sum(0)
        squares += (x * x).sum(0)
        count += valid.sum(0)
        samples += int(live.sum())
        privileged = world.privileged()
        action, teacher_state = teacher.act(
            privileged, obs, privileged["agent_velocity"], teacher_state
        )
        result = world.step(action)
        live &= ~(result.terminated | result.truncated)
        obs = result.obs
    return total, squares, count, samples


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    device = torch.device(args.device)
    contract = SensorContractV2.from_json(ROOT / "configs/v2/sensor_contract.json")
    policy = build_policy("VO2", ModelConfigV2()).to(device).eval()
    parts = [
        accumulate(motion, args.episodes // 2, policy, contract, device, offset)
        for offset, motion in ((0, "curved"), (args.episodes // 2, "wander"))
    ]
    total = sum(p[0] for p in parts)
    squares = sum(p[1] for p in parts)
    count = sum(p[2] for p in parts)
    samples = sum(p[3] for p in parts)
    fitted = count > 0
    mean = torch.where(fitted, total / count.clamp_min(1), 0.0)
    var = torch.where(fitted, squares / count.clamp_min(1) - mean * mean, 1.0).clamp_min(0)
    norm = InputNorm(
        policy.input_norm.layout,
        policy.input_norm.std_floor,
        mean=mean.float().cpu(),
        std=var.sqrt().float().cpu(),
        fitted=True,
        n_samples=samples,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    norm.save(args.output)
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=True
    ).stdout.strip()
    report = {
        "artifact": str(args.output),
        "sha256": hashlib.sha256(args.output.read_bytes()).hexdigest(),
        "commit": commit,
        "world": "v1compat (hide_target)",
        "episodes": args.episodes,
        "samples": samples,
        "unfitted_columns": int((~fitted).sum()),
        "unfitted_column_index": torch.nonzero(~fitted).flatten().cpu().tolist(),
        "std_below_floor": int(((var.sqrt() < policy.input_norm.std_floor) & fitted).sum()),
        "note": "wind channels are never valid in v1compat; refit on both worlds after #133",
    }
    args.output.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "unfitted_column_index"}, indent=2))


if __name__ == "__main__":
    main()
