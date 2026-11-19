"""Fit the fixed v2 input normalization from teacher episodes (plan 3.4, #137).

DAgger's teacher drives (OcclusionTeacher: the privileged teacher outside occlusion, the
observation-only rule during it, with the arena's walls; #161 D4); the policy only senses, so
the code statistics describe the states a good tracker visits. Rows split evenly over the two
worlds (v1compat, and wind at the protocol's U = 0.5 m/s, #169), the two target motions and the
four scenarios, and every row contributes its first 60 s episode. Each code column is averaged
only over steps where its channel is valid (flytrack_v2.model.sensory.fit_input_norm, the
model's own fit), so the wind columns come from the wind world's rows only. A column never
valid in any row would keep mean 0 and std 1 and is listed as unfitted.

Usage: python scripts/v2/fit_input_norm.py --episodes 200 \
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
from flytrack_v2.model.sensory import fit_input_norm
from flytrack_v2.world.teacher import OcclusionTeacher, PrivilegedTeacher
from flytrack_v2.world.v1compat import V1CompatWorld
from flytrack_v2.world.wind import WindConfig
from torch import Tensor

ROOT = Path(__file__).resolve().parents[2]
SCENARIOS = ("normal", "occlusion", "noise", "conflict")
WIND_SPEED_M_S = 0.5  # configs/v2/evaluation_protocol.yaml tuning.wind_speed_m_s (#169)
EPISODE_SEED_BASE = 1_900_000_000  # disjoint from training, v1 evaluation and the v1 audit


@torch.no_grad()
def collect(
    kind: str, motion: str, rows: int, policy, contract, device, seed_offset: int
) -> Tensor:
    """Sensory codes [T, code_size] of every step of each row's first episode."""
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
        wind=WindConfig(speed_m_s=WIND_SPEED_M_S) if kind == "wind" else None,
    )
    teacher = OcclusionTeacher(PrivilegedTeacher(contract), walls=world.walls)
    seeds = torch.arange(rows, device=device) + EPISODE_SEED_BASE + seed_offset
    obs = world.reset(None, seeds)
    sensory = policy.initial_sensory_state(rows, device)
    teacher_state = teacher.initial_state(rows, device)
    live = torch.ones(rows, dtype=torch.bool, device=device)
    codes = []
    for _ in range(int(np.ceil(60 / contract.control_dt))):
        code, sensory = policy.sense(obs, sensory)
        codes.append(code[live].cpu())
        privileged = world.privileged()
        action, teacher_state = teacher.act(
            privileged, obs, privileged["agent_velocity"], teacher_state
        )
        result = world.step(action)
        live &= ~(result.terminated | result.truncated)
        obs = result.obs
    return torch.cat(codes)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--episodes", type=int, default=200, help="split over 2 worlds x 2 motions")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    device = torch.device(args.device)
    contract = SensorContractV2.from_json(ROOT / "configs/v2/sensor_contract.json")
    policy = build_policy("VO2", ModelConfigV2()).to(device).eval()
    rows = args.episodes // 4
    parts = [(kind, motion) for kind in ("v1compat", "wind") for motion in ("curved", "wander")]
    codes = torch.cat(
        [
            collect(kind, motion, rows, policy, contract, device, i * rows)
            for i, (kind, motion) in enumerate(parts)
        ]
    )
    # The model's own fit (#137 review): each channel's valid rows only; a channel that is
    # never valid keeps mean 0 and std 1.
    norm = fit_input_norm(codes, policy.input_norm.layout, policy.input_norm.std_floor)
    width = sum(norm.layout.widths)
    flags = codes[:, width:]
    valid = (flags > 0.5).index_select(1, norm.validity_index)
    fitted = valid.any(0)
    samples = len(codes)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    norm.save(args.output)
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=True
    ).stdout.strip()
    report = {
        "artifact": str(args.output),
        "sha256": hashlib.sha256(args.output.read_bytes()).hexdigest(),
        "commit": commit,
        "worlds": f"v1compat and wind (U = {WIND_SPEED_M_S} m/s), hide_target, half each",
        "teacher": "OcclusionTeacher (walls on)",
        "episodes": rows * len(parts),
        "samples": samples,
        "unfitted_columns": int((~fitted).sum()),
        "unfitted_column_index": torch.nonzero(~fitted).flatten().cpu().tolist(),
        "std_below_floor": int(((norm.std < norm.std_floor) & fitted).sum()),
    }
    args.output.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "unfitted_column_index"}, indent=2))


if __name__ == "__main__":
    main()
