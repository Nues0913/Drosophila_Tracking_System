"""#138: reproduce the expert's development check of VO2 + ChemController (11-21).

PYTHONPATH=<worktree>/python python chem_dev.py ARTIFACT TRAIN_SEED OUT.json [DEVICE] [E0] [N]

Wind world, development episodes pilot_held_out base + 100000 * TRAIN_SEED + E0 .. E0+N-1
(default 800..807: outside the scored e = 0..99), the same numbers in both motions. Per motion:
the occlusion scenario with plain VO2, VO2 + controller, VO2 + hold-only controller, and the
controller under the occlusion-window odor swap and 90 degree wind rotation (E1); and every
other scenario with plain VO2 and VO2 + controller, with the controller's engagement.
"""

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

import flytrack_v2

ROOT = Path(flytrack_v2.__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("evaluate_v2", ROOT / "scripts/v2/evaluate_v2.py")
ev = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ev)
sys.path.insert(0, str(Path(__file__).resolve().parent))
from chem_controller import PARAMS, ChemController  # noqa: E402
from flytrack_v2.contracts import SensorContractV2  # noqa: E402
from flytrack_v2.eval.runner import (  # noqa: E402
    ObservationAblation,
    V2Actor,
    rotate_wind,
    run_episodes,
    swap_odor,
    to_records,
)

artifact, train_seed, out = Path(sys.argv[1]), int(sys.argv[2]), Path(sys.argv[3])
device = torch.device(sys.argv[4] if len(sys.argv) > 4 else "cpu")
E0 = int(sys.argv[5]) if len(sys.argv) > 5 else 800
N = int(sys.argv[6]) if len(sys.argv) > 6 else 8
protocol = yaml.safe_load((ROOT / "configs/v2/evaluation_protocol.yaml").read_text())
contract = SensorContractV2.from_json(ROOT / "configs/v2/sensor_contract.json")
scale = protocol["scenarios"]["noise"]["scale"]
window = ev.occlusion_window(protocol, ev.scenario_configs(scale, protocol)["occlusion"])
max_steps = int(np.ceil(protocol["episode_seconds"] / contract.control_dt)) + 5
base = protocol["episode_seeds"]["pilot_held_out"]
seeds = [base["base"] + base["per_train_seed"] * train_seed + E0 + i for i in range(N)]
policy, card, _ = ev.load_artifact(artifact)
policy = policy.to(device).eval()


def run(motion, scenario, arm):
    world, rows = ev.build_world(contract, "wind", motion, (scenario,), scale, protocol, N, device)
    actor = V2Actor(policy)
    ctrl = None
    if arm != "vo2":
        ctrl = actor = ChemController(actor, contract, use_chem=arm != "hold_only")
    if arm in ("odor_swap", "wind_rot90"):
        transform = swap_odor if arm == "odor_swap" else rotate_wind(90.0, contract.antenna_angle_deg)
        actor = ObservationAblation(actor, transform, window=window, control_dt=contract.control_dt)
    result = run_episodes(world, actor, seeds, max_steps=max_steps)
    records = to_records(result, rows, run_id=f"dev:{arm}", variant="VO2", train_seed=train_seed,
                         world="wind", motion=motion, artifact_sha256="", commit="")
    row = {
        "success": [int(r["success"]) for r in records],
        "collision": [int(r["collision"]) for r in records],
        "during": result.occlusion_phases["during_tracking_fraction"],
    }
    if ctrl is not None:
        s = {k: v.cpu().numpy() for k, v in ctrl.stats.items()}
        row["engaged_fraction"] = float(s["active"].sum() / s["steps"].sum())
        row["chem_fraction"] = float(s["chem"].sum() / s["steps"].sum())
        row["engaged_episodes"] = int((s["active"] > 0).sum())
    return row


report = {"artifact": str(artifact), "train_seed": train_seed, "episode_seeds": seeds,
          "params": PARAMS, "window_s": window, "results": {}}
for motion in ("curved", "wander"):
    res = {}
    for arm in ("vo2", "ctrl", "hold_only", "odor_swap", "wind_rot90"):
        res[f"occlusion/{arm}"] = run(motion, "occlusion", arm)
    for scenario in ("normal", "noise", "conflict"):
        for arm in ("vo2", "ctrl"):
            res[f"{scenario}/{arm}"] = run(motion, scenario, arm)
    ctl = res["occlusion/ctrl"]["during"]
    for arm in ("odor_swap", "wind_rot90"):
        d = [b - a for b, a in zip(ctl, res[f"occlusion/{arm}"]["during"]) if b == b and a == a]
        res[f"e1/{arm}"] = float(np.mean(d)) if d else None
    report["results"][motion] = res
    for key, row in res.items():
        if isinstance(row, dict):
            line = {"succ": f"{sum(row['success'])}/{N}", "coll": f"{sum(row['collision'])}/{N}",
                    "during": round(float(np.nanmean(row["during"])), 3)
                    if not all(v != v for v in row["during"]) else None}
            line.update({k: row[k] for k in ("engaged_fraction", "chem_fraction",
                                              "engaged_episodes") if k in row})
        else:
            line = row
        print(motion, key, line, flush=True)
out.write_text(json.dumps(report, indent=1) + "\n")
