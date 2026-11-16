"""Describe an S4 run beyond the registered judgment (PM, #35; descriptive only). Run in flytrack-ml:0.1 from the
worktree root (PYTHONPATH=python), after run_s4.sh:  python explore/s4/analyze.py OUT_JSON [BC_ROOT]
(BC_ROOT defaults to runs/s4/bc; seeds are the seed_* directories with an actor).
Per seed: each round's forward-row calibration (rows above the std floor, the recalibration's bias shift and
weight ratios), each round's last-epoch gradient balance (scale, how often it hit max_scale, primary and secondary
norms), and the trained actor's forward-row weights: the 10 rows with the largest |w| / std contribution
(|w| * raw std * scale, i.e. the latent spread each row adds), and the DNp09 pool weights.
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

from flytrack_core.model import load_policy_artifact

torch.set_num_threads(1)
root = Path(sys.argv[2] if len(sys.argv) > 2 else "runs/s4/bc") / "VO-late"
report = {"rule": __doc__, "seeds": {}}
for base in sorted(p.parent.parent for p in root.glob("seed_*/actor/model_card.json")):
    seed = int(base.name.split("_")[1])
    rounds = []
    for path in sorted(base.glob("forward_row_calibration_round_*.json")):
        c = json.loads(path.read_text())
        rounds.append({"file": path.name, "rows_above_floor": len(c["raw_std"]) - c["floored_rows"],
                       "teacher_v_std": c["teacher_v_std"], "recalibration": c["recalibration"],
                       "closed_form_held_out_corr": {k: v["held_out_corr"] for k, v in c["closed_form"].items()}})
    metrics = json.loads((base / "training_metrics.json").read_text())
    balance = []
    for r in metrics:
        last = [e for e in r["epochs"] if isinstance(e, dict) and "balance" in e]
        if last:
            balance.append({"round": r["round"], "epochs": len(r["epochs"]), **last[-1]["balance"]})
    policy, _, _ = load_policy_artifact(str(base / "actor"), device="cpu")
    readout = policy.readout
    types = readout.forward_row_types
    names = [f"b{b}:{t}" for b in range(policy.branches) for t in types]
    w = readout.forward_row_weight.detach().numpy()
    scale = readout.forward_row_scale.numpy()
    last_std = np.asarray(json.loads(sorted(base.glob("forward_row_calibration_round_*.json"))[-1].read_text())["raw_std"])
    spread = np.abs(w) * last_std * scale
    top = np.argsort(-spread)[:10]
    report["seeds"][seed] = {
        "rounds": rounds, "balance": balance,
        "forward_weight_dnp09": readout.forward_weight.detach().tolist(), "forward_bias": float(readout.forward_bias.detach()),
        "row_weight_norm": float(np.linalg.norm(w)),
        "top_rows": [{"row": names[j], "weight": float(w[j]), "latent_sd": float(spread[j])} for j in top],
    }
    print(seed, json.dumps(report["seeds"][seed]["top_rows"][:5]), flush=True)
Path(sys.argv[1]).write_text(json.dumps(report, indent=1))
