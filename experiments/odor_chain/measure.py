"""Modality sensitivity and DNa01/DNa02 working point of one policy.

Usage: measure.py OUT.json ARTIFACT
       measure.py OUT.json --init CONFIG.yaml   (freshly built policy, config seed)
Uses the same 200 teacher-driven observations (seed 1.61e9) and sensory codes
as the diag/scale rounds, so ratios are directly comparable with them:
sensitivity from scripts/diagnostics/modality_sensitivity.py, DN voltage
medians and sigmoid slopes as in runs/odor_gain/scale/measure.py.
"""

import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent

from common import codes, modality, observations, replay  # noqa: E402
from flytrack_core.model import build_policy, load_policy_artifact  # noqa: E402
from flytrack_core.model.graph import Graph  # noqa: E402
from flytrack_train.config import contract_from_config, read_config  # noqa: E402

STEPS = 8 if os.environ.get("SMOKE", "0") == "1" else 200


@torch.no_grad()
def dn_summary(policy, graph):
    trace = replay(policy, codes(STEPS))
    policy.set_lesion("none")
    result = {}
    for kind in ("DNa01", "DNa02"):
        selected = (graph.type_names[graph.type_index] == kind) & policy.core.motor_mask.numpy()
        voltage = trace["voltage"][:, selected]
        slope = torch.sigmoid(torch.as_tensor(voltage) * policy.core.motor_rate_beta).numpy()
        result[kind] = {
            "n": int(selected.sum()),
            "voltage_median": float(np.median(voltage)),
            "slope_median": float(np.median(slope)),
        }
    return result


@torch.no_grad()
def main():
    torch.set_num_threads(1)
    out = Path(sys.argv[1])
    if sys.argv[2] == "--init":
        config = read_config(sys.argv[3])
        torch.manual_seed(int(config.get("seed", 0)))
        policy = build_policy(
            config["variant"],
            config["graph_path"],
            contract_from_config(config),
            frontend_path=config["frontend_path"],
            **config.get("model", {}),
        )
        graph = Graph.load(config["graph_path"])
        source = "init:" + sys.argv[3]
    else:
        policy, _, _ = load_policy_artifact(sys.argv[2], device="cpu")
        graph = Graph.load(Path(sys.argv[2]) / "graph.npz")
        source = sys.argv[2]
    policy.eval()
    sensitivity = modality.sensitivity(policy, observations(STEPS), torch.device("cpu"))
    report = {
        "source": source,
        "odor_input_gain": float(policy.model_config["odor_input_gain"]),
        "steps": STEPS,
        "seed": modality.DEFAULT_SEED,
        "sensitivity": sensitivity,
        "odor_to_vision_ratio": sensitivity["odor_to_vision_ratio"],
        "no_odor": sensitivity["no_odor"]["max_abs_delta_omega"],
        "odor_swap_lr": sensitivity["odor_swap_lr"]["max_abs_delta_omega"],
        "no_vision": sensitivity["no_vision"]["max_abs_delta_omega"],
        "dn": dn_summary(policy, graph),
        "core_bias_max_abs": float(policy.core.bias.abs().max()),
    }
    for key in ("odor_to_vision_ratio", "no_odor", "odor_swap_lr", "no_vision"):
        if not math.isfinite(report[key]):
            raise ValueError(f"Nonfinite {key}")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=1, allow_nan=False) + "\n")
    print(
        source,
        f"ratio {report['odor_to_vision_ratio']:.3g}",
        {kind: round(row["voltage_median"], 3) for kind, row in report["dn"].items()},
        flush=True,
    )


if __name__ == "__main__":
    main()
