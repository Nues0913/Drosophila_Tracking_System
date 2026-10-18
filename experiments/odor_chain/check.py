"""Fail unless a finished stage has a complete, finite artifact and finite metrics.

Usage: check.py STAGE_DIR CONFIG.yaml
The artifact's odor_input_gain must equal the config's: warm starts keep the
init artifact's model card, so a mismatched init_artifact would go unnoticed.
"""

import json
import math
import sys
from pathlib import Path

import torch
from flytrack_core.model import load_policy_artifact
from flytrack_train.config import read_config


def finite(value, where):
    if isinstance(value, dict):
        for key, item in value.items():
            finite(item, f"{where}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            finite(item, f"{where}[{index}]")
    elif isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"Nonfinite value at {where}")


def tensors_finite(state, where):
    for key, value in state.items():
        if isinstance(value, dict):
            tensors_finite(value, f"{where}.{key}")
        elif torch.is_tensor(value) and value.is_floating_point():
            if not bool(torch.isfinite(value).all()):
                raise ValueError(f"Nonfinite tensor {where}.{key}")


def main():
    stage, config = Path(sys.argv[1]), read_config(sys.argv[2])
    progress = json.loads((stage / "progress.json").read_text())
    if progress.get("phase") != "complete":
        raise ValueError(f"{stage} is not complete: {progress.get('phase')}")
    metrics = json.loads((stage / "training_metrics.json").read_text())
    if not metrics:
        raise ValueError(f"{stage} has no training metrics")
    finite(metrics, "training_metrics")
    with torch.no_grad():
        policy, _, card = load_policy_artifact(stage / "actor", device="cpu")
        tensors_finite(dict(policy.state_dict()), "actor")
    expected = float(config.get("model", {}).get("odor_input_gain", 1.0))
    if float(policy.model_config["odor_input_gain"]) != expected:
        raise ValueError(
            f"{stage} odor_input_gain {policy.model_config['odor_input_gain']} != {expected}"
        )
    research = stage / "pretraining_checkpoint.pt"
    if research.exists():
        checkpoint = torch.load(research, map_location="cpu", weights_only=True)
        tensors_finite(checkpoint["decoder"], "decoder")
        for key, value in policy.state_dict().items():
            if not torch.equal(value, checkpoint["actor"][key]):
                raise ValueError(f"Research checkpoint and actor differ at {key}")
    print(
        json.dumps(
            {
                "stage": str(stage),
                "environment_steps": card["metadata"]["environment_steps"],
                "odor_input_gain": policy.model_config.get("odor_input_gain"),
            }
        )
    )


if __name__ == "__main__":
    main()
