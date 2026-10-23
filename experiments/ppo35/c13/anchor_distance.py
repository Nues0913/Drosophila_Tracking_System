"""Read CPU checkpoints and measure the configured trainable actor displacement."""

import math
from pathlib import Path

from c11.summarize_c11 import optional


def actor_distance(policy, baseline, train):
    import torch
    from flytrack_train.imitation import parameter_groups

    # Use configured rates, before critic warmup temporarily sets actor lr to zero.
    groups = parameter_groups(policy, {**train, "learning_rate": train.get("learning_rate", 3e-4)})
    included = {id(p) for group in groups if group["lr"] != 0 for p in group["params"]}
    reference = baseline.state_dict() if hasattr(baseline, "state_dict") else baseline
    squared = torch.zeros((), dtype=torch.float64, device="cpu")
    for name, parameter in policy.named_parameters():
        if not parameter.requires_grad or id(parameter) not in included:
            continue
        current = parameter.detach().to(device="cpu", dtype=torch.float64)
        original = reference[name].detach().to(device="cpu", dtype=torch.float64)
        if current.shape != original.shape:
            raise ValueError("BC actor parameter shape differs")
        squared += (current - original).square().sum(dtype=torch.float64)
    value = float(squared.sqrt())
    if not math.isfinite(value):
        raise ValueError("Nonfinite BC actor distance")
    return value


def nearest_update(records, step):
    if not records:
        return {"training_update_step": None, "bc_anchor_distance": None}
    row = min(records, key=lambda r: (abs(r["environment_steps"] - step), r["environment_steps"]))
    return {
        "training_update_step": row["environment_steps"],
        "bc_anchor_distance": row.get("bc_anchor_distance"),
    }


def distance_records(folder, config, bc_actor):
    import torch
    from flytrack_train.config import prepare_policy

    torch.set_num_threads(1)
    policy, _ = prepare_policy(config, torch.device("cpu"))
    baseline = torch.load(Path(bc_actor) / "weights.pt", map_location="cpu", weights_only=True)
    policy.load_state_dict(baseline, strict=True)
    metrics = optional(folder / "ppo_c11/training_metrics.json", [])
    result = [
        {
            "step": 0,
            "boundary": 0,
            "actor_distance_l2": actor_distance(policy, baseline, config["train"]),
            **nearest_update(metrics, 0),
        }
    ]
    index = optional(folder / "snapshots/index.json", {"snapshots": []})
    for row in sorted(index["snapshots"], key=lambda r: r["step"]):
        checkpoint = torch.load(row["path"], map_location="cpu", weights_only=True)
        if checkpoint["environment_steps"] != row["step"]:
            raise ValueError("Snapshot checkpoint step differs")
        policy.load_state_dict(checkpoint["actor"], strict=True)
        result.append(
            {
                "step": row["step"],
                "boundary": row["boundary"],
                "actor_distance_l2": actor_distance(policy, baseline, config["train"]),
                **nearest_update(metrics, row["step"]),
            }
        )
    return result
