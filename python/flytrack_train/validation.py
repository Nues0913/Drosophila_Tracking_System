"""Held-out sensory decoding checks, distinct from fitting losses and actor export."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from flytrack_core.model import load_policy_artifact

from .batches import tensor_observation
from .config import make_env, project_source_fingerprint, write_json
from .recovery import Recovery


def validate_pretraining(
    config, output: Path, source: Path, checkpoint_path: Path, device: torch.device
):
    policy, contract, card = load_policy_artifact(source, device=str(device))
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    modality = checkpoint["modality"]
    if modality not in {"vision", "odor"}:
        raise ValueError("Auxiliary validation modality must be vision or odor")
    for name, value in policy.state_dict().items():
        if not torch.equal(value, checkpoint["actor"][name].to(device)):
            raise ValueError("Auxiliary decoder checkpoint and actor weights differ")
    mask = checkpoint["mask"].to(device=device, dtype=torch.bool)
    decoder = torch.nn.Linear(int(mask.sum()), 4 if modality == "vision" else 3).to(device)
    decoder.load_state_dict(checkpoint["decoder"])
    policy.eval()
    decoder.eval()
    validation = config.get("validation", {})
    episodes = int(validation.get("episodes", 20))
    seed_base = int(validation.get("environment_seed", 1_500_000_000))
    if (
        episodes <= 0
        or not 1_500_000_000 <= seed_base < 1_700_000_000
        or seed_base + episodes >= 1_700_000_000
    ):
        raise ValueError(
            "Held-out decoder validation requires positive episodes and seeds in [1.5e9,1.7e9)"
        )
    seed = int(config.get("seed", 0))
    seed_base += seed * 100_000
    if seed < 0 or seed_base + episodes >= 1_700_000_000:
        raise ValueError("Validation seed plus training-seed offset exceeds its held-out range")
    threshold = float(validation.get("bearing_mae_deg", 15.0 if modality == "vision" else 30.0))
    reports = []
    env = make_env(config, contract)
    progress = Recovery(
        output,
        {"validation": validation, "source": str(source)},
        "validate-pretrain",
        seed,
        device=device,
    )
    try:
        with torch.no_grad():
            for episode in range(episodes):
                observation, info = env.reset(seed=seed_base + episode)
                state = policy.initial_state(1, device)
                (
                    errors,
                    presence_correct,
                    presence_count,
                    true_positive,
                    false_positive,
                    false_negative,
                ) = [], 0, 0, 0, 0, 0
                while True:
                    masked = {
                        key: np.array(observation[key], copy=True)
                        for key in ("vision", "odor", "valid")
                    }
                    present = (
                        bool(info["target_visible"])
                        if modality == "vision"
                        else bool(
                            np.max(observation["odor"])
                            > float(config.get("train", {}).get("odor_presence_threshold", 0.01))
                            and observation["valid"][2] > 0
                        )
                    )
                    if modality == "vision":
                        masked["odor"][:] = 0
                        masked["valid"][2] = 0
                    else:
                        masked["vision"][:] = 0
                        masked["valid"][:2] = 0
                    _, state, activity = policy(tensor_observation([masked], device), state)
                    prediction = decoder(activity[:, mask])[0]
                    if not bool(torch.isfinite(prediction).all()):
                        raise FloatingPointError("Held-out decoder produced nonfinite values")
                    predicted_present = bool(prediction[2] >= 0)
                    presence_count += 1
                    presence_correct += predicted_present == present
                    true_positive += predicted_present and present
                    false_positive += predicted_present and not present
                    false_negative += not predicted_present and present
                    if present:
                        predicted_bearing = float(torch.atan2(prediction[0], prediction[1]))
                        difference = predicted_bearing - float(info["target_bearing_rad"])
                        errors.append(
                            abs(float(np.arctan2(np.sin(difference), np.cos(difference))))
                            * 180
                            / np.pi
                        )
                    observation, _, terminated, truncated, info = env.step(
                        np.asarray(info["teacher_action"], np.float32)
                    )
                    progress.progress(
                        phase="validation",
                        episode=episode + 1,
                        total_episodes=episodes,
                        steps=presence_count,
                    )
                    if terminated or truncated:
                        break
                reports.append(
                    {
                        "episode": episode,
                        "environment_seed": seed_base + episode,
                        "bearing_mae_deg": float(np.mean(errors)) if errors else None,
                        "bearing_error_sum_deg": float(np.sum(errors)),
                        "present_steps": len(errors),
                        "presence_correct": presence_correct,
                        "steps": presence_count,
                        "presence_tp": true_positive,
                        "presence_fp": false_positive,
                        "presence_fn": false_negative,
                    }
                )
                write_json(output / "episodes.json", reports)
    finally:
        env.close()
    count = sum(row["present_steps"] for row in reports)
    mae = sum(row["bearing_error_sum_deg"] for row in reports) / count if count else None
    summary = {
        "modality": modality,
        "episodes": episodes,
        "environment_seed_base": seed_base,
        "bearing_mae_deg": mae,
        "bearing_mae_threshold_deg": threshold,
        "present_steps": count,
        "presence_accuracy": sum(row["presence_correct"] for row in reports)
        / sum(row["steps"] for row in reports),
        "passed": mae is not None and mae <= threshold,
        "source_actor": str(source),
        "checkpoint": str(checkpoint_path),
        "actor_training_ledger": card.get("metadata", {}).get("stage_budget_ledger", []),
        "source_code": project_source_fingerprint(),
        "teacher_truth_enters_actor": False,
    }
    write_json(output / "metrics.json", summary)
    progress.progress(
        force=True,
        phase="complete",
        **{k: summary[k] for k in ("passed", "bearing_mae_deg", "presence_accuracy")},
    )
    return summary
