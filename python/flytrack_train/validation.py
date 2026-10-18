"""Held-out sensory decoding checks, distinct from fitting losses and actor export."""

from __future__ import annotations

from functools import partial
from pathlib import Path

import numpy as np
import torch
from flytrack_core.model import load_policy_artifact

from .batches import SequenceModel, tensor_observation
from .config import make_env, project_source_fingerprint, write_json
from .parallel import EnvPool
from .pretrain import PopulationRMSDecoder
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
    state = checkpoint["decoder"]
    outputs = 4 if modality == "vision" else 3
    if "input_rms" in state:
        # decoder_input_scale: population_rms keeps its frozen scales as buffers.
        if not bool(state["input_initialized"]):
            raise ValueError("Population-RMS decoder checkpoint has uninitialized input scales")
        decoder = PopulationRMSDecoder(int(mask.sum()), outputs, state["input_convergence"])
    else:
        decoder = torch.nn.Linear(int(mask.sum()), outputs)
    decoder = decoder.to(device)
    decoder.load_state_dict(state)
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
    progress = Recovery(
        output,
        {"validation": validation, "source": str(source)},
        "validate-pretrain",
        seed,
        device=device,
    )
    odor_threshold = float(config.get("train", {}).get("odor_presence_threshold", 0.01))
    lanes = min(episodes, int(validation.get("batch_episodes", 32)))
    model = SequenceModel(policy, device)
    queue = list(range(episodes))
    running: dict[int, dict] = {}
    observations, infos, finished = {}, {}, {}
    sensory, central = model.initial(lanes)
    initial_sensory, initial_central = model.initial(1)
    pool = EnvPool([partial(make_env, config, contract) for _ in range(lanes)])
    try:
        with torch.no_grad(), model.cache():
            while queue or running:
                for slot in range(lanes):
                    if slot in running or not queue:
                        continue
                    episode = queue.pop(0)
                    observations[slot], infos[slot] = pool.reset(slot, seed_base + episode)
                    if sensory is not None:
                        sensory[slot] = initial_sensory[0]
                    central[slot] = initial_central[0]
                    running[slot] = {
                        "episode": episode,
                        "errors": [],
                        "presence_correct": 0,
                        "presence_count": 0,
                        "true_positive": 0,
                        "false_positive": 0,
                        "false_negative": 0,
                    }
                active = sorted(running)
                masked_batch, presence = [], []
                for slot in active:
                    observation, info = observations[slot], infos[slot]
                    masked = {
                        key: np.array(observation[key], copy=True)
                        for key in ("vision", "odor", "valid")
                    }
                    presence.append(
                        bool(info["target_visible"])
                        if modality == "vision"
                        else bool(
                            np.max(observation["odor"]) > odor_threshold
                            and observation["valid"][2] > 0
                        )
                    )
                    if modality == "vision":
                        masked["odor"][:] = 0
                        masked["valid"][2] = 0
                    else:
                        masked["vision"][:] = 0
                        masked["valid"][:2] = 0
                    masked_batch.append(masked)
                index = torch.tensor(active, device=device)
                inputs, new_sensory = model.observe(
                    tensor_observation(masked_batch, device),
                    None if sensory is None else sensory[index],
                )
                _, new_central, activity = model.step(inputs, central[index])
                if sensory is not None:
                    sensory[index] = new_sensory
                central[index] = new_central
                predictions = decoder(activity[:, mask])
                if not bool(torch.isfinite(predictions).all()):
                    raise FloatingPointError("Held-out decoder produced nonfinite values")
                predictions = predictions.cpu().numpy()
                actions = {}
                for row, slot in enumerate(active):
                    entry, info, present = running[slot], infos[slot], presence[row]
                    prediction = predictions[row]
                    predicted_present = bool(prediction[2] >= 0)
                    entry["presence_count"] += 1
                    entry["presence_correct"] += predicted_present == present
                    entry["true_positive"] += predicted_present and present
                    entry["false_positive"] += predicted_present and not present
                    entry["false_negative"] += not predicted_present and present
                    if present:
                        predicted_bearing = float(np.arctan2(prediction[0], prediction[1]))
                        difference = predicted_bearing - float(info["target_bearing_rad"])
                        entry["errors"].append(
                            abs(float(np.arctan2(np.sin(difference), np.cos(difference))))
                            * 180
                            / np.pi
                        )
                    actions[slot] = np.asarray(info["teacher_action"], np.float32)
                for slot, (observation, _, terminated, truncated, info) in pool.step(
                    actions
                ).items():
                    observations[slot], infos[slot] = observation, info
                    if terminated or truncated:
                        entry = running.pop(slot)
                        finished[entry["episode"]] = entry
                progress.progress(
                    phase="validation", episodes_done=len(finished), total_episodes=episodes
                )
    finally:
        pool.close()
    reports = []
    for episode in range(episodes):
        entry = finished[episode]
        errors = entry["errors"]
        reports.append(
            {
                "episode": episode,
                "environment_seed": seed_base + episode,
                "bearing_mae_deg": float(np.mean(errors)) if errors else None,
                "bearing_error_sum_deg": float(np.sum(errors)),
                "present_steps": len(errors),
                "presence_correct": entry["presence_correct"],
                "steps": entry["presence_count"],
                "presence_tp": entry["true_positive"],
                "presence_fp": entry["false_positive"],
                "presence_fn": entry["false_negative"],
            }
        )
    write_json(output / "episodes.json", reports)
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
