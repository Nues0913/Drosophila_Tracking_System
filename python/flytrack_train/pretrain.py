"""Supervised unimodal decoding; disposable heads are never exported."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from .batches import Episode, pack_episode
from .config import (
    TRAIN_ENVIRONMENT_SEED_LIMIT,
    make_env,
    prepare_policy,
    save_actor,
    seed_all,
    write_json,
)
from .recovery import EpisodeStore, Recovery, atomic_torch_save, fit_sequence_epochs, restore_rng


def collect_pretraining(
    config,
    contract,
    policy,
    modality,
    episodes,
    rng,
    device,
    total_steps: int | None = None,
    initial_dataset=None,
    initial_steps=0,
    on_episode=None,
    on_progress=None,
):
    dataset: list[Episode] = list(initial_dataset or [])
    environment_steps = initial_steps
    env = make_env(config, contract)
    try:
        while (
            environment_steps < total_steps if total_steps is not None else len(dataset) < episodes
        ):
            observation, info = env.reset(seed=int(rng.integers(0, TRAIN_ENVIRONMENT_SEED_LIMIT)))
            observations, labels = [], []
            previous_bearing = None
            while True:
                bearing = float(info["target_bearing_rad"])
                angular_velocity = (
                    0.0
                    if previous_bearing is None
                    else np.arctan2(
                        np.sin(bearing - previous_bearing), np.cos(bearing - previous_bearing)
                    )
                    / contract.control_dt
                )
                present = (
                    bool(info["target_visible"])
                    if modality == "vision"
                    else bool(
                        np.max(observation["odor"])
                        > float(config.get("train", {}).get("odor_presence_threshold", 0.01))
                        and observation["valid"][2] > 0
                    )
                )
                masked = {
                    key: np.array(observation[key], copy=True)
                    for key in ("vision", "odor", "valid")
                }
                if modality == "vision":
                    masked["odor"][:] = 0
                    masked["valid"][2] = 0
                else:
                    masked["vision"][:] = 0
                    masked["valid"][:2] = 0
                observations.append(masked)
                labels.append(
                    {
                        "bearing": np.asarray([np.sin(bearing), np.cos(bearing)], dtype=np.float32),
                        "present": np.asarray(float(present), dtype=np.float32),
                        "angular_velocity": np.asarray(angular_velocity, dtype=np.float32),
                    }
                )
                # Teacher following and occasional exploration broaden the sequences.
                # Privileged state remains outside the actor input.
                action = np.asarray(info["teacher_action"], dtype=np.float32)
                if rng.random() < float(
                    config.get("train", {}).get("exploration_probability", 0.2)
                ):
                    action = np.asarray(
                        [
                            rng.uniform(0, contract.v_max),
                            rng.uniform(-contract.omega_max, contract.omega_max),
                        ],
                        dtype=np.float32,
                    )
                previous_bearing = bearing
                observation, _, terminated, truncated, info = env.step(action)
                environment_steps += 1
                if on_progress:
                    on_progress(environment_steps)
                if (
                    terminated
                    or truncated
                    or (total_steps is not None and environment_steps >= total_steps)
                ):
                    break
            dataset.append(pack_episode(observations, labels))
            if on_episode:
                on_episode(dataset[-1], len(dataset), environment_steps)
    finally:
        env.close()
    return dataset, environment_steps


def train_pretrain(config: dict[str, Any], output: Path, seed: int, device: torch.device) -> None:
    seed_all(seed)
    config = dict(config)
    config.setdefault("variant", "V")
    variant = config.get("variant", "V")
    modality = config.get("train", {}).get(
        "modality", "vision" if variant == "V" else "odor" if variant == "O" else None
    )
    if modality not in {"vision", "odor"}:
        raise ValueError("Pretraining requires train.modality: vision or odor (inferred for V/O)")
    policy, contract = prepare_policy(config, device)
    train = config.get("train", {})
    if (
        int(train.get("epochs", 20)) <= 0
        or int(train.get("sequence_length", 32)) <= 0
        or float(train.get("learning_rate", 1e-3)) <= 0
    ):
        raise ValueError("Pretraining epochs, sequence_length, and learning_rate must be positive")
    rng = np.random.default_rng(seed)
    count = int(train.get("episodes", 100))
    total_steps = int(train.get("total_steps", 120000))
    if count <= 0 or total_steps <= 0:
        raise ValueError("Pretraining collection budget must be positive")
    groups = policy.population_masks
    # Read out vision projection and convergence. Odor includes ORN projection.
    mask = (groups["visual"] if modality == "vision" else groups["olfactory"]) | groups[
        "convergence"
    ]
    if not bool(mask.any()):
        raise ValueError(
            f"Graph lacks {modality} projection/convergence nodes for auxiliary decoding"
        )
    output_count = 4 if modality == "vision" else 3
    decoder = nn.Linear(int(mask.sum()), output_count).to(device)
    parameters = list(policy.parameters()) + list(decoder.parameters())
    optimizer = torch.optim.Adam(parameters, lr=float(train.get("learning_rate", 1e-3)))
    recovery = Recovery(output, config, "pretrain", seed, device=device)
    store = EpisodeStore(output / "dataset")
    checkpoint = recovery.checkpoint
    environment_steps = checkpoint["environment_steps"] if checkpoint else 0
    dataset = store.load(checkpoint["dataset_count"]) if checkpoint else []
    if checkpoint:
        policy.load_state_dict(checkpoint["actor"])
        decoder.load_state_dict(checkpoint["decoder"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        restore_rng(checkpoint["rng"], rng)

    def save(phase, fit_state=None):
        recovery.save(
            rng,
            actor=policy.state_dict(),
            decoder=decoder.state_dict(),
            optimizer=optimizer.state_dict(),
            phase=phase,
            dataset_count=len(dataset),
            environment_steps=environment_steps,
            fit_state=fit_state,
        )

    def collected(episode, episode_count, steps):
        nonlocal environment_steps, dataset
        store.append(episode_count - 1, episode)
        environment_steps = steps
        # The collector owns its list; the outer list only tracks committed shards.
        dataset.append(episode)
        save("collect")
        recovery.progress(
            force=True,
            phase="collect",
            environment_steps=steps,
            total_steps=total_steps,
            episodes=episode_count,
        )

    if checkpoint is None or checkpoint["phase"] == "collect":
        dataset, environment_steps = collect_pretraining(
            config,
            contract,
            policy,
            modality,
            count,
            rng,
            device,
            total_steps,
            initial_dataset=dataset,
            initial_steps=environment_steps,
            on_episode=collected,
            on_progress=lambda steps: recovery.progress(
                phase="collect", environment_steps=steps, total_steps=total_steps
            ),
        )
        save("fit")

    def window_loss(episode, start, end, state):
        outputs, activities = [], []
        for step in range(start, end):
            obs = {
                key: torch.as_tensor(values[step : step + 1], device=device)
                for key, values in episode.observations.items()
            }
            _, state, activity = policy(obs, state)
            outputs.append(decoder(activity[:, mask])[0])
            activities.append(activity.square().mean())
        prediction = torch.stack(outputs)
        target = torch.as_tensor(episode.labels["bearing"][start:end], device=device)
        present = torch.as_tensor(episode.labels["present"][start:end], device=device)
        denom = present.sum().clamp_min(1)
        bearing_loss = ((prediction[:, :2] - target).square().sum(-1) * present).sum() / denom
        loss = bearing_loss + nn.functional.binary_cross_entropy_with_logits(
            prediction[:, 2], present
        )
        if modality == "vision":
            velocity = torch.as_tensor(episode.labels["angular_velocity"][start:end], device=device)
            loss = (
                loss
                + float(train.get("angular_velocity_coef", 0.1))
                * ((prediction[:, 3] - velocity).square() * present).sum()
                / denom
            )
        loss = loss + float(train.get("activity_coef", 1e-4)) * torch.stack(activities).mean()
        return loss, state

    reports = fit_sequence_epochs(
        policy,
        dataset,
        {"epochs": 20, **train},
        optimizer,
        parameters,
        rng,
        device,
        window_loss,
        lambda state: save("fit", state),
        lambda **values: recovery.progress(
            phase="fit", environment_steps=environment_steps, total_steps=total_steps, **values
        ),
        checkpoint.get("fit_state") if checkpoint and checkpoint["phase"] == "fit" else None,
    )
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "training_metrics.json", reports)
    # Keep decoder in the research checkpoint for diagnosis, not in the deployed actor artifact.
    atomic_torch_save(
        {
            "actor": policy.state_dict(),
            "decoder": decoder.state_dict(),
            "mask": mask.cpu(),
            "modality": modality,
            "config": config,
        },
        output / "pretraining_checkpoint.pt",
    )
    save_actor(
        output / "actor",
        policy,
        config,
        contract,
        stage="pretrain",
        modality=modality,
        seed=seed,
        environment_steps=environment_steps,
        algorithm="supervised_pretrain",
        auxiliary_decoder_exported=False,
    )
    recovery.progress(
        force=True, phase="complete", environment_steps=environment_steps, total_steps=total_steps
    )
