"""Supervised unimodal decoding; disposable heads are never exported."""

from __future__ import annotations

from functools import partial
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from .batches import SequenceModel, pack_episode
from .collection import PretrainDatasetCache, collect_lockstep, pretrain_stream_identity
from .config import (
    make_env,
    prepare_policy,
    save_actor,
    seed_all,
    write_json,
)
from .parallel import EnvPool, default_workers
from .recovery import Recovery, atomic_torch_save, fit_sequence_epochs, restore_rng

PRETRAIN_STREAMS = {"vision": 1, "odor": 2}


def pretraining_actor(config, contract, modality):
    """Teacher following and occasional exploration broaden the sequences.

    Privileged state remains outside the actor input; labels come from ``info``.
    """
    train = config.get("train", {})
    exploration = float(train.get("exploration_probability", 0.2))
    threshold = float(train.get("odor_presence_threshold", 0.01))

    def act(running, observations, infos):
        actions = {}
        for slot, entry in running.items():
            observation, info = observations[slot], infos[slot]
            bearing = float(info["target_bearing_rad"])
            previous = entry.extra.get("previous_bearing")
            angular_velocity = (
                0.0
                if previous is None
                else np.arctan2(np.sin(bearing - previous), np.cos(bearing - previous))
                / contract.control_dt
            )
            present = (
                bool(info["target_visible"])
                if modality == "vision"
                else bool(np.max(observation["odor"]) > threshold and observation["valid"][2] > 0)
            )
            masked = {
                key: np.array(observation[key], copy=True) for key in ("vision", "odor", "valid")
            }
            if modality == "vision":
                masked["odor"][:] = 0
                masked["valid"][2] = 0
            else:
                masked["vision"][:] = 0
                masked["valid"][:2] = 0
            entry.observations.append(masked)
            entry.labels.append(
                {
                    "bearing": np.asarray([np.sin(bearing), np.cos(bearing)], dtype=np.float32),
                    "present": np.asarray(float(present), dtype=np.float32),
                    "angular_velocity": np.asarray(angular_velocity, dtype=np.float32),
                }
            )
            action = np.asarray(info["teacher_action"], dtype=np.float32)
            if entry.rng.random() < exploration:
                action = np.asarray(
                    [
                        entry.rng.uniform(0, contract.v_max),
                        entry.rng.uniform(-contract.omega_max, contract.omega_max),
                    ],
                    dtype=np.float32,
                )
            entry.extra["previous_bearing"] = bearing
            actions[slot] = action
        return actions

    return act


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
    # Pretraining episodes do not depend on the actor: one stream per seed,
    # modality and world is shared by all variants when dataset_cache is set.
    cache = PretrainDatasetCache(
        Path(config["dataset_cache"]) / "pretrain" if config.get("dataset_cache") else output,
        pretrain_stream_identity(config, modality, seed),
    )
    checkpoint = recovery.checkpoint
    environment_steps = checkpoint["environment_steps"] if checkpoint else 0
    if checkpoint:
        policy.load_state_dict(checkpoint["actor"])
        decoder.load_state_dict(checkpoint["decoder"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        restore_rng(checkpoint["rng"], rng)
    dataset_count = checkpoint["dataset_count"] if checkpoint else 0

    def save(phase, fit_state=None):
        recovery.save(
            rng,
            actor=policy.state_dict(),
            decoder=decoder.state_dict(),
            optimizer=optimizer.state_dict(),
            phase=phase,
            dataset_count=dataset_count,
            environment_steps=environment_steps,
            fit_state=fit_state,
        )

    if checkpoint is None or checkpoint["phase"] == "collect":
        with cache.locked():
            lengths = cache.lengths()
            available = sum(lengths)
            if available < total_steps:
                pool = EnvPool(
                    [partial(make_env, config, contract) for _ in range(max(1, default_workers()))]
                )

                def collected(index, episode, steps):
                    nonlocal environment_steps, dataset_count
                    cache.append(index, episode, lengths)
                    environment_steps, dataset_count = min(steps, total_steps), index + 1
                    save("collect")
                    recovery.progress(
                        force=True,
                        phase="collect",
                        environment_steps=environment_steps,
                        total_steps=total_steps,
                        episodes=dataset_count,
                    )

                try:
                    collect_lockstep(
                        pool,
                        seed=seed,
                        stream=PRETRAIN_STREAMS[modality],
                        first_index=len(lengths),
                        committed_steps=available,
                        step_budget=total_steps,
                        act=pretraining_actor(config, contract, modality),
                        finish=lambda entry: pack_episode(entry.observations, entry.labels),
                        on_episode=collected,
                        on_progress=lambda steps: recovery.progress(
                            phase="collect", environment_steps=steps, total_steps=total_steps
                        ),
                        truncate=False,
                    )
                finally:
                    pool.close()
        dataset, environment_steps = cache.load_prefix(total_steps)
        dataset_count = len(dataset)
        write_json(
            output / "dataset.json",
            {
                "cache": str(cache.directory),
                "stream_sha256": cache.key,
                "episodes": dataset_count,
                "environment_steps": environment_steps,
            },
        )
        save("fit")
    else:
        dataset, environment_steps = cache.load_prefix(total_steps)
        if len(dataset) != dataset_count:
            raise ValueError("Recovery dataset count differs from the cached episode stream")

    model = SequenceModel(policy, device)
    inputs = model.encode(dataset)
    labels = [
        {key: torch.as_tensor(value, device=device) for key, value in episode.labels.items()}
        for episode in dataset
    ]
    # Raw camera frames are no longer needed once encoded (about 1.4 GB per 240k steps).
    del dataset
    velocity_coef = float(train.get("angular_velocity_coef", 0.1))
    activity_coef = float(train.get("activity_coef", 1e-4))

    def window_loss(outputs, targets, step_mask):
        # Per lane, exactly the single-window loss: bearing/velocity averaged
        # over present steps, presence and activity over the lane's steps.
        activity = outputs["activity"]
        prediction = decoder(activity[..., mask])
        present = targets["present"] * step_mask
        denom = present.sum(0).clamp_min(1)
        steps = step_mask.sum(0).clamp_min(1)
        bearing_loss = ((prediction[..., :2] - targets["bearing"]).square().sum(-1) * present).sum(
            0
        ) / denom
        presence_loss = (
            nn.functional.binary_cross_entropy_with_logits(
                prediction[..., 2], targets["present"], reduction="none"
            )
            * step_mask
        ).sum(0) / steps
        loss = bearing_loss + presence_loss
        if modality == "vision":
            loss = (
                loss
                + velocity_coef
                * ((prediction[..., 3] - targets["angular_velocity"]).square() * present).sum(0)
                / denom
            )
        activity_loss = (activity.square().mean(-1) * step_mask).sum(0) / steps
        return loss + activity_coef * activity_loss

    reports = fit_sequence_epochs(
        model,
        inputs,
        labels,
        {"epochs": 20, **train},
        optimizer,
        parameters,
        rng,
        window_loss,
        lambda state: save("fit", state),
        lambda force=False, phase="fit", **values: recovery.progress(
            force=force,
            phase=phase,
            environment_steps=environment_steps,
            total_steps=total_steps,
            **values,
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
