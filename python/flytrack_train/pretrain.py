"""Supervised unimodal decoding; disposable heads are never exported."""

from __future__ import annotations

from functools import partial
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from .batches import LaneSchedule, SequenceModel, pack_episode
from .collection import PretrainDatasetCache, collect_lockstep, pretrain_stream_identity
from .config import (
    make_env,
    prepare_policy,
    save_actor,
    seed_all,
    write_json,
)
from .imitation import parameter_groups
from .losses import activity_penalty
from .parallel import EnvPool, default_workers
from .recovery import (
    Recovery,
    _window_batch,
    atomic_torch_save,
    fit_sequence_epochs,
    held_out_episodes,
    restore_rng,
)

PRETRAIN_STREAMS = {"vision": 1, "odor": 2}


class PopulationRMSDecoder(nn.Linear):
    """Disposable head with one frozen RMS per disjoint input population.

    Weights and bias start at zero: unit-RMS inputs are not mean-centered, so a
    random head began at losses of 9-15. Zero predicts sin/cos 0 and presence
    logit 0 (about 1 + ln 2). The default random draw is still consumed.
    """

    def __init__(self, in_features: int, out_features: int, convergence: torch.Tensor):
        super().__init__(in_features, out_features)
        with torch.no_grad():
            self.weight.zero_()
            self.bias.zero_()
        self.register_buffer("input_convergence", convergence.bool().clone())
        self.register_buffer("input_rms", torch.ones(2))
        self.register_buffer("input_initialized", torch.tensor(False))

    def forward(self, activity):
        return super().forward(activity / self.input_rms[self.input_convergence.long()])


@torch.no_grad()
def initialize_decoder_inputs(decoder, model, inputs, mask, train):
    """Use the first batch of complete encoded episodes before any updates.

    Held-out episodes and padded steps never enter the statistics. No random
    draws are consumed, and initialized recovery buffers are used unchanged.
    """
    if bool(decoder.input_initialized):
        return
    lengths = [len(next(iter(item.values()))) for item in inputs]
    held = set(held_out_episodes(len(inputs), int(train.get("validation_every", 10))))
    fitted = [index for index in range(len(inputs)) if index not in held]
    lanes = int(train.get("batch_episodes", 16))
    span = max(lengths[index] for index in fitted[:lanes])
    schedule = LaneSchedule(lengths, fitted, lanes, span)
    _, central = model.initial(lanes)
    batch, _, step_mask, central = _window_batch(
        model, schedule.next(), inputs, [{} for _ in inputs], central
    )
    # Graph population labels overlap: convergence belongs to its own group,
    # and the modality group contains the remaining selected nodes.
    populations = (~decoder.input_convergence, decoder.input_convergence)
    squared_sum = torch.zeros(2, device=model.device, dtype=torch.float64)
    units = torch.stack([population.sum() for population in populations])
    count = 0
    with model.cache():
        for step in range(len(step_mask)):
            _, updated, activity = model.step(
                {key: values[step] for key, values in batch.items()}, central
            )
            valid = step_mask[step].bool()
            central = torch.where(valid.unsqueeze(-1), updated, central)
            values = activity[valid][..., mask].double()
            squared_sum += torch.stack(
                [values[:, population].square().sum() for population in populations]
            )
            count += len(values)
    # An absent population has no decoder inputs; give its unused scalar the floor.
    rms = (squared_sum / (count * units).clamp_min(1)).sqrt().clamp_min(1e-3)
    if not bool(torch.isfinite(rms).all()):
        raise ValueError("Decoder input RMS must be finite")
    decoder.input_rms.copy_(rms)
    decoder.input_initialized.fill_(True)


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
    if "decoder_input_standardize" in train:
        raise ValueError("decoder_input_standardize was removed; use decoder_input_scale")
    input_scale = train.get("decoder_input_scale", "none")
    if not isinstance(input_scale, str) or input_scale not in {"none", "population_rms"}:
        raise ValueError("decoder_input_scale must be none or population_rms")
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
    decoder = (
        PopulationRMSDecoder(int(mask.sum()), output_count, groups["convergence"][mask])
        if input_scale == "population_rms"
        else nn.Linear(int(mask.sum()), output_count)
    ).to(device)
    parameters = list(policy.parameters()) + list(decoder.parameters())
    # Same groups as BC (one rate unless overridden). 38 vision updates moved the
    # neuron-type biases by up to 0.04 and odor's effect on omega fell ~2000x (#4);
    # restoring the biases brought it back, but freezing them also slowed vision
    # learning. core_bias_learning_rate: 0 freezes them for such comparisons.
    optimizer = torch.optim.Adam(
        [
            *parameter_groups(policy, train),
            {"params": list(decoder.parameters()), "lr": float(train.get("learning_rate", 1e-3))},
        ]
    )
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
    if input_scale == "population_rms":
        initialize_decoder_inputs(decoder, model, inputs, mask, train)
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
        activity_loss = (activity_penalty(policy, activity) * step_mask).sum(0) / steps
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
