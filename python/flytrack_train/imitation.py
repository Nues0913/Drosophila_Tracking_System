"""Teacher-only BC collection followed by on-policy DAgger aggregation."""

from __future__ import annotations

from functools import partial
from pathlib import Path
from typing import Any

import numpy as np
import torch
from flytrack_core.model.policy import DEFAULT_CONFIG, ConnectomePolicy

from .batches import Episode, SequenceModel, pack_episode, tensor_observation
from .collection import collect_lockstep
from .config import (
    make_env,
    prepare_policy,
    save_actor,
    seed_all,
    write_json,
)
from .extra_turn import EXTRA_TURN_KEYS, calibrate_extra_turns
from .forward_rows import FORWARD_ROW_KEYS, calibrate_forward_rows
from .losses import activity_penalty
from .parallel import EnvPool, default_workers
from .parameters import parameter_groups
from .recovery import (
    EpisodeStore,
    Recovery,
    fit_sequence_epochs,
    held_out_episodes,
    restore_rng,
)
from .turn_bias import center_turn_bias

BC_STREAM = 10


def dagger_actor(model: SequenceModel, contract, teacher_probability: float):
    """Teacher labels every step; the acting policy mixes teacher and student.

    The student runs batched over all slots with its own recurrent state per
    slot. It always advances (as in deployment), whoever's action is applied.
    """

    def start(slot, entry):
        if teacher_probability < 1:
            entry.extra["sensory"], entry.extra["central"] = model.initial(1)

    def act(running, observations, infos):
        slots = list(running)
        for slot in slots:
            entry = running[slot]
            entry.observations.append(
                {
                    key: np.array(observations[slot][key], copy=True)
                    for key in ("vision", "odor", "valid")
                }
            )
            entry.labels.append(
                {"action": np.asarray(infos[slot]["teacher_action"], dtype=np.float32).copy()}
            )
        actions = {
            slot: np.asarray(infos[slot]["teacher_action"], dtype=np.float32) for slot in slots
        }
        if teacher_probability >= 1:
            return actions
        with torch.no_grad(), model.cache():
            obs = tensor_observation([observations[slot] for slot in slots], model.device)
            sensory = (
                torch.cat([running[slot].extra["sensory"] for slot in slots])
                if model.split
                else None
            )
            central = torch.cat([running[slot].extra["central"] for slot in slots])
            inputs, sensory = model.observe(obs, sensory)
            student, central, _ = model.step(inputs, central)
        student = student.cpu().numpy()
        for row, slot in enumerate(slots):
            entry = running[slot]
            if model.split:
                entry.extra["sensory"] = sensory[row : row + 1]
            entry.extra["central"] = central[row : row + 1]
            if entry.rng.random() >= teacher_probability:
                actions[slot] = student[row]
        return actions

    return start, act


def fit_behavior(
    model: SequenceModel,
    inputs,
    episodes: list[Episode],
    train: dict[str, Any],
    contract,
    optimizer,
    rng,
    fit_state=None,
    on_checkpoint=None,
    on_progress=None,
):
    if not episodes:
        raise ValueError("Behavior cloning requires collected episodes")
    device = model.device
    scale = torch.tensor([contract.v_max, contract.omega_max], device=device)
    labels = [{"action": torch.as_tensor(e.labels["action"], device=device)} for e in episodes]
    activity_coef = float(train.get("activity_coef", 1e-4))

    def window_loss(outputs, targets, step_mask):
        steps = step_mask.sum(0).clamp_min(1)
        error = ((outputs["action"] - targets["action"]) / scale).square().mean(-1)
        activity = activity_penalty(model.policy, outputs["activity"])
        return ((error + activity_coef * activity) * step_mask).sum(0) / steps

    balance = None
    settings = train.get("speed_gradient_balance")
    if settings:
        # #35 S2 (exploration): the speed term's gradient on every non-readout parameter is rescaled
        # to target_ratio x the turn (+ activity) term's; the readout keeps the plain gradient.
        def split(outputs, targets, step_mask):
            steps = step_mask.sum(0).clamp_min(1)
            squared = ((outputs["action"] - targets["action"]) / scale).square()
            activity = activity_penalty(model.policy, outputs["activity"])
            speed = (0.5 * squared[..., 0] * step_mask).sum(0) / steps
            turn = ((0.5 * squared[..., 1] + activity_coef * activity) * step_mask).sum(0) / steps
            return turn, speed

        # Only parameters the optimizer actually updates (lr > 0): frozen groups such as the BC
        # core bias keep their plain gradient, so they do not inflate the clipped norm.
        learning_rate = {
            id(parameter): float(group["lr"])
            for group in optimizer.param_groups
            for parameter in group["params"]
        }
        upstream = [
            parameter
            for name, parameter in model.policy.named_parameters()
            if not name.startswith("readout.")
            and parameter.requires_grad
            and learning_rate.get(id(parameter), 0.0) > 0
        ]
        if not upstream:
            raise ValueError("speed_gradient_balance found no trainable non-readout parameters")
        balance = {
            "split": split,
            "upstream": upstream,
            "target_ratio": float(settings["target_ratio"]),
            "max_scale": float(settings.get("max_scale", 1e4)),
        }

    return fit_sequence_epochs(
        model,
        inputs,
        labels,
        train,
        optimizer,
        list(model.policy.parameters()),
        rng,
        window_loss,
        on_checkpoint or (lambda state: None),
        on_progress or (lambda **values: None),
        fit_state,
        balance,
    )


def train_bc(config: dict[str, Any], output: Path, seed: int, device: torch.device) -> None:
    seed_all(seed)
    policy, contract = prepare_policy(config, device)
    train = config.get("train", {})
    if (
        int(train.get("epochs", 10)) <= 0
        or int(train.get("sequence_length", 32)) <= 0
        or float(train.get("learning_rate", 1e-3)) <= 0
    ):
        raise ValueError("BC epochs, sequence_length, and learning_rate must be positive")
    if not 0 <= float(train.get("dagger_beta", 0.5)) <= 1:
        raise ValueError("DAgger beta must be between zero and one")
    rng = np.random.default_rng(seed)
    rounds = int(train.get("dagger_rounds", 3))
    initial_episodes = int(train.get("teacher_episodes", 50))
    dagger_episodes = int(train.get("dagger_episodes_per_round", 25))
    teacher_steps = int(train.get("teacher_steps", 60000))
    dagger_steps = int(train.get("dagger_steps_per_round", 30000))
    if initial_episodes <= 0 or dagger_episodes <= 0 or rounds < 0:
        raise ValueError("BC requires positive episode counts and nonnegative DAgger rounds")
    # #4 B3 (formal spec): the rows join the actor and the optimizer now at zero weight,
    # and take their constants from the round-0 teacher episodes before the first fit.
    calibration = train.get("calibrate_extra_turn")
    if calibration:
        if not isinstance(policy, ConnectomePolicy) or policy.readout.mode != "structured":
            raise ValueError("calibrate_extra_turn needs a connectome policy, structured readout")
        if policy.readout.extra_turn_types:
            raise ValueError("calibrate_extra_turn needs an actor without extra turn rows")
        types = list(calibration["types"])
        eps = float(calibration.get("eps", DEFAULT_CONFIG["readout_extra_turn_eps"]))
        # New rows use the fixed center form even on a card from before it (#67 review).
        policy.readout.extra_center_ramp = True
        policy.model_config["readout_extra_turn_center_ramp"] = True
        policy.set_readout_extra_turns(types, [0.0] * len(types), [1.0] * len(types), eps)
    # #104 D3 (opt-in): detached forward-speed rows, also zero-weight until calibrated.
    forward_calibration = train.get("calibrate_forward_rows")
    if forward_calibration:
        if not isinstance(policy, ConnectomePolicy) or policy.readout.mode != "structured":
            raise ValueError("calibrate_forward_rows needs a connectome policy, structured readout")
        if policy.readout.forward_row_types:
            raise ValueError("calibrate_forward_rows needs an actor without forward rows")
        forward_types = list(forward_calibration["types"])
        width = len(forward_types) * policy.branches
        policy.set_readout_forward_rows(forward_types, [0.0] * width, [1.0] * width)
    optimizer = torch.optim.Adam(parameter_groups(policy, train))
    if teacher_steps <= 0 or dagger_steps <= 0:
        raise ValueError("Fair BC/DAgger budgets must be positive")
    recovery = Recovery(output, config, "bc", seed, device=device)
    store = EpisodeStore(output / "dataset")
    checkpoint = recovery.checkpoint
    episodes = store.load(checkpoint["dataset_count"]) if checkpoint else []
    environment_steps = checkpoint["environment_steps"] if checkpoint else 0
    reports = checkpoint["reports"] if checkpoint else []
    first_round = checkpoint["round_index"] if checkpoint else 0
    # The row constants are buffers outside the state dict; filled once, never rebound.
    calibrated = dict((checkpoint.get("extra_turn_calibration") if checkpoint else None) or {})
    if calibrated:
        policy.set_readout_extra_turns(*(calibrated[key] for key in EXTRA_TURN_KEYS))
    forward_calibrated = dict(
        (checkpoint.get("forward_row_calibration") if checkpoint else None) or {}
    )
    if forward_calibrated:
        policy.set_readout_forward_rows(*(forward_calibrated[key] for key in FORWARD_ROW_KEYS))
    if checkpoint:
        policy.load_state_dict(checkpoint["actor"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        restore_rng(checkpoint["rng"], rng)
    model = SequenceModel(policy, device)
    encoded: list[dict[str, torch.Tensor]] = []
    for round_index in range(first_round, rounds + 1):
        # The final round is fully student driven; labels always come from the privileged teacher.
        probability = (
            1.0
            if round_index == 0
            else float(train.get("dagger_beta", 0.5)) ** round_index
            if round_index < rounds
            else 0.0
        )
        continuing = checkpoint is not None and round_index == first_round
        phase = checkpoint["phase"] if continuing else "collect"
        base_count = checkpoint["collection_base_count"] if continuing else len(episodes)
        round_steps = checkpoint["round_steps"] if continuing else 0
        base_steps = environment_steps - round_steps
        budget = teacher_steps if round_index == 0 else dagger_steps
        collection = {"round_steps": round_steps}

        def save(
            phase,
            fit_state=None,
            base_count=base_count,
            round_index=round_index,
            collection=collection,
        ):
            recovery.save(
                rng,
                actor=policy.state_dict(),
                optimizer=optimizer.state_dict(),
                phase=phase,
                dataset_count=len(episodes),
                collection_base_count=base_count,
                environment_steps=environment_steps,
                round_steps=collection["round_steps"],
                round_index=round_index,
                reports=reports,
                fit_state=fit_state,
                extra_turn_calibration=calibrated or None,
                forward_row_calibration=forward_calibrated or None,
            )

        def collected(
            episode,
            count,
            steps,
            base_count=base_count,
            base_steps=base_steps,
            round_index=round_index,
            collection=collection,
        ):
            nonlocal environment_steps
            store.append(base_count + count - 1, episode)
            episodes.append(episode)
            collection["round_steps"], environment_steps = steps, base_steps + steps
            save("collect")
            recovery.progress(
                force=True,
                phase="collect",
                round=round_index,
                environment_steps=environment_steps,
                total_steps=teacher_steps + rounds * dagger_steps,
            )

        if phase == "collect":
            start, act = dagger_actor(model, contract, probability)
            pool = EnvPool(
                [
                    partial(make_env, config, contract, allow_scenario_mix=True)
                    for _ in range(max(1, default_workers()))
                ]
            )
            try:
                collect_lockstep(
                    pool,
                    seed=seed,
                    stream=BC_STREAM + round_index,
                    first_index=len(episodes) - base_count,
                    committed_steps=round_steps,
                    step_budget=budget,
                    act=act,
                    finish=lambda entry: pack_episode(entry.observations, entry.labels),
                    on_episode=lambda index, episode, steps, collected=collected: collected(
                        episode, index + 1, steps
                    ),
                    on_start=start,
                    on_progress=lambda steps, round_index=round_index, base_steps=base_steps: (
                        recovery.progress(
                            phase="collect",
                            round=round_index,
                            environment_steps=base_steps + steps,
                            total_steps=teacher_steps + rounds * dagger_steps,
                        )
                    ),
                )
            finally:
                pool.close()
            save("fit")
        # The frozen sensory code of an episode never changes; encode new ones only.
        encoded.extend(model.encode(episodes[len(encoded) :]))
        if calibration and not calibrated:
            # Training episodes only: the held-out ones are BC's own validation split.
            held = set(held_out_episodes(len(episodes), int(train.get("validation_every", 10))))
            report = calibrate_extra_turns(policy, model, episodes, encoded, held)
            calibrated.update(report["model"])
            policy.set_readout_extra_turns(*(calibrated[key] for key in EXTRA_TURN_KEYS))
            write_json(output / "extra_turn_calibration.json", report)
        if forward_calibration and not forward_calibrated:
            held = set(held_out_episodes(len(episodes), int(train.get("validation_every", 10))))
            report = calibrate_forward_rows(
                policy,
                model,
                episodes,
                encoded,
                held,
                int(forward_calibration.get("burn_in_steps", 60)),
            )
            forward_calibrated.update(report["model"])
            policy.set_readout_forward_rows(
                *(forward_calibrated[key] for key in FORWARD_ROW_KEYS)
            )
            write_json(output / "forward_row_calibration.json", report)
        fit_reports = fit_behavior(
            model,
            encoded,
            episodes,
            train,
            contract,
            optimizer,
            rng,
            checkpoint.get("fit_state") if continuing and phase == "fit" else None,
            on_checkpoint=lambda state: save("fit", state),
            on_progress=lambda round_index=round_index, force=False, phase="fit", **values: (
                recovery.progress(
                    force=force,
                    phase=phase,
                    round=round_index,
                    environment_steps=environment_steps,
                    **values,
                )
            ),
        )
        reports.append(
            {
                "round": round_index,
                "teacher_probability": probability,
                "environment_steps": environment_steps,
                "epochs": fit_reports,
            }
        )
        write_json(output / "training_metrics.json", reports)
        # Commit a clean next-round boundary before collecting more data.
        recovery.save(
            rng,
            actor=policy.state_dict(),
            optimizer=optimizer.state_dict(),
            phase="collect",
            dataset_count=len(episodes),
            collection_base_count=len(episodes),
            round_steps=0,
            environment_steps=environment_steps,
            round_index=round_index + 1,
            reports=reports,
            fit_state=None,
            extra_turn_calibration=calibrated or None,
            forward_row_calibration=forward_calibrated or None,
        )
    output.mkdir(parents=True, exist_ok=True)
    centering_metadata = {}
    if train.get("center_turn_bias", False):
        centering = center_turn_bias(policy, model, episodes, train)
        centering_metadata["turn_bias_centering"] = centering
        # Keep the rounds list and all epoch fields compatible with existing readers.
        reports[-1] = {**reports[-1], "turn_bias_centering": centering}
    write_json(output / "training_metrics.json", reports)
    save_actor(
        output / "actor",
        policy,
        config,
        contract,
        stage="bc_dagger",
        seed=seed,
        environment_steps=environment_steps,
        algorithm="bc_dagger",
        dataset_episodes=len(episodes),
        dagger_rounds=rounds,
        **centering_metadata,
    )
    recovery.progress(
        force=True,
        phase="complete",
        environment_steps=environment_steps,
        total_steps=teacher_steps + rounds * dagger_steps,
    )
