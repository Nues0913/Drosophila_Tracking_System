"""Teacher-only BC collection followed by on-policy DAgger aggregation."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch

from .batches import Episode, pack_episode, tensor_observation
from .config import (
    TRAIN_ENVIRONMENT_SEED_LIMIT,
    make_env,
    prepare_policy,
    save_actor,
    seed_all,
    write_json,
)
from .recovery import EpisodeStore, Recovery, fit_sequence_epochs, restore_rng


def collect_episodes(
    config,
    contract,
    policy,
    count: int,
    rng: np.random.Generator,
    device,
    teacher_probability: float,
    step_budget: int | None = None,
    initial_episodes=None,
    initial_steps=0,
    on_episode=None,
    on_progress=None,
):
    episodes: list[Episode] = list(initial_episodes or [])
    steps = initial_steps
    env = make_env(config, contract)
    try:
        policy.eval()
        while steps < step_budget if step_budget is not None else len(episodes) < count:
            observation, info = env.reset(seed=int(rng.integers(0, TRAIN_ENVIRONMENT_SEED_LIMIT)))
            state = policy.initial_state(1, device)
            observations, labels = [], []
            while True:
                teacher_action = np.asarray(info["teacher_action"], dtype=np.float32)
                observations.append(
                    {
                        key: np.array(observation[key], copy=True)
                        for key in ("vision", "odor", "valid")
                    }
                )
                labels.append({"action": teacher_action.copy()})
                with torch.no_grad():
                    student_action, state, _ = policy(
                        tensor_observation([observation], device), state
                    )
                action = (
                    teacher_action
                    if rng.random() < teacher_probability
                    else student_action[0].cpu().numpy()
                )
                observation, _, terminated, truncated, info = env.step(action)
                steps += 1
                if on_progress:
                    on_progress(steps)
                if terminated or truncated or (step_budget is not None and steps >= step_budget):
                    break
            episodes.append(pack_episode(observations, labels))
            if on_episode:
                on_episode(episodes[-1], len(episodes), steps)
    finally:
        env.close()
    return episodes, steps


def fit_behavior(
    policy,
    episodes: list[Episode],
    train: dict[str, Any],
    contract,
    optimizer,
    rng,
    device,
    fit_state=None,
    on_checkpoint=None,
    on_progress=None,
):
    if not episodes:
        raise ValueError("Behavior cloning requires collected episodes")
    scale = torch.tensor([contract.v_max, contract.omega_max], device=device)

    def window_loss(episode, start, end, state):
        predictions, activities = [], []
        for step in range(start, end):
            obs = {
                key: torch.as_tensor(values[step : step + 1], device=device)
                for key, values in episode.observations.items()
            }
            prediction, state, activity = policy(obs, state)
            predictions.append(prediction[0])
            activities.append(activity.square().mean())
        target = torch.as_tensor(episode.labels["action"][start:end], device=device)
        loss = ((torch.stack(predictions) - target) / scale).square().mean()
        loss = loss + float(train.get("activity_coef", 1e-4)) * torch.stack(activities).mean()
        return loss, state

    return fit_sequence_epochs(
        policy,
        episodes,
        train,
        optimizer,
        list(policy.parameters()),
        rng,
        device,
        window_loss,
        on_checkpoint or (lambda state: None),
        on_progress or (lambda **values: None),
        fit_state,
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
    optimizer = torch.optim.Adam(policy.parameters(), lr=float(train.get("learning_rate", 1e-3)))
    if teacher_steps <= 0 or dagger_steps <= 0:
        raise ValueError("Fair BC/DAgger budgets must be positive")
    recovery = Recovery(output, config, "bc", seed, device=device)
    store = EpisodeStore(output / "dataset")
    checkpoint = recovery.checkpoint
    episodes = store.load(checkpoint["dataset_count"]) if checkpoint else []
    environment_steps = checkpoint["environment_steps"] if checkpoint else 0
    reports = checkpoint["reports"] if checkpoint else []
    first_round = checkpoint["round_index"] if checkpoint else 0
    if checkpoint:
        policy.load_state_dict(checkpoint["actor"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        restore_rng(checkpoint["rng"], rng)
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
            collect_episodes(
                config,
                contract,
                policy,
                initial_episodes if round_index == 0 else dagger_episodes,
                rng,
                device,
                probability,
                budget,
                initial_episodes=episodes[base_count:],
                initial_steps=round_steps,
                on_episode=collected,
                on_progress=lambda steps, round_index=round_index, base_steps=base_steps: (
                    recovery.progress(
                        phase="collect",
                        round=round_index,
                        environment_steps=base_steps + steps,
                        total_steps=teacher_steps + rounds * dagger_steps,
                    )
                ),
            )
            save("fit")
        fit_reports = fit_behavior(
            policy,
            episodes,
            train,
            contract,
            optimizer,
            rng,
            device,
            checkpoint.get("fit_state") if continuing and phase == "fit" else None,
            on_checkpoint=lambda state: save("fit", state),
            on_progress=lambda round_index=round_index, **values: recovery.progress(
                phase="fit", round=round_index, environment_steps=environment_steps, **values
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
        )
    output.mkdir(parents=True, exist_ok=True)
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
    )
    recovery.progress(
        force=True,
        phase="complete",
        environment_steps=environment_steps,
        total_steps=teacher_steps + rounds * dagger_steps,
    )
