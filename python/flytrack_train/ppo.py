"""Recurrent PPO with explicit terminal observations and asymmetric critic."""

from __future__ import annotations

import time
from functools import partial
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from .batches import SequenceModel, reset_state, tensor_observation
from .config import (
    TRAIN_ENVIRONMENT_SEED_LIMIT,
    make_env,
    prepare_policy,
    save_actor,
    seed_all,
    write_json,
)
from .distributions import BoundedGaussian, generalized_advantage
from .imitation import parameter_groups
from .losses import activity_penalty
from .parallel import EnvPool
from .recovery import Recovery, restore_rng


class PrivilegedCritic(nn.Module):
    def __init__(self, state_size: int = 12, hidden_size: int = 128):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(state_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, state):
        return self.network(state).squeeze(-1)


def _privileged(infos: list[dict[str, Any]], device: torch.device):
    values = np.stack([info["privileged_state"] for info in infos]).astype(np.float32)
    if values.ndim != 2 or values.shape[1] != 12 or not np.isfinite(values).all():
        raise ValueError("Privileged critic requires finite states of shape [batch,12]")
    return torch.as_tensor(values, device=device)


def curriculum_stage(train: dict[str, Any], progress: float) -> dict[str, Any]:
    stages = train.get(
        "curriculum",
        [
            {
                "until": 0.15,
                "target_motion": "stationary",
                "plume_model": "gaussian",
                "scenario": "normal",
            },
            {
                "until": 0.35,
                "target_motion": "linear",
                "plume_model": "gaussian",
                "scenario": "normal",
            },
            {
                "until": 0.6,
                "target_motion": ["curved", "wander"],
                "plume_model": "puff",
                "scenario": "normal",
            },
            {
                "until": 0.85,
                "target_motion": ["curved", "wander"],
                "plume_model": "puff",
                "scenario": "occlusion",
            },
            {
                "until": 1.0,
                "target_motion": ["curved", "wander"],
                "plume_model": "puff",
                "scenario": "conflict",
            },
        ],
    )
    if not stages or any(float(s["until"]) <= 0 for s in stages) or float(stages[-1]["until"]) < 1:
        raise ValueError("Curriculum must finish at progress >=1")
    if any(
        float(a["until"]) >= float(b["until"]) for a, b in zip(stages, stages[1:], strict=False)
    ):
        raise ValueError("Curriculum progress thresholds must be increasing")
    return {
        k: v
        for k, v in next(s for s in stages if progress <= float(s["until"])).items()
        if k != "until"
    }


def train_ppo(config: dict[str, Any], output: Path, seed: int, device: torch.device) -> None:
    seed_all(seed)
    policy, contract = prepare_policy(config, device)
    train = config.get("train", {})
    n_envs = int(train.get("num_envs", 8))
    rollout_steps = int(train.get("rollout_steps", 256))
    sequence_length = int(train.get("sequence_length", 32))
    total_steps = int(train.get("total_steps", 2_000_000))
    if (
        n_envs <= 0
        or sequence_length <= 0
        or rollout_steps < sequence_length
        or total_steps < n_envs
    ):
        raise ValueError(
            "PPO requires positive environments, sequence length, and a nonempty rollout"
        )
    # Avoid hidden extra interactions: final rollout can be shorter than the normal rollout.
    if total_steps % n_envs:
        raise ValueError("total_steps must be divisible by num_envs for exact comparison budgets")
    critic = PrivilegedCritic(hidden_size=int(train.get("critic_hidden_size", 128))).to(device)
    parameters = list(policy.parameters()) + list(critic.parameters())
    # Actor groups as in pretraining and BC, so a bias freeze (#4) holds through PPO.
    learning_rate = float(train.get("learning_rate", 3e-4))
    optimizer = torch.optim.Adam(
        [
            *parameter_groups(policy, {**train, "learning_rate": learning_rate}),
            {"params": list(critic.parameters()), "lr": learning_rate},
        ],
        eps=1e-5,
    )
    gamma, gae_lambda = float(train.get("gamma", 0.99)), float(train.get("gae_lambda", 0.95))
    clip = float(train.get("clip_coef", 0.2))
    epochs = int(train.get("update_epochs", 4))
    chunk_batch = int(train.get("sequence_batch_size", 16))
    max_grad_norm = float(train.get("max_grad_norm", 0.5))
    if (
        epochs <= 0
        or chunk_batch <= 0
        or max_grad_norm <= 0
        or not 0 <= gamma <= 1
        or not 0 <= gae_lambda <= 1
    ):
        raise ValueError("Invalid PPO optimization or discount settings")
    output.mkdir(parents=True, exist_ok=True)
    observations, infos = [], []
    pool = None
    model = SequenceModel(policy, device)
    sensory, state = model.initial(n_envs)
    episode_start = torch.ones(n_envs, device=device, dtype=torch.bool)
    rng = np.random.default_rng(seed)
    recovery = Recovery(output, config, "ppo", seed, device=device)
    consumed, iteration = 0, 0
    reports, episode_returns, running_returns = [], [], np.zeros(n_envs)
    start_time = time.perf_counter()
    stage = curriculum_stage(train, 0)
    env_seeds = [(seed * 10000 + index) % TRAIN_ENVIRONMENT_SEED_LIMIT for index in range(n_envs)]
    env_stages = [dict(stage) for _ in range(n_envs)]
    action_histories: list[list[list[float]]] = [[] for _ in range(n_envs)]
    elapsed_offset = 0.0
    checkpoint = recovery.checkpoint
    if checkpoint is not None:
        policy.load_state_dict(checkpoint["actor"])
        critic.load_state_dict(checkpoint["critic"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        consumed, iteration = checkpoint["environment_steps"], checkpoint["iteration"]
        reports = checkpoint["reports"]
        episode_returns = checkpoint["episode_returns"]
        running_returns = checkpoint["running_returns"].numpy().copy()
        state = checkpoint["recurrent_state"].to(device)
        if checkpoint.get("sensory_state") is not None:
            sensory = checkpoint["sensory_state"].to(device)
        episode_start = checkpoint["episode_start"].to(device)
        env_seeds, env_stages = checkpoint["env_seeds"], checkpoint["env_stages"]
        action_histories = checkpoint["action_histories"]
        elapsed_offset = float(checkpoint["elapsed_s"])
    # Curriculum of the world each slot currently holds (changes only at resets).
    env_curricula = [dict(item) for item in env_stages]
    try:
        pool = EnvPool([partial(make_env, config, contract, **item) for item in env_stages])
        for index in range(n_envs):
            observation, info = pool.reset(index, env_seeds[index])
            if action_histories[index]:
                observation, _, terminated, truncated, info = pool.replay(
                    index, action_histories[index]
                )
                if terminated or truncated:
                    raise ValueError("Recovery replay crossed an unrecorded episode reset")
            if checkpoint is not None:
                for key in ("vision", "odor", "valid"):
                    if not np.array_equal(
                        observation[key], checkpoint["observations"][key][index].numpy()
                    ):
                        raise ValueError(
                            f"Recovery world replay differs for environment {index} "
                            f"observation {key}"
                        )
                if not np.array_equal(
                    info["privileged_state"], checkpoint["privileged"][index].numpy()
                ):
                    raise ValueError(
                        f"Recovery world replay differs for environment {index} privileged state"
                    )
                if bool(episode_start[index]) != (not action_histories[index]):
                    raise ValueError(
                        "Recovery reset boundary differs from recorded recurrent state"
                    )
                pool.set(index, "_episode_id", checkpoint["env_episode_ids"][index])
                info["episode_id"] = checkpoint["env_episode_ids"][index]
            observations.append(observation)
            infos.append(info)
        if checkpoint is not None:
            # Rebuilding models/worlds may consume random values. Restore the
            # trainer streams last, after deterministic world replay completes.
            restore_rng(checkpoint["rng"], rng)
        recovery.progress(
            force=True,
            phase="rollout",
            environment_steps=consumed,
            total_steps=total_steps,
            resumed=checkpoint is not None,
        )
        while consumed < total_steps:
            iteration += 1
            steps = min(rollout_steps, (total_steps - consumed) // n_envs)
            buffer = {
                key: []
                for key in (
                    "inputs",
                    "privileged",
                    "state",
                    "start",
                    "latent",
                    "logprob",
                    "value",
                    "reward",
                    "next_value",
                    "terminated",
                    "truncated",
                )
            }
            policy.train()
            initial_sensory, initial_central = model.initial(n_envs)
            for _ in range(steps):
                batch_obs = tensor_observation(observations, device)
                privileged = _privileged(infos, device)
                state = reset_state(state, episode_start, initial_central)
                if sensory is not None:
                    sensory = reset_state(sensory, episode_start, initial_sensory)
                with torch.no_grad(), model.cache():
                    # The frozen sensory code is stored and reused by every update.
                    inputs, next_sensory = model.observe(batch_obs, sensory)
                    mean, next_state, _ = model.step(inputs, state)
                    distribution = BoundedGaussian(
                        mean, policy.log_std, contract.v_max, contract.omega_max
                    )
                    actions, latent, logprob = distribution.sample()
                    value = critic(privileged)
                rewards, terminated, truncated, final_infos = [], [], [], []
                next_observations, next_infos = [], []
                host_actions = actions.cpu().numpy()
                results = pool.step({index: host_actions[index] for index in range(n_envs)})
                for index in range(n_envs):
                    observation, reward, terminal, timeout, info = results[index]
                    action_histories[index].append(actions[index].detach().cpu().tolist())
                    final_infos.append(info)  # Always the actual final state, before any reset.
                    rewards.append(reward)
                    terminated.append(terminal)
                    truncated.append(timeout)
                    running_returns[index] += reward
                    if terminal or timeout:
                        episode_returns.append(float(running_returns[index]))
                        running_returns[index] = 0
                        current_stage = curriculum_stage(
                            train, min(1, (consumed + n_envs) / total_steps)
                        )
                        # Curriculum changes only at episode boundaries, never mid-episode.
                        if current_stage != env_curricula[index]:
                            pool.remake(index, partial(make_env, config, contract, **current_stage))
                            env_curricula[index] = dict(current_stage)
                        env_seeds[index] = int(rng.integers(0, TRAIN_ENVIRONMENT_SEED_LIMIT))
                        env_stages[index] = current_stage
                        action_histories[index] = []
                        observation, info = pool.reset(index, env_seeds[index])
                    next_observations.append(observation)
                    next_infos.append(info)
                with torch.no_grad():
                    final_values = critic(_privileged(final_infos, device))
                row = {
                    "inputs": inputs,
                    "privileged": privileged,
                    "state": state.detach(),
                    "start": episode_start,
                    "latent": latent,
                    "logprob": logprob,
                    "value": value,
                    "reward": torch.as_tensor(rewards, device=device, dtype=torch.float32),
                    "next_value": final_values,
                    "terminated": torch.as_tensor(terminated, device=device, dtype=torch.bool),
                    "truncated": torch.as_tensor(truncated, device=device, dtype=torch.bool),
                }
                for key, value in row.items():
                    buffer[key].append(value)
                observations, infos, state = next_observations, next_infos, next_state.detach()
                sensory = next_sensory
                episode_start = row["terminated"] | row["truncated"]
                consumed += n_envs
                recovery.progress(
                    phase="rollout",
                    environment_steps=consumed,
                    total_steps=total_steps,
                    iteration=iteration,
                )
            data = {key: torch.stack(values) for key, values in buffer.items() if key != "inputs"}
            data["inputs"] = {
                key: torch.stack([item[key] for item in buffer["inputs"]])
                for key in buffer["inputs"][0]
            }
            advantage, returns = generalized_advantage(
                data["reward"],
                data["value"],
                data["next_value"],
                data["terminated"],
                data["truncated"],
                gamma,
                gae_lambda,
            )
            advantage = (advantage - advantage.mean()) / (advantage.std(unbiased=False) + 1e-8)
            descriptors = [
                (offset, env_index, min(offset + sequence_length, steps))
                for offset in range(0, steps, sequence_length)
                for env_index in range(n_envs)
            ]
            diagnostics = []
            stop_early = False
            for _ in range(epochs):
                rng.shuffle(descriptors)
                # Equal-length chunks are grouped; the last short chunk needs no padded gradients.
                buckets: dict[int, list[tuple[int, int, int]]] = {}
                for descriptor in descriptors:
                    buckets.setdefault(descriptor[2] - descriptor[0], []).append(descriptor)
                for length, chunks in buckets.items():
                    for first in range(0, len(chunks), chunk_batch):
                        selected = chunks[first : first + chunk_batch]
                        starts = torch.tensor([s for s, _, _ in selected], device=device)
                        env_indices = torch.tensor([e for _, e, _ in selected], device=device)
                        chunk_state = data["state"][starts, env_indices].detach()
                        logprobs, entropies, activities = [], [], []
                        _, chunk_initial = model.initial(len(selected))
                        with model.cache():
                            for offset in range(length):
                                times = starts + offset
                                chunk_state = reset_state(
                                    chunk_state, data["start"][times, env_indices], chunk_initial
                                )
                                inputs = {
                                    key: values[times, env_indices]
                                    for key, values in data["inputs"].items()
                                }
                                mean, chunk_state, activity = model.step(inputs, chunk_state)
                                dist = BoundedGaussian(
                                    mean, policy.log_std, contract.v_max, contract.omega_max
                                )
                                logprobs.append(
                                    dist.log_prob_latent(data["latent"][times, env_indices])
                                )
                                entropies.append(dist.entropy())
                                activities.append(activity_penalty(policy, activity).mean())
                        times = starts.unsqueeze(0) + torch.arange(length, device=device).unsqueeze(
                            1
                        )
                        indices = env_indices.unsqueeze(0).expand(length, -1)
                        new_logprob = torch.stack(logprobs)
                        old_logprob = data["logprob"][times, indices]
                        logratio = new_logprob - old_logprob
                        ratio = logratio.exp()
                        adv = advantage[times, indices]
                        actor_loss = torch.maximum(
                            -adv * ratio, -adv * ratio.clamp(1 - clip, 1 + clip)
                        ).mean()
                        value = critic(data["privileged"][times, indices])
                        value_target, old_value = (
                            returns[times, indices],
                            data["value"][times, indices],
                        )
                        value_clipped = old_value + (value - old_value).clamp(-clip, clip)
                        critic_loss = (
                            0.5
                            * torch.maximum(
                                (value - value_target).square(),
                                (value_clipped - value_target).square(),
                            ).mean()
                        )
                        entropy = torch.stack(entropies).mean()
                        activity_loss = torch.stack(activities).mean()
                        loss = (
                            actor_loss
                            + float(train.get("value_coef", 0.5)) * critic_loss
                            - float(train.get("entropy_coef", 0.01)) * entropy
                        )
                        loss = loss + float(train.get("activity_coef", 1e-4)) * activity_loss
                        if not bool(torch.isfinite(loss)):
                            raise FloatingPointError(
                                "PPO loss is nonfinite; inspect actor activity and reward scale"
                            )
                        optimizer.zero_grad(set_to_none=True)
                        loss.backward()
                        nn.utils.clip_grad_norm_(parameters, max_grad_norm, error_if_nonfinite=True)
                        optimizer.step()
                        approx_kl = ((ratio - 1) - logratio).mean().detach().item()
                        diagnostics.append(
                            [actor_loss.item(), critic_loss.item(), entropy.item(), approx_kl]
                        )
                        if train.get("target_kl") and approx_kl > float(train["target_kl"]):
                            stop_early = True
                            break
                    if stop_early:
                        break
                if stop_early:
                    break
            average = np.mean(diagnostics, axis=0)
            reports.append(
                {
                    "iteration": iteration,
                    "environment_steps": consumed,
                    "actor_loss": float(average[0]),
                    "critic_loss": float(average[1]),
                    "entropy": float(average[2]),
                    "approx_kl": float(average[3]),
                    "mean_last_20_episode_return": float(np.mean(episode_returns[-20:]))
                    if episode_returns
                    else None,
                    "elapsed_s": elapsed_offset + time.perf_counter() - start_time,
                }
            )
            write_json(output / "training_metrics.json", reports)
            # Keep the critic/optimizer for research recovery; exported actors exclude both.
            recovery.save(
                rng,
                actor=policy.state_dict(),
                critic=critic.state_dict(),
                optimizer=optimizer.state_dict(),
                environment_steps=consumed,
                iteration=iteration,
                reports=reports,
                episode_returns=episode_returns,
                running_returns=torch.as_tensor(running_returns),
                recurrent_state=state.detach().cpu(),
                sensory_state=None if sensory is None else sensory.detach().cpu(),
                episode_start=episode_start.cpu(),
                observations={
                    k: v.cpu() for k, v in tensor_observation(observations, device).items()
                },
                privileged=_privileged(infos, device).cpu(),
                env_seeds=env_seeds,
                env_stages=env_stages,
                action_histories=action_histories,
                env_episode_ids=[pool.get(index, "_episode_id") for index in range(n_envs)],
                elapsed_s=elapsed_offset + time.perf_counter() - start_time,
            )
            recovery.progress(
                force=True,
                phase="checkpoint",
                environment_steps=consumed,
                total_steps=total_steps,
                iteration=iteration,
                **{
                    k: reports[-1][k]
                    for k in ("actor_loss", "critic_loss", "mean_last_20_episode_return")
                },
            )
        save_actor(
            output / "actor",
            policy,
            config,
            contract,
            stage="ppo",
            seed=seed,
            environment_steps=consumed,
            algorithm="recurrent_ppo",
            sequence_length=sequence_length,
        )
        recovery.progress(
            force=True, phase="complete", environment_steps=consumed, total_steps=total_steps
        )
    finally:
        if pool is not None:
            pool.close()
