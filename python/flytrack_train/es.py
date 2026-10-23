"""Mirrored OpenAI-style evolution strategies, using the same interaction budget for every actor."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from .batches import tensor_observation
from .config import (
    TRAIN_ENVIRONMENT_SEED_LIMIT,
    make_env,
    prepare_policy,
    save_actor,
    seed_all,
    write_json,
)
from .imitation import parameter_groups
from .ppo import curriculum_stage
from .recovery import Recovery, restore_rng


def centered_ranks(values: np.ndarray) -> np.ndarray:
    """Average rank for ties avoids an arbitrary perturbation preference."""
    flat = np.asarray(values, dtype=np.float64).reshape(-1)
    if not np.isfinite(flat).all():
        raise FloatingPointError("ES fitness values must be finite before rank normalization")
    if len(flat) < 2:
        raise ValueError("Rank normalization requires at least two evaluations")
    order = np.argsort(flat, kind="stable")
    ranks = np.empty(len(flat), dtype=np.float64)
    first = 0
    while first < len(flat):
        last = first + 1
        while last < len(flat) and flat[order[last]] == flat[order[first]]:
            last += 1
        ranks[order[first:last]] = (first + last - 1) / 2
        first = last
    return (ranks / (len(flat) - 1) - 0.5).reshape(np.asarray(values).shape)


def fixed_budget_return(policy, config, contract, steps: int, seed: int, device, curriculum):
    env = make_env(config, contract, **curriculum)
    total = 0.0
    episode = 0
    try:
        observation, _ = env.reset(seed=seed)
        state = policy.initial_state(1, device)
        policy.eval()
        with torch.no_grad():
            for _ in range(steps):
                action, state, _ = policy(tensor_observation([observation], device), state)
                observation, reward, terminated, truncated, _ = env.step(action[0].cpu().numpy())
                total += reward
                if terminated or truncated:
                    episode += 1
                    observation, _ = env.reset(seed=(seed + episode) % TRAIN_ENVIRONMENT_SEED_LIMIT)
                    state = policy.initial_state(1, device)
    finally:
        env.close()
    return total / steps


def train_es(config: dict[str, Any], output: Path, seed: int, device: torch.device) -> None:
    seed_all(seed)
    policy, contract = prepare_policy(config, device)
    train = config.get("train", {})
    directions = int(train.get("population_size", 16))
    rollout_steps = int(train.get("evaluation_steps", 1250))
    total_steps = int(train.get("total_steps", 2_000_000))
    sigma = float(train.get("sigma", 0.02))
    generation_steps = 2 * directions * rollout_steps
    if (
        directions <= 0
        or rollout_steps <= 0
        or sigma <= 0
        or total_steps <= 0
        or total_steps % generation_steps
    ):
        raise ValueError(
            "ES total_steps must be a positive multiple of 2*population_size*evaluation_steps"
        )
    # ES has one step size: a zero group rate (#4: core_bias_learning_rate: 0) keeps
    # those parameters out of the search; any other group rate cannot be honoured.
    for name in ("core_bias_learning_rate", "readout_learning_rate", "readout_bias_learning_rate"):
        if train.get(name) is not None and float(train[name]) != 0:
            raise ValueError(f"ES supports only 0 (frozen) for {name}")
    frozen = {
        id(p)
        for group in parameter_groups(policy, {**train, "learning_rate": 1.0})
        if group["lr"] == 0
        for p in group["params"]
    }
    parameters = [p for p in policy.parameters() if p.requires_grad and id(p) not in frozen]
    theta = nn.Parameter(nn.utils.parameters_to_vector(parameters).detach().clone())
    optimizer = torch.optim.Adam([theta], lr=float(train.get("learning_rate", 0.01)))
    generator = torch.Generator(device=device).manual_seed(seed)
    rng = np.random.default_rng(seed)
    reports = []
    output.mkdir(parents=True, exist_ok=True)
    recovery = Recovery(output, config, "es", seed, device=device)
    first_generation = 0
    if recovery.checkpoint is not None:
        checkpoint = recovery.checkpoint
        policy.load_state_dict(checkpoint["actor"])
        theta.data.copy_(checkpoint["theta"].to(device))
        optimizer.load_state_dict(checkpoint["optimizer"])
        generator.set_state(checkpoint["perturbation_generator"].cpu())
        restore_rng(checkpoint["rng"], rng)
        reports = checkpoint["reports"]
        first_generation = checkpoint["environment_steps"] // generation_steps
    recovery.progress(
        force=True,
        phase="generation",
        environment_steps=first_generation * generation_steps,
        total_steps=total_steps,
    )
    for generation in range(first_generation, total_steps // generation_steps):
        noise = torch.randn((directions, len(theta)), device=device, generator=generator)
        scores = np.empty((directions, 2), dtype=np.float64)
        stage = curriculum_stage(train, generation * generation_steps / total_steps)
        for direction in range(directions):
            # Common random numbers for +epsilon/-epsilon reduce environmental variance.
            evaluation_seed = int(rng.integers(0, TRAIN_ENVIRONMENT_SEED_LIMIT))
            for sign_index, sign in enumerate((1, -1)):
                nn.utils.vector_to_parameters(
                    (theta.detach() + sign * sigma * noise[direction]).clone(), parameters
                )
                scores[direction, sign_index] = fixed_budget_return(
                    policy, config, contract, rollout_steps, evaluation_seed, device, stage
                )
                recovery.progress(
                    phase="generation",
                    environment_steps=generation * generation_steps
                    + (2 * direction + sign_index + 1) * rollout_steps,
                    total_steps=total_steps,
                    generation=generation + 1,
                )
        ranks = torch.as_tensor(centered_ranks(scores), device=device, dtype=theta.dtype)
        gradient = ((ranks[:, 0] - ranks[:, 1]).unsqueeze(1) * noise).sum(0) / (
            2 * directions * sigma
        )
        optimizer.zero_grad(set_to_none=True)
        # Adam minimizes; negate the estimated ascent direction and add optional L2 shrinkage.
        theta.grad = -gradient + float(train.get("weight_decay", 0.001)) * theta.detach()
        optimizer.step()
        nn.utils.vector_to_parameters(theta.detach().clone(), parameters)
        consumed = (generation + 1) * generation_steps
        reports.append(
            {
                "generation": generation + 1,
                "environment_steps": consumed,
                "mean_perturbed_reward_per_step": float(scores.mean()),
                "max_perturbed_reward_per_step": float(scores.max()),
                "gradient_norm": float(gradient.norm()),
            }
        )
        write_json(output / "training_metrics.json", reports)
        recovery.save(
            rng,
            actor=policy.state_dict(),
            theta=theta.detach(),
            optimizer=optimizer.state_dict(),
            environment_steps=consumed,
            perturbation_generator=generator.get_state(),
            reports=reports,
        )
        recovery.progress(
            force=True,
            phase="checkpoint",
            environment_steps=consumed,
            total_steps=total_steps,
            generation=generation + 1,
        )
    save_actor(
        output / "actor",
        policy,
        config,
        contract,
        stage="es",
        seed=seed,
        environment_steps=total_steps,
        algorithm="mirrored_openai_es",
    )
    recovery.progress(
        force=True, phase="complete", environment_steps=total_steps, total_steps=total_steps
    )
