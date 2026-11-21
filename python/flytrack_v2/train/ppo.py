"""PPO-v2: fine-tune a Stage 1 policy with recurrent PPO (plan 3.8 Stage 2, #138).

- Asymmetric actor-critic: the actor is the exported Stage 1 policy and sees observations only;
  the critic is an MLP on the privileged state (target relative position and velocity, walls,
  occlusion, wind, decoy, episode time).
- Recurrent PPO: each rollout stores the central state before its first step and the sensory
  codes (the frontend is frozen), so an update replays the same 32 steps from that state;
  rows that reset inside the rollout restart from the rest state the rollout used.
- Warm-up (PIRLNav, #161 D2): the first updates train the critic only; the actor learning rate
  then ramps up linearly and decays linearly to the end.
- Kickstarting (#161 D1): the loss adds lambda * KL(pi_teacher || pi_theta) to the frozen
  Stage 1 policy, which runs on the same codes with its own recurrent state; lambda falls
  linearly to 0 over the first part of the actor updates, after which the teacher is not run.
- Reward: the world's track_v2 (#181). A time limit, and the forced end of a desynchronised
  first episode, bootstrap from the value of the state before the reset; only a collision
  terminates.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import Tensor, nn

from flytrack_v2.interfaces import reset_rows
from flytrack_v2.model import export_policy, load_policy, parameter_groups
from flytrack_v2.model.policy import ConnectomePolicyV2, attach_plastic_projection
from flytrack_v2.monitor import EngineeringFailure, RunMonitor

from .config import CONFIG_DIR, MOTION_NAMES, STOP_INTERVAL, WORLD_NAMES, _override, read_yaml
from .data import RolloutCollector, _concatenate

CRITIC_KEYS = (
    "agent_pose",
    "agent_velocity",
    "target_pose",
    "target_velocity",
    "occluded",
    "target_visible",
    "wind_velocity",
    "decoy_active",
    "decoy_position",
    "time_s",
)
CRITIC_FEATURES = 24


# ---------------------------------------------------------------------------------------------
# Configuration


def load_ppo_config(path: Path, host: str, overrides: Sequence[str] = ()) -> dict:
    """The PPO settings with --override applied, and the host's micro-batch (hosts.yaml)."""
    defaults = read_yaml(CONFIG_DIR / "ppo.yaml")
    config = read_yaml(path)
    if config.keys() - {"plastic_lr_scale"} != defaults.keys() - {"plastic_lr_scale"}:
        raise ValueError(
            "PPO config must have the same keys as ppo.yaml plus optional plastic_lr_scale"
        )
    for expression in overrides:
        if expression.startswith("plastic_lr_scale="):
            config.setdefault("plastic_lr_scale", 1.0)
        _override(config, expression)
    if "plastic_lr_scale" in config:
        scale = config["plastic_lr_scale"]
        if type(scale) not in (int, float) or not math.isfinite(scale) or scale <= 0:
            raise ValueError("plastic_lr_scale must be positive and finite")
    hosts = read_yaml(CONFIG_DIR / "hosts.yaml")
    if host not in hosts:
        raise ValueError(f"Unknown host {host!r}; hosts.yaml has {sorted(hosts)}")
    rows = config["envs_per_instance"] * len(WORLD_NAMES) * len(MOTION_NAMES)
    minibatch = rows // config["minibatches"]
    if rows % config["minibatches"]:
        raise ValueError("The environment count must be divisible by minibatches")
    micro = min(hosts[host]["micro_batch"], minibatch)
    if minibatch % micro:
        raise ValueError("The minibatch must be divisible by the host micro-batch")
    warmup = config["warmup"]
    updates = math.ceil(config["total_env_steps"] / (rows * config["rollout_steps"]))
    # The actor trains during its learning-rate ramp already, so one update past the critic-only
    # warm-up is enough (#188: 40 updates with a 20 + 20 warm-up were refused).
    if updates <= warmup["critic_only_updates"]:
        raise ValueError("total_env_steps leaves no actor updates after the critic warm-up")
    return {
        **config,
        "host": host,
        "rows": rows,
        "minibatch_rows": minibatch,
        "micro_batch": micro,
        "updates": updates,
    }


@dataclass(frozen=True)
class Schedule:
    actor_on: bool
    actor_lr: float
    critic_lr: float
    kick_lambda: float


def schedule(update: int, config: Mapping) -> Schedule:
    """Learning rates and the kickstarting weight of update ``update`` (0-based)."""
    total = config["updates"]
    warmup = config["warmup"]
    critic_only, ramp = warmup["critic_only_updates"], warmup["actor_ramp_updates"]
    actor_updates = total - critic_only
    k = update - critic_only
    if k < 0:
        actor_lr = 0.0
    elif k < ramp:
        actor_lr = config["actor_lr"] * (k + 1) / ramp
    else:
        actor_lr = config["actor_lr"] * max(0.0, 1 - (k - ramp) / max(1, actor_updates - ramp))
    critic_lr = config["critic_lr"] * max(0.0, 1 - update / total)
    kick = config["kickstart"]
    span = kick["anneal_fraction"] * actor_updates
    kick_lambda = kick["initial_lambda"] * (max(0.0, 1 - max(k, 0) / span) if span > 0 else 0.0)
    return Schedule(k >= 0, actor_lr, critic_lr, kick_lambda)


def steps_teacher(plan: Schedule) -> bool:
    """Run the frozen teacher in this update's rollout. While lambda > 0 it steps every
    rollout, the critic-only warm-up included, so its recurrent state has the student's
    history when the KL starts (#192 R7); the KL is only computed once the actor trains."""
    return plan.kick_lambda > 0


# ---------------------------------------------------------------------------------------------
# Critic


def critic_features(
    privileged: Mapping[str, Tensor], walls: Sequence[float], episode_seconds: float, v_max: float
) -> Tensor:
    """[B, CRITIC_FEATURES] egocentric privileged features, each of order 1."""
    f = torch.float32
    pose = privileged["agent_pose"].to(f)
    x, y, heading = pose.unbind(-1)
    cos, sin = heading.cos(), heading.sin()
    xmin, ymin, xmax, ymax = walls
    half = 0.5 * max(xmax - xmin, ymax - ymin)

    def ego(vx: Tensor, vy: Tensor) -> tuple[Tensor, Tensor]:
        return cos * vx + sin * vy, -sin * vx + cos * vy

    target = privileged["target_pose"].to(f)
    tx, ty = ego(target[:, 0] - x, target[:, 1] - y)
    distance = torch.hypot(tx, ty)
    bearing = torch.atan2(ty, tx)
    tvx, tvy = ego(*privileged["target_velocity"].to(f).unbind(-1))
    wx, wy = ego(*privileged["wind_velocity"].to(f).unbind(-1))
    decoy = privileged["decoy_position"].to(f)
    active = privileged["decoy_active"].to(f)
    dx, dy = ego(decoy[:, 0] - x, decoy[:, 1] - y)
    velocity = privileged["agent_velocity"].to(f)
    walls_near = torch.stack((x - xmin, xmax - x, y - ymin, ymax - y), -1) / half
    columns = [
        tx / half,
        ty / half,
        distance / half,
        bearing.cos(),
        bearing.sin(),
        tvx / v_max,
        tvy / v_max,
        velocity[:, 0] / v_max,
        velocity[:, 1],
        *walls_near.unbind(-1),
        walls_near.amin(-1),
        cos,
        sin,
        privileged["occluded"].to(f),
        privileged["target_visible"].to(f),
        wx,
        wy,
        active,
        active * dx / half,
        active * dy / half,
        privileged["time_s"].to(f) / episode_seconds,
    ]
    return torch.stack(columns, -1)


class Critic(nn.Module):
    """MLP value of the privileged features, in units of 1 / (1 - gamma) (returns of order 1)."""

    def __init__(self, n_features: int = CRITIC_FEATURES, hidden: Sequence[int] = (256, 256)):
        super().__init__()
        layers: list[nn.Module] = []
        width = n_features
        for size in hidden:
            linear = nn.Linear(width, size)
            nn.init.orthogonal_(linear.weight, math.sqrt(2))
            nn.init.zeros_(linear.bias)
            layers += [linear, nn.Tanh()]
            width = size
        head = nn.Linear(width, 1)
        nn.init.orthogonal_(head.weight, 1.0)
        nn.init.zeros_(head.bias)
        self.net = nn.Sequential(*layers, head)

    def forward(self, features: Tensor) -> Tensor:
        return self.net(features).squeeze(-1)


# ---------------------------------------------------------------------------------------------
# Rollouts, advantages and the update


@dataclass
class Rollout:
    """One PPO batch: T steps of N environments, on one device."""

    code: Tensor  # [T, N, code]
    latent: Tensor  # [T, N, 2], the sampled latent (log_prob is exact on it)
    logp: Tensor  # [T, N]
    features: Tensor  # [T, N, F], critic features of the state each action saw
    reward: Tensor  # [T, N]
    # [T, N] bool: the episode ends without a bootstrap: a collision, or the task's time limit,
    # which the critic sees as the episode time feature (#188 review; Pardo et al. 2018)
    terminated: Tensor
    truncated: Tensor  # [T, N] bool: cut short by a forced first-episode reset; bootstraps
    final_features: Tensor  # [T, N, F], the state before the reset where truncated
    last_features: Tensor  # [N, F], after the last step
    central: Tensor  # [N, S], the central state before the first step
    initial: Tensor  # [1, S], the rest state rows restart from
    teacher_mean: Tensor | None = None  # [T, N, 2] when kickstarting
    teacher_log_std: Tensor | None = None  # [2]


@dataclass(frozen=True)
class PPOSettings:
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip: float = 0.2
    target_kl: float = 0.02
    value_coef: float = 0.5
    entropy_coef: float = 0.001
    activity_coef: float = 1e-4
    grad_clip: float = 0.5
    epochs: int = 4
    minibatches: int = 8

    @classmethod
    def from_config(cls, config: Mapping) -> PPOSettings:
        return cls(**{name: config[name] for name in cls.__dataclass_fields__})


@torch.no_grad()
def advantages_and_returns(
    critic: Critic, rollout: Rollout, gamma: float, gae_lambda: float
) -> tuple[Tensor, Tensor, Tensor]:
    """GAE in return units: (advantages [T, N], returns [T, N], values [T, N]).

    A terminated episode (a collision or the 60 s end of the task) ends the return; a truncated
    one (a forced first-episode reset) bootstraps from the state before the reset; the
    advantage never crosses an episode boundary.
    """
    scale = 1.0 / (1.0 - gamma)
    values = critic(rollout.features) * scale
    final = critic(rollout.final_features) * scale
    following = torch.cat((values[1:], (critic(rollout.last_features) * scale)[None]))
    done = rollout.terminated | rollout.truncated
    next_value = torch.where(rollout.truncated, final, following)
    next_value = torch.where(rollout.terminated, torch.zeros_like(next_value), next_value)
    advantages = torch.zeros_like(values)
    running = torch.zeros_like(values[0])
    for t in reversed(range(len(values))):
        delta = rollout.reward[t] + gamma * next_value[t] - values[t]
        running = delta + gamma * gae_lambda * (~done[t]).to(values.dtype) * running
        advantages[t] = running
    return advantages, advantages + values, values


def gaussian_kl(mean_p: Tensor, log_std_p: Tensor, mean_q: Tensor, log_std_q: Tensor) -> Tensor:
    """KL(p || q) of diagonal Gaussians, summed over the last dimension."""
    log_std_p, log_std_q = log_std_p.expand_as(mean_p), log_std_q.expand_as(mean_q)
    var_p, var_q = (2 * log_std_p).exp(), (2 * log_std_q).exp()
    return (log_std_q - log_std_p + (var_p + (mean_p - mean_q).square()) / (2 * var_q) - 0.5).sum(
        -1
    )


def _actor_replay(policy, rollout: Rollout, rows: Tensor):
    """Replay the rollout's steps for ``rows`` from the stored state, with gradients."""
    graph = isinstance(policy, ConnectomePolicyV2)
    state = rollout.central[rows]
    initial = rollout.initial
    logp, entropy, means, activity = [], [], [], []
    for t in range(len(rollout.code)):
        if graph:
            step = policy.forward_code(rollout.code[t, rows], state, non_readout_activity=True)
            activity.append(step.aux["activity_sq"])
        else:
            step = policy.forward_code(rollout.code[t, rows], state)
            activity.append(step.central_state.square().mean(-1))
        logp.append(step.dist.log_prob(rollout.latent[t, rows]))
        entropy.append(step.dist.entropy())
        means.append(step.dist.mean)
        ended = rollout.terminated[t, rows] | rollout.truncated[t, rows]
        state = reset_rows(step.central_state, initial, ended)
    return torch.stack(logp), torch.stack(entropy), torch.stack(means), torch.stack(activity)


def _clip_and_step(optimizer, max_norm: float) -> Tensor:
    parameters = [
        p for group in optimizer.param_groups for p in group["params"] if p.grad is not None
    ]
    norm = torch.nn.utils.get_total_norm([p.grad for p in parameters])
    if bool(torch.isfinite(norm)):
        torch.nn.utils.clip_grads_with_norm_(parameters, max_norm, norm)
        optimizer.step()
    return norm


def ppo_update(
    policy,
    critic: Critic,
    actor_optimizer,
    critic_optimizer,
    rollout: Rollout,
    settings: PPOSettings,
    *,
    actor_on: bool,
    kick_lambda: float,
    micro_batch: int,
    generator: torch.Generator,
) -> dict[str, Tensor]:
    """Epochs of minibatch updates over environments; returns the means of the last epoch."""
    advantages, returns, values = advantages_and_returns(
        critic, rollout, settings.gamma, settings.gae_lambda
    )
    scale = 1.0 - settings.gamma
    target = returns * scale
    normalized = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    n = rollout.code.shape[1]
    minibatch = n // settings.minibatches
    kick = kick_lambda > 0 and rollout.teacher_mean is not None
    metrics: dict[str, Tensor] = {}
    for _epoch in range(settings.epochs):
        order = torch.randperm(n, generator=generator, device="cpu").to(rollout.code.device)
        sums: dict[str, Tensor] = {}
        for start in range(0, minibatch * settings.minibatches, minibatch):
            rows = order[start : start + minibatch]
            actor_optimizer.zero_grad(set_to_none=True)
            critic_optimizer.zero_grad(set_to_none=True)
            terms: dict[str, Tensor] = {}
            for part in rows.split(micro_batch):
                weight = len(part) / len(rows)
                value_loss = 0.5 * (critic(rollout.features[:, part]) - target[:, part]).square()
                loss = settings.value_coef * value_loss.mean()
                parts = {"value_loss": value_loss.mean().detach()}
                if actor_on:
                    logp, entropy, means, activity = _actor_replay(policy, rollout, part)
                    log_ratio = logp - rollout.logp[:, part]
                    ratio = log_ratio.exp()
                    adv = normalized[:, part]
                    clipped = ratio.clamp(1 - settings.clip, 1 + settings.clip)
                    policy_loss = -torch.minimum(ratio * adv, clipped * adv).mean()
                    loss = (
                        loss
                        + policy_loss
                        - settings.entropy_coef * entropy.mean()
                        + settings.activity_coef * activity.mean()
                    )
                    if kick:
                        kl = gaussian_kl(
                            rollout.teacher_mean[:, part],
                            rollout.teacher_log_std,
                            means,
                            policy.log_std,
                        ).mean()
                        loss = loss + kick_lambda * kl
                        parts["kl_teacher"] = kl.detach()
                    with torch.no_grad():
                        parts.update(
                            policy_loss=policy_loss.detach(),
                            entropy=entropy.mean(),
                            approx_kl=((ratio - 1) - log_ratio).mean(),
                            clip_fraction=((ratio - 1).abs() > settings.clip).float().mean(),
                        )
                (loss * weight).backward()
                for name, value in parts.items():
                    terms[name] = terms.get(name, 0.0) + value * weight
            terms["critic_grad_norm"] = _clip_and_step(critic_optimizer, settings.grad_clip)
            if actor_on:
                terms["actor_grad_norm"] = _clip_and_step(actor_optimizer, settings.grad_clip)
            for name, value in terms.items():
                sums[name] = sums.get(name, 0.0) + value / settings.minibatches
        metrics = sums
        if actor_on and float(sums["approx_kl"]) > 1.5 * settings.target_kl:
            break
    with torch.no_grad():
        residual = (returns - values).var()
        metrics["explained_variance"] = 1 - residual / returns.var().clamp_min(1e-8)
        metrics["value_mean"] = values.mean()
        metrics["return_mean"] = returns.mean()
        metrics["epochs_run"] = torch.tensor(float(_epoch + 1))
    return metrics


# ---------------------------------------------------------------------------------------------
# Collection on the GPU worlds


class PPOCollector(RolloutCollector):
    """WP4a's four worlds and first-episode desync; the student samples its own actions."""

    def __init__(self, policy, teacher, config: dict, seed: int, device: torch.device):
        super().__init__(policy, config, seed, device)
        self.teacher_policy = teacher
        self.sample_rng = torch.Generator(device=device).manual_seed(seed ^ 0x9905)
        self.teacher_initial = teacher.initial_central_state(self.rows, device)
        self.teacher_central = self.teacher_initial.clone()
        self.walls = self.worlds[0].walls
        self.episode_seconds = config["episode_seconds"]
        self.v_max = policy.contract.v_max

    def _features(self, privileged: Mapping[str, Tensor]) -> Tensor:
        return critic_features(privileged, self.walls, self.episode_seconds, self.v_max)

    def _privileged(self) -> dict[str, Tensor]:
        return _concatenate([world.privileged() for world in self.worlds], CRITIC_KEYS)

    def checkpoint(self) -> dict:
        saved = super().checkpoint()
        saved["teacher_central"] = self.teacher_central.cpu()
        saved["sample_rng"] = self.sample_rng.get_state()
        return saved

    def restore(self, saved: dict) -> None:
        super().restore(saved)
        self.teacher_central = saved["teacher_central"].to(self.device)
        self.sample_rng.set_state(saved["sample_rng"])

    @torch.no_grad()
    def collect_ppo(self, steps: int, *, kickstart: bool) -> tuple[Rollout, dict[str, Tensor]]:
        policy, device, rows = self.policy, self.device, self.rows
        initial = policy.initial_central_state(rows, device)
        central = self.central.clone()
        f = CRITIC_FEATURES
        buffers = {
            "code": torch.empty(steps, rows, policy.code_size, device=device),
            "latent": torch.empty(steps, rows, 2, device=device),
            "logp": torch.empty(steps, rows, device=device),
            "features": torch.empty(steps, rows, f, device=device),
            "reward": torch.empty(steps, rows, device=device),
            "terminated": torch.empty(steps, rows, dtype=torch.bool, device=device),
            "truncated": torch.empty(steps, rows, dtype=torch.bool, device=device),
            "final_features": torch.zeros(steps, rows, f, device=device),
        }
        teacher_mean = torch.empty(steps, rows, 2, device=device) if kickstart else None
        ended = torch.zeros((), device=device)
        stats = {name: torch.zeros((), device=device) for name in ("success", "collision")}
        reward_sum = torch.zeros((), device=device)
        width = self.worlds[0].num_envs
        for t in range(steps):
            code, self.sensory = policy.sense(self.obs, self.sensory)
            step = policy.forward_code(code, self.central)
            action, latent = step.dist.sample(self.sample_rng)
            buffers["code"][t] = code
            buffers["latent"][t] = latent
            buffers["logp"][t] = step.dist.log_prob(latent)
            buffers["features"][t] = self._features(self._privileged())
            if kickstart:
                taught = self.teacher_policy.forward_code(code, self.teacher_central)
                teacher_mean[t] = taught.dist.mean
                teacher_next = taught.central_state
            results = [
                world.step(part)
                for world, part in zip(self.worlds, action.split(width), strict=True)
            ]
            terminated = torch.cat([r.terminated for r in results])
            truncated = torch.cat([r.truncated for r in results])
            final = self._features(_concatenate([r.final_privileged for r in results], CRITIC_KEYS))
            observations = [r.obs for r in results]
            resets = [r.terminated | r.truncated for r in results]

            def before_reset(world_id: int, mask: Tensor, final=final) -> None:
                state = self._features(
                    {
                        k: v
                        for k, v in self.worlds[world_id].privileged().items()
                        if k in CRITIC_KEYS
                    }
                )
                block = slice(world_id * width, (world_id + 1) * width)
                final[block] = torch.where(mask[:, None], state, final[block])

            self._first_episode_resets(observations, resets, before_reset)
            reset = torch.cat(resets)
            forced = reset & ~(terminated | truncated)
            buffers["reward"][t] = torch.cat([r.reward for r in results]).to(torch.float32)
            # The 60 s limit is the task's end (evaluation episodes are 60 s too) and the critic
            # sees the episode time, so it terminates; only a forced first-episode end, which
            # the task does not have, bootstraps (#188 review).
            buffers["terminated"][t] = terminated | truncated
            buffers["truncated"][t] = forced
            buffers["final_features"][t] = torch.where(forced[:, None], final, 0.0)
            natural = terminated | truncated
            info = {k: torch.cat([r.info[k] for r in results]) for k in ("episode_success",)}
            ended += natural.sum()
            stats["success"] += (info["episode_success"].to(torch.float32) * natural).sum()
            stats["collision"] += terminated.sum()
            reward_sum += buffers["reward"][t].sum()
            self.obs = self._observations(observations)
            self.sensory = policy.reset_sensory_state(self.sensory, reset)
            self.central = reset_rows(step.central_state, initial, reset)
            if kickstart:
                self.teacher_central = reset_rows(teacher_next, self.teacher_initial, reset)
            else:
                self.teacher_central = reset_rows(self.teacher_central, self.teacher_initial, reset)
        rollout = Rollout(
            **buffers,
            last_features=self._features(self._privileged()),
            central=central,
            initial=initial[:1],
            teacher_mean=teacher_mean,
            teacher_log_std=self.teacher_policy.log_std.detach() if kickstart else None,
        )
        denominator = ended.clamp_min(1)
        metrics = {
            "episodes_ended": ended,
            "episode_success_rate": stats["success"] / denominator,
            "episode_collision_rate": stats["collision"] / denominator,
            "reward_per_step": reward_sum / (steps * rows),
        }
        return rollout, metrics


# ---------------------------------------------------------------------------------------------
# The run


def check_init_seed(init: Path, card: Mapping, seed: int) -> None:
    """--seed must be the Stage 1 artifact's training seed: the export records it as
    provenance.train_seed, which picks the evaluation episodes, so Stage 1 and Stage 2 pair by
    it (#188 review). Without one in the card, no seed_<n> part of the path may contradict it."""
    stage1 = card.get("provenance", {}).get("train_seed")
    in_path = {int(part[5:]) for part in Path(init).parts if re.fullmatch(r"seed_\d+", part)}
    if stage1 is not None and stage1 != seed:
        raise ValueError(f"--seed {seed} differs from the Stage 1 training seed {stage1}")
    if stage1 is None and in_path and seed not in in_path:
        raise ValueError(f"--seed {seed} contradicts the Stage 1 path {init}")


def _sha256_json(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


class PPORun:
    def __init__(self, config: dict, init: Path, seed: int, device: torch.device, out: Path):
        self.config, self.seed, self.device, self.out = config, seed, device, out
        self.init = init
        self.init_card = json.loads((init / "model_card.json").read_text(encoding="utf-8"))
        check_init_seed(init, self.init_card, seed)
        self.init_sha256 = hashlib.sha256((init / "SHA256SUMS").read_bytes()).hexdigest()
        self.policy = load_policy(init, device=device)
        self.policy.train()
        self.teacher = load_policy(init, device=device)
        for parameter in self.teacher.parameters():
            parameter.requires_grad_(False)
        self.critic = Critic(CRITIC_FEATURES, config["critic_hidden"]).to(device)
        self.actor_optimizer = torch.optim.Adam(
            parameter_groups(self.policy, 0.0, plastic_lr_scale=config.get("plastic_lr_scale", 1.0))
        )
        attach_plastic_projection(self.actor_optimizer, self.policy)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=config["critic_lr"])
        self.settings = PPOSettings.from_config(config)
        self.update_rng = torch.Generator().manual_seed(seed ^ 0x0BB0)
        self.update = 0
        self.env_steps = 0
        self.collector: PPOCollector | None = None
        # The resume check covers the seed, the host and the device too: resuming with another
        # --seed would keep the old collector RNG but export the new train_seed (#188 review).
        self.resolved = {
            **config,
            "seed": seed,
            "device": str(device),
            "init_artifact_sha256": self.init_sha256,
        }
        self.config_sha256 = _sha256_json(self.resolved)

    @property
    def variant(self) -> str:
        return self.init_card["variant"]

    def _set_rates(self, plan: Schedule) -> None:
        for group in self.actor_optimizer.param_groups:
            group["lr"] = plan.actor_lr * group.get("lr_scale", 1.0)
        for group in self.critic_optimizer.param_groups:
            group["lr"] = plan.critic_lr

    def checkpoint_path(self) -> Path:
        return self.out / "checkpoint.pt"

    def save(self) -> None:
        state = {
            "config_sha256": self.config_sha256,
            "update": self.update,
            "env_steps": self.env_steps,
            "policy": self.policy.state_dict(),
            "critic": self.critic.state_dict(),
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "critic_optimizer": self.critic_optimizer.state_dict(),
            "update_rng": self.update_rng.get_state(),
            "collector": self.collector.checkpoint(),
        }
        temporary = self.checkpoint_path().with_suffix(".tmp")
        torch.save(state, temporary)
        temporary.replace(self.checkpoint_path())

    def prepare(self, resume: bool) -> None:
        self.out.mkdir(parents=True, exist_ok=True)
        config_path = self.out / "config.json"
        if resume:
            saved = torch.load(self.checkpoint_path(), map_location="cpu", weights_only=False)
            if saved["config_sha256"] != self.config_sha256:
                raise ValueError("--resume needs the same config, seed, host, device and init")
            self.policy.load_state_dict(saved["policy"])
            self.critic.load_state_dict(saved["critic"])
            self.actor_optimizer.load_state_dict(saved["actor_optimizer"])
            self.critic_optimizer.load_state_dict(saved["critic_optimizer"])
            self.update_rng.set_state(saved["update_rng"])
            self.update, self.env_steps = saved["update"], saved["env_steps"]
            self.collector = PPOCollector(
                self.policy, self.teacher, self.config, self.seed, self.device
            )
            self.collector.restore(saved["collector"])
        else:
            if self.checkpoint_path().exists():
                raise ValueError(f"{self.out} already has a checkpoint; use --resume")
            config_path.write_text(json.dumps(self.resolved, indent=2) + "\n", encoding="utf-8")
            self.collector = PPOCollector(
                self.policy, self.teacher, self.config, self.seed, self.device
            )

    def should_stop(self) -> bool:
        return (self.out / "STOP").exists()

    def run(self) -> None:
        config = self.config
        steps_per_update = config["rows"] * config["rollout_steps"]
        kl = self.settings.target_kl * config["kl_stop"]["factor"]

        def on_failure(reason: str, record: dict) -> None:
            (self.out / "failure.json").write_text(
                json.dumps({"reason": reason, "record": record}, indent=2) + "\n", encoding="utf-8"
            )

        with RunMonitor(
            self.out,
            # A non-finite gradient skips its optimizer step; failing here keeps a run that has
            # stopped learning from looking healthy (#188 review).
            # policy_loss and approx_kl as well: a NaN KL never counts toward the kl_stop
            # streak, since NaN > threshold is false (#188 review)
            finite_keys=(
                "value_loss",
                "critic_grad_norm",
                "actor_grad_norm",
                "policy_loss",
                "approx_kl",
            ),
            on_failure=on_failure,
            start_env_steps=self.env_steps,
        ) as monitor:
            monitor.watch("approx_kl", kl, config["kl_stop"]["patience"])
            while self.update < config["updates"]:
                plan = schedule(self.update, config)
                self._set_rates(plan)
                started = time.perf_counter()
                rollout, collected = self.collector.collect_ppo(
                    config["rollout_steps"], kickstart=steps_teacher(plan)
                )
                collect_s = time.perf_counter() - started
                metrics = ppo_update(
                    self.policy, self.critic, self.actor_optimizer, self.critic_optimizer,
                    rollout, self.settings, actor_on=plan.actor_on, kick_lambda=plan.kick_lambda,
                    micro_batch=config["micro_batch"], generator=self.update_rng,
                )  # fmt: skip
                del rollout
                self.update += 1
                self.env_steps += steps_per_update
                scalars = {
                    **collected,
                    **metrics,
                    "actor_lr": plan.actor_lr,
                    "critic_lr": plan.critic_lr,
                    "kick_lambda": plan.kick_lambda,
                    "actor_on": float(plan.actor_on),
                    "log_std_speed": self.policy.log_std[0].detach(),
                    "log_std_turn": self.policy.log_std[1].detach(),
                    "collect_s": collect_s,
                    "update_s": time.perf_counter() - started - collect_s,
                }
                if self.device.type == "cuda":
                    scalars["max_memory_allocated"] = float(
                        torch.cuda.max_memory_allocated(self.device)
                    )
                monitor.log_update(self.update, self.env_steps, scalars)
                if (
                    self.update % config["checkpoint_every"] == 0
                    or self.update % STOP_INTERVAL == 0
                ):
                    self.save()
                if self.should_stop():
                    self.save()
                    print(f"Stopped at update {self.update}; resume with --resume", flush=True)
                    return
        self.save()
        provenance = self.init_card["provenance"]
        export_policy(
            self.policy,
            self.out / "final",
            provenance={
                "git_sha": os.environ.get("GIT_COMMIT") or os.environ.get("GITHUB_SHA"),
                "dirty": None,
                "config_sha256": self.config_sha256,
                "stages": [*provenance["stages"], {"name": "ppo", **self.resolved}],
                "train_seed": self.seed,
            },
        )
        print(f"Finished {self.update} updates and {self.env_steps} environment steps", flush=True)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, default=CONFIG_DIR / "ppo.yaml")
    parser.add_argument("--init", type=Path, required=True, help="the Stage 1 artifact (final/)")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--host", required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--override", action="extend", nargs="+", metavar="KEY=VALUE", default=[])
    args = parser.parse_args(argv)
    if args.seed < 0:
        parser.error("--seed must be nonnegative")
    try:
        config = load_ppo_config(args.config, args.host, args.override)
        device = torch.device(args.device)
        if device.type == "cuda":
            if device.index is None:
                device = torch.device("cuda", torch.cuda.current_device())
            torch.cuda.set_device(device)
            torch.cuda.set_per_process_memory_fraction(1.0, device)
        torch.manual_seed(args.seed)
        run = PPORun(config, args.init, args.seed, device, args.out)
        run.prepare(args.resume)
        run.run()
    except EngineeringFailure as error:
        print(f"Engineering failure: {error}", flush=True)
        return error.exit_code
    except (ValueError, OSError) as error:
        parser.error(str(error))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
