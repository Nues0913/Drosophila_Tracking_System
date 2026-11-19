"""Recurrent imitation loss, accumulated gradients and held-out probes."""

from __future__ import annotations

import math
from copy import deepcopy
from dataclasses import dataclass

import numpy as np
import torch
from torch import Tensor

from flytrack_v2.interfaces import ACTION_EPS, ActionDistribution, reset_rows, unsquash
from flytrack_v2.model.policy import ConnectomePolicyV2

from .data import SequenceBatch


@dataclass(frozen=True)
class LossSettings:
    burn_in: int
    activity_coef: float
    label_margin: float | tuple[float, float]  # one margin, or (speed, turn)


def label_margins(config) -> tuple[float, float]:
    """(speed, turn) margins: from label_latent_bound b when set, so every label's latent is
    within +-b (speed v_max sigmoid(z), turn omega_max tanh(z), #184 Decision), else
    label_margin for both (configs before #189)."""
    bound = config.get("label_latent_bound")
    if bound is None:
        return config["label_margin"], config["label_margin"]
    return 1.0 / (1.0 + math.exp(bound)), 1.0 - math.tanh(bound)


def shrink_labels(
    action: Tensor, v_max: float, omega_max: float, margin: float | tuple[float, float]
) -> Tensor:
    """Keep labels away from the inverse squash limits for nonnegative speed and signed turn."""
    # A YAML label_margin may load as an int (e.g. 0); only a pair is (speed, turn).
    speed_margin, turn_margin = margin if isinstance(margin, (tuple, list)) else (margin, margin)
    turn_bound = (1 - turn_margin) * omega_max
    speed = action[..., 0].clamp(speed_margin * v_max, (1 - speed_margin) * v_max)
    turn = action[..., 1].clamp(-turn_bound, turn_bound)
    return torch.stack((speed, turn), dim=-1)


@torch.no_grad()
def label_latent_quantiles(
    action: Tensor, v_max: float, omega_max: float, margin: float | tuple[float, float]
) -> dict[str, float]:
    """Measure the latent targets over all collected labels in a round."""
    percentiles = (1, 10, 500, 990, 999)
    latent = unsquash(shrink_labels(action, v_max, omega_max, margin), v_max, omega_max)
    quantiles = np.quantile(latent.reshape(-1, 2).numpy(), [p / 1000 for p in percentiles], axis=0)
    return {
        f"label_latent_{name}_p{percentile:03d}": float(value)
        for percentile, pair in zip(percentiles, quantiles, strict=True)
        for name, value in zip(("speed", "turn"), pair, strict=True)
    }


@torch.no_grad()
def burn_in_state(policy, code: Tensor, reset: Tensor, state: Tensor, initial: Tensor) -> Tensor:
    state = state.detach()
    for tick, ended in zip(code, reset, strict=True):
        state = reset_rows(policy.forward_code(tick, state).central_state, initial, ended)
    return state


def _latent_log_prob_by_action(dist: ActionDistribution, latent: Tensor) -> Tensor:
    """Return the latent log probability for speed and turn separately."""
    log_std = dist.log_std.expand_as(dist.mean)
    z = (latent - dist.mean) * torch.exp(-log_std)
    return -0.5 * z.square() - log_std - 0.5 * math.log(2 * math.pi)


def sequence_loss(
    policy, batch: SequenceBatch, initial: Tensor, settings: LossSettings
) -> dict[str, Tensor]:
    burn_in = settings.burn_in
    state = burn_in_state(
        policy, batch.code[:burn_in], batch.reset[:burn_in], batch.initial, initial
    )
    graph = isinstance(policy, ConnectomePolicyV2)
    means, activity = [], []
    for code, ended in zip(batch.code[burn_in:], batch.reset[burn_in:], strict=True):
        if graph:
            step = policy.forward_code(code, state, non_readout_activity=True)
            activity.append(step.aux["activity_sq"])
        else:
            step = policy.forward_code(code, state)
            activity.append(step.central_state.square().mean(-1))
        means.append(step.dist.mean)
        state = reset_rows(step.central_state, initial, ended)

    dist = ActionDistribution(
        torch.stack(means), policy.log_std, policy.contract.v_max, policy.contract.omega_max
    )
    action = batch.action[burn_in:]
    speed, turn = action.unbind(-1)
    speed_low = speed <= ACTION_EPS * dist.v_max
    speed_high = speed >= (1 - ACTION_EPS) * dist.v_max
    turn_saturated = turn.abs() >= (1 - ACTION_EPS) * dist.omega_max
    saturation = speed_low | speed_high | turn_saturated
    labels = shrink_labels(action, dist.v_max, dist.omega_max, settings.label_margin)
    latent = unsquash(labels, dist.v_max, dist.omega_max)
    log_prob = _latent_log_prob_by_action(dist, latent)
    nll = -log_prob.sum(-1)
    nll_speed, nll_turn = -log_prob.mean(dim=(0, 1))
    penalty = settings.activity_coef * torch.stack(activity).mean()
    return {
        "loss": nll.mean() + penalty,
        "nll": nll.mean(),
        "nll_speed": nll_speed,
        "nll_turn": nll_turn,
        "activity_penalty": penalty,
        "saturation_fraction": saturation.float().mean(),
        "saturation_fraction_speed_low": speed_low.float().mean(),
        "saturation_fraction_speed_high": speed_high.float().mean(),
        "saturation_fraction_turn": turn_saturated.float().mean(),
        "saturated_nll": torch.where(saturation, nll.detach(), 0.0).mean(),
    }


def _micro_batches(batch: SequenceBatch, size: int):
    for i in range(0, batch.initial.size(0), size):
        rows = slice(i, i + size)
        yield SequenceBatch(
            batch.code[:, rows],
            batch.action[:, rows],
            batch.reset[:, rows],
            batch.initial[rows],
            batch.tags[rows],
            batch.occluded[:, rows],
        )


def batch_metrics(
    policy, batch: SequenceBatch, settings: LossSettings, micro_batch: int, *, backward: bool
) -> dict[str, Tensor]:
    device = policy.log_std.device
    total = batch.initial.size(0)
    initial = policy.initial_central_state(1, device)
    metrics = {}
    for part in _micro_batches(batch.to(device), micro_batch):
        fraction = part.initial.size(0) / total
        values = sequence_loss(policy, part, initial, settings)
        if backward:
            (values["loss"] * fraction).backward()
        for name, value in values.items():
            weighted = value.detach() * fraction
            metrics[name] = metrics[name] + weighted if name in metrics else weighted
    return metrics


def first_batch_gradient_norms(
    policy: ConnectomePolicyV2,
    original_readout: dict[str, Tensor],
    batch: SequenceBatch,
    settings: LossSettings,
    micro_batch: int,
) -> dict:
    """Probe the two NLL terms on the same core without populating parameter gradients."""
    parts = tuple(_micro_batches(batch.to(policy.log_std.device), micro_batch))
    initial = policy.initial_central_state(1, policy.log_std.device)
    groups = {
        "g": policy.core.gain_theta,
        "tau": policy.core.tau_theta,
        "bias": policy.core.bias,
        "lc_rf": policy.projection.lc_rf,
        "missing_bias": policy.projection.missing_bias,
    }
    active = [param for param in groups.values() if param.requires_grad]
    total = batch.initial.size(0)

    def component_norms(component: str) -> dict[str, float]:
        accumulated = {param: torch.zeros_like(param) for param in active}
        for part in parts:
            nll = sequence_loss(policy, part, initial, settings)[component]
            grads = torch.autograd.grad(
                nll * (part.initial.size(0) / total), active, allow_unused=True
            )
            for param, grad in zip(active, grads, strict=True):
                if grad is not None:
                    accumulated[param].add_(grad)
        norms = torch.stack(
            [
                accumulated[param].norm() if param in accumulated else param.new_zeros(())
                for param in groups.values()
            ]
        )
        return dict(
            zip((*groups, "total"), torch.cat((norms, norms.norm()[None])).tolist(), strict=True)
        )

    def measure() -> dict:
        speed, turn = component_norms("nll_speed"), component_norms("nll_turn")
        return {
            group: {
                "speed": speed[group] if math.isfinite(speed[group]) else None,
                "turn": turn[group] if math.isfinite(turn[group]) else None,
                "speed_to_turn": (
                    speed[group] / turn[group]
                    if turn[group] > 0 and math.isfinite(turn[group] + speed[group])
                    else None
                ),
            }
            for group in speed
        }

    fitted = policy.readout
    original = deepcopy(fitted)
    original.load_state_dict(original_readout)
    normalized = measure()
    try:
        policy.readout = original
        unnormalized = measure()
    finally:
        policy.readout = fitted
    return {"normalized": normalized, "unnormalized": unnormalized}


def make_scheduler(optimizer, config: dict, updates: int):
    final_ratio = config["lr_final"] / config["lr"]

    def scale(index):
        progress = min(index, updates - 1) / max(1, updates - 1)
        return final_ratio + (1 - final_ratio) * 0.5 * (1 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, scale)


def train_update(
    policy,
    optimizer,
    scheduler,
    batch: SequenceBatch,
    settings: LossSettings,
    micro_batch: int,
    grad_clip: float,
) -> dict[str, Tensor | float]:
    lr = optimizer.param_groups[0]["lr"]
    optimizer.zero_grad(set_to_none=True)
    metrics = batch_metrics(policy, batch, settings, micro_batch, backward=True)
    parameters = [
        p for group in optimizer.param_groups for p in group["params"] if p.grad is not None
    ]
    grad_norm = torch.nn.utils.get_total_norm([parameter.grad for parameter in parameters])
    finite = torch.stack([grad_norm, metrics["loss"], metrics["nll"], metrics["activity_penalty"]])
    if bool(torch.isfinite(finite).all()):
        torch.nn.utils.clip_grads_with_norm_(parameters, grad_clip, grad_norm)
        optimizer.step()
        scheduler.step()
    saturation_nll = metrics.pop("saturated_nll")
    nll = metrics["nll"]
    device = policy.log_std.device
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    return {
        **metrics,
        "saturated_nll_fraction": torch.where(nll != 0, saturation_nll / nll, 0.0),
        "grad_norm": grad_norm,
        "lr": lr,
        "log_std_speed": policy.log_std[0].detach(),
        "log_std_turn": policy.log_std[1].detach(),
        "max_memory_allocated": (
            torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0
        ),
    }


@torch.no_grad()
def health_metrics(
    policy, batch: SequenceBatch, burn_in: int, micro_batch: int
) -> dict[str, Tensor]:
    """Compare mode actions from the same burn-in state, with and without swapping odor."""
    device = policy.log_std.device
    initial = policy.initial_central_state(1, device)
    eye, _, odor, _ = policy.input_norm.layout.widths
    left = slice(2 * eye, 2 * eye + odor // 2)
    right = slice(left.stop, left.stop + odor // 2)
    speeds, differences, dn_rates = [], [], []
    for part in _micro_batches(batch.to(device), micro_batch):
        state = burn_in_state(
            policy, part.code[:burn_in], part.reset[:burn_in], part.initial, initial
        )
        swapped_state = state
        for code, ended in zip(part.code[burn_in:], part.reset[burn_in:], strict=True):
            swapped = code.clone()
            swapped[:, left], swapped[:, right] = code[:, right], code[:, left]
            normal = policy.forward_code(code, state)
            swapped_step = policy.forward_code(swapped, swapped_state)
            action, alternate = normal.dist.mode(), swapped_step.dist.mode()
            speeds.append(action[:, 0])
            differences.append((action[:, 1] - alternate[:, 1]).abs())
            if "dn_rates" in normal.aux:
                dn_rates.append(normal.aux["dn_rates"])
            state = reset_rows(normal.central_state, initial, ended)
            swapped_state = reset_rows(swapped_step.central_state, initial, ended)
    metrics = {
        "health_speed_std_m_s": torch.cat(speeds).std(correction=0),
        "health_odor_swap_delta_omega": torch.cat(differences).mean(),
    }
    if dn_rates:
        rates = torch.cat(dn_rates)
        dn_mean = rates.mean(0)
        metrics["health_dn_min_mean_rate"] = dn_mean.min()
        metrics["health_dn_silent"] = (~(dn_mean > 1e-4)).sum()
        if bool(policy.readout.rate_norm_fitted):
            drift = policy.readout.normalize_rates(rates).abs()
            metrics["readout_drift_mean_abs"] = drift.mean()
            metrics["readout_drift_max_abs_per_dn"] = drift.amax(0)
    return metrics
