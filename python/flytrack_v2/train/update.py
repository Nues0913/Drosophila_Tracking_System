"""Recurrent imitation loss, accumulated gradients and held-out probes."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor

from flytrack_v2.interfaces import ActionDistribution, reset_rows, unsquash
from flytrack_v2.model.policy import ConnectomePolicyV2

from .data import SequenceBatch


@dataclass(frozen=True)
class LossSettings:
    burn_in: int
    activity_coef: float
    label_margin: float


@torch.no_grad()
def burn_in_state(policy, code: Tensor, reset: Tensor, state: Tensor, initial: Tensor) -> Tensor:
    state = state.detach()
    for tick, ended in zip(code, reset, strict=True):
        state = reset_rows(policy.forward_code(tick, state).central_state, initial, ended)
    return state


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
    limits = action.new_tensor((dist.v_max, dist.omega_max))
    saturation = (action.abs() >= (1 - 1e-6) * limits).any(-1)
    bound = (1 - settings.label_margin) * limits
    latent = unsquash(action.clamp(-bound, bound), dist.v_max, dist.omega_max)
    nll = -dist.log_prob(latent)
    penalty = settings.activity_coef * torch.stack(activity).mean()
    return {
        "loss": nll.mean() + penalty,
        "nll": nll.mean(),
        "activity_penalty": penalty,
        "saturation_fraction": saturation.float().mean(),
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
        dn_mean = torch.cat(dn_rates).mean(0)
        metrics["health_dn_min_mean_rate"] = dn_mean.min()
        metrics["health_dn_silent"] = (~(dn_mean > 1e-4)).sum()
    return metrics
