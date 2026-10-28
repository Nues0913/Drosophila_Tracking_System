"""Shared actor optimizer groups and their effective parameter budget (#77)."""

from __future__ import annotations

from typing import Any

import torch
from flytrack_core.model import count_trainable_parameters


def parameter_groups(policy, train: dict[str, Any]) -> list[dict[str, Any]]:
    """Adam groups with optional core-bias, readout, and readout-bias rates.

    ``readout_bias_learning_rate`` matches names starting with ``readout.`` and
    ending with ``_bias``; it overrides ``readout_learning_rate``, then ``learning_rate``.
    ``core_bias_learning_rate`` independently overrides the rate for ``core.bias*``.
    A zero rate freezes that group. Without overrides this is one group with every parameter.
    """
    base = float(train.get("learning_rate", 1e-3))
    rates = {
        "core.bias": train.get("core_bias_learning_rate"),
        "readout.": train.get("readout_learning_rate"),
    }
    readout_bias_rate = train.get("readout_bias_learning_rate")
    if any(rate is not None and float(rate) < 0 for rate in (*rates.values(), readout_bias_rate)):
        raise ValueError("Group learning rates must be nonnegative")
    groups: dict[float, list[torch.nn.Parameter]] = {}
    for name, parameter in policy.named_parameters():
        rate = next(
            (r for prefix, r in rates.items() if r is not None and name.startswith(prefix)), base
        )
        if readout_bias_rate is not None and name.startswith("readout.") and name.endswith("_bias"):
            rate = readout_bias_rate
        groups.setdefault(float(rate), []).append(parameter)
    return [{"params": params, "lr": rate} for rate, params in groups.items()]


def count_effective_parameters(policy, train: dict[str, Any]) -> dict[str, Any]:
    """Count actor capacity using the optimizer's actual grouping function.

    All counts include only requires_grad parameters: declared is total capacity,
    zero_lr is the frozen subset at lr == 0, effective is the subset at lr > 0.
    Frozen frontend weights and auxiliary decoders/critics are excluded.
    """
    names = {id(parameter): name for name, parameter in policy.named_parameters()}
    groups, zero_lr_parameters = [], []
    zero_lr = effective = 0
    for group in parameter_groups(policy, train):
        parameters = [
            {"name": names[id(parameter)], "numel": parameter.numel()}
            for parameter in group["params"]
            if parameter.requires_grad
        ]
        numel = sum(parameter["numel"] for parameter in parameters)
        rate = float(group["lr"])
        groups.append({"lr": rate, "numel": numel, "parameters": parameters})
        if rate == 0:
            zero_lr += numel
            zero_lr_parameters.extend(parameters)
        elif rate > 0:
            effective += numel
    return {
        "declared": count_trainable_parameters(policy),
        "zero_lr": zero_lr,
        "effective": effective,
        "groups": groups,
        "zero_lr_parameters": zero_lr_parameters,
    }
