"""Resolve DAgger settings, host settings and the integer rollout schedule."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, fields
from fractions import Fraction
from pathlib import Path

import yaml

from flytrack_v2.model.config import ModelConfigV2

ROOT = Path(__file__).resolve().parents[3]
CONFIG_DIR = ROOT / "configs/v2/train"
WORLD_NAMES = ("v1compat", "wind")
MOTION_NAMES = ("curved", "wander")
SCENARIO_NAMES = ("normal", "occlusion", "noise", "conflict")
STATE_STRIDE = 16
STOP_INTERVAL = 50
_TRAINER_MODEL_FIELDS = {"init_seed", "structure_seed", "activation_checkpoint"}
_MODEL_FIELDS = {field.name for field in fields(ModelConfigV2)}


def read_yaml(path: Path) -> dict:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a mapping")
    return value


def scenario_rows(weights: dict[str, float], rows: int) -> list[str]:
    """Allocate fixed rows by largest remainder, breaking ties in SCENARIO_NAMES order."""
    if type(rows) is not int or rows < 1 or set(weights) != set(SCENARIO_NAMES):
        raise ValueError("Expected a positive row count and the four DAgger scenarios")
    fractions = [Fraction(str(weights[name])) for name in SCENARIO_NAMES]
    if min(fractions) < 0 or sum(fractions) != 1:
        raise ValueError("Scenario weights must be nonnegative and sum to 1")
    quotas = [rows * weight for weight in fractions]
    counts = [int(quota) for quota in quotas]
    order = sorted(range(len(counts)), key=lambda i: quotas[i] - counts[i], reverse=True)
    for i in order[: rows - sum(counts)]:
        counts[i] += 1
    return [name for name, count in zip(SCENARIO_NAMES, counts, strict=True) for _ in range(count)]


@dataclass(frozen=True)
class RoundPlan:
    beta: float
    ticks: int
    first_snapshot: int
    updates: int


def plan_rounds(config: dict) -> list[RoundPlan]:
    rows = len(WORLD_NAMES) * len(MOTION_NAMES) * config["envs_per_instance"]
    length = config["sequence"]["length"]
    window = config["sequence"]["burn_in"] + length
    ticks_requested = math.ceil(config["rounds"]["env_steps_per_round"] / rows)
    plans = []
    elapsed = 0
    for beta in config["rounds"]["beta"]:
        first_snapshot = -elapsed % STATE_STRIDE
        ticks = max(ticks_requested, first_snapshot + window)
        updates = math.ceil(
            config["replay_ratio"] * ticks * rows / (config["batch_sequences"] * length)
        )
        plans.append(RoundPlan(beta, ticks, first_snapshot, updates))
        elapsed += ticks
    return plans


def _validate_model_key(key: str) -> None:
    if key in _TRAINER_MODEL_FIELDS:
        raise ValueError(f"model.{key} is controlled by the trainer")
    if key not in _MODEL_FIELDS:
        raise ValueError(f"Unknown model config key {key!r}")


def validate_model_overrides(model: Mapping) -> None:
    if not isinstance(model, Mapping):
        raise ValueError("DAgger setting model must be a mapping")
    for key in model:
        _validate_model_key(key)


def _override(config: dict, expression: str) -> None:
    key, separator, raw = expression.partition("=")
    parts = key.split(".")
    if separator and len(parts) == 2 and parts[0] == "model":
        _validate_model_key(parts[1])
        config["model"][parts[1]] = yaml.safe_load(raw)
        return
    node = config
    for part in parts[:-1]:
        if not isinstance(node.get(part), dict):
            raise ValueError(f"Unknown DAgger setting {key!r}")
        node = node[part]
    if not separator or parts[-1] not in node or isinstance(node[parts[-1]], dict):
        raise ValueError(f"Expected an existing setting and YAML value, got {expression!r}")
    node[parts[-1]] = yaml.safe_load(raw)


def _integer(value, name: str, minimum: int = 1) -> None:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")


def _number(value, name: str, low: float, high: float = math.inf, *, inclusive=False) -> None:
    if (
        type(value) not in (float, int)
        or not math.isfinite(value)
        or not (low <= value <= high if inclusive else low < value < high)
    ):
        raise ValueError(f"{name} is outside its allowed range")


def load_config(path: Path, host: str, overrides: list[str] | None = None) -> dict:
    defaults = read_yaml(CONFIG_DIR / "dagger.yaml")
    config = read_yaml(path)
    config.setdefault("desync_first_episode", False)
    config.setdefault("readout_normalization", defaults["readout_normalization"])
    # Configs written before #184 labelled around the applied action.
    config.setdefault("teacher_limit_acceleration", True)
    config.setdefault("model", {})
    if set(config) != set(defaults):
        raise ValueError("DAgger config must have the same keys as dagger.yaml")
    for key, default in defaults.items():
        if (
            key != "model"
            and isinstance(default, dict)
            and (not isinstance(config[key], dict) or set(config[key]) != set(default))
        ):
            raise ValueError(f"DAgger setting {key} must have keys {', '.join(default)}")
    validate_model_overrides(config["model"])
    for expression in overrides or ():
        _override(config, expression)

    for key, names in (("worlds", WORLD_NAMES), ("motions", MOTION_NAMES)):
        if config[key] != dict.fromkeys(names, 0.5):
            raise ValueError(f"{key} must use equal halves with one world instance per combination")
    _integer(config["envs_per_instance"], "envs_per_instance")
    scenario_rows(config["scenarios"], config["envs_per_instance"])
    _number(config["wind_speed_m_s"], "wind_speed_m_s", 0)
    _number(config["episode_seconds"], "episode_seconds", 0)
    if config["plume_model"] != "puff":
        raise ValueError("The DAgger world requires the puff plume")
    beta = config["rounds"]["beta"]
    if not isinstance(beta, list) or not beta:
        raise ValueError("rounds.beta must be a nonempty list")
    for value in beta:
        _number(value, "beta", 0, 1, inclusive=True)
    _integer(config["rounds"]["env_steps_per_round"], "rounds.env_steps_per_round")
    _number(config["replay_ratio"], "replay_ratio", 0)
    _integer(config["sequence"]["length"], "sequence.length")
    _integer(config["sequence"]["burn_in"], "sequence.burn_in", 0)
    _integer(config["batch_sequences"], "batch_sequences")
    optimizer = config["optimizer"]
    if optimizer["name"] != "adam" or optimizer["schedule"] != "cosine_over_all_updates":
        raise ValueError("DAgger uses Adam with cosine_over_all_updates")
    for key in ("lr", "lr_final", "grad_clip"):
        _number(optimizer[key], f"optimizer.{key}", 0)
    if optimizer["lr_final"] > optimizer["lr"]:
        raise ValueError("optimizer.lr_final must be <= optimizer.lr")
    _number(config["activity_coef"], "activity_coef", 0, math.inf, inclusive=True)
    _number(config["label_margin"], "label_margin", 0, 1)
    if config["label_margin"] > 0.5:
        raise ValueError("label_margin must be <= 0.5 to leave a valid speed interval")
    _integer(config["held_out"]["every_updates"], "held_out.every_updates")
    _integer(config["held_out"]["sequences"], "held_out.sequences")
    _number(config["held_out"]["fraction"], "held_out.fraction", 0, 1)
    for key in (
        "activation_checkpoint",
        "desync_first_episode",
        "readout_normalization",
        "teacher_limit_acceleration",
    ):
        if type(config[key]) is not bool:
            raise ValueError(f"{key} must be boolean")

    hosts = read_yaml(CONFIG_DIR / "hosts.yaml")
    if host not in hosts:
        raise ValueError(f"Unknown host {host!r}, choose {', '.join(hosts)}")
    host_micro_batch = hosts[host]["micro_batch"]
    _integer(host_micro_batch, f"{host}.micro_batch")
    micro_batch = min(host_micro_batch, config["batch_sequences"])
    if config["batch_sequences"] % micro_batch:
        raise ValueError("batch_sequences must be divisible by micro_batch")
    return {
        **config,
        "host": host,
        "host_micro_batch": host_micro_batch,
        "micro_batch": micro_batch,
        "state_stride": STATE_STRIDE,
        "stop_interval": STOP_INTERVAL,
    }
