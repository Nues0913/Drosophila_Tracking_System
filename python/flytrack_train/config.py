"""Configuration and reproducibility at the command boundary."""

from __future__ import annotations

import hashlib
import json
import os
import random
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from flytrack_core.contracts import SensorContract, file_sha256
from flytrack_core.model import (
    build_policy,
    load_policy_artifact,
    save_policy_artifact,
    transplant_pretrained_modalities,
)
from flytrack_core.model.policy import DEFAULT_CONFIG, ConnectomePolicy

from .worlds import make_env  # noqa: F401 - re-exported for trainers

TRAIN_ENVIRONMENT_SEED_LIMIT = 1_000_000_000
TRAIN_ENVIRONMENT_SEED_POLICY = {
    "minimum_inclusive": 0,
    "maximum_exclusive": TRAIN_ENVIRONMENT_SEED_LIMIT,
    "purpose": "training only; held-out calibration/evaluation use separate seed ranges",
}


def project_source_fingerprint() -> dict[str, Any]:
    """Hash executable project code only; never inspect user configuration/secrets."""
    root = Path(__file__).resolve().parents[2]
    paths = []
    for package in ("flytrack_core", "flytrack_train", "flytrack_sim"):
        paths.extend((root / "python" / package).rglob("*.py"))
    paths.extend((root / "scripts").glob("*.py"))
    files = {
        str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(paths)
    }
    digest = hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest()
    return {"sha256": digest, "files": files}


def read_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open(encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    if not isinstance(config, dict):
        raise ValueError("Configuration must be a YAML mapping")
    return config


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def contract_from_config(config: dict[str, Any]) -> SensorContract:
    if "sensor_contract_path" in config:
        path = Path(config["sensor_contract_path"])
        with path.open(encoding="utf-8") as stream:
            data = yaml.safe_load(stream) if path.suffix in {".yml", ".yaml"} else json.load(stream)
        return SensorContract.from_dict(data)
    return SensorContract.from_dict(config.get("sensor_contract", {}))


def prepare_policy(config: dict[str, Any], device: torch.device):
    contract = contract_from_config(config)
    if config.get("init_artifact"):
        policy, loaded_contract, initial_card = load_policy_artifact(
            config["init_artifact"], device=str(device)
        )
        policy._training_ledger = list(
            initial_card.get("metadata", {}).get("stage_budget_ledger", [])
        )
        if loaded_contract.sha256() != contract.sha256():
            raise ValueError("Initial artifact sensor contract differs from training configuration")
        if policy.variant != config.get("variant", "VO"):
            raise ValueError("Initial artifact variant differs from training configuration")
        if (
            file_sha256(config["graph_path"]) != policy.graph_sha256
            or file_sha256(config["frontend_path"]) != policy.frontend_sha256
        ):
            raise ValueError(
                "Initial artifact graph/frontend differs from configured training sources"
            )
        if isinstance(policy, ConnectomePolicy) and "motor_rate_beta" in config.get("model", {}):
            # Explicit warm-start migration; ordinary loading preserves old card semantics.
            policy.core.motor_rate_beta = config["model"]["motor_rate_beta"]
            policy.model_config["motor_rate_beta"] = policy.core.motor_rate_beta
        model = config.get("model", {})
        if "odor_shared_adaptation" in model:
            # Explicit warm-start migration (#4 P2): the ORN encoder is fixed, not learned.
            policy.encoder.shared_adaptation = bool(model["odor_shared_adaptation"])
            policy.model_config["odor_shared_adaptation"] = policy.encoder.shared_adaptation
        if isinstance(policy, ConnectomePolicy) and model.get("readout_extra_turn_types"):
            # Explicit warm-start migration (#4 B3): new rows, calibrated on this actor, start
            # at zero weight, so the loaded behavior is unchanged until training moves them.
            requested = {
                key: model.get(key, DEFAULT_CONFIG[key])
                for key in (
                    "readout_extra_turn_types",
                    "readout_extra_turn_center",
                    "readout_extra_turn_scale",
                    "readout_extra_turn_eps",
                )
            }
            if policy.readout.extra_turn_types and any(
                policy.model_config[key] != value for key, value in requested.items()
            ):
                # Silently recalibrating trained rows would change the loaded behavior.
                raise ValueError("Initial artifact already has different extra turn rows")
            policy.set_readout_extra_turns(*requested.values())
    else:
        policy = build_policy(
            config.get("variant", "VO"),
            config["graph_path"],
            contract,
            frontend_path=config.get("frontend_path"),
            **config.get("model", {}),
        ).to(device)
    pretrained = config.get("pretrained_artifacts", {})
    if pretrained:
        if config.get("init_artifact"):
            raise ValueError("Use init_artifact or pretrained_artifacts, not both")
        transplant_pretrained_modalities(
            policy, vision_artifact=pretrained.get("vision"), odor_artifact=pretrained.get("odor")
        )
        policy._training_ledger = []
        for path in dict.fromkeys(pretrained.values()):
            _, _, card = load_policy_artifact(path, device="cpu")
            policy._training_ledger.extend(card.get("metadata", {}).get("stage_budget_ledger", []))
    return policy, contract


def save_actor(
    directory: Path, policy, config: dict[str, Any], contract: SensorContract, **metadata: Any
) -> None:
    entry = {
        key: metadata[key]
        for key in ("stage", "algorithm", "environment_steps", "modality")
        if key in metadata
    }
    ledger = [*getattr(policy, "_training_ledger", []), entry]
    metadata["stage_budget_ledger"] = ledger
    metadata["total_environment_steps"] = sum(
        int(row.get("environment_steps", 0)) for row in ledger
    )
    metadata["config"] = config
    metadata["environment_seed_policy"] = TRAIN_ENVIRONMENT_SEED_POLICY
    entry["environment_seed_policy"] = TRAIN_ENVIRONMENT_SEED_POLICY
    metadata["source_code"] = project_source_fingerprint()
    entry["source_code_sha256"] = metadata["source_code"]["sha256"]
    save_policy_artifact(
        directory,
        policy,
        policy.model_config,
        contract,
        config["graph_path"],
        frontend_path=config.get("frontend_path"),
        metadata=metadata,
    )


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    # NaN is not portable JSON: censored/unobserved metrics become explicit nulls.
    def clean(value):
        if isinstance(value, dict):
            return {str(k): clean(v) for k, v in value.items()}
        if isinstance(value, (tuple, list)):
            return [clean(v) for v in value]
        if isinstance(value, np.ndarray):
            return clean(value.tolist())
        if isinstance(value, np.generic):
            return clean(value.item())
        if isinstance(value, float) and not np.isfinite(value):
            return None
        return value

    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(
                json.dumps(clean(data), ensure_ascii=False, indent=2, allow_nan=False) + "\n"
            )
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)
