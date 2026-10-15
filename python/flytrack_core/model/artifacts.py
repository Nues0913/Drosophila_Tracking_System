"""Portable, hash-checked model artifacts and modality-specific warm starts."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from flytrack_core.contracts import SensorContract
from flytrack_core.sensing.hex_eye import compute_directions

from .graph import Graph
from .policy import ConnectomePolicy, build_policy, count_trainable_parameters


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def direction_table_sha256(contract: SensorContract) -> str:
    table = np.asarray(compute_directions(contract), dtype="<f8", order="C")
    return hashlib.sha256(table.tobytes()).hexdigest()


def _atomic_json(path: Path, content: Any) -> None:
    fd, temporary = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(content, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
        os.replace(temporary, path)
    finally:
        if Path(temporary).exists():
            Path(temporary).unlink()


def _atomic_copy(source: Path, target: Path) -> None:
    if source.resolve() == target.resolve():
        return
    fd, temporary = tempfile.mkstemp(dir=target.parent, prefix=f".{target.name}.")
    os.close(fd)
    try:
        shutil.copyfile(source, temporary)
        os.replace(temporary, target)
    finally:
        if Path(temporary).exists():
            Path(temporary).unlink()


def save_policy_artifact(
    directory: str | Path,
    policy: nn.Module,
    model_config: dict[str, Any],
    contract: SensorContract,
    graph_path: str | Path,
    frontend_path: str | Path | None = None,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Commit card last; incomplete/interleaved artifacts fail hash validation.

    policy.model_config is authoritative because auto readout and GRU width are
    resolved at construction. model_config records the caller's requested config.
    """
    if frontend_path is None:
        raise ValueError("Saving a deployable artifact requires its authentic frontend file")
    if contract.sha256() != policy.contract.sha256():
        raise ValueError("Policy and artifact sensor contracts differ")
    if (
        file_sha256(graph_path) != policy.graph_sha256
        or file_sha256(frontend_path) != policy.frontend_sha256
    ):
        raise ValueError(
            "Artifact graph/frontend must be the exact sources used to construct this policy"
        )
    graph = Graph.load(graph_path)
    if graph.n_nodes != policy.n_nodes or not torch.equal(
        torch.as_tensor(graph.node_ids), policy.node_ids.detach().cpu()
    ):
        raise ValueError("Artifact graph does not match policy nodes")
    destination = Path(directory)
    destination.mkdir(parents=True, exist_ok=True)
    _atomic_copy(Path(graph_path), destination / "graph.npz")
    _atomic_copy(Path(frontend_path), destination / "frontend.npz")
    _atomic_json(destination / "sensor_contract.json", contract.to_dict())
    fd, temporary = tempfile.mkstemp(dir=destination, prefix=".weights.")
    os.close(fd)
    try:
        # Only tensors and primitive containers are serialized, never nn.Module.
        torch.save(
            {key: tensor.detach().cpu() for key, tensor in policy.state_dict().items()}, temporary
        )
        os.replace(temporary, destination / "weights.pt")
    finally:
        if Path(temporary).exists():
            Path(temporary).unlink()
    files = ("graph.npz", "frontend.npz", "sensor_contract.json", "weights.pt")
    card = {
        "schema_version": 1,
        "variant": policy.variant,
        "runtime_lesion": policy.lesion,
        "model_config": policy.model_config,
        "requested_model_config": model_config,
        "sensor_contract_sha256": contract.sha256(),
        "direction_table_sha256": direction_table_sha256(contract),
        "control_dt": contract.control_dt,
        "neural_dt": contract.neural_dt,
        "history_steps": contract.history_steps,
        "state_size": policy.state_size,
        "n_nodes": policy.n_nodes,
        "trainable_parameters": count_trainable_parameters(policy),
        "files": {name: file_sha256(destination / name) for name in files},
        "frontend_source": policy.frontend.metadata,
        "observation_keys": ["vision", "odor", "valid"],
        "action_units": ["m/s", "rad/s"],
        "missing_sensor_policy": (
            "mask input and freeze corresponding frontend/ORN state; "
            "learned seed-only missing current"
        ),
        "visual_projection": (
            "same-side evenly allocated angular pooling bins; "
            "configurable engineering assumption, no measured RF"
        ),
        "shared_sensory_entry": (
            "all variants receive the same fixed LC/ORN seed projection; "
            "GRU receives seed currents plus valid flags, never unprojected frontend channels"
        ),
        "single_modality_ablation": (
            "shared full central graph and parameter count; "
            "other modality input and valid mask suppressed"
        ),
        "state_semantics": (
            "persistent recurrent state; history_steps is burn-in/storage "
            "and is not replayed each control step"
        ),
        "metadata": metadata or {},
    }
    if isinstance(policy, ConnectomePolicy):
        card["readout"] = policy.readout.mode
        card["rs_accepted_swaps"] = policy.core.random_swaps
        card["rs_sharing"] = (
            "original type-pair labels retained on edge slots" if policy.variant == "RS" else None
        )
        card["late_fusion"] = (
            "two independent states with shared type parameters; DN concatenation only"
            if policy.variant == "VO-late"
            else None
        )
    else:
        card["activity_semantics"] = (
            "repeated diagnostic GRU hidden channels; no biological node correspondence"
        )
    _atomic_json(destination / "model_card.json", card)
    return card


def load_policy_artifact(
    directory: str | Path, device: str | torch.device = "cpu"
) -> tuple[nn.Module, SensorContract, dict[str, Any]]:
    source = Path(directory)
    card = json.loads((source / "model_card.json").read_text(encoding="utf-8"))
    if card.get("schema_version") != 1:
        raise ValueError("Unsupported policy artifact schema")
    required = {"graph.npz", "frontend.npz", "sensor_contract.json", "weights.pt"}
    if set(card.get("files", {})) != required:
        raise ValueError("Artifact card must identify the four exact required files")
    for name, expected in card["files"].items():
        if file_sha256(source / name) != expected:
            raise ValueError(f"Artifact checksum mismatch: {name}")
    contract = SensorContract.from_dict(
        json.loads((source / "sensor_contract.json").read_text(encoding="utf-8"))
    )
    if contract.sha256() != card["sensor_contract_sha256"]:
        raise ValueError("Artifact sensor contract checksum mismatch")
    if direction_table_sha256(contract) != card["direction_table_sha256"]:
        raise ValueError("Artifact ommatidium direction table differs from current geometry code")
    for key, value in (
        ("control_dt", contract.control_dt),
        ("neural_dt", contract.neural_dt),
        ("history_steps", contract.history_steps),
    ):
        if card[key] != value:
            raise ValueError(f"Artifact card disagrees with sensor contract: {key}")
    policy = build_policy(
        card["variant"],
        source / "graph.npz",
        contract,
        source / "frontend.npz",
        **card["model_config"],
    )
    weights = torch.load(source / "weights.pt", map_location="cpu", weights_only=True)
    if not isinstance(weights, dict) or not all(
        isinstance(key, str) and isinstance(value, torch.Tensor) for key, value in weights.items()
    ):
        raise ValueError("Artifact weights must be a plain tensor state_dict")
    if any(not torch.isfinite(value).all() for value in weights.values()):
        raise ValueError("Artifact contains nonfinite learned weights")
    policy.load_state_dict(weights, strict=True)
    policy.set_lesion(card.get("runtime_lesion", "none"))
    if (
        policy.state_size != card["state_size"]
        or count_trainable_parameters(policy) != card["trainable_parameters"]
    ):
        raise ValueError(
            "Artifact shape or parameter count differs from reconstructed architecture"
        )
    policy.to(device).eval()
    policy.artifact_card = card
    return policy, contract, card


def transplant_pretrained_modalities(
    policy: nn.Module,
    vision_artifact: str | Path | None = None,
    odor_artifact: str | Path | None = None,
) -> None:
    """Transfer shared type parameters; average overlaps instead of overwriting.

    Motor readouts and PPO log_std remain initialized for the destination task.
    GRU requires its own modality pretraining; type weights cannot be transplanted.
    """
    if not isinstance(policy, ConnectomePolicy):
        raise ValueError(
            "Modality type-parameter transplantation is limited to connectome-family policies"
        )
    sources: list[tuple[str, nn.Module, dict[str, Any]]] = []
    for modality, directory in (("visual", vision_artifact), ("olfactory", odor_artifact)):
        if directory is None:
            continue
        source, _, card = load_policy_artifact(directory, device=policy.log_std.device)
        if not isinstance(source, ConnectomePolicy):
            raise ValueError("Warm-start source must be a connectome-family artifact")
        if (
            source.contract.sha256() != policy.contract.sha256()
            or source.frontend.metadata != policy.frontend.metadata
        ):
            raise ValueError("Warm-start sensor/frontend contracts differ")
        if (
            source.graph_sha256 != policy.graph_sha256
            or source.frontend_sha256 != policy.frontend_sha256
            or not torch.equal(source.node_ids, policy.node_ids)
            or not torch.equal(source.core.type_index, policy.core.type_index)
        ):
            raise ValueError("Warm-start sources must use the identical annotated central graph")
        sources.append((modality, source, card))
    if not sources:
        return
    with torch.no_grad():
        for destination_name in ("raw_alpha", "raw_tau", "bias", "missing_bias"):
            destination = (
                policy.missing_bias
                if destination_name == "missing_bias"
                else getattr(policy.core, destination_name)
            )
            total, counts = torch.zeros_like(destination), torch.zeros_like(destination)
            for modality, source, _ in sources:
                type_mask = torch.zeros_like(policy.core.bias, dtype=torch.bool)
                type_mask[policy.core.type_index[policy.population_masks[modality]]] = True
                if destination_name == "raw_alpha":
                    mask = (
                        type_mask[policy.core.pair_types[:, 0]]
                        & type_mask[policy.core.pair_types[:, 1]]
                    )
                    source_index = {
                        tuple(pair): index
                        for index, pair in enumerate(source.core.pair_types.detach().cpu().tolist())
                    }
                    matched = [
                        source_index.get(tuple(pair), -1)
                        for pair in policy.core.pair_types.detach().cpu().tolist()
                    ]
                    index = torch.tensor(matched, dtype=torch.long, device=destination.device)
                    mask = mask & (index >= 0)
                    value = source.core.raw_alpha[index.clamp_min(0)]
                elif destination_name == "missing_bias":
                    mask = type_mask[policy.missing_types]
                    value = source.missing_bias
                else:
                    mask = type_mask
                    value = getattr(source.core, destination_name)
                total += torch.where(mask, value, 0)
                counts += mask.to(counts.dtype)
            destination.copy_(torch.where(counts > 0, total / counts.clamp_min(1), destination))
