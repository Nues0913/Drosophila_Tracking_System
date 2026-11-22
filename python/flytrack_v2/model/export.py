"""Self-contained, hash-checked artifacts for v2 policies."""

from __future__ import annotations

import json
import shutil
from collections.abc import Mapping
from dataclasses import fields
from pathlib import Path
from tempfile import TemporaryDirectory

import torch
from flytrack_core.contracts import file_sha256

from flytrack_v2 import SCHEMA_VERSION, SPEC_VERSION
from flytrack_v2.artifacts import GRAPH_FREE_VARIANTS, required_files, validate_model_card
from flytrack_v2.contracts import SensorContractV2
from flytrack_v2.frontend import NodeMajorFlyvis

from .config import PLASTIC_DEFAULTS, ModelConfigV2
from .graph import GraphV2
from .policy import ConnectomePolicyV2, GRUPolicyV2, build_policy
from .sensory import InputNorm

_CARD = "model_card.json"
_SUMS = "SHA256SUMS"


def export_policy(
    policy: ConnectomePolicyV2 | GRUPolicyV2,
    out_dir: str | Path,
    *,
    provenance: Mapping,
) -> Path:
    """Export the policy's current weights and resolved components, returning the directory."""
    policy.refresh_rest_state(full=True)
    variant = policy.variant
    names = required_files(variant)
    destination = Path(out_dir)
    destination.mkdir(parents=True, exist_ok=True)

    # Stage the complete artifact so failures while writing or validating leave prior files
    # intact. Publish the checksum manifest last.
    with TemporaryDirectory(dir=destination) as temporary:
        staging = Path(temporary)
        torch.save(
            {key: tensor.detach().cpu() for key, tensor in policy.state_dict().items()},
            staging / "weights.pt",
        )
        frontend = policy.encoder.frontend
        shutil.copyfile(frontend.source_path, staging / "frontend.npz")
        policy.input_norm.save(staging / "input_norm.npz")
        policy.contract.write_json(staging / "sensor_contract.json")

        model_config = policy.config.to_dict(include_plastic_defaults=True)
        model_config.update(
            variant=variant,
            visual_bins=frontend.frozen.visual_bins,
            steer_types=[],
            speed_types=[],
            readout_body_ids={"steer": {}, "speed": {}},
        )
        card = {
            "spec_version": SPEC_VERSION,
            "schema_version": SCHEMA_VERSION,
            "variant": variant,
            "sensor_contract_sha256": policy.contract.sha256(),
            **_contract_fields(policy.contract),
            "state": _state_fields(policy),
            "model_config": model_config,
            "provenance": dict(provenance),
        }
        if isinstance(policy, ConnectomePolicyV2):
            policy.graph.save(staging / "graph.npz")
            card["graph_origin_sha256"] = policy.graph.origin_sha256
            model_config.update(
                steer_types=list(policy.readout.steer_types),
                speed_types=list(policy.readout.speed_types),
                readout_body_ids={
                    "steer": policy.readout.steer_body_ids,
                    "speed": policy.readout.speed_body_ids,
                },
            )
        else:
            model_config["hidden_size"] = policy.central_state_size

        files = {name: file_sha256(staging / name) for name in names if name not in (_CARD, _SUMS)}
        if files["frontend.npz"] != frontend.source_sha256:
            raise ValueError("The frontend source has changed since policy construction")
        if (
            frontend.frozen.control_dt != policy.contract.control_dt
            or frontend.frozen.neural_dt != policy.contract.neural_dt
            or frontend.n_ommatidia != policy.contract.n_ommatidia
        ):
            raise ValueError("The frontend timing and eye geometry must match the sensor contract")
        card["files"] = files
        (staging / _CARD).write_text(
            json.dumps(card, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8"
        )
        checksums = {**files, _CARD: file_sha256(staging / _CARD)}
        (staging / _SUMS).write_text(
            "".join(f"{digest}  {name}\n" for name, digest in checksums.items()),
            encoding="utf-8",
        )
        validate_model_card(card)
        for name in names:
            (staging / name).replace(destination / name)
    return destination


@torch.inference_mode(False)
def load_policy(
    artifact_dir: str | Path, *, device: str | torch.device = "cpu"
) -> ConnectomePolicyV2 | GRUPolicyV2:
    """Load a v2 policy for inference using only its exported artifact directory."""
    source = Path(artifact_dir)
    checksums = _verify_checksums(source)
    card = json.loads((source / _CARD).read_text(encoding="utf-8"))
    if not isinstance(card, dict) or card.get("files") != checksums:
        raise ValueError("Model card files must match the verified SHA256SUMS entries")
    validate_model_card(card)

    contract = SensorContractV2.from_json(source / "sensor_contract.json")
    if contract.sha256() != card["sensor_contract_sha256"]:
        raise ValueError("Model card sensor_contract_sha256 does not match the sensor contract")
    for key, value in _contract_fields(contract).items():
        if card[key] != value:
            raise ValueError(f"Model card {key} does not match the sensor contract")

    variant = card["variant"]
    config_values = dict(card["model_config"])
    config_values.pop("variant")
    readout_body_ids = config_values.pop("readout_body_ids", None)
    hidden_size = config_values.pop("hidden_size", None) if variant in GRAPH_FREE_VARIANTS else None
    missing = {field.name for field in fields(ModelConfigV2)} - config_values.keys()
    missing.difference_update(PLASTIC_DEFAULTS, {"visual_offset_init"})
    missing.discard("odor_adaptation_fraction")  # Older cards: complete adaptation (1.0).
    missing.discard("odor_adaptation_fractions")  # Older cards: one fraction for all.
    if missing:
        raise ValueError(f"Model card model_config is missing {sorted(missing)}")
    config = ModelConfigV2.from_dict(config_values)
    graph = None
    if variant not in GRAPH_FREE_VARIANTS:
        graph = GraphV2.load(source / "graph.npz")
        if card.get("graph_origin_sha256", graph.origin_sha256) != graph.origin_sha256:
            raise ValueError("Model card graph_origin_sha256 does not match graph.npz")

    policy = build_policy(
        variant,
        config,
        contract=contract,
        graph=graph,
        frontend=NodeMajorFlyvis(source / "frontend.npz", contract, config.visual_bins),
        input_norm=InputNorm.load(source / "input_norm.npz", config.norm_std_floor),
        from_artifact=True,
        readout_body_ids=readout_body_ids,
    )
    if variant in GRAPH_FREE_VARIANTS and (
        type(hidden_size) is not int or hidden_size != policy.central_state_size
    ):
        raise ValueError("GRU2 model_config hidden_size must match the resolved GRU2 width")
    if card["state"] != _state_fields(policy):
        raise ValueError("Model card state dimensions do not match the reconstructed policy")
    weights = torch.load(source / "weights.pt", weights_only=True, map_location="cpu")
    policy.load_state_dict(weights, strict=True)
    policy.to(device).eval()
    policy.refresh_rest_state(full=True)
    return policy


def _verify_checksums(source: Path) -> dict[str, str]:
    allowed = set(required_files("VO2")) - {_SUMS}
    checksums = {}
    for line in (source / _SUMS).read_text(encoding="ascii").splitlines():
        digest, marker, name = line[:64], line[64:66], line[66:]
        if (
            marker not in ("  ", " *")
            or any(char not in "0123456789abcdef" for char in digest)
            or name not in allowed
            or name in checksums
        ):
            raise ValueError("Invalid SHA256SUMS entry")
        checksums[name] = digest
    if not set(required_files("GRU2")) - {_SUMS} <= checksums.keys():
        raise ValueError("SHA256SUMS is missing required files")
    for name, digest in checksums.items():
        path = source / name
        if not path.is_file() or file_sha256(path) != digest:
            raise ValueError(f"SHA256SUMS checksum mismatch: {name}")
    del checksums[_CARD]
    return checksums


def _contract_fields(contract: SensorContractV2) -> dict:
    return {
        "control_dt": contract.control_dt,
        "neural_dt": contract.neural_dt,
        "n_ommatidia": contract.n_ommatidia,
        "action_limits": {"v_max": contract.v_max, "omega_max": contract.omega_max},
    }


def _state_fields(policy: ConnectomePolicyV2 | GRUPolicyV2) -> dict[str, int]:
    return {"central_state_size": policy.central_state_size, "code_size": policy.code_size}
