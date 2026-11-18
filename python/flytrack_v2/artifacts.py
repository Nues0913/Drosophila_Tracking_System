"""Exported v2 artifact layout and model card schema 4 (plan 5.3, frozen at M0).

v1 cards carry no spec_version; validate_model_card refuses them, so v1 weights cannot be
loaded onto a v2 graph or readout (and the v2 loaders refuse to mix the two).

An artifact must rebuild its policy without the training config (#161 C4): model_config holds
the fully resolved model settings (readout type lists, gain parameterization, structure seed,
input mapping), and graph.npz is the graph the policy actually ran on. For RW2 and ER2 that is
the rewired graph, so a loader never rewires again.
"""

from __future__ import annotations

from collections.abc import Mapping

from flytrack_v2 import SCHEMA_VERSION, SPEC_VERSION

GRAPH_FREE_VARIANTS = ("GRU2",)
VARIANTS = ("VO2", "V2", "O2", "RW2", "ER2", "GRU2")


def required_files(variant: str) -> tuple[str, ...]:
    files = ["weights.pt", "frontend.npz", "input_norm.npz", "sensor_contract.json"]
    if variant not in GRAPH_FREE_VARIANTS:
        files.append("graph.npz")
    return (*files, "model_card.json", "SHA256SUMS")


_REQUIRED = {
    "spec_version": str,
    "schema_version": int,
    "variant": str,
    "sensor_contract_sha256": str,
    "files": dict,  # file name -> sha256, for every required file except the card and sums
    "state": dict,  # central_state_size, code_size (the sensory state is internal)
    "control_dt": float,
    "neural_dt": float,
    "n_ommatidia": int,
    "action_limits": dict,  # v_max, omega_max
    "model_config": dict,  # fully resolved model settings; rebuilds the policy with graph.npz
    "provenance": dict,  # git_sha, dirty, config_sha256, stages
}
_NESTED = {
    "state": ("central_state_size", "code_size"),
    "action_limits": ("v_max", "omega_max"),
    "provenance": ("git_sha", "dirty", "config_sha256", "stages"),
}


def validate_model_card(card: Mapping) -> None:
    if card.get("spec_version") != SPEC_VERSION:
        raise ValueError(
            "Not a v2 model card; v1 artifacts have no spec_version and cannot be mixed with v2"
        )
    if card.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"Model card schema_version must be {SCHEMA_VERSION}")
    for key, kind in _REQUIRED.items():
        value = card.get(key)
        if kind is float and isinstance(value, int) and not isinstance(value, bool):
            value = float(value)
        if not isinstance(value, kind) or isinstance(value, bool) and kind is not bool:
            raise ValueError(f"Model card field {key} must be {kind.__name__}")
    for key, names in _NESTED.items():
        missing = [name for name in names if name not in card[key]]
        if missing:
            raise ValueError(f"Model card {key} is missing {missing}")
    if card["variant"] not in VARIANTS:
        raise ValueError(f"Unknown v2 variant {card['variant']}; choose {VARIANTS}")
    config = card["model_config"]
    if config.get("variant") != card["variant"]:
        raise ValueError("model_config.variant must equal the card's variant")
    if card["variant"] in ("RW2", "ER2") and not isinstance(config.get("structure_seed"), int):
        raise ValueError(f"{card['variant']} model_config needs the integer structure_seed")
    expected = set(required_files(card["variant"])) - {"model_card.json", "SHA256SUMS"}
    if set(card["files"]) != expected:
        raise ValueError(f"Model card files must list exactly {sorted(expected)}")
