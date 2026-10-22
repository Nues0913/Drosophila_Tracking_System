"""Restore only the two preregistered readout biases: OUT BASE BC."""

import sys
from pathlib import Path

from diag.common import portable
from diag.make_restore import bit_equal
from flytrack_core.model import load_policy_artifact, save_policy_artifact
from protocol100 import actor_hashes, validate_actor, write

NAMES = ("readout.turn_bias", "readout.forward_bias")


def restore_state(base, bc):
    for name in NAMES:
        if name not in base or name not in bc:
            raise ValueError(f"Missing parameter: {name}")
    if base.keys() != bc.keys():
        raise ValueError("Actor state keys differ")
    for name in NAMES:
        if base[name].shape != bc[name].shape or base[name].dtype != bc[name].dtype:
            raise ValueError(f"Parameter shape/dtype differs: {name}")
    return {k: (bc[k] if k in NAMES else v).clone() for k, v in base.items()}


def check(state, base, bc):
    return {
        "other_state_bitwise_base": all(
            bit_equal(v, base[k]) for k, v in state.items() if k not in NAMES
        ),
        "bias_bitwise_bc": all(bit_equal(state[k], bc[k]) for k in NAMES),
        "replaced": [{"name": k, "elements": base[k].numel()} for k in NAMES],
        "remaining_bc_differences": [
            {
                "name": k,
                "elements": v.numel(),
                "max_abs_difference": float((v.double() - bc[k].double()).abs().max()),
            }
            for k, v in state.items()
            if not bit_equal(v, bc[k])
        ],
    }


def main():
    out, base_path, bc_path = map(Path, sys.argv[1:])
    for path in (base_path, bc_path):
        validate_actor(path)
    policy, contract, card = load_policy_artifact(base_path, device="cpu")
    bc, _, _ = load_policy_artifact(bc_path, device="cpu")
    base = {k: v.clone() for k, v in policy.state_dict().items()}
    baseline = bc.state_dict()
    policy.load_state_dict(restore_state(base, baseline))
    save_policy_artifact(
        out / "actor",
        policy,
        card["requested_model_config"],
        contract,
        base_path / "graph.npz",
        base_path / "frontend.npz",
        metadata=portable(
            {
                "diagnostic": "readout bias restore",
                "base": str(base_path),
                "bc": str(bc_path),
                "sources_sha256": {"base": actor_hashes(base_path), "bc": actor_hashes(bc_path)},
            }
        ),
    )
    restored, _, _ = load_policy_artifact(out / "actor", device="cpu")
    checks = check(restored.state_dict(), base, baseline)
    if not checks["other_state_bitwise_base"] or not checks["bias_bitwise_bc"]:
        raise ValueError("Restored artifact failed bitwise checks")
    write(out.parent / f"{out.name}_check.json", checks)


if __name__ == "__main__":
    main()
