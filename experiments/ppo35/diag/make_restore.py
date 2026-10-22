"""Restore exactly the core bias group using the repository artifact API."""

import sys
from pathlib import Path

import torch
from common import portable
from flytrack_core.model import load_policy_artifact, save_policy_artifact
from flytrack_train.imitation import parameter_groups
from protocol100 import actor_hashes, validate_actor, write


def bit_equal(a, b):
    return (
        a.dtype == b.dtype
        and a.shape == b.shape
        and torch.equal(
            a.detach().cpu().contiguous().reshape(-1).view(torch.uint8),
            b.detach().cpu().contiguous().reshape(-1).view(torch.uint8),
        )
    )


def check(restored, c10, bc, names):
    return {
        "other_state_bitwise_c10": all(
            bit_equal(value, c10[key]) for key, value in restored.items() if key not in names
        ),
        "bias_bitwise_bc": all(bit_equal(restored[key], bc[key]) for key in names),
    }


def main():
    out, c10_path, bc_path = map(Path, sys.argv[1:])
    for path in (c10_path, bc_path):
        validate_actor(path)
    policy, contract, card = load_policy_artifact(c10_path, device="cpu")
    bc, _, _ = load_policy_artifact(bc_path, device="cpu")
    before = {k: v.clone() for k, v in policy.state_dict().items()}
    baseline = bc.state_dict()
    if before.keys() != baseline.keys():
        raise ValueError("Actor state keys differ")
    names = {k for k, _ in policy.named_parameters() if k.startswith("core.bias")}
    groups = parameter_groups(policy, {"learning_rate": 1, "core_bias_learning_rate": 2})
    selected = {id(p) for group in groups if group["lr"] == 2 for p in group["params"]}
    if not names or selected != {id(p) for k, p in policy.named_parameters() if k in names}:
        raise ValueError("Bias group differs")
    state = {k: baseline[k] if k in names else v for k, v in before.items()}
    policy.load_state_dict(state)
    out.mkdir(parents=True, exist_ok=True)
    save_policy_artifact(
        out / "restore/actor",
        policy,
        card["requested_model_config"],
        contract,
        c10_path / "graph.npz",
        c10_path / "frontend.npz",
        metadata=portable(
            {
                "diagnostic": "core.bias restore",
                "c10": str(c10_path),
                "bc": str(bc_path),
                "sources_sha256": {"c10": actor_hashes(c10_path), "bc": actor_hashes(bc_path)},
            }
        ),
    )
    restored, _, _ = load_policy_artifact(out / "restore/actor", device="cpu")
    checks = check(restored.state_dict(), before, baseline, names)
    if not all(checks.values()):
        raise ValueError("Restored artifact failed bitwise checks")
    differences = [
        {
            "name": k,
            "elements": v.numel(),
            "max_abs_difference": float((v.double() - baseline[k].double()).abs().max()),
        }
        for k, v in before.items()
        if not bit_equal(v, baseline[k])
    ]
    write(
        out / "restore_check.json",
        {
            **checks,
            "replaced": [{"name": k, "elements": before[k].numel()} for k in sorted(names)],
            "bc_c10_differences": differences,
        },
    )


if __name__ == "__main__":
    main()
