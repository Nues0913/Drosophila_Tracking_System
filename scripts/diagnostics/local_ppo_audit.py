"""Bind local PPO preservation evidence to the exact actor and BC reference.

Reports both the production PPO DN threshold and the original BC threshold.
Behavioral superiority/noninferiority still requires paired episode analysis.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import torch
from flytrack_core.model import load_policy_artifact
from flytrack_train.config import write_json

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
from dn_health import health_gate  # noqa: E402


def audit(baseline_path, actor_path, health_path, expected_steps=65536):
    for path in (baseline_path, actor_path, health_path):
        if not path.resolve().is_relative_to(ROOT):
            raise ValueError("Use artifacts and evidence within this workspace")
    baseline, _, base_card = load_policy_artifact(baseline_path, device="cpu")
    actor, _, card = load_policy_artifact(actor_path, device="cpu")
    health = json.loads(health_path.read_text())
    if health.get("source_actor_files") != card["files"]:
        raise ValueError("DN health evidence belongs to different actor files")
    if Path(health["source_actor"]).resolve() != actor_path.resolve():
        raise ValueError("DN health evidence names a different actor")
    metadata = card["metadata"]
    configuration = metadata.get("config", {})
    settings = configuration.get("local_experiment", {})
    parent = configuration.get("init_artifact")
    changes = {}
    expected_state = baseline.state_dict()
    actual_state = actor.state_dict()
    same_shapes = actual_state.keys() == expected_state.keys() and all(
        value.shape == expected_state[name].shape for name, value in actual_state.items()
    )
    if same_shapes:
        for name, value in actual_state.items():
            changes[name] = float((value.double() - expected_state[name].double()).abs().max())
    ppo_health = health_gate(health, "ppo")
    bc_health = health_gate(health, "bc")
    checks = {
        "ppo_stage": metadata.get("stage") == "ppo"
        and metadata.get("algorithm") == "recurrent_ppo",
        "exact_training_budget": metadata.get("environment_steps") == expected_steps,
        "same_bc_parent": parent is not None and Path(parent).resolve() == baseline_path.resolve(),
        "same_variant": baseline.variant == actor.variant,
        "same_graph": baseline.graph_sha256 == actor.graph_sha256,
        "same_frontend": baseline.frontend_sha256 == actor.frontend_sha256,
        "same_sensor_contract": baseline.contract.sha256() == actor.contract.sha256(),
        "same_model_equations": baseline.model_config == actor.model_config,
        "same_parameter_shapes": same_shapes,
        "no_rate_counterfactual": not health.get("counterfactual_rate_override", True),
        "no_diagnostic_weight_update": not health.get("weights_updated", True),
        "ppo_dn_health": ppo_health["passed"],
        "original_bc_control_floor": bc_health["passed"],
    }
    trained_prefix = {"readout": "readout.", "readout_forward": "readout.forward"}.get(
        settings.get("actor_subset")
    )
    if trained_prefix:
        checks["frozen_parameters_unchanged"] = same_shapes and all(
            value == 0 for name, value in changes.items()
            if not name.startswith(trained_prefix) and name != "log_std"
        )
        checks["readout_updated"] = any(
            value > 0 for name, value in changes.items() if name.startswith(trained_prefix)
        )
    bound = settings.get("change_bound")
    if bound:
        bounded = [value.double() - expected_state[name].double()
                   for name, value in actual_state.items()
                   if name.startswith(tuple(bound["prefixes"]))]
        norm = float(torch.cat([v.flatten() for v in bounded]).norm())
        checks["change_within_declared_bound"] = norm <= float(bound["radius"]) * (1 + 1e-5)
    if settings.get("freeze_log_std"):
        checks["exploration_fixed_as_declared"] = torch.equal(
            actor.log_std, torch.full_like(actor.log_std, settings["initial_log_std"])
        )
    return {
        "passed": all(checks.values()), "checks": checks,
        "baseline": str(baseline_path), "actor": str(actor_path),
        "baseline_model_card_sha256": hashlib.sha256(
            (baseline_path / "model_card.json").read_bytes()
        ).hexdigest(),
        "actor_model_card_sha256": hashlib.sha256(
            (actor_path / "model_card.json").read_bytes()
        ).hexdigest(),
        "health_file_sha256": hashlib.sha256(health_path.read_bytes()).hexdigest(),
        "baseline_actor_files": base_card["files"],
        "actor_files": card["files"],
        "parameter_max_abs_changes": changes,
        "ppo_dn_gate": ppo_health, "bc_dn_gate": bc_health,
        "scope": "Artifact and control preservation; paired behavioral gains are separate.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--actor", type=Path, required=True)
    parser.add_argument("--health", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-steps", type=int, default=65536)
    args = parser.parse_args()
    if not args.output.resolve().is_relative_to(ROOT):
        parser.error("Output must stay inside this workspace")
    torch.set_num_threads(1)
    report = audit(args.baseline, args.actor, args.health, args.expected_steps)
    write_json(args.output, report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
