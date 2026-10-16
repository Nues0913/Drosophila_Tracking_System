#!/usr/bin/env python3
"""Read-only, teacher-driven held-out DN and action-gradient diagnostic (no fitting)."""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
from flytrack_core.model import load_policy_artifact
from flytrack_core.model.graph import Graph
from flytrack_core.model.policy import ConnectomePolicy
from flytrack_train.batches import tensor_observation
from flytrack_train.config import read_config, write_json
from flytrack_train.worlds import make_env


def heldout_seeds(config, episodes):
    validation = config.get("validation", {})
    base = int(validation.get("environment_seed", 1_500_000_000))
    seed = int(config.get("seed", 0))
    first = base + seed * 100_000
    if (
        episodes < 1
        or seed < 0
        or not 1_500_000_000 <= base < 1_700_000_000
        or not 1_500_000_000 <= first < first + episodes <= 1_700_000_000
    ):
        raise ValueError("DN health requires held-out environment seeds in [1.5e9,1.7e9)")
    return list(range(first, first + episodes))


def distribution(values):
    values = values.detach().double().flatten()
    quantiles = torch.quantile(values, values.new_tensor([0, 0.05, 0.5, 0.95, 1])).tolist()
    return dict(zip(("min", "p05", "median", "p95", "max"), quantiles, strict=True)) | {
        "mean": float(values.mean())
    }


def finite_json(value):
    if isinstance(value, dict):
        return {key: finite_json(item) for key, item in value.items()}
    if isinstance(value, list):
        return [finite_json(item) for item in value]
    return None if isinstance(value, float) and not math.isfinite(value) else value


def health_gate(report, stage):
    """A positive softplus floor alone is insufficient evidence of motor control."""
    checks = {
        "finite": report["finite"],
        "active_motor_count": report["effective_active_motor_count"] >= 4,
        "central_gradient": report["gradient_norms"]["core"] > 1e-6,
        "not_all_silent": not report["all_motor_silent"],
    }
    # Pretraining has no motor target, so steering control is only required once
    # the readout has been trained (BC onward); earlier stages check DN liveness.
    if report["visual_intervention_applicable"] and stage != "pretrain":
        checks["eye_swap_response"] = report["eye_swap_abs_omega_delta"]["median"] >= 0.05
    if stage == "bc":
        improvement = report["omega_mae_improvement_over_zero"]
        checks["bc_omega_error"] = improvement is not None and improvement >= 0.2
    return {"passed": all(checks.values()), "checks": checks}


def diagnose(
    policy,
    graph,
    config,
    *,
    episodes=4,
    steps=64,
    burn_in=16,
    gradient_steps=16,
    rate_threshold=1e-3,
    stage="pretrain",
):
    if not isinstance(policy, ConnectomePolicy):
        return {
            "skipped": True,
            "reason": "GRU has no biological DN correspondence",
            "passed": True,
        }
    if steps < 1 or burn_in < 0 or not 1 <= gradient_steps <= steps:
        raise ValueError("Require steps > 0, burn_in >= 0 and 1 <= gradient_steps <= steps")
    if not math.isfinite(rate_threshold) or rate_threshold <= 0:
        raise ValueError("Effective rate threshold must be positive and finite")
    seeds = heldout_seeds(config, episodes)
    device = policy.core.bias.device
    motor = policy.population_turn | policy.population_forward
    indices = motor.nonzero().flatten()
    all_rates, all_voltages, actions, targets, deltas = [], [], [], [], []
    gradient_sums = {
        name: torch.zeros_like(parameter)
        for name, parameter in policy.named_parameters()
        if name.startswith(("core.", "readout."))
    }
    named = dict(policy.named_parameters())
    loss_sum, loss_elements, code_delta_max = 0.0, 0, 0.0
    measured_steps = []
    policy.eval()
    for seed in seeds:
        env = make_env(config, policy.contract)
        codes, labels, probe_initial = [], [], None
        count = 0
        try:
            observation, info = env.reset(seed=seed)
            with torch.no_grad(), policy.adjacency_cache():
                sensory = policy.encoder.initial_state(2, device)
                central = policy.central_initial_state(2, device)
                for step in range(burn_in + steps):
                    swapped = {**observation, "vision": observation["vision"][::-1].copy()}
                    obs = tensor_observation([observation, swapped], device)
                    code, sensory = policy.sense(obs, sensory)
                    if step == burn_in:
                        probe_initial = central[:1].detach().clone()
                    action, central, activity = policy.forward_code(code, central)
                    if step >= burn_in:
                        count += 1
                        all_rates.append(activity[0, indices].cpu())
                        # Retain branch voltages separately; averaging can hide a dead branch.
                        voltage = central[0].reshape(policy.branches, policy.n_nodes)
                        all_voltages.append(voltage[:, indices].cpu())
                        actions.append(action[0].cpu())
                        teacher = torch.as_tensor(info["teacher_action"], device=device)
                        targets.append(teacher.cpu())
                        deltas.append((action[0, 1] - action[1, 1]).abs().cpu())
                        code_delta_max = max(
                            code_delta_max,
                            float((code[0] - code[1]).abs().max()),
                        )
                        if len(codes) < gradient_steps:
                            codes.append(code[:1].detach())
                            labels.append(teacher)
                    observation, _, terminated, truncated, info = env.step(
                        np.asarray(info["teacher_action"], dtype=np.float32)
                    )
                    if terminated or truncated:
                        break
        finally:
            env.close()
        if not codes:
            raise ValueError(f"Held-out episode {seed} ended before any measured step")
        measured_steps.append(count)
        # One task-only gradient probe per episode. No optimizer, backward accumulation,
        # weight edits, decoder, activity penalty, or training seed is involved.
        with torch.enable_grad(), policy.adjacency_cache():
            central = probe_initial
            predictions = []
            for code in codes:
                prediction, central, _ = policy.forward_code(code, central)
                predictions.append(prediction[0])
            prediction = torch.stack(predictions)
            scale = prediction.new_tensor([policy.contract.v_max, policy.contract.omega_max])
            loss = ((prediction - torch.stack(labels)) / scale).square().sum()
            gradients = torch.autograd.grad(
                loss, [named[name] for name in gradient_sums], allow_unused=True
            )
        loss_sum += float(loss.detach())
        loss_elements += prediction.numel()
        for name, gradient in zip(gradient_sums, gradients, strict=True):
            if gradient is not None:
                gradient_sums[name] += gradient.detach()
    rates, voltages = torch.stack(all_rates), torch.stack(all_voltages)
    action, target = torch.stack(actions), torch.stack(targets)
    norms = {name: float((value / loss_elements).norm()) for name, value in gradient_sums.items()}
    for prefix in ("core", "readout"):
        norms[prefix] = math.sqrt(
            sum(value**2 for name, value in norms.items() if name.startswith(prefix + "."))
        )
    effective_fraction = (rates > rate_threshold).float().mean(0)
    zero_mae = float(target[:, 1].abs().mean())
    omega_mae = float((action[:, 1] - target[:, 1]).abs().mean())
    finite = all(
        bool(torch.isfinite(value).all()) for value in (rates, voltages, action, target)
    ) and all(math.isfinite(value) for value in [loss_sum, *norms.values()])
    report = {
        "variant": policy.variant,
        "lesion": policy.lesion,
        "motor_rate_beta": policy.model_config.get("motor_rate_beta"),
        "stage": stage,
        "environment_seeds": seeds,
        "measured_steps_per_episode": measured_steps,
        "burn_in": burn_in,
        "gradient_steps_per_episode": gradient_steps,
        "trajectory": "teacher-driven; same original observations with only eye images swapped",
        "weights_updated": False,
        "finite": finite,
        "effective_rate_threshold": rate_threshold,
        "effective_active_motor_count": int((effective_fraction >= 0.1).sum()),
        "all_motor_silent": not bool((rates > 0).any()),
        "visual_intervention_applicable": policy.variant != "O" and policy.lesion != "vision",
        "paired_sensory_code_max_delta": code_delta_max,
        "eye_swap_abs_omega_delta": distribution(torch.stack(deltas)),
        "action_min": action.amin(0).tolist(),
        "action_max": action.amax(0).tolist(),
        "normalized_action_mse": loss_sum / loss_elements,
        "gradient_norms": norms,
        "omega_mae": omega_mae,
        "zero_omega_mae": zero_mae,
        "omega_mae_improvement_over_zero": 1 - omega_mae / zero_mae if zero_mae > 0 else None,
        "motor_neurons": [
            {
                "node_id": int(graph.node_ids[index]),
                "type": str(graph.type_names[graph.type_index[index]]),
                "side": int(graph.side[index]),
                "positive_rate_fraction": float((rates[:, column] > 0).float().mean()),
                "effective_active_fraction": float(effective_fraction[column]),
                "rate": distribution(rates[:, column]),
                "voltage_by_branch": [
                    distribution(voltages[:, branch, column]) for branch in range(policy.branches)
                ],
            }
            for column, index in enumerate(indices.cpu().tolist())
        ],
        "thresholds": {
            "minimum_effective_active_motor_count": 4,
            "minimum_active_fraction": 0.1,
            "minimum_median_eye_swap_omega_delta": 0.05,
            "minimum_core_gradient_norm_exclusive": 1e-6,
            "minimum_bc_omega_error_improvement": 0.2,
        },
    }
    report.update(health_gate(report, stage))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--config", type=Path, help="Otherwise use the artifact's recorded config")
    parser.add_argument("--output", type=Path, help="Optional JSON report; artifact is read-only")
    parser.add_argument("--episodes", type=int, default=4)
    parser.add_argument("--steps", type=int, default=64, help="Measured steps per episode")
    parser.add_argument("--burn-in", type=int, default=16)
    parser.add_argument("--gradient-steps", type=int, default=16)
    parser.add_argument("--rate-threshold", type=float, default=1e-3)
    parser.add_argument("--stage", choices=("pretrain", "bc", "ppo", "es"), default="pretrain")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument(
        "--sparse", action="store_true", help="Use the equivalent CSR central operator"
    )
    parser.add_argument(
        "--motor-rate-beta",
        type=float,
        help="Counterfactual rate override in memory only; does not change weights/artifact",
    )
    parser.add_argument("--require-healthy", action="store_true", help="Exit 2 if gate fails")
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("--threads must be positive")
    started = time.perf_counter()
    torch.set_num_threads(args.threads)
    policy, _, card = load_policy_artifact(args.artifact, device="cpu")
    config = read_config(args.config) if args.config else card.get("metadata", {}).get("config", {})
    if args.motor_rate_beta is not None:
        if not math.isfinite(args.motor_rate_beta) or args.motor_rate_beta <= 0:
            parser.error("--motor-rate-beta must be positive and finite")
        if not isinstance(policy, ConnectomePolicy):
            parser.error("--motor-rate-beta requires a connectome policy")
        policy.core.motor_rate_beta = args.motor_rate_beta
        policy.model_config["motor_rate_beta"] = args.motor_rate_beta
    if args.sparse and isinstance(policy, ConnectomePolicy):
        policy.core.dense = False
    report = diagnose(
        policy,
        Graph.load(args.artifact / "graph.npz"),
        config,
        episodes=args.episodes,
        steps=args.steps,
        burn_in=args.burn_in,
        gradient_steps=args.gradient_steps,
        rate_threshold=args.rate_threshold,
        stage=args.stage,
    )
    report.update(
        source_actor=str(args.artifact),
        source_actor_files=card["files"],
        counterfactual_rate_override=args.motor_rate_beta is not None,
        central_operator="csr" if args.sparse else "artifact setting",
        diagnostic_config=config,
        diagnostic_seconds=time.perf_counter() - started,
    )
    if args.output:
        write_json(args.output, report)
    # write_json handles nonfinite diagnostics as null; stdout must also be portable JSON.
    print(json.dumps(finite_json(report), indent=2, allow_nan=False))
    if args.require_healthy and not report["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
