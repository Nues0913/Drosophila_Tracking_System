"""Resolve real actor parameter budgets and write six-variant/seed stage configs.

This command constructs actors to count parameters but does not collect data or
train. Run it only during the later validation phase after genuine artifacts exist.
"""

from __future__ import annotations

import argparse
import copy
import json
import shlex
from pathlib import Path

import yaml
from flytrack_core.contracts import SensorContract
from flytrack_core.model import assert_parameter_budget, build_policy
from flytrack_train.parameters import count_effective_parameters


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-config", type=Path, default=Path("configs/model/variants.yaml"))
    parser.add_argument(
        "--graph", type=Path, default=Path("data/subgraphs/olfactory_v2_nokc/graph.npz")
    )
    parser.add_argument("--frontend", type=Path, default=Path("artifacts/frontend/frontend.npz"))
    parser.add_argument(
        "--sensor-contract", type=Path, default=Path("configs/sensor_contract.json")
    )
    parser.add_argument("--output", type=Path, default=Path("runs/experiment_configs"))
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(5)))
    # #35 Decision 6382764327: the formal chain ends at BC/DAgger; PPO/ES stay
    # available as an optional stage after BC.
    parser.add_argument("--algorithm", choices=("none", "ppo", "es"), default="none")
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--jobs", type=int, default=3, help="Concurrent (variant, seed) chains in run_all.sh"
    )
    parser.add_argument(
        "--dataset-cache",
        default="runs/datasets",
        help="Pretraining episode streams shared by all variants of a seed",
    )
    args = parser.parse_args()
    if not args.seeds or len(set(args.seeds)) != len(args.seeds) or min(args.seeds) < 0:
        parser.error("seeds must be distinct nonnegative integers")
    options = yaml.safe_load(args.model_config.read_text(encoding="utf-8"))
    contract = SensorContract.from_json(args.sensor_contract)
    reference = options["reference"]
    common = options.get("common", {})
    bc_template = yaml.safe_load(Path("configs/train/bc.yaml").read_text(encoding="utf-8"))
    # #4 B3: BC adds these rows (1 weight per branch), so the budget counts them too.
    extra_turn = dict(options.get("extra_turn_rows") or {})
    extra_variants = set(extra_turn.pop("variants", []))
    if not extra_variants <= set(options["variants"]) - {"GRU"}:
        parser.error("extra_turn_rows variants must be connectome variants of the matrix")

    def bc_train(variant):
        # Use this exact train section both for counting and for the emitted BC stage.
        train = copy.deepcopy(bc_template["train"])
        train.pop("calibrate_extra_turn", None)
        if variant in extra_variants:
            train["calibrate_extra_turn"] = dict(extra_turn)
        return train

    def with_extra_turn_rows(variant, policy):
        if variant in extra_variants:
            types = list(extra_turn["types"])
            policy.set_readout_extra_turns(
                types, [0.0] * len(types), [1.0] * len(types), float(extra_turn["eps"])
            )
        return policy

    reference_policy = with_extra_turn_rows(
        reference,
        build_policy(
            reference,
            args.graph,
            contract,
            args.frontend,
            **{**common, **options.get("overrides", {}).get(reference, {}), "budget_target": None},
        ),
    )
    target = count_effective_parameters(reference_policy, bc_train(reference))["effective"]
    configs = {}
    models = {reference: reference_policy}
    for variant in options["variants"]:
        resolved = {
            **common,
            **options.get("overrides", {}).get(variant, {}),
            # Only GRU resolves architecture from this target. Connectome actors
            # are checked below using BC-effective counts, not declared capacity.
            "budget_target": target if variant == "GRU" else None,
        }
        model = build_policy(variant, args.graph, contract, args.frontend, **resolved)
        # The stage configs build the actor without the rows; BC calibrates them.
        configs[variant] = dict(model.model_config)
        models[variant] = with_extra_turn_rows(variant, model)
    parameter_counts = {
        variant: count_effective_parameters(model, bc_train(variant))
        for variant, model in models.items()
    }
    counts = assert_parameter_budget(
        models,
        reference=reference,
        tolerance=float(options["tolerance"]),
        effective_counts={
            variant: count["effective"] for variant, count in parameter_counts.items()
        },
    )
    for count in parameter_counts.values():
        count["relative_to_effective_vo_percent"] = 100 * (count["effective"] / target - 1)
    # Free all duplicated frozen frontends before invoking any actual learner.
    del models, reference_policy
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "parameter_budget.json").write_text(
        json.dumps(
            {
                "reference": reference,
                "budget_basis": (
                    "effective BC actor parameters (requires_grad and lr > 0), including B3"
                ),
                "reference_effective_parameters": target,
                "tolerance": options["tolerance"],
                # Preserve this legacy key as declared capacity for existing readers.
                "actor_trainable_parameters": {
                    variant: count["declared"] for variant, count in parameter_counts.items()
                },
                "actor_trainable_parameters_basis": "declared (requires_grad), including B3",
                "parameter_counts": parameter_counts,
                "bc_learning_rates": {
                    key: bc_template["train"].get(key, 1e-3 if key == "learning_rate" else None)
                    for key in (
                        "learning_rate",
                        "readout_learning_rate",
                        "core_bias_learning_rate",
                        "readout_bias_learning_rate",
                    )
                },
                "gru_resolution": {
                    "budget_target": target,
                    "hidden_size": configs["GRU"]["hidden_size"],
                    "gru_input_size": configs["GRU"]["gru_input_size"],
                    "effective": counts["GRU"],
                },
                "extra_turn_rows": {**extra_turn, "variants": sorted(extra_variants)},
                "resolved_models": configs,
            },
            indent=2,
        )
        + "\n"
    )
    for variant, model_config in configs.items():
        for seed in args.seeds:
            chain = (
                [("pretrain_vision", "vision", 240000)]
                if variant == "V"
                else [("pretrain_odor", "odor", 240000)]
                if variant == "O"
                else [("pretrain_vision", "vision", 120000), ("pretrain_odor", "odor", 120000)]
            )
            chain.append(("bc", None, None))
            if args.algorithm != "none":
                chain.append((args.algorithm, None, None))
            previous = None
            for template_name, modality, steps in chain:
                config = (
                    bc_template
                    if template_name == "bc"
                    else yaml.safe_load(Path(f"configs/train/{template_name}.yaml").read_text())
                )
                config = copy.deepcopy(config)
                config.update(
                    variant=variant,
                    seed=seed,
                    model=model_config,
                    graph_path=str(args.graph),
                    frontend_path=str(args.frontend),
                    sensor_contract_path=str(args.sensor_contract),
                )
                if modality:
                    config["dataset_cache"] = args.dataset_cache
                config["scenario_configs"] = {
                    name: f"configs/scenarios/{name}.yaml"
                    for name in ("normal", "occlusion", "noise", "conflict")
                }
                config.pop("init_artifact", None)
                config.pop("pretrained_artifacts", None)
                if template_name == "bc":
                    config["train"] = bc_train(variant)
                if previous:
                    config["init_artifact"] = str(previous)
                if modality:
                    output = Path(f"runs/pretrain/{variant}/seed_{seed}/{modality}")
                    config["train"]["total_steps"] = steps
                elif template_name == "bc":
                    output = Path(f"runs/bc/{variant}/seed_{seed}")
                else:
                    output = Path(f"runs/{variant}/seed_{seed}")
                config["output"] = str(output)
                config_file = args.output / variant / f"seed_{seed}" / f"{template_name}.yaml"
                config_file.parent.mkdir(parents=True, exist_ok=True)
                config_file.write_text(
                    yaml.safe_dump(config, allow_unicode=True, sort_keys=False), encoding="utf-8"
                )
                previous = output / "actor"
    # Stages of a chain stay ordered; independent chains share the GPU/CPUs.
    # Extra arguments (e.g. --seeds 2, --variants VO) are passed through.
    commands = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        "# Run from the project root after reviewing budgets.",
        f"python scripts/run_experiments.py --configs {shlex.quote(str(args.output))} "
        f"--algorithm {args.algorithm} --device {shlex.quote(args.device)} "
        f'--jobs {args.jobs} "$@"',
    ]
    (args.output / "run_all.sh").write_text("\n".join(commands) + "\n", encoding="utf-8")
    print(
        f"Resolved effective BC actor counts: {counts}; "
        f"review {args.output / 'run_all.sh'} before running"
    )


if __name__ == "__main__":
    main()
