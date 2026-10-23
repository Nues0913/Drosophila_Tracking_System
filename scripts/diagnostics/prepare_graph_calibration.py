"""Write isolated, short #55 cost-probe configs from resolved model budgets.

Does not train or launch a runner. These are timing probes, not formal budgets
or quality evidence. No old-graph checkpoint is used. Existing output refused.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml
from flytrack_core.contracts import file_sha256


def prepare(budget_path: Path, graph: Path, output: Path, seed: int) -> dict:
    budget = json.loads(budget_path.read_text())
    output.mkdir(parents=True, exist_ok=False)
    rows = []
    for variant, model in budget["resolved_models"].items():
        # Independent stages avoid judging tracking quality from a short warmstart.
        # PPO is deliberately from initialization: its timing is only a first bound.
        for stage in ("pretrain_vision", "pretrain_odor", "bc", "ppo"):
            if (variant == "V" and stage == "pretrain_odor") or (
                variant == "O" and stage == "pretrain_vision"
            ):
                continue
            config = yaml.safe_load(Path(f"configs/train/{stage}.yaml").read_text())
            config.pop("init_artifact", None)
            config.pop("pretrained_artifacts", None)
            config.update(
                variant=variant,
                seed=seed,
                model=model,
                graph_path=str(graph),
                output=str(output / "runs" / variant / stage),
            )
            train = config["train"]
            formal = dict(train)
            if stage.startswith("pretrain"):
                train.update(total_steps=12000, epochs=2)
                config["dataset_cache"] = str(output / "datasets")
            elif stage == "bc":
                train.update(
                    teacher_steps=12000, dagger_steps_per_round=6000, dagger_rounds=1, epochs=2
                )
            else:
                train["total_steps"] = 8192
            path = output / "configs" / variant / f"{stage}.yaml"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(yaml.safe_dump(config, sort_keys=False))
            rows.append(
                {
                    "variant": variant,
                    "stage": stage,
                    "config": str(path),
                    "probe_train": train,
                    "formal_template_train": formal,
                }
            )
    report = {
        "diagnostic_only": True,
        "graph": str(graph),
        "graph_sha256": file_sha256(graph),
        "parameter_budget_sha256": file_sha256(budget_path),
        "seed": seed,
        "stages": rows,
        "formal_pretrain_total_steps_per_chain": 240000,
        "formal_bc_steps_per_chain": 150000,
        "formal_ppo_steps_per_chain": 2000000,
        "formal_variants": 6,
        "formal_seeds": 5,
        "gpu_hours_measured": None,
        "full_matrix_48_to_72h_verified": False,
    }
    (output / "plan.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--parameter-budget", type=Path, required=True)
    p.add_argument("--graph", type=Path, default=Path("data/subgraphs/olfactory_v2/graph.npz"))
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    if args.seed < 0:
        p.error("seed must be nonnegative")
    report = prepare(args.parameter_budget, args.graph, args.output, args.seed)
    print(f"Prepared {len(report['stages'])} isolated stage probes; no training launched.")


if __name__ == "__main__":
    main()
