"""Bind a resumable C10 run to code, inputs and its single preregistered change."""

import copy
import os
from pathlib import Path

import yaml
from protocol100 import SEED_START, digest, environment, read, validate_actor, write

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def differences(left, right, prefix=""):
    if isinstance(left, dict) and isinstance(right, dict):
        result = {}
        for key in sorted(left.keys() | right.keys()):
            name = f"{prefix}.{key}" if prefix else key
            if key not in left or key not in right:
                result[name] = {"c9": left.get(key), "c10": right.get(key)}
            else:
                result.update(differences(left[key], right[key], name))
        return result
    return {} if left == right else {prefix: {"c9": left, "c10": right}}


def candidate_config(base, out, bc):
    config = copy.deepcopy(base)
    config["output"] = str(out / "ppo_c10")
    config["init_artifact"] = str(bc)
    config["train"]["core_bias_learning_rate"] = 0.001
    changes = differences(
        {k: v for k, v in base.items() if k not in {"output", "init_artifact"}},
        {k: v for k, v in config.items() if k not in {"output", "init_artifact"}},
    )
    if changes != {"train.core_bias_learning_rate": {"c9": 0.0, "c10": 0.001}}:
        raise ValueError(f"Unexpected C10 config differences: {changes}")
    if config["variant"] != "V" or config["seed"] != 0:
        raise ValueError("C10 must be V, seed 0")
    if config["train"]["total_steps"] != 500_000 or config["train"]["critic_warmup_updates"] != 10:
        raise ValueError("C10 training budget/warmup differs")
    return config, changes


def prepare():
    out = Path(os.environ.get("OUT", "runs/c10")).resolve()
    smoke = os.environ.get("SMOKE", "0") == "1"
    if smoke and out == ROOT / "runs/c10":
        raise ValueError("SMOKE must use a separate OUT (for example runs/c10_smoke)")
    inputs = {
        name: str(Path(os.environ[name]).resolve())
        for name in ("BC_ACTOR", "MEASURE_PY", "MEASURE_REPO")
    }
    inputs.update(
        C9_ACTOR=str(ROOT / "runs/t3c/ppo_bc/actor"),
        C9_DN=str(ROOT / "runs/t3c/dn_ppo_bc.json"),
    )
    if not Path(inputs["MEASURE_PY"]).is_relative_to(inputs["MEASURE_REPO"]):
        raise ValueError("MEASURE_PY must be inside MEASURE_REPO")
    validate_actor(inputs["BC_ACTOR"])
    base = yaml.safe_load((HERE / "ppo_bc.yaml").read_text())
    config, changes = candidate_config(base, out, Path(inputs["BC_ACTOR"]))
    formal = copy.deepcopy(config)
    evaluation = yaml.safe_load((HERE / "eval.yaml").read_text())
    if smoke:
        config["train"].update(
            num_envs=1,
            total_steps=8,
            rollout_steps=4,
            sequence_length=2,
            sequence_batch_size=2,
            update_epochs=1,
            critic_hidden_size=8,
            critic_warmup_updates=1,
        )
        config["env"]["episode_seconds"] = 0.2
        evaluation["env"]["episode_seconds"] = 0.2
    paths = list(HERE.glob("*.py")) + list(HERE.glob("*.sh")) + list(HERE.glob("*.yaml"))
    for folder in (
        "python/flytrack_core",
        "python/flytrack_train",
        "python/flytrack_sim",
        "scripts",
    ):
        paths.extend((ROOT / folder).rglob("*.py"))
    paths.extend((ROOT / "configs/scenarios").glob("*.yaml"))
    paths.extend(
        ROOT / name
        for name in (
            "data/subgraphs/main/graph.npz",
            "artifacts/frontend/frontend.npz",
            "configs/sensor_contract.json",
            "runs/t3c_pack/t3c_c9_7b6a722/trained_actor_SHA256SUMS",
            "runs/t3c/dn_ppo_bc.json",
        )
    )
    measure_dir = Path(inputs["MEASURE_PY"]).parent
    paths.extend(measure_dir.glob("*.py"))
    paths.extend((measure_dir / "fixtures").glob("*.npz"))
    paths.extend((Path(inputs["MEASURE_REPO"]) / "python").rglob("*.py"))
    paths.append(Path(inputs["MEASURE_REPO"]) / "scripts/diagnostics/modality_sensitivity.py")
    card = validate_actor(inputs["BC_ACTOR"])
    paths.extend(Path(inputs["BC_ACTOR"]) / name for name in ("model_card.json", *card["files"]))
    manifest = {
        "protocol": "C10-protocol100-v1",
        "smoke": smoke,
        "inputs": inputs,
        "configs": {"ppo_c10.yaml": config, "eval.yaml": evaluation},
        "formal_c10": formal,
        "c9_config_diff_excluding_output_init_artifact": changes,
        "smoke_overrides": differences(formal, config) if smoke else {},
        "episodes": 1 if smoke else 100,
        "seed_start": SEED_START,
        "device": "cpu" if smoke else "cuda",
        "deterministic": True,
        "environment": environment(),
        "budgets_s": {"a": 1800, "b": 3600},
        "budget_allocation": {"a": ["bc_eval", "c9_eval"], "b": ["c10_train", "c10_eval"]},
        "sha256": {str(path): digest(path) for path in sorted(set(paths))},
    }
    out.mkdir(parents=True, exist_ok=True)
    target = out / "manifest.json"
    if target.exists() and read(target) != manifest:
        raise ValueError("Run inputs/code/mode changed; use a new OUT")
    for name, value in manifest["configs"].items():
        path = out / name
        encoded = yaml.safe_dump(value, sort_keys=False)
        if path.exists() and path.read_text() != encoded:
            raise ValueError(f"Prepared config changed: {path}")
        if not path.exists():
            path.write_text(encoded)
    if not target.exists():
        write(target, manifest)
    print(f"Prepared {out}; smoke={smoke}", flush=True)
    return manifest


if __name__ == "__main__":
    prepare()
