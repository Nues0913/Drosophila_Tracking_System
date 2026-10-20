"""Resolve environment inputs and bind a resumable run to unchanged inputs."""

import hashlib
import json
import os
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    out = Path(os.environ.get("OUT", "runs/t3c")).resolve()
    smoke = os.environ.get("SMOKE", "0") == "1"
    inputs = {
        name: str(Path(os.environ[name]).resolve())
        for name in ("BC_ACTOR", "PRETRAIN_ACTOR", "MEASURE_PY", "MEASURE_REPO")
    }
    if not Path(inputs["MEASURE_PY"]).is_relative_to(inputs["MEASURE_REPO"]):
        raise ValueError("MEASURE_PY must be inside MEASURE_REPO")
    for name in ("BC_ACTOR", "PRETRAIN_ACTOR"):
        path = Path(inputs[name])
        card = json.loads((path / "model_card.json").read_text())
        if card["variant"] != "V":
            raise ValueError(f"{name} must be a V actor")
        for filename, expected in card["files"].items():
            if digest(path / filename) != expected:
                raise ValueError(f"{name}: checksum mismatch for {filename}")
    configs = {}
    for arm, parent in (("bc", "BC_ACTOR"), ("direct", "PRETRAIN_ACTOR")):
        config = yaml.safe_load((HERE / f"ppo_{arm}.yaml").read_text())
        config["init_artifact"] = inputs[parent]
        config["output"] = str(out / f"ppo_{arm}")
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
        configs[f"ppo_{arm}.yaml"] = config
    evaluation = yaml.safe_load((HERE / "eval.yaml").read_text())
    if smoke:
        evaluation["env"]["episode_seconds"] = 0.2
    configs["eval.yaml"] = evaluation
    # All code, frozen diagnostic fixtures, and graph/frontend/scenarios are bound.
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
        )
    )
    measure_repo = Path(inputs["MEASURE_REPO"])
    measure_dir = Path(inputs["MEASURE_PY"]).parent
    paths.extend(measure_dir.glob("*.py"))
    paths.extend((measure_dir / "fixtures").glob("*.npz"))
    paths.append(measure_repo / "scripts/diagnostics/modality_sensitivity.py")
    paths.extend((measure_repo / "python").rglob("*.py"))
    paths.extend(Path(inputs[name]) / "model_card.json" for name in ("BC_ACTOR", "PRETRAIN_ACTOR"))
    manifest = {
        "smoke": smoke,
        "inputs": inputs,
        "configs": configs,
        "sha256": {str(path): digest(path) for path in sorted(set(paths))},
    }
    out.mkdir(parents=True, exist_ok=True)
    target = out / "manifest.json"
    if target.exists() and json.loads(target.read_text()) != manifest:
        raise ValueError("Run inputs/code/mode changed; use a new OUT instead of reusing results")
    for name, config in configs.items():
        encoded = yaml.safe_dump(config, sort_keys=False)
        path = out / name
        if path.exists() and path.read_text() != encoded:
            raise ValueError(f"Prepared config changed: {path}")
        if not path.exists():
            path.write_text(encoded)
    if not target.exists():
        target.write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n")
    print(f"Prepared {out}; smoke={smoke}")


if __name__ == "__main__":
    main()
