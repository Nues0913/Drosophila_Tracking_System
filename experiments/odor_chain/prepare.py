"""Prepare one seed/group selection without changing the seed-0 formal settings.

SEED defaults to 0; CHAIN_GROUPS defaults to G0 G1 G2 G3; OUT defaults to
runs/odor_gain/chain_seed<seed>. SMOKE=1 writes minimal CPU configs under OUT/smoke.
Each group gets its own vision stage so warm starts preserve its model card.
"""

import copy
import json
import os
from pathlib import Path

import yaml

TEMPLATES = {
    "vision": Path("configs/train/pretrain_vision.yaml"),
    "odor": Path("configs/train/pretrain_odor.yaml"),
    "bc": Path("configs/train/bc.yaml"),
}
# group: (odor_input_gain, core_bias_learning_rate, dn_floor_coef, decoder_input_scale);
# None keeps the template unchanged (G0 is the current formal chain).
GROUPS = {
    "G0": (None, None, None, None),
    "G1": (10, 0.0, 0.0, "population_rms"),
    "G2": (100, 0.0, 0.0, "population_rms"),
    "G3": (10, 0.0, 0.01, "population_rms"),
}
SMOKE_TRAIN = {
    "vision": {"total_steps": 24, "epochs": 1, "batch_episodes": 2},
    "odor": {"total_steps": 24, "epochs": 1, "batch_episodes": 2},
    "bc": {"teacher_steps": 24, "dagger_rounds": 0, "epochs": 1, "batch_episodes": 2},
}
SMOKE_VALIDATION = {"episodes": 1, "batch_episodes": 1}


def stage_config(template, group, stage, base, seed):
    gain, bias_rate, floor, scale = GROUPS[group]
    config = copy.deepcopy(template)
    config["variant"], config["seed"] = "VO", seed
    config["output"] = str(base / group / stage)
    previous = {"odor": "vision", "bc": "odor"}.get(stage)
    if previous:
        config["init_artifact"] = str(base / group / previous / "actor")
    else:
        config.pop("init_artifact", None)
    if stage != "bc":
        # Identical teacher/exploration episode stream for every group; the cache
        # key covers seed, modality, world and episode source code.
        config["dataset_cache"] = str(base / "datasets")
    train = config["train"]
    if gain is not None:
        config["model"] = {**config["model"], "odor_input_gain": gain}
    if bias_rate is not None:
        train["core_bias_learning_rate"] = bias_rate
    if floor is not None:
        train["dn_floor_coef"] = floor
    if scale is not None and stage != "bc":
        train["decoder_input_scale"] = scale
    return config


def write(path, config):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(config, sort_keys=False))


def main():
    seed = int(os.environ.get("SEED", "0"))
    groups = os.environ.get("CHAIN_GROUPS", "G0 G1 G2 G3").split()
    out = Path(os.environ.get("OUT", f"runs/odor_gain/chain_seed{seed}"))
    smoke = os.environ.get("SMOKE", "0")
    if seed < 0:
        raise ValueError("SEED must be nonnegative")
    if not groups or len(set(groups)) != len(groups) or any(g not in GROUPS for g in groups):
        raise ValueError("CHAIN_GROUPS must select distinct groups from G0 G1 G2 G3")
    if out.is_absolute() or ".." in out.parts or len(out.parts) < 2 or out.parts[0] != "runs":
        raise ValueError("OUT must be a repository-relative directory under runs/")
    if smoke not in {"0", "1"}:
        raise ValueError("SMOKE must be 0 or 1")
    base = out / "smoke" if smoke == "1" else out
    templates = {stage: yaml.safe_load(path.read_text()) for stage, path in TEMPLATES.items()}
    assert templates["bc"]["train"]["core_bias_learning_rate"] == 0.0
    assert templates["bc"]["env"]["scenario"] == "normal"
    manifest = {
        "seed": seed,
        "groups": groups,
        "smoke": smoke == "1",
        "templates": {stage: str(path) for stage, path in TEMPLATES.items()},
        "configs": {},
    }
    for group in groups:
        for stage in ("vision", "odor", "bc"):
            config = stage_config(templates[stage], group, stage, base, seed)
            if smoke == "1":
                config["train"].update(SMOKE_TRAIN[stage])
                # Retain the 8-10 s occlusion interval in episode evaluation.
                config["env"]["episode_seconds"] = 12
                if stage != "bc":
                    config["validation"] = dict(SMOKE_VALIDATION)
            manifest["configs"][f"{group}_{stage}"] = config
    path = base / "configs/manifest.json"
    if path.exists() and json.loads(path.read_text()) != manifest:
        raise ValueError(
            "Existing OUT belongs to a different run; use its original settings or a new OUT"
        )
    for name, config in manifest["configs"].items():
        write(base / "configs" / f"{name}.yaml", config)
    path.write_text(json.dumps(manifest, indent=1) + "\n")


if __name__ == "__main__":
    main()
