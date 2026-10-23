"""Prepare C13 bindings; never modify Python sources or delivered results."""

import copy
import importlib
import inspect
import os
import re
from pathlib import Path

import yaml
from c11.prepare_c11 import main_hash_check, other_training
from c12.prepare_c12 import differences, python_inventory, save_bound
from protocol100 import validate_actor

from c13.c13_common import (
    FORMAL_SEED,
    LAMBDA,
    ROOT,
    RULES,
    SEEDS,
    SMOKE_SEED,
    SNAP_SEED,
    actor_files,
    digest,
    portable,
    read,
    relative,
    seeds,
)


def config_diff(base, candidate, seed):
    changes = differences(base, candidate)
    if (
        seed not in (0, 1)
        or set(changes) != {"output", "train.bc_anchor_coef"}
        or "bc_anchor_coef" in base["train"]
        or candidate["train"].get("bc_anchor_coef") != LAMBDA
        or isinstance(candidate["train"].get("bc_anchor_coef"), bool)
        or base["seed"] != seed
    ):
        raise ValueError("Unexpected C13 config differences (extra, missing or lambda)")
    return {k: {"c12": v["c11"], "c13": v["c12"]} for k, v in changes.items()}


def candidate_config(base, out, seed):
    config = copy.deepcopy(base)
    # Retain the shared C11 artifact basename used by C12's trainer and sidecar.
    config["output"] = str(out / f"s{seed}" / "ppo_c11")
    config["train"]["bc_anchor_coef"] = LAMBDA
    config_diff(base, config, seed)
    return config


def anchor_support():
    module = importlib.import_module("flytrack_train.ppo")
    if not callable(
        getattr(module, "_bc_anchor_loss", None)
    ) or "bc_anchor_coef" not in inspect.getsource(module):
        raise ValueError("main 尚未合併 bc_anchor_coef")


def reference_check(root, smoke):
    reference = root / "runs/c13"
    path = reference / "main_python.sha256"
    result = main_hash_check(root, path, smoke)
    expected = {}
    if path.is_file():
        for line in path.read_text().splitlines():
            fields = line.split(maxsplit=1)
            if len(fields) == 2:
                expected[fields[1].removeprefix("*").removeprefix("./")] = fields[0].lower()
    errors = []
    if expected != python_inventory(root):
        errors.append("Complete python/ inventory differs from main reference")
    commit_path = reference / "main_python.commit"
    commit = commit_path.read_text().strip() if commit_path.is_file() else None
    if commit is None or re.fullmatch(r"[a-fA-F0-9]{40}", commit) is None:
        errors.append("C13 main_python.commit missing or invalid")
    if errors and not smoke:
        raise ValueError("; ".join(errors))
    for message in errors:
        print(f"WARNING: {message}", flush=True)
    return {**result, "commit": commit, "warnings": result["warnings"] + errors}


def training_preflight():
    other_training()
    for proc in Path("/proc").iterdir():
        if not proc.name.isdigit() or int(proc.name) == os.getpid():
            continue
        try:
            argv = (proc / "cmdline").read_bytes().split(b"\0")
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
        if any(p in argv for p in (b"c12.c12_jobs", b"c13.c13_jobs")) and any(
            p in argv for p in (b"s0_train", b"s1_train")
        ):
            raise ValueError("Other C12/C13 training process")


def prepare(out, smoke):
    out = Path(relative(out))
    if out != Path("runs/c13_smoke" if smoke else "runs/c13"):
        raise ValueError("OUT differs from C13 mode")
    start, snap_start = (SMOKE_SEED, SMOKE_SEED + 1000) if smoke else (FORMAL_SEED, SNAP_SEED)
    episodes, snap_episodes = (1, 1) if smoke else (100, 20)
    for key, expected, count in (
        ("SEED_START", start, episodes),
        ("SNAP_SEED_START", snap_start, snap_episodes),
    ):
        actual = int(os.environ.get(key, expected))
        seeds(actual, count, smoke)
        if actual != expected:
            raise ValueError("Seed differs from C13 protocol")
    anchor_support()
    main = reference_check(ROOT, smoke)
    training_preflight()
    prior_folder = ROOT / "runs/c12"
    prior = read(prior_folder / "manifest.json")
    inputs = {
        key: relative(os.environ.get(key, prior["inputs"][key]))
        for key in ("BC_ACTOR", "MEASURE_REPO", "MEASURE_PY")
    }
    if "MEASURE_REPO" in os.environ and "MEASURE_PY" not in os.environ:
        inputs["MEASURE_PY"] = str(
            Path(inputs["MEASURE_REPO"]) / "experiments/odor_chain/measure.py"
        )
    if (
        not Path(inputs["MEASURE_PY"])
        .resolve()
        .is_relative_to(Path(inputs["MEASURE_REPO"]).resolve())
    ):
        raise ValueError("MEASURE_PY must be inside MEASURE_REPO")
    validate_actor(inputs["BC_ACTOR"])
    for path in actor_files(inputs["BC_ACTOR"]):
        original = str(Path(prior["inputs"]["BC_ACTOR"]) / path.name)
        if digest(path) != prior["sha256"][original]:
            raise ValueError("BC actor differs from C12 manifest")
    bases = {s: yaml.safe_load((prior_folder / f"ppo_c12_{s}.yaml").read_text()) for s in SEEDS}
    for base in bases.values():
        if Path(base["init_artifact"]).resolve() != Path(inputs["BC_ACTOR"]).resolve():
            raise ValueError("BC_ACTOR differs from C12 init_artifact")
    formal = {s: candidate_config(bases[s], out, i) for i, s in enumerate(SEEDS)}
    configs = copy.deepcopy(formal)
    evaluation = yaml.safe_load((prior_folder / "eval.yaml").read_text())
    evaluation["seed"] = start
    if smoke:
        for config in configs.values():
            config["train"].update(
                num_envs=1,
                rollout_steps=12,
                total_steps=36,
                sequence_length=2,
                sequence_batch_size=2,
                update_epochs=1,
                critic_hidden_size=8,
                critic_warmup_updates=1,
            )
            config["env"]["episode_seconds"] = 0.2
        evaluation["env"]["episode_seconds"] = 0.2
    sources = [ROOT / p for p in prior["sha256"]]
    for folder in ("c11", "c12", "c13"):
        for pattern in ("*.py", "*.sh"):
            sources += list((ROOT / "experiments/ppo35" / folder).glob(pattern))
    sources += [
        prior_folder / name
        for name in ("manifest.json", "eval.yaml", "acceptance.json", "dn_bc.json")
    ]
    sources += [prior_folder / f"ppo_c12_{s}.yaml" for s in SEEDS]
    for seed in SEEDS:
        folder = prior_folder / seed
        sources += [
            folder / name
            for name in (
                "snap_eval.json",
                "snap_dn.json",
                "snapshots/index.json",
                "ppo_c11/training_metrics.json",
            )
        ]
        sources += list((folder / "snapshots").glob("*.pt"))
        sources += list((folder / "snapshots").glob("dn_step_*.json"))
    sources += actor_files(inputs["BC_ACTOR"])
    sources += list((Path(inputs["MEASURE_REPO"]) / "python").rglob("*.py"))
    for pattern in ("*.py", "*.npz"):
        sources += list(Path(inputs["MEASURE_PY"]).parent.rglob(pattern))
    sources += [Path(inputs["MEASURE_REPO"]) / "scripts/diagnostics/modality_sensitivity.py"]
    sources += [p for p in (ROOT / "runs/c13").glob("main_python.*") if p.is_file()]
    if os.environ.get("MONITOR_SH"):
        sources.append(Path(os.environ["MONITOR_SH"]))
    manifest = {
        "protocol": "C13-protocol100-v1",
        "smoke": smoke,
        "inputs": inputs,
        "configs": {"eval.yaml": evaluation, **{f"ppo_c13_{s}.yaml": configs[s] for s in SEEDS}},
        "formal_c13": formal,
        "c12_config_diff": {s: config_diff(bases[s], formal[s], i) for i, s in enumerate(SEEDS)},
        "smoke_overrides": {s: differences(formal[s], configs[s]) for s in SEEDS} if smoke else {},
        "device": "cpu" if smoke else "cuda",
        "episodes": episodes,
        "seed_start": start,
        "snap_episodes": snap_episodes,
        "snap_seed_start": snap_start,
        "snapshot_interval": 12 if smoke else 50_000,
        "gpu_budget_s": 9000,
        "main_python": main,
        "python_sha256": python_inventory(ROOT),
        "sha256": {relative(p): digest(p) for p in sorted(set(sources))},
        "rules": RULES,
        "execution": portable(
            {
                k: os.environ.get(k)
                for k in ("LANES", "WORKERS", "PPO_WORKERS", "CUDA_VISIBLE_DEVICES", "MONITOR_SH")
            }
        ),
        "snap_scheduling": (
            "After both training receipts; concurrent with GPU evaluation; "
            "nice 10; two CPUs per seed; workers=0"
        ),
    }
    if (out / "manifest.json").exists() and read(out / "manifest.json") != manifest:
        raise ValueError("Run inputs/code/mode changed; restore inputs before resuming")
    out.mkdir(parents=True, exist_ok=True)
    for name, config in manifest["configs"].items():
        save_bound(out / name, config, True)
    save_bound(out / "manifest.json", manifest)
    for seed in SEEDS:
        folder = out / seed
        folder.mkdir(exist_ok=True)
        child = {
            **manifest,
            "seed_name": seed,
            "configs": {"ppo_c11.yaml": configs[seed], "eval.yaml": evaluation},
        }
        save_bound(folder / "manifest.json", child)
        save_bound(folder / "ppo_c11.yaml", configs[seed], True)
        save_bound(folder / "eval.yaml", evaluation, True)
    print(f"Prepared {out}; smoke={smoke}", flush=True)
    return manifest


if __name__ == "__main__":
    smoke = os.environ.get("SMOKE", "0") == "1"
    prepare(Path(os.environ.get("OUT", "runs/c13_smoke" if smoke else "runs/c13")), smoke)
