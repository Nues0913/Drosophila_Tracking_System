"""Prepare two bound C12 runs without modifying prior results or using git."""

import copy
import os
from pathlib import Path

import yaml
from c11.prepare_c11 import differences as c11_differences
from c11.prepare_c11 import main_hash_check, other_training
from protocol100 import validate_actor

from c12.c12_common import (
    FORMAL_SEED,
    MAIN_COMMIT,
    ROOT,
    RULES,
    SEEDS,
    SMOKE_SEED,
    SNAP_SEED,
    UNTIL,
    actor_files,
    digest,
    portable,
    read,
    relative,
    seeds,
    write,
)


def differences(left, right):
    return {
        key: {"c11": row["c10"], "c12": row["c11"]}
        for key, row in c11_differences(left, right).items()
    }


def python_inventory(root):
    # C12 requires the entire source inventory, including any PPO35 tests.
    return {
        p.relative_to(root).as_posix(): digest(p)
        for p in sorted((root / "python").rglob("*"))
        if p.is_file() and "__pycache__" not in p.parts and p.suffix not in (".pyc", ".pyo")
    }


def config_diff(base, candidate, seed):
    changes = differences(base, candidate)
    expected = {"output", "train.curriculum"} | ({"seed"} if seed == 1 else set())
    if seed not in (0, 1) or set(changes) != expected:
        raise ValueError("Unexpected C12 config differences (extra or missing)")
    curriculum = copy.deepcopy(base["train"]["curriculum"][2:])
    if len(curriculum) != 4:
        raise ValueError("C11 must have six curriculum stages")
    for row, until in zip(curriculum, UNTIL, strict=True):
        row["until"] = until
    if candidate["train"]["curriculum"] != curriculum or candidate["seed"] != seed:
        raise ValueError("C12 curriculum content/until or PPO seed differs")
    if (
        candidate["train"]["total_steps"] != 500_000
        or candidate["train"]["readout_bias_learning_rate"] != 0
        or candidate["train"]["critic_warmup_updates"] != 10
    ):
        raise ValueError("C11 fixed settings differ")
    return changes


def candidate_config(base, out, seed):
    config = copy.deepcopy(base)
    config["output"] = str(out / f"s{seed}" / "ppo_c11")
    config["seed"] = seed
    config["train"]["curriculum"] = copy.deepcopy(base["train"]["curriculum"][2:])
    for row, until in zip(config["train"]["curriculum"], UNTIL, strict=True):
        row["until"] = until
    config_diff(base, config, seed)
    return config


def reference_check(root, smoke):
    reference = root / "runs/c12"
    result = main_hash_check(root, reference / "main_python.sha256", smoke)
    path = reference / "main_python.sha256"
    if path.is_file():
        expected = {}
        for line in path.read_text().splitlines():
            fields = line.split(maxsplit=1)
            if len(fields) == 2:
                expected[fields[1].removeprefix("*").removeprefix("./")] = fields[0].lower()
        if expected != python_inventory(root):
            message = "Complete python/ inventory differs from pinned main reference"
            if not smoke:
                raise ValueError(message)
            print(f"WARNING: {message}", flush=True)
            result["warnings"].append(message)
    commit_path = reference / "main_python.commit"
    commit = commit_path.read_text().strip() if commit_path.is_file() else None
    if commit != MAIN_COMMIT:
        message = "C12 main_python.commit missing or differs from pinned main"
        if not smoke:
            raise ValueError(message)
        print(f"WARNING: {message}", flush=True)
        result["warnings"].append(message)
    return {**result, "commit": commit, "expected_commit": MAIN_COMMIT}


def save_bound(path, value, yaml_file=False):
    if yaml_file:
        encoded = yaml.safe_dump(value, sort_keys=False)
        if path.exists() and path.read_text() != encoded:
            raise ValueError("Prepared config changed")
        if not path.exists():
            path.write_text(encoded)
    else:
        if path.exists() and read(path) != value:
            raise ValueError("Run inputs/code/mode changed; restore inputs before resuming")
        if not path.exists():
            write(path, value)


def prepare(out, smoke):
    out = Path(relative(out))
    if out != Path("runs/c12_smoke" if smoke else "runs/c12"):
        raise ValueError("OUT differs from C12 mode")
    start = SMOKE_SEED if smoke else FORMAL_SEED
    snap_start = SMOKE_SEED + 1000 if smoke else SNAP_SEED
    episodes, snap_episodes = (1, 1) if smoke else (100, 20)
    for key, expected, count in (
        ("SEED_START", start, episodes),
        ("SNAP_SEED_START", snap_start, snap_episodes),
    ):
        actual = int(os.environ.get(key, expected))
        seeds(actual, count, smoke)
        if actual != expected:
            raise ValueError("Seed differs from C12 protocol")
    main = reference_check(ROOT, smoke)
    other_training()
    # Recognize the C12 wrapper before it imports flytrack_train.
    for proc in Path("/proc").iterdir():
        if not proc.name.isdigit() or int(proc.name) == os.getpid():
            continue
        try:
            argv = (proc / "cmdline").read_bytes().split(b"\0")
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
        if b"c12.c12_jobs" in argv and any(x in argv for x in (b"s0_train", b"s1_train")):
            raise ValueError("Other C12 training process")
    c10 = read(ROOT / "runs/c10/manifest.json")
    inputs = {
        key: relative(os.environ.get(key, c10["inputs"][key]))
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
    for p in actor_files(inputs["BC_ACTOR"]):
        original = str(Path(c10["inputs"]["BC_ACTOR"]) / p.name)
        if digest(p) != c10["sha256"][original]:
            raise ValueError("BC actor differs from C10 manifest")
    base = yaml.safe_load((ROOT / "runs/c11/ppo_c11.yaml").read_text())
    if Path(base["init_artifact"]).resolve() != Path(inputs["BC_ACTOR"]).resolve():
        raise ValueError("BC_ACTOR differs from C11 init_artifact")
    evaluation = yaml.safe_load((ROOT / "runs/c11/eval.yaml").read_text())
    evaluation["seed"] = start
    formal = {s: candidate_config(base, out, i) for i, s in enumerate(SEEDS)}
    configs = copy.deepcopy(formal)
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
    sources = list((ROOT / "experiments/ppo35/c12").glob("*.py"))
    sources += list((ROOT / "experiments/ppo35/c12").glob("*.sh"))
    sources += list((ROOT / "experiments/ppo35/c11").glob("*.py"))
    sources += list((ROOT / "experiments/ppo35").glob("*.py"))
    sources += [
        ROOT / "runs/c10/manifest.json",
        ROOT / "runs/c11/ppo_c11.yaml",
        ROOT / "runs/c11/eval.yaml",
    ]
    for folder in ("runs/c11", "runs/c11_s1"):
        sources += [
            ROOT / folder / name
            for name in ("manifest.json", "acceptance.json", "snap_eval.json", "snap_dn.json")
        ]
        sources += list((ROOT / folder / "snapshots").glob("dn_step_*.json"))
    sources += actor_files(inputs["BC_ACTOR"])
    sources += [Path(base[k]) for k in ("graph_path", "frontend_path", "sensor_contract_path")]
    sources += list((ROOT / "configs/scenarios").glob("*.yaml"))
    sources += list((Path(inputs["MEASURE_REPO"]) / "python").rglob("*.py"))
    for pattern in ("*.py", "*.npz"):
        sources += list(Path(inputs["MEASURE_PY"]).parent.rglob(pattern))
    sources += [Path(inputs["MEASURE_REPO"]) / "scripts/diagnostics/modality_sensitivity.py"]
    sources += [p for p in (ROOT / "runs/c12").glob("main_python.*") if p.is_file()]
    if os.environ.get("MONITOR_SH"):
        sources.append(Path(os.environ["MONITOR_SH"]))
    manifest = {
        "protocol": "C12-protocol100-v1",
        "smoke": smoke,
        "inputs": inputs,
        "configs": {"eval.yaml": evaluation, **{f"ppo_c12_{s}.yaml": configs[s] for s in SEEDS}},
        "formal_c12": formal,
        "c11_config_diff": {s: config_diff(base, formal[s], i) for i, s in enumerate(SEEDS)},
        "smoke_overrides": {s: differences(formal[s], configs[s]) for s in SEEDS} if smoke else {},
        "device": "cpu" if smoke else "cuda",
        "episodes": episodes,
        "seed_start": start,
        "snap_episodes": snap_episodes,
        "snap_seed_start": snap_start,
        "snapshot_interval": 12 if smoke else 50_000,
        "gpu_budget_s": 8100,
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
    # Check every existing binding before writing any config, including on resume.
    if (out / "manifest.json").exists() and read(out / "manifest.json") != manifest:
        raise ValueError("Run inputs/code/mode changed; restore inputs before resuming")
    out.mkdir(parents=True, exist_ok=True)
    for name, config in manifest["configs"].items():
        save_bound(out / name, config, True)
    save_bound(out / "manifest.json", manifest)
    for s in SEEDS:
        folder = out / s
        folder.mkdir(exist_ok=True)
        # Alias C11's internal artifact/config names to reuse its trainer and sidecar.
        child = {
            **manifest,
            "seed_name": s,
            "configs": {"ppo_c11.yaml": configs[s], "eval.yaml": evaluation},
        }
        save_bound(folder / "manifest.json", child)
        save_bound(folder / "ppo_c11.yaml", configs[s], True)
        save_bound(folder / "eval.yaml", evaluation, True)
    print(f"Prepared {out}; smoke={smoke}", flush=True)
    return manifest


if __name__ == "__main__":
    smoke = os.environ.get("SMOKE", "0") == "1"
    prepare(Path(os.environ.get("OUT", "runs/c12_smoke" if smoke else "runs/c12")), smoke)
