"""Prepare C11, bind resume inputs, and perform read-only preflight checks."""

import copy
import fnmatch
import os
import re
from pathlib import Path, PurePosixPath

import yaml
from c11_common import (
    FORMAL_SEED,
    HERE,
    ROOT,
    SMOKE_SEED,
    SNAP_SEED,
    actor_files,
    digest,
    portable,
    read,
    relative,
    seeds,
    write,
)
from prepare_c10 import differences as c10_differences
from protocol100 import validate_actor


def differences(left, right):
    return {
        key: {"c10": row["c9"], "c11": row["c10"]}
        for key, row in c10_differences(left, right).items()
    }


def config_diff(base, candidate):
    # Compare equivalent path spellings consistently; emitted files use relative paths.
    changes = differences(portable(base), portable(candidate))
    expected = {
        "output": {"c10": portable(base["output"]), "c11": candidate["output"]},
        "train.readout_bias_learning_rate": {"c10": None, "c11": 0},
    }
    if changes != expected or "readout_bias_learning_rate" in base["train"]:
        raise ValueError("Unexpected C11 config differences")
    if (
        candidate["variant"] != "V"
        or candidate["seed"] != 0
        or candidate["train"]["total_steps"] != 500_000
        or candidate["train"]["core_bias_learning_rate"] != 0.001
        or candidate["train"]["readout_learning_rate"] != 0.01
        or candidate["train"]["critic_warmup_updates"] != 10
    ):
        raise ValueError("C10 reference settings differ")
    return changes


def candidate_config(base, out):
    config = portable(copy.deepcopy(base))
    config["output"] = str(out / "ppo_c11")
    config["train"]["readout_bias_learning_rate"] = 0
    config_diff(base, config)
    return config


def excluded(name):
    return fnmatch.fnmatchcase(name, "python/tests/test_ppo35_*.py")


def python_inventory(root):
    return {
        p.relative_to(root).as_posix(): digest(p)
        for p in sorted((root / "python").rglob("*"))
        if p.is_file()
        and "__pycache__" not in p.parts
        and not p.name.endswith((".pyc", ".pyo"))
        and not excluded(p.relative_to(root).as_posix())
    }


def main_hash_check(root, path, smoke):
    errors, expected = [], {}
    if not path.is_file():
        errors.append("main_python.sha256 missing")
    else:
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            fields = line.split(maxsplit=1)
            if len(fields) != 2 or not re.fullmatch(r"[a-fA-F0-9]{64}", fields[0]):
                errors.append("Invalid sha256sum entry")
                continue
            checksum, name = fields
            name = name.removeprefix("*").removeprefix("./")
            p = PurePosixPath(name)
            if p.is_absolute() or ".." in p.parts or not name.startswith("python/"):
                errors.append("Invalid reference path")
            elif not excluded(name):
                if name in expected:
                    errors.append("Duplicate reference path")
                expected[name] = checksum.lower()
        actual = python_inventory(root)
        if expected != actual:
            errors.append("python/ inventory or hashes differ from main reference")
        if not expected:
            errors.append("Empty main reference")
    if errors and not smoke:
        raise ValueError("; ".join(errors))
    for error in errors:
        print(f"WARNING: {error}", flush=True)
    return {"sha256": digest(path) if path.exists() else None, "warnings": errors}


def training_command(tokens):
    if any("flytrack_train" in p for p in tokens) and any(
        p in {"ppo", "bc", "pretrain"} for p in tokens
    ):
        return True
    if any(Path(p).name in {"run_c10.py", "driver.py"} for p in tokens) and any(
        p in {"train", "train-entry"} for p in tokens
    ):
        return True
    return any(Path(p).name == "jobs.py" for p in tokens) and "c11_train" in tokens


def other_training():
    matches = []
    for proc in Path("/proc").iterdir():
        if not proc.name.isdigit() or int(proc.name) == os.getpid():
            continue
        try:
            argv = (proc / "cmdline").read_bytes().split(b"\0")
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
        tokens = [p.decode(errors="replace") for p in argv if p]
        if training_command(tokens):
            matches.append(int(proc.name))
    if matches:
        raise ValueError(f"Other training process(es): {matches}")


def prepare(out, smoke):
    out = Path(relative(out))
    if (smoke and out != Path("runs/c11_smoke")) or (not smoke and out != Path("runs/c11")):
        raise ValueError("Use runs/c11_smoke for smoke and runs/c11 for formal runs")
    out.mkdir(parents=True, exist_ok=True)
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
    other_training()
    main = main_hash_check(ROOT, ROOT / "runs/c11/main_python.sha256", smoke)
    main_ref = os.environ.get("MAIN_REF", "origin/main")
    commit_file = ROOT / "runs/c11/main_python.commit"
    commit = os.environ.get("MAIN_COMMIT")
    if commit is None:
        commit = main_ref if re.fullmatch(r"[0-9a-f]{40}", main_ref) else None
    if commit is None and commit_file.exists():
        commit = commit_file.read_text().strip()
    if not smoke and (commit is None or not re.fullmatch(r"[0-9a-f]{40}", commit)):
        raise ValueError("Provide reference commit via MAIN_COMMIT or main_python.commit")
    base = yaml.safe_load((ROOT / "runs/c10/ppo_c10.yaml").read_text())
    formal = candidate_config(base, out)
    if Path(formal["init_artifact"]).resolve() != Path(inputs["BC_ACTOR"]).resolve():
        raise ValueError("BC_ACTOR differs from C10 init_artifact")
    config = copy.deepcopy(formal)
    evaluation = portable(yaml.safe_load((ROOT / "runs/c10/eval.yaml").read_text()))
    evaluation["seed"] = SMOKE_SEED if smoke else FORMAL_SEED
    if smoke:
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
    seed = int(os.environ.get("SEED_START", evaluation["seed"]))
    snap_seed = int(os.environ.get("SNAP_SEED_START", SMOKE_SEED + 1000 if smoke else SNAP_SEED))
    episodes, snap_episodes = (1, 1) if smoke else (100, 20)
    seeds(seed, episodes, smoke)
    seeds(snap_seed, snap_episodes, smoke)
    if seed != evaluation["seed"] or snap_seed != (SMOKE_SEED + 1000 if smoke else SNAP_SEED):
        raise ValueError("Seed differs from C11 protocol")
    sources = list(HERE.glob("*.py")) + list(HERE.glob("*.sh"))
    sources += list(HERE.parent.glob("*.py"))
    sources += [ROOT / "runs/c10" / n for n in ("manifest.json", "ppo_c10.yaml", "eval.yaml")]
    sources += [
        ROOT / "runs/c10" / n
        for n in (
            "eval_c10.json",
            "dn_c10.json",
            "acceptance_c10_only.json",
            "ppo_c10/actor/weights.pt",
        )
        if (ROOT / "runs/c10" / n).exists()
    ]
    if os.environ.get("MONITOR_SH"):
        sources.append(Path(os.environ["MONITOR_SH"]))
    sources += actor_files(inputs["BC_ACTOR"])
    sources += list((Path(inputs["MEASURE_REPO"]) / "python").rglob("*.py"))
    sources += list(Path(inputs["MEASURE_PY"]).parent.rglob("*.py"))
    sources += list(Path(inputs["MEASURE_PY"]).parent.rglob("*.npz"))
    sources += [Path(inputs["MEASURE_REPO"]) / "scripts/diagnostics/modality_sensitivity.py"]
    sources += [Path(formal[k]) for k in ("graph_path", "frontend_path", "sensor_contract_path")]
    sources += list((ROOT / "configs/scenarios").glob("*.yaml"))
    manifest = {
        "protocol": "C11-protocol100-v1",
        "smoke": smoke,
        "inputs": inputs,
        "configs": {"ppo_c11.yaml": config, "eval.yaml": evaluation},
        "formal_c11": formal,
        "c10_config_diff": config_diff(base, formal),
        "smoke_overrides": differences(formal, config) if smoke else {},
        "device": "cpu" if smoke else "cuda",
        "episodes": episodes,
        "seed_start": seed,
        "snap_episodes": snap_episodes,
        "snap_seed_start": snap_seed,
        "snapshot_interval": config["train"]["num_envs"] * config["train"]["rollout_steps"]
        if smoke
        else 50_000,
        "gpu_budget_s": 4500,
        "main_ref": main_ref,
        "main_commit": commit,
        "main_python": main,
        "python_sha256": python_inventory(ROOT),
        "sha256": {relative(p): digest(p) for p in sorted(set(sources))},
        "execution": portable(
            {
                k: os.environ.get(k)
                for k in (
                    "LANES",
                    "WORKERS",
                    "PPO_WORKERS",
                    "CUDA_VISIBLE_DEVICES",
                    "MONITOR_SH",
                )
            }
        ),
        "snap_scheduling": "Concurrent CPU chain; nice 10; at most two affinity CPUs; workers=0",
    }
    target = out / "manifest.json"
    if target.exists() and read(target) != manifest:
        raise ValueError("Run inputs/code/mode changed; restore inputs before resuming")
    for name, value in manifest["configs"].items():
        path = out / name
        encoded = yaml.safe_dump(value, sort_keys=False)
        if path.exists() and path.read_text() != encoded:
            raise ValueError("Prepared config changed")
        if not path.exists():
            path.write_text(encoded)
    if not target.exists():
        write(target, manifest)
    print(f"Prepared {out}; smoke={smoke}", flush=True)
    return manifest


if __name__ == "__main__":
    smoke_mode = os.environ.get("SMOKE", "0") == "1"
    prepare(Path(os.environ.get("OUT", "runs/c11_smoke" if smoke_mode else "runs/c11")), smoke_mode)
