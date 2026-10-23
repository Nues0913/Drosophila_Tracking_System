"""Stage entry points; inherited protocol evaluation and CPU diagnostics."""

import os
import subprocess
import sys
from pathlib import Path

from c11_common import ROOT, manifest_profile, portable, read, relative, seeds, write
from protocol100 import actor_hashes, dn_rows, validate_eval


def train(out, extra):
    import torch
    from flytrack_train.__main__ import main
    from protocol100 import environment

    manifest = read(out / "manifest.json")
    if not manifest["smoke"] and (
        not torch.cuda.is_available() or "4060" not in torch.cuda.get_device_name(0)
    ):
        raise ValueError("Formal training requires the 4060 CUDA device")
    write(
        out / "train_runtime.json",
        {
            "device": manifest["device"],
            "device_name": "CPU" if manifest["smoke"] else torch.cuda.get_device_name(0),
            "torch_version": str(torch.__version__),
            "deterministic": torch.are_deterministic_algorithms_enabled(),
            "environment": environment(),
        },
    )
    sys.argv = [
        "flytrack_train",
        "ppo",
        "--config",
        str(out / "ppo_c11.yaml"),
        "--device",
        manifest["device"],
        "--workers",
        os.environ["PPO_WORKERS"],
        *extra,
    ]
    main()


def evaluate(out, name, *, seed_guard=None):
    import eval100

    m = read(out / "manifest.json")
    if seed_guard is None:
        seeds(m["seed_start"], m["episodes"], m["smoke"], manifest_profile(m).name)
    else:
        seed_guard(m["seed_start"], m["episodes"], m["smoke"])
    actor = str(out / "ppo_c11/actor") if name == "c11" else m["inputs"]["BC_ACTOR"]
    path = out / f"eval_{name}.tmp.json"
    os.environ.update(EPISODES=str(m["episodes"]), SEED_START=str(m["seed_start"]))
    sys.argv = ["eval100", str(path), str(out / "eval.yaml"), actor]
    eval100.main()
    report = portable(read(path))
    validate_eval(
        report,
        [actor],
        episodes=m["episodes"],
        seed_start=m["seed_start"],
        smoke=m["smoke"],
        env=m["configs"]["eval.yaml"]["env"],
    )
    write(out / f"eval_{name}.json", report)
    path.unlink()


def measure(out, actor, destination):
    m = read(out / "manifest.json")
    inputs = m["inputs"]
    env = {
        **os.environ,
        "CUDA_VISIBLE_DEVICES": "",
        "PYTHONPATH": os.pathsep.join(
            (
                str(ROOT / "python"),
                str(ROOT / inputs["MEASURE_REPO"] / "python"),
            )
        ),
    }
    subprocess.run(
        [sys.executable, inputs["MEASURE_PY"], str(destination), str(actor)],
        cwd=ROOT,
        env=env,
        stdin=subprocess.DEVNULL,
        check=True,
    )
    report = portable(read(destination))
    dn_rows(report, str(actor), 8 if m["smoke"] else 200)
    write(destination, report)


def snapshot_actors(out):
    m = read(out / "manifest.json")
    return [{"step": 0, "actor": m["inputs"]["BC_ACTOR"]}, *read(out / "snap_build.json")["actors"]]


def snap_eval(out, *, items=None, seed_guard=None):
    from functools import partial

    import numpy as np
    import torch
    from eval100 import Actor, run
    from flytrack_core.model.sparse import enable_deterministic
    from flytrack_train.config import contract_from_config
    from flytrack_train.parallel import EnvPool
    from flytrack_train.worlds import make_env

    torch.set_num_threads(1)
    enable_deterministic()
    m = read(out / "manifest.json")
    config = m["configs"]["eval.yaml"]
    episodes, start = m["snap_episodes"], m["snap_seed_start"]
    if seed_guard is None:
        seeds(start, episodes, m["smoke"], manifest_profile(m).name)
    else:
        seed_guard(start, episodes, m["smoke"])
    contract = contract_from_config(config)

    class RecordingPool(EnvPool):
        def step(self, actions):
            values = super().step(actions)
            for slot, (_, _, _, _, info) in values.items():
                omegas.append(float(info.get("applied_action", actions[slot])[1]))
            return values

    rows = []
    for item in snapshot_actors(out) if items is None else items:
        omegas = []
        actor = Actor(item["actor"], torch.device("cpu"))
        pool = RecordingPool([partial(make_env, config, contract, scenario="normal")], workers=0)
        try:
            report = run(pool, [0], actor, contract, episodes, config["env"], start)
        finally:
            pool.close()
        rows.append(
            {
                **item,
                "actor_sha256": actor_hashes(item["actor"]),
                "device": "cpu",
                "seed_start": start,
                "episodes": episodes,
                "normal": report,
                "tracked_fraction": report["tracked_fraction"],
                "omega_median": float(np.median(omegas)),
                "turn_bias": float(actor.policy.readout.turn_bias.detach().abs().max()),
                "core_bias_max_abs": float(actor.policy.core.bias.detach().abs().max()),
            }
        )
    write(out / "snap_eval.json", {"rows": rows})


def snap_dn(out):
    rows = []
    for item in snapshot_actors(out):
        destination = out / "snapshots" / f"dn_step_{item['step']}.json"
        measure(out, item["actor"], destination)
        rows.append({**item, "report": str(destination)})
    write(out / "snap_dn.json", {"rows": rows})


def acceptance_result(out):
    from accept100 import compare

    m = read(out / "manifest.json")
    seeds(m["seed_start"], m["episodes"], m["smoke"], manifest_profile(m).name)
    reports = {name: read(out / f"eval_{name}.json") for name in ("bc", "c11")}
    actors = {"bc": m["inputs"]["BC_ACTOR"], "c11": str(out / "ppo_c11/actor")}
    for name, report in reports.items():
        validate_eval(
            report,
            [actors[name]],
            episodes=m["episodes"],
            seed_start=m["seed_start"],
            smoke=m["smoke"],
            env=m["configs"]["eval.yaml"]["env"],
        )
    return {
        "smoke": m["smoke"],
        **compare(
            reports["bc"]["policies"][actors["bc"]],
            reports["c11"]["policies"][actors["c11"]],
            read(out / "dn_c11.json"),
            episodes=m["episodes"],
            seed_start=m["seed_start"],
            smoke=m["smoke"],
        ),
    }


def execute(out, stage):
    if stage == "c11_train":
        extra = ["--resume"] if (out / "ppo_c11/training_checkpoint.pt").exists() else []
        train(out, extra)
    elif stage in ("c11_eval", "bc_eval"):
        evaluate(out, stage.removesuffix("_eval"))
    elif stage == "dn_c11":
        measure(out, out / "ppo_c11/actor", out / "dn_c11.json")
    elif stage == "snap_build":
        from snapshots import build_all

        build_all(out)
    elif stage == "snap_eval":
        snap_eval(out)
    elif stage == "snap_dn":
        snap_dn(out)
    elif stage == "acceptance":
        write(out / "acceptance.json", acceptance_result(out))
    elif stage == "summary":
        from summarize_c11 import summarize

        summarize(out)
    else:
        raise ValueError("Unknown stage")


if __name__ == "__main__":
    execute(Path(relative(sys.argv[1])), sys.argv[2])
