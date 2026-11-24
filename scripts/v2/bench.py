"""Throughput benchmark for the v2 throughput gate (plan 4.3, #139).

``python scripts/v2/bench.py v1 --host-label 4060`` measures the v1 baseline on this host:

- env: one FlyTrackEnv on one CPU thread, environment steps per second (bench item a);
- policy: the VO policy step (sense + core, no gradient) at several batch sizes (item b);
- ppo: v1 PPO end to end with the formal ppo.yaml batch (8 environments x 256 steps).
  The run is timed at two lengths, and the difference gives steps per second without
  start-up time (item d).

``python scripts/v2/bench.py v2 --host-label 2060`` measures the v2 rollout on this host:

- rollout: one batched world per world kind (v1compat, and wind at the protocol's U), the
  four scenarios mixed and both motions, the VO2 policy step (sense + core, no gradient,
  deterministic action) and DAgger's teacher label (OcclusionTeacher) every step; batched
  environment steps per second at each batch size. It bounds the end-to-end rate from above
  (#161 F): training's backward pass and optimiser come with WP4a, and the 100 timed steps
  of a 1,200-step episode include no episode reset (#174 review).

The JSON written to runs/bench/ is the baseline every hot-path v2 PR is compared against. It
identifies the host only by --host-label, never by its name.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[2]


def _git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=ROOT, capture_output=True, text=True, check=True
    ).stdout.strip()


def environment(host_label: str, device: str) -> dict:
    cuda = device.startswith("cuda") and torch.cuda.is_available()
    return {
        "host_label": host_label,
        "commit": _git("rev-parse", "HEAD"),
        "dirty": bool(_git("status", "--porcelain", "--untracked-files=no")),
        "utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "device": device,
        "gpu": torch.cuda.get_device_name(0) if cuda else None,
        "cpu": platform.processor() or platform.machine(),
        "cpu_count": os.cpu_count(),
    }


def bench_env(steps: int = 300) -> dict:
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        return _bench_env(steps)
    finally:
        torch.set_num_threads(threads)  # the policy bench on --device cpu keeps its threads


def _bench_env(steps: int) -> dict:
    from flytrack_core.contracts import SensorContract
    from flytrack_sim.env import FlyTrackEnv

    contract = SensorContract.from_json(ROOT / "configs/sensor_contract.json")
    env = FlyTrackEnv(
        contract,
        scenario="normal",
        episode_seconds=60,
        target_motion="curved",
        plume_model="puff",
    )
    env.reset(seed=0)
    rng = np.random.default_rng(0)
    for _ in range(20):
        env.step(np.array([0.2, 0.0], np.float32))
    start = time.perf_counter()
    for index in range(steps):
        action = np.array([0.2, rng.uniform(-0.5, 0.5)], np.float32)
        _, _, terminated, truncated, _ = env.step(action)
        if terminated or truncated:
            env.reset(seed=index)
    seconds = time.perf_counter() - start
    return {"steps": steps, "ms_per_step": 1000 * seconds / steps, "steps_per_s": steps / seconds}


def bench_policy(device: str, batches: list[int], steps: int = 40) -> dict:
    from flytrack_core.contracts import SensorContract
    from flytrack_core.model.policy import build_policy

    train = yaml.safe_load((ROOT / "configs/train/ppo.yaml").read_text(encoding="utf-8"))
    contract = SensorContract.from_json(ROOT / train["sensor_contract_path"])
    common = yaml.safe_load((ROOT / "configs/model/variants.yaml").read_text(encoding="utf-8"))
    config = {**common["common"], **train["model"]}
    policy = build_policy(
        "VO", ROOT / train["graph_path"], contract, ROOT / train["frontend_path"], **config
    ).to(device)
    results = {}
    with torch.no_grad():
        for batch in batches:
            obs = {
                "vision": torch.rand(batch, 2, contract.n_ommatidia, device=device),
                "odor": torch.rand(batch, 2, device=device),
                "valid": torch.ones(batch, 3, device=device),
            }
            state = policy.initial_state(batch, device)
            for _ in range(5):
                _, state, _ = policy(obs, state)
            if device.startswith("cuda"):
                torch.cuda.synchronize()
            start = time.perf_counter()
            for _ in range(steps):
                _, state, _ = policy(obs, state)
            if device.startswith("cuda"):
                torch.cuda.synchronize()
            ms = 1000 * (time.perf_counter() - start) / steps
            results[str(batch)] = {"ms_per_step": ms, "samples_per_s": 1000 * batch / ms}
    return results


def ppo_config(updates: int, output: Path) -> dict:
    """Formal v1 ppo.yaml with a short budget, a fresh actor and a private output."""
    config = yaml.safe_load((ROOT / "configs/train/ppo.yaml").read_text(encoding="utf-8"))
    train = config["train"]
    train["total_steps"] = updates * train["num_envs"] * train["rollout_steps"]
    config.pop("init_artifact", None)
    config["output"] = str(output)
    for key in ("sensor_contract_path", "graph_path", "frontend_path"):
        config[key] = str(ROOT / config[key])
    config["scenario_configs"] = {
        name: str(ROOT / path) for name, path in config["scenario_configs"].items()
    }
    return config


def _time_ppo(updates: int, device: str, workdir: Path) -> tuple[int, float]:
    config = ppo_config(updates, workdir / f"ppo_{updates}")
    path = workdir / f"ppo_{updates}.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    command = [sys.executable, "-m", "flytrack_train", "ppo", "--config", str(path)]
    start = time.perf_counter()
    done = subprocess.run([*command, "--device", device], cwd=ROOT, capture_output=True, text=True)
    if done.returncode:
        tail = "\n".join(done.stderr.splitlines()[-20:])
        raise RuntimeError(f"v1 PPO exited with {done.returncode}:\n{tail}")
    return config["train"]["total_steps"], time.perf_counter() - start


def bench_ppo(device: str, short: int = 2, long: int = 6) -> dict:
    (ROOT / "runs").mkdir(exist_ok=True)  # runs/ is not in git; a fresh clone has none
    with tempfile.TemporaryDirectory(prefix="bench_ppo_", dir=ROOT / "runs") as workdir:
        steps_short, seconds_short = _time_ppo(short, device, Path(workdir))
        steps_long, seconds_long = _time_ppo(long, device, Path(workdir))
    return {
        "updates": [short, long],
        "steps": [steps_short, steps_long],
        "wall_s": [seconds_short, seconds_long],
        "steps_per_s": (steps_long - steps_short) / (seconds_long - seconds_short),
    }


def _time_rollout(world, teacher, policy, batch, device, warmup, steps) -> float:
    from flytrack_v2.interfaces import policy_step

    with torch.no_grad():
        obs = world.reset()
        sensory = policy.initial_sensory_state(batch, device)
        central = policy.initial_central_state(batch, device)
        label_state = teacher.initial_state(batch, device)
        reset = None
        wall = 0.0
        for index in range(warmup + steps):
            if index == warmup:
                if device.startswith("cuda"):
                    torch.cuda.synchronize()
                started = time.perf_counter()
            step, sensory, _ = policy_step(policy, obs, sensory, central, reset)
            central = step.central_state
            privileged = world.privileged()
            _, label_state = teacher.act(privileged, obs, privileged["agent_velocity"], label_state)
            result = world.step(step.dist.mode())
            reset = result.terminated | result.truncated
            obs = result.obs
        if device.startswith("cuda"):
            torch.cuda.synchronize()
        wall = time.perf_counter() - started
    return wall


def bench_rollout_v2(
    device: str, batches: list[int], worlds: list[str], warmup: int = 20, steps: int = 100
) -> dict:
    """Batched v2 rollout steps per second: world step + VO2 policy step + teacher label."""
    from flytrack_v2.contracts import SensorContractV2
    from flytrack_v2.model import InputNorm, ModelConfigV2, build_policy
    from flytrack_v2.world.teacher import OcclusionTeacher, PrivilegedTeacher
    from flytrack_v2.world.v1compat import V1CompatWorld
    from flytrack_v2.world.wind import WindConfig

    contract = SensorContractV2.from_json(ROOT / "configs/v2/sensor_contract.json")
    protocol = yaml.safe_load(
        (ROOT / "configs/v2/evaluation_protocol.yaml").read_text(encoding="utf-8")
    )
    wind_speed = protocol.get("tuning", {}).get("wind_speed_m_s", 0.5)
    if not isinstance(wind_speed, (int, float)):
        wind_speed = 0.5
    norm_path = ROOT / "artifacts/input_norm_v2/input_norm.npz"
    norm = InputNorm.load(norm_path) if norm_path.exists() else None
    policy = build_policy("VO2", ModelConfigV2(), input_norm=norm).to(device).eval()
    scenarios = ("normal", "occlusion", "noise", "conflict")
    configs = {
        name: yaml.safe_load((ROOT / f"configs/scenarios/{name}.yaml").read_text(encoding="utf-8"))
        for name in scenarios
    }
    results = {}
    for kind in worlds:
        for batch in batches:
            world = V1CompatWorld(
                contract,
                batch,
                scenarios=[scenarios[i % 4] for i in range(batch)],
                options={"episode_seconds": 60, "target_motion": "curved", "plume_model": "puff"},
                scenario_configs=configs,
                device=device,
                wind=WindConfig(speed_m_s=float(wind_speed)) if kind == "wind" else None,
            )
            teacher = OcclusionTeacher(PrivilegedTeacher(contract), walls=world.walls)
            wall = _time_rollout(world, teacher, policy, batch, device, warmup, steps)
            results[f"{kind}/{batch}"] = {
                "ms_per_batched_step": 1000 * wall / steps,
                "env_steps_per_s": batch * steps / wall,
            }
            print(f"rollout {kind} B={batch}: {results[f'{kind}/{batch}']}", flush=True)
            del world
            if device.startswith("cuda"):
                torch.cuda.empty_cache()
    return {
        "worlds": worlds,
        "input_norm": str(norm_path.relative_to(ROOT)) if norm is not None else None,
        "steps_timed": steps,
        "results": results,
        "peak_memory_mib": (
            torch.cuda.max_memory_allocated() / 2**20 if device.startswith("cuda") else None
        ),
    }


def main(argv: list[str] | None = None) -> dict:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("pipeline", choices=["v1", "v2"])
    parser.add_argument("--host-label", required=True, help="host label for the benchmark")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batches", default="1,8,256,1024")
    parser.add_argument("--skip", default="", help="comma list of env, policy, ppo")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--worlds", default="v1compat,wind", help="v2 rollout worlds")
    args = parser.parse_args(argv)
    skip = {item for item in args.skip.split(",") if item}
    result = {"pipeline": args.pipeline, **environment(args.host_label, args.device)}
    if args.pipeline == "v2":
        batches = [int(batch) for batch in args.batches.split(",")]
        worlds = [item for item in args.worlds.split(",") if item]
        result["rollout"] = bench_rollout_v2(args.device, batches, worlds)
        skip |= {"env", "policy", "ppo"}
    if "env" not in skip:
        result["env"] = bench_env()
    if "policy" not in skip:
        batches = [int(batch) for batch in args.batches.split(",")]
        result["policy"] = bench_policy(args.device, batches)
    if "ppo" not in skip:
        result["ppo"] = bench_ppo(args.device)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output = args.output or ROOT / f"runs/bench/{args.pipeline}_{args.host_label}_{stamp}.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    print(f"written {output}")
    return result


if __name__ == "__main__":
    main()
