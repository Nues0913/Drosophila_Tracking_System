"""Throughput of the v2 worlds (#132, #133 acceptance; plan 3.3: 1,024 envs <= 5 ms/step).

``python scripts/v2/bench_world.py --host-label 4070`` steps the world without a model, with a
pursuit action computed from the privileged state (so episodes end mostly by time, as in
training), and reports milliseconds per step for each batch size and scenario mix:

- steady: steps in which no row ended;
- one_reset: steps in which exactly one row ended (the reset graph runs);
- expected: the mean over a 60 s episode schedule, P(reset) = 1 - exp(-B / 1200).

The JSON goes to runs/bench/ and names the host only by --host-label.
"""

from __future__ import annotations

import argparse
import json
import math
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import torch
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python"))

from flytrack_v2.contracts import SensorContractV2  # noqa: E402
from flytrack_v2.world.v1compat import V1CompatWorld  # noqa: E402
from flytrack_v2.world.wind import WindConfig  # noqa: E402

SCENARIOS = ("normal", "occlusion", "noise", "conflict")


def _git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True).stdout.strip()


def _pursuit(world: V1CompatWorld) -> torch.Tensor:
    p = world.privileged()
    d = p["target_pose"][:, :2] - p["agent_pose"][:, :2]
    bearing = torch.atan2(d[:, 1], d[:, 0]) - p["agent_pose"][:, 2]
    bearing = torch.remainder(bearing + math.pi, 2 * math.pi) - math.pi
    return torch.stack(((d.norm(dim=-1) - 1.0).clamp(0, 0.5), (2 * bearing).clamp(-1.5, 1.5)), -1)


def bench(
    batch: int, scenarios: tuple[str, ...], steps: int, device: str, world_kind: str = "v1compat"
) -> dict:
    contract = SensorContractV2.from_json(ROOT / "configs/v2/sensor_contract.json")
    configs = {
        name: yaml.safe_load((ROOT / f"configs/scenarios/{name}.yaml").read_text())
        for name in SCENARIOS
    }
    world = V1CompatWorld(
        contract,
        batch,
        [scenarios[i % len(scenarios)] for i in range(batch)],
        {"episode_seconds": 60, "target_motion": ["curved", "wander"]},
        device=device,
        scenario_configs=configs,
        wind=WindConfig() if world_kind == "wind" else None,
    )
    world.reset()
    for _ in range(20):  # graph capture and warm-up
        world.step(_pursuit(world))
    action = _pursuit(world)

    def timed(force_reset: bool) -> float:
        torch.cuda.synchronize()
        start = time.perf_counter()
        for i in range(steps):
            if force_reset:
                world._state["time"][i % batch] = world.episode_seconds  # ends this step
            world.step(action)
        torch.cuda.synchronize()
        return 1e3 * (time.perf_counter() - start) / steps

    steady, one_reset = timed(False), timed(True)
    p_reset = 1 - math.exp(-batch / world.max_steps)
    return {
        "world": world_kind,
        "batch": batch,
        "scenarios": list(scenarios),
        "steady_ms": round(steady, 3),
        "one_reset_ms": round(one_reset, 3),
        "expected_ms": round(steady + p_reset * (one_reset - steady), 3),
        "env_steps_per_s": round(batch / (steady + p_reset * (one_reset - steady)) * 1e3),
        "peak_mib": round(torch.cuda.max_memory_allocated() / 2**20),
        "graph": world.graph,
    }


def main(argv: list[str] | None = None) -> dict:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--host-label", required=True, help="2060, 4060 or 4070")
    parser.add_argument("--batches", default="256,1024,4096")
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--worlds", default="v1compat,wind")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    if not torch.cuda.is_available():
        raise SystemExit("the world bench needs CUDA")
    results = []
    for world_kind in args.worlds.split(","):
        for batch in (int(b) for b in args.batches.split(",")):
            for scenarios in (("normal",), SCENARIOS):
                torch.cuda.reset_peak_memory_stats()
                results.append(bench(batch, scenarios, args.steps, "cuda", world_kind))
                print(json.dumps(results[-1]), flush=True)
                torch.cuda.empty_cache()
    report = {
        "mode": "world",
        "host_label": args.host_label,
        "commit": _git("rev-parse", "HEAD"),
        "dirty": bool(_git("status", "--porcelain")),
        "utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "gpu": torch.cuda.get_device_name(),
        "results": results,
    }
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output = args.output or ROOT / f"runs/bench/world_{args.host_label}_{stamp}.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"written {output}")
    return report


if __name__ == "__main__":
    main()
