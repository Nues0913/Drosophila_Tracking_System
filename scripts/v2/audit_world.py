"""World audit for every world, scenario and target motion (#132, #133; plan 3.3).

``python scripts/v2/audit_world.py --host-label 4070`` runs flytrack_v2.world.audit for the
four scenarios and both motions (100 episodes each by default) and writes one JSON to
runs/audit/. Exit status 1 if any cell fails.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python"))

from flytrack_v2.contracts import SCENARIOS, SensorContractV2  # noqa: E402
from flytrack_v2.world.audit import audit_world  # noqa: E402
from flytrack_v2.world.wind import WindConfig  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--host-label", required=True)
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--worlds", default="v1compat,wind")
    parser.add_argument("--wind-speed", type=float, default=WindConfig().speed_m_s)
    args = parser.parse_args(argv)
    contract = SensorContractV2.from_json(ROOT / "configs/v2/sensor_contract.json")
    configs = {
        name: yaml.safe_load((ROOT / f"configs/scenarios/{name}.yaml").read_text())
        for name in SCENARIOS
    }
    cells = []
    for world_kind in args.worlds.split(","):
        for motion in ("curved", "wander"):
            for scenario in SCENARIOS:
                wind = WindConfig(speed_m_s=args.wind_speed) if world_kind == "wind" else None
                report = audit_world(
                    contract,
                    scenario,
                    {"episode_seconds": 60, "target_motion": motion},
                    episodes=args.episodes,
                    device=args.device,
                    scenario_configs=configs,
                    wind=wind,
                )
                report["target_motion"] = motion
                report["world"] = world_kind
                if wind is not None:
                    report["wind"] = vars(wind)
                best = max(report["blind_policies"].items(), key=lambda kv: kv[1]["success_rate"])
                odor = report.get("occlusion_teachers", {})
                extra = "".join(f" {k}={v['success_rate']:.2f}" for k, v in odor.items())
                print(
                    f"{world_kind:8s} {motion:7s} {scenario:9s} passed={report['passed']} "
                    f"teacher={report['teacher']['success_rate']:.2f} "
                    f"best blind {best[0]}={best[1]['success_rate']:.2f}{extra}",
                    flush=True,
                )
                cells.append(report)
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True
    ).stdout.strip()
    dirty = bool(
        subprocess.run(
            ["git", "status", "--porcelain"], cwd=ROOT, capture_output=True, text=True
        ).stdout.strip()
    )
    out = {
        "audit": "world-v1compat",
        "host_label": args.host_label,
        "commit": commit,
        "dirty": dirty,
        "utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "device": args.device,
        "passed": all(cell["passed"] for cell in cells),
        "cells": cells,
    }
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = args.output or ROOT / f"runs/audit/world_{args.host_label}_{stamp}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print(f"written {path} passed={out['passed']}")
    return 0 if out["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
