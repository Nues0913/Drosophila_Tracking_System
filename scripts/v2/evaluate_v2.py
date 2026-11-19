"""Evaluate v2 policy artifacts on the registered worlds, scenarios and motions (#139, #144).

Per artifact (runs/.../<variant>/seed_<n>/<artifact dir>), per world (v1compat, wind) and
motion, one batched world runs the four scenarios at the protocol's noise scale, plus a noise
world per sensitivity scale. Episode seeds come from the protocol's episode_seeds range
(formal_evaluation or pilot_held_out): base + 100000 * train_seed + e, the same e in every
scenario, world and variant, so comparisons pair. Writes episodes.jsonl (docs/v2/metrics.md),
secondary.jsonl (the A3 speed slope and E1 occlusion phases per episode) and manifest.json.

Optional secondary runs on the same episodes:
- --speed-clamp: every actor with its speed held at 0.20 m/s (#161 A3), normal scenario;
- --ablations: VO2 in the wind world's occlusion scenario with the odor swapped and with the
  wind rotated by 90 degrees (#161 E1), each during the occlusion only (the attribution, #192
  R5) and over the whole episode.

The summary (flytrack_v2.eval.summary) is built from episodes.jsonl afterwards.

Usage: python scripts/v2/evaluate_v2.py --artifacts DIR [DIR ...] --seeds formal_evaluation \
           --output runs/evaluation_v2/formal
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import time
from pathlib import Path

import numpy as np
import yaml
from flytrack_v2.contracts import SensorContractV2
from flytrack_v2.eval.runner import (
    ObservationAblation,
    SpeedClamp,
    V2Actor,
    rotate_wind,
    run_episodes,
    swap_odor,
    to_records,
)
from flytrack_v2.metrics import validate_episode_record
from flytrack_v2.model import load_policy
from flytrack_v2.world.v1compat import V1CompatWorld

ROOT = Path(__file__).resolve().parents[2]
SCENARIOS = ("normal", "occlusion", "noise", "conflict")
NOISE_KEYS = (
    "image_noise_std",
    "brightness_std",
    "odor_noise_std",
    "image_dropout",
    "odor_dropout",
)


# #161 E1 ablations, VO2 in the wind world's occlusion scenario: (channel, timing). The
# occlusion-window runs are the registered attribution (#192 R5): paired episodes are identical
# until the occlusion starts, so a change is the channel's use while the target is hidden. The
# whole-episode runs say whether the controller depends on the channel at all.
ABLATIONS = {
    "odor_swap_occlusion": ("odor", "occlusion"),
    "wind_rotate_90_occlusion": ("wind", "occlusion"),
    "odor_swap": ("odor", "episode"),
    "wind_rotate_90": ("wind", "episode"),
}


def occlusion_window(protocol: dict) -> tuple[float, float]:
    occlusion = protocol["scenarios"]["occlusion"]
    start = float(occlusion["occlusion_start_s"])
    return start, start + float(occlusion["occlusion_duration_s"])


def load_artifact(path: Path):
    """(policy on the CPU, card, artifact sha256) through the single v2 loading entry.

    load_policy (#176, #161 C4) verifies SHA256SUMS, the card and the contract and rebuilds the
    policy from the artifact directory alone. The artifact's identity is the sha256 of its
    SHA256SUMS, which covers every file.
    """
    policy = load_policy(path, device="cpu")
    card = json.loads((path / "model_card.json").read_text(encoding="utf-8"))
    digest = hashlib.sha256((path / "SHA256SUMS").read_bytes()).hexdigest()
    return policy, card, digest


def train_seed_of(path: Path, card: dict) -> int:
    seed = card.get("provenance", {}).get("train_seed")
    if seed is None:
        match = re.search(r"seed_(\d+)", str(path))
        if match is None:
            raise ValueError(f"{path}: no train seed in the card or the path (seed_<n>)")
        seed = match.group(1)
    return int(seed)


def scenario_configs(noise_scale: float, protocol: dict) -> dict[str, dict]:
    base = protocol["scenarios"]["noise"]["base_at_scale_1"]
    configs = {}
    for name in SCENARIOS:
        config = yaml.safe_load(
            (ROOT / "configs/scenarios" / f"{name}.yaml").read_text(encoding="utf-8")
        )
        if name == "noise":
            config.update({key: base[key] * noise_scale for key in NOISE_KEYS})
        configs[name] = config
    return configs


def build_world(contract, kind, motion, scenarios, noise_scale, protocol, episodes, device):
    rows = [{"scenario": s, "noise_scale": noise_scale} for s in scenarios for _ in range(episodes)]
    wind = None
    if kind == "wind":
        from flytrack_v2.world.wind import WindConfig  # #169

        wind = WindConfig(speed_m_s=protocol["tuning"]["wind_speed_m_s"])
    world = V1CompatWorld(
        contract,
        num_envs=len(rows),
        scenarios=[row["scenario"] for row in rows],
        options={
            "episode_seconds": protocol["episode_seconds"],
            "target_motion": motion,
            "plume_model": "puff",
            "domain_randomization": False,
        },
        scenario_configs=scenario_configs(noise_scale, protocol),
        device=device,
        reward="track_v2",
        occlusion_view=protocol["occlusion_view"]["v2"],
        **({"wind": wind} if wind is not None else {}),
    )
    return world, rows


def episode_seeds(protocol: dict, use: str, train_seed: int, episodes: int, scenarios: int):
    entry = protocol["episode_seeds"][use]
    one = [entry["base"] + entry["per_train_seed"] * train_seed + e for e in range(episodes)]
    return one * scenarios


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--artifacts", nargs="+", type=Path, required=True)
    parser.add_argument(
        "--protocol", type=Path, default=ROOT / "configs/v2/evaluation_protocol.yaml"
    )
    parser.add_argument("--seeds", choices=("formal_evaluation", "pilot_held_out"), required=True)
    parser.add_argument("--worlds", nargs="+", default=None)
    parser.add_argument("--motions", nargs="+", default=None)
    parser.add_argument("--episodes", type=int, default=None)
    parser.add_argument("--no-sensitivity", action="store_true")
    parser.add_argument("--speed-clamp", action="store_true")
    parser.add_argument("--ablations", action="store_true")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    protocol = yaml.safe_load(args.protocol.read_text(encoding="utf-8"))
    contract = SensorContractV2.from_json(ROOT / "configs/v2/sensor_contract.json")
    worlds = args.worlds or protocol["worlds"]
    motions = args.motions or [protocol["motions"]["primary"], protocol["motions"]["secondary"]]
    episodes = args.episodes or protocol["episodes_per_cell"]
    scale = protocol["scenarios"]["noise"]["scale"]
    sensitivity = (
        [] if args.no_sensitivity else protocol["scenarios"]["noise"]["sensitivity_scales"]
    )
    max_steps = int(np.ceil(protocol["episode_seconds"] / contract.control_dt)) + 5
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=True
    ).stdout.strip()
    args.output.mkdir(parents=True, exist_ok=True)
    runs = []  # (label, world kind, motion, scenarios, noise scale)
    for kind in worlds:
        for motion in motions:
            runs.append(("main", kind, motion, SCENARIOS, scale))
            runs += [("noise_sensitivity", kind, motion, ("noise",), s) for s in sensitivity]
            if args.speed_clamp:
                runs.append(("speed_clamp", kind, motion, ("normal",), scale))
            if args.ablations and kind == "wind":
                runs += [(name, kind, motion, ("occlusion",), scale) for name in ABLATIONS]
    manifest = {
        "commit": commit,
        "protocol_sha256": hashlib.sha256(args.protocol.read_bytes()).hexdigest(),
        "seeds": args.seeds,
        "episodes": episodes,
        "runs": [],
        "artifacts": [],
    }
    # Each artifact is loaded and verified once, kept on the CPU, and moved to the device only
    # for its own runs, so the GPU holds one policy at a time.
    loaded = [(path, *load_artifact(path)) for path in args.artifacts]
    invalid = 0
    with (
        (args.output / "episodes.jsonl").open("w", encoding="utf-8") as records_out,
        (args.output / "secondary.jsonl").open("w", encoding="utf-8") as secondary_out,
    ):
        for label, kind, motion, scenarios, noise_scale in runs:
            world, rows = build_world(
                contract, kind, motion, scenarios, noise_scale, protocol, episodes, args.device
            )
            for path, policy, card, digest in loaded:
                variant, seed = card["variant"], train_seed_of(path, card)
                if label in ABLATIONS and variant != "VO2":
                    continue
                actor = V2Actor(policy.to(args.device))
                if label == "speed_clamp":
                    actor = SpeedClamp(actor, 0.20)
                elif label in ABLATIONS:
                    channel, timing = ABLATIONS[label]
                    transform = (
                        swap_odor
                        if channel == "odor"
                        else rotate_wind(90.0, contract.antenna_angle_deg)
                    )
                    window = occlusion_window(protocol) if timing == "occlusion" else None
                    actor = ObservationAblation(
                        actor, transform, window=window, control_dt=contract.control_dt
                    )
                started = time.perf_counter()
                seeds = episode_seeds(protocol, args.seeds, seed, episodes, len(scenarios))
                result = run_episodes(world, actor, seeds, max_steps=max_steps)
                policy.to("cpu")
                records = to_records(
                    result, rows, run_id=f"{args.output.name}:{label}", variant=variant,
                    train_seed=seed, world=kind, motion=motion, artifact_sha256=digest,
                    commit=commit,
                )  # fmt: skip
                for i, record in enumerate(records):
                    try:
                        validate_episode_record(record)
                    except ValueError:
                        invalid += 1
                    if label in ("main", "noise_sensitivity"):
                        records_out.write(json.dumps(record) + "\n")
                    extra = {key: values[i] for key, values in result.speed_distance.items()}
                    extra.update({k: v[i] for k, v in result.occlusion_phases.items()})
                    extra.update(
                        run=label, variant=variant, train_seed=seed, world=kind, motion=motion,
                        scenario=record["scenario"], noise_scale=record["noise_scale"],
                        episode_seed=record["episode_seed"], success=record["success"],
                        collision=record["collision"],
                        tracking_fraction=record["tracking_fraction"],
                    )  # fmt: skip
                    secondary_out.write(json.dumps(extra) + "\n")
                seconds = time.perf_counter() - started
                manifest["runs"].append(
                    {
                        "run": label,
                        "world": kind,
                        "motion": motion,
                        "scenarios": list(scenarios),
                        "noise_scale": noise_scale,
                        "variant": variant,
                        "train_seed": seed,
                        "seconds": seconds,
                    }
                )
                print(f"{label} {kind} {motion} {variant} seed {seed}: {seconds:.0f} s", flush=True)
    manifest["artifacts"] = [str(p) for p in args.artifacts]
    manifest["invalid_records"] = invalid
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=1) + "\n")
    print(f"{invalid} records failed validate_episode_record", flush=True)


if __name__ == "__main__":
    main()
