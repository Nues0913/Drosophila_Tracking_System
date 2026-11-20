"""Evaluate v2 policy artifacts on the registered worlds, scenarios and motions (#139, #144).

Per artifact (runs/.../<variant>/seed_<n>/<artifact dir>), per world (v1compat, wind) and
motion, one batched world runs the four scenarios at the protocol's noise scale, plus a noise
world per sensitivity scale. Episode seeds come from the protocol's episode_seeds range
(formal_evaluation or pilot_held_out): base + 100000 * train_seed + e, the same e in every
scenario, world and variant, so comparisons pair. Writes episodes.jsonl (docs/v2/metrics.md),
secondary.jsonl (the A3 speed slope and E1 occlusion phases per episode) and manifest.json.

Optional secondary runs on the same episodes:
- --speed-clamp: every actor with its speed held at 0.20 m/s (#161 A3), normal scenario;
- --ablations: VO2 and GRU2 in the wind world's occlusion scenario with the odor swapped and
  with the wind rotated by 90 degrees (#161 E1), each during the occlusion only (the
  attribution, #192 R5, against occlusion_baseline: the same occlusion-only batch unablated)
  and over the whole episode. GRU2's is descriptive (#141).

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
from flytrack_v2.eval.records import episode_numbers
from flytrack_v2.eval.runner import (
    ObservationAblation,
    SpeedClamp,
    V1Actor,
    V2Actor,
    json_line,
    rotate_wind,
    run_episodes,
    swap_odor,
    to_records,
)
from flytrack_v2.metrics import V1_REFERENCE_PREFIX, validate_episode_record
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
# occlusion-window runs are the registered attribution (#192 R5): paired episodes agree until
# the occlusion starts up to float32 rounding, so a change is the channel's use while the target
# is hidden. Their control is occlusion_baseline, the same occlusion-only batch without an
# ablation: a different batch composition would reorder the float operations (#194 review).
# The whole-episode runs say whether the controller depends on the channel at all.
ABLATIONS = {
    "occlusion_baseline": (None, "episode"),
    "odor_swap_occlusion": ("odor", "occlusion"),
    "wind_rotate_90_occlusion": ("wind", "occlusion"),
    "odor_swap": ("odor", "episode"),
    "wind_rotate_90": ("wind", "episode"),
}


# The runs whose positions are compared when the occlusion starts (#194 review): every
# occlusion-window ablation against occlusion_baseline, the same batch unablated.
PREMISE_RUNS = ("occlusion_baseline", "odor_swap_occlusion", "wind_rotate_90_occlusion")

# The variants the ablations run on: VO2's E1 is the registered attribution; GRU2's, on the same
# inputs and episodes, is descriptive and says whether a learned controller uses odor or wind
# while the target is hidden (registered before M4, #141).
E1_VARIANTS = ("VO2", "GRU2")


def e1_premise(snapshots: dict, variants: dict | None = None) -> list[dict]:
    """Per artifact, motion and window ablation: how far each paired episode's agent is from
    the baseline's when the occlusion starts (max, and the fraction farther than 1 cm).
    ``variants`` maps an artifact to its variant, so each variant's E1 reads its own pairs."""
    rows = []
    for (path, motion, label), xy in sorted(snapshots.items()):
        if label == "occlusion_baseline":
            continue
        base = np.asarray(snapshots[(path, motion, "occlusion_baseline")])
        distance = np.linalg.norm(np.asarray(xy) - base, axis=1)
        row = {
            "artifact": path,
            "motion": motion,
            "run": label,
            "max_m": float(distance.max()),
            "fraction_over_1cm": float((distance > 0.01).mean()),
        }
        if variants is not None:
            row["variant"] = variants[path]
        rows.append(row)
    return rows


def occlusion_window(protocol: dict, world_config: dict | None = None) -> tuple[float, float]:
    """The protocol's occlusion [start, end) s; with ``world_config`` (the occlusion scenario
    file the world reads) the two must agree (#194 review)."""
    occlusion = protocol["scenarios"]["occlusion"]
    start = float(occlusion["occlusion_start_s"])
    window = start, start + float(occlusion["occlusion_duration_s"])
    if world_config is not None:
        world_start = float(world_config["occlusion_start_s"])
        world_window = world_start, world_start + float(world_config["occlusion_duration_s"])
        if world_window != window:
            raise ValueError(
                f"the protocol's occlusion {window} differs from the world's {world_window}"
            )
    return window


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
    """The card's provenance.train_seed; the path's seed_<n> only when the card has none (#192 R3).

    The training seed picks the episode numbers, so a wrong one silently changes what an actor is
    evaluated on. A card seed that no seed_<n> component of the path agrees with, or a path
    with two different seed_<n> components and no card seed, is an error.
    """
    seed = card.get("provenance", {}).get("train_seed")
    in_path = {int(part[5:]) for part in Path(path).parts if re.fullmatch(r"seed_\d+", part)}
    if seed is not None:
        if type(seed) is not int or seed < 0:
            raise ValueError(f"{path}: provenance.train_seed must be a nonnegative int")
        if in_path and seed not in in_path:
            raise ValueError(f"{path}: the card's train seed {seed} contradicts the path")
        return seed
    if len(in_path) != 1:
        raise ValueError(
            f"{path}: no train seed in the card, and the path has {len(in_path)} seed_<n> parts"
        )
    return in_path.pop()


def load_v1_artifact(path: Path):
    """(policy on the CPU, card, weights sha256) of a formal-v1 actor <variant>/seed_<n>/actor.

    Its records carry the variant "v1:<variant>" and the seed from the path, which formal-v1's
    layout fixes (release formal-v1); evaluation uses that seed's v2 episode numbers, so G-D
    pairs VO2 and v1 VO-late of one training seed on the same episodes (#192 R2).
    """
    from flytrack_core.model import load_policy_artifact

    seed_dir, variant_dir = path.parent, path.parent.parent
    if path.name != "actor" or not re.fullmatch(r"seed_\d+", seed_dir.name):
        raise ValueError(f"{path}: a formal-v1 actor is <variant>/seed_<n>/actor")
    policy, _, card = load_policy_artifact(path, device="cpu")
    policy.eval()
    card = {
        **card,
        "v2_variant": f"{V1_REFERENCE_PREFIX}{variant_dir.name}",
        "v2_train_seed": int(seed_dir.name[5:]),
    }
    return policy, card, card["files"]["weights.pt"]


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
    return list(episode_numbers(protocol, use, train_seed, episodes)) * scenarios


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--artifacts", nargs="+", type=Path, required=True)
    parser.add_argument(
        "--v1-artifacts",
        nargs="*",
        type=Path,
        default=[],
        help="formal-v1 actors (<variant>/seed_<n>/actor), evaluated in v1compat's main run",
    )
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
        # every actor, formal-v1's included, sees the v2 occlusion view (#197 review)
        "occlusion_view": protocol["occlusion_view"]["v2"],
    }
    # Each artifact is loaded and verified once, kept on the CPU, and moved to the device only
    # for its own runs, so the GPU holds one policy at a time.
    loaded = [("v2", path, *load_artifact(path)) for path in args.artifacts]
    loaded += [("v1", path, *load_v1_artifact(path)) for path in args.v1_artifacts]
    snapshots: dict[tuple, list] = {}
    snapshot_variants: dict[str, str] = {}
    invalid = 0
    with (
        (args.output / "episodes.jsonl").open("w", encoding="utf-8") as records_out,
        (args.output / "secondary.jsonl").open("w", encoding="utf-8") as secondary_out,
    ):
        for label, kind, motion, scenarios, noise_scale in runs:
            world, rows = build_world(
                contract, kind, motion, scenarios, noise_scale, protocol, episodes, args.device
            )
            for generation, path, policy, card, digest in loaded:
                if generation == "v1":
                    # G-D's reference (#192 R2): v1 has no wind sense and is compared in
                    # v1compat only, on the same episodes as the v2 actors of its seed.
                    if label != "main" or kind != "v1compat":
                        continue
                    variant, seed = card["v2_variant"], card["v2_train_seed"]
                else:
                    variant, seed = card["variant"], train_seed_of(path, card)
                if label in ABLATIONS and variant not in E1_VARIANTS:
                    continue
                actor = (V1Actor if generation == "v1" else V2Actor)(policy.to(args.device))
                if label == "speed_clamp":
                    actor = SpeedClamp(actor, 0.20)
                elif label in ABLATIONS and ABLATIONS[label][0] is not None:
                    channel, timing = ABLATIONS[label]
                    transform = (
                        swap_odor
                        if channel == "odor"
                        else rotate_wind(90.0, contract.antenna_angle_deg)
                    )
                    window = None
                    if timing == "occlusion":
                        world_config = scenario_configs(noise_scale, protocol)["occlusion"]
                        window = occlusion_window(protocol, world_config)
                    actor = ObservationAblation(
                        actor, transform, window=window, control_dt=contract.control_dt
                    )
                started = time.perf_counter()
                seeds = episode_seeds(protocol, args.seeds, seed, episodes, len(scenarios))
                snapshot_step = None
                if label in ABLATIONS and label in PREMISE_RUNS:
                    start = occlusion_window(protocol)[0]
                    snapshot_step = round(start / contract.control_dt)
                result = run_episodes(
                    world, actor, seeds, max_steps=max_steps, snapshot_step=snapshot_step
                )
                if result.snapshot_xy is not None:
                    snapshots[(str(path), motion, label)] = result.snapshot_xy
                    snapshot_variants[str(path)] = variant
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
                        records_out.write(json_line(record))
                    extra = {key: values[i] for key, values in result.speed_distance.items()}
                    extra.update({k: v[i] for k, v in result.occlusion_phases.items()})
                    extra.update(
                        run=label, variant=variant, train_seed=seed, world=kind, motion=motion,
                        scenario=record["scenario"], noise_scale=record["noise_scale"],
                        episode_seed=record["episode_seed"], success=record["success"],
                        collision=record["collision"],
                        tracking_fraction=record["tracking_fraction"],
                    )  # fmt: skip
                    secondary_out.write(json_line(extra))
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
    manifest["e1_premise"] = e1_premise(snapshots, snapshot_variants)
    manifest["artifacts"] = [str(p) for p in args.artifacts]
    manifest["v1_artifacts"] = [str(p) for p in args.v1_artifacts]
    manifest["invalid_records"] = invalid
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=1) + "\n")
    print(f"{invalid} records failed validate_episode_record", flush=True)


if __name__ == "__main__":
    main()
