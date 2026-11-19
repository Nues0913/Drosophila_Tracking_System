"""Measure fixed trajectories under the track_v2 reward (#161 D3)."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from statistics import mean

import torch
import yaml
from flytrack_v2.contracts import SensorContractV2
from flytrack_v2.world.teacher import PrivilegedTeacher
from flytrack_v2.world.v1compat import V1CompatWorld, wrap

ROOT = Path(__file__).resolve().parents[2]
GAMMAS = (0.95, 0.99, 0.999)
# Dedicated reward_tests seed range, separate from world_audit (#173 review).
SEED_START = 1_640_000_000
MOTIONS = ("curved", "wander")
TRAJECTORIES = ("A", "B1", "B2", "C10", "C10stop", "C30", "D")
SPEED_FRACTIONS = {"B1": 0.9, "B2": 0.6}
SWITCH_SECONDS = {"C10": 10, "C10stop": 10, "C30": 30, "D": 10}
STOP_WALL_DISTANCE_M = 0.5
RETURN_FIELDS = (
    "return",
    *(f"discounted_return_{gamma}" for gamma in GAMMAS),
    *(f"discounted_return_from_switch_{gamma}" for gamma in GAMMAS),
)
EPISODE_FIELDS = ("steps", "collision", "tracking_fraction", "mean_distance_m")


def _wall_distances(pose, walls):
    x, y = pose[:, :2].unbind(-1)
    xmin, ymin, xmax, ymax = walls
    return torch.stack((x - xmin, y - ymin, xmax - x, ymax - y), dim=-1)


@torch.no_grad()
def measure_trajectories(
    motion: str,
    *,
    rows: int = 8,
    episode_seconds: int = 60,
    device: str = "cpu",
    trajectories: tuple[str, ...] = TRAJECTORIES,
) -> dict:
    """Run each actor on the same seeds and record one episode per row."""
    if "C10stop" in trajectories and "C10" not in trajectories:
        raise ValueError("C10stop requires C10 to verify the shared path")
    contract = SensorContractV2.from_json(ROOT / "configs/v2/sensor_contract.json")
    normal = yaml.safe_load((ROOT / "configs/scenarios/normal.yaml").read_text(encoding="utf-8"))
    batch = rows * len(trajectories)
    world = V1CompatWorld(
        contract,
        batch,
        "normal",
        {"episode_seconds": episode_seconds, "target_motion": motion, "plume_model": "puff"},
        device=device,
        scenario_configs={"normal": normal},
        reward="track_v2",
    )
    teacher = PrivilegedTeacher(contract, world.following)
    dt = contract.control_dt
    state = teacher.initial_state(batch, world.device)
    seeds = list(range(SEED_START, SEED_START + rows))
    obs = world.reset(episode_seeds=torch.tensor(seeds, device=device).repeat(len(trajectories)))
    target_speed = world.trajectory.speed_m_s  # norm of target_velocity for both motions
    speeds = {name: fraction * target_speed for name, fraction in SPEED_FRACTIONS.items()}
    slices = {name: slice(i * rows, (i + 1) * rows) for i, name in enumerate(trajectories)}
    switch_steps = {name: round(s / dt) for name, s in SWITCH_SECONDS.items()}
    headings = {}
    wall_indices = {}
    stop_block = slices.get("C10stop")
    if stop_block is not None:
        braking = torch.zeros(rows, dtype=torch.bool, device=device)
        matched_prefix = torch.ones_like(braking)
        stop_wall_distance = torch.zeros(rows, dtype=torch.float64, device=device)
        stop_speed = torch.zeros_like(stop_wall_distance)
        speed_step = world.limits.linear_accel_max * dt
        # For each initial braking speed, sum the speeds applied by the world on
        # the current step and all subsequent steps under maximum deceleration.
        n = math.ceil(world.limits.v_max / speed_step)
        brake_speeds = torch.arange(n + 1, dtype=torch.float64, device=device) * speed_step
        brake_distances = brake_speeds.cumsum(dim=0) * dt
    rewards = torch.zeros(world.max_steps, batch, dtype=torch.float64, device=device)
    done = torch.zeros(batch, dtype=torch.bool, device=device)
    metrics = {
        field: torch.zeros(batch, dtype=torch.float64, device=device) for field in EPISODE_FIELDS
    }

    for step in range(world.max_steps):
        privileged = world.privileged()
        action, state = teacher.act(privileged, obs, privileged["agent_velocity"], state)
        for name, block in slices.items():
            if name in speeds:
                action[block, 0] = speeds[name]
            if name not in switch_steps or step < switch_steps[name]:
                continue
            pose = privileged["agent_pose"][block]
            if step == switch_steps[name]:
                # Choose the wall once at the switch. Opposite that wall, D has room to drift.
                wall_indices[name] = _wall_distances(pose, world.walls).argmin(dim=1, keepdim=True)
                normals = pose.new_tensor((math.pi, -math.pi / 2, 0, math.pi / 2))
                headings[name] = normals[wall_indices[name][:, 0]] + (math.pi if name == "D" else 0)
            error = wrap(headings[name] - pose[:, 2])
            action[block, 1] = teacher.turn_gain * error
            if name == "D":
                action[block, 0] = 0.05
            else:
                # Turn in place, then drive at v_max along the outward normal.
                action[block, 0] = torch.where(error.abs() <= 0.1, world.limits.v_max, 0.0)
            if name == "C10stop":
                clearance = _wall_distances(pose, world.walls).gather(1, wall_indices[name])[:, 0]
                remaining = (clearance.to(torch.float64) - STOP_WALL_DISTANCE_M).clamp_min(0.0)
                # Interpolate the braking curve to use the last full-speed step
                # possible and still settle at the requested wall clearance.
                hi = torch.searchsorted(brake_distances, remaining).clamp(1, n)
                lo = hi - 1
                fraction = (remaining - brake_distances[lo]) / (
                    brake_distances[hi] - brake_distances[lo]
                )
                allowed_speed = torch.lerp(brake_speeds[lo], brake_speeds[hi], fraction.clamp(0, 1))
                start_braking = ~braking & (action[block, 0] > allowed_speed)
                action[block] = torch.where(braking[:, None], 0.0, action[block])
                action[block, 0] = torch.where(start_braking, allowed_speed, action[block, 0])
                action[block, 1] = torch.where(start_braking, 0.0, action[block, 1])
                braking |= start_braking

        if stop_block is not None:
            # Compare the requests exactly, before braking or the first episode ends.
            same_action = (action[stop_block] == action[slices["C10"]]).all(dim=1)
            matched_prefix &= done[stop_block] | braking | same_action

        result = world.step(action)
        obs = result.obs
        rewards[step] = torch.where(~done, result.reward, 0.0)
        ended = ~done & (result.terminated | result.truncated)
        for field, values in metrics.items():
            metrics[field] = torch.where(ended, result.info[f"episode_{field}"], values)
        if stop_block is not None:
            final = result.final_privileged
            distance = _wall_distances(final["agent_pose"][stop_block], world.walls).amin(dim=1)
            stop_wall_distance = torch.where(ended[stop_block], distance, stop_wall_distance)
            speed = final["agent_velocity"][stop_block, 0]
            stop_speed = torch.where(ended[stop_block], speed, stop_speed)
        done |= ended
        if (step + 1) % 20 == 0 and bool(done.all()):
            break
    if not bool(done.all()):
        raise RuntimeError("a fixed-trajectory episode did not end within max_steps")
    if stop_block is not None and not bool(matched_prefix.all()):
        raise RuntimeError("C10stop actions differ from C10 before braking")

    steps = torch.arange(world.max_steps, dtype=torch.float64, device=device)
    weights = [gamma**steps for gamma in GAMMAS]

    def discounted(start_s, block):
        start = round(start_s / dt)
        tail = rewards[start:, block]
        # The first reward after the switch has weight 1, including a collision on that step.
        return [(tail * weight[: len(tail), None]).sum(dim=0) for weight in weights]

    cells = {}
    for name, block in slices.items():
        start_s = SWITCH_SECONDS.get(name, 10)
        values = {
            **dict(
                zip(
                    RETURN_FIELDS,
                    [
                        rewards[:, block].sum(dim=0),
                        *discounted(0, block),
                        *discounted(start_s, block),
                    ],
                    strict=True,
                )
            ),
            **{field: tensor[block] for field, tensor in metrics.items()},
        }
        if name == "C10stop":
            values["final_wall_distance_m"] = stop_wall_distance
            values["final_speed_m_s"] = stop_speed
        per_row = {key: tensor.tolist() for key, tensor in values.items()}
        per_row["collision"] = [bool(value) for value in per_row["collision"]]
        per_row["steps"] = [int(value) for value in per_row["steps"]]
        cells[name] = {
            "discount_from_s": start_s,
            "episodes": [
                {"seed": seed, **{key: column[i] for key, column in per_row.items()}}
                for i, seed in enumerate(seeds)
            ],
            "summary": {
                key: {"mean": mean(column), "min": min(column), "max": max(column)}
                for key, column in per_row.items()
            },
        }
    return {
        "target_speed_m_s": target_speed,
        "requested_speeds_m_s": {
            **speeds,
            "C10": world.limits.v_max,
            "C10stop": world.limits.v_max,
            "C30": world.limits.v_max,
            "D": 0.05,
        },
        "trajectories": cells,
    }


def _print_table(motions: dict) -> None:
    for motion, report in motions.items():
        print(f"{motion}: target {report['target_speed_m_s']:.3f} m/s", flush=True)
    discount_labels = [f"{start}/{gamma}" for start in ("0s", "switch") for gamma in GAMMAS]
    returns_header = " ".join(f"{label:>12}" for label in ("return", *discount_labels))
    print(
        f"motion    actor  speed m/s {returns_header}  switch s    steps  collision"
        "   tracking   distance m"
    )
    for motion, report in motions.items():
        for name, cell in report["trajectories"].items():
            row = {key: stats["mean"] for key, stats in cell["summary"].items()}
            speed = report["requested_speeds_m_s"].get(name)
            speed_label = "teacher" if speed is None else f"{speed:.3f}"
            returns = " ".join(f"{row[field]:12.3f}" for field in RETURN_FIELDS)
            print(
                f"{motion:7} {name:>7} {speed_label:>10} {returns} "
                f"{cell['discount_from_s']:9d} {row['steps']:8.1f} {row['collision']:10.1%} "
                f"{row['tracking_fraction']:10.1%} {row['mean_distance_m']:12.2f}"
            )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--rows", type=int, default=8, help="paired seeds per motion (default: 8)")
    parser.add_argument("--out", type=Path, default=ROOT / "runs/track_v2_trajectories.json")
    args = parser.parse_args()
    if args.rows < 1:
        parser.error("--rows must be positive")
    if torch.device(args.device).type == "cpu":
        torch.set_num_threads(1)
    motions = {}
    for motion in MOTIONS:
        print(f"Measuring {motion} on {args.device} ({args.rows} rows per actor)", flush=True)
        motions[motion] = measure_trajectories(motion, rows=args.rows, device=args.device)
    report = {
        "scenario": "normal",
        "reward": "track_v2",
        "plume_model": "puff",
        "episode_seconds": 60,
        "gammas": GAMMAS,
        "device": args.device,
        "rows_per_motion": args.rows,
        "motions": motions,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    _print_table(motions)
    print(f"Written {args.out}")


if __name__ == "__main__":
    main()
