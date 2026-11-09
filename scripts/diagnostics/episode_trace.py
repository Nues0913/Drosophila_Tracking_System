"""Replay selected CPU episodes using exactly lesion_analysis's inference path.

The JSONL records describe the observation on which the policy made its decision:
time_s, agent_pose, target_pose, teacher_action, target_distance_m,
target_bearing_rad, target_visible and occluded come from reset's info (step 0),
then the preceding step's info. In particular teacher_action NEVER comes from
the info returned after applying the action being labelled. applied_action,
collision, post_time_s, post_agent_pose and post_target_pose come from that step.
EpisodeMetrics deliberately consumes post-step info, exactly as lesion_analysis.

Actions are [v (m/s), omega (rad/s)], poses are [x (m), y (m), yaw (rad)].
Current mass is absolute incoming recurrent edge-current mass, not attribution.
Segmentation uses the decision-time occluded flag, including at window boundaries.
Descriptive turning/speed statistics and teacher comparisons use post-step
info["applied_action"]. Like teacher_action, it is clipped with the same previous
action, speed bounds and acceleration limits, making comparisons consistent.
Raw policy action and applied_action are both retained in the trace. Integrals use the actual
elapsed step duration (including a shortened terminal collision step).
"""

from __future__ import annotations

import argparse
import contextlib
import gzip
import json
from pathlib import Path

import numpy as np

if __package__:
    from ._a1_common import clean, provenance, sha256, write_csv, write_json
else:
    from _a1_common import clean, provenance, sha256, write_csv, write_json

MASS_NAMES = ("visual_exclusive", "odor_exclusive", "shared")
NOT_AVAILABLE = "NOT AVAILABLE"
REQUIRED_DN_TYPES = ("DNp09", "DNa01", "DNa02", "DNb05")


class DNRecorder:
    """Observe the actual structured readout inputs without changing any tensors.

    Only scalar snapshots survive a forward call; pop() clears each step and
    the context manager removes the hook even when replay raises an exception.
    """

    def __init__(self, policy, card, lesion):
        if lesion != "none":
            raise ValueError("--record-dn requires --lesion none")
        self.readout = getattr(policy, "readout", None)
        if card["variant"] == "GRU" or getattr(self.readout, "mode", None) != "structured":
            raise ValueError("--record-dn requires a structured DN readout; GRU/linear unsupported")
        readout = self.readout
        if readout.branches not in (1, 2):
            raise ValueError("--record-dn requires one or two readout branches")
        graph = readout._graph
        used = (readout.difference != 0).any(0) | (readout.forward_pool != 0).any(0)
        for name in ("extra_left", "extra_right"):
            pool = getattr(readout, name, None)
            if pool is not None:
                used = used | (pool != 0).any(0)
        used = used.cpu().numpy()
        graph_types = {
            str(graph.type_names[type_id]): type_id
            for type_id in sorted(set(graph.type_index.tolist()))
        }
        self.in_readout = {
            kind: bool(used[graph.type_index == type_id].any())
            for kind, type_id in graph_types.items()
        }
        self.pools = {}
        for kind, type_id in graph_types.items():
            if not self.in_readout[kind] and kind not in REQUIRED_DN_TYPES:
                continue
            nodes = graph.type_index == type_id
            self.pools[kind] = {
                name: np.flatnonzero(mask).tolist()
                for name, mask in (
                    ("left", nodes & (graph.side == -1)),
                    ("right", nodes & (graph.side == 1)),
                    ("all", nodes),
                )
            }
        missing = set(REQUIRED_DN_TYPES) - graph_types.keys()
        missing_sides = {
            kind: [side for side in ("left", "right") if not self.pools[kind][side]]
            for kind in REQUIRED_DN_TYPES
            if kind in self.pools and any(not self.pools[kind][side] for side in ("left", "right"))
        }
        if missing or missing_sides:
            raise ValueError(
                f"--record-dn missing required DN types in graph: {sorted(missing)}; "
                f"missing required DN sides: {missing_sides}; graph types: {sorted(graph_types)}"
            )
        self.required_types_not_in_readout = [
            kind for kind in REQUIRED_DN_TYPES if not self.in_readout[kind]
        ]
        self.pending = None
        self.handle = None

    def __enter__(self):
        self.handle = self.readout.register_forward_hook(self._capture)
        return self

    def _capture(self, readout, inputs, output):
        import torch

        if self.pending is not None:
            raise RuntimeError("DN snapshot was not consumed before the next readout call")
        activities = inputs[0]
        names = ("central",) if readout.branches == 1 else ("visual", "odor")
        with torch.no_grad():
            branches = [
                {
                    "name": name,
                    "types": {
                        kind: {
                            **{
                                side: float(activity[0, indices].mean().cpu()) if indices else None
                                for side, indices in pools.items()
                            },
                            "in_readout": self.in_readout[kind],
                        }
                        for kind, pools in self.pools.items()
                    },
                }
                for name, activity in zip(names, activities, strict=True)
            ]
            forward = torch.cat([a @ readout.forward_pool.T for a in activities], dim=1)
            self.pending = {
                "branches": branches,
                "forward_latent": float(
                    (forward @ readout.forward_weight + readout.forward_bias)[0]
                ),
                "omega_latent": float(readout.omega_latent(activities)[0]),
            }

    def pop(self):
        if self.pending is None:
            raise RuntimeError("Policy did not call the structured readout")
        snapshot, self.pending = self.pending, None
        return snapshot

    def __exit__(self, *exc):
        self.handle.remove()
        self.pending = None


def collision_types(env, agent_pose, target_pose) -> list[str] | str:
    """Apply env._has_collision's exact geometry; retain simultaneous contacts.

    No epsilon is introduced: wall inequalities are strict, disk contacts use <=.
    Missing or unusable geometry yields NOT AVAILABLE instead of an attribution.
    """
    try:
        world = env.unwrapped
        walls = np.asarray(world.walls)
        obstacles = world.obstacles
        agent_radius = world.agent_radius_m
        target_radius = world.target_radius_m
        xy = np.asarray(agent_pose)[:2]
        target = np.asarray(target_pose)[:2]
        if walls.shape != (4,) or xy.shape != (2,) or target.shape != (2,):
            return NOT_AVAILABLE
        if not np.all(np.isfinite([*walls, *xy, *target, agent_radius, target_radius])):
            return NOT_AVAILABLE
        result = []
        if np.any(xy < walls[:2] + agent_radius) or np.any(xy > walls[2:] - agent_radius):
            result.append("wall")
        if np.linalg.norm(xy - target) <= agent_radius + target_radius:
            result.append("target")
        obstacle_hit = False
        for obstacle in obstacles:
            if not np.all(np.isfinite([obstacle["x"], obstacle["y"], obstacle["radius"]])):
                return NOT_AVAILABLE
            if (
                np.linalg.norm(xy - [obstacle["x"], obstacle["y"]])
                <= agent_radius + obstacle["radius"]
            ):
                obstacle_hit = True
        if obstacle_hit:
            result.append("obstacle")
        return result
    except (AttributeError, KeyError, TypeError, ValueError, IndexError):
        return NOT_AVAILABLE


def nearest_wall_clearance(env, pose) -> float | None:
    """Signed clearance from the agent disk to the rectangular wall boundary."""
    try:
        world = env.unwrapped
        walls = np.asarray(world.walls)
        xy = np.asarray(pose)[:2]
        if walls.shape != (4,):
            return None
        return float(min(*(xy - walls[:2]), *(walls[2:] - xy)) - world.agent_radius_m)
    except (AttributeError, TypeError, ValueError):
        return None


def angle_difference(a: float, b: float) -> float:
    return float((a - b + np.pi) % (2 * np.pi) - np.pi)


def parse_episodes(value: str) -> list[int]:
    episodes = set()
    try:
        for part in value.split(","):
            bounds = part.strip().split("-")
            if len(bounds) == 1:
                episodes.add(int(bounds[0]))
            elif len(bounds) == 2:
                first, last = map(int, bounds)
                if first > last:
                    raise ValueError("reversed range")
                episodes.update(range(first, last + 1))
            else:
                raise ValueError("invalid range")
        if not episodes or min(episodes) < 0:
            raise ValueError("negative or empty episode selection")
    except ValueError as error:
        raise argparse.ArgumentTypeError("Use nonnegative episodes, e.g. 0,3,5-7") from error
    return sorted(episodes)


class Descriptors:
    """Keep scalar accumulators, short occlusion windows and one distance series."""

    def __init__(self):
        self.steps = self.v_difference = self.abs_policy_omega = self.abs_teacher_omega = 0
        self.omega_difference = self.bearing = 0.0
        self.distances = []
        self.times = []
        self.windows = []
        self.window = None

    def append(self, before: dict, after: dict, env) -> None:
        action = after["applied_action"]
        teacher = before["teacher_action"]
        self.steps += 1
        self.v_difference += float(action[0]) - float(teacher[0])
        self.abs_policy_omega += abs(float(action[1]))
        self.abs_teacher_omega += abs(float(teacher[1]))
        self.omega_difference += float(action[1]) - float(teacher[1])
        self.bearing += float(before["target_bearing_rad"])
        self.distances.append(float(before["target_distance_m"]))
        self.times.append(float(before["time_s"]))
        if not before["occluded"]:
            self.window = None
            return
        pose = np.asarray(before["agent_pose"])
        end_pose = np.asarray(after["agent_pose"])
        dt = float(after["time_s"] - before["time_s"])
        if self.window is None:
            self.window = {
                "start_time_s": float(before["time_s"]),
                "start_xy": pose[:2].tolist(),
                "old_target_xy": np.asarray(before["target_pose"])[:2].tolist(),
                "onset_heading_target_angle_rad": abs(float(before["target_bearing_rad"])),
                "steps": 0,
                "sum_v": 0.0,
                "sum_abs_omega": 0.0,
                "signed_turn_rad": 0.0,
                "absolute_turn_rad": 0.0,
                "path_length_m": 0.0,
                "old_target_angle_sum_rad": 0.0,
                "min_wall_clearance_m": None,
            }
            self.windows.append(self.window)
        window = self.window
        window["steps"] += 1
        window["end_time_s"] = float(after["time_s"])
        window["end_xy"] = end_pose[:2].tolist()
        window["sum_v"] += float(action[0])
        window["sum_abs_omega"] += abs(float(action[1]))
        window["signed_turn_rad"] += float(action[1]) * dt
        window["absolute_turn_rad"] += abs(float(action[1])) * dt
        window["path_length_m"] += float(np.linalg.norm(end_pose[:2] - pose[:2]))
        direction = np.asarray(window["old_target_xy"]) - pose[:2]
        window["old_target_angle_sum_rad"] += abs(
            angle_difference(float(pose[2]), float(np.arctan2(direction[1], direction[0])))
        )
        for position in (pose, end_pose):
            clearance = nearest_wall_clearance(env, position)
            if clearance is not None:
                previous = window["min_wall_clearance_m"]
                window["min_wall_clearance_m"] = (
                    clearance if previous is None else min(clearance, previous)
                )

    def result(self, end_time_s: float) -> dict:
        windows = []
        for window in self.windows:
            n = window["steps"]
            path = window["path_length_m"]
            straight = float(np.linalg.norm(np.subtract(window["end_xy"], window["start_xy"])))
            windows.append(
                {
                    "start_time_s": window["start_time_s"],
                    "end_time_s": window["end_time_s"],
                    "steps": n,
                    "mean_v_m_s": window["sum_v"] / n,
                    "mean_abs_omega_rad_s": window["sum_abs_omega"] / n,
                    "abs_net_turn_rad": abs(window["signed_turn_rad"]),
                    "total_abs_turn_rad": window["absolute_turn_rad"],
                    "path_length_m": path,
                    "straight_displacement_m": straight,
                    "straightness": straight / path if path > 0 else None,
                    "onset_heading_target_angle_rad": window["onset_heading_target_angle_rad"],
                    "mean_heading_old_target_angle_rad": window["old_target_angle_sum_rad"] / n,
                    "min_wall_clearance_m": window["min_wall_clearance_m"],
                }
            )
        n = sum(window["steps"] for window in self.windows)
        path = sum(window["path_length_m"] for window in self.windows)
        clearance = [
            w["min_wall_clearance_m"] for w in windows if w["min_wall_clearance_m"] is not None
        ]
        occlusion = None
        if n:
            occlusion = {
                "steps": n,
                "mean_v_m_s": sum(w["sum_v"] for w in self.windows) / n,
                "mean_abs_omega_rad_s": sum(w["sum_abs_omega"] for w in self.windows) / n,
                "abs_net_turn_rad": abs(sum(w["signed_turn_rad"] for w in self.windows)),
                "total_abs_turn_rad": sum(w["absolute_turn_rad"] for w in self.windows),
                "straightness": sum(w["straight_displacement_m"] for w in windows) / path
                if path > 0
                else None,
                "onset_heading_target_angle_rad": windows[0]["onset_heading_target_angle_rad"],
                "mean_heading_old_target_angle_rad": sum(
                    w["old_target_angle_sum_rad"] for w in self.windows
                )
                / n,
                "min_wall_clearance_m": min(clearance) if clearance else None,
                "windows": windows,
            }
        tail = [d for t, d in zip(self.times, self.distances, strict=True) if t >= end_time_s - 10]
        return {
            "occlusion": occlusion,
            "whole_episode": {
                "mean_policy_minus_teacher_v_m_s": self.v_difference / self.steps,
                "abs_omega_policy_teacher_ratio": self.abs_policy_omega / self.abs_teacher_omega
                if self.abs_teacher_omega > 0
                else None,
                "mean_policy_minus_teacher_omega_rad_s": self.omega_difference / self.steps,
                "mean_signed_bearing_rad": self.bearing / self.steps,
                "median_distance_m": float(np.median(self.distances)),
                "last_10s_mean_distance_m": float(np.mean(tail)) if tail else None,
            },
        }


def mass_summary(masses: dict, steps: int) -> dict:
    total = sum(masses.values())
    return {
        "steps": steps,
        "mass": dict(masses),
        "fraction": {name: value / total if total > 0 else None for name, value in masses.items()},
    }


def replay_episode(policy, contract, card, config, env, args, seed, episode, trace_path):
    import torch
    from flytrack_core.metrics import EpisodeMetrics
    from flytrack_train.evaluation import evaluation_environment_seed

    recorder = DNRecorder(policy, card, args.lesion) if getattr(args, "record_dn", False) else None
    options = config.get("evaluation", {})
    env_seed = evaluation_environment_seed(
        seed, episode, int(options.get("environment_seed", 1700000000))
    )
    observation, decision_info = env.reset(seed=env_seed)
    state = policy.initial_state(1, args.device)
    bearing_threshold = options.get("success_bearing_deg", 45.0)
    metrics = EpisodeMetrics(
        contract.control_dt,
        float(options.get("success_distance_m", 2.0)),
        float(options.get("success_time_fraction", 0.8)),
        None if bearing_threshold is None else float(bearing_threshold),
    )
    masses = {name: 0.0 for name in MASS_NAMES}
    segments = {
        segment: {name: 0.0 for name in MASS_NAMES} for segment in ("occluded", "non_occluded")
    }
    segment_steps = {segment: 0 for segment in segments}
    analyzed_steps = 0
    descriptors = Descriptors()
    step = 0
    with (
        recorder if recorder is not None else contextlib.nullcontext(),
        gzip.open(trace_path, "wt", encoding="utf-8") as stream,
        torch.no_grad(),
    ):
        while True:
            tensors = {
                key: torch.as_tensor(value, device=args.device).unsqueeze(0)
                for key, value in observation.items()
            }
            action, state, activity = policy(tensors, state)
            if recorder is not None:
                dn = recorder.pop()
            step_mass = {}
            segment = "occluded" if decision_info["occluded"] else "non_occluded"
            segment_steps[segment] += 1
            if card["variant"] != "GRU":
                contribution = policy.convergence_current_contributions(activity)
                for name, value in contribution.items():
                    # Keep the same conversion and accumulation order as lesion_analysis.
                    step_mass[name] = float(value.sum().cpu())
                    masses[name] += step_mass[name]
                    segments[segment][name] += step_mass[name]
                analyzed_steps += 1
            physical_action = action[0].cpu().numpy()
            observation, _, terminated, truncated, info = env.step(physical_action)
            metrics.append(
                info["target_distance_m"],
                info["target_bearing_rad"],
                float(info["applied_action"][1]),
                info["collision"],
                info["occluded"],
                info["time_s"],
            )
            descriptors.append(decision_info, info, env)
            record = {
                "step": step,
                **{
                    key: np.asarray(decision_info[key]).tolist()
                    for key in (
                        "time_s",
                        "agent_pose",
                        "target_pose",
                        "teacher_action",
                        "target_distance_m",
                        "target_bearing_rad",
                        "target_visible",
                        "occluded",
                    )
                },
                "action": physical_action.tolist(),
                "applied_action": np.asarray(info["applied_action"]).tolist(),
                "collision": bool(info["collision"]),
                "post_time_s": float(info["time_s"]),
                "post_agent_pose": np.asarray(info["agent_pose"]).tolist(),
                "post_target_pose": np.asarray(info["target_pose"]).tolist(),
            }
            if card["variant"] != "GRU":
                record["current_mass"] = step_mass
            if recorder is not None:
                record["dn"] = dn
            stream.write(json.dumps(clean(record), allow_nan=False) + "\n")
            step += 1
            decision_info = info
            if terminated or truncated:
                break
    row = {
        "variant": card["variant"],
        "lesion": args.lesion,
        "scenario": args.scenario,
        "seed": seed,
        "episode": episode,
        "environment_seed": env_seed,
        **metrics.result(),
        "analyzed_steps": analyzed_steps,
    }
    if card["variant"] != "GRU":
        total = sum(masses.values())
        for name, value in masses.items():
            row[f"{name}_fraction"] = value / total if total > 0 else None
        row["current_total"] = mass_summary(masses, analyzed_steps)
        row["current_segments"] = {
            key: mass_summary(value, segment_steps[key]) for key, value in segments.items()
        }
    row.update(
        collision_types=collision_types(env, info["agent_pose"], info["target_pose"])
        if info["collision"]
        else [],
        collision_time_s=float(info["time_s"]) if info["collision"] else None,
        descriptors=descriptors.result(float(info["time_s"])),
        trace=str(trace_path.relative_to(args.output)),
    )
    if recorder is not None:
        world = env.unwrapped
        row["world"] = {
            "walls": np.asarray(world.walls).tolist(),
            "obstacles": world.obstacles,
            "agent_radius_m": float(world.agent_radius_m),
            "target_radius_m": float(world.target_radius_m),
        }
    return clean(row)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", required=True, type=Path)
    parser.add_argument("--config", type=Path, default=Path("configs/train/evaluate.yaml"))
    parser.add_argument(
        "--scenario", choices=("normal", "occlusion", "noise", "conflict"), required=True
    )
    parser.add_argument("--lesion", choices=("none", "vision", "odor"), default="none")
    parser.add_argument("--episodes", type=parse_episodes, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu",), default="cpu")
    parser.add_argument("--record-dn", action="store_true")
    args = parser.parse_args()
    import yaml
    from flytrack_core.model import load_policy_artifact
    from flytrack_train.config import contract_from_config, make_env

    policy, contract, card = load_policy_artifact(args.artifact, device=args.device)
    if args.record_dn:
        recorder = DNRecorder(policy, card, args.lesion)  # Validate before creating any output.
    if not hasattr(policy, "set_lesion"):
        raise ValueError("Policy does not expose the declared lesion analysis interface")
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError("Evaluation configuration must be a YAML mapping")
    if contract_from_config(config).sha256() != contract.sha256():
        raise ValueError("Artifact sensor contract differs from evaluation configuration")
    try:
        seed = int(card["metadata"]["seed"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("Artifact model card must include an integer training seed") from error
    if seed < 0:
        raise ValueError("Artifact training seed must be nonnegative")
    trace_dir = args.output / "trace"
    trace_dir.mkdir(parents=True, exist_ok=True)
    inputs = [args.config, *sorted(args.artifact.rglob("*"))]
    inputs = [path for path in inputs if path.is_file()]
    if "sensor_contract_path" in config:
        inputs.append(Path(config["sensor_contract_path"]))
    if args.scenario in config.get("scenario_configs", {}):
        inputs.append(Path(config["scenario_configs"][args.scenario]))
    notes = {
        "provenance": provenance(inputs),
        "model_card": card,
        "control_dt_s": contract.control_dt,
        "alignment": __doc__,
        "multi_window_semantics": "Reset the old target position at each occlusion onset; "
        "aggregate straightness is sum of window displacements / sum of window path lengths. "
        "The aggregate onset angle is the first window's; every onset is also in windows.",
        "undefined": "Zero path length, zero teacher angular sum and zero current mass yield null. "
        "Missing wall geometry yields null clearance; missing collision geometry: NOT AVAILABLE.",
    }
    if args.record_dn:
        notes.update(
            record_dn=True,
            dn_definition="Readout-input rates, never the policy's branch-averaged activity. "
            "branches follow readout order: central, or visual then odor. types contains "
            "the union of required DNp09/DNa01/DNa02/DNb05 and every type used by "
            "difference, forward_pool or extra turn pools. in_readout is true iff a node "
            "of that type has a nonzero entry in difference, forward_pool, extra_left "
            "or extra_right; false means the recorded rates are not used by the readout. "
            "left/right are means over that type's nodes on side -1/+1; all averages every node "
            "of that type (including side 0). Required types must exist on both sides "
            "in the graph; an absent side of any other type is null. forward_latent "
            "is before sigmoid; omega_latent is before tanh, including bias and extra turns. "
            "world stores unwrapped walls [xmin,ymin,xmax,ymax], obstacle disks and radii.",
            required_dn_types_not_in_readout=recorder.required_types_not_in_readout,
        )
    policy.set_lesion(args.lesion)
    env = make_env(config, contract, scenario=args.scenario)
    rows = []
    try:
        for episode in args.episodes:
            filename = (
                f"{card['variant']}_s{seed}_{args.scenario}_{args.lesion}_ep{episode}.jsonl.gz"
            )
            rows.append(
                replay_episode(
                    policy, contract, card, config, env, args, seed, episode, trace_dir / filename
                )
            )
            print(f"{filename}: {rows[-1]['samples']} steps", flush=True)
    finally:
        env.close()
        policy.set_lesion("none")
    notes["outputs_sha256"] = {row["trace"]: sha256(args.output / row["trace"]) for row in rows}
    write_json(args.output / "episodes.json", rows)
    write_csv(args.output / "episodes.csv", rows)
    write_json(args.output / "analysis_notes.json", notes)


if __name__ == "__main__":
    main()
