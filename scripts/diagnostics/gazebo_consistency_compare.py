"""2D/Gazebo input consistency (#88, formerly #23): Gazebo inputs vs the 2D world at the same poses.

Usage (ML image, repo root):
  python scripts/diagnostics/gazebo_consistency_compare.py \
      EXPORT.npz configs/train/evaluate.yaml OUT.json
EXPORT.npz comes from gazebo_consistency_export.py (ROS image). Open-loop and descriptive: no model
runs and nothing is changed.

- Episodes come from the exported /world/scenario events (environment seed, start time, arena,
  obstacles, radii, plume model and config); only samples inside an active episode are compared.
- Vision: each Gazebo hex-eye frame (the policy's own input) vs the 2D ray cast at the agent and
  target poses linearly interpolated at the frame stamp, with the 2D evaluation's fixed intensities
  (domain_randomization false) and camera geometry. Per frame: Pearson r and mean absolute
  difference (MAD). "Target ommatidia" are those whose 2D value changes when the target is removed;
  their mean contrast to the rest of the frame is reported for both worlds.
- Odor: each Gazebo reading vs the 2D plume (same model, config and seed) advanced to the same
  episode time with the recorded target position as the source, sampled at
  odor_sensor_positions(recorded agent pose).
- Acceptance (G0c Decision on #88, fixed before any behaviour was examined), per eye over all
  frames: median r >= 0.9, median MAD <= 0.05, and the median target contrast has the 2D sign.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ACCEPT_MEDIAN_R = 0.9
ACCEPT_MEDIAN_MAD = 0.05


def interpolate(track: np.ndarray, t: float) -> np.ndarray | None:
    """[t, x, y, yaw] rows sorted by t -> pose at t, or None outside the recorded range."""
    i = int(np.searchsorted(track[:, 0], t))
    if i == 0 or i >= len(track):
        return None
    a, b = track[i - 1], track[i]
    w = (t - a[0]) / (b[0] - a[0]) if b[0] > a[0] else 0.0
    yaw = a[3] + w * np.angle(np.exp(1j * (b[3] - a[3])))
    return np.array([a[1] + w * (b[1] - a[1]), a[2] + w * (b[2] - a[2]), yaw])


def pearson(x: np.ndarray, y: np.ndarray) -> float:
    if x.std() < 1e-12 or y.std() < 1e-12:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def episode_windows(events: list[dict]) -> list[tuple[int, float, float, dict]]:
    """(episode id, start, end, parameters) from the first active event of each episode."""
    windows = []
    first = {}
    for event in events:
        if event["active"]:
            first.setdefault(event["episode_id"], event)
    for episode_id, event in sorted(first.items()):
        p = event["parameters"]
        start = float(p["episode_start_s"])
        ends = [
            e["stamp"]
            for e in events
            if e["episode_id"] == episode_id and not e["active"] and e["stamp"] > start
        ]
        windows.append((episode_id, start, min(ends, default=np.inf), p))
    return windows


def render_2d(env, agent, target, p, eye, with_target=True):
    half = float(p["arena_size_m"]) / 2
    cx, cy = float(p["arena_center_x"]), float(p["arena_center_y"])
    walls = np.array([cx - half, cy - half, cx + half, cy + half])
    disks = [
        {"height": env.target_height_m, "intensity": env._wall_intensity, **o}
        for o in p.get("obstacles", [])
    ]
    if with_target:
        disks.append(
            {
                "x": target[0],
                "y": target[1],
                "radius": float(p["target_radius_m"]),
                "height": env.target_height_m,
                "intensity": env._target_intensity,
            }
        )
    return env.retina.raycast(
        agent,
        walls,
        disks,
        eye,
        subrays=env.subrays,
        wall_intensity=env._wall_intensity,
        camera_height_m=env.camera_height_m,
        camera_forward_m=env.camera_forward_m,
        camera_baseline_m=env.camera_baseline_m,
    )


def compare(data, env, contract) -> dict:
    from flytrack_core.sensing.olfaction import odor_sensor_positions
    from flytrack_core.world import make_plume

    events = json.loads(str(data["events_json"]))
    agent, target = data["agent"], data["target"]
    frames = {0: [], 1: []}
    odor_rows = []
    episodes = []
    for episode_id, start, end, p in episode_windows(events):
        for eye in (0, 1):
            for t, gz in zip(data[f"vision{eye}_t"], data[f"vision{eye}"], strict=True):
                if not start < t < end:
                    continue
                a, g = interpolate(agent, t), interpolate(target, t)
                if a is None or g is None:
                    continue
                d2 = render_2d(env, a, g, p, eye)
                on = np.abs(d2 - render_2d(env, a, g, p, eye, with_target=False)) > 1e-6
                row = {"r": pearson(gz, d2), "mad": float(np.abs(gz - d2).mean())}
                if on.any() and (~on).any():
                    row["gz_contrast"] = float(gz[on].mean() - gz[~on].mean())
                    row["d2_contrast"] = float(d2[on].mean() - d2[~on].mean())
                frames[eye].append(row)
        plume = make_plume(p["plume_model"], p["plume_config"], int(p["environment_seed"]))
        count = 0
        for t, left, right, valid in data["odor"]:
            if not (start < t < end and valid):
                continue
            a, g = interpolate(agent, t), interpolate(target, t)
            if a is None or g is None:
                continue
            plume.advance(t - start, np.asarray([g[:2]]))
            d2 = plume.concentration(odor_sensor_positions(a, contract.odor_baseline_m))
            odor_rows.append((left, right, float(d2[0]), float(d2[1])))
            count += 1
        episodes.append(
            {
                "episode_id": episode_id,
                "environment_seed": int(p["environment_seed"]),
                "odor": count,
            }
        )

    def median(rows, key):
        values = [r[key] for r in rows if key in r and np.isfinite(r[key])]
        return float(np.median(values)) if values else None

    eyes = {}
    for eye, rows in frames.items():
        summary = {
            "frames": len(rows),
            **{
                f"{q}_{key}": float(np.percentile([r[key] for r in rows], q))
                for key in ("r", "mad")
                for q in (5, 50, 95)
                if rows
            },
            "median_gz_target_contrast": median(rows, "gz_contrast"),
            "median_2d_target_contrast": median(rows, "d2_contrast"),
        }
        summary["accepted"] = bool(
            rows
            and summary["50_r"] >= ACCEPT_MEDIAN_R
            and summary["50_mad"] <= ACCEPT_MEDIAN_MAD
            and summary["median_gz_target_contrast"] is not None
            and np.sign(summary["median_gz_target_contrast"])
            == np.sign(summary["median_2d_target_contrast"])
        )
        eyes["left" if eye == 0 else "right"] = summary
    odor = np.array(odor_rows) if odor_rows else np.zeros((0, 4))
    return {
        "rule": __doc__,
        "episodes": episodes,
        "vision": eyes,
        "accepted": all(e["accepted"] for e in eyes.values()),
        "odor": None
        if not len(odor)
        else {
            "samples": int(len(odor)),
            "r_left": pearson(odor[:, 0], odor[:, 2]),
            "r_right": pearson(odor[:, 1], odor[:, 3]),
            "median_abs_diff": float(np.median(np.abs(odor[:, :2] - odor[:, 2:]))),
            "left_right_sign_agreement": float(
                np.mean(np.sign(odor[:, 0] - odor[:, 1]) == np.sign(odor[:, 2] - odor[:, 3]))
            ),
        },
    }


def main() -> None:
    import yaml
    from flytrack_train.config import contract_from_config, make_env

    export, config_path, output = (Path(a) for a in sys.argv[1:4])
    data = np.load(export)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    contract = contract_from_config(config)
    if contract.sha256() != str(data["contract_sha256"]):
        raise ValueError("The 2D config and the Gazebo actor use different sensor contracts")
    env = make_env(config, contract, scenario="normal").unwrapped
    env.reset(seed=0)
    if env.domain_randomization:
        raise ValueError("Compare with the evaluation world: domain_randomization must be false")
    result = compare(data, env, contract)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    for name, eye in result["vision"].items():
        print(
            f"{name}: frames {eye['frames']} median r {eye['50_r']:.3f} MAD {eye['50_mad']:.3f} "
            f"target contrast gazebo {eye['median_gz_target_contrast']:+.3f} "
            f"2D {eye['median_2d_target_contrast']:+.3f} accepted {eye['accepted']}"
        )
    print("odor", result["odor"])
    print("ACCEPTED" if result["accepted"] else "NOT ACCEPTED")


if __name__ == "__main__":
    main()
