"""Episode metrics for one actor with an optional lesion, normal and occlusion only.

Usage: episode_eval.py OUT.json CONFIG.yaml ARTIFACT LESION
LESION is none or odor (policy.set_lesion). Reuses the lockstep evaluation of
scripts/diagnostics/policy_episode_eval.py (seeds from 1.6e9, EPISODES env,
default 30) and adds the mean |target bearing| over occluded steps
(occlusion scenario: 2 s from 8 s). No fixed-action baselines are run.
"""

import importlib.util
import json
import os
import sys
from functools import partial
from pathlib import Path

import numpy as np
import torch
import yaml
from flytrack_core.metrics import EpisodeMetrics
from flytrack_train.config import contract_from_config
from flytrack_train.parallel import EnvPool
from flytrack_train.worlds import make_env

REPO = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location(
    "policy_episode_eval", REPO / "scripts/diagnostics/policy_episode_eval.py"
)
episode_eval = importlib.util.module_from_spec(spec)
spec.loader.exec_module(episode_eval)
SCENARIOS = ("normal", "occlusion")


def run(pool, slots, act, contract, episodes, env_options):
    """policy_episode_eval.run plus occluded-step bearing errors."""
    results, omegas, teachers = [], [], []
    for first in range(0, episodes, len(slots)):
        lanes = slots[: min(len(slots), episodes - first)]
        obs, infos = zip(
            *(pool.reset(slot, episode_eval.BASE_SEED + first + i) for i, slot in enumerate(lanes)),
            strict=True,
        )
        obs, infos = list(obs), list(infos)
        metrics = [EpisodeMetrics.from_env_options(contract.control_dt, env_options) for _ in lanes]
        occluded = [[] for _ in lanes]
        live = list(range(len(lanes)))
        act.reset(len(lanes))
        while live:
            actions = act(obs, infos, live)
            stepped = pool.step({lanes[row]: actions[i] for i, row in enumerate(live)})
            still = []
            for i, row in enumerate(live):
                observation, _, terminated, truncated, info = stepped[lanes[row]]
                applied = info.get("applied_action", actions[i])
                omegas.append(float(applied[1]))
                teachers.append(float(infos[row]["teacher_action"][1]))
                metrics[row].append(
                    info["target_distance_m"],
                    info["target_bearing_rad"],
                    applied[1],
                    info["collision"],
                    info.get("occluded", False),
                    info.get("time_s"),
                )
                if info.get("occluded", False):
                    occluded[row].append(abs(float(np.rad2deg(info["target_bearing_rad"]))))
                obs[row], infos[row] = observation, info
                if not (terminated or truncated):
                    still.append(row)
            live = still
        for metric, errors in zip(metrics, occluded, strict=True):
            results.append(
                {
                    **metric.result(),
                    "occlusion_bearing_error_deg": float(np.mean(errors)) if errors else np.nan,
                    "occlusion_samples": len(errors),
                }
            )
    summary = {k: float(np.nanmean([r[k] for r in results])) for k in results[0]}
    summary["per_episode"] = [
        {**row, "seed": episode_eval.BASE_SEED + index} for index, row in enumerate(results)
    ]
    omegas, teachers = np.asarray(omegas), np.asarray(teachers)
    summary["omega_std"] = float(omegas.std())
    summary["teacher_omega_std"] = float(teachers.std())
    summary["omega_corr_with_teacher"] = (
        float(np.corrcoef(omegas, teachers)[0, 1]) if omegas.std() > 0 else 0.0
    )
    return summary


class LesionedActor(episode_eval.Actor):
    def __init__(self, path, device, lesion):
        super().__init__(path, device)
        self.policy.set_lesion(lesion)


def finite_json(value):
    if isinstance(value, dict):
        return {key: finite_json(item) for key, item in value.items()}
    if isinstance(value, list):
        return [finite_json(item) for item in value]
    return None if isinstance(value, float) and not np.isfinite(value) else value


def main():
    out, config_path, artifact, lesion = Path(sys.argv[1]), sys.argv[2], sys.argv[3], sys.argv[4]
    if lesion not in ("none", "odor"):
        raise ValueError("LESION must be none or odor")
    config = yaml.safe_load(open(config_path, encoding="utf-8"))
    episodes = int(os.environ.get("EPISODES", 30))
    lanes = min(episodes, int(os.environ.get("LANES", 30)))
    contract = contract_from_config(config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    act = LesionedActor(artifact, device, lesion)
    worker_option = os.environ.get("WORKERS", "auto")
    workers = (
        max(1, min(8, (os.cpu_count() or 2) - 1)) if worker_option == "auto" else int(worker_option)
    )
    pool = EnvPool(
        [partial(make_env, config, contract, scenario=s) for s in SCENARIOS for _ in range(lanes)],
        workers=workers,
    )
    try:
        report = {
            "artifact": artifact,
            "lesion": lesion,
            "env": config["env"],
            "episodes": episodes,
            "base_seed": episode_eval.BASE_SEED,
            "device": str(device),
            "scenarios": {
                s: run(
                    pool,
                    list(range(i * lanes, (i + 1) * lanes)),
                    act,
                    contract,
                    episodes,
                    config["env"],
                )
                for i, s in enumerate(SCENARIOS)
            },
        }
    finally:
        pool.close()
    occlusion = report["scenarios"]["occlusion"]
    if not np.isfinite(occlusion["occlusion_bearing_error_deg"]):
        raise ValueError("Occlusion scenario recorded no occluded steps")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(finite_json(report), indent=1, allow_nan=False) + "\n")
    print(
        artifact,
        lesion,
        {
            s: (round(v["success"], 2), round(v["mean_bearing_error_deg"], 1))
            for s, v in report["scenarios"].items()
        },
        "occluded bearing",
        round(occlusion["occlusion_bearing_error_deg"], 1),
        flush=True,
    )


if __name__ == "__main__":
    main()
