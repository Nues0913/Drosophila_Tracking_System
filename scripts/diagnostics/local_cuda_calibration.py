"""Calibrate CPU/CUDA actor outputs and actor-only throughput on stored data."""

from __future__ import annotations

import argparse
import hashlib
import time
from pathlib import Path

import numpy as np
import torch
from flytrack_train.config import write_json
from flytrack_train.recovery import EpisodeStore
from policy_episode_eval import Actor

ROOT = Path(__file__).resolve().parents[2]


def run(actor, episodes, batch_size, steps):
    actor.reset(batch_size)
    live = list(range(batch_size))
    output = []
    started = time.perf_counter()
    for step in range(steps):
        observations = [
            {key: value[step] for key, value in episode.observations.items()}
            for episode in episodes[:batch_size]
        ]
        # Actor materializes actions on CPU on every step, as the simulator does.
        output.append(np.asarray(actor(observations, [], live)))
    return np.stack(output), time.perf_counter() - started


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batches", type=int, nargs="+", default=[8, 24])
    parser.add_argument("--steps", type=int, default=128)
    parser.add_argument("--repeats", type=int, default=2)
    args = parser.parse_args()
    if not args.output.resolve().is_relative_to(ROOT):
        parser.error("Output must stay in this workspace")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available to this process")
    torch.set_num_threads(1)
    episodes = EpisodeStore(args.dataset).load(max(args.batches))
    steps = min(args.steps, min(len(e) for e in episodes))
    report = {
        "gpu": torch.cuda.get_device_name(0), "torch": str(torch.__version__),
        "artifact": str(args.artifact),
        "model_card_sha256": hashlib.sha256(
            (args.artifact / "model_card.json").read_bytes()
        ).hexdigest(),
        "calibration_script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "steps_per_sequence": steps, "repeats": args.repeats,
        "scope": "Recorded-observation actor inference only; excludes rendering and PPO updates.",
        "runtime_identity_interchange_verified": False,
        "rows": [],
    }
    reference = {}
    for device in ("cpu", "cuda"):
        actor = Actor(args.artifact, torch.device(device))
        for batch in args.batches:
            run(actor, episodes, batch, min(8, steps))
            if device == "cuda":
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
            elapsed, maximum_error = [], 0.0
            for _ in range(args.repeats):
                actions, duration = run(actor, episodes, batch, steps)
                elapsed.append(duration)
                if device == "cpu":
                    reference[batch] = actions
                else:
                    error = float(np.max(np.abs(actions - reference[batch])))
                    maximum_error = max(maximum_error, error)
                    np.testing.assert_allclose(actions, reference[batch], atol=1e-4, rtol=1e-4)
            row = {
                "device": device, "batch": batch,
                "median_seconds": float(np.median(elapsed)),
                "actor_frames_per_second": batch * steps / float(np.median(elapsed)),
                "maximum_action_abs_error_vs_cpu": maximum_error,
                "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated()
                if device == "cuda" else None,
            }
            report["rows"].append(row)
            write_json(args.output, report)
            print(row, flush=True)
        del actor
        if device == "cuda":
            torch.cuda.empty_cache()
    report["numeric_parity_passed"] = True
    report["complete"] = True
    write_json(args.output, report)


if __name__ == "__main__":
    main()
