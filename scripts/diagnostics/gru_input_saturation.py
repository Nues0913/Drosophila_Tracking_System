"""Measure initialized GRU bottleneck saturation on normal teacher rollouts (#85).

Run from the stage configuration's project root so its relative source paths resolve:
gru_input_saturation.py OUT.json STAGE_CONFIG.yaml --compensation both
STAGE_CONFIG must be the GRU BC stage configuration (GRU/seed_0/bc.yaml),
whose puff plume matches evaluation and the PM reference environment. The
pretrain_vision configuration uses a gaussian plume and underestimates saturation.
Use a CPU memory limit, for example prlimit --as=8000000000.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import flytrack_core
import numpy as np
import torch
from flytrack_core.model import build_policy
from flytrack_train.batches import SequenceModel, tensor_observation
from flytrack_train.config import contract_from_config, project_source_fingerprint, read_config
from modality_sensitivity import teacher_observations
from torch.nn import functional as F

# The shared venv may be editable against another checkout. Refuse that silently
# different implementation without hard-coding the checkout's absolute location.
ROOT = Path(__file__).resolve().parents[2]
assert Path(flytrack_core.__file__).resolve().parent == ROOT / "python/flytrack_core", (
    "Set PYTHONPATH to this checkout's python directory"
)

THRESHOLDS = {"saturation_fraction_max": 0.05, "vision_odor_std_ratio_min": 0.02}


@torch.no_grad()
def measure(policy, episodes):
    """Capture the actual Linear input; average temporal std per episode/unit."""
    if not isinstance(policy.input_projection, torch.nn.Sequential):
        raise ValueError("The diagnostic requires a resolved GRU input bottleneck")
    linear = policy.input_projection[0]
    encoder = policy.encoder
    vision = torch.where(encoder.visual_inputs[encoder.seed_indices])[0]
    odor = torch.where(encoder.odor_inputs[encoder.seed_indices])[0]
    if torch.any(encoder.visual_inputs & encoder.odor_inputs):
        raise ValueError("Modality attribution requires disjoint visual/odor seeds")
    model = SequenceModel(policy, "cpu")
    rows = []

    def capture(module, inputs):
        features = inputs[0]
        rows.append(
            (
                F.linear(features, linear.weight, linear.bias)[0].numpy().copy(),
                F.linear(features[:, vision], linear.weight[:, vision])[0].numpy().copy(),
                F.linear(features[:, odor], linear.weight[:, odor])[0].numpy().copy(),
                float(features[:, vision].abs().mean()),
                float(features[:, odor].abs().mean()),
            )
        )

    hook = linear.register_forward_pre_hook(capture)
    all_rows = []
    vision_stds, odor_stds = [], []
    try:
        for observations in episodes:
            sensory, central = model.initial(1)
            rows.clear()
            for observation in observations:
                with model.cache():
                    inputs, sensory = model.observe(
                        tensor_observation([observation], model.device), sensory
                    )
                    _, central, _ = model.step(inputs, central)
            if not rows:
                raise ValueError("Teacher episodes must contain at least one observation")
            # Population std (ddof=0), independently over time in each episode/unit.
            vision_stds.extend(np.asarray([row[1] for row in rows], dtype=np.float64).std(0))
            odor_stds.extend(np.asarray([row[2] for row in rows], dtype=np.float64).std(0))
            all_rows.extend(rows)
    finally:
        hook.remove()
    absolute = np.abs(np.stack([row[0] for row in all_rows]))
    vision_std, odor_std = float(np.mean(vision_stds)), float(np.mean(odor_stds))
    ratio = vision_std / odor_std if odor_std > 0 else None
    saturation = float(np.mean(absolute > 3))
    return {
        "saturation_fraction": saturation,
        "vision_odor_std_ratio": ratio,
        "vision_temporal_std_mean": vision_std,
        "odor_temporal_std_mean": odor_std,
        "abs_preactivation_median": float(np.median(absolute)),
        "abs_preactivation_p95": float(np.percentile(absolute, 95)),
        "vision_feature_mean_abs": float(np.mean([row[3] for row in all_rows])),
        "odor_feature_mean_abs": float(np.mean([row[4] for row in all_rows])),
        "episode_steps": [len(episode) for episode in episodes],
        "vision_seed_count": len(vision),
        "odor_seed_count": len(odor),
        "seed_overlap_count": int((encoder.visual_inputs & encoder.odor_inputs).sum()),
        "pass": saturation <= THRESHOLDS["saturation_fraction_max"]
        and ratio is not None
        and ratio >= THRESHOLDS["vision_odor_std_ratio_min"],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument(
        "stage_config", type=Path, help="GRU BC stage YAML (puff plume, matching evaluation)"
    )
    parser.add_argument("--compensation", choices=("false", "true", "both"), default="both")
    parser.add_argument("--torch-seeds", type=int, nargs="+", default=list(range(5)))
    parser.add_argument("--env-seeds", type=int, nargs="+", default=[1610000011, 1610000012])
    parser.add_argument("--steps", type=int, default=600)
    args = parser.parse_args()
    if args.steps < 1:
        parser.error("steps must be positive")
    started = time.perf_counter()
    torch.set_num_threads(1)
    config = read_config(args.stage_config)
    if config.get("variant") != "GRU":
        parser.error("stage_config must configure GRU")
    contract = contract_from_config(config)
    episodes = [teacher_observations(config, contract, args.steps, seed) for seed in args.env_seeds]
    report = {
        "stage_config": str(args.stage_config),
        "stage_config_sha256": hashlib.sha256(args.stage_config.read_bytes()).hexdigest(),
        "stage_settings": config,
        "source_sha256": project_source_fingerprint()["sha256"],
        "diagnostic_script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "device": "cpu",
        "torch_num_threads": 1,
        "torch_seeds": args.torch_seeds,
        "env_seeds": args.env_seeds,
        "requested_steps": args.steps,
        "episode_steps": [len(episode) for episode in episodes],
        "temporal_std_ddof": 0,
        "thresholds": THRESHOLDS,
        "results": [],
    }
    modes = [False, True] if args.compensation == "both" else [args.compensation == "true"]
    print("torch_seed compensation saturation_fraction vision_odor_std_ratio pass", flush=True)
    for seed in args.torch_seeds:
        for compensation in modes:
            torch.manual_seed(seed)
            policy = (
                build_policy(
                    "GRU",
                    config["graph_path"],
                    contract,
                    config["frontend_path"],
                    **{**config.get("model", {}), "gru_odor_gain_compensation": compensation},
                )
                .cpu()
                .eval()
            )
            row = {
                "torch_seed": seed,
                "compensation": compensation,
                "model_config": dict(policy.model_config),
                **measure(policy, episodes),
            }
            report["results"].append(row)
            ratio = row["vision_odor_std_ratio"]
            ratio_text = "undefined" if ratio is None else f"{ratio:.9f}"
            print(
                f"{seed:10d} {str(compensation):12s} "
                f"{row['saturation_fraction']:.9f} {ratio_text} {row['pass']}",
                flush=True,
            )
            del policy
    report["elapsed_seconds"] = time.perf_counter() - started
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
