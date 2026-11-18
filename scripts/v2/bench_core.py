#!/usr/bin/env python3
"""Measure the v2 actor's frozen sensing, cached-code inference and optional training."""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path
from time import perf_counter

import torch
from flytrack_core.contracts import file_sha256
from flytrack_v2.artifacts import VARIANTS
from flytrack_v2.contracts import obs_shapes
from flytrack_v2.model import ConnectomePolicyV2, ModelConfigV2, build_policy

ROOT = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=VARIANTS, default="VO2")
    parser.add_argument("--batch", type=int, default=1024)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--train-batch", type=int, default=256)
    parser.add_argument("--train-seq", type=int, default=80)
    parser.add_argument(
        "--activation-checkpoint",
        choices=("on", "off"),
        help="override activation checkpointing for training only (default: model config)",
    )
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if min(args.batch, args.steps, args.train_batch, args.train_seq) <= 0 or args.warmup < 0:
        parser.error("batches, steps and sequence length must be positive; warmup may be zero")
    return args


def random_obs(policy, batch, device, generator):
    return {
        key: torch.ones(shape, device=device)
        if key == "valid"
        else torch.rand(shape, device=device, generator=generator)
        for key, shape in obs_shapes(policy.contract, batch).items()
    }


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


@torch.no_grad()
def inference(policy, args, device, generator):
    sensory = policy.initial_sensory_state(args.batch, device)
    central = policy.initial_central_state(args.batch, device)
    obs = random_obs(policy, args.batch, device, generator)
    # Warm both paths before making either measurement, including CUDA front-end compilation.
    code, sensory = policy.sense(obs, sensory)
    for _ in range(args.warmup):
        code, sensory = policy.sense(obs, sensory)
        central = policy.forward_code(code, central).central_state

    synchronize(device)
    start = perf_counter()
    for _ in range(args.steps):
        code, sensory = policy.sense(obs, sensory)
    synchronize(device)
    sense_ms = (perf_counter() - start) * 1000 / args.steps

    synchronize(device)
    start = perf_counter()
    for _ in range(args.steps):
        central = policy.forward_code(code, central).central_state
    synchronize(device)
    return sense_ms, (perf_counter() - start) * 1000 / args.steps


def train_sequence(policy, args, device, generator):
    if args.activation_checkpoint is not None:
        policy.config = replace(
            policy.config, activation_checkpoint=args.activation_checkpoint == "on"
        )
        if isinstance(policy, ConnectomePolicyV2):
            policy.core.config = policy.config

    sensory = policy.initial_sensory_state(args.train_batch, device)
    codes = []
    for _ in range(args.train_seq):
        obs = random_obs(policy, args.train_batch, device, generator)
        code, sensory = policy.sense(obs, sensory)
        codes.append(code)
    del sensory, obs
    state = policy.initial_central_state(args.train_batch, device)
    policy.train()
    synchronize(device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    start = perf_counter()
    losses = []
    for code in codes:
        step = policy.forward_code(code, state)
        state = step.central_state
        loss = -step.dist.log_prob(torch.zeros_like(step.dist.mean)).mean()
        if "activity_sq" in step.aux:
            loss = loss + 1e-4 * step.aux["activity_sq"].mean()
        losses.append(loss)
    torch.stack(losses).mean().backward()
    synchronize(device)
    return {
        "batch": args.train_batch,
        "sequence_length": args.train_seq,
        "activation_checkpoint": policy.config.activation_checkpoint,
        "forward_backward_ms": (perf_counter() - start) * 1000,
        "peak_cuda_allocated_bytes": (
            torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None
        ),
    }


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    config = ModelConfigV2()
    seed = 0 if args.variant in ("RW2", "ER2") else None
    policy = build_policy(args.variant, config, structure_seed=seed).to(device).eval()
    generator = torch.Generator(device=device).manual_seed(0)
    sense_ms, forward_ms = inference(policy, args, device, generator)
    ratio = forward_ms / sense_ms
    results = {
        "variant": args.variant,
        "structure_seed": seed,
        "device": str(device),
        "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU",
        "torch_version": torch.__version__,
        "batch": args.batch,
        "steps": args.steps,
        "warmup": args.warmup,
        "sense_ms_per_step": sense_ms,
        "forward_code_ms_per_step": forward_ms,
        "forward_to_sense_ratio": ratio,
        "ratio_le_0_20": ratio <= 0.20,
        "graph": {"path": config.graph_path, "sha256": file_sha256(ROOT / config.graph_path)},
        "frontend": {
            "path": config.frontend_path,
            "sha256": file_sha256(ROOT / config.frontend_path),
        },
        "train": None if args.skip_train else train_sequence(policy, args, device, generator),
    }
    output = json.dumps(results, indent=2)
    print(output)
    if args.output is not None:
        args.output.write_text(output + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
