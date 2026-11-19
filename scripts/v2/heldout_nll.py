"""Compare exported policies on aligned held-out DAgger replay windows.

The trainer starts at a saved pre-sense snapshot and scores the suffix after
burn-in. Draw a full window at snapshot tick t with burn-in B = max(--burn-in)
and score code, actions and resets at [t+B, t+B+length) for every b. For a
shorter burn-in b, start at t+B-b and use the replay snapshot at that tick,
following the trainer's initialization and slicing. All B-b offsets must
therefore be multiples of the snapshot stride. Resets inside a window are
handled by the trainer's loss.
"""

from __future__ import annotations

import argparse
import json
import re
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import torch
from flytrack_v2.model import ConnectomePolicyV2, GRUPolicyV2, load_policy
from flytrack_v2.train.config import STATE_STRIDE
from flytrack_v2.train.data import ReplayShard, SequenceBatch, StepData, _allocate_steps
from flytrack_v2.train.update import LossSettings, batch_metrics, label_margins
from torch import Tensor

Policy = ConnectomePolicyV2 | GRUPolicyV2
NLL_KEYS = ("nll", "nll_speed", "nll_turn")


def replay_paths(run: Path) -> list[Path]:
    rounds = []
    for path in run.glob("round_*/replay.pt"):
        match = re.fullmatch(r"round_([1-9][0-9]*)", path.parent.name)
        if match and path.is_file():
            rounds.append((int(match[1]), path))
    if not rounds:
        raise ValueError(f"No round_<k>/replay.pt files found in {run}")
    return [path for _, path in sorted(rounds)]


def held_out_draws(
    paths: list[Path], window: int, size: int, seed: int
) -> Iterator[tuple[Path, Tensor]]:
    """Match Replay.sample's draw, retaining draw positions for the metric batches."""
    population = []
    for path in paths:
        shard = ReplayShard.load(path)
        rows = torch.where(shard.held_out)[0]
        starts = max(
            0, 1 + (shard.steps.code.size(1) - shard.first_snapshot - window) // STATE_STRIDE
        )
        population.append((path, rows, starts))
        del shard

    total = sum(len(rows) * starts for _, rows, starts in population)
    if not total:
        raise ValueError("Replay has no complete held-out sequences for the requested window")
    rng = torch.Generator().manual_seed(seed)
    draws = torch.randperm(total, generator=rng)[:size].clone()
    offset = 0
    for path, rows, starts in population:
        end = offset + len(rows) * starts
        positions = torch.where((draws >= offset) & (draws < end))[0]
        local = draws[positions] - offset
        offset = end
        if local.numel():
            snaps = local // len(rows)
            envs = rows[local % len(rows)]
            order = (envs * starts + snaps).argsort()
            yield path, torch.stack((positions[order], envs[order], snaps[order]), dim=1)


def read_batch(
    draws: list[tuple[Path, Tensor]],
    first: int,
    size: int,
    window: int,
    offsets: list[int],
    owner: Policy,
) -> tuple[StepData, dict[int, Tensor], Tensor]:
    """Read one draw-order batch, mapping one shard at a time."""
    steps = _allocate_steps(window, size, owner.code_size, "cpu")
    initials = {offset: torch.empty(size, owner.central_state_size) for offset in offsets}
    tags = None
    for path, picks in draws:
        part = picks[(picks[:, 0] >= first) & (picks[:, 0] < first + size)]
        if not len(part):
            continue
        positions, rows, snaps = part.unbind(dim=1)
        dest = positions - first
        shard = ReplayShard.load(path)
        if (
            shard.steps.code.size(-1) != owner.code_size
            or shard.states.size(-1) != owner.central_state_size
        ):
            raise ValueError(f"{path} has code or state dimensions incompatible with its policy")
        starts = shard.first_snapshot + snaps * STATE_STRIDE
        ticks = starts[:, None] + torch.arange(window)
        for target, source in zip(steps, shard.steps, strict=True):
            values = source[rows[:, None], ticks].transpose(0, 1)
            target.index_copy_(1, dest, values.to(target.dtype))
        for offset, initial in initials.items():
            initial[dest] = shard.states[rows, snaps + offset // STATE_STRIDE].float()
        if tags is None:
            tags = shard.tags.new_empty((size, shard.tags.size(-1)))
        tags[dest] = shard.tags[rows]
        del source, shard
    return steps, initials, tags


def aligned_batch(steps: StepData, initial: Tensor, tags: Tensor, offset: int) -> SequenceBatch:
    return SequenceBatch(
        steps.code[offset:],
        steps.action[offset:],
        steps.reset[offset:],
        initial,
        tags,
        steps.occluded[offset:],
    )


def can_share_replay(a: Policy, b: Policy) -> bool:
    if (
        a.variant != b.variant
        or a.contract.sha256() != b.contract.sha256()
        or a.encoder.layout != b.encoder.layout
        or a.encoder.frontend.source_sha256 != b.encoder.frontend.source_sha256
        or a.central_state_size != b.central_state_size
    ):
        return False
    if isinstance(a, ConnectomePolicyV2) and isinstance(b, ConnectomePolicyV2):
        return np.array_equal(a.graph.node_ids, b.graph.node_ids)
    return True


@torch.no_grad()
def probe(args: argparse.Namespace) -> dict:
    runs = dict(zip(("A", "B"), args.runs, strict=True))
    configs = {
        name: json.loads((run / "config.json").read_text(encoding="utf-8"))
        for name, run in runs.items()
    }
    device = torch.device(args.device)
    policies = {name: load_policy(run / "final", device=device) for name, run in runs.items()}
    shared = can_share_replay(policies["A"], policies["B"])
    results = []
    for data, run in runs.items():
        config = configs[data]
        if config.get("state_stride", STATE_STRIDE) != STATE_STRIDE:
            raise ValueError(f"{run} has a snapshot stride different from {STATE_STRIDE}")
        length = config["sequence"]["length"]
        batch_size = args.batch_size if args.batch_size is not None else config["micro_batch"]
        margins = label_margins(config)
        settings = {b: LossSettings(b, config["activity_coef"], margins) for b in args.burn_in}
        evaluated = {data: policies[data]}
        if data == "A" and shared:
            evaluated["B"] = policies["B"]
        totals = {
            (name, b): torch.zeros(len(NLL_KEYS), device=device)
            for name in evaluated
            for b in settings
        }
        longest = max(settings)
        window = longest + length
        offsets = [longest - b for b in settings]
        draws = list(held_out_draws(replay_paths(run), window, args.sequences, args.seed))
        count = sum(len(picks) for _, picks in draws)
        for first in range(0, count, batch_size):
            size = min(batch_size, count - first)
            steps, initials, tags = read_batch(draws, first, size, window, offsets, policies[data])
            steps = StepData(*(value.to(device) for value in steps))
            tags = tags.to(device)
            for burn_in, loss in settings.items():
                offset = longest - burn_in
                batch = aligned_batch(steps, initials[offset].to(device), tags, offset)
                for name, policy in evaluated.items():
                    metrics = batch_metrics(policy, batch, loss, batch_size, backward=False)
                    values = torch.stack([metrics[key] for key in NLL_KEYS])
                    totals[name, burn_in].add_(values * (size / count))
        for (weights, b), total in totals.items():
            results.append(
                {
                    "weights": weights,
                    "data": data,
                    "burn_in": b,
                    "samples": count,
                    "scored_steps": count * length,
                    **dict(zip(NLL_KEYS, total.tolist(), strict=True)),
                }
            )
    return {
        "runs": {name: str(run) for name, run in runs.items()},
        "seed": args.seed,
        "requested_sequences": args.sequences,
        "shared_replay": "A" if shared else None,
        "results": results,
    }


def render_markdown(report: dict) -> str:
    shared = (
        "Both policies scored the same held-out sequences from replay A."
        if report["shared_replay"]
        else "Shared replay unavailable because the policy code or state layouts are incompatible."
    )
    lines = [
        "# Held-out NLL",
        "",
        *(f"- {name} run at `{path}`" for name, path in report["runs"].items()),
        f"- Seed {report['seed']} with up to {report['requested_sequences']} sequences per replay.",
        "",
        shared,
        "",
        "| Weights | Replay | Burn-in | Sequences | Scored steps | NLL | Speed NLL | Turn NLL |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in report["results"]:
        metrics = " | ".join(f"{row[key]:.8g}" for key in NLL_KEYS)
        lines.append(
            f"| {row['weights']} | {row['data']} | {row['burn_in']} | {row['samples']} | "
            f"{row['scored_steps']} | {metrics} |"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--runs", nargs=2, required=True, type=Path, metavar=("A", "B"))
    parser.add_argument("--burn-in", nargs="+", required=True, type=int)
    parser.add_argument("--sequences", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, help="override the replay run's micro_batch")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    args.burn_in = sorted(set(args.burn_in))
    if min(args.burn_in) < 0 or args.seed < 0:
        parser.error("--burn-in and --seed must be nonnegative")
    if args.sequences < 1 or (args.batch_size is not None and args.batch_size < 1):
        parser.error("--sequences and --batch-size must be positive")
    if any((args.burn_in[-1] - b) % STATE_STRIDE for b in args.burn_in):
        parser.error(f"--burn-in values must differ by multiples of {STATE_STRIDE} ticks")

    report = probe(args)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "heldout_nll.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    (args.out / "heldout_nll.md").write_text(render_markdown(report), encoding="utf-8")


if __name__ == "__main__":
    main()
