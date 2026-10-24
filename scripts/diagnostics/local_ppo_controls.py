"""Evaluation-only controls for the readout update direction (study_plan controls).

dose: BC plus signed amounts along the mean unit change of chosen PPO actors.
build: derive actors from BC plus a readout change: alpha times the consensus
(mean over development PPO seeds), its forward-only and turn-only parts, and
random directions with the consensus norm. Core, frontend and log_std are BC's.
analyze: paired curved/normal deltas of each control against the same BC report.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
import yaml
from flytrack_core.model import load_policy_artifact, save_policy_artifact
from flytrack_train.config import write_json
from local_ppo_compare import METRICS, paired_delta

ROOT = Path(__file__).resolve().parents[2]
PRIMARY = "curved/normal"
ALPHAS = (0.5, 1.0, 2.0)
SHAM_DIRECTIONS = 3
SHAM_SEED = 20261024


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def readout_delta(base, actor):
    reference = base.state_dict()
    return {name: (value - reference[name]).double()
            for name, value in actor.state_dict().items() if name.startswith("readout.")}


def flat(delta):
    return torch.cat([delta[name].flatten() for name in sorted(delta)])


def unflat(vector, like):
    out, start = {}, 0
    for name in sorted(like):
        size = like[name].numel()
        out[name] = vector[start : start + size].reshape(like[name].shape)
        start += size
    return out


def build(args):
    base, contract, card = load_policy_artifact(ROOT / args.baseline, device="cpu")
    sources, deltas = [], []
    for seed in args.seeds:
        config = yaml.safe_load((args.configs / f"seed_{seed}.yaml").read_text())
        if Path(config["init_artifact"]) != args.baseline:
            raise ValueError(f"Seed {seed} was trained from another BC start")
        actor_path = ROOT / config["output"] / "actor"
        actor, _, _ = load_policy_artifact(actor_path, device="cpu")
        deltas.append(readout_delta(base, actor))
        sources.append({"seed": seed, "actor": str(actor_path.relative_to(ROOT)),
                        "model_card_sha256": digest(actor_path / "model_card.json")})
    consensus = {name: torch.stack([d[name] for d in deltas]).mean(0) for name in deltas[0]}
    norm = float(flat(consensus).norm())
    controls = {f"consensus_alpha_{alpha:g}": (alpha, {k: alpha * v for k, v in consensus.items()})
                for alpha in ALPHAS}
    controls["consensus_forward_only"] = (1.0, {
        k: v if k.startswith("readout.forward") else torch.zeros_like(v)
        for k, v in consensus.items()})
    controls["consensus_turn_only"] = (1.0, {
        k: v if k.startswith("readout.turn") else torch.zeros_like(v)
        for k, v in consensus.items()})
    if args.per_seed_components:
        # Development diagnostic: split each seed's own change into its two parts.
        for source, delta in zip(sources, deltas, strict=True):
            for part in ("forward", "turn"):
                controls[f"seed_{source['seed']}_{part}_only"] = (1.0, {
                    k: v if k.startswith(f"readout.{part}") else torch.zeros_like(v)
                    for k, v in delta.items()})
    generator = torch.Generator().manual_seed(SHAM_SEED)
    for index in range(SHAM_DIRECTIONS):
        direction = torch.randn(flat(consensus).numel(), generator=generator, dtype=torch.float64)
        controls[f"sham_random_{index}"] = (1.0, unflat(direction / direction.norm() * norm,
                                                        consensus))
    manifest = {"baseline": str(args.baseline), "recipe": args.recipe, "sources": sources,
                "consensus_norm": norm,
                "consensus": {k: v.tolist() for k, v in consensus.items()},
                "per_seed_delta": [{k: v.tolist() for k, v in d.items()} for d in deltas],
                "controls": {}}
    for name, (alpha, delta) in controls.items():
        # Sparse CSR buffers cannot be deep-copied; reload BC for every control.
        policy, _, _ = load_policy_artifact(ROOT / args.baseline, device="cpu")
        with torch.no_grad():
            for parameter_name, parameter in policy.named_parameters():
                if parameter_name in delta:
                    parameter.add_(delta[parameter_name].to(parameter.dtype))
        if name.startswith("sham"):
            group, seed = name.rsplit("_", 1)[0], int(name.rsplit("_", 1)[1])
        elif name.startswith("seed_"):
            group, seed = name.split("_", 2)[2], int(name.split("_")[1])
        else:
            group, seed = name, 0
        metadata = {
            "stage": "control", "algorithm": "derived_readout_change", "seed": seed,
            "config": {"init_artifact": str(args.baseline), "local_experiment": {
                "name": group, "alpha": alpha, "derived_from": sources,
                "sham_seed": SHAM_SEED if name.startswith("sham") else None}},
        }
        destination = ROOT / args.output / name / "actor"
        save_policy_artifact(destination, policy, policy.model_config, contract,
                             ROOT / args.baseline / "graph.npz",
                             frontend_path=ROOT / args.baseline / "frontend.npz",
                             metadata=metadata)
        reloaded, _, _ = load_policy_artifact(destination, device="cpu")
        change = readout_delta(base, reloaded)
        others = {k: float((v - base.state_dict()[k]).abs().max())
                  for k, v in reloaded.state_dict().items() if not k.startswith("readout.")
                  and v.dtype.is_floating_point}
        if any(others.values()):
            raise ValueError(f"{name}: a non-readout parameter changed")
        manifest["controls"][name] = {
            "actor": str(destination.relative_to(ROOT)), "group": group, "alpha": alpha,
            "readout_change": {k: v.tolist() for k, v in change.items()},
            "readout_change_norm": float(flat(change).norm()),
            "cosine_with_consensus": float(flat(change) @ flat(consensus)
                                           / (flat(change).norm() * flat(consensus).norm())),
        }
    write_json(ROOT / args.output / "controls_manifest.json", manifest)
    print(json.dumps({k: {"norm": v["readout_change_norm"], "cos": v["cosine_with_consensus"]}
                      for k, v in manifest["controls"].items()}, indent=2))


def dose(args):
    """BC plus a signed dose along the mean unit change of the given PPO actors."""
    base, contract, _ = load_policy_artifact(ROOT / args.baseline, device="cpu")
    prefixes = tuple(f"readout.{part}" for part in args.parts)
    units, sources = [], []
    for actor_path in args.runs:
        actor, _, _ = load_policy_artifact(ROOT / actor_path, device="cpu")
        delta = {k: v for k, v in readout_delta(base, actor).items() if k.startswith(prefixes)}
        units.append(flat(delta) / flat(delta).norm())
        sources.append({"actor": str(actor_path),
                        "model_card_sha256": digest(ROOT / actor_path / "model_card.json"),
                        "change_norm": float(flat(delta).norm())})
    direction = torch.stack(units).mean(0)
    direction = direction / direction.norm()
    template = {k: v for k, v in readout_delta(base, base).items() if k.startswith(prefixes)}
    manifest = {"baseline": str(args.baseline), "parts": args.parts, "sources": sources,
                "direction": {k: v.tolist() for k, v in unflat(direction, template).items()},
                "cosine_with_direction": [float(u @ direction) for u in units], "doses": {}}
    for amount in args.doses:
        name = f"dose_{amount:+g}"
        change = unflat(direction * amount, template)
        policy, _, _ = load_policy_artifact(ROOT / args.baseline, device="cpu")
        with torch.no_grad():
            for parameter_name, parameter in policy.named_parameters():
                if parameter_name in change:
                    parameter.add_(change[parameter_name].to(parameter.dtype))
        destination = ROOT / args.output / name / "actor"
        save_policy_artifact(destination, policy, policy.model_config, contract,
                             ROOT / args.baseline / "graph.npz",
                             frontend_path=ROOT / args.baseline / "frontend.npz",
                             metadata={"stage": "control", "algorithm": "derived_readout_change",
                                       "seed": 0, "config": {
                                           "init_artifact": str(args.baseline),
                                           "local_experiment": {"name": name, "dose": amount,
                                                                "derived_from": sources}}})
        manifest["doses"][name] = {"actor": str(destination.relative_to(ROOT)), "dose": amount,
                                   "readout_change": {k: v.tolist() for k, v in change.items()}}
    write_json(ROOT / args.output / "dose_manifest.json", manifest)
    print(json.dumps({"cosines": manifest["cosine_with_direction"],
                      "direction": manifest["direction"]}, indent=2))


def analyze(args):
    baseline = json.loads(args.baseline_report.read_text())
    horizon = float(baseline["config"].get("env", {}).get("episode_seconds", 60.0))
    base_rows = next(p for p in baseline["policies"].values()
                     if not p["metadata"].get("control"))["cells"][PRIMARY]["episodes"]
    result = {"baseline_report": str(args.baseline_report), "cell": PRIMARY, "controls": {}}
    for path in args.reports:
        report = json.loads(path.read_text())
        for key in ("config_sha256", "implementation", "environment_seeds"):
            if report[key] != baseline[key]:
                raise ValueError(f"{path}: {key} differs from the BC report")
        for name, policy in report["policies"].items():
            delta = paired_delta(base_rows, policy["cells"][PRIMARY]["episodes"], horizon)
            result["controls"][Path(name).parent.name] = {
                "group": policy["metadata"]["local_experiment"]["name"],
                "paired_mean_delta": delta,
                "summary": {k: policy["cells"][PRIMARY]["summary"][k] for k in METRICS
                            if k in policy["cells"][PRIMARY]["summary"]},
            }
    shams = [v["paired_mean_delta"] for v in result["controls"].values()
             if v["group"] == "sham_random"]
    if shams:
        result["sham_range"] = {k: [float(np.min([s[k] for s in shams])),
                                    float(np.max([s[k] for s in shams]))] for k in METRICS}
    write_json(args.output, result)
    print(json.dumps({k: {m: round(v["paired_mean_delta"][m], 4) for m in (
        "episode_return", "tracked_fraction", "full_horizon_tracked_fraction", "success",
        "collision_rate")} for k, v in result["controls"].items()}, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    make = commands.add_parser("build")
    make.add_argument("--recipe", required=True)
    make.add_argument("--baseline", type=Path, required=True)
    make.add_argument("--configs", type=Path, required=True)
    make.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    make.add_argument("--output", type=Path, required=True)
    make.add_argument("--per-seed-components", action="store_true",
                      help="Also build forward-only and turn-only parts of each seed's change")
    doses = commands.add_parser("dose")
    doses.add_argument("--baseline", type=Path, required=True)
    doses.add_argument("--runs", nargs="+", type=Path, required=True)
    doses.add_argument("--parts", nargs="+", default=["turn"], choices=["turn", "forward"])
    doses.add_argument("--doses", nargs="+", type=float, required=True)
    doses.add_argument("--output", type=Path, required=True)
    study = commands.add_parser("analyze")
    study.add_argument("--baseline-report", type=Path, required=True)
    study.add_argument("--reports", nargs="+", type=Path, required=True)
    study.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not (ROOT / args.output).resolve().is_relative_to(ROOT):
        parser.error("Output must stay in this workspace")
    torch.set_num_threads(1)
    {"build": build, "dose": dose, "analyze": analyze}[args.command](args)


if __name__ == "__main__":
    main()
