"""Workspace-only wrapper for controlled PPO actor parameter subsets.

Uses the production PPO implementation. The only hook sets actor trainability
and an optional initial exploration scale before production recovery loading.
Every experiment records the wrapper hash and the selected parameter names.
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import torch
from flytrack_train import ppo
from flytrack_train.config import read_config, write_json
from flytrack_train.parallel import set_default_workers
from torch.optim.optimizer import register_optimizer_step_post_hook

ROOT = Path(__file__).resolve().parents[2]


# Parameter-name prefixes each subset may train; "full" trains everything.
SUBSETS = {"readout": ("readout.",), "readout_forward": ("readout.forward",)}


def configure_actor(policy, settings):
    subset = settings.get("actor_subset", "full")
    if subset != "full" and subset not in SUBSETS:
        raise ValueError(f"actor_subset must be full or one of {sorted(SUBSETS)}")
    freeze_std = settings.get("freeze_log_std", False)
    initial_log_std = settings.get("initial_log_std")
    if initial_log_std is not None:
        if not isinstance(initial_log_std, (int, float)) or not -5 <= initial_log_std <= 0:
            raise ValueError("initial_log_std must be a number in [-5, 0]")
        with torch.no_grad():
            policy.log_std.fill_(initial_log_std)
    selected = []
    for name, parameter in policy.named_parameters():
        enabled = subset == "full" or name.startswith(SUBSETS[subset])
        if name == "log_std" and freeze_std:
            enabled = False
        parameter.requires_grad_(enabled)
        if enabled:
            selected.append(name)
    if not selected:
        raise ValueError("The selected subset has no parameters")
    return selected


def bounded_change(policy, settings):
    """Optional BC-centred bound: after every optimizer step, project the change of
    the named parameters (relative to the starting BC) back into an L2 ball."""
    bound = settings.get("change_bound")
    if not bound:
        return None
    radius, prefixes = float(bound["radius"]), tuple(bound["prefixes"])
    if radius <= 0 or not prefixes:
        raise ValueError("change_bound needs a positive radius and parameter prefixes")
    named = [(n, p) for n, p in policy.named_parameters() if n.startswith(prefixes)]
    reference = [p.detach().clone() for _, p in named]
    ledger = {"radius": radius, "names": [n for n, _ in named], "steps": 0, "projections": 0}

    @torch.no_grad()
    def project(optimizer, *_):
        change = torch.cat([(p - r).flatten() for (_, p), r in zip(named, reference, strict=True)])
        norm = float(change.norm())
        ledger["steps"] += 1
        if norm > radius:
            ledger["projections"] += 1
            for (_, p), r in zip(named, reference, strict=True):
                p.copy_(r + (p - r) * (radius / norm))
        ledger["final_norm"] = min(norm, radius)

    return register_optimizer_step_post_hook(project), ledger


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--device", default="cpu", help="Recorded in the artifact config")
    args = parser.parse_args()
    config = read_config(args.config)
    output = Path(config["output"])
    if not output.resolve().is_relative_to(ROOT):
        parser.error("All experiment output must stay in this workspace")
    if not Path(config["init_artifact"]).resolve().is_relative_to(ROOT):
        parser.error("Use a baseline artifact inside this workspace")
    if args.resume:
        config["resume"] = True
    settings = config.setdefault("local_experiment", {})
    if args.device != settings.get("device", "cpu"):
        parser.error("--device must match local_experiment.device in the configuration")
    settings["runner_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    torch.set_num_threads(args.threads)
    set_default_workers(args.workers)
    original = ppo.prepare_policy
    original_save = ppo.save_actor

    def prepare(local_config, device):
        policy, contract = original(local_config, device)
        policy._local_original_trainability = {
            name: parameter.requires_grad for name, parameter in policy.named_parameters()
        }
        selected = configure_actor(policy, settings)
        # The reference is the starting BC: recovery loads checkpoints after this.
        bounds.append(bounded_change(policy, settings))
        output.mkdir(parents=True, exist_ok=True)
        write_json(output / "actor_selection.json", {
            "settings": settings, "trainable_names": selected,
            "note": "Production zero-learning-rate groups can further freeze parameters.",
        })
        return policy, contract

    def save(directory, policy, *positional, **keywords):
        # The portable artifact records architecture capacity, not this run's
        # temporary optimization subset. Ordinary loading restores architecture
        # defaults and validates its parameter count against the model card.
        flags = {name: p.requires_grad for name, p in policy.named_parameters()}
        try:
            for name, parameter in policy.named_parameters():
                parameter.requires_grad_(policy._local_original_trainability[name])
            original_save(directory, policy, *positional, **keywords)
        finally:
            for name, parameter in policy.named_parameters():
                parameter.requires_grad_(flags[name])

    bounds = []
    ppo.prepare_policy = prepare
    ppo.save_actor = save
    try:
        ppo.train_ppo(config, output, int(config["seed"]), torch.device(args.device))
    finally:
        ppo.prepare_policy = original
        ppo.save_actor = original_save
        for bound in filter(None, bounds):
            bound[0].remove()
            write_json(output / "change_bound.json", bound[1])


if __name__ == "__main__":
    main()
