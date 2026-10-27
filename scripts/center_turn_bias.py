#!/usr/bin/env python3
"""Center an existing BC actor using its recorded training dataset (#73).

Run from the repository root with PYTHONPATH=python. --output is the new actor
artifact directory, e.g. runs/r73_centered/A_seed0/actor.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import shutil
from pathlib import Path

from flytrack_core.model import load_policy_artifact
from flytrack_core.model.artifacts import file_sha256
from flytrack_train.batches import SequenceModel
from flytrack_train.config import read_config, save_actor
from flytrack_train.recovery import EpisodeStore
from flytrack_train.turn_bias import center_turn_bias


def center_artifact(
    source: Path,
    output: Path,
    *,
    config: dict | None = None,
    device: str = "cpu",
    overwrite: bool = False,
) -> dict:
    source, output = source.resolve(), output.resolve()
    if output == source or source in output.parents or output in source.parents:
        raise ValueError("Output must not overlap the source BC directory")
    if output.exists() and not overwrite:
        raise FileExistsError(f"Output already exists: {output}; use --overwrite")
    actor, dataset, metrics = source / "actor", source / "dataset", source / "training_metrics.json"
    if not actor.is_dir() or not dataset.is_dir():
        raise ValueError("Source BC directory must contain actor/ and dataset/")
    policy, contract, card = load_policy_artifact(actor, device=device)
    recorded_config = card["metadata"]["config"]
    if config is not None and int(config.get("train", {}).get("validation_every", 10)) != int(
        recorded_config.get("train", {}).get("validation_every", 10)
    ):
        raise ValueError(
            "Override train.validation_every must match the artifact's recorded config; "
            "changing it would change the held-out split"
        )
    config = copy.deepcopy(config if config is not None else recorded_config)
    for key in ("graph_path", "frontend_path"):
        config[key] = recorded_config[key]
    paths = sorted(dataset.glob("episode_*.npz"))
    episodes = EpisodeStore(dataset).load(len(paths))
    report = center_turn_bias(
        policy, SequenceModel(policy, device), episodes, config.get("train", {})
    )
    try:
        source_path = {"path": os.path.relpath(actor)}
    except ValueError:  # Different drives cannot be expressed as a relative path.
        source_path = {"path": actor.name, "path_is_relative": False}
    provenance = {
        **source_path,
        "files": {
            str(path.relative_to(actor)): file_sha256(path)
            for path in sorted(actor.rglob("*"))
            if path.is_file()
        },
        "training_metrics_sha256": file_sha256(metrics),
    }
    metadata = dict(card.get("metadata", {}))
    # Re-save this BC stage, retaining its ledger without counting it twice.
    ledger = metadata.pop("stage_budget_ledger", [])
    policy._training_ledger = ledger[:-1]
    for key in ("config", "total_environment_steps", "environment_seed_policy", "source_code"):
        metadata.pop(key, None)
    metadata.update(turn_bias_centering=report, source_artifact=provenance, applied_post_hoc=True)
    if output.exists():
        if output.is_dir():
            shutil.rmtree(output)
        else:
            output.unlink()
    # Read authenticated bundled files without recording their local paths.
    save_actor(
        output,
        policy,
        config,
        contract,
        graph_path=actor / "graph.npz",
        frontend_path=actor / "frontend.npz",
        **metadata,
    )
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="Existing BC output with actor/ and dataset/")
    parser.add_argument("--config", type=Path, help="Override the artifact's recorded config")
    parser.add_argument("--output", type=Path, required=True, help="New actor artifact directory")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    config = read_config(args.config) if args.config else None
    report = center_artifact(
        args.source, args.output, config=config, device=args.device, overwrite=args.overwrite
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
