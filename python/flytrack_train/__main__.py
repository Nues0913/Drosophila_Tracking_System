"""Command entry point; importing this package does not start training."""

from __future__ import annotations

import argparse
import hashlib
import shutil
from pathlib import Path

import torch
from flytrack_core.model import load_policy_artifact

from .config import contract_from_config, read_config, write_json


def export_actor(config, source: Path, destination: Path):
    _, contract, card = load_policy_artifact(source, device="cpu")
    if contract.sha256() != contract_from_config(config).sha256():
        raise ValueError("Export sensor contract does not match configuration")
    if (
        destination.exists()
        or source.resolve() == destination.resolve()
        or source.resolve() in destination.resolve().parents
    ):
        raise ValueError("Export destination must be a new directory outside its source artifact")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.mkdir()
    for name in ("model_card.json", *card["files"]):
        shutil.copyfile(source / name, destination / name)
    checksums = {
        str(path.relative_to(destination)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(destination.rglob("*"))
        if path.is_file()
    }
    write_json(
        destination / "export_manifest.json",
        {
            "source": str(source),
            "sensor_contract_sha256": contract.sha256(),
            "files_sha256": checksums,
            "contains_critic": False,
            "contains_auxiliary_decoder": False,
        },
    )


def main():
    parser = argparse.ArgumentParser(description="FlyTrack offline learning and evaluation")
    parser.add_argument(
        "command",
        choices=("pretrain", "bc", "ppo", "es", "evaluate", "export", "validate-pretrain"),
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", help="Override the output directory in YAML")
    parser.add_argument("--seed", type=int, help="Training seed override")
    parser.add_argument("--device", default="auto", help="cpu, cuda, cuda:0, or auto")
    parser.add_argument("--artifact", help="Existing actor artifact for export")
    parser.add_argument(
        "--checkpoint", help="Research auxiliary decoder checkpoint for validate-pretrain"
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume the same stage from its source-bound recovery checkpoint",
    )
    parser.add_argument(
        "--variant",
        choices=("V", "O", "VO", "VO-late", "GRU", "RS"),
        help="Override variant before model construction",
    )
    args = parser.parse_args()
    config = read_config(args.config)
    if args.resume:
        config["resume"] = True
    if args.variant:
        config["variant"] = args.variant
    seed = int(args.seed if args.seed is not None else config.get("seed", 0))
    output = Path(
        args.output or config.get("output", f"runs/{config.get('variant', 'VO')}/seed_{seed}")
    )
    device = (
        torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if args.device == "auto"
        else torch.device(args.device)
    )
    if args.command == "validate-pretrain":
        from .validation import validate_pretraining

        if not args.output:
            output = output / "validation"
        if not args.artifact or not args.checkpoint:
            parser.error("validate-pretrain requires --artifact and --checkpoint")
        result = validate_pretraining(
            config, output, Path(args.artifact), Path(args.checkpoint), device
        )
        if not result["passed"]:
            raise SystemExit("Held-out sensory decoding did not pass its preregistered threshold")
        return
    if args.command == "export":
        source = args.artifact or config.get("artifact")
        if not source:
            parser.error("export requires --artifact or artifact in YAML")
        export_actor(config, Path(source), output)
        return
    if args.command == "evaluate":
        from .evaluation import evaluate

        evaluate(config, output, device)
    else:
        if args.command == "pretrain":
            from .pretrain import train_pretrain as run
        elif args.command == "bc":
            from .imitation import train_bc as run
        elif args.command == "ppo":
            from .ppo import train_ppo as run
        else:
            from .es import train_es as run
        run(config, output, seed, device)


if __name__ == "__main__":
    main()
