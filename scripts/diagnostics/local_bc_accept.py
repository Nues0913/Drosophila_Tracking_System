"""Accept an additional BC start with the same evidence as the first one.

Requires the existing BC DN gate and a full 60-second closed-loop development
evaluation (CPU evaluator, as for BC seed 0), then binds both by SHA-256.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

from flytrack_train.config import write_json

ROOT = Path(__file__).resolve().parents[2]
STUDY = ROOT / "runs/local_ppo_bc"
sys.path.insert(0, str(ROOT / "scripts"))
from dn_health import health_gate  # noqa: E402


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--name", required=True, help="File stem, e.g. bc1")
    args = parser.parse_args()
    if not (ROOT / args.artifact).resolve().is_relative_to(ROOT):
        parser.error("Use an artifact inside this workspace")
    progress = json.loads((ROOT / args.artifact).parent.joinpath("progress.json").read_text())
    if progress.get("phase") != "complete":
        raise ValueError("BC training has not completed")
    config = STUDY / "configs/study_eval.yaml"
    health = STUDY / f"{args.name}_final_health.json"
    evaluation = STUDY / f"{args.name}_development.json"
    if not health.exists():
        with (STUDY / f"{args.name}_final_health.log").open("a") as handle:
            subprocess.run([sys.executable, str(ROOT / "scripts/dn_health.py"),
                            "--artifact", str(args.artifact), "--config", str(config),
                            "--stage", "bc", "--sparse", "--output", str(health)],
                           cwd=ROOT, stdout=handle, stderr=subprocess.STDOUT, check=True)
    if not evaluation.exists() or not json.loads(evaluation.read_text()).get("complete"):
        with (STUDY / f"{args.name}_development.log").open("a") as handle:
            subprocess.run([sys.executable, str(ROOT / "scripts/diagnostics/local_ppo_eval.py"),
                            "--config", str(config), "--output", str(evaluation),
                            "--artifacts", str(args.artifact), "--episodes", "24",
                            "--lanes", "24", "--workers", "2"],
                           cwd=ROOT, stdout=handle, stderr=subprocess.STDOUT, check=True)
    report = json.loads(health.read_text())
    gate = health_gate(report, "bc")
    if report.get("source_actor_files") != json.loads(
        (ROOT / args.artifact / "model_card.json").read_text()
    )["files"]:
        raise ValueError("DN evidence belongs to different actor files")
    summary = json.loads(evaluation.read_text())["policies"][str(args.artifact)]["cells"][
        "curved/normal"]["summary"]
    acceptance = {
        "accepted": gate["passed"],
        "artifact": str(args.artifact),
        "model_card_sha256": digest(ROOT / args.artifact / "model_card.json"),
        "dn_health_file": health.name, "dn_health_sha256": digest(health),
        "dn_gate": gate,
        "evaluation_file": evaluation.name, "evaluation_sha256": digest(evaluation),
        "scope": "additional BC start, old main graph; development episodes only",
        "summary": summary,
    }
    write_json(STUDY / f"baseline_acceptance_{args.name}.json", acceptance)
    print(json.dumps({"accepted": gate["passed"], "checks": gate["checks"],
                      **{k: summary[k] for k in ("episode_return", "success", "collision_rate",
                                                 "tracked_fraction")}}, indent=2))


if __name__ == "__main__":
    main()
