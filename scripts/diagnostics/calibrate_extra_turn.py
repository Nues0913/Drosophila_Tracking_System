"""Calibrate the opt-in extra turn rows (#4 B3) on stored BC episodes.

Usage: [DEVICE=cpu] calibrate_extra_turn.py [--shared-adaptation] [--eps EPS] OUT.json
       BC_DATASET_DIR EPISODE_COUNT ARTIFACT TYPE [TYPE ...]
The artifact is run over the stored episodes (normally the round-0 teacher episodes
of a BC run in the target world). For each TYPE the normalized L/R difference
(L - R) / max(L + R, eps) is collected on the training episodes (every 10th is held
out, as in BC), skipping steps with L + R < eps (both sides silent, as in the first
~3 s of an episode), and center = mean, scale = 1 / std are written as the
``readout_extra_turn_*`` model entries for the BC config; the computation is
flytrack_train.extra_turn, which formal BC runs call on their round-0 episodes
(``train.calibrate_extra_turn``). --shared-adaptation encodes the episodes with
the shared bilateral ORN adaptation (#4 P2), as a warm-started BC run would.
"""

import argparse
import json
import os
from pathlib import Path

from flytrack_core.model import load_policy_artifact
from flytrack_core.model.policy import ConnectomePolicy
from flytrack_train.batches import SequenceModel
from flytrack_train.extra_turn import calibrate_extra_turns
from flytrack_train.recovery import EpisodeStore, held_out_episodes


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("out", type=Path)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("count", type=int)
    parser.add_argument("artifact")
    parser.add_argument("types", nargs="+")
    parser.add_argument("--shared-adaptation", action="store_true")
    parser.add_argument("--eps", type=float, default=0.1)
    args = parser.parse_args()
    # Explicit and recorded, never picked from what the host happens to have (#63 review).
    device = os.environ.get("DEVICE", "cpu")
    episodes = EpisodeStore(args.dataset).load(args.count)
    held = set(held_out_episodes(args.count, 10))
    policy, _, card = load_policy_artifact(args.artifact, device=device)
    if not isinstance(policy, ConnectomePolicy) or policy.readout.mode != "structured":
        raise SystemExit("Needs a connectome policy with the structured DN readout")
    policy.eval()
    policy.encoder.shared_adaptation = args.shared_adaptation
    # Placeholder constants (in memory only; nothing is saved): only extra_turn_terms is used.
    placeholder = [0.0] * len(args.types), [1.0] * len(args.types)
    policy.set_readout_extra_turns(args.types, *placeholder, args.eps)
    model = SequenceModel(policy, device)
    try:
        calibration = calibrate_extra_turns(policy, model, episodes, model.encode(episodes), held)
    except ValueError as error:
        raise SystemExit(str(error)) from None
    result = {
        "artifact": str(args.artifact),
        "artifact_variant": card["variant"],
        "device": device,
        "dataset": str(args.dataset),
        "episodes": args.count,
        **calibration,
    }
    args.out.write_text(json.dumps(result, indent=1))
    fits = result["closed_form"]
    print(json.dumps(result["model"]), json.dumps({k: v["held_out_corr"] for k, v in fits.items()}))


if __name__ == "__main__":
    main()
