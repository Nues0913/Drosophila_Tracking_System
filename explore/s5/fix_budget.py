"""Re-save an S5 actor with metadata.actor_parameter_budget recomputed (PM, #35 S5).

  python explore/s5/fix_budget.py ACTOR_DIR
The first fit_readout.py copied the formal card's budget (15,790 declared) although the actor now declares the
268 row weights too, and the evaluation refused the artifact. Weights and everything else are unchanged.
"""
import json
import sys

import torch

from flytrack_core.model import load_policy_artifact, save_policy_artifact
from flytrack_train.config import count_effective_parameters

actor = sys.argv[1]
policy, contract, card = load_policy_artifact(actor, device="cpu")
before = torch.load(actor + "/weights.pt", map_location="cpu", weights_only=True)
metadata = dict(card["metadata"])
old = metadata["actor_parameter_budget"]
metadata["actor_parameter_budget"] = count_effective_parameters(policy, metadata["config"]["train"])
save_policy_artifact(actor, policy, card["requested_model_config"], contract, actor + "/graph.npz", actor + "/frontend.npz", metadata)
after = torch.load(actor + "/weights.pt", map_location="cpu", weights_only=True)
assert before.keys() == after.keys() and all(torch.equal(before[k], after[k]) for k in before)
print(json.dumps({"before": old, "after": metadata["actor_parameter_budget"]}))
