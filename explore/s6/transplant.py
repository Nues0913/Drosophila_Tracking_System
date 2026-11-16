"""Copy an actor's learned parameters onto the S6 derived graph (PM, #35 S6 exploration).

  python explore/s6/transplant.py SOURCE_ACTOR GRAPH_NPZ OUT_ACTOR
Run in flytrack-ml:0.1 from the repo root (PYTHONPATH=python). The policy is rebuilt on GRAPH_NPZ with the
source card's model config. Type-pair conductances, time constants, biases, the readout and every other
parameter keep the source values; a type pair that does not exist in the source graph (S6: LC9 -> DNp09) keeps
the model's own initial conductance (init_alpha, as in a fresh policy). The stage budget ledger of the source
card is carried over so that BC continues the same chain.
"""
import json
import sys

import numpy as np
import torch

from flytrack_core.model import load_policy_artifact, save_policy_artifact
from flytrack_core.model.policy import build_policy

source, graph_path, out = sys.argv[1], sys.argv[2], sys.argv[3]
old, contract, card = load_policy_artifact(source, device="cpu")
policy = build_policy(card["variant"], graph_path, contract, source + "/frontend.npz", **card["model_config"])
old_names = np.asarray(old.readout._graph.type_names).astype(str)
new_names = np.asarray(policy.readout._graph.type_names).astype(str)
assert np.array_equal(old_names, new_names), "the derived graph must not add or remove cell types"
state = {key: value.clone() for key, value in old.state_dict().items()}
old_pairs = {tuple(pair): i for i, pair in enumerate(old.core.pair_types.tolist())}
alpha = policy.core.raw_alpha.detach().clone()
fresh = []
for i, pair in enumerate(policy.core.pair_types.tolist()):
    if tuple(pair) in old_pairs:
        alpha[i] = state["core.raw_alpha"][old_pairs[tuple(pair)]]
    else:
        fresh.append([str(new_names[pair[0]]), str(new_names[pair[1]])])
assert len(alpha) == len(old_pairs) + len(fresh)
state["core.raw_alpha"] = alpha
policy.load_state_dict(state, strict=True)
policy.set_lesion(card.get("runtime_lesion", "none"))
metadata = dict(card.get("metadata", {}))
metadata["s6_transplant"] = {"source_card_sha256_of_weights": card["files"]["weights.pt"], "new_type_pairs": fresh,
                             "new_pair_alpha": policy.core.init_alpha, "source_nodes": int(old.n_nodes), "nodes": int(policy.n_nodes)}
new_card = save_policy_artifact(out, policy, card["requested_model_config"], contract, graph_path, source + "/frontend.npz", metadata)
print(json.dumps({"new_type_pairs": fresh, "new_pair_alpha": policy.core.init_alpha, "nodes": int(policy.n_nodes),
                  "trainable_parameters": new_card["trainable_parameters"], "source_trainable_parameters": card["trainable_parameters"]}))
