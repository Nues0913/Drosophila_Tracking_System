# ruff: noqa: E501, B023  (#138 exploratory analysis output lines)
"""#138 A3 proposal: the type pairs on two-synapse paths from the chemical senses to the
steering DNs, with an anatomical cap on each gain (described; the selection rule only).

PYTHONPATH=<worktree>/python python plastic_paths.py GRAPH.npz OUT.json

Rule (structure only, fixed before any training):
- Sources: the odor seed ORNs (ORN_DM1, ORN_VA2) and the wind inputs (input_mask GROUP_WIND,
  the JO neurons). Targets: the steer readout DN types (DNa01, DNa02, DNa03, DNb05, DNp09).
- A label (type pair with its ipsi/contra code) is plastic when it is either hop of a
  source -> X -> target path: source -> X with X presynaptic to a target, or X -> target with X
  postsynaptic to a source.
- Cap: per postsynaptic cell of the label, the largest existing excitatory input type's
  summed weight (signed_fraction x the initial gain, as the model) over this label's weight on
  that cell; the label's cap is the smallest over its cells, clipped to [1, 24] (the external
  review's rule, 7058912295 follow-up): a learned gain may grow to the cap times its initial
  value, never stronger than the strongest input type the cell already has. Floor 0.25.
"""

import json
import sys
from pathlib import Path

import numpy as np
from flytrack_core.contracts import GROUP_WIND
from flytrack_v2.model.graph import GraphV2

STEER = ("DNa01", "DNa02", "DNa03", "DNb05", "DNp09")
graph = GraphV2.load(Path(sys.argv[1]))
names = graph.type_names[graph.type_index]
pre, post, lp = graph.pre, graph.post, graph.label_pairs
frac = graph.synapse_count / graph.fan_in[post]
source = np.isin(names, ["ORN_DM1", "ORN_VA2"]) | ((graph.input_mask & GROUP_WIND) != 0)
target = np.isin(names, STEER)
into_target = np.zeros(graph.n_nodes, bool)
into_target[pre[target[post]]] = True  # X presynaptic to a target cell
from_source = np.zeros(graph.n_nodes, bool)
from_source[post[source[pre]]] = True  # X postsynaptic to a source cell
first = source[pre] & into_target[post]
second = from_source[pre] & target[post]
plastic = np.unique(graph.edge_label[first | second])
excitatory = graph.sign[pre] > 0
rows = []
for label in plastic:
    edges = graph.edge_label == label
    caps = []
    for cell in np.unique(post[edges]):
        into = post == cell
        by_type = np.bincount(
            graph.type_index[pre[into & excitatory]],
            weights=frac[into & excitatory],
            minlength=len(graph.type_names),
        )
        own = frac[edges & into].sum()
        caps.append(by_type.max() / own if own > 0 else 1.0)
    a, b = lp[label][:2]
    code = int(lp[label][2]) if lp.shape[1] > 2 else 0
    rows.append({
        "label": int(label), "pre": str(graph.type_names[a]), "post": str(graph.type_names[b]),
        "code": code, "hop": "first" if (edges & first).any() else "second",
        "edges": int(edges.sum()), "synapses": float(graph.synapse_count[edges].sum()),
        "sign": int(np.sign(graph.sign[pre[edges]].sum())),
        "cap": float(np.clip(min(caps), 1.0, 24.0)),
    })  # fmt: skip
report = {
    "graph_sha256": graph.source_sha256,
    "rule": "both hops of every chemical-sense -> X -> steer-DN path; cap clip [1, 24]",
    "steer_types": list(STEER),
    "n_labels": len(rows),
    "labels": rows,
}
Path(sys.argv[2]).write_text(json.dumps(report, indent=1) + "\n")
print(len(rows), "plastic labels;", sum(r["hop"] == "first" for r in rows), "first hop")
for r in rows:
    print(
        f"  {r['hop']:6s} {r['pre']:14s} -> {r['post']:14s} code {r['code']} edges {r['edges']:3d} "
        f"syn {r['synapses']:5.0f} sign {r['sign']:+d} cap {r['cap']:5.2f}"
    )
