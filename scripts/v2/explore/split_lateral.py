"""#138 A2: the A graph (v2_cx) with ipsilateral and contralateral gains on the odor seeds' outputs.

PYTHONPATH=python python scripts/v2/explore/split_lateral.py IN_GRAPH.npz OUT_DIR

Rule (registered before training, structure only): every type pair whose presynaptic type is an
odor seed type (ORN_DM1, ORN_VA2) and which has both an ipsilateral and a contralateral edge
(both ends lateral) gets two gains instead of one: ipsilateral (pre side == post side) and
contralateral. Left -> left shares with right -> right and left -> right with right -> left, so
the mirror symmetry stays; wiring, synapse counts and signs are unchanged. Basis: one ORN's
ipsilateral axon branch releases more transmitter than its contralateral branch (Gaudry et al.,
Nature 2013), a property of the presynaptic branch and so of every partner on it.

The model needs no change: GraphV2 shares a gain per distinct row of edge_type_pairs, so the
split is a third column (0 shared, 1 ipsilateral, 2 contralateral, 3 an edge with a midline end
in a split pair). RW2 keeps each edge's label when it swaps targets within a side pair, and ER2
carries the labels with the sampled edges, so both controls get the same number of gains.
"""

import json
import sys
from pathlib import Path

import numpy as np
from flytrack_v2.model.graph import GraphV2

SEED_TYPES = ("ORN_DM1", "ORN_VA2")
source, out = Path(sys.argv[1]), Path(sys.argv[2])
graph = GraphV2.load(source)
names = graph.type_names[graph.type_index]
pre, post, side = graph.pre, graph.post, graph.side
lateral = (side[pre] != 0) & (side[post] != 0)
ipsi = lateral & (side[pre] == side[post])
contra = lateral & (side[pre] != side[post])
pairs = graph.label_pairs[graph.edge_label]  # [E, 2] type indices
if pairs.shape[1] != 2:
    raise ValueError("the input graph already has a split column")
split_pairs = []
for label, (a, b) in enumerate(graph.label_pairs):
    if graph.type_names[a] not in SEED_TYPES:
        continue
    edges = graph.edge_label == label
    if (edges & ipsi).any() and (edges & contra).any():
        split_pairs.append(label)
in_split = np.isin(graph.edge_label, split_pairs)
code = np.where(~in_split, 0, np.where(ipsi, 1, np.where(contra, 2, 3)))
labelled = GraphV2(
    **{
        name: getattr(graph, name)
        for name in (
            "node_ids", "type_index", "type_names", "side", "sign", "pre", "post",
            "synapse_count", "input_mask", "group_mask", "brain_input_synapses",
            "subgraph_input_synapses", "protected",
        )
    },
    edge_type_pairs=np.column_stack((pairs, code)).astype(np.int64),
    origin_sha256=graph.source_sha256,
)  # fmt: skip
out.mkdir(parents=True, exist_ok=True)
labelled.save(out / "graph.npz")
reloaded = GraphV2.load(out / "graph.npz")
assert np.array_equal(reloaded.edge_label, labelled.edge_label)
detail = []
for label in split_pairs:
    a, b = graph.label_pairs[label]
    edges = graph.edge_label == label
    detail.append({
        "pair": f"{graph.type_names[a]}->{graph.type_names[b]}",
        "ipsi_edges": int((edges & ipsi).sum()),
        "contra_edges": int((edges & contra).sum()),
        "midline_edges": int((edges & ~lateral).sum()),
        "ipsi_synapses": float(graph.synapse_count[edges & ipsi].sum()),
        "contra_synapses": float(graph.synapse_count[edges & contra].sum()),
    })  # fmt: skip
report = {
    "source_graph_sha256": graph.source_sha256,
    "graph_sha256": reloaded.source_sha256,
    "seed_types": list(SEED_TYPES),
    "gains_before": int(len(graph.label_pairs)),
    "gains_after": int(len(labelled.label_pairs)),
    "split_pairs": detail,
}
(out / "manifest.json").write_text(json.dumps(report, indent=1) + "\n")
print(report["gains_before"], "->", report["gains_after"], "gains;", len(detail), "pairs split")
