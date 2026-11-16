"""S6 derived graph (PM, #35 exploration; the formal subgraph is NOT changed).

  python -I explore/s6/derive_graph.py RAW_DIR SUBGRAPH_DIR OUT_DIR   (flytrack-ml:0.1; needs pyarrow, pandas)
The extraction keeps a cell-to-cell edge only if it has >= 50 synapses, and drops seed cells that keep no edge.
A seed population that converges on the forward DN through many small edges disappears: LC9 -> DNp09 is DNp09's
largest input (1,977 synapses, 12.4 %); none of it is kept and only 21 of 219 LC9 cells are in the subgraph.
Rule here, for the forward DN (DNp09) and the visual seed types only: apply the 50-synapse threshold to the summed
synapses from a seed TYPE onto one DNp09 cell; a type passes only if it passes for BOTH DNp09 cells (LC9 only:
LC10a and LPLC2 reach 50 on one side only). For a passing type, every published cell of that type with >= 1
synapse onto DNp09 becomes a node (type, Dale sign, group and input masks as the cells of that type already in the
subgraph; side from the publisher's somaSide) and every edge from that type onto DNp09 is kept with its real
synapse count. Nothing else changes: no other new edges, the other nodes and the 31,682 edges are untouched.
New nodes are appended after the existing ones. Writes OUT_DIR/graph.npz and OUT_DIR/derivation.json.
"""
import json
import os
import sys

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.feather as feather
import pyarrow.ipc as ipc

raw, sub, out = sys.argv[1], sys.argv[2], sys.argv[3]
THRESHOLD, TARGET, SEEDS = 50, "DNp09", ["LC10a", "LC10d", "LC9", "LC11", "LPLC2"]
g = dict(np.load(sub + "/graph.npz", allow_pickle=False))
ids = g["node_ids"]
index = {int(b): i for i, b in enumerate(ids)}
type_names = g["type_names"].astype(str)
names = type_names[g["type_index"]]
targets = [int(b) for b in ids[names == TARGET]]
ann = feather.read_table(raw + "/body-annotations-male-cns-v1.0-minconf-0.5.feather",
                         columns=["bodyId", "type", "somaSide"]).to_pandas().set_index("bodyId")
weights = ipc.open_file(raw + "/connectome-weights-male-cns-v1.0-minconf-0.5.feather")
wanted = pa.array(targets, pa.int64())
parts = []
for i in range(weights.num_record_batches):
    batch = weights.get_batch(i)
    mask = pc.is_in(batch.column("body_post"), value_set=wanted)
    if pc.any(mask).as_py():
        parts.append(batch.filter(mask).to_pandas())
e = pd.concat(parts, ignore_index=True)
total = int(e.weight.sum())
e["type"] = e.body_pre.map(ann.type)
e["pre_side"] = e.body_pre.map(ann.somaSide)
e["post_side"] = e.body_post.map(ann.somaSide)
existing = set(zip(g["pre"].tolist(), g["post"].tolist()))
kept_before = int(sum(w for a, b, w in zip(e.body_pre, e.body_post, e.weight) if (index.get(a), index.get(b)) in existing))
seed_edges = e[e.type.isin(SEEDS)]
by_type = seed_edges.groupby(["type", "body_post"]).weight.sum()
passing = sorted(t for t in SEEDS if all(by_type.get((t, p), 0) >= THRESHOLD for p in targets))
side_code = {}
for name in SEEDS:  # publisher side label -> the subgraph's side code, from cells already in the subgraph
    for body in ids[names == name]:
        side_code.setdefault(str(ann.somaSide.get(int(body))), set()).add(int(g["side"][index[int(body)]]))
assert all(len(v) == 1 for v in side_code.values()), side_code
side_code = {k: v.pop() for k, v in side_code.items()}
new_ids, new_type, new_sign, new_side, new_group, new_input = [], [], [], [], [], []
pre, post, count = [], [], []
report_types = []
for name in passing:
    part = seed_edges[seed_edges.type == name]
    template = np.flatnonzero(names == name)[0]
    added_nodes = 0
    for body in sorted(set(part.body_pre)):
        if int(body) not in index:
            label = str(ann.somaSide.get(int(body)))
            if label not in side_code:
                continue  # unknown side: not injected, not added
            index[int(body)] = len(ids) + len(new_ids)
            new_ids.append(int(body)); new_type.append(g["type_index"][template]); new_sign.append(g["sign"][template])
            new_side.append(side_code[label]); new_group.append(g["group_mask"][template]); new_input.append(g["input_mask"][template])
            added_nodes += 1
    added_edges = added_syn = 0
    for a, b, w in zip(part.body_pre, part.body_post, part.weight):
        if int(a) in index and (index[int(a)], index[int(b)]) not in existing:
            pre.append(index[int(a)]); post.append(index[int(b)]); count.append(float(w)); added_edges += 1; added_syn += int(w)
    sides = part.groupby(["pre_side", "post_side"]).weight.sum()
    report_types.append({"type": name, "sign": int(g["sign"][template]), "published_cells": int((ann.type == name).sum()),
                         "cells_in_subgraph_before": int((names == name).sum()), "cells_with_synapse_onto_target": int(part.body_pre.nunique()),
                         "added_nodes": added_nodes, "added_edges": added_edges, "added_synapses": added_syn,
                         "synapses_by_pre_side_post_side": {f"{a}->{b}": int(w) for (a, b), w in sides.items()},
                         "edge_synapses_p50_max": [float(part.weight.median()), int(part.weight.max())]})
new = dict(g)
for key, extra in (("node_ids", new_ids), ("type_index", new_type), ("sign", new_sign), ("side", new_side),
                   ("group_mask", new_group), ("input_mask", new_input), ("pre", pre), ("post", post), ("synapse_count", count)):
    new[key] = np.concatenate([g[key], np.asarray(extra, dtype=g[key].dtype)])
assert len(set(zip(new["pre"].tolist(), new["post"].tolist()))) == len(new["pre"])
assert len(np.unique(new["node_ids"])) == len(new["node_ids"])
os.makedirs(out, exist_ok=True)
np.savez_compressed(out + "/graph.npz", **new)
report = {"rule": __doc__, "target_bodies": targets, "target_input_synapses": total, "seed_types_passing": passing,
          "seed_type_synapses_onto_target": {t: int(seed_edges.weight[seed_edges.type == t].sum()) for t in SEEDS},
          "kept_share_before": kept_before / total, "kept_share_after": (kept_before + sum(count)) / total,
          "nodes_before": int(len(ids)), "nodes_after": int(len(new["node_ids"])), "edges_before": int(len(g["pre"])),
          "edges_after": int(len(new["pre"])), "types": report_types}
json.dump(report, open(out + "/derivation.json", "w"), indent=1)
print(json.dumps({k: v for k, v in report.items() if k != "rule"}, indent=1))
