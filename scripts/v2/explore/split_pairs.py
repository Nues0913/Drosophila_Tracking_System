"""#138 A + ipsi/contra gains: which type pairs to split, from the graph's structure only.

PYTHONPATH=<worktree>/python python split_pairs.py GRAPH.npz OUT.json [HOPS] [MAX_EXTRA]

Linear propagation in the model's own weight form: w_e = g * n_e / fan_in(post), every g at the
same initial value, unsigned (as #138 7053575411) and signed. Odor seeds (ORN_DM1, ORN_VA2) to
the readout DNs (DNa01, DNa02, DNa03, DNb05, DNp09) within HOPS hops.

1. Flow: each edge's share of the odor -> readout-DN flow, F(pre) * w_e * B(post), F the summed
   forward reach from the odor seeds and B the summed backward reach from the DNs (hops <= HOPS
   each). A type pair's flow is the sum over its edges.
2. Candidates: type pairs with both an ipsilateral and a contralateral edge (both ends lateral),
   by flow, plus ORN_DM1 -> DM1_lPN and ORN_VA2 -> VA2_adPN (ipsilateral release, Gaudry 2013),
   until the extra gains reach MAX_EXTRA.
3. Capacity: the laterality index at each DN type, (L - R | left odor) - (L - R | right odor) over
   the mean drive, at equal gains and with every split pair at ipsi x K, contra x 1/K.
"""

import json
import sys
from pathlib import Path

import numpy as np
import scipy.sparse as sp
from flytrack_v2.model.graph import GraphV2

graph = GraphV2.load(Path(sys.argv[1]))
out = Path(sys.argv[2])
HOPS = int(sys.argv[3]) if len(sys.argv) > 3 else 8
MAX_EXTRA = int(sys.argv[4]) if len(sys.argv) > 4 else 350
names = graph.type_names[graph.type_index]
n = graph.n_nodes
pre, post = graph.pre, graph.post
frac = graph.synapse_count / graph.fan_in[post]
sign = graph.sign[pre].astype(float)
side = graph.side
lateral = (side[pre] != 0) & (side[post] != 0)
ipsi = lateral & (side[pre] == side[post])
contra = lateral & (side[pre] != side[post])
pair_names = np.array([f"{a}->{b}" for a, b in zip(names[pre], names[post])])
odor = np.flatnonzero(np.isin(names, ["ORN_DM1", "ORN_VA2"]))
dn_types = ["DNa01", "DNa02", "DNa03", "DNb05", "DNp09"]
dns = np.flatnonzero(np.isin(names, dn_types))


def matrix(weights):
    return sp.csr_matrix((weights, (post, pre)), shape=(n, n))  # x_next = M @ x


def reach(M, start, hops):
    x, total = start.copy(), np.zeros(n)
    for _ in range(hops):
        x = M @ x
        total += x
    return total


M = matrix(frac)
start = np.zeros(n)
start[odor] = 1.0
F = reach(M, start, HOPS - 1) + start
end = np.zeros(n)
end[dns] = 1.0
B = reach(M.T.tocsr(), end, HOPS - 1) + end
edge_flow = F[pre] * frac * B[post]
total_flow = edge_flow.sum()
pairs = {}
for p in np.unique(pair_names):
    m = pair_names == p
    pairs[p] = {
        "flow": float(edge_flow[m].sum() / total_flow),
        "ipsi_edges": int((m & ipsi).sum()),
        "contra_edges": int((m & contra).sum()),
        "mid_edges": int((m & ~lateral).sum()),
        "ipsi_synapses": float(graph.synapse_count[m & ipsi].sum()),
        "contra_synapses": float(graph.synapse_count[m & contra].sum()),
    }
ranked = sorted(pairs, key=lambda p: -pairs[p]["flow"])
first = ["ORN_DM1->DM1_lPN", "ORN_VA2->VA2_adPN"]


def extra(p):  # added gains when p is split into ipsi / contra (/ mid)
    r = pairs[p]
    return (r["ipsi_edges"] > 0) + (r["contra_edges"] > 0) + (r["mid_edges"] > 0) - 1


chosen, budget, covered = [], 0, 0.0
for p in first + ranked:
    r = pairs[p]
    if p in chosen or not (r["ipsi_edges"] and r["contra_edges"]):
        continue
    if budget + extra(p) > MAX_EXTRA:
        break
    chosen.append(p)
    budget += extra(p)
    covered += r["flow"]


def laterality(weights, signed):
    Mw = matrix(weights * (sign if signed else 1.0))
    result = {}
    drive = {}
    for s in (-1, 1):
        x = np.zeros(n)
        x[odor[side[odor] == s]] = 1.0
        drive[s] = reach(Mw, x, HOPS)
    for t in dn_types:
        left = np.flatnonzero((names == t) & (side == -1))
        right = np.flatnonzero((names == t) & (side == 1))
        d = {s: drive[s][left].sum() - drive[s][right].sum() for s in (-1, 1)}
        mean = np.mean([abs(drive[s][left].sum()) + abs(drive[s][right].sum()) for s in (-1, 1)]) / 2
        result[t] = float((d[-1] - d[1]) / mean) if mean else None
    return result


is_split = np.isin(pair_names, chosen)
capacity = {}
for K in (1.0, 2.0, 4.0):
    scale = np.where(is_split & ipsi, K, np.where(is_split & contra, 1 / K, 1.0))
    capacity[f"K={K:g}"] = {
        "unsigned": laterality(frac * scale, False),
        "signed": laterality(frac * scale, True),
    }
report = {
    "graph_sha256": graph.source_sha256,
    "hops": HOPS,
    "max_extra": MAX_EXTRA,
    "labels_before": int(len(graph.label_pairs)),
    "extra_gains": int(budget),
    "chosen_pairs": chosen,
    "chosen_flow_share": covered,
    "top_pairs": [{"pair": p, **pairs[p]} for p in ranked[:40]],
    "chosen_detail": [{"pair": p, **pairs[p]} for p in chosen],
    "capacity": capacity,
}
out.write_text(json.dumps(report, indent=1) + "\n")
print("labels", report["labels_before"], "+", budget, "chosen", len(chosen), "flow share", round(covered, 3))
for k, v in capacity.items():
    print(k, {t: round(x, 3) for t, x in v["unsigned"].items()}, "signed", {t: round(x, 3) for t, x in v["signed"].items()})
