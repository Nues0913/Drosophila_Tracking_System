# ruff: noqa: E501, B023  (#138 exploratory analysis output lines)
"""#138 A: learned ipsi/contra gains and the odor laterality they give, per stage (described only).

cd <A worktree> && OMP_NUM_THREADS=2 PYTHONPATH=python python learned_gains.py RUN_DIR [STAGE ...]

RUN_DIR holds stage1/final and stage2/final. For each stage: the ipsi/contra gain ratio of every
split type pair (the 5 ORN -> PN pairs first, where Gaudry 2013 predicts > 1), and the odor
laterality (linear, 8 hops, as #138 7053575411) at equal gains and at the learned gains: at the
PNs (left odor, 1 hop, left minus right over their mean) and at the readout DNs
((L - R | left odor) - (L - R | right odor) over the mean drive), unsigned and signed.
"""

import sys
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import torch

sys.path.insert(0, "scripts/v2")
import evaluate_v2 as ev  # noqa: E402

HOPS = 8
ORN_PN = {"DM1_lPN", "DM2_lPN", "M_l2PNl20", "VA2_adPN", "VA7m_lPN"}
run = Path(sys.argv[1])
stages = sys.argv[2:] or ["stage1", "stage2"]


def laterality(g, gains, signed):
    names = g.type_names[g.type_index]
    n, side = g.n_nodes, g.side
    w = gains[g.edge_label] * g.synapse_count / g.fan_in[g.post]
    if signed:
        w = w * g.sign[g.pre]
    M = sp.csr_matrix((w, (g.post, g.pre)), shape=(n, n))
    odor = np.flatnonzero(np.isin(names, ["ORN_DM1", "ORN_VA2"]))
    drive = {}
    for s in (-1, 1):
        x = np.zeros(n)
        x[odor[side[odor] == s]] = 1
        total = np.zeros(n)
        for _ in range(HOPS):
            x = M @ x
            total += x
        drive[s] = total
    dn = {}
    for t in ("DNa01", "DNa02", "DNa03", "DNb05", "DNp09"):
        L = np.flatnonzero((names == t) & (side == -1))
        R = np.flatnonzero((names == t) & (side == 1))
        d = {s: drive[s][L].sum() - drive[s][R].sum() for s in (-1, 1)}
        mean = np.mean([abs(drive[s][L].sum()) + abs(drive[s][R].sum()) for s in (-1, 1)]) / 2
        dn[t] = round(float((d[-1] - d[1]) / mean), 3)
    pn = {}
    x = np.zeros(n)
    x[odor[side[odor] == -1]] = 1
    y = M @ x
    for t in ("DM1_lPN", "VA2_adPN"):
        L = np.flatnonzero((names == t) & (side == -1))
        R = np.flatnonzero((names == t) & (side == 1))
        pn[t] = round(float((y[L].sum() - y[R].sum()) / ((y[L].sum() + y[R].sum()) / 2)), 3)
    return dn, pn


for stage in stages:
    pol, card, _ = ev.load_artifact(run / stage / "final")
    g = pol.graph
    with torch.no_grad():
        gains = pol.core.gains().double().numpy()
    lp, names = g.label_pairs, g.type_names
    print(
        f"== {stage}: gain median {np.median(gains):.3f}, p5/p95 {np.round(np.percentile(gains, [5, 95]), 3)}"
    )
    rows = []
    for a, b in sorted({(int(x), int(y)) for x, y, c in lp if c > 0}):
        gi = gains[np.flatnonzero((lp[:, 0] == a) & (lp[:, 1] == b) & (lp[:, 2] == 1))[0]]
        gc = gains[np.flatnonzero((lp[:, 0] == a) & (lp[:, 1] == b) & (lp[:, 2] == 2))[0]]
        rows.append((str(names[b]) in ORN_PN, f"{names[a]}->{names[b]}", gi, gc, gi / gc))
    for pn_pair, pair, gi, gc, r in sorted(rows, key=lambda r: (not r[0], -abs(np.log(r[4])))):
        print(f"  {'*' if pn_pair else ' '} {pair:28s} ipsi {gi:.3f} contra {gc:.3f} ratio {r:.3f}")
    equal = np.full_like(gains, float(np.median(gains)))
    for label, gg in (("equal g", equal), ("learned g", gains)):
        u, pn = laterality(g, gg, False)
        s, _ = laterality(g, gg, True)
        print(f"  {label}: PN (left odor, 1 hop) {pn}\n    DN unsigned {u}\n    DN signed   {s}")
