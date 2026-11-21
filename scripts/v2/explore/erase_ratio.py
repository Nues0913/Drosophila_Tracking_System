# ruff: noqa: E501, B023  (#138 exploratory analysis output lines)
"""#138 7057874102 check 1: learned ipsi/contra ratio vs the ratio that erases 1-hop laterality.

cd <A worktree> && OMP_NUM_THREADS=2 PYTHONPATH=python python erase_ratio.py RUN_DIR [RUN_DIR ...]

Erasing ratio of a split pair a -> b, per odor side s: with odor on side s only, the b cell on
side s gets g_ipsi * sum(signed_fraction of ipsi edges) and the other side's b gets
g_contra * sum(contra fractions); the 1-hop left-right difference at b is 0 when
g_ipsi / g_contra = contra sum / ipsi sum (unsigned, as MiLong's table). Reported per side and
their geometric mean; the correlation of log learned ratio with log erasing ratio over the
pairs where both sides are finite (strong, slope near 1: active erasing; flat: common drift).
"""

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, "scripts/v2")
import evaluate_v2 as ev  # noqa: E402

ORN_PN = {"DM1_lPN", "DM2_lPN", "M_l2PNl20", "VA2_adPN", "VA7m_lPN"}
for run in map(Path, sys.argv[1:]):
    for stage in ("stage1", "stage2"):
        final = run / stage / "final"
        if not (final / "SHA256SUMS").is_file():
            continue
        pol, _, _ = ev.load_artifact(final)
        g = pol.graph
        with torch.no_grad():
            gains = pol.core.gains().double().numpy()
        lp, names, side = g.label_pairs, g.type_names, g.side
        frac = g.synapse_count / g.fan_in[g.post]
        rows = []
        for a, b in sorted({(int(x), int(y)) for x, y, c in lp if c > 0}):
            li = np.flatnonzero((lp[:, 0] == a) & (lp[:, 1] == b) & (lp[:, 2] == 1))[0]
            lc = np.flatnonzero((lp[:, 0] == a) & (lp[:, 1] == b) & (lp[:, 2] == 2))[0]
            erase = []
            for s in (-1, 1):
                pre_s = side[g.pre] == s
                ipsi = (g.edge_label == li) & pre_s & (side[g.post] == s)
                contra = (g.edge_label == lc) & pre_s & (side[g.post] == -s)
                i, c = frac[ipsi].sum(), frac[contra].sum()
                erase.append(c / i if i > 0 and c > 0 else np.nan)
            rows.append(
                (f"{names[a]}->{names[b]}", str(names[b]) in ORN_PN, gains[li] / gains[lc], *erase)
            )
        learned = np.array([r[2] for r in rows])
        er = np.array([np.sqrt(r[3] * r[4]) for r in rows])
        ok = np.isfinite(er)
        x, y = np.log(er[ok]), np.log(learned[ok])
        slope = np.polyfit(x, y, 1)[0]
        print(
            f"== {run.parent.name}/{run.name} {stage}: {ok.sum()}/{len(rows)} pairs finite on both sides; "
            f"corr(log learned, log erase) {np.corrcoef(x, y)[0, 1]:+.2f}, slope {slope:+.3f}; "
            f"learned ratio median {np.median(learned):.2f}, IQR {np.percentile(learned, 25):.2f}-{np.percentile(learned, 75):.2f}"
        )
        for name, pn, r, e1, e2 in sorted(rows, key=lambda r: (not r[1], r[0])):
            print(
                f"  {'*' if pn else ' '} {name:26s} learned {r:.3f}  erase L {e1:7.3f} R {e2:7.3f}"
            )
