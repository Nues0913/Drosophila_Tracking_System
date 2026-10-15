"""Audit candidate seed/DN types and thresholds against raw MaleCNS tables.

Run from the repository root; requires data/raw.
"""

import json
import time
from pathlib import Path

import numpy as np
import yaml
from connectome_extract.download import FILES
from connectome_extract.graph import assign_signs, in_sorted, select_types
from connectome_extract.tables import edge_batches, load_annotations, load_neurotransmitters

start = time.monotonic()
config = yaml.safe_load(Path("configs/extraction/main.yaml").read_text())
raw = Path(config["raw_dir"])
n = assign_signs(
    load_annotations(raw).merge(
        load_neurotransmitters(raw), on="id", how="left", validate="one_to_one"
    ),
    0.5,
)
s = config["selection"]
visual = select_types(n, s["visual_seed_patterns"], "visual")
odor = select_types(n, s["odor_seed_patterns"], "odor") & set(n[n.side != 0].id)
motors = select_types(n, s["turn_dn_patterns"], "turn") | select_types(
    n, s["forward_dn_patterns"], "forward"
)
seeds = visual | odor
allowed = set(
    n[n.superclass.isin(config["allowed_superclasses"]) & n.sign.notna() & n.type.ne("")].id
)
allowed_array = np.array(sorted(allowed), dtype=np.int64)
parts = []
for pre, post, w in edge_batches(raw / FILES["weights"]):
    mask = w >= 20
    pre, post, w = pre[mask], post[mask], w[mask]
    mask = in_sorted(pre, allowed_array) & in_sorted(post, allowed_array)
    if mask.any():
        parts.append(np.column_stack((pre[mask], post[mask], w[mask].astype(np.int64))))
all_edges = np.concatenate(parts)
del parts
print("central known-sign/type edge cache", all_edges.shape, flush=True)


def bfs(pre, post, seeds, h):
    d = {v: 0 for v in seeds}
    frontier = set(seeds)
    for hop in range(1, h + 1):
        if not frontier:
            break
        dest = set(post[in_sorted(pre, np.array(sorted(frontier), dtype=np.int64))].tolist())
        frontier = dest - d.keys()
        d.update(dict.fromkeys(frontier, hop))
    return d


rows = []
seed_types = {t: set(g.id) & seeds for t, g in n[n.id.isin(seeds)].groupby("type")}
for threshold in [20, 30, 40, 50, 55, 60, 65, 70, 75, 80, 90, 100]:
    e = all_edges[all_edges[:, 2] >= threshold]
    pre, post = e[:, 0], e[:, 1]
    short = bfs(post, pre, motors, 8)
    distances = {
        t: min((short[b] for b in bodies if b in short), default=None)
        for t, bodies in seed_types.items()
    }
    for uh in [4, 5]:
        v = bfs(pre, post, visual, 3)
        o = bfs(pre, post, odor, 3)
        back = bfs(post, pre, motors, uh)
        keep = (v.keys() | o.keys()) & back.keys()
        for _ in range(10):
            previous = keep
            ids = np.array(sorted(keep), dtype=np.int64)
            mask = in_sorted(pre, ids) & in_sorted(post, ids)
            p, q = pre[mask], post[mask]
            v = bfs(p, q, visual & keep, 3)
            o = bfs(p, q, odor & keep, 3)
            back = bfs(q, p, motors & keep, uh)
            keep = (v.keys() | o.keys()) & back.keys()
            if keep == previous:
                break
        row = {
            "threshold": threshold,
            "downstream_hops": 3,
            "upstream_hops": uh,
            "neurons": len(keep),
            "edges": int(len(p)),
            "missing_seed_types": [t for t, b in seed_types.items() if not b & keep],
            "seed_minimum_motor_distance": distances,
            "retained_seed_counts": {t: len(b & keep) for t, b in seed_types.items()},
            "convergence": len(v.keys() & o.keys()),
        }
        rows.append(row)
        print(json.dumps(row), flush=True)
Path("artifacts/data_execution/candidate_scan.json").write_text(
    json.dumps(
        {
            "edge_cache_rows": len(all_edges),
            "allowed_superclasses": config["allowed_superclasses"],
            "results": rows,
            "elapsed_s": time.monotonic() - start,
        },
        indent=2,
        sort_keys=True,
    )
    + "\n"
)
