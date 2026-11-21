# ruff: noqa: E501, B023  (#138 exploratory analysis output lines)
"""#138 A2 diagnosis: where the left/right odor signal fades between the ORNs and the turn.

cd <A worktree> && OMP_NUM_THREADS=3 PYTHONPATH=python python odor_trace.py STAGE_DIR [STEPS]

STAGE_DIR is an A artifact's stage directory (final/ and failure/batch.pt). On the wind-world
sequences whose first STEPS (16) steps after the burn-in are all occluded with no reset, the
policy runs twice from the burn-in state: the recorded codes, and the codes with the left and
right ORN columns swapped on every step (E1's odor swap, held). At the last step, per cell type:
mean |rate difference| between the two runs, the type's mean rate, and for paired types the
change of its left - right rate difference. Types are grouped by their hop distance from the
odor seed ORNs (shortest path along synapses); the chain named in #138 is listed on its own.
Then the turn: steer readout difference and the mode turn difference.
"""

import sys
from collections import deque
from pathlib import Path

import numpy as np
import torch
from flytrack_v2.model.export import load_policy
from flytrack_v2.train.update import burn_in_state

CHAIN = ["ORN_DM1", "ORN_VA2", "DM1_lPN", "VA2_adPN", "LHCENT8", "FB5AB", "hDeltaC", "FC2B",
         "PFL3", "DNa01", "DNa02", "DNa03", "DNb05", "DNp09"]  # fmt: skip
stage = Path(sys.argv[1])
steps = int(sys.argv[2]) if len(sys.argv) > 2 else 16
batch = torch.load(stage / "failure/batch.pt", weights_only=False, map_location="cpu")
policy = load_policy(stage / "final", device="cpu")
graph = policy.graph
names = graph.type_names[graph.type_index]
side = graph.side
code, reset, occluded, tags = (batch[k] for k in ("code", "reset", "occluded", "tags"))
code = code.float()
burn = 16
rows = torch.where(
    (tags[:, 0] == 1) & occluded[burn : burn + steps].all(0) & ~reset[burn : burn + steps].any(0)
)[0]
print(f"{stage}: {len(rows)} sequences, {steps} steps of held odor swap")

# hop distance from the odor seed ORNs
n = graph.n_nodes
adjacency = [[] for _ in range(n)]
for a, b in zip(graph.pre.tolist(), graph.post.tolist(), strict=True):
    adjacency[a].append(b)
hops = np.full(n, -1)
queue = deque(np.flatnonzero(np.isin(names, ["ORN_DM1", "ORN_VA2"])).tolist())
for start in queue:
    hops[start] = 0
while queue:
    node = queue.popleft()
    for nxt in adjacency[node]:
        if hops[nxt] < 0:
            hops[nxt] = hops[node] + 1
            queue.append(nxt)

initial = policy.initial_central_state(1, "cpu")
with torch.no_grad():
    state = burn_in_state(
        policy, code[:burn, rows], reset[:burn, rows], batch["initial"][rows].float(), initial
    )
    a = b = state
    for t in range(steps):
        c = code[burn + t, rows]
        swapped = c.clone()
        swapped[:, 288], swapped[:, 289] = c[:, 289], c[:, 288]
        step_a, step_b = policy.forward_code(c, a), policy.forward_code(swapped, b)
        a, b = step_a.central_state, step_b.central_state
    rate_a, rate_b = policy.core.rates(a).numpy(), policy.core.rates(b).numpy()
    turn = (step_a.dist.mode()[:, 1] - step_b.dist.mode()[:, 1]).abs().mean().item()
    latent = (step_a.dist.mean[:, 1] - step_b.dist.mean[:, 1]).abs().mean().item()
    print(f"mode turn |diff| {turn:.5f} rad/s, latent {latent:.5f}")
    # The steer readout by DN type: (normalized rates @ steer_means)[:, k] * steer_weight[k]
    # (#138 review: DNb05 carries most of A's residual odor turn).
    readout = policy.readout
    parts = [
        (readout.normalize_rates(step.aux["dn_rates"]) @ readout.steer_means) * readout.steer_weight
        for step in (step_a, step_b)
    ]
    share = (parts[0] - parts[1]).abs().mean(0)
    types = getattr(readout, "steer_types", None) or [f"type{k}" for k in range(len(share))]
    total = float(share.sum())
    print(
        "steer latent |diff| by DN type:",
        {t: f"{float(v):.2e} ({float(v) / total:.0%})" for t, v in zip(types, share, strict=True)},
    )
    print(
        "ORN code |L-R| mean",
        float((code[burn + steps - 1, rows, 288] - code[burn + steps - 1, rows, 289]).abs().mean()),
    )

diff = np.abs(rate_a - rate_b).mean(0)
mean_rate = rate_a.mean(0)


def row(t):
    m = names == t
    left, right = m & (side == -1), m & (side == 1)
    lat = ""
    if left.any() and right.any():
        lr_a = rate_a[:, left].mean(1) - rate_a[:, right].mean(1)
        lr_b = rate_b[:, left].mean(1) - rate_b[:, right].mean(1)
        lat = (
            f"  L-R change {np.abs(lr_a - lr_b).mean():.2e} (L-R itself {np.abs(lr_a).mean():.2e})"
        )
    hop = int(hops[m].min()) if (hops[m] >= 0).any() else -1
    return f"{t:14s} hop {hop:2d} n {m.sum():4d}  |d rate| {diff[m].mean():.2e}  rate {mean_rate[m].mean():.2e}  rel {diff[m].mean() / max(mean_rate[m].mean(), 1e-12):.2e}{lat}"


print("-- chain")
for t in CHAIN:
    if (names == t).any():
        print(row(t))
print("-- by hop: the 4 types with the largest |d rate| at each distance")
for h in range(0, 9):
    types = np.unique(names[hops == h])
    if not len(types):
        continue
    ranked = sorted(types, key=lambda t: -diff[names == t].mean())
    level = diff[hops == h]
    print(
        f"hop {h}: {len(types)} types, {int((hops == h).sum())} cells, |d rate| mean {level.mean():.2e} max {level.max():.2e}"
    )
    for t in ranked[:4]:
        print("   " + row(t))
