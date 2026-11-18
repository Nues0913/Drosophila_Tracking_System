"""Streaming eligibility rules; type-pair totals use the whole typed brain."""

from pathlib import Path

import numpy as np
import pandas as pd

from .tables import edge_batches


def in_sorted(values: np.ndarray, candidates: np.ndarray) -> np.ndarray:
    positions = np.searchsorted(candidates, values)
    mask = positions < len(candidates)
    mask[mask] &= candidates[positions[mask]] == values[mask]
    return mask


class EdgeRule:
    def __init__(self, config, nodes, path, columns, minimum=5):
        self.name = config.get("edge_rule", "single_edge")
        if self.name not in ("single_edge", "type_sum"):
            raise ValueError(f"Unknown edge_rule: {self.name}")
        self.minimum = (
            float(config.get("edge_min_synapses", 10)) if self.name == "type_sum" else minimum
        )
        self.type_minimum = float(config.get("type_min_synapses", 50))
        if any(not np.isfinite(v) or v <= 0 for v in (self.minimum, self.type_minimum)):
            raise ValueError("Edge and type synapse thresholds must be finite and positive")
        self.pairs = np.array([], dtype=np.int64)
        if self.name == "single_edge":
            return
        typed = nodes[nodes.type.ne("")].sort_values("id")
        self.ids = typed.id.to_numpy(np.int64)
        self.codes, names = pd.factorize(typed.type, sort=True)
        self.type_count = len(names)
        keys, weights = [], []
        for pre, post, weight in edge_batches(path, columns):
            keep = weight >= self.minimum
            pre, post, weight = pre[keep], post[keep], weight[keep]
            keep = in_sorted(pre, self.ids) & in_sorted(post, self.ids)
            pre, post, weight = pre[keep], post[keep], weight[keep]
            keys.append(self.pair_codes(pre, post))
            weights.append(weight)
        if keys:
            unique, inverse = np.unique(np.concatenate(keys), return_inverse=True)
            totals = np.bincount(inverse, weights=np.concatenate(weights))
            self.pairs = unique[totals >= self.type_minimum]

    def pair_codes(self, pre, post):
        return (
            self.codes[np.searchsorted(self.ids, pre)] * self.type_count
            + self.codes[np.searchsorted(self.ids, post)]
        )

    def mask(self, pre, post, weight):
        keep = weight >= self.minimum
        if self.name == "single_edge":
            return keep
        positions = np.flatnonzero(keep)
        typed = in_sorted(pre[positions], self.ids) & in_sorted(post[positions], self.ids)
        keep[positions[~typed]] = False
        positions = positions[typed]
        keep[positions] = in_sorted(self.pair_codes(pre[positions], post[positions]), self.pairs)
        return keep


def collect_edges(path: Path, columns: dict, allowed: set[int], rule: EdgeRule) -> pd.DataFrame:
    ids = np.array(sorted(allowed), dtype=np.int64)
    parts = []
    for pre, post, weight in edge_batches(path, columns):
        keep = rule.mask(pre, post, weight)
        pre, post, weight = pre[keep], post[keep], weight[keep]
        keep = in_sorted(pre, ids) & in_sorted(post, ids)
        if keep.any():
            parts.append(
                pd.DataFrame({"pre": pre[keep], "post": post[keep], "synapse_count": weight[keep]})
            )
    if not parts:
        return pd.DataFrame(columns=["pre", "post", "synapse_count"])
    return (
        pd.concat(parts, ignore_index=True)
        .groupby(["pre", "post"], sort=True, as_index=False)
        .synapse_count.sum()
    )


def brain_inputs(path: Path, columns: dict, ids) -> dict[int, float]:
    """Sum every raw weight, including untyped / out-of-scope / subthreshold inputs."""
    ids = np.array(sorted(ids), dtype=np.int64)
    totals = np.zeros(len(ids), dtype=np.float64)
    for _, post, weight in edge_batches(path, columns):
        keep = in_sorted(post, ids)
        totals += np.bincount(
            np.searchsorted(ids, post[keep]), weights=weight[keep], minlength=len(ids)
        )
    return dict(zip(ids.tolist(), totals.tolist(), strict=True))
