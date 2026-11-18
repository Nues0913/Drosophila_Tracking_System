"""Generate the bodyId list for the v2 bilateral DN speed readout."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[2]
GRAPH = Path("data/subgraphs/v2_main/graph.npz")
OUTPUT = Path("configs/v2/model/readout_bilateral_dn.yaml")
MIN_COVERAGE = 0.40


def readout_spec() -> dict:
    graph_path = ROOT / GRAPH
    data = graph_path.read_bytes()
    dn_types = json.loads(graph_path.with_name("manifest.json").read_text())["dn_types"]
    with np.load(io.BytesIO(data), allow_pickle=False) as graph:
        names = graph["type_names"][graph["type_index"]]
        sides = graph["side"]
        ids = graph["node_ids"]
        brain = graph["brain_input_synapses"].astype(np.float64)
        subgraph = graph["subgraph_input_synapses"].astype(np.float64)
    coverage = np.divide(subgraph, brain, out=np.zeros_like(brain), where=brain > 0)
    rows = []
    for name in sorted(dn_types):
        matching = names == name
        left = np.flatnonzero(matching & (sides == -1))
        right = np.flatnonzero(matching & (sides == 1))
        if left.size != 1 or right.size != 1:
            continue
        left, right = left.item(), right.item()
        if min(coverage[left], coverage[right]) >= MIN_COVERAGE:
            rows.append(
                {
                    "type": name,
                    "left_body_id": int(ids[left]),
                    "right_body_id": int(ids[right]),
                    "left_coverage": round(float(coverage[left]), 6),
                    "right_coverage": round(float(coverage[right]), 6),
                }
            )
    return {
        "source": {
            "graph": GRAPH.as_posix(),
            "graph_sha256": hashlib.sha256(data).hexdigest(),
            "rule": (
                "exactly one left and one right cell, "
                f"both with input coverage >= {MIN_COVERAGE:.2f}"
            ),
        },
        "types": rows,
    }


def _flow(values: dict) -> str:
    return yaml.safe_dump(values, sort_keys=False, default_flow_style=True, width=240).strip()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="check the saved list without writing")
    args = parser.parse_args()
    spec = readout_spec()
    rows = spec["types"]
    lines = [f"source: {_flow(spec['source'])}", "types:", *(f"  - {_flow(row)}" for row in rows)]
    contents = "\n".join(lines) + "\n"
    output_path = ROOT / OUTPUT
    if args.check:
        if not output_path.is_file() or output_path.read_text() != contents:
            raise SystemExit(f"{OUTPUT} differs from the {len(rows)} generated types")
    else:
        output_path.write_text(contents)
    print(f"{len(rows)} bilateral DN types in {OUTPUT}")


if __name__ == "__main__":
    main()
