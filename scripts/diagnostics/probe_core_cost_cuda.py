"""Light CUDA probe of dense vs sparse central-core cost on the #55 graphs.

Run from the repository root with PYTHONPATH=python; argv[1] is the output JSON.
Median of five timed repeats after one warm-up. A local spot check, not a host
calibration: compare only rows from the same run.
"""

import json
import sys
import time
from pathlib import Path

import torch
import yaml
from flytrack_core.contracts import SensorContract
from flytrack_core.model import build_policy

CASES = [("olfactory_v2", 5000), ("olfactory_v2_nokc", 5000), ("olfactory_v2_nokc", 4000)]
# (batch, ticks, backward): a rollout-sized forward and an update-sized forward/backward.
SHAPES = [(4, 40, False), (64, 16, True)]


def timed(model, code, batch: int, ticks: int, backward: bool, device) -> float:
    state = model.central_initial_state(batch, device)
    start = time.perf_counter()
    with torch.set_grad_enabled(backward), model.adjacency_cache():
        total = 0
        for _ in range(ticks):
            action, state, _ = model.forward_code(code, state)
            total = total + action.square().mean()
        if backward:
            total.backward()
    torch.cuda.synchronize()
    model.zero_grad(set_to_none=True)
    return time.perf_counter() - start


def main():
    device = torch.device("cuda")
    common = yaml.safe_load(Path("configs/model/variants.yaml").read_text())["common"]
    contract = SensorContract.from_json("configs/sensor_contract.json")
    rows = []
    for graph, threshold in CASES:
        for variant in ("VO", "VO-late"):
            model = build_policy(
                variant,
                Path(f"data/subgraphs/{graph}/graph.npz"),
                contract,
                "artifacts/frontend/frontend.npz",
                **dict(common, dense_threshold=threshold),
            ).to(device)
            row = {"graph": graph, "variant": variant, "dense": model.core.dense}
            for batch, ticks, backward in SHAPES:
                obs = {
                    "vision": torch.zeros(
                        batch, 2, model.encoder.frontend.n_ommatidia, device=device
                    ),
                    "odor": torch.full((batch, 2), 0.02, device=device),
                    "valid": torch.ones(batch, 3, device=device),
                }
                with torch.no_grad():
                    code, _ = model.sense(obs, model.encoder.initial_state(batch, device))
                timed(model, code, batch, ticks, backward, device)
                torch.cuda.reset_peak_memory_stats()
                times = sorted(timed(model, code, batch, ticks, backward, device) for _ in range(5))
                key = f"b{batch}_t{ticks}_{'fwdbwd' if backward else 'fwd'}"
                row[f"{key}_median_ms"] = round(times[2] * 1000, 2)
                row[f"{key}_peak_MiB"] = round(torch.cuda.max_memory_allocated() / 2**20, 1)
            print(json.dumps(row), flush=True)
            rows.append(row)
            del model
            torch.cuda.empty_cache()
    result = {"device": torch.cuda.get_device_name(0), "torch": torch.__version__, "rows": rows}
    Path(sys.argv[1]).write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
