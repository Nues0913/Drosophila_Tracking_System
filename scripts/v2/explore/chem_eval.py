"""#138: the registered evaluation (evaluate_v2.py, unchanged) with every v2 actor wrapped in
ChemController.

PYTHONPATH=<worktree>/python python chem_eval.py [--hold-only] <evaluate_v2.py arguments>

Same worlds, scenarios, episode seeds, ablations and records as evaluate_v2.py; only the actor
differs, so the output pairs episode by episode with the plain run of the same artifact and
commit. --hold-only never uses odor or wind (the control for the memory part of the rule).
The manifest gains the controller's parameters and this directory's script hashes.
"""

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import flytrack_v2

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from chem_controller import PARAMS, ChemController  # noqa: E402
from flytrack_v2.contracts import SensorContractV2  # noqa: E402
from flytrack_v2.eval.runner import V2Actor  # noqa: E402

ROOT = Path(flytrack_v2.__file__).resolve().parents[2]
hold_only = "--hold-only" in sys.argv
sys.argv = [a for a in sys.argv if a != "--hold-only"]
spec = importlib.util.spec_from_file_location("evaluate_v2", ROOT / "scripts/v2/evaluate_v2.py")
ev = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ev)
contract = SensorContractV2.from_json(ROOT / "configs/v2/sensor_contract.json")


def wrapped(policy):
    return ChemController(V2Actor(policy), contract, use_chem=not hold_only)


ev.V2Actor = wrapped
ev.main()
out = Path(sys.argv[sys.argv.index("--output") + 1])
manifest = json.loads((out / "manifest.json").read_text())
manifest["chem_controller"] = {
    "params": PARAMS,
    "hold_only": hold_only,
    "sha256": {
        name: hashlib.sha256((HERE / name).read_bytes()).hexdigest()
        for name in ("chem_controller.py", "chem_eval.py")
    },
}
(out / "manifest.json").write_text(json.dumps(manifest, indent=1) + "\n")
