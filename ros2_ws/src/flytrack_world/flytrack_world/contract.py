"""Load the deployment sensor contract that every world node must be given (#140, E9)."""

import json
from pathlib import Path

from flytrack_core.contracts import SensorContract


def load_required_contract(path: str) -> SensorContract:
    """Read the contract JSON at ``path``; an empty path is an error, not a silent default.

    Before #140 an empty ``sensor_contract_json`` made the world nodes fall back to
    ``SensorContract()``, so a launch that forgot the parameter ran on defaults that could differ
    from the policy's artifact without any warning.
    """
    if not str(path):
        raise ValueError(
            "sensor_contract_json is required: pass the deployment sensor contract JSON path"
        )
    return SensorContract.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))
