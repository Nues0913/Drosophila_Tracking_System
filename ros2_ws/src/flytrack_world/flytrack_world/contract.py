"""Load the deployment sensor contract that every world node must be given (#140, E9)."""

import json
from pathlib import Path

from flytrack_core.contracts import SensorContract

# The fields SensorContractV2 adds to v1's (docs/v2/interfaces.md section 1).
V2_WIND_FIELDS = ("wind_hz", "antenna_angle_deg", "wind_reference_m_s")


def load_required_contract(path: str) -> SensorContract:
    """Read the contract JSON at ``path``; an empty path is an error, not a silent default.

    Before #140 an empty ``sensor_contract_json`` made the world nodes fall back to
    ``SensorContract()``, so a launch that forgot the parameter ran on defaults that could differ
    from the policy's artifact without any warning.

    A contract with any of the v2 wind fields is a ``SensorContractV2`` and must carry all of
    them: the dataclass defaults would otherwise fill a missing one in silently.
    """
    if not str(path):
        raise ValueError(
            "sensor_contract_json is required: pass the deployment sensor contract JSON path"
        )
    values = json.loads(Path(path).read_text(encoding="utf-8"))
    present = [name for name in V2_WIND_FIELDS if name in values]
    if not present:
        return SensorContract.from_dict(values)
    missing = sorted(set(V2_WIND_FIELDS) - set(present))
    if missing:
        raise ValueError(f"v2 sensor contract is missing wind fields: {missing}")
    from flytrack_v2.contracts import SensorContractV2  # v2 deployments only

    return SensorContractV2.from_dict(values)


def has_wind(contract) -> bool:
    """True for a v2 contract, whose world publishes the antenna reading."""
    return all(hasattr(contract, name) for name in V2_WIND_FIELDS)
