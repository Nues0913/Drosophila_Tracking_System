"""Check the deployment sensor contract against the model artifact (#140, E9)."""

import json
from dataclasses import fields
from pathlib import Path

from flytrack_core.contracts import SensorContract


def check_deployment_contract(path: str, artifact_contract: SensorContract) -> SensorContract:
    """Require the deployment contract at ``path`` and refuse it unless it equals the artifact's.

    Before #140 an empty ``sensor_contract_json`` skipped the comparison, so the policy ran
    without confirming that the world publishes what the model was trained on.
    """
    if not str(path):
        raise ValueError(
            "sensor_contract_json is required: pass the deployment sensor contract JSON path"
        )
    # Parse with the artifact's contract class, and require every one of its fields: a v2
    # deployment contract is refused for a v1 artifact (unknown wind fields), and a v1 one for a
    # v2 artifact (missing wind fields; the dataclass defaults would otherwise fill them in and
    # reproduce the v2 hash).
    values = json.loads(Path(path).read_text(encoding="utf-8"))
    missing = {field.name for field in fields(artifact_contract)} - set(values)
    if missing:
        raise ValueError(
            f"Deployment sensor contract differs from model artifact: missing {sorted(missing)}"
        )
    try:
        expected = type(artifact_contract).from_dict(values)
    except (TypeError, ValueError) as error:
        raise ValueError(
            f"Deployment sensor contract differs from model artifact: {error}"
        ) from error
    if expected.sha256() != artifact_contract.sha256():
        raise ValueError("Deployment sensor contract differs from model artifact")
    return expected
