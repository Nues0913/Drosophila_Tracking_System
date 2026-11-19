"""Check the deployment sensor contract against the model artifact (#140, E9)."""

import json
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
    expected = SensorContract.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))
    if expected.sha256() != artifact_contract.sha256():
        raise ValueError("Deployment sensor contract differs from model artifact")
    return expected
