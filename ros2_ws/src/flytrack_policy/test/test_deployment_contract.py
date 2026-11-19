import json
from dataclasses import replace

import pytest
from flytrack_core.contracts import SensorContract
from flytrack_policy.contract import check_deployment_contract


def write(tmp_path, contract):
    path = tmp_path / "contract.json"
    path.write_text(json.dumps(contract.to_dict()), encoding="utf-8")
    return str(path)


def test_empty_path_is_refused_instead_of_skipping_the_check():
    with pytest.raises(ValueError, match="sensor_contract_json is required"):
        check_deployment_contract("", SensorContract())


def test_matching_contract_is_accepted(tmp_path):
    artifact = SensorContract()
    assert check_deployment_contract(write(tmp_path, artifact), artifact).sha256() == (
        artifact.sha256()
    )


def test_different_contract_is_refused(tmp_path):
    artifact = SensorContract()
    deployed = replace(artifact, odor_hz=artifact.odor_hz * 2)
    with pytest.raises(ValueError, match="differs from model artifact"):
        check_deployment_contract(write(tmp_path, deployed), artifact)
