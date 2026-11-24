import json
from dataclasses import replace

import pytest
from flytrack_core.contracts import SensorContract
from flytrack_world.contract import load_required_contract


def test_empty_path_is_refused_instead_of_using_defaults():
    with pytest.raises(ValueError, match="sensor_contract_json is required"):
        load_required_contract("")


def test_given_contract_is_loaded_not_the_default(tmp_path):
    changed = replace(SensorContract(), odor_hz=SensorContract().odor_hz * 2)
    path = tmp_path / "contract.json"
    path.write_text(json.dumps(changed.to_dict()), encoding="utf-8")

    loaded = load_required_contract(str(path))

    assert loaded.sha256() == changed.sha256()
    assert loaded.sha256() != SensorContract().sha256()


def test_missing_file_is_an_error(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_required_contract(str(tmp_path / "absent.json"))
