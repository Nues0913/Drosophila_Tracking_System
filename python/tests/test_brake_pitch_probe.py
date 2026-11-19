import importlib.util
import json
import math
from pathlib import Path

import pytest

SPEC = importlib.util.spec_from_file_location(
    "brake_pitch_probe",
    Path(__file__).resolve().parents[2] / "scripts/diagnostics/brake_pitch_probe.py",
)
probe = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(probe)


def test_pitch_is_positive_when_the_nose_goes_down():
    angle = math.radians(35)
    # rotation about +y by +angle turns +x towards -z: nose down for a forward-x body
    pitch, roll = probe.pitch_roll((math.cos(angle / 2), 0.0, math.sin(angle / 2), 0.0))
    assert pitch == pytest.approx(angle)
    assert roll == pytest.approx(0.0)


def test_phase_cycle():
    assert probe.phase_at(0.0) == "drive"
    assert probe.phase_at(probe.DRIVE_S + 0.01) == "brake"
    assert probe.phase_at(probe.DRIVE_S + probe.BRAKE_S + 0.01) == "drive"


def test_summarize_takes_the_largest_magnitude_per_phase(tmp_path):
    rows = [
        {"episode_id": 0, "phase": "drive", "pitch": -0.01, "roll": 0.0},
        {"episode_id": 0, "phase": "brake", "pitch": 0.5, "roll": -0.02},
        {"episode_id": 1, "phase": "brake", "pitch": 0.1, "roll": 0.0},
    ]
    path = tmp_path / "pitch.jsonl"
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    summary = probe.summarize(path)
    assert summary["samples"] == 3
    assert summary["overall"]["max_abs_pitch_deg_brake"] == pytest.approx(math.degrees(0.5))
    assert summary["episodes"]["1"]["max_abs_pitch_deg_brake"] == pytest.approx(math.degrees(0.1))
