"""#116 G0f: Gazebo contact check (default-off world option, scripted collisions, analysis)."""

import copy
import importlib.util
import math
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]
BRINGUP = ROOT / "ros2_ws/src/flytrack_bringup"


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, ROOT / path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


world_mod = load("contact_world_test", "ros2_ws/src/flytrack_world/flytrack_world/contact_world.py")
driver = load("contact_probe_driver_test", "scripts/diagnostics/contact_probe_driver.py")
analysis = load("contact_check_analyze_test", "scripts/diagnostics/contact_check_analyze.py")
WORLD = (BRINGUP / "worlds/tracking.sdf").read_text(encoding="utf-8")
AGENT = (BRINGUP / "models/agent/model.sdf").read_text(encoding="utf-8")


def strip_contact(model):
    model = copy.deepcopy(model)
    for link in model.findall("link"):
        for sensor in link.findall("sensor"):
            if sensor.get("type") == "contact":
                link.remove(sensor)
    return model


def canonical(element):
    return ET.tostring(element, encoding="unicode")


def test_formal_world_and_agent_have_no_contact_sensing():
    assert "contact" not in WORLD.lower()
    assert 'type="contact"' not in AGENT


def test_contact_world_only_inlines_the_agent_and_adds_sensors_and_the_system():
    derived = ET.fromstring(world_mod.contact_world(WORLD, AGENT)).find("world")
    formal = ET.fromstring(WORLD).find("world")
    agent = ET.fromstring(AGENT).find("model")
    models = [m for m in derived.findall("model") if m.get("name") == "agent"]
    assert len(models) == 1
    # The inlined agent is the formal model plus contact sensors, nothing else.
    assert canonical(strip_contact(models[0])) == canonical(agent)
    links = {name: collisions for name, collisions in world_mod.contact_links(agent)}
    assert set(links) == {"base_link", "left_wheel", "right_wheel", "caster", "front_caster"}
    for link in models[0].findall("link"):
        sensors = [s for s in link.findall("sensor") if s.get("type") == "contact"]
        assert [s.findtext("contact/collision") for s in sensors] == links[link.get("name")]
        assert {s.findtext("topic") for s in sensors} == {
            f"/model/agent/contact/{link.get('name')}"
        }
    # Every other world element is unchanged and in the same order.
    plugins = [p for p in derived.findall("plugin") if p.get("filename") == "gz-sim-contact-system"]
    assert len(plugins) == 1
    rest = [canonical(e) for e in derived if e not in models and e not in plugins]
    formal_rest = [
        canonical(e)
        for e in formal
        if not (e.tag == "include" and e.findtext("uri") == "model://agent")
    ]
    assert rest == formal_rest
    order = [e.tag for e in derived]
    assert order.index("plugin") < order.index("model")


def test_contact_world_rejects_a_world_without_the_agent_include():
    with pytest.raises(ValueError):
        world_mod.contact_world(WORLD.replace("model://agent", "model://other"), AGENT)


def test_contact_events_drop_the_floor_and_keep_walls_and_target():
    names = [
        ("agent::left_wheel::wheel", "floor::floor::floor", 1e-8),
        ("north_wall::wall::wall", "agent::base_link::body", 2e-4),
    ]
    event = world_mod.contact_event("base_link", 3.5, names)
    assert event == {
        "stamp": 3.5,
        "link": "base_link",
        "contacts": [
            {"self": "agent::base_link::body", "other": "north_wall::wall::wall", "depth": 2e-4}
        ],
    }
    assert world_mod.contact_event("left_wheel", 1.0, names[:1]) is None


def test_world_launch_and_compose_keep_contact_check_off_by_default():
    launch = (BRINGUP / "launch/world.launch.py").read_text(encoding="utf-8")
    assert '"contact_check", default_value="false"' in launch
    compose = yaml.safe_load((ROOT / "compose.yaml").read_text(encoding="utf-8"))
    command = compose["services"]["sim"]["command"]
    assert "contact_check:=${CONTACT_CHECK:-false}" in command
    assert "episode_seconds:=${EPISODE_SECONDS:-60}" in command
    assert "/sim/contact_events" not in (
        ROOT / "ros2_ws/src/flytrack_eval/flytrack_eval/bag_recorder_node.py"
    ).read_text(encoding="utf-8")


@pytest.mark.parametrize("wall", ["north", "east", "south", "west"])
@pytest.mark.parametrize("angle", [0, 15, 30, 45, 60, 70, 80])
@pytest.mark.parametrize("side", [1, -1])
def test_wall_approach_meets_the_chosen_wall_at_the_planned_incidence(wall, angle, side):
    manoeuvre = {"kind": "wall", "wall": wall, "incidence_deg": angle, "side": side}
    x, y, hx, hy, heading = driver.wall_approach(manoeuvre)
    limit = 5.0 - 0.18
    assert abs(x) < limit and abs(y) < limit and max(abs(hx), abs(hy)) == pytest.approx(limit)
    length = max(driver.LEAD_M, driver.LEAD_DEPTH_M / math.cos(math.radians(angle)))
    assert math.hypot(hx - x, hy - y) == pytest.approx(length)
    assert driver.first_wall(x, y, heading, 5.0, 0.18) == wall
    nx, ny = driver.NORMAL_INTO_WALL[wall]
    cos_incidence = math.cos(heading) * nx + math.sin(heading) * ny
    assert math.degrees(math.acos(min(1.0, cos_incidence))) == pytest.approx(angle, abs=1e-6)


@pytest.mark.parametrize("manoeuvre", [m for m in driver.default_plan() if m["kind"] == "wall"])
def test_wall_step_drives_from_a_spawn_to_the_planned_wall(manoeuvre):
    """Unicycle integration of wall_step at 20 Hz (no physics): constant speed, lead-in, hit."""
    pose = [1.0, -0.5, 2.0]
    state, dt = {}, 0.05
    for _ in range(4000):
        v, omega = driver.wall_step(state, pose, manoeuvre)
        assert v == driver.SPEED_M_S  # never brakes
        pose = [
            pose[0] + v * math.cos(pose[2]) * dt,
            pose[1] + v * math.sin(pose[2]) * dt,
            driver.wrap(pose[2] + omega * dt),
        ]
        if max(abs(pose[0]), abs(pose[1])) > 5.0 - 0.18:
            break
    else:
        pytest.fail("never reached a wall")
    nx, ny = driver.NORMAL_INTO_WALL[manoeuvre["wall"]]
    assert pose[0] * nx + pose[1] * ny > 5.0 - 0.18
    *_, heading = driver.wall_approach(manoeuvre)
    assert abs(driver.wrap(pose[2] - heading)) < math.radians(3.0)


def test_default_plan_covers_all_walls_angles_and_target_sides():
    plan = driver.default_plan()
    assert len(plan) >= 30
    walls = [m for m in plan if m["kind"] == "wall"]
    assert {m["wall"] for m in walls} == set(driver.WALLS)
    assert {m["incidence_deg"] for m in walls} >= {0, 45, 60, 70, 80}
    offsets = [m["offset_m"] for m in plan if m["kind"] == "target"]
    assert 0.0 in offsets and 0.25 in offsets and -0.25 in offsets


def test_target_aim_offsets_perpendicular_to_the_line_of_sight():
    agent, target = (0.0, 0.0, 0.0), (2.0, 0.0, 0.0)
    assert driver.target_aim(agent, target, 0.0) == pytest.approx(0.0)
    assert driver.target_aim(agent, target, 0.25) == pytest.approx(math.atan2(0.25, 2.0))
    assert driver.target_aim(agent, target, -0.25) == pytest.approx(-math.atan2(0.25, 2.0))


def scenario_event(stamp, episode, active, phase, collision, start=10.0):
    return {
        "stamp": stamp,
        "episode_id": episode,
        "seed": 100 + episode,
        "scenario": "normal",
        "active": active,
        "parameters": {
            "phase": phase,
            "collision": collision,
            "episode_start_s": start,
            "arena_size_m": 10.0,
            "arena_center_x": 0.0,
            "arena_center_y": 0.0,
            "agent_radius_m": 0.18,
            "target_radius_m": 0.15,
            "obstacles": [],
        },
    }


def contact(stamp, other):
    return {
        "stamp": stamp,
        "link": "base_link",
        "contacts": [{"self": "agent::base_link::body", "other": other, "depth": 0.0}],
    }


def test_analysis_counts_contacts_before_the_flag_and_times_them():
    events = [
        scenario_event(9.0, 0, False, "resetting", False),
        scenario_event(10.0, 0, True, "tracking", False, start=10.0),
        scenario_event(12.05, 0, False, "completed", True, start=10.0),
        scenario_event(20.0, 1, True, "tracking", False, start=20.0),
        scenario_event(80.0, 1, False, "completed", False, start=20.0),
    ]
    agent = [[10.0 + 0.02 * i, 4.70 + 0.006 * i, 0.0, 0.0] for i in range(110)]
    target = [[10.0, -2.0, 0.0, 0.0]]
    export = {
        "events": events,
        "agent": agent,
        "target": target,
        "contacts": [
            contact(12.03, "east_wall::wall::wall"),  # before the tick that set the flag
            contact(12.20, "east_wall::wall::wall"),  # after it
            contact(9.5, "east_wall::wall::wall"),  # before the episode: ignored
        ],
    }
    result = analysis.analyse(export)
    first, second = result["episodes"]
    assert first["flag_s"] == 12.05 and first["contacts_before_flag"] == 1
    assert first["contacts_before_flag_objects"] == ["east_wall"]
    assert first["contact_minus_flag_s"] == pytest.approx(-0.02)
    # x = 4.70 + 0.006 i first exceeds 4.82 at i = 21 (t = 10.42).
    assert first["geometric_s"] == pytest.approx(10.42)
    assert second["flag_s"] is None and second["contacts_before_flag"] == 0
    summary = result["summary"]
    assert summary["geometric_flags"] == 1 and summary["flags_with_contact"] == 1
    assert summary["episodes_with_contact_before_flag"] == 1
    assert summary["objects_before_flag"] == ["east_wall"]
    assert summary["unflagged_with_contact"] == 0
