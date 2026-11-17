"""#116 G0f: the installed contact world option (colcon; the ML suite has the detailed checks)."""

import xml.etree.ElementTree as ET
from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from flytrack_world.contact_monitor import ContactMonitorNode
from flytrack_world.contact_world import contact_event, contact_links, write_contact_world


def test_installed_world_gets_one_contact_sensor_per_agent_collision(tmp_path):
    share = Path(get_package_share_directory("flytrack_bringup"))
    agent = share / "models/agent/model.sdf"
    output = write_contact_world(share / "worlds/tracking.sdf", agent, tmp_path / "w.sdf")
    world = ET.parse(output).getroot().find("world")
    sensors = world.findall("model[@name='agent']/link/sensor[@type='contact']")
    links = contact_links(ET.parse(agent).getroot().find("model"))
    assert len(sensors) == sum(len(collisions) for _, collisions in links)
    assert world.find("plugin[@filename='gz-sim-contact-system']") is not None
    assert "contact" not in (share / "worlds/tracking.sdf").read_text().lower()


def test_monitor_drops_floor_contacts():
    assert ContactMonitorNode.__name__ == "ContactMonitorNode"
    floor = [("agent::caster::ball", "floor::floor::floor", 0.0)]
    assert contact_event("caster", 1.0, floor) is None
    wall = [("agent::base_link::body", "east_wall::wall::wall", 0.0)]
    assert contact_event("base_link", 1.0, wall)["contacts"][0]["other"] == "east_wall::wall::wall"
