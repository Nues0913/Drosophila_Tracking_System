"""Gazebo world with contact sensors on the agent (#116 G0f, off by default).

The formal world (``tracking.sdf`` including ``model://agent``) is never modified. With
``contact_check:=true`` the world launch writes a derived copy: the agent model is inlined with
one contact sensor per collision-bearing link, and the world loads ``gz-sim-contact-system``.
Nothing else changes (same poses, collisions, cameras, plugins), so the physics is the formal
world's; ``python/tests/test_gazebo_contact_world.py`` checks that.
"""

from __future__ import annotations

import copy
import xml.etree.ElementTree as ET
from pathlib import Path

CONTACT_SYSTEM = ("gz-sim-contact-system", "gz::sim::systems::Contact")
GZ_TOPIC_PREFIX = "/model/agent/contact"
ROS_TOPIC_PREFIX = "/sim/contact"
EVENTS_TOPIC = "/sim/contact_events"
FLOOR_MARKERS = ("floor",)


def contact_links(agent_model: ET.Element) -> list[tuple[str, list[str]]]:
    """(link name, collision names) for every agent link that has a collision."""
    return [
        (link.get("name"), [c.get("name") for c in link.findall("collision")])
        for link in agent_model.findall("link")
        if link.findall("collision")
    ]


def contact_topic(link: str, prefix: str = GZ_TOPIC_PREFIX) -> str:
    return f"{prefix}/{link}"


def _sensor(link: str, collision: str, update_rate: float) -> ET.Element:
    sensor = ET.Element("sensor", name=f"{link}_{collision}_contact", type="contact")
    ET.SubElement(sensor, "always_on").text = "true"
    ET.SubElement(sensor, "update_rate").text = f"{update_rate:g}"
    ET.SubElement(sensor, "topic").text = contact_topic(link)
    contact = ET.SubElement(sensor, "contact")
    ET.SubElement(contact, "collision").text = collision
    ET.SubElement(contact, "topic").text = contact_topic(link)
    return sensor


def contact_world(world_sdf: str, agent_sdf: str, update_rate: float = 1000.0) -> str:
    """Return the formal world with the agent inlined and contact sensors on its links."""
    world_root = ET.fromstring(world_sdf)
    world = world_root.find("world")
    agent_model = copy.deepcopy(ET.fromstring(agent_sdf).find("model"))
    if world is None or agent_model is None or agent_model.get("name") != "agent":
        raise ValueError("expected a <world> and an <sdf><model name='agent'>")
    includes = [
        inc for inc in world.findall("include") if (inc.findtext("uri") or "") == "model://agent"
    ]
    if len(includes) != 1 or includes[0].find("pose") is not None:
        raise ValueError("the world must include model://agent exactly once, without a pose")
    name = includes[0].findtext("name") or agent_model.get("name")
    agent_model.set("name", name)
    for link in agent_model.findall("link"):
        for collision in link.findall("collision"):
            link.append(_sensor(link.get("name"), collision.get("name"), update_rate))
    children = list(world)
    index = children.index(includes[0])
    world.remove(includes[0])
    world.insert(index, agent_model)
    last_plugin = max(i for i, child in enumerate(list(world)) if child.tag == "plugin")
    filename, plugin_name = CONTACT_SYSTEM
    world.insert(last_plugin + 1, ET.Element("plugin", filename=filename, name=plugin_name))
    return '<?xml version="1.0"?>\n' + ET.tostring(world_root, encoding="unicode") + "\n"


def is_floor(name: str) -> bool:
    return any(marker in name.split("::")[0] for marker in FLOOR_MARKERS)


def contact_event(link: str, stamp_s: float, contacts) -> dict | None:
    """Non-floor contacts of one message; ``contacts`` are (name1, name2, max depth) triples."""
    kept = []
    for first, second, depth in contacts:
        mine, other = (first, second) if first.startswith("agent") else (second, first)
        if is_floor(other):
            continue
        kept.append({"self": mine, "other": other, "depth": float(depth)})
    if not kept:
        return None
    return {"stamp": stamp_s, "link": link, "contacts": kept}


def write_contact_world(world_path: Path, agent_path: Path, output: Path) -> Path:
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        contact_world(world_path.read_text(encoding="utf-8"), agent_path.read_text("utf-8")),
        encoding="utf-8",
    )
    return output
