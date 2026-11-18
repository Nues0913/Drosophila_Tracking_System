"""Record the demo camera images and the scenario events to files (#123 V2, demo_camera only).

The demo cameras are not used by the policy or the evaluator. This node writes, under
``output_dir``:
- one ``{name}.frames`` file per image topic: a sequence of records
  ``<stamp_ns int64><width uint32><height uint32><length uint32><zlib(RGB8 rows)>``
  (little-endian; the stamp is the image header stamp, simulation clock);
- ``scenario_events.jsonl``: every /world/scenario event (header stamp, phase, occluded,
  episode start, episode).
Subscriptions are best effort (sensor data), so the recorder never applies back-pressure on the
bridge. ``read_frames`` reads a ``.frames`` file back.
"""

from __future__ import annotations

import json
import struct
import zlib
from pathlib import Path

from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data

from flytrack_world.common import EVENT_QOS, spin_main

HEADER = struct.Struct("<qIII")


def file_name(topic: str) -> str:
    return topic.strip("/").replace("/", "_") + ".frames"


def encode_frame(stamp_ns: int, width: int, height: int, rgb: bytes, level: int = 1) -> bytes:
    if len(rgb) != width * height * 3:
        raise ValueError("expected tightly packed RGB8 rows")
    payload = zlib.compress(rgb, level)
    return HEADER.pack(stamp_ns, width, height, len(payload)) + payload


def read_frames(path: Path):
    """Yield (stamp_ns, width, height, rgb bytes) from a ``.frames`` file."""
    with open(path, "rb") as handle:
        while True:
            head = handle.read(HEADER.size)
            if not head:
                return
            if len(head) != HEADER.size:
                raise ValueError(f"{path}: truncated record header")
            stamp, width, height, length = HEADER.unpack(head)
            payload = handle.read(length)
            if len(payload) != length:
                raise ValueError(f"{path}: truncated record")
            yield stamp, width, height, zlib.decompress(payload)


def packed_rgb(message) -> bytes:
    if message.encoding not in ("rgb8", "8UC3"):
        raise ValueError(f"unsupported encoding {message.encoding}")
    row = message.width * 3
    data = bytes(message.data)
    if message.step == row:
        return data
    return b"".join(data[r * message.step : r * message.step + row] for r in range(message.height))


class DemoRecorderNode(Node):
    def __init__(self):
        super().__init__("demo_recorder")
        from flytrack_interfaces.msg import ScenarioEvent
        from sensor_msgs.msg import Image

        self.declare_parameter(
            "topics", ["/demo/camera/follow/image", "/demo/camera/overview/image"]
        )
        self.declare_parameter("output_dir", "/workspace/runs/demo_frames")
        out = Path(self.get_parameter("output_dir").value)
        out.mkdir(parents=True, exist_ok=True)
        self.counts: dict[str, int] = {}
        self.files = {}
        for topic in self.get_parameter("topics").value:
            self.files[topic] = open(out / file_name(topic), "ab")
            self.counts[topic] = 0
            self.create_subscription(
                Image,
                topic,
                lambda message, topic=topic: self._image(topic, message),
                qos_profile_sensor_data,
            )
        self.events = open(out / "scenario_events.jsonl", "a", encoding="utf-8")
        self.create_subscription(ScenarioEvent, "/world/scenario", self._event, EVENT_QOS)

    def _image(self, topic, message):
        stamp = message.header.stamp.sec * 1_000_000_000 + message.header.stamp.nanosec
        record = encode_frame(stamp, message.width, message.height, packed_rgb(message))
        self.files[topic].write(record)
        self.files[topic].flush()
        self.counts[topic] += 1

    def _event(self, message):
        stamp = message.header.stamp.sec + message.header.stamp.nanosec * 1e-9
        try:
            values = json.loads(message.parameters_json)
        except ValueError:
            values = {}
        row = {
            "stamp": stamp,
            "episode_id": int(message.episode_id),
            "scenario": message.scenario,
            "seed": int(message.seed),
            "active": bool(message.active),
            "phase": values.get("phase"),
            "occluded": values.get("occluded"),
            "episode_start_s": values.get("episode_start_s"),
        }
        self.events.write(json.dumps(row, sort_keys=True) + "\n")
        self.events.flush()

    def destroy_node(self):
        for handle in self.files.values():
            handle.close()
        self.events.close()
        self.get_logger().info(f"demo frames recorded: {self.counts}")
        return super().destroy_node()


def main(args=None):
    spin_main(DemoRecorderNode, args)
