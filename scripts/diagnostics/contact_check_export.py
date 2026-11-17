"""Export a contact-check rosbag to JSON (#116 G0f).

Usage (ROS image):
  python3 scripts/diagnostics/contact_check_export.py BAG_DIR OUT.json
Record the bag with flytrack_eval's bag_recorder_node and topics /clock /world/scenario
/world/agent/pose /world/target/pose /sim/contact_events (and /sim/contact_probe for run (b))
while the world runs with contact_check:=true. All times are header stamps (simulation clock),
except /sim/contact_probe, whose records carry no stamp and are kept in bag order.
Pose rows are [t, x, y, yaw, z, pitch] (pitch positive = nose down, radians).
Analyse with contact_check_analyze.py (no ROS needed).
"""

import json
import math
import sys
from pathlib import Path

import rosbag2_py
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message

bag, output = Path(sys.argv[1]), Path(sys.argv[2])
reader = rosbag2_py.SequentialReader()
reader.open(
    rosbag2_py.StorageOptions(uri=str(bag), storage_id="sqlite3"),
    rosbag2_py.ConverterOptions("cdr", "cdr"),
)
types = {t.name: get_message(t.type) for t in reader.get_all_topics_and_types()}
wanted = {
    "/world/scenario",
    "/world/agent/pose",
    "/world/target/pose",
    "/sim/contact_events",
    "/sim/contact_probe",
}
missing = {"/world/scenario", "/world/agent/pose", "/sim/contact_events"} - set(types)
if missing:
    raise SystemExit(f"bag lacks {sorted(missing)}")
reader.set_filter(rosbag2_py.StorageFilter(topics=sorted(wanted & set(types))))
result = {"events": [], "agent": [], "target": [], "contacts": [], "probe": []}


def stamp(header):
    return header.stamp.sec + header.stamp.nanosec * 1e-9


def yaw(q):
    return math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))


def pitch(q):
    """Positive = nose down (rotation about the body y axis)."""
    return math.asin(max(-1.0, min(1.0, 2 * (q.w * q.y - q.z * q.x))))


while reader.has_next():
    topic, data, _ = reader.read_next()
    message = deserialize_message(data, types[topic])
    if topic == "/world/scenario":
        result["events"].append(
            {
                "stamp": stamp(message.header),
                "episode_id": int(message.episode_id),
                "seed": int(message.seed),
                "scenario": message.scenario,
                "active": bool(message.active),
                "parameters": json.loads(message.parameters_json),
            }
        )
    elif topic in ("/world/agent/pose", "/world/target/pose"):
        p = message.pose
        key = "agent" if "agent" in topic else "target"
        q = p.orientation
        row = [stamp(message.header), p.position.x, p.position.y, yaw(q), p.position.z, pitch(q)]
        result[key].append(row)
    elif topic == "/sim/contact_events":
        result["contacts"].append(json.loads(message.data))
    else:
        result["probe"].append(json.loads(message.data))
for key in ("agent", "target"):
    result[key].sort()
result["contacts"].sort(key=lambda event: event["stamp"])
output.write_text(json.dumps(result) + "\n", encoding="utf-8")
print({key: len(value) for key, value in result.items()})
