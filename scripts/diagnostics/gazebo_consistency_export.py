"""Export a Gazebo rosbag for the 2D/Gazebo input consistency check (#88, formerly #23).

Usage (ROS image, e.g. docker run ... flytrack-ros:0.1):
  python3 scripts/diagnostics/gazebo_consistency_export.py BAG_DIR SENSOR_CONTRACT_JSON OUT.npz \
      [KERNEL]
KERNEL is the policy node's vision_kernel: "subrays" (default since #111 G0e) or "gaussian"
(the kernel used before G0e, e.g. to re-export a G0c/G2 bag as the policy saw it).
Record the bag with flytrack_eval's bag_recorder_node (default topics) during a Gazebo cell.
Camera frames are converted exactly as the policy node does (image_to_grayscale, then
HexEye(contract).sample_image with the same kernel), so the exported vectors are the policy's
vision input. Also
exported: agent/target poses [t, x, y, yaw], odor [t, left, right, valid] and the /world/scenario
events (JSON). Times are header stamps (simulation clock). Compare with
gazebo_consistency_compare.py in the ML image.
"""

import json
import sys
from pathlib import Path

import numpy as np
import rosbag2_py
from flytrack_core.contracts import SensorContract
from flytrack_core.sensing.hex_eye import HexEye
from flytrack_policy.node import image_to_grayscale
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message

bag, contract_path, output = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
kernel = sys.argv[4] if len(sys.argv) > 4 else "subrays"
contract = SensorContract.from_dict(json.loads(contract_path.read_text(encoding="utf-8")))
eye = HexEye(contract)
reader = rosbag2_py.SequentialReader()
reader.open(
    rosbag2_py.StorageOptions(uri=str(bag), storage_id="sqlite3"),
    rosbag2_py.ConverterOptions("cdr", "cdr"),
)
types = {t.name: get_message(t.type) for t in reader.get_all_topics_and_types()}
wanted = {
    "/fly/camera/left/image_raw",
    "/fly/camera/right/image_raw",
    "/world/agent/pose",
    "/world/target/pose",
    "/fly/odor",
    "/world/scenario",
}
reader.set_filter(rosbag2_py.StorageFilter(topics=sorted(wanted & set(types))))
vision = {0: [], 1: []}
gray_stats = {0: [], 1: []}
poses = {"/world/agent/pose": [], "/world/target/pose": []}
odor, events = [], []


def stamp(header):
    return header.stamp.sec + header.stamp.nanosec * 1e-9


def yaw(q):
    return float(np.arctan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z)))


while reader.has_next():
    topic, data, _ = reader.read_next()
    message = deserialize_message(data, types[topic])
    if topic.startswith("/fly/camera/"):
        index = 0 if "/left/" in topic else 1
        gray = image_to_grayscale(message)
        values = eye.sample_image(gray, eye=index, kernel=kernel)
        vision[index].append((stamp(message.header), values))
        gray_stats[index].append((stamp(message.header), float(gray.mean()), float(gray.std())))
    elif topic in poses:
        p = message.pose
        poses[topic].append((stamp(message.header), p.position.x, p.position.y, yaw(p.orientation)))
    elif topic == "/fly/odor":
        odor.append((stamp(message.header), message.left, message.right, float(message.valid)))
    elif topic == "/world/scenario":
        events.append(
            {
                "stamp": stamp(message.header),
                "episode_id": int(message.episode_id),
                "seed": int(message.seed),
                "scenario": message.scenario,
                "active": bool(message.active),
                "parameters": json.loads(message.parameters_json),
            }
        )
arrays = {}
for index in (0, 1):
    frames = sorted(vision[index], key=lambda item: item[0])
    arrays[f"vision{index}_t"] = np.array([t for t, _ in frames])
    arrays[f"vision{index}"] = np.stack([v for _, v in frames]).astype(np.float32)
    arrays[f"gray{index}"] = np.array(sorted(gray_stats[index]))
arrays["agent"] = np.array(sorted(poses["/world/agent/pose"]))
arrays["target"] = np.array(sorted(poses["/world/target/pose"]))
arrays["odor"] = np.array(sorted(odor))
arrays["events_json"] = np.array(json.dumps(events, sort_keys=True))
arrays["contract_sha256"] = np.array(contract.sha256())
arrays["vision_kernel"] = np.array(kernel)
np.savez_compressed(output, **arrays)
print({k: v.shape for k, v in arrays.items()}, len(events), "events")
