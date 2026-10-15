"""Fetch verified Gazebo camera sources and add a source-publication timing hook.

Upstream gz-sensors is Apache-2.0. The original file license headers remain in
the downloaded build tree. Only the camera library is rebuilt as an image-local
overlay; no host or upstream installation is replaced.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import urllib.request
from pathlib import Path

TAG = "gz-sensors10_10.0.2"
BASE = f"https://raw.githubusercontent.com/gazebosim/gz-sensors/{TAG}/src/"
FILES = {
    "CameraSensor.cc": "705c56bd817eff1e30f9d9e50b54b006707e623d62ab09628fcb2965825b1c35",
    "CameraSensorUtil.cc": "bb5f1112909188687768f69860a9035c0c2555db8e60d041a749c26d3dd80c57",
    "CameraSensorUtil.hh": "f9c968072fddcf7d746db9dbf0078ec446baa680fa82bdf104dca539a8f8531a",
}


def replace_once(source: str, original: str, replacement: str) -> str:
    if source.count(original) != 1:
        raise ValueError(f"Timing patch requires exactly one upstream hook: {original!r}")
    return source.replace(original, replacement)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    for name, expected in FILES.items():
        with urllib.request.urlopen(BASE + name, timeout=60) as response:
            payload = response.read()
        if hashlib.sha256(payload).hexdigest() != expected:
            raise ValueError(f"Upstream SHA-256 differs for {name}; refusing to compile")
        (args.output / name).write_bytes(payload)
    path = args.output / "CameraSensor.cc"
    source = path.read_text()
    source = replace_once(
        source,
        "#include <gz/msgs/image.pb.h>",
        "#include <gz/msgs/image.pb.h>\n#include <gz/msgs/stringmsg.pb.h>\n"
        "#include <cstdlib>\n#include <time.h>",
    )
    source = replace_once(
        source,
        "  public: transport::Node::Publisher pub;",
        "  public: transport::Node::Publisher pub;\n"
        "  public: transport::Node::Publisher timingPub;",
    )
    source = replace_once(
        source,
        "  if (!this->dataPtr->pub)\n",
        "  this->dataPtr->timingPub =\n"
        "      this->dataPtr->node.Advertise<gz::msgs::StringMsg>(\n"
        '          this->Topic() + "/source_timing");\n'
        "  if (!this->dataPtr->pub)\n",
    )
    hook = """      // Flytrack timing starts when the complete sensor message exists,
      // immediately before its native Gazebo Transport publication.
      struct timespec wall;
      if (clock_gettime(CLOCK_MONOTONIC, &wall) == 0)
      {
        const auto ns = static_cast<unsigned long long>(wall.tv_sec) *
                        1000000000ULL + wall.tv_nsec;
        const char *domain = std::getenv("FLYTRACK_CLOCK_DOMAIN");
        gz::msgs::StringMsg timing;
        timing.set_data(std::to_string(msg.header().stamp().sec()) + "," +
                        std::to_string(msg.header().stamp().nsec()) + "," +
                        std::to_string(ns) + "," + (domain ? domain : ""));
        this->dataPtr->timingPub.Publish(timing);
      }
      this->dataPtr->pub.Publish(msg);"""
    source = replace_once(source, "      this->dataPtr->pub.Publish(msg);", hook)
    path.write_text(source)
    (args.output / "source_manifest.json").write_text(
        json.dumps(
            {
                "upstream": BASE,
                "upstream_sha256": FILES,
                "patched_camera_sha256": hashlib.sha256(source.encode()).hexdigest(),
                "basis": "complete sensor image message, immediately before Gazebo publication",
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
