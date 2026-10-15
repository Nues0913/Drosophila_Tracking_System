"""Docker health probe: succeeds only after two increasing simulation clock messages."""

import time

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from rosgraph_msgs.msg import Clock


def main():
    rclpy.init()
    node = Node("flytrack_clock_probe")
    stamps = []
    subscription = node.create_subscription(
        Clock,
        "/clock",
        lambda msg: stamps.append(msg.clock.sec + msg.clock.nanosec * 1e-9),
        QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT),
    )
    deadline = time.monotonic() + 3.0
    ok = False
    while time.monotonic() < deadline:
        rclpy.spin_once(node, timeout_sec=0.1)
        if len(stamps) >= 2 and stamps[-1] > stamps[0]:
            ok = True
            break
    node.destroy_subscription(subscription)
    node.destroy_node()
    rclpy.shutdown()
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
