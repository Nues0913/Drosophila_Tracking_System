"""Forward the agent's non-floor contacts as compact JSON events (#116 G0f, contact_check only).

The wheels and the caster touch the floor at every physics step, so the raw contact topics are
not recorded. This node keeps every contact whose other collision is not the floor and publishes
one std_msgs/String per contact message on /sim/contact_events:
{"stamp": sim seconds, "link": agent link, "contacts": [{"self", "other", "depth"}]}.
The stamp is the contact message's header stamp (simulation clock).
"""

from __future__ import annotations

import json

from rclpy.node import Node
from rclpy.qos import QoSDurabilityPolicy, QoSProfile, QoSReliabilityPolicy
from std_msgs.msg import String

from flytrack_world.common import spin_main
from flytrack_world.contact_world import EVENTS_TOPIC, ROS_TOPIC_PREFIX, contact_event

EVENTS_QOS = QoSProfile(
    depth=1000,
    reliability=QoSReliabilityPolicy.RELIABLE,
    durability=QoSDurabilityPolicy.VOLATILE,
)


class ContactMonitorNode(Node):
    def __init__(self):
        super().__init__("contact_monitor")
        from ros_gz_interfaces.msg import Contacts

        self.declare_parameter("links", ["base_link"])
        self.publisher = self.create_publisher(String, EVENTS_TOPIC, EVENTS_QOS)
        self.forwarded = 0
        for link in self.get_parameter("links").value:
            self.create_subscription(
                Contacts,
                f"{ROS_TOPIC_PREFIX}/{link}",
                lambda message, link=link: self._contacts(link, message),
                100,
            )

    def _contacts(self, link, message):
        stamp = message.header.stamp.sec + message.header.stamp.nanosec * 1e-9
        event = contact_event(
            link,
            stamp,
            [
                (c.collision1.name, c.collision2.name, max(c.depths, default=0.0))
                for c in message.contacts
            ],
        )
        if event is not None:
            self.publisher.publish(String(data=json.dumps(event, sort_keys=True)))
            self.forwarded += 1


def main(args=None):
    spin_main(ContactMonitorNode, args)
