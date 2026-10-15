"""Run after building ROS packages and exporting a real artifact.

ARTIFACT=/absolute/actor SENSOR_CONTRACT=/absolute/contract.json launch_test <this file>
No simulator or sensory messages are needed to inspect the actor subscription graph.
"""

import os
import tempfile
import time
import unittest

import launch
import launch_ros.actions
import launch_testing
import launch_testing.actions
import rclpy


def generate_test_description():
    artifact = os.environ.get("ARTIFACT")
    contract = os.environ.get("SENSOR_CONTRACT")
    if not artifact or not contract:
        raise RuntimeError(
            "Truth isolation test requires ARTIFACT and SENSOR_CONTRACT environment variables"
        )
    actor = launch_ros.actions.Node(
        package="flytrack_policy",
        executable="connectome_policy_node",
        name="policy_under_test",
        output="screen",
        parameters=[{"artifact": artifact, "sensor_contract_json": contract, "use_sim_time": True}],
    )
    evaluator = launch_ros.actions.Node(
        package="flytrack_eval",
        executable="evaluator_node",
        name="evaluator_under_test",
        output="screen",
        parameters=[
            {
                "use_sim_time": True,
                "output_dir": tempfile.mkdtemp(prefix="flytrack_eval_isolation_"),
            }
        ],
    )
    return launch.LaunchDescription([actor, evaluator, launch_testing.actions.ReadyToTest()]), {
        "actor": actor,
        "evaluator": evaluator,
    }


class TestActorSubscriptions(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        rclpy.init()
        cls.observer = rclpy.create_node("truth_isolation_observer")

    @classmethod
    def tearDownClass(cls):
        cls.observer.destroy_node()
        rclpy.shutdown()

    def test_no_privileged_topics(self):
        deadline = time.monotonic() + 30
        topics = set()
        while time.monotonic() < deadline:
            rclpy.spin_once(self.observer, timeout_sec=0.1)
            if ("policy_under_test", "/") not in self.observer.get_node_names_and_namespaces():
                continue
            topics = {
                name
                for name, _ in self.observer.get_subscriber_names_and_types_by_node(
                    "policy_under_test", "/"
                )
            }
            if "/fly/odor" in topics and "/fly/camera/left/image_raw" in topics:
                break
        self.assertIn("/fly/odor", topics, "Policy never reached initialized subscription state")
        self.assertIn("/fly/camera/left/image_raw", topics)
        self.assertIn("/fly/camera/right/image_raw", topics)
        self.assertFalse(
            any(
                topic.startswith("/world/")
                or topic.startswith("/sim/")
                or topic.startswith("/eval/")
                for topic in topics
            ),
            topics,
        )
        allowed_framework = {"/clock", "/parameter_events"}
        self.assertTrue(
            all(topic.startswith("/fly/") or topic in allowed_framework for topic in topics), topics
        )

    def test_evaluator_never_publishes_control(self):
        deadline = time.monotonic() + 30
        publications = set()
        while time.monotonic() < deadline:
            rclpy.spin_once(self.observer, timeout_sec=0.1)
            if ("evaluator_under_test", "/") not in self.observer.get_node_names_and_namespaces():
                continue
            publications = {
                name
                for name, _ in self.observer.get_publisher_names_and_types_by_node(
                    "evaluator_under_test", "/"
                )
            }
            if "/eval/metrics" in publications:
                break
        self.assertIn("/eval/metrics", publications, "Evaluator never initialized")
        self.assertNotIn("/cmd_vel", publications)
        self.assertTrue(
            publications <= {"/eval/metrics", "/parameter_events", "/rosout"}, publications
        )
