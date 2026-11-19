"""Thin ROS policy node: subscribe to corrupted sensory data only."""

from __future__ import annotations

import signal
import time

import numpy as np
import rclpy
from flytrack_core.sensing.hex_eye import CAMERA_KERNELS, HexEye
from flytrack_interfaces.msg import NeuralActivity, OdorReading, WindReading
from geometry_msgs.msg import Twist
from rclpy.duration import Duration
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from rclpy.signals import SignalHandlerOptions
from sensor_msgs.msg import Image
from std_msgs.msg import Float64
from std_srvs.srv import Trigger

from .artifact import load_deployment_policy
from .contract import check_deployment_contract
from .profiling import PolicyProfiler
from .runtime import PolicyRuntime
from .timing import SENSORS as TRACED_SENSORS
from .timing import ControlTracer

SENSOR_TOPICS = (
    "/fly/camera/left/image_raw",
    "/fly/camera/right/image_raw",
    "/fly/odor",
)
WIND_TOPIC = "/fly/wind"
WORLDS = ("v1compat", "wind")


def image_to_grayscale(message: Image) -> np.ndarray:
    """Decode declared strides and encodings without a cv_bridge dependency."""
    encoding = message.encoding.lower()
    channels = {"mono8": 1, "rgb8": 3, "bgr8": 3, "rgba8": 4, "bgra8": 4}
    if message.width < 1 or message.height < 1 or message.step < 1:
        raise ValueError("Empty camera image")
    if len(message.data) < message.height * message.step:
        raise ValueError("Truncated camera image")
    if encoding == "32fc1":
        if message.step < message.width * 4:
            raise ValueError("Camera stride is shorter than one row")
        dtype = np.dtype(">f4" if message.is_bigendian else "<f4")
        image = np.ndarray(
            (message.height, message.width),
            dtype=dtype,
            buffer=message.data,
            strides=(message.step, 4),
        ).astype(np.float32)
        return np.clip(image, 0, 1)
    if encoding not in channels:
        raise ValueError(f"Unsupported camera encoding: {message.encoding}")
    count = channels[encoding]
    if message.step < message.width * count:
        raise ValueError("Camera stride is shorter than one row")
    rows = np.frombuffer(message.data, dtype=np.uint8, count=message.height * message.step).reshape(
        message.height, message.step
    )
    image = (
        rows[:, : message.width * count]
        .reshape(message.height, message.width, count)
        .astype(np.float32)
        / 255
    )
    if count == 1:
        return image[..., 0]
    rgb = image[..., :3]
    if encoding.startswith("bgr"):
        rgb = rgb[..., ::-1]
    return rgb @ np.array([0.299, 0.587, 0.114], dtype=np.float32)


def stamp_seconds(header) -> float:
    return header.stamp.sec + header.stamp.nanosec * 1e-9


class ConnectomePolicyNode(Node):
    def __init__(self):
        super().__init__("connectome_policy_node")
        if not self.has_parameter("use_sim_time"):
            self.declare_parameter("use_sim_time", True)
        if not self.get_parameter("use_sim_time").value:
            raise ValueError("connectome_policy_node requires use_sim_time=true")
        self.declare_parameter("artifact", "")
        self.declare_parameter("sensor_contract_json", "")
        self.declare_parameter("device", "cpu")
        self.declare_parameter("profiler_enabled", False)
        self.declare_parameter("profiler_output_path", "")
        # #111 G0e: "subrays" samples the camera with the 2D ray cast's quadrature (the kernel the
        # actors were trained on); "gaussian" reproduces the runs made before G0e (formal-v1 G2).
        self.declare_parameter("vision_kernel", "subrays")
        # #140: "wind" feeds a v2 policy the antenna reading from /fly/wind; "v1compat" gives it
        # wind 0 with valid 0, as the 2D v1compat world does.
        self.declare_parameter("world", "v1compat")
        profiler_enabled = bool(self.get_parameter("profiler_enabled").value)
        profiler_output_path = str(self.get_parameter("profiler_output_path").value)
        if profiler_enabled and not profiler_output_path:
            raise ValueError("profiler_output_path is required when profiler_enabled=true")
        artifact = str(self.get_parameter("artifact").value)
        if not artifact:
            raise ValueError(
                "The artifact parameter must identify a complete exported policy directory"
            )
        device = str(self.get_parameter("device").value)
        policy, contract, self.card, self.artifact_spec = load_deployment_policy(
            artifact, device=device
        )
        check_deployment_contract(str(self.get_parameter("sensor_contract_json").value), contract)
        self.contract = contract
        self.vision_kernel = str(self.get_parameter("vision_kernel").value)
        if self.vision_kernel not in CAMERA_KERNELS:
            raise ValueError(f"vision_kernel must be one of {CAMERA_KERNELS}")
        self.world = str(self.get_parameter("world").value)
        if self.world not in WORLDS:
            raise ValueError(f"world must be one of {WORLDS}, got {self.world!r}")
        if self.world == "wind" and self.artifact_spec != "v2":
            raise ValueError("world=wind needs a v2 policy: a v1 actor has no wind input")
        self.eye = HexEye(contract)
        self.runtime = PolicyRuntime(
            policy,
            contract,
            device,
            profiler_enabled=profiler_enabled,
            wind=self.world == "wind",
        )
        self.tracer = ControlTracer(self)
        camera_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST, depth=2, reliability=ReliabilityPolicy.BEST_EFFORT
        )
        odor_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST, depth=5, reliability=ReliabilityPolicy.BEST_EFFORT
        )
        command_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST, depth=1, reliability=ReliabilityPolicy.RELIABLE
        )
        activity_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST, depth=2, reliability=ReliabilityPolicy.BEST_EFFORT
        )
        self.command_publisher = self.create_publisher(Twist, "/cmd_vel", command_qos)
        self.activity_publisher = self.create_publisher(
            NeuralActivity, "/fly/neural_activity", activity_qos
        )
        self.latency_publisher = self.create_publisher(
            Float64, "/fly/inference_latency_ms", odor_qos
        )
        self.camera_subscriptions = [
            self.create_subscription(
                Image, SENSOR_TOPICS[0], lambda msg: self.camera_callback(msg, 0), camera_qos
            ),
            self.create_subscription(
                Image, SENSOR_TOPICS[1], lambda msg: self.camera_callback(msg, 1), camera_qos
            ),
        ]
        self.odor_subscription = self.create_subscription(
            OdorReading, SENSOR_TOPICS[2], self.odor_callback, odor_qos
        )
        self.wind_subscription = (
            self.create_subscription(WindReading, WIND_TOPIC, self.wind_callback, odor_qos)
            if self.world == "wind"
            else None
        )
        self.reset_service = self.create_service(
            Trigger, "/connectome_policy_node/reset", self.reset_callback
        )
        self.timer = self.create_timer(contract.control_dt, self.control_callback)
        self.profiler = PolicyProfiler(
            profiler_enabled,
            profiler_output_path,
            metadata={"device": device, "control_dt_s": contract.control_dt},
        )
        if profiler_enabled:
            self.tracer.on_trace_state = self.profiler.record_trace_state
        self.last_activity_stamp = -float("inf")
        self.get_logger().info(
            f"Loaded {self.artifact_spec} {self.card['variant']} policy with "
            f"{self.card.get('trainable_parameters', 'unrecorded')} trainable parameters"
        )

    def camera_callback(self, message: Image, eye: int) -> None:
        received_ns = time.monotonic_ns()
        sensor = ("left", "right")[eye]
        profile = self.profiler.begin_sensor_callback(sensor, message.header, received_ns)
        accepted = False
        try:
            if (
                message.width != self.contract.camera_width
                or message.height != self.contract.camera_height
            ):
                raise ValueError("Image dimensions differ from sensor contract")
            decode_started_ns = time.monotonic_ns() if profile is not None else 0
            gray = image_to_grayscale(message)
            if profile is not None:
                self.profiler.span(profile, "image_decode", decode_started_ns, time.monotonic_ns())
            if not np.isfinite(gray).all():
                self.runtime.mark_missing(eye, stamp_seconds(message.header))
                return
            sample_started_ns = time.monotonic_ns() if profile is not None else 0
            values = self.eye.sample_image(gray, eye=eye, kernel=self.vision_kernel)
            if profile is not None:
                self.profiler.span(
                    profile, "hex_eye_sample", sample_started_ns, time.monotonic_ns()
                )
            update_started_ns = time.monotonic_ns() if profile is not None else 0
            accepted = self.runtime.update_vision(eye, values, stamp_seconds(message.header))
            if profile is not None:
                self.profiler.span(profile, "buffer_update", update_started_ns, time.monotonic_ns())
            if accepted:
                self.tracer.received(eye, message.header, received_ns)
        except (ValueError, TypeError) as error:
            self.runtime.mark_missing(eye, stamp_seconds(message.header))
            self.get_logger().warning(f"Rejected camera sample: {error}", throttle_duration_sec=5)
        finally:
            self.profiler.end_sensor_callback(profile, time.monotonic_ns(), accepted)

    def odor_callback(self, message: OdorReading) -> None:
        received_ns = time.monotonic_ns()
        profile = self.profiler.begin_sensor_callback("odor", message.header, received_ns)
        accepted = False
        try:
            update_started_ns = time.monotonic_ns() if profile is not None else 0
            accepted = self.runtime.update_odor(
                np.array([message.left, message.right], dtype=np.float32),
                stamp_seconds(message.header),
                valid=bool(message.valid),
            )
            if profile is not None:
                self.profiler.span(profile, "buffer_update", update_started_ns, time.monotonic_ns())
            if accepted:
                self.tracer.received(2, message.header, received_ns)
        finally:
            self.profiler.end_sensor_callback(profile, time.monotonic_ns(), accepted)

    def wind_callback(self, message: WindReading) -> None:
        # Not in the causal ControlTrace, which has a slot for each of v1's three sensors.
        self.runtime.update_wind(
            np.array([message.left, message.right], dtype=np.float32),
            stamp_seconds(message.header),
            valid=bool(message.valid),
        )

    def reset_callback(
        self, request: Trigger.Request, response: Trigger.Response
    ) -> Trigger.Response:
        now = self.get_clock().now()
        self.runtime.reset(minimum_stamp=now.nanoseconds * 1e-9)
        discarded = self.tracer.reset()
        self.profiler.record_reset(
            now.to_msg(), time.monotonic_ns(), discarded_trace_sequences=discarded
        )
        self.last_activity_stamp = -float("inf")
        self.publish_stop()
        response.success, response.message = True, "Policy state and sensor buffers reset"
        return response

    def control_callback(self) -> None:
        started_ns = time.monotonic_ns()
        profile_sequence = self.profiler.next_command_sequence()
        now = self.get_clock().now()
        seconds = now.nanoseconds * 1e-9
        try:
            command, activity, valid = self.runtime.tick(seconds)
        except (ValueError, RuntimeError) as error:
            self.runtime.reset(minimum_stamp=seconds)
            publish_started_ns = time.monotonic_ns()
            self.publish_stop()
            published_ns = time.monotonic_ns()
            self.get_logger().error(f"Policy inference failed: {error}", throttle_duration_sec=5)
            self.profiler.record_command(
                command_sequence=profile_sequence,
                trace_sequence=None,
                trace_status="not_submitted",
                sim_stamp=now.to_msg(),
                callback_entry_ns=started_ns,
                callback_exit_ns=time.monotonic_ns(),
                publish_start_ns=publish_started_ns,
                publish_done_ns=published_ns,
                spans_ns={"tick_error": [started_ns, publish_started_ns]},
                samples=self.profiler.sensor_references(
                    self.runtime.stamps, self.tracer.latest, [False, False, False]
                ),
                inference_performed=False,
                valid=[False, False, False],
            )
            return
        message = Twist()
        message.linear.x, message.angular.z = float(command[0]), float(command[1])
        publish_started_ns = time.monotonic_ns()
        self.command_publisher.publish(message)
        published_ns = time.monotonic_ns()
        # The trace and profiler cover v1's three sensors; the wind slot (index 3) is left out.
        traced = len(TRACED_SENSORS)
        samples = self.profiler.sensor_references(
            self.runtime.stamps[:traced], self.tracer.latest, valid[:traced]
        )
        trace_sequence, trace_status = self.tracer.command(
            now.to_msg(),
            started_ns,
            published_ns,
            valid[:traced],
            self.runtime.stamps[:traced],
            self.runtime.last_inference_latency_ms is not None,
        )
        if self.runtime.last_inference_latency_ms is not None:
            latency_message = Float64()
            latency_message.data = self.runtime.last_inference_latency_ms
            self.latency_publisher.publish(latency_message)
        if seconds < self.last_activity_stamp:
            self.last_activity_stamp = -float("inf")
        if activity is not None and seconds - self.last_activity_stamp >= 0.1 - 1e-6:
            activity_message = NeuralActivity()
            activity_message.header.stamp = now.to_msg()
            activity_message.header.frame_id = "base_link"
            activity_message.node_ids = self.runtime.policy.node_ids.detach().cpu().tolist()
            activity_message.activity = activity.astype(np.float32).tolist()
            activity_message.variant = self.card["variant"]
            self.activity_publisher.publish(activity_message)
            self.last_activity_stamp = seconds
        self.profiler.record_command(
            command_sequence=profile_sequence,
            trace_sequence=trace_sequence,
            trace_status=trace_status,
            sim_stamp=now.to_msg(),
            callback_entry_ns=started_ns,
            callback_exit_ns=time.monotonic_ns(),
            publish_start_ns=publish_started_ns,
            publish_done_ns=published_ns,
            spans_ns=self.runtime.last_profile_spans_ns,
            samples=samples,
            inference_performed=self.runtime.last_inference_latency_ms is not None,
            valid=valid,
        )

    def publish_stop(self) -> None:
        self.command_publisher.publish(Twist())


def _interrupt(signum, frame):
    raise KeyboardInterrupt


def main(args=None) -> None:
    # rclpy's own SIGINT handler shut the context down before the finally block,
    # so the zero command never left and Gazebo kept the last cmd_vel. Python
    # handles SIGINT/SIGTERM here instead; spin_once with a timeout returns to the
    # interpreter so the interrupt is raised while the context is still valid. One
    # executor for the whole run, as rclpy.spin() does: rclpy.spin_once(node) adds
    # and removes the node on every call.
    rclpy.init(args=args, signal_handler_options=SignalHandlerOptions.NO)
    signal.signal(signal.SIGTERM, _interrupt)
    node = executor = None
    try:
        node = ConnectomePolicyNode()
        executor = SingleThreadedExecutor()
        executor.add_node(node)
        while rclpy.ok():
            executor.spin_once(timeout_sec=0.1)
    except KeyboardInterrupt:
        pass
    finally:
        if node is not None and rclpy.ok():
            node.publish_stop()
            # Reliable QoS: let subscribers acknowledge before teardown.
            node.command_publisher.wait_for_all_acked(Duration(seconds=1.0))
        # Unlike the global executor, our own is not closed by rclpy.shutdown().
        if executor is not None:
            executor.shutdown()
        if node is not None:
            node.profiler.close()
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
