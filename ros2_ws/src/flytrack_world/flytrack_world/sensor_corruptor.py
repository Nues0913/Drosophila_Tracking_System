"""Corrupt raw sensor frames and preserve acquisition headers unchanged.

The antenna reading (``/sim/wind``, v2 wind world) is relayed to ``/fly/wind`` with the noise
scenario's odor dropout drawn independently, and no further noise: the world already adds the
antenna noise, and the 2D wind world gives the wind no extra noise in the noise scenario (#169).
"""

import time
from copy import deepcopy

import numpy as np
from flytrack_core.sensing import SensorCorruptor
from flytrack_interfaces.msg import OdorReading, ScenarioEvent, WindReading
from sensor_msgs.msg import Image
from std_msgs.msg import String

from .common import CAMERA_QOS, EVENT_QOS, SENSOR_QOS, WorldNode, spin_main, stamp_seconds
from .timing import TIMING_QOS, SensorTimingRelay
from .wind_field import wind_dropped, wind_rng


def decode_image(message: Image) -> np.ndarray:
    encodings = {
        "mono8": (np.uint8, 1),
        "8UC1": (np.uint8, 1),
        "mono16": (np.uint16, 1),
        "16UC1": (np.uint16, 1),
        "rgb8": (np.uint8, 3),
        "bgr8": (np.uint8, 3),
        "rgba8": (np.uint8, 4),
        "bgra8": (np.uint8, 4),
        "32FC1": (np.float32, 1),
    }
    if message.encoding not in encodings:
        raise ValueError(f"unsupported camera encoding {message.encoding}")
    dtype, channels = encodings[message.encoding]
    dtype = np.dtype(dtype).newbyteorder(">" if message.is_bigendian else "<")
    if message.step < message.width * channels * dtype.itemsize:
        raise ValueError("image step is shorter than a pixel row")
    if len(message.data) < message.height * message.step:
        raise ValueError("camera data shorter than height*step")
    pixels = np.ndarray(
        (message.height, message.width, channels),
        dtype=dtype,
        buffer=bytes(message.data),
        strides=(message.step, channels * dtype.itemsize, dtype.itemsize),
    )
    values = pixels.astype(np.float32)
    if np.issubdtype(dtype, np.integer):
        values /= np.iinfo(dtype).max
    if channels == 1:
        values = values[..., 0]
    else:
        rgb = values[..., :3]
        if message.encoding in ("bgr8", "bgra8"):
            rgb = rgb[..., ::-1]
        values = rgb @ np.asarray([0.299, 0.587, 0.114], np.float32)
    return np.clip(np.nan_to_num(values, nan=0.0, posinf=1.0, neginf=0.0), 0, 1)


class SensorCorruptorNode(WorldNode):
    def __init__(self):
        super().__init__("sensor_corruptor")
        self.corruptor = SensorCorruptor(self.base_seed)
        self.wind_rng = wind_rng(self.base_seed)
        self.timing = SensorTimingRelay(self)
        self.camera_publishers = [
            self.create_publisher(Image, f"/fly/camera/{eye}/image_raw", CAMERA_QOS)
            for eye in ("left", "right")
        ]
        self.odor_publisher = self.create_publisher(OdorReading, "/fly/odor", SENSOR_QOS)
        for index, eye in enumerate(("left", "right")):
            self.create_subscription(
                String,
                f"/sim/camera/{eye}/source_timing",
                lambda message, sensor=eye: self.timing.camera_source(sensor, message),
                TIMING_QOS,
            )
            self.create_subscription(
                Image,
                f"/sim/camera/{eye}/image_raw",
                lambda message, eye_index=index: self._image(message, eye_index),
                CAMERA_QOS,
            )
        self.wind_publisher = self.create_publisher(WindReading, "/fly/wind", SENSOR_QOS)
        self.create_subscription(OdorReading, "/sim/odor", self._odor, SENSOR_QOS)
        self.create_subscription(WindReading, "/sim/wind", self._wind, SENSOR_QOS)
        self.create_subscription(ScenarioEvent, "/world/scenario", self._scenario, EVENT_QOS)

    def _scenario(self, message):
        new, _ = self.consume_event(message)
        if new:
            self.corruptor.reset(int(message.seed))
            self.wind_rng = wind_rng(int(message.seed))
            self.timing.reset()

    def _image(self, message, eye: int):
        ingress_ns = time.monotonic_ns()
        stamp = stamp_seconds(message.header.stamp)
        if not self.active or stamp <= self.started_at_s:
            return
        try:
            image = decode_image(message)
            corrupted, valid = self.corruptor.image(
                image, self.scenario_spec.corruption_at(stamp - self.started_at_s), eye
            )
            if not valid:
                return  # Drop transport frame; policy expires its previous stamp at 200 ms.
            output = Image()
            output.header = deepcopy(message.header)
            output.height, output.width = message.height, message.width
            output.encoding, output.is_bigendian, output.step = "mono8", 0, message.width
            output.data = np.rint(corrupted * 255).astype(np.uint8).tobytes()
            forward_ns = time.monotonic_ns()
            self.camera_publishers[eye].publish(output)
            self.timing.record(output.header, ("left", "right")[eye], ingress_ns, forward_ns)
        except ValueError as error:
            self.get_logger().error(str(error))

    def _odor(self, message):
        ingress_ns = time.monotonic_ns()
        stamp = stamp_seconds(message.header.stamp)
        if not self.active or stamp <= self.started_at_s:
            return
        if not message.valid:
            output = deepcopy(message)
            output.left = output.right = 0.0
            self.odor_publisher.publish(output)
            self.timing.record(
                output.header,
                "odor",
                ingress_ns,
                time.monotonic_ns(),
                message.source_monotonic_ns,
                message.clock_domain,
            )
            return
        corrupted, valid = self.corruptor.odor(
            np.asarray([message.left, message.right]),
            self.scenario_spec.corruption_at(stamp - self.started_at_s),
        )
        if valid:
            output = deepcopy(message)
            output.left, output.right = float(corrupted[0]), float(corrupted[1])
            forward_ns = time.monotonic_ns()
            self.odor_publisher.publish(output)
            self.timing.record(
                output.header,
                "odor",
                ingress_ns,
                forward_ns,
                message.source_monotonic_ns,
                message.clock_domain,
            )

    def _wind(self, message):
        ingress_ns = time.monotonic_ns()
        stamp = stamp_seconds(message.header.stamp)
        if not self.active or stamp <= self.started_at_s:
            return
        config = self.scenario_spec.corruption_at(stamp - self.started_at_s)
        if message.valid and wind_dropped(self.wind_rng, config.odor_dropout):
            return  # Drop the sample; the policy expires its previous stamp at the watchdog.
        output = deepcopy(message)
        if not message.valid:
            output.left = output.right = 0.0
        forward_ns = time.monotonic_ns()
        self.wind_publisher.publish(output)
        self.timing.record(
            output.header,
            "wind",
            ingress_ns,
            forward_ns,
            message.source_monotonic_ns,
            message.clock_domain,
        )


def main(args=None):
    spin_main(SensorCorruptorNode, args)
