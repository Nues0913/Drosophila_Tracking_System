"""A deterministic binocular hexagonal retina and its camera-pixel projection.

Directions are (yaw, pitch) radians in the robot frame. Image coordinates use
rectilinear camera optics, +x forward, +y left and +z up. No SciPy/ROS dependency
is needed: the sparse projector stores only contributing pixel indices.

The 2D world renders each ommatidium as the mean of ``subrays`` quadrature rays
(``HexEye.raycast``); the trained actors only ever saw that kernel. A camera image
is therefore sampled by default with the same rays (``kernel="subrays"``), each
read bilinearly from the pixels. The older full-Gaussian kernel (``"gaussian"``,
3 sigma support) is wider than the 7-ray quadrature and lowered the target
contrast in Gazebo by about 10% (#111 G0e); it stays available to reproduce runs
made before G0e.
"""

import hashlib
import math
from dataclasses import dataclass

import numpy as np

from flytrack_core.contracts import SensorContract


def axial_coordinates(radius: int) -> np.ndarray:
    if radius < 0:
        raise ValueError("eye radius must be nonnegative")
    return np.asarray(
        [
            (q, r)
            for r in range(-radius, radius + 1)
            for q in range(-radius, radius + 1)
            if abs(q + r) <= radius
        ],
        dtype=np.int32,
    )


DEFAULT_SUBRAYS = 7
CAMERA_KERNELS = ("subrays", "gaussian")


def subray_offsets(sigma_rad: float, subrays: int) -> list[tuple[float, float]]:
    """(dyaw, dpitch) of each quadrature ray; ray 0 is the ommatidium axis.

    Shared by the 2D ray cast and the camera projection so both use the same kernel.
    """
    if subrays < 1:
        raise ValueError("subrays must be positive")
    offsets = []
    for sample_index in range(subrays):
        if sample_index == 0:
            offsets.append((0.0, 0.0))
            continue
        radial = math.sqrt(-2 * math.log(1 - (sample_index - 0.5) / (subrays - 1)))
        angle = sample_index * math.pi * (3 - math.sqrt(5))
        offsets.append((sigma_rad * radial * math.cos(angle), sigma_rad * radial * math.sin(angle)))
    return offsets


def compute_directions(contract: SensorContract | None = None) -> np.ndarray:
    """Return [2, N, 2]; index 0 is the left eye, index 1 the right eye."""
    contract = contract or SensorContract()
    coordinates = axial_coordinates(contract.eye_radius).astype(np.float64)
    scale = max(contract.eye_radius, 1)
    horizontal = (coordinates[:, 0] + coordinates[:, 1] / 2) / scale
    vertical = coordinates[:, 1] / scale
    # Start with a hexagonal lattice on the pinhole projection plane, then
    # recover angular rays. Independently linear yaw/pitch would push corner
    # ommatidia outside the rectangular camera frustum.
    projected_horizontal = horizontal * math.tan(math.radians(contract.horizontal_fov_deg / 2))
    projected_vertical = vertical * math.tan(math.radians(contract.vertical_fov_deg / 2))
    yaw = np.arctan(projected_horizontal)
    pitch = np.arctan(projected_vertical / np.sqrt(1 + projected_horizontal**2))
    left = np.stack((yaw + math.radians(contract.eye_yaw_deg), pitch), axis=-1)
    right = np.stack((yaw - math.radians(contract.eye_yaw_deg), pitch), axis=-1)
    return np.stack((left, right))


@dataclass(frozen=True)
class GaussianProjection:
    """COO-like N×(height*width) matrix, without dense zero storage."""

    rows: np.ndarray
    columns: np.ndarray
    weights: np.ndarray
    n_rows: int
    image_shape: tuple[int, int]

    def sample(self, image: np.ndarray) -> np.ndarray:
        image = np.asarray(image)
        source_dtype = image.dtype
        if image.shape[:2] != self.image_shape:
            raise ValueError("image shape differs from projection")
        if image.ndim == 3:
            if image.shape[-1] < 3:
                image = image[..., 0]
            else:
                image = image[..., :3] @ np.asarray([0.299, 0.587, 0.114])
        if image.ndim != 2:
            raise ValueError("expected grayscale or RGB image")
        values = image.astype(np.float32, copy=False)
        if np.issubdtype(source_dtype, np.integer):
            values = values / np.iinfo(source_dtype).max
        values = np.nan_to_num(values, nan=0.0, posinf=1.0, neginf=0.0)
        result = np.bincount(
            self.rows, weights=self.weights * values.ravel()[self.columns], minlength=self.n_rows
        )
        return np.clip(result, 0.0, 1.0).astype(np.float32)


class HexEye:
    def __init__(self, contract: SensorContract | None = None):
        self.contract = contract or SensorContract()
        self.directions_rad = compute_directions(self.contract)
        self.axial = axial_coordinates(self.contract.eye_radius)
        self._projectors: dict[tuple, GaussianProjection] = {}

    def sha256(self) -> str:
        """Hash the canonical little-endian float64 direction table in C order."""
        return hashlib.sha256(self.directions_rad.astype("<f8").tobytes(order="C")).hexdigest()

    def projection(
        self,
        eye: int = 0,
        width: int | None = None,
        height: int | None = None,
        kernel: str = "subrays",
        subrays: int = DEFAULT_SUBRAYS,
    ) -> GaussianProjection:
        if eye not in (0, 1):
            raise ValueError("eye must be 0 (left) or 1 (right)")
        if kernel not in CAMERA_KERNELS:
            raise ValueError(f"kernel must be one of {CAMERA_KERNELS}")
        width = width or self.contract.camera_width
        height = height or self.contract.camera_height
        if min(width, height) < 2:
            raise ValueError("camera dimensions must be at least 2")
        key = (eye, width, height, kernel, subrays if kernel == "subrays" else None)
        if key in self._projectors:
            return self._projectors[key]
        if kernel == "subrays":
            projector = self._subray_projection(eye, width, height, subrays)
        else:
            projector = self._gaussian_projection(eye, width, height)
        self._projectors[key] = projector
        return projector

    def _subray_projection(
        self, eye: int, width: int, height: int, subrays: int
    ) -> GaussianProjection:
        """Mean of the raycast's admissible quadrature rays, each read bilinearly."""
        center_yaw = math.radians(self.contract.eye_yaw_deg) * (1 if eye == 0 else -1)
        fx = width / (2 * math.tan(math.radians(self.contract.horizontal_fov_deg / 2)))
        fy = height / (2 * math.tan(math.radians(self.contract.vertical_fov_deg / 2)))
        half_hfov = math.radians(self.contract.horizontal_fov_deg / 2)
        tan_half_vfov = math.tan(math.radians(self.contract.vertical_fov_deg / 2))
        offsets = subray_offsets(math.radians(self.contract.acceptance_angle_deg), subrays)
        rows, columns, weights = [], [], []
        for index, (yaw, pitch) in enumerate(self.directions_rad[eye]):
            points = []
            for dyaw, dpitch in offsets:
                # Same admissibility test as raycast: rays outside the frustum are dropped.
                relative_yaw = yaw + dyaw - center_yaw
                ray_pitch = pitch + dpitch
                if abs(relative_yaw) > half_hfov + 1e-10 or (
                    abs(math.tan(ray_pitch) / math.cos(relative_yaw)) > tan_half_vfov + 1e-10
                ):
                    continue
                px = (width - 1) / 2 - fx * math.tan(relative_yaw)
                py = (height - 1) / 2 - fy * math.tan(ray_pitch) / math.cos(relative_yaw)
                points.append((min(max(px, 0.0), width - 1.0), min(max(py, 0.0), height - 1.0)))
            for px, py in points:
                x0, y0 = min(int(px), width - 2), min(int(py), height - 2)
                ax, ay = px - x0, py - y0
                for x, y, w in (
                    (x0, y0, (1 - ax) * (1 - ay)),
                    (x0 + 1, y0, ax * (1 - ay)),
                    (x0, y0 + 1, (1 - ax) * ay),
                    (x0 + 1, y0 + 1, ax * ay),
                ):
                    if w > 0:
                        rows.append(index)
                        columns.append(y * width + x)
                        weights.append(w / len(points))
        return GaussianProjection(
            np.asarray(rows, dtype=np.int64),
            np.asarray(columns, dtype=np.int64),
            np.asarray(weights, dtype=np.float32),
            len(self.axial),
            (height, width),
        )

    def _gaussian_projection(self, eye: int, width: int, height: int) -> GaussianProjection:
        """Pre-G0e kernel: a full Gaussian over the pixels (3 sigma support)."""
        center_yaw = math.radians(self.contract.eye_yaw_deg) * (1 if eye == 0 else -1)
        fx = width / (2 * math.tan(math.radians(self.contract.horizontal_fov_deg / 2)))
        fy = height / (2 * math.tan(math.radians(self.contract.vertical_fov_deg / 2)))
        sigma_angle = math.radians(self.contract.acceptance_angle_deg)
        rows, columns, weights = [], [], []
        for index, (yaw, pitch) in enumerate(self.directions_rad[eye]):
            relative_yaw = yaw - center_yaw
            px = (width - 1) / 2 - fx * math.tan(relative_yaw)
            py = (height - 1) / 2 - fy * math.tan(pitch) / math.cos(relative_yaw)
            # A Gaussian acceptance angle is interpreted as one standard deviation.
            sx = max(0.5, fx * sigma_angle / math.cos(relative_yaw) ** 2)
            sy = max(0.5, fy * sigma_angle / math.cos(pitch) ** 2 / math.cos(relative_yaw))
            xmin, xmax = max(0, int(px - 3 * sx)), min(width - 1, int(px + 3 * sx) + 1)
            ymin, ymax = max(0, int(py - 3 * sy)), min(height - 1, int(py + 3 * sy) + 1)
            if xmin > xmax or ymin > ymax:
                continue
            xx, yy = np.meshgrid(np.arange(xmin, xmax + 1), np.arange(ymin, ymax + 1))
            local_weights = np.exp(-0.5 * (((xx - px) / sx) ** 2 + ((yy - py) / sy) ** 2))
            local_weights /= local_weights.sum()
            rows.extend([index] * xx.size)
            columns.extend((yy * width + xx).ravel().tolist())
            weights.extend(local_weights.ravel().tolist())
        return GaussianProjection(
            np.asarray(rows, dtype=np.int64),
            np.asarray(columns, dtype=np.int64),
            np.asarray(weights, dtype=np.float32),
            len(self.axial),
            (height, width),
        )

    def sample_image(
        self,
        image: np.ndarray,
        eye: int = 0,
        kernel: str = "subrays",
        subrays: int = DEFAULT_SUBRAYS,
    ) -> np.ndarray:
        return self.projection(eye, image.shape[1], image.shape[0], kernel, subrays).sample(image)

    def raycast(
        self,
        agent_pose: np.ndarray,
        walls: np.ndarray,
        disks: list[dict],
        eye: int = 0,
        *,
        subrays: int = DEFAULT_SUBRAYS,
        camera_height_m: float = 0.27,
        camera_forward_m: float = 0.14,
        camera_baseline_m: float = 0.12,
        background: float = 0.15,
        wall_intensity: float = 0.45,
    ) -> np.ndarray:
        """Render the same angular retina through Gaussian quadrature subrays.

        walls is [xmin,ymin,xmax,ymax]. Disks use x,y,radius,height,intensity.
        This is a 2.5D approximation: vertical rays intersect extruded disks and
        boundary walls, while the robot dynamics remain planar.
        """
        offsets = subray_offsets(math.radians(self.contract.acceptance_angle_deg), subrays)
        pose = np.asarray(agent_pose, dtype=np.float64).copy()
        yaw = pose[2]
        pose[:2] += camera_forward_m * np.asarray([np.cos(yaw), np.sin(yaw)])
        pose[:2] += (
            camera_baseline_m
            / 2
            * (1 if eye == 0 else -1)
            * np.asarray([-np.sin(yaw), np.cos(yaw)])
        )
        directions = self.directions_rad[eye]
        samples = np.full((subrays, len(directions)), background, dtype=np.float64)
        admissible = np.zeros_like(samples, dtype=bool)
        optical_yaw = math.radians(self.contract.eye_yaw_deg) * (1 if eye == 0 else -1)
        half_hfov = math.radians(self.contract.horizontal_fov_deg / 2)
        tan_half_vfov = math.tan(math.radians(self.contract.vertical_fov_deg / 2))
        for sample_index, (dyaw, dpitch) in enumerate(offsets):
            yaw = directions[:, 0] + pose[2] + dyaw
            pitch = directions[:, 1] + dpitch
            relative_yaw = directions[:, 0] + dyaw - optical_yaw
            admissible[sample_index] = (np.abs(relative_yaw) <= half_hfov + 1e-10) & (
                np.abs(np.tan(pitch) / np.cos(relative_yaw)) <= tan_half_vfov + 1e-10
            )
            rays = np.stack((np.cos(yaw), np.sin(yaw)), axis=-1)
            nearest = np.full(len(rays), np.inf)
            # Boundary walls are vertical up to 1 m.
            for axis, boundary in ((0, walls[0]), (0, walls[2]), (1, walls[1]), (1, walls[3])):
                component = rays[:, axis]
                distance = np.divide(
                    boundary - pose[axis],
                    component,
                    out=np.full(len(rays), np.inf),
                    where=np.abs(component) > 1e-10,
                )
                other = pose[1 - axis] + distance * rays[:, 1 - axis]
                lower, upper = (walls[1], walls[3]) if axis == 0 else (walls[0], walls[2])
                altitude = camera_height_m + distance * np.tan(pitch)
                hit = (
                    (distance > 0)
                    & (distance < nearest)
                    & (other >= lower)
                    & (other <= upper)
                    & (altitude >= 0)
                    & (altitude <= 1.0)
                )
                nearest[hit] = distance[hit]
                samples[sample_index, hit] = wall_intensity
            for disk in disks:
                displacement = np.asarray([disk["x"], disk["y"]]) - pose[:2]
                along = rays @ displacement
                discriminant = disk["radius"] ** 2 - (displacement @ displacement - along**2)
                distance = along - np.sqrt(np.maximum(discriminant, 0))
                altitude = camera_height_m + distance * np.tan(pitch)
                hit = (
                    (discriminant >= 0)
                    & (distance > 0)
                    & (distance < nearest)
                    & (altitude >= 0)
                    & (altitude <= disk.get("height", 0.4))
                )
                nearest[hit] = distance[hit]
                samples[sample_index, hit] = disk.get("intensity", 0.9)
        count = admissible.sum(axis=0)
        result = np.divide(
            (samples * admissible).sum(axis=0),
            count,
            out=np.full(len(directions), background),
            where=count > 0,
        )
        return np.clip(result, 0, 1).astype(np.float32)


def sample_image(
    image: np.ndarray,
    contract: SensorContract | None = None,
    eye: int = 0,
    kernel: str = "subrays",
) -> np.ndarray:
    """Convenience API; retain a HexEye instance for repeated camera frames."""
    return HexEye(contract).sample_image(image, eye, kernel)
