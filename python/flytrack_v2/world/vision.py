"""Batched ray casting of the binocular hexagonal retina (#132, plan 3.3).

A vectorised port of v1's ``HexEye.raycast``: every environment, eye, quadrature subray and
ommatidium is one ray, intersected with the boundary walls and the extruded disks (target,
decoys, obstacles). The geometry, the frustum admissibility test, the hit order and the mean
over admissible subrays follow v1, so the result matches it within float tolerance; there is
no bitwise claim (#131).

Two paths compute the same thing:

- ``eager``: plain torch ops, any device. It is the reference for the tests.
- ``fused`` (CUDA): one jiterator kernel for the walls and one per disk, per subray. Jiterator
  compiles with NVRTC, which ships with torch, so no host C compiler is needed (torch.compile's
  Triton backend needs one). At 1,024 environments it is several times faster than eager.

Ties between surfaces at the same distance go to the earlier one in v1's loop order: v1 only
overwrites a hit with a strictly closer one, and both paths test ``distance < nearest`` in the
same order (walls, then disks in order).
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import torch
from flytrack_core.contracts import SensorContract
from flytrack_core.sensing.hex_eye import DEFAULT_SUBRAYS, compute_directions, subray_offsets
from torch import Tensor


@dataclass(frozen=True)
class CameraGeometry:
    """v1 ``raycast`` keyword defaults; FlyTrackEnv passes the same values."""

    height_m: float = 0.27
    forward_m: float = 0.14
    baseline_m: float = 0.12
    background: float = 0.15
    wall_height_m: float = 1.0


_WALL_KERNEL = """
template <typename T> void flytrack_wall(T yaw, T tan_pitch, T heading, T px, T py, T wall_value,
    T xmin, T ymin, T xmax, T ymax, T camera_h, T wall_h, T background, T& nearest, T& value) {
  T a = yaw + heading;
  T rx = cos(a);
  T ry = sin(a);
  T inf = INFINITY;
  T dx = fabs(rx) > T(1e-10) ? ((rx > T(0) ? xmax : xmin) - px) / rx : inf;
  T dy = fabs(ry) > T(1e-10) ? ((ry > T(0) ? ymax : ymin) - py) / ry : inf;
  T d = dx < dy ? dx : dy;
  T alt = camera_h + d * tan_pitch;
  bool hit = alt >= T(0) && alt <= wall_h;
  nearest = hit ? d : inf;
  value = hit ? wall_value : background;
}
"""

_DISK_KERNEL = """
template <typename T> void flytrack_disk(T yaw, T tan_pitch, T heading, T nearest_in, T dx, T dy,
    T radius_sq, T height, T camera_h, T& nearest, T& hit) {
  T a = yaw + heading;
  T rx = cos(a);
  T ry = sin(a);
  T along = rx * dx + ry * dy;
  T disc = radius_sq - ((dx * dx + dy * dy) - along * along);
  T d = along - sqrt(disc > T(0) ? disc : T(0));
  T alt = camera_h + d * tan_pitch;
  bool h = disc >= T(0) && d > T(0) && d < nearest_in && alt >= T(0) && alt <= height;
  nearest = h ? d : nearest_in;
  hit = h ? T(1) : T(0);
}
"""

_KERNELS: dict[str, object] = {}


def _kernels():
    """Build the jiterator functions once per process (NVRTC compiles on first call)."""
    if not _KERNELS:
        from torch.cuda import jiterator

        _KERNELS["wall"] = jiterator._create_multi_output_jit_fn(
            _WALL_KERNEL,
            num_outputs=2,
            xmin=0.0,
            ymin=0.0,
            xmax=0.0,
            ymax=0.0,
            camera_h=0.0,
            wall_h=0.0,
            background=0.0,
        )
        _KERNELS["disk"] = jiterator._create_multi_output_jit_fn(
            _DISK_KERNEL, num_outputs=2, camera_h=0.0
        )
    return _KERNELS["wall"], _KERNELS["disk"]


class BatchedRetina:
    """Precomputed ray tables for both eyes; ``render`` casts a whole batch.

    The tables are [2 eyes, S subrays, N ommatidia]: yaw relative to the heading, tan(pitch),
    and each subray's weight, 1/count over the subrays inside the camera frustum (v1's
    admissibility test) and 0 outside it. An ommatidium with no admissible subray shows the
    background, as in v1.

    The arena walls (xmin, ymin, xmax, ymax) are fixed per retina: every environment of a world
    shares one arena, and the fused kernel takes them as scalars.
    """

    def __init__(
        self,
        contract: SensorContract,
        walls: Sequence[float],
        device: torch.device | str = "cpu",
        dtype: torch.dtype = torch.float32,
        subrays: int = DEFAULT_SUBRAYS,
        camera: CameraGeometry | None = None,
        path: str = "auto",
    ):
        if path not in ("auto", "eager", "fused"):
            raise ValueError("path must be auto, eager or fused")
        self.contract = contract
        self.walls = tuple(float(value) for value in walls)
        if len(self.walls) != 4 or not (
            self.walls[0] < self.walls[2] and self.walls[1] < self.walls[3]
        ):
            raise ValueError("walls must be xmin, ymin, xmax, ymax with min < max")
        self.device = torch.device(device)
        self.dtype = dtype
        self.subrays = subrays
        self.camera = camera or CameraGeometry()
        if path == "auto":
            path = "fused" if self.device.type == "cuda" else "eager"
        if path == "fused" and self.device.type != "cuda":
            raise ValueError("the fused path needs CUDA")
        self.path = path
        directions = torch.as_tensor(compute_directions(contract), dtype=torch.float64)  # [2,N,2]
        offsets = torch.as_tensor(
            subray_offsets(math.radians(contract.acceptance_angle_deg), subrays),
            dtype=torch.float64,
        )  # [S,2]
        yaw = directions[:, None, :, 0] + offsets[None, :, None, 0]  # [2,S,N]
        pitch = directions[:, None, :, 1] + offsets[None, :, None, 1]
        optical = math.radians(contract.eye_yaw_deg) * torch.tensor(
            [1.0, -1.0], dtype=torch.float64
        )
        relative = yaw - optical[:, None, None]
        half_hfov = math.radians(contract.horizontal_fov_deg / 2)
        tan_half_vfov = math.tan(math.radians(contract.vertical_fov_deg / 2))
        admissible = (relative.abs() <= half_hfov + 1e-10) & (
            (torch.tan(pitch) / torch.cos(relative)).abs() <= tan_half_vfov + 1e-10
        )
        count = admissible.sum(1)  # [2,N]
        weight = admissible / count.clamp(min=1)[:, None, :]
        # Subray-major copies so each subray's slice is contiguous.
        self.ray_yaw = yaw.transpose(0, 1).contiguous().to(self.device, dtype)  # [S,2,N]
        self.ray_tan_pitch = torch.tan(pitch).transpose(0, 1).contiguous().to(self.device, dtype)
        self.ray_weight = weight.transpose(0, 1).contiguous().to(self.device, dtype)
        self.empty = (count == 0).to(self.device)
        # Eye offset in the body frame: forward, then half the baseline left (+) or right (-).
        self.eye_side = torch.tensor([1.0, -1.0], device=self.device, dtype=dtype)
        # Device constants built once, so render() makes no host-to-device copy and can be
        # captured in a CUDA graph.
        self._inf = torch.tensor(math.inf, device=self.device, dtype=dtype)
        self._walls = torch.tensor(self.walls, device=self.device, dtype=dtype)

    @property
    def n_ommatidia(self) -> int:
        return int(self.ray_yaw.shape[-1])

    def eye_positions(self, agent_pose: Tensor) -> Tensor:
        """[B,2 eyes,2] camera centres in the world frame."""
        pose = agent_pose.to(self.dtype)
        cos, sin = torch.cos(pose[:, 2]), torch.sin(pose[:, 2])
        forward = self.camera.forward_m * torch.stack((cos, sin), -1)  # [B,2]
        left = torch.stack((-sin, cos), -1)  # [B,2]
        lateral = self.camera.baseline_m / 2 * self.eye_side[None, :, None] * left[:, None, :]
        return pose[:, None, :2] + forward[:, None, :] + lateral

    def render(self, agent_pose: Tensor, wall_intensity: Tensor, disks: Tensor) -> Tensor:
        """Return [B,2,N] float32 luminance in [0, 1].

        - agent_pose: [B,3] x, y, yaw. The cameras must be inside the arena; the world keeps
          them there (a collision stops the body 0.18 m from a wall, the cameras sit within
          0.15 m of the body centre).
        - wall_intensity: [B].
        - disks: [B,D,5] as x, y, radius, height, intensity. Pad unused slots with a negative
          height: v1's altitude test (0 <= altitude <= height) then never passes.
        """
        dtype = self.dtype
        eye = self.eye_positions(agent_pose)  # [B,2,2]
        px, py = eye[..., 0, None], eye[..., 1, None]  # [B,2,1]
        heading = agent_pose[:, 2].to(dtype)[:, None, None]  # [B,1,1]
        disks = disks.to(dtype)
        wall_value = wall_intensity.to(dtype)[:, None, None]
        # Per-disk terms that do not depend on the ray: [B,2,1] each.
        dx = [disks[:, None, k, 0, None] - px for k in range(disks.shape[1])]
        dy = [disks[:, None, k, 1, None] - py for k in range(disks.shape[1])]
        radius_sq = [disks[:, None, k, 2, None].square() for k in range(disks.shape[1])]
        height = [disks[:, None, k, 3, None] for k in range(disks.shape[1])]
        intensity = [disks[:, None, k, 4, None] for k in range(disks.shape[1])]
        total = torch.zeros(
            agent_pose.shape[0], 2, self.n_ommatidia, device=agent_pose.device, dtype=dtype
        )
        # One subray at a time keeps every temporary at [B,2,N], small enough to stay in cache.
        for s in range(self.subrays):
            yaw, tan_pitch = self.ray_yaw[s][None], self.ray_tan_pitch[s][None]  # [1,2,N]
            if self.path == "fused":
                nearest, value = self._fused_walls(yaw, tan_pitch, heading, px, py, wall_value)
                disk_kernel = _kernels()[1]
                for k in range(disks.shape[1]):
                    nearest, hit = disk_kernel(
                        yaw,
                        tan_pitch,
                        heading,
                        nearest,
                        dx[k],
                        dy[k],
                        radius_sq[k],
                        height[k],
                        camera_h=self.camera.height_m,
                    )
                    value = torch.where(hit > 0, intensity[k], value)
            else:
                nearest, value = self._eager_walls(yaw, tan_pitch, heading, px, py, wall_value)
                for k in range(disks.shape[1]):
                    nearest, value = self._eager_disk(
                        yaw, tan_pitch, heading, nearest, value, dx[k], dy[k], radius_sq[k],
                        height[k], intensity[k],
                    )  # fmt: skip
            total.addcmul_(value, self.ray_weight[s][None])
        total = torch.where(self.empty[None], self.camera.background, total)
        return total.clamp(0, 1).to(torch.float32)

    def _fused_walls(self, yaw, tan_pitch, heading, px, py, wall_value):
        xmin, ymin, xmax, ymax = self.walls
        return _kernels()[0](
            yaw,
            tan_pitch,
            heading,
            px,
            py,
            wall_value,
            xmin=xmin,
            ymin=ymin,
            xmax=xmax,
            ymax=ymax,
            camera_h=self.camera.height_m,
            wall_h=self.camera.wall_height_m,
            background=self.camera.background,
        )

    def _eager_walls(self, yaw, tan_pitch, heading, px, py, wall_value):
        """First wall along each ray, if its height range is in view.

        The cameras sit inside the arena, so the first wall a ray meets is the one it leaves
        through: x max or x min by the sign of rx, y max or y min by the sign of ry. v1 tests
        all four planes, but the other three intersect beyond the arena edge and fail its
        bounds check; all walls share one intensity, so a corner tie changes nothing.
        """
        absolute = yaw + heading
        rx, ry = torch.cos(absolute), torch.sin(absolute)
        w = self._walls
        ux, uy = rx.abs() > 1e-10, ry.abs() > 1e-10
        to_x = torch.where(rx > 0, w[2], w[0]) - px
        to_y = torch.where(ry > 0, w[3], w[1]) - py
        nearest = torch.minimum(
            torch.where(ux, to_x / torch.where(ux, rx, 1), self._inf),
            torch.where(uy, to_y / torch.where(uy, ry, 1), self._inf),
        )
        altitude = self.camera.height_m + nearest * tan_pitch
        hit = (altitude >= 0) & (altitude <= self.camera.wall_height_m)
        return torch.where(hit, nearest, self._inf), torch.where(
            hit, wall_value, self.camera.background
        )

    def _eager_disk(
        self, yaw, tan_pitch, heading, nearest, value, dx, dy, radius_sq, height, intensity
    ):
        absolute = yaw + heading
        rx, ry = torch.cos(absolute), torch.sin(absolute)
        along = rx * dx + ry * dy
        discriminant = radius_sq - ((dx.square() + dy.square()) - along.square())
        distance = along - torch.sqrt(discriminant.clamp(min=0))
        altitude = self.camera.height_m + distance * tan_pitch
        hit = (
            (discriminant >= 0)
            & (distance > 0)
            & (distance < nearest)
            & (altitude >= 0)
            & (altitude <= height)
        )
        return torch.where(hit, distance, nearest), torch.where(hit, intensity, value)
