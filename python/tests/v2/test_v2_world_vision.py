"""The batched v2 retina matches v1's HexEye.raycast within tolerance (WP2a, #132)."""

from pathlib import Path

import numpy as np
import pytest
import torch
from flytrack_core.contracts import SensorContract
from flytrack_core.sensing import HexEye
from flytrack_v2.world import BatchedRetina

ROOT = Path(__file__).resolve().parents[3]
WALLS = (-5.0, -5.0, 5.0, 5.0)
TOLERANCE = 1e-4  # plan 3.3: fixed pose set, tolerance, not bits
# float32 rounding can move a subray that grazes an edge (wall foot, disk rim) to the other
# side, changing that ommatidium by one subray's share of a contrast step (at most 1/7 here).
# float64 meets TOLERANCE everywhere; float32 may do this on a few values in a million.
SUBRAY_SHARE = 1 / 7
FLIP_FRACTION = 1e-5


def _assert_close(out, reference, dtype):
    error = np.abs(out - reference)
    if dtype == torch.float64:
        np.testing.assert_allclose(out, reference, rtol=0, atol=TOLERANCE)
        return
    assert error.max() <= SUBRAY_SHARE + TOLERANCE
    assert (error > TOLERANCE).mean() <= FLIP_FRACTION


@pytest.fixture(scope="module")
def contract():
    return SensorContract.from_json(ROOT / "configs/sensor_contract.json")


def _pose_set(count: int, seed: int = 0):
    """Poses that stress the geometry: against a wall, in a corner, the target touching the
    body (the collision distance), the target over the cameras, and ordinary scenes. Each has
    the target disk and, on odd rows, an obstacle; even rows pad the obstacle slot."""
    rng = np.random.default_rng(seed)
    poses, disks = [], []
    for k in range(count):
        kind = k % 5
        if kind == 0:
            pose = [rng.choice([-1, 1]) * 4.82, rng.uniform(-4.8, 4.8), rng.uniform(-np.pi, np.pi)]
            if k % 2:
                pose = [pose[1], pose[0], pose[2]]
        elif kind == 1:
            pose = [rng.choice([-1, 1]) * 4.82, rng.choice([-1, 1]) * 4.82, rng.uniform(-3, 3)]
        else:
            pose = [*rng.uniform(-4.5, 4.5, 2), rng.uniform(-np.pi, np.pi)]
        if kind == 2:
            distance, angle = 0.33, pose[2] + rng.uniform(-0.5, 0.5)
        elif kind == 3:
            distance, angle = 0.15, pose[2]
        else:
            distance, angle = rng.uniform(0.4, 6), rng.uniform(-np.pi, np.pi)
        target = [
            pose[0] + distance * np.cos(angle),
            pose[1] + distance * np.sin(angle),
            0.15 * rng.uniform(0.8, 1.2),
            0.5,
            rng.uniform(0.75, 1.0),
        ]
        obstacle = [*rng.uniform(-4, 4, 2), 0.3, 0.4 if k % 2 else -1.0, 0.6]
        poses.append(pose)
        disks.append([obstacle, target])
    return np.asarray(poses), np.asarray(disks), rng.uniform(0.25, 0.65, count)


def _v1(contract, poses, disks, wall_intensity):
    eye = HexEye(contract)
    out = np.zeros((len(poses), 2, contract.n_ommatidia), np.float32)
    for b, pose in enumerate(poses):
        scene = [
            dict(x=d[0], y=d[1], radius=d[2], height=d[3], intensity=d[4])
            for d in disks[b]
            if d[3] >= 0
        ]
        for side in (0, 1):
            out[b, side] = eye.raycast(
                pose, np.asarray(WALLS), scene, side, wall_intensity=wall_intensity[b]
            )
    return out


def _render(contract, device, dtype, path, poses, disks, wall_intensity):
    retina = BatchedRetina(contract, WALLS, device, dtype, path=path)
    tensors = [torch.as_tensor(x, device=device) for x in (poses, wall_intensity, disks)]
    return retina.render(*tensors).cpu().numpy()


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_eager_matches_v1_raycast(contract, dtype):
    poses, disks, wall_intensity = _pose_set(40)
    reference = _v1(contract, poses, disks, wall_intensity)
    out = _render(contract, "cpu", dtype, "eager", poses, disks, wall_intensity)
    assert out.shape == (40, 2, contract.n_ommatidia) and out.dtype == np.float32
    _assert_close(out, reference, dtype)
    # The scenes are not trivial: walls, target and background all appear.
    assert len(np.unique(np.round(reference, 3))) > 50


def test_padded_disk_slot_never_hits(contract):
    poses, disks, wall_intensity = _pose_set(10, seed=3)
    disks[:, 0, 3] = -1.0  # pad every obstacle slot
    padded = _render(contract, "cpu", torch.float32, "eager", poses, disks, wall_intensity)
    alone = _render(contract, "cpu", torch.float32, "eager", poses, disks[:, 1:], wall_intensity)
    np.testing.assert_array_equal(padded, alone)


def test_rejects_bad_walls_and_paths(contract):
    with pytest.raises(ValueError, match="walls"):
        BatchedRetina(contract, (5.0, -5.0, -5.0, 5.0))
    with pytest.raises(ValueError, match="CUDA"):
        BatchedRetina(contract, WALLS, "cpu", path="fused")
    assert BatchedRetina(contract, WALLS, "cpu").path == "eager"


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="the fused path runs on CUDA")
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_fused_matches_eager_and_v1_on_cuda(contract, dtype):
    poses, disks, wall_intensity = _pose_set(200, seed=1)
    reference = _v1(contract, poses, disks, wall_intensity)
    fused = _render(contract, "cuda", dtype, "fused", poses, disks, wall_intensity)
    eager = _render(contract, "cuda", dtype, "eager", poses, disks, wall_intensity)
    _assert_close(fused, reference, dtype)
    # Same arithmetic on the same device: the two paths agree to rounding.
    _assert_close(fused, eager, dtype)


def test_per_eye_poses_render_each_eye_at_its_own_pose(contract):
    poses, disks, wall_intensity = _pose_set(8, seed=5)
    other, other_disks, _ = _pose_set(8, seed=6)
    retina = BatchedRetina(contract, WALLS, "cpu", torch.float64)
    left = retina.render(
        torch.as_tensor(poses), torch.as_tensor(wall_intensity), torch.as_tensor(disks)
    )
    right = retina.render(
        torch.as_tensor(other), torch.as_tensor(wall_intensity), torch.as_tensor(other_disks)
    )
    both = retina.render(
        torch.as_tensor(np.stack((poses, other), 1)),
        torch.as_tensor(wall_intensity),
        torch.as_tensor(np.stack((disks, other_disks), 1)),
    )
    torch.testing.assert_close(both[:, 0], left[:, 0], rtol=0, atol=0)
    torch.testing.assert_close(both[:, 1], right[:, 1], rtol=0, atol=0)


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="the fused path runs on CUDA")
def test_render_target_matches_render_on_cuda(contract):
    """The world's two-kernel target render against the general disk render."""
    poses, disks, wall_intensity = _pose_set(300, seed=8)
    target = torch.as_tensor(disks[:, 1], device="cuda")  # x, y, radius, height 0.5, intensity
    retina = BatchedRetina(contract, WALLS, "cuda", torch.float32)
    pose, light = (
        torch.as_tensor(poses, device="cuda"),
        torch.as_tensor(wall_intensity, device="cuda"),
    )
    fast = retina.render_target(pose, light, target[:, :2], target[:, 2], 0.5, target[:, 4])
    general = retina.render(pose, light, target[:, None])
    _assert_close(fast.cpu().numpy(), general.cpu().numpy(), torch.float32)
