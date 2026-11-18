"""The wind world: wind field, filament plume and antenna sense (WP2b, #133; #161 C1)."""

import math
from pathlib import Path

import pytest
import torch
import yaml
from flytrack_v2.contracts import SensorContractV2, validate_obs
from flytrack_v2.world.v1compat import V1CompatWorld
from flytrack_v2.world.wind import WindConfig, antenna, upwind_angle

ROOT = Path(__file__).resolve().parents[3]


@pytest.fixture(scope="module")
def contract():
    return SensorContractV2.from_json(ROOT / "configs/v2/sensor_contract.json")


def _configs():
    return {
        name: yaml.safe_load((ROOT / f"configs/scenarios/{name}.yaml").read_text())
        for name in ("normal", "occlusion", "noise", "conflict")
    }


def _world(contract, wind: WindConfig, num_envs=4, scenario="normal", device="cpu", **kw):
    return V1CompatWorld(
        contract, num_envs, scenario, {"episode_seconds": 60, "target_motion": "curved"},
        seed=7, device=device, dtype=torch.float64, scenario_configs=_configs(), wind=wind, **kw,
    )  # fmt: skip


def test_antenna_four_directions_and_mirror_symmetry(contract):
    """#161 C1: left = k cos(theta - alpha), right = k cos(theta + alpha), theta where the wind
    comes from (left positive). Front: equal; left: left > right; behind: equal, negative."""
    k = 0.5 / contract.wind_reference_m_s  # U = 0.5
    c45 = math.cos(math.radians(45))
    heading = torch.zeros(4, dtype=torch.float64)
    # The wind blows *to* the opposite of where it comes from.
    comes_from = torch.tensor([0.0, math.pi / 2, -math.pi / 2, math.pi], dtype=torch.float64)
    wind = -0.5 * torch.stack((torch.cos(comes_from), torch.sin(comes_from)), -1)
    reading = antenna(wind, heading, contract.antenna_angle_deg, contract.wind_reference_m_s)
    expected = torch.tensor(
        [[c45, c45], [c45, -c45], [-c45, c45], [-c45, -c45]], dtype=torch.float64
    )
    torch.testing.assert_close(reading, k * expected)
    # Mirror the wind across the heading: the readings swap sides.
    angles = torch.linspace(-3, 3, 61, dtype=torch.float64)
    blow = torch.stack((torch.cos(angles), torch.sin(angles)), -1) * 0.4
    mirrored = blow * torch.tensor([1.0, -1.0], dtype=torch.float64)
    zero = torch.zeros(61, dtype=torch.float64)
    a = antenna(blow, zero, contract.antenna_angle_deg, contract.wind_reference_m_s)
    b = antenna(mirrored, zero, contract.antenna_angle_deg, contract.wind_reference_m_s)
    torch.testing.assert_close(a, b.flip(-1))
    # The reading determines theta again.
    theta = torch.remainder(angles + math.pi + math.pi, 2 * math.pi) - math.pi  # comes from
    torch.testing.assert_close(upwind_angle(a, contract.antenna_angle_deg), theta)


def test_world_observes_the_antennae(contract):
    world = _world(contract, WindConfig(antenna_noise_std=0.0))
    obs = world.reset()
    validate_obs(obs, contract, check_values=False)
    assert torch.all(obs["valid"][:, 3] == 1)
    for _ in range(7):
        result = world.step(torch.tensor([[0.2, 0.4]], dtype=torch.float32).expand(4, 2))
    p = result.final_privileged
    expected = antenna(
        p["wind_velocity"], p["agent_pose"][:, 2],
        contract.antenna_angle_deg, contract.wind_reference_m_s,
    )  # fmt: skip
    # The last antenna sample is at most one odor period old: compare within its change.
    torch.testing.assert_close(result.final_obs["wind"].double(), expected, rtol=0, atol=0.05)
    assert torch.allclose(
        p["wind_velocity"].norm(dim=-1), torch.full((4,), 0.5, dtype=torch.float64)
    )
    # v1compat has no wind sense.
    plain = V1CompatWorld(contract, 2, "normal", {"episode_seconds": 60})
    obs = plain.reset()
    assert torch.all(obs["valid"][:, 3] == 0) and torch.all(obs["wind"] == 0)
    assert torch.all(plain.privileged()["wind_velocity"] == 0)


def test_filaments_drift_with_the_wind(contract):
    """No turbulence: every held filament moves by exactly wind * dt per plume tick."""
    world = _world(contract, WindConfig(turbulent_diffusion_m2_s=0.0, direction_std_deg=0.0))
    world.reset()
    for _ in range(20):
        world.step(torch.zeros(4, 2))
    s = world._state
    before, seq0, tick0 = s["puff_xy"].clone(), s["puff_seq"].clone(), s["tick"].clone()
    world.step(torch.zeros(4, 2))
    ticks = (s["tick"] - tick0).to(torch.float64)
    kept = (seq0 >= 0) & (s["puff_seq"] == seq0)
    moved = s["puff_xy"] - before
    expected = s["wind"][:, None] * (ticks * world.plume.integration_dt)[:, None, None]
    assert bool(kept.any())
    torch.testing.assert_close(moved[kept], expected.expand_as(moved)[kept])
    # The wind blows the plume away from the target: downwind of it.
    held = s["puff_seq"] >= 0
    offset = s["puff_xy"] - s["target"][:, None, :2]
    along = (offset * s["wind"][:, None]).sum(-1)
    assert bool(((along > 0) | ~held).float().mean(1).gt(0.9).all())


def test_concentration_mass_is_conserved_without_decay(contract):
    """Each filament is a 2D Gaussian of mass puff_mass: the field integrates to n * mass."""
    config = WindConfig(decay_tau_s=math.inf)
    world = _world(contract, config, num_envs=1)
    world.reset()
    for _ in range(40):  # 2 s: about 20 filaments, none expired
        world.step(torch.zeros(1, 2))
    s = world._state
    held = s["puff_seq"] >= 0
    count = int(held.sum())
    xy = s["puff_xy"][held]
    low, high = xy.min(0).values - 1.5, xy.max(0).values + 1.5
    step = 0.01
    gx = torch.arange(float(low[0]), float(high[0]), step, dtype=torch.float64)
    gy = torch.arange(float(low[1]), float(high[1]), step, dtype=torch.float64)
    grid = torch.stack(torch.meshgrid(gx, gy, indexing="ij"), -1).reshape(-1, 2)
    total = 0.0
    for chunk in grid.split(4096):
        points = chunk.reshape(1, -1, 2)  # [1, P, 2] as "sensor points"
        terms = world._concentration({**s, "puff_xy": s["puff_xy"]}, points)
        total += float(terms.double().sum()) * step * step
    assert count >= 15
    assert total == pytest.approx(count * config.puff_mass, rel=1e-3)


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="the fused kernels need CUDA")
def test_wind_world_fused_matches_eager_on_cuda(contract):
    worlds = [
        V1CompatWorld(
            contract,
            48,
            ["normal", "occlusion", "noise", "conflict"] * 12,
            {"episode_seconds": 3, "target_motion": ["curved", "wander"]},
            seed=5,
            device="cuda",
            scenario_configs=_configs(),
            wind=WindConfig(),
            **kw,
        )  # fmt: skip
        for kw in ({}, {"fused": False})
    ]
    for world in worlds:
        world.reset()
    generator = torch.Generator().manual_seed(1)
    for _ in range(70):
        action = torch.rand(48, 2, generator=generator) * torch.tensor([0.5, 3.0]) - torch.tensor(
            [0, 1.5]
        )
        fast, slow = (world.step(action) for world in worlds)
        torch.testing.assert_close(
            fast.final_obs["wind"], slow.final_obs["wind"], rtol=0, atol=1e-5
        )
        torch.testing.assert_close(fast.final_obs["valid"], slow.final_obs["valid"])
        torch.testing.assert_close(
            fast.final_obs["odor"], slow.final_obs["odor"], rtol=1e-4, atol=1e-6
        )
        torch.testing.assert_close(
            fast.final_privileged["wind_velocity"], slow.final_privileged["wind_velocity"]
        )
