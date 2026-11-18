"""Where the v1compat world gets its random numbers (#132, plan 3.3 parity).

The world asks for draws through one small interface, and two classes answer it:

- ``HashDraws``: the production source. Each draw is a counter-based hash of (episode seed,
  stream, counter), so it costs no state, needs no host sync and does not depend on the
  agent's actions or on other rows.
- ``V1Draws``: the parity source. It replays the numpy streams v1's FlyTrackEnv consumes for
  the same episode seed (layout, trajectory phase, domain randomisation, wander path, puff
  turbulence, sensor corruption) in v1's order. With it, v1compat and v1 see the same random
  numbers, so a fixed action sequence can be compared step by step within tolerance. It
  loops over rows on the host and is meant for tests only.

Every method returns standard draws (uniform values already mapped to v1's ranges, or standard
normals that the world scales), so both sources feed the same world code.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch
from flytrack_core.world.layout import LAYOUT_STREAM, LayoutConfig
from torch import Tensor

from flytrack_v2.world import rng

# Hash stream identifiers.
_LAYOUT, _PHASE, _DOMAIN, _WANDER, _PLUME, _ODOR = 1, 2, 3, 4, 5, 8
PLUME_STREAM = _PLUME  # kernels.plume_tick continues this key chain
_VISION = (6, 7)
_WANDER_V1_STREAM = 0x5EED  # flytrack_core.world.trajectories


@dataclass(frozen=True)
class LayoutRanges:
    """The layout and domain-randomisation ranges the draws are mapped to."""

    layout: LayoutConfig
    motions: tuple[str, ...]
    center: tuple[float, float]
    domain_randomization: bool


class HashDraws:
    def __init__(self, ranges: LayoutRanges):
        self.ranges = ranges

    def episode(self, mask: Tensor, seeds: Tensor) -> dict[str, Tensor]:
        r, layout = self.ranges, self.ranges.layout
        index = torch.arange(8, device=seeds.device)[None]
        layout_u = rng.draw(seeds, _LAYOUT, index, 0, False)  # [B,8]
        other = rng.draw(seeds, _PHASE, index[:, :4], 0, False)  # phase, domain randomisation

        def u(column: int) -> Tensor:
            return layout_u[:, column]

        values = {
            "direction": torch.where(u(0) < 0.5, -1.0, 1.0)
            if layout.randomize_target_direction
            else torch.ones_like(seeds, dtype=torch.float64),
            "lateral_m": (2 * u(1) - 1) * layout.start_lateral_offset_m,
            "yaw_offset_rad": torch.deg2rad((2 * u(2) - 1) * layout.start_heading_offset_deg),
            "motion": (u(3) * len(r.motions)).long().clamp(max=len(r.motions) - 1),
            "start_x": r.center[0] + (2 * u(4) - 1) * layout.start_region_half_m,
            "start_y": r.center[1] + (2 * u(5) - 1) * layout.start_region_half_m,
            "heading_rad": (2 * u(6) - 1) * math.pi,
            "radius_m": layout.curve_radius_min_m
            + u(7) * (layout.curve_radius_max_m - layout.curve_radius_min_m),
            "phase_rad": (2 * other[:, 0] - 1) * math.pi,
        }
        values.update(_domain(r.domain_randomization, lambda i: other[:, 1 + i], seeds))
        return values

    def wander(self, mask: Tensor, seeds: Tensor, index: Tensor) -> Tensor:
        """Standard normals for wander path steps ``index`` [B,K]."""
        return rng.draw(seeds, _WANDER, index, 0, True)

    def turbulence(self, mask: Tensor, seeds: Tensor, tick: Tensor, sequence: Tensor) -> Tensor:
        """[B,C,2] normals: key(seed, stream, tick, puff number), then one sub-key per axis
        (kernels.plume_tick draws the same numbers inside the fused tick)."""
        k = rng.key(seeds[:, None, None], _PLUME, tick[:, None, None], sequence[..., None])
        k = rng.key(k, torch.arange(2, device=seeds.device))
        return rng.normal(k)

    def vision(self, mask, seeds, eye, event, blackout, dropout, brightness_std, noise_std):
        """[B,E] samples: (dropped, brightness normal). Per-row parameters are [B]."""
        p = dropout[:, None]
        u = rng.draw(seeds, _VISION[eye], event, 0, False)
        dropped = mask & ~blackout & (p > 0) & (u < p)
        return dropped, rng.draw(seeds, _VISION[eye], event, 1, True)

    def vision_noise(self, seeds, eye, event, size: int) -> Tensor:
        sub = 2 + torch.arange(size, device=seeds.device)
        return rng.draw(seeds, _VISION[eye], event[:, None], sub, True, fast=True).to(torch.float32)

    def odor(self, mask, seeds, event, dropout, noise_std):
        """[B,E] samples: (dropped, [B,E,2] noise normals)."""
        p = dropout[:, None]
        dropped = mask & (p > 0) & (rng.draw(seeds, _ODOR, event, 0, False) < p)
        sub = 1 + torch.arange(2, device=seeds.device)
        return dropped, rng.draw(seeds, _ODOR, event[..., None], sub, True)


def _domain(enabled: bool, uniform, like: Tensor) -> dict[str, Tensor]:
    if not enabled:
        ones = torch.ones_like(like, dtype=torch.float64)
        return {"wall_intensity": 0.45 * ones, "target_intensity": 0.9 * ones, "visual_scale": ones}
    return {
        "wall_intensity": 0.25 + 0.4 * uniform(0),
        "target_intensity": 0.75 + 0.25 * uniform(1),
        "visual_scale": 0.8 + 0.4 * uniform(2),
    }


class V1Draws:
    """v1's numpy streams, per row, in v1's consumption order (tests only)."""

    def __init__(self, ranges: LayoutRanges, num_envs: int, noise_size: int):
        self.ranges = ranges
        self.noise_size = noise_size
        self._rows: list[dict] = [{} for _ in range(num_envs)]

    def _row(self, b: int) -> dict:
        return self._rows[b]

    def episode(self, mask: Tensor, seeds: Tensor) -> dict[str, Tensor]:
        r, layout = self.ranges, self.ranges.layout
        names = ("direction", "lateral_m", "yaw_offset_rad", "motion", "start_x", "start_y")
        names += ("heading_rad", "radius_m", "phase_rad", "wall_intensity", "target_intensity")
        names += ("visual_scale",)
        out = {name: np.zeros(len(seeds)) for name in names}
        for b in torch.nonzero(mask).flatten().tolist():
            seed = int(seeds[b])
            # flytrack_core.world.layout.sample_episode_layout, draw for draw.
            g = np.random.default_rng([seed, LAYOUT_STREAM])
            out["direction"][b] = (
                float(g.choice((-1.0, 1.0))) if layout.randomize_target_direction else 1.0
            )
            out["lateral_m"][b] = g.uniform(-1, 1) * layout.start_lateral_offset_m
            out["yaw_offset_rad"][b] = np.deg2rad(
                g.uniform(-1, 1) * layout.start_heading_offset_deg
            )
            motion = int(g.integers(len(r.motions))) if len(r.motions) > 1 else 0
            out["motion"][b] = motion
            out["radius_m"][b] = math.nan
            if layout.randomize_target_layout:
                half = layout.start_region_half_m
                start = np.asarray(r.center) + g.uniform(-half, half, size=2)
                out["start_x"][b], out["start_y"][b] = start
                out["heading_rad"][b] = float(g.uniform(-np.pi, np.pi))
                if r.motions[motion] == "curved":
                    out["radius_m"][b] = float(
                        g.uniform(layout.curve_radius_min_m, layout.curve_radius_max_m)
                    )
            # TargetTrajectory's phase, and FlyTrackEnv.reset(seed=...)'s np_random, which
            # gymnasium seeds as default_rng(seed): both start from the same stream.
            out["phase_rad"][b] = np.random.default_rng(seed).uniform(-np.pi, np.pi)
            if r.domain_randomization:
                g = np.random.default_rng(seed)
                out["wall_intensity"][b] = float(g.uniform(0.25, 0.65))
                out["target_intensity"][b] = float(g.uniform(0.75, 1.0))
                out["visual_scale"][b] = float(g.uniform(0.8, 1.2))
            else:
                out["wall_intensity"][b], out["target_intensity"][b] = 0.45, 0.9
                out["visual_scale"][b] = 1.0
            children = np.random.SeedSequence(seed).spawn(3)
            self._rows[b] = {
                "wander": np.random.default_rng([seed, _WANDER_V1_STREAM]),
                "plume": np.random.default_rng(seed),
                "vision": [np.random.default_rng(children[0]), np.random.default_rng(children[1])],
                "odor": np.random.default_rng(children[2]),
                "noise": [{}, {}],
            }
        device = seeds.device
        result = {name: torch.as_tensor(value, device=device) for name, value in out.items()}
        result["motion"] = result["motion"].long()
        return result

    def wander(self, mask: Tensor, seeds: Tensor, index: Tensor) -> Tensor:
        """The wander stream is one sequence per episode: draw number m is the m-th normal."""
        out = np.zeros(tuple(index.shape))
        has, where = mask.cpu().numpy(), index.cpu().numpy()
        for b in np.flatnonzero(has.any(1)).tolist():
            row = self._row(b)
            cache = row.setdefault("wander_draws", [])
            for k in np.flatnonzero(has[b]).tolist():
                m = int(where[b, k])
                while len(cache) <= m:
                    cache.append(row["wander"].standard_normal())
                out[b, k] = cache[m]
        return torch.as_tensor(out, device=seeds.device)

    def turbulence(self, mask: Tensor, seeds: Tensor, tick: Tensor, sequence: Tensor) -> Tensor:
        out = np.zeros((*sequence.shape, 2))
        seq = sequence.cpu().numpy()
        for b in torch.nonzero(mask).flatten().tolist():
            slots = np.flatnonzero(seq[b] >= 0)
            if len(slots):
                order = slots[np.argsort(seq[b, slots])]  # v1 keeps puffs in emission order
                out[b, order] = self._row(b)["plume"].normal(0, 1.0, (len(order), 2))
        return torch.as_tensor(out, device=seeds.device)

    def vision(self, mask, seeds, eye, event, blackout, dropout, brightness_std, noise_std):
        dropped = np.zeros(tuple(mask.shape), bool)
        brightness = np.zeros(tuple(mask.shape))
        has = mask.cpu().numpy()
        for e in range(has.shape[1]):  # samples in time order
            for b in np.flatnonzero(has[:, e]).tolist():
                if bool(blackout[b, e]):
                    continue  # SensorCorruptor.image returns before drawing
                g = self._row(b)["vision"][eye]
                p = float(dropout[b])
                if p > 0 and g.random() < p:
                    dropped[b, e] = True
                    continue
                if float(brightness_std[b]):
                    brightness[b, e] = g.normal(0, 1.0)
                if float(noise_std[b]):
                    noise = g.normal(0, 1.0, self.noise_size)
                    self._row(b)["noise"][eye][int(event[b, e])] = noise
        device = seeds.device
        return torch.as_tensor(dropped, device=device), torch.as_tensor(brightness, device=device)

    def vision_noise(self, seeds, eye, event, size: int) -> Tensor:
        out = np.zeros((len(seeds), size), np.float32)
        for b in range(len(seeds)):
            stored = self._rows[b].get("noise", [{}, {}])[eye].pop(int(event[b]), None)
            if stored is not None:
                out[b] = stored
            self._rows[b].get("noise", [{}, {}])[eye].clear()
        return torch.as_tensor(out, device=seeds.device)

    def odor(self, mask, seeds, event, dropout, noise_std):
        dropped = np.zeros(tuple(mask.shape), bool)
        noise = np.zeros((*mask.shape, 2))
        has = mask.cpu().numpy()
        for e in range(has.shape[1]):
            for b in np.flatnonzero(has[:, e]).tolist():
                g = self._row(b)["odor"]
                p = float(dropout[b])
                if p > 0 and g.random() < p:
                    dropped[b, e] = True
                    continue
                if float(noise_std[b]):
                    noise[b, e] = g.normal(0, 1.0, 2)
        device = seeds.device
        return torch.as_tensor(dropped, device=device), torch.as_tensor(noise, device=device)
