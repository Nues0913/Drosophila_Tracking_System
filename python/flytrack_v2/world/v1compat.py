"""``v1compat``: v1's FlyTrackEnv as one batched world (#132, plan 3.3).

Geometry, target motion (curved, wander), the four scenarios, the puff-trail plume, sensor
timing and corruption, collisions, reward and the success rule follow v1's FlyTrackEnv. The
world reads its parameters by constructing v1's FlyTrackEnv with the same options, so parsing,
defaults and the reward's survival bound are v1's own. Wind is absent: ``wind`` observations
and the wind valid flag are 0.

Sensor timing (interfaces.md): each control step is split at every neural_dt boundary and at
each vision and odor sample time, exactly as v1's loop, with v1's float64 clock arithmetic.
Rows advance in lockstep through a fixed number of substeps; a row that has reached the end of
its step, or collided, is masked. Vision samples taken inside the loop are recorded (pose,
target, time, corruption draws) and only the last valid one per eye is rendered after the
loop: an earlier sample in the same step is overwritten in v1 before anyone reads it.

Randomness comes from a ``draws`` source (draws.py): counter-based hashes in production, or
v1's numpy streams for parity tests. Either way it depends on the episode seed only.
"""

from __future__ import annotations

import math
from dataclasses import fields, replace
from typing import Any

import torch
from flytrack_core.contracts import SensorContract
from flytrack_core.world.plume import make_plume
from flytrack_core.world.trajectories import WANDER_DT_S, TrajectoryConfig
from flytrack_sim.env import FlyTrackEnv
from torch import Tensor

from flytrack_v2.contracts import SUCCESS_RULE, SensorContractV2
from flytrack_v2.interfaces import StepResult
from flytrack_v2.world import kernels, rng
from flytrack_v2.world.draws import HashDraws, LayoutRanges, V1Draws
from flytrack_v2.world.vision import BatchedRetina, CameraGeometry
from flytrack_v2.world.wind import WindConfig, antenna

WIND_STREAM, WIND_SENSE_STREAM, WIND_EPISODE_STREAM = 9, 10, 11  # hash streams (wind world)
MAX_SUBSTEPS = 7  # check_substeps verifies it; v1's schedule uses at most 6
MAX_TICKS = 6  # plume ticks (integration_dt 0.01) per 0.05 s control step, with float slack
MAX_WANDER_STEPS = 7  # wander path points added per control step
PATH_SLOTS = 16  # wander path window
VISION_SLOTS, ODOR_SLOTS = 2, 3  # samples per control step at 30 Hz and 50 Hz
_UNSUPPORTED = (
    "obstacles",
    "initial_agent_pose",
    "occlusion_odor_steering",
    "sensing",
    "render_mode",
    "arena_center_xy",
)
_FIXED_LAYOUT_KEYS = ("radius_m", "direction", "initial_x", "initial_y", "heading_rad", "motion")


def _v1_contract(contract: SensorContractV2) -> SensorContract:
    """The v1 fields of a v2 contract (FlyTrackEnv takes a v1 SensorContract)."""
    names = {field.name for field in fields(SensorContract)}
    return SensorContract(**{k: v for k, v in contract.to_dict().items() if k in names})


def track_v2_reward(
    tracked: Tensor, distance: Tensor, collision: Tensor, change: Tensor, following: float
) -> Tensor:
    """PPO reward aligned with the tracking success rule (plan 3.8, #161 D3).

    Each collision-free step earns at least −0.05 · ‖Δa‖ because the +0.9
    survival bonus covers the maximum distance penalty. A wall collision skips
    that bonus, subtracts 10, and ends the episode, so the collision step always
    earns less than a step spent holding still safely.

    In the fixed trajectories in test_v2_track_v2_reward.py, stopping beats
    hitting the wall on total return and on discounted return at γ = 0.95,
    0.99, and 0.999. Braking incurs an action-change penalty, so this ordering
    can reverse at very small γ, when only the first few steps carry much weight.
    """
    survival = 0.3 * 3.0  # +0.9 per non-collision step, per the #161 D3 Decision.
    error = (distance.to(torch.float64) - following).abs().clamp(max=3.0)
    hit = collision.to(torch.float64)
    return (
        tracked.to(torch.float64)
        + (survival * (1.0 - hit) - 0.3 * error)
        - 10.0 * hit
        - 0.05 * change.to(torch.float64).norm(dim=-1)
    )


def _clone(tree):
    if isinstance(tree, Tensor):
        return tree.clone()
    if isinstance(tree, dict):
        return {name: _clone(value) for name, value in tree.items()}
    return type(tree)(_clone(value) for value in tree)


def wrap(angle: Tensor) -> Tensor:
    return torch.remainder(angle + math.pi, 2 * math.pi) - math.pi


class V1CompatWorld:
    """BatchedWorld (interfaces.py) reproducing v1's FlyTrackEnv.

    - scenarios: one name for every row, or one per row (rows keep their scenario).
    - options: FlyTrackEnv keyword options (configs/train/*.yaml ``env``), e.g.
      episode_seconds, target_motion, domain_randomization, scenario_config, plume_config.
      Options v1compat does not reproduce raise.
    - draws: "hash" (default) or "v1" (parity tests: v1's numpy streams; host loops).
    - reward (#161 D3): "track_v2" (plan 3.8, for PPO) or "v1" (FlyTrackEnv's, for parity).
    - occlusion_view (#161 C2): "hide_target" renders the scene without the target while the
      scenario occludes it; "blackout" is v1's black image (parity tests, and v1 actors that
      were trained on it). valid stays 1 either way: 0 only means a lost reading.
    - wind: None for ``v1compat`` (v1's fixed 0.15 m/s plume drift, no wind sense), or a
      WindConfig for the ``wind`` world (#133, wind.py): a meandering wind, a filament plume
      advected by it, and the antenna sense in ``wind`` and ``valid[:, 3]``.
    - fused, graph: on CUDA with hash draws, the hot loops run as fused kernels (kernels.py)
      and each step replays two CUDA graphs (physics, then the automatic reset when a row
      ended; one host sync decides). fused=False keeps the eager torch reference on CUDA.
    - env_index_offset, episode_seed_namespace (#137): the derived seed of episode k of row i
      is ``episode_seed(seed, i + env_index_offset, k) | episode_seed_namespace``. The offset
      lets several worlds share one global row numbering (e.g. one world per motion); the
      namespace keeps a range of seeds apart (training uses ``2**40``). Explicit
      ``episode_seeds`` are used as given.
    """

    def __init__(
        self,
        contract: SensorContractV2,
        num_envs: int,
        scenarios: str | list[str] = "normal",
        options: dict[str, Any] | None = None,
        seed: int = 0,
        device: torch.device | str = "cpu",
        dtype: torch.dtype = torch.float32,
        draws: str = "hash",
        scenario_configs: dict[str, dict] | None = None,
        check_substeps: bool = False,
        graph: bool = True,
        fused: bool | None = None,
        reward: str = "track_v2",
        occlusion_view: str = "hide_target",
        wind: WindConfig | dict | None = None,
        env_index_offset: int = 0,
        episode_seed_namespace: int = 0,
    ):
        def is_int(value):
            return isinstance(value, int) and not isinstance(value, bool)

        if not is_int(env_index_offset) or env_index_offset < 0:
            raise ValueError("env_index_offset must be a nonnegative int")
        if (
            not is_int(episode_seed_namespace)
            or not 0 <= episode_seed_namespace < 1 << rng.SEED_BITS
        ):
            raise ValueError("episode_seed_namespace must be an int in [0, 2**52)")
        self.env_index_offset = env_index_offset
        self.episode_seed_namespace = episode_seed_namespace
        if reward not in ("track_v2", "v1"):
            raise ValueError("reward must be track_v2 or v1")
        if occlusion_view not in ("hide_target", "blackout"):
            raise ValueError("occlusion_view must be hide_target or blackout")
        self.reward_kind, self.occlusion_view = reward, occlusion_view
        options = dict(options or {})
        for name in _UNSUPPORTED:
            if name in options:
                raise ValueError(f"v1compat does not reproduce the v1 option {name}")
        if options.get("plume_model", "puff") != "puff":
            raise ValueError("v1compat reproduces the puff plume only")
        if set(options.get("trajectory_config", {}) or {}) & set(_FIXED_LAYOUT_KEYS):
            raise ValueError(
                "v1compat samples the layout; fixed trajectory geometry is unsupported"
            )
        self.contract = contract
        self.num_envs = int(num_envs)
        self.device = torch.device(device)
        self.dtype = dtype
        self.seed = int(seed)
        self.check_substeps = check_substeps
        names = [scenarios] * self.num_envs if isinstance(scenarios, str) else list(scenarios)
        if len(names) != self.num_envs:
            raise ValueError("give one scenario, or one per environment")
        self.scenario_names = names
        envs = {}
        for name in sorted(set(names)):
            scenario_options = dict(options)
            file_config = dict((scenario_configs or {}).get(name, {}) or {})
            file_config.pop("name", None)
            scenario_options["scenario_config"] = {
                **file_config,
                **(options.get("scenario_config", {}) or {}),
            }
            envs[name] = FlyTrackEnv(
                contract=_v1_contract(contract), scenario=name, **scenario_options
            )
        reference = envs[names[0]]
        self._reference = reference
        for env in envs.values():
            if env.survival_bonus != reference.survival_bonus:
                raise ValueError("all scenarios of one world must share the reward bound")
        self.walls = tuple(float(v) for v in reference.walls)
        self.episode_seconds = reference.episode_seconds
        self.max_steps = reference.max_steps
        self.motions = reference.motion_choices
        if set(self.motions) - {"curved", "wander"}:
            raise ValueError("v2 target motions are curved and wander")
        trajectory = {
            "speed_m_s": reference.target_speed_m_s,
            "center_x": float(reference.arena_center[0]),
            "center_y": float(reference.arena_center[1]),
            "xmin": self.walls[0] + 0.5,
            "ymin": self.walls[1] + 0.5,
            "xmax": self.walls[2] - 0.5,
            "ymax": self.walls[3] - 0.5,
            **(reference.trajectory_config or {}),
        }
        self.trajectory = TrajectoryConfig.from_dict(
            {
                **trajectory,
                "motion": "wander",
                "initial_x": trajectory["center_x"],
                "initial_y": trajectory["center_y"],
            }
        )
        self.plume = make_plume(reference.plume_model, reference.plume_config, 0).config
        self.wind = WindConfig.from_dict(wind) if isinstance(wind, dict) else wind
        if self.wind is not None:
            if draws != "hash":
                raise ValueError("the wind world has no v1 counterpart: use hash draws")
            if contract.wind_hz != contract.odor_hz:
                raise ValueError(
                    "the wind world samples wind on the odor clock (wind_hz = odor_hz)"
                )
            w = self.wind
            self.plume = replace(
                self.plume, emission_hz=w.emission_hz, lifetime_s=w.lifetime_s,
                initial_sigma_m=w.initial_sigma_m, diffusion_m2_s=w.diffusion_m2_s,
                turbulent_diffusion_m2_s=w.turbulent_diffusion_m2_s, puff_mass=w.puff_mass,
                # PlumeConfig needs a finite decay; exp(-age / 1e30) is exactly 1 in float64.
                decay_tau_s=w.decay_tau_s if math.isfinite(w.decay_tau_s) else 1e30,
                wind_x=0.0, wind_y=0.0,
            )  # fmt: skip
        self.layout = reference.layout
        ranges = LayoutRanges(
            self.layout,
            self.motions,
            (self.trajectory.center_x, self.trajectory.center_y),
            reference.domain_randomization,
        )
        # The fused CUDA kernels (kernels.py) reproduce the hash draws; the eager torch code is
        # the reference and the only path for v1's numpy draws.
        self.fused = self.device.type == "cuda" and draws == "hash" and fused is not False
        if fused and not self.fused:
            raise ValueError("the fused kernels need CUDA and hash draws")
        if draws == "hash":
            self.draws = HashDraws(ranges)
        elif draws == "v1":
            self.draws = V1Draws(ranges, self.num_envs, contract.n_ommatidia)
        else:
            raise ValueError("draws must be hash or v1")
        self.camera = CameraGeometry(
            reference.camera_height_m, reference.camera_forward_m, reference.camera_baseline_m
        )
        self.retina = BatchedRetina(
            contract, self.walls, self.device, dtype, reference.subrays, self.camera,
            path="fused" if self.fused else "eager",
        )  # fmt: skip
        self._env_params(envs)
        # Puff slots: lifetime x emission rate per source, plus margin.
        sources = 2 if "conflict" in names else 1
        per_source = int(self.plume.lifetime_s * self.plume.emission_hz) + 3
        if self.wind is not None:
            # At most `capacity` filaments per source (#192 R6): the conflict decoy gets its own
            # slots, so the target's trail is as long as in normal. The sources emit alternately
            # into one ring per environment, so a ring of sources x per_source slots drops each
            # source's oldest filament.
            per_source = min(per_source, self.wind.capacity)
        self.puff_capacity = sources * per_source
        self.sources = sources
        self.env_index = torch.arange(self.num_envs, device=self.device)
        self._seed_rows = self.env_index + self.env_index_offset
        self._path_slots = torch.arange(PATH_SLOTS, device=self.device)
        self._puff_slots = torch.arange(self.puff_capacity, device=self.device)
        self._event_slots = torch.arange(max(VISION_SLOTS, ODOR_SLOTS), device=self.device)
        self._plume_wind = torch.tensor(
            [self.plume.wind_x, self.plume.wind_y], dtype=torch.float64, device=self.device
        )
        self._substeps = torch.arange(MAX_SUBSTEPS, device=self.device)
        self._false = torch.zeros((), dtype=torch.bool, device=self.device)
        self.reset_render_rows = min(self.num_envs, max(32, self.num_envs // 16))
        self._tick_slots = torch.arange(MAX_TICKS, device=self.device)
        self._wander_slots = torch.arange(MAX_WANDER_STEPS, device=self.device)
        lim = self.limits
        self._action_low = torch.tensor([0.0, -lim.omega_max], device=self.device)
        self._action_high = torch.tensor([lim.v_max, lim.omega_max], device=self.device)
        self._action_delta = contract.control_dt * torch.tensor(
            [lim.linear_accel_max, lim.angular_accel_max], dtype=torch.float64, device=self.device
        )
        self._any_image_noise = bool(self.noise["image_noise_std"].gt(0).any())
        self.graph = self.fused and not check_substeps and graph
        self._graph: torch.cuda.CUDAGraph | None = None
        self._lo = torch.tensor(self.walls[:2], dtype=torch.float64, device=self.device)
        self._hi = torch.tensor(self.walls[2:], dtype=torch.float64, device=self.device)
        self.episode_index = torch.zeros(self.num_envs, dtype=torch.int64, device=self.device)
        self._state: dict[str, Tensor] = {}

    # ----------------------------------------------------------------------------------
    # Parameters

    def _env_params(self, envs: dict[str, FlyTrackEnv]) -> None:
        def column(fn, dtype=torch.float64):
            return torch.tensor(
                [fn(envs[n]) for n in self.scenario_names], dtype=dtype, device=self.device
            )

        ref = self._reference
        self.agent_radius = ref.agent_radius_m
        self.target_radius = ref.target_radius_m
        self.target_height = ref.target_height_m
        self.following = ref.following_distance_m
        self.tolerance = ref.distance_tolerance_m
        self.cap = ref.distance_penalty_cap_m
        self.lost_distance = ref.lost_distance_m
        self.weights = dict(ref.reward_weights)
        self.survival = float(ref.survival_bonus)
        rule = SUCCESS_RULE
        if (ref.success_distance_m, ref.success_bearing_deg, ref.success_fraction) != (
            rule.distance_m,
            rule.bearing_deg,
            rule.fraction,
        ):
            raise ValueError("v1compat keeps v1's success rule (contracts.SUCCESS_RULE)")
        self.limits = ref.limits
        self.is_occlusion = column(lambda e: e.scenario_spec.name == "occlusion", torch.bool)
        self.is_noise = column(lambda e: e.scenario_spec.name == "noise", torch.bool)
        self.is_conflict = column(lambda e: e.scenario_spec.name == "conflict", torch.bool)
        self.occlusion_start = column(lambda e: e.scenario_spec.occlusion_start_s)
        self.occlusion_end = column(
            lambda e: e.scenario_spec.occlusion_start_s + e.scenario_spec.occlusion_duration_s
        )
        self.decoy = torch.stack(
            (column(lambda e: e.scenario_spec.decoy_x), column(lambda e: e.scenario_spec.decoy_y)),
            -1,
        )
        noise = {
            key: column(
                lambda e, k=key: (
                    getattr(e.scenario_spec, k) if e.scenario_spec.name == "noise" else 0.0
                )
            )
            for key in (
                "image_noise_std",
                "brightness_std",
                "image_dropout",
                "odor_noise_std",
                "odor_dropout",
            )
        }
        self.noise = noise

    # ----------------------------------------------------------------------------------
    # Target motion

    def _wander_step(self, x, y, heading, rate, draw):
        """One WANDER_DT_S step of v1's wander path (trajectories._extend_wander)."""
        c = self.trajectory
        decay = math.exp(-WANDER_DT_S / c.wander_turn_tau_s)
        noise = c.wander_turn_rate_std * math.sqrt(1 - decay**2)
        cx, cy = (c.xmin + c.xmax) / 2, (c.ymin + c.ymax) / 2
        rate = (decay * rate + noise * draw).clamp(-c.wander_max_turn_rate, c.wander_max_turn_rate)
        wall = torch.minimum(
            torch.minimum(x - c.xmin, c.xmax - x), torch.minimum(y - c.ymin, c.ymax - y)
        )
        proximity = ((c.wander_wall_margin_m - wall) / c.wander_wall_margin_m).clamp(0, 1)
        inward = wrap(torch.atan2(cy - y, cx - x) - heading)
        turn = (rate + c.wander_wall_gain * proximity * inward).clamp(
            -c.wander_max_turn_rate, c.wander_max_turn_rate
        )
        heading = wrap(heading + turn * WANDER_DT_S)
        x = x + c.speed_m_s * torch.cos(heading) * WANDER_DT_S
        y = y + c.speed_m_s * torch.sin(heading) * WANDER_DT_S
        outside = (x < c.xmin) | (x > c.xmax) | (y < c.ymin) | (y > c.ymax)
        x2, y2 = x.clamp(c.xmin, c.xmax), y.clamp(c.ymin, c.ymax)
        heading = torch.where(outside, torch.atan2(cy - y2, cx - x2), heading)
        return x2, y2, heading, rate

    def _rebase_wander(self, s: dict[str, Tensor]) -> None:
        """Drop path entries v1 can no longer read: keep index floor(t / dt) - 1 onward."""
        keep_from = (torch.floor(s["time"] / WANDER_DT_S).long() - 1).clamp(min=0)
        shift = (keep_from - s["path_base"]).clamp(min=0)
        index = (self._path_slots[None] + shift[:, None]).clamp(max=PATH_SLOTS - 1)
        s["path"] = s["path"].gather(1, index[..., None].expand(-1, -1, 3))
        s["path_base"] = s["path_base"] + shift
        s["path_filled"] = (s["path_filled"] - shift).clamp(min=0)

    def _extend_wander(
        self, s: dict[str, Tensor], mask: Tensor, until: Tensor, steps: int = MAX_WANDER_STEPS
    ) -> None:
        """Extend the wander path to cover every index v1 reads up to time ``until``.

        state(t) reads indices floor(t / dt) and floor(t / dt) + 1; one more entry absorbs the
        float division (floor(0.2 / 0.01) is 19 but floor(0.21 / 0.01) is 21). Path entry i is
        index path_base + i; entry m + 1 comes from entry m and wander draw number m, so the
        draws are consumed in v1's order.
        """
        needed = torch.floor(until / WANDER_DT_S).long() + 3 - s["path_base"]
        if self.check_substeps and bool((mask & s["wander"] & (needed > PATH_SLOTS)).any()):
            raise RuntimeError("PATH_SLOTS is too small for one control step")
        grows = mask & s["wander"]
        count = torch.where(grows, (needed - s["path_filled"]).clamp(min=0), 0)
        if isinstance(self.draws, V1Draws) and not bool(count.any()):
            return
        slots = self._wander_slots[None, :steps]
        start = s["path_base"][:, None] + s["path_filled"][:, None] - 1
        draws = self.draws.wander(slots < count[:, None], s["seed"], start + slots)
        rows = self.env_index
        for i in range(steps):
            grow = i < count
            last = (s["path_filled"] - 1).clamp(min=0)
            point = s["path"][rows, last]
            if self.fused:
                x, y, h, rate = kernels.wander_step(
                    point[:, 0], point[:, 1], point[:, 2], s["wander_rate"], draws[:, i],
                    self.trajectory, WANDER_DT_S,
                )  # fmt: skip
            else:
                x, y, h, rate = self._wander_step(
                    point[:, 0], point[:, 1], point[:, 2], s["wander_rate"], draws[:, i]
                )
            here = s["path_filled"].clamp(max=PATH_SLOTS - 1)
            s["path"][rows, here] = torch.where(
                grow[:, None], torch.stack((x, y, h), -1), s["path"][rows, here]
            )
            s["wander_rate"] = torch.where(grow, rate, s["wander_rate"])
            s["path_filled"] = torch.where(grow, s["path_filled"] + 1, s["path_filled"])
        if self.check_substeps and bool((grows & (s["path_filled"] < needed)).any()):
            raise RuntimeError("MAX_WANDER_STEPS is too small for one control step")

    def _target_state(self, s: dict[str, Tensor], time: Tensor) -> tuple[Tensor, Tensor]:
        """[B,K,3] pose and [B,K,2] velocity at episode times ``time`` [B,K] (float64)."""
        c = self.trajectory
        phase, direction = s["phase"][:, None], s["direction"][:, None]
        radius = s["radius"][:, None]
        angle = phase + direction * c.speed_m_s * time / radius
        curved_xy = torch.stack(
            (c.center_x + radius * torch.cos(angle), c.center_y + radius * torch.sin(angle)), -1
        )
        curved_v = (direction * c.speed_m_s)[..., None] * torch.stack(
            (-torch.sin(angle), torch.cos(angle)), -1
        )
        curved_h = angle + direction * math.pi / 2
        position = time / WANDER_DT_S
        index = torch.floor(position)
        fraction = position - index
        slot = (index.long() - s["path_base"][:, None]).clamp(0, PATH_SLOTS - 2)
        if self.check_substeps:
            reads = s["wander"][:, None] & (slot + 1 >= s["path_filled"][:, None])
            if bool(reads.any()):
                raise RuntimeError("the wander path does not cover the requested time")
        start = s["path"].gather(1, slot[..., None].expand(-1, -1, 3))
        end = s["path"].gather(1, (slot + 1)[..., None].expand(-1, -1, 3))
        wander_xy = (1 - fraction)[..., None] * start[..., :2] + fraction[..., None] * end[..., :2]
        wander_h = wrap(start[..., 2] + fraction * wrap(end[..., 2] - start[..., 2]))
        wander_v = c.speed_m_s * torch.stack((torch.cos(wander_h), torch.sin(wander_h)), -1)
        w = s["wander"][:, None]
        xy = torch.where(w[..., None], wander_xy, curved_xy)
        heading = wrap(torch.where(w, wander_h, curved_h))
        velocity = torch.where(w[..., None], wander_v, curved_v)
        return torch.cat((xy, heading[..., None]), -1), velocity

    # ----------------------------------------------------------------------------------
    # Plume

    def _plume_tick(self, s: dict[str, Tensor], tick: Tensor, target_xy: Tensor) -> None:
        """One PuffPlume tick for rows in ``tick``; ``target_xy`` is the target at the tick time.

        v1's order: age, advect, turbulence (for every held puff), drop puffs past their
        lifetime, then emit at the sources (the target, then the decoy in conflict). In the
        wind world the wind direction first takes its OU step for this tick.
        """
        p = self.plume
        s["tick"] = torch.where(tick, s["tick"] + 1, s["tick"])
        if self.wind is not None:
            self._wind_tick(s, tick)
        if self.fused:
            x, y, ages, s["puff_seq"] = kernels.plume_tick(
                s["puff_xy"][..., 0], s["puff_xy"][..., 1], s["puff_age"], s["puff_seq"],
                tick, s["seed"], s["tick"], s["wind"], p,
            )  # fmt: skip
            centers = torch.stack((x, y), -1)
            self._emit(s, tick, target_xy, centers, ages)
            return
        held = tick[:, None] & (s["puff_seq"] >= 0)
        ages = torch.where(held, s["puff_age"] + p.integration_dt, s["puff_age"])
        centers = torch.where(
            held[..., None], s["puff_xy"] + s["wind"][:, None] * p.integration_dt, s["puff_xy"]
        )
        if p.turbulent_diffusion_m2_s:
            sequence = torch.where(held, s["puff_seq"], -1)
            z = self.draws.turbulence(tick & held.any(1), s["seed"], s["tick"], sequence)
            sigma = math.sqrt(2 * p.turbulent_diffusion_m2_s * p.integration_dt)
            centers = torch.where(held[..., None], centers + sigma * z, centers)
        dead = held & (ages > p.lifetime_s)
        s["puff_seq"] = torch.where(dead, -1, s["puff_seq"])
        self._emit(s, tick, target_xy, centers, ages)

    def _wind_tick(self, s: dict[str, Tensor], tick: Tensor) -> None:
        """The wind direction's OU step for rows in ``tick`` (wind world)."""
        w, dt = self.wind, self.plume.integration_dt
        decay = math.exp(-dt / w.direction_tau_s)
        sigma = math.radians(w.direction_std_deg) * math.sqrt(1 - decay**2)
        z = rng.draw(s["seed"], WIND_STREAM, s["tick"][:, None], 0, True)[:, 0]
        s["wind_eta"] = torch.where(tick, decay * s["wind_eta"] + sigma * z, s["wind_eta"])
        s["wind"] = self._wind_vector(s)

    def _wind_vector(self, s: dict[str, Tensor]) -> Tensor:
        angle = s["wind_dir"] + s["wind_eta"]
        return self.wind.speed_m_s * torch.stack((torch.cos(angle), torch.sin(angle)), -1)

    def _emit(self, s, tick: Tensor, target_xy: Tensor, centers: Tensor, ages: Tensor) -> None:
        """PuffPlume emission at the tick: one puff per source once the credit reaches 1."""
        p = self.plume
        credit = torch.where(tick, s["credit"] + p.emission_hz * p.integration_dt, s["credit"])
        emitted = torch.where(tick, torch.floor(credit + 1e-12), 0.0)
        s["credit"] = credit - emitted
        sources = torch.stack((target_xy, self.decoy), 1)  # [B,2,2]
        count = torch.where(self.is_conflict, 2, 1) * emitted.long()  # puffs this tick
        cap = self.puff_capacity
        rows = self.env_index
        for j in range(self.sources):  # credit 0.08 per tick: at most one puff per source
            new = tick & (j < count)
            n = s["emitted"] + j
            slot = torch.remainder(n, cap)
            source = torch.where(self.is_conflict[:, None], sources[:, j % 2], sources[:, 0])
            centers[rows, slot] = torch.where(new[:, None], source, centers[rows, slot])
            ages[rows, slot] = torch.where(new, 0.0, ages[rows, slot])
            s["puff_seq"][rows, slot] = torch.where(new, n, s["puff_seq"][rows, slot])
        s["emitted"] = s["emitted"] + count
        s["puff_xy"], s["puff_age"] = centers, ages

    def _concentration(self, s: dict[str, Tensor], points: Tensor) -> Tensor:
        """[B,2] float32 concentration at [B,2,2] points (PuffPlume.concentration)."""
        p = self.plume
        if self.fused:
            xy = s["puff_xy"]
            return kernels.concentration(
                points, xy[..., 0], xy[..., 1], s["puff_age"], s["puff_seq"], p
            )
        held = s["puff_seq"] >= 0
        variance = p.initial_sigma_m**2 + 2 * p.diffusion_m2_s * s["puff_age"]  # [B,C]
        density = p.puff_mass * torch.exp(-s["puff_age"] / p.decay_tau_s) / (2 * math.pi * variance)
        squared = (points[:, :, None, :] - s["puff_xy"][:, None]).square().sum(-1)  # [B,2,C]
        terms = torch.where(
            held[:, None], torch.exp(-squared / (2 * variance[:, None])) * density[:, None], 0.0
        )
        return (p.background + terms.sum(-1)).to(torch.float32)

    # ----------------------------------------------------------------------------------
    # Sensors

    def _occluded(self, time: Tensor) -> Tensor:
        shape = (self.num_envs,) + (1,) * (time.ndim - 1)
        return (
            self.is_occlusion.reshape(shape)
            & (self.occlusion_start.reshape(shape) <= time)
            & (time < self.occlusion_end.reshape(shape))
        )

    def _odor_points(self, pose: Tensor) -> Tensor:
        """[...,2,2] left and right odor sensor positions from [...,3] poses."""
        yaw = pose[..., 2]
        forward = torch.stack((torch.cos(yaw), torch.sin(yaw)), -1)
        left = torch.stack((-torch.sin(yaw), torch.cos(yaw)), -1)
        center = pose[..., :2] + 0.15 * forward
        half = self.contract.odor_baseline_m / 2
        return torch.stack((center + half * left, center - half * left), -2)

    def _vision_events(self, s, has: Tensor, times: Tensor, poses: Tensor, targets: Tensor):
        """The vision samples of this step, [B,E] in time order: draw, keep the last valid one
        per eye for rendering. An earlier sample is overwritten in v1 before anyone reads it."""
        events = s["vision_events"][:, None] + self._event_slots[: has.shape[1]]
        blackout = self._occluded(times)
        n = self.noise
        for eye in (0, 1):
            dropped, z = self.draws.vision(
                has, s["seed"], eye, events, blackout,
                n["image_dropout"], n["brightness_std"], n["image_noise_std"],
            )  # fmt: skip
            for e in range(has.shape[1]):
                keep = has[:, e] & ~dropped[:, e]
                m = keep[:, None]
                s["eye_pose"][:, eye] = torch.where(m, poses[:, e], s["eye_pose"][:, eye])
                s["eye_target"][:, eye] = torch.where(m, targets[:, e], s["eye_target"][:, eye])
                s["eye_blackout"][:, eye] = torch.where(
                    keep, blackout[:, e], s["eye_blackout"][:, eye]
                )
                s["eye_brightness"][:, eye] = torch.where(
                    keep, z[:, e], s["eye_brightness"][:, eye]
                )
                s["eye_event"][:, eye] = torch.where(keep, events[:, e], s["eye_event"][:, eye])
                s["eye_new"][:, eye] |= keep
                s["stamps"][:, eye] = torch.where(keep, times[:, e], s["stamps"][:, eye])
        s["vision_events"] = s["vision_events"] + has.sum(1)

    def _odor_events(self, s, has: Tensor, times: Tensor, poses: Tensor, ticks: Tensor):
        """The odor samples of this step, [B,E] in time order: draw, and return the last valid
        one per row (whether any, its time, sensor points, plume tick and noise draw)."""
        events = s["odor_events"][:, None] + self._event_slots[: has.shape[1]]
        dropped, z = self.draws.odor(
            has, s["seed"], events, self.noise["odor_dropout"], self.noise["odor_noise_std"]
        )
        keep = has & ~dropped
        last = (keep.long() * (self._event_slots[: has.shape[1]] + 1)).max(1).values - 1
        chosen = last >= 0
        k = last.clamp(min=0)
        s["odor_events"] = s["odor_events"] + has.sum(1)
        return (
            chosen,
            times.gather(1, k[:, None])[:, 0],
            self._odor_points(poses.gather(1, k[:, None, None].expand(-1, 1, 3))[:, 0]),
            ticks.gather(1, k[:, None])[:, 0],
            z.gather(1, k[:, None, None].expand(-1, 1, 2))[:, 0],
        )

    def _apply_odor(self, s, chosen, time, concentration, z) -> None:
        value = concentration.to(torch.float64) + self.noise["odor_noise_std"][:, None] * z
        value = value.clamp(min=0).to(torch.float32)
        s["odor"] = torch.where(chosen[:, None], value, s["odor"])
        s["stamps"][:, 2] = torch.where(chosen, time, s["stamps"][:, 2])

    def _wind_events(self, s, has: Tensor, times: Tensor, poses: Tensor, ticks: Tensor):
        """The antenna samples of this step, [B,E] in time order, on the odor clock: draw,
        and return the last valid one (whether any, time, heading, plume tick, noise).

        Dropout is the noise scenario's odor dropout; the antenna noise is always on."""
        events = s["wind_events"][:, None] + self._event_slots[: has.shape[1]]
        p = self.noise["odor_dropout"][:, None]
        dropped = has & (p > 0) & (rng.draw(s["seed"], WIND_SENSE_STREAM, events, 0, False) < p)
        sub = 1 + torch.arange(2, device=self.device)
        z = rng.draw(s["seed"], WIND_SENSE_STREAM, events[..., None], sub, True)
        keep = has & ~dropped
        last = (keep.long() * (self._event_slots[: has.shape[1]] + 1)).max(1).values - 1
        k = last.clamp(min=0)
        s["wind_events"] = s["wind_events"] + has.sum(1)
        return (
            last >= 0,
            times.gather(1, k[:, None])[:, 0],
            poses.gather(1, k[:, None, None].expand(-1, 1, 3))[:, 0, 2],
            ticks.gather(1, k[:, None])[:, 0],
            z.gather(1, k[:, None, None].expand(-1, 1, 2))[:, 0],
        )

    def _apply_wind(self, s, chosen, time, heading, tick, z, wind: Tensor) -> None:
        c = self.contract
        reading = antenna(wind, heading, c.antenna_angle_deg, c.wind_reference_m_s)
        reading = (reading + self.wind.antenna_noise_std * z).to(torch.float32)
        s["wind_obs"] = torch.where(chosen[:, None], reading, s["wind_obs"])
        s["stamps"][:, 3] = torch.where(chosen, time, s["stamps"][:, 3])

    def _render_vision(self, s: dict[str, Tensor], rows: Tensor | None = None) -> None:
        """Render the recorded samples (``rows``: an index subset, or all rows)."""
        index = rows if rows is not None else self.env_index
        if index.numel() == 0:
            return
        new = s["eye_new"][index]
        pose = s["eye_pose"][index].to(self.dtype)
        target = s["eye_target"][index]
        k = len(index)
        hidden = s["eye_blackout"][index]  # [k,2]: the scenario occludes the target
        radius = (self.target_radius * s["visual_scale"][index])[:, None].expand(k, 2)
        if self.occlusion_view == "hide_target":
            # A NaN radius fails every hit test: the eye sees the arena without the target.
            radius = torch.where(hidden, math.nan, radius)
        raw = self.retina.render_target(
            pose,
            s["wall_intensity"][index],
            target,
            radius,
            self.target_height,
            s["target_intensity"][index],
        )  # [k,2,N]
        for eye in (0, 1):
            image = raw[:, eye]
            factor = torch.ones(k, dtype=torch.float64, device=self.device)
            std = self.noise["brightness_std"][index]
            factor = torch.where(
                std > 0, (1 + std * s["eye_brightness"][index, eye]).clamp(min=0), factor
            )
            image = image * factor.to(torch.float32)[:, None]
            noise_std = self.noise["image_noise_std"][index]
            if self._any_image_noise:
                z = self.draws.vision_noise(
                    s["seed"][index], eye, s["eye_event"][index, eye], image.shape[1]
                )
                # numpy adds the float64 noise to the float32 image in float64, then rounds.
                image = (image.to(torch.float64) + noise_std[:, None] * z.to(torch.float64)).to(
                    torch.float32
                )
            image = image.clamp(0, 1)
            if self.occlusion_view == "blackout":
                image = torch.where(hidden[:, eye, None], 0.0, image)
            s["vision"][index, eye] = torch.where(new[:, eye, None], image, s["vision"][index, eye])
        if rows is None:
            s["eye_new"] = torch.zeros_like(s["eye_new"])
        else:
            s["eye_new"].index_put_((index,), self._false)  # a device value: no host copy

    def _observation(self, s: dict[str, Tensor]) -> dict[str, Tensor]:
        valid = (s["time"][:, None] - s["stamps"] <= self.contract.watchdog_s).to(torch.float32)
        return {
            "vision": s["vision"] * valid[:, :2, None],
            "odor": s["odor"] * valid[:, 2:3],
            "wind": s["wind_obs"] * valid[:, 3:4],  # never sampled in v1compat: valid is 0
            "valid": valid,
        }

    # ----------------------------------------------------------------------------------
    # Episodes

    def _new_state(self) -> dict[str, Tensor]:
        b, n, c, dev, f64 = (
            self.num_envs,
            self.contract.n_ommatidia,
            self.puff_capacity,
            self.device,
            torch.float64,
        )
        z = lambda *shape, dtype=f64: torch.zeros(*shape, dtype=dtype, device=dev)  # noqa: E731
        return {
            "seed": z(b, dtype=torch.int64),
            "pose": z(b, 3),
            "action": z(b, 2, dtype=torch.float32),
            "time": z(b),
            "steps": z(b, dtype=torch.int64),
            "collision": z(b, dtype=torch.bool),
            "target": z(b, 3),
            "target_velocity": z(b, 2),
            "wander": z(b, dtype=torch.bool),
            "direction": z(b),
            "radius": z(b),
            "phase": z(b),
            "path": z(b, PATH_SLOTS, 3),
            "path_base": z(b, dtype=torch.int64),
            "path_filled": z(b, dtype=torch.int64),
            "wander_rate": z(b),
            "wall_intensity": z(b),
            "target_intensity": z(b),
            "visual_scale": z(b),
            "tick": z(b, dtype=torch.int64),
            "credit": z(b),
            "emitted": z(b, dtype=torch.int64),
            "puff_xy": z(b, c, 2),
            "puff_age": z(b, c),
            "puff_seq": z(b, c, dtype=torch.int64) - 1,
            "vision": z(b, 2, n, dtype=torch.float32),
            "odor": z(b, 2, dtype=torch.float32),
            "stamps": z(b, 4) - math.inf,
            "wind": z(b, 2),  # world wind velocity (v1compat: the plume's fixed drift)
            "wind_dir": z(b),
            "wind_eta": z(b),
            "wind_obs": z(b, 2, dtype=torch.float32),
            "next_vision": z(b),
            "next_odor": z(b),
            "vision_events": z(b, dtype=torch.int64),
            "odor_events": z(b, dtype=torch.int64),
            "wind_events": z(b, dtype=torch.int64),
            "eye_pose": z(b, 2, 3),
            "eye_target": z(b, 2, 2),
            "eye_blackout": z(b, 2, dtype=torch.bool),
            "eye_brightness": z(b, 2),
            "eye_event": z(b, 2, dtype=torch.int64),
            "eye_new": z(b, 2, dtype=torch.bool),
            # Episode accumulators.
            "previous_distance": z(b),
            "previous_occluded": z(b, dtype=torch.bool),
            "return": z(b),
            "distance_sum": z(b),
            "bearing_sum": z(b),
            "speed_sum": z(b),
            "speed_sq_sum": z(b),
            "tracked": z(b),
            "near": z(b),
            "occluded_steps": z(b),
            "occluded_tracked": z(b),
            "reacquire_from": z(b) + math.nan,
            "reacquired": z(b) + math.nan,
        }

    def reset(
        self, mask: Tensor | None = None, episode_seeds: Tensor | None = None
    ) -> dict[str, Tensor]:
        # The first reset starts episode 0 everywhere. A later one starts the next episode on
        # the masked rows, like the automatic reset, unless explicit seeds are given (#137).
        if not self._state:
            self._state = self._new_state()
            mask = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
        else:
            if mask is None:
                mask = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
            if episode_seeds is None:
                self.episode_index.add_(mask.to(self.device).long())
        static = self._begin()
        self._reset_rows(mask.to(self.device), episode_seeds)
        self._render_vision(self._state, torch.nonzero(mask).flatten())
        self._commit(static)
        return self._observation(self._state)

    def _reset_rows(self, mask: Tensor, episode_seeds: Tensor | None) -> None:
        s = self._state
        if episode_seeds is None:
            seeds = rng.episode_seed(self.seed, self._seed_rows, self.episode_index)
            if self.episode_seed_namespace:
                seeds = seeds | self.episode_seed_namespace
        else:
            seeds = episode_seeds.to(self.device, torch.int64)
            if bool(((seeds < 0) | (seeds >= 1 << rng.SEED_BITS)).any()):
                raise ValueError("episode seeds must be in [0, 2**52)")
        fresh = self._new_state()
        for name, value in fresh.items():
            shape = (self.num_envs,) + (1,) * (value.ndim - 1)
            s[name] = torch.where(mask.reshape(shape), value, s[name])
        s["seed"] = torch.where(mask, seeds, s["seed"])
        d = self.draws.episode(mask, s["seed"])
        wander = (
            d["motion"] == self.motions.index("wander")
            if "wander" in self.motions
            else torch.zeros_like(mask)
        )
        for name, key in (
            ("direction", "direction"),
            ("radius", "radius_m"),
            ("phase", "phase_rad"),
            ("wall_intensity", "wall_intensity"),
            ("target_intensity", "target_intensity"),
            ("visual_scale", "visual_scale"),
        ):
            s[name] = torch.where(mask, d[key].to(torch.float64), s[name])
        s["radius"] = torch.where(mask & wander, 1.0, s["radius"])  # unused, finite
        s["wander"] = torch.where(mask, wander, s["wander"])
        start = torch.stack((d["start_x"], d["start_y"], d["heading_rad"]), -1).to(torch.float64)
        if not self.layout.randomize_target_layout:
            start = torch.tensor(
                [self.trajectory.center_x, self.trajectory.center_y, 0.0],
                dtype=torch.float64,
                device=self.device,
            ).expand(self.num_envs, 3)
        s["path"][:, 0] = torch.where(mask[:, None], start, s["path"][:, 0])
        s["path_filled"] = torch.where(mask, 1, s["path_filled"])
        self._extend_wander(s, mask, s["time"], steps=2)  # indices 0..2 at t = 0
        target, velocity = (x[:, 0] for x in self._target_state(s, s["time"][:, None]))
        s["target"] = torch.where(mask[:, None], target, s["target"])
        s["target_velocity"] = torch.where(mask[:, None], velocity, s["target_velocity"])
        # Agent behind the target, shifted sideways (layout.sample_episode_layout).
        heading = target[:, 2]
        lateral, distance = d["lateral_m"].to(torch.float64), self.layout.start_distance_m
        yaw_offset = d["yaw_offset_rad"].to(torch.float64)
        agent = torch.stack(
            (
                target[:, 0] - distance * torch.cos(heading) - lateral * torch.sin(heading),
                target[:, 1] - distance * torch.sin(heading) + lateral * torch.cos(heading),
                torch.where(yaw_offset != 0, wrap(heading + yaw_offset), heading),
            ),
            -1,
        )
        s["pose"] = torch.where(mask[:, None], agent, s["pose"])
        s["previous_distance"] = torch.where(
            mask, (s["target"][:, :2] - s["pose"][:, :2]).norm(dim=-1), s["previous_distance"]
        )
        s["previous_occluded"] = torch.where(
            mask, self._occluded(s["time"]), s["previous_occluded"]
        )
        if self.wind is None:  # v1's plume drifts with a fixed wind
            s["wind"] = torch.where(mask[:, None], self._plume_wind, s["wind"])
        else:  # a direction per episode, plus its stationary OU meander
            u = rng.draw(s["seed"], WIND_EPISODE_STREAM, self._event_slots[None, :2], 0, False)
            z = rng.draw(s["seed"], WIND_EPISODE_STREAM, self._event_slots[None, :1], 1, True)
            s["wind_dir"] = torch.where(mask, (2 * u[:, 0] - 1) * math.pi, s["wind_dir"])
            eta = math.radians(self.wind.direction_std_deg) * z[:, 0]
            s["wind_eta"] = torch.where(mask, eta, s["wind_eta"])
            s["wind"] = torch.where(mask[:, None], self._wind_vector(s), s["wind"])
        # Sensors at t = 0 (FlyTrackEnv.reset); the plume is still empty.
        one = mask[:, None]
        self._vision_events(
            s, one, s["time"][:, None], s["pose"][:, None], s["target"][:, None, :2]
        )
        s["next_vision"] = torch.where(
            mask, s["next_vision"] + 1 / self.contract.vision_hz, s["next_vision"]
        )
        chosen, time, points, _, z = self._odor_events(
            s, one, s["time"][:, None], s["pose"][:, None], s["tick"][:, None]
        )
        self._apply_odor(s, chosen, time, self._concentration(s, points), z)
        if self.wind is not None:
            sample = self._wind_events(
                s, one, s["time"][:, None], s["pose"][:, None], s["tick"][:, None]
            )
            self._apply_wind(s, *sample, s["wind"])
        s["next_odor"] = torch.where(
            mask, s["next_odor"] + 1 / self.contract.odor_hz, s["next_odor"]
        )

    # ----------------------------------------------------------------------------------
    # Stepping

    def _clip(self, action: Tensor, previous: Tensor) -> Tensor:
        """ActionLimits.clip: bounds in float32, the acceleration limit in float64, as v1."""
        low, high, delta = self._action_low, self._action_high, self._action_delta
        value = torch.minimum(torch.maximum(action.to(torch.float32), low), high)
        previous64 = previous.to(torch.float64)
        value = torch.minimum(
            torch.maximum(value.to(torch.float64), previous64 - delta), previous64 + delta
        )
        return torch.minimum(torch.maximum(value, low.double()), high.double()).to(torch.float32)

    def _collides(self, pose: Tensor, target: Tensor) -> Tensor:
        xy = pose[..., :2]
        wall = ((xy < self._lo + self.agent_radius) | (xy > self._hi - self.agent_radius)).any(-1)
        touching = (xy - target[..., :2]).norm(dim=-1) <= self.agent_radius + self.target_radius
        return wall | touching

    def privileged(self) -> dict[str, Tensor]:
        s = self._state
        bearing = self._distance_bearing(s)[1]
        occluded = self._occluded(s["time"])
        half = math.radians(self.contract.horizontal_fov_deg / 2)
        center = math.radians(self.contract.eye_yaw_deg)
        in_view = (wrap(bearing - center).abs() <= half) | (wrap(bearing + center).abs() <= half)
        f = self.dtype
        return {
            "agent_pose": s["pose"].to(f, copy=True),
            "agent_velocity": s["action"].to(f, copy=True),
            "target_pose": s["target"].to(f, copy=True),
            "target_velocity": s["target_velocity"].to(f, copy=True),
            "occluded": occluded,
            "target_visible": in_view & ~occluded,
            "wind_velocity": (
                s["wind"].to(f, copy=True)
                if self.wind is not None
                else torch.zeros(self.num_envs, 2, dtype=f, device=self.device)
            ),
            "decoy_active": self.is_conflict.clone(),
            "decoy_position": torch.where(self.is_conflict[:, None], self.decoy, 0.0).to(f),
            "time_s": s["time"].to(f, copy=True),
            "episode_seed": s["seed"].clone(),
        }

    def step(self, action: Tensor) -> StepResult:
        action = action.to(self.device)
        if self.graph:
            if self._graph is None:
                self._capture(action)
            self._graph_action.copy_(action)
            self._graph.replay()
            outputs = _clone(self._graph_outputs[:6])
            ended = int(self._graph_outputs[6].sum())  # the one host sync per step
            if ended:
                self._reset_graph.replay()
            if ended > self.reset_render_rows:
                # More rows ended at once than the graph renders (e.g. synchronised
                # truncation): render the rest here.
                static = self._begin()
                self._render_vision(
                    self._state, torch.nonzero(self._state["eye_new"].any(1)).flatten()
                )
                self._commit(static)
        else:
            outputs = self._physics(action)
            ended = outputs[6]
            rows = torch.nonzero(ended).flatten()  # the one host sync per step
            if rows.numel():
                self._auto_reset(ended, rows)
        reward, terminated, truncated, info, final_obs, final_privileged = outputs[:6]
        obs = self._observation(self._state)
        return StepResult(obs, reward, terminated, truncated, info, final_obs, final_privileged)

    def _auto_reset(self, ended: Tensor, rows: Tensor | None) -> None:
        """Start the next episode on rows that ended (all rows masked when rows is None)."""
        self.episode_index.add_(ended.long())
        static = self._begin()
        self._reset_rows(ended, None)
        if rows is None:
            # Inside the reset graph: render the first reset_render_rows ended rows; the
            # others (rare) are rendered by step() after the replay.
            order = torch.sort(ended.to(torch.uint8), descending=True, stable=True).indices
            rows = order[: self.reset_render_rows]
        self._render_vision(self._state, rows)
        self._commit(static)

    def _begin(self) -> dict[str, Tensor]:
        """Work on a copy of the state dict; _commit writes the results back in place, so the
        state tensors keep their addresses (what a CUDA graph replays into)."""
        static = self._state
        self._state = dict(static)
        return static

    def _commit(self, static: dict[str, Tensor]) -> None:
        for name, value in self._state.items():
            if value is not static[name]:
                static[name].copy_(value)
        self._state = static

    def release_rollout_graphs(self) -> None:
        """Free the CUDA graphs and their memory pool; the state is kept, and the next step
        captures them again."""
        self._graph = None
        for name in ("_reset_graph", "_graph_outputs", "_graph_action"):
            self.__dict__.pop(name, None)

    def _capture(self, action: Tensor) -> None:
        """Record _physics and the automatic reset as two CUDA graphs. Warm-up runs mutate
        the state, so it is restored afterwards."""
        snapshot = {name: value.clone() for name, value in self._state.items()}
        episode_index = self.episode_index.clone()
        self._graph_action = action.clone()
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(2):
                outputs = self._physics(self._graph_action)
                self._auto_reset(torch.zeros_like(outputs[6]), None)
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            self._graph_outputs = self._physics(self._graph_action)
        reset_graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(reset_graph, pool=graph.pool()):
            self._auto_reset(self._graph_outputs[6], None)
        for name, value in snapshot.items():
            self._state[name].copy_(value)
        self.episode_index.copy_(episode_index)
        self._graph, self._reset_graph = graph, reset_graph

    def _physics(self, action: Tensor) -> tuple:
        """Everything in a step before the automatic reset; no host sync on the hash path."""
        static = self._begin()
        s = self._state
        dt_control = self.contract.control_dt
        applied = self._clip(action, s["action"])
        stale = ~(self._observation(s)["valid"] > 0).any(-1)  # all four channels lost
        applied = torch.where(stale[:, None], 0.0, applied)
        # v1 unpacks float32 scalars, so v / omega is a float32 division.
        straight = applied[:, 1].abs() < 1e-8
        ratio = (applied[:, 0] / torch.where(straight, 1.0, applied[:, 1])).to(torch.float64)
        end = torch.clamp((s["steps"] + 1).to(torch.float64) * dt_control, max=self.episode_seconds)
        # Pass 1: v1's clock and the agent's motion at every substep, as if nothing collided.
        t, nv, no, pose = s["time"], s["next_vision"], s["next_odor"], s["pose"]
        times, poses, active, v_due, o_due, nvs, nos = [], [], [], [], [], [], []
        for _ in range(MAX_SUBSTEPS):
            if self.fused:
                before = t
                t, nv, no, live, vision, odor = kernels.clock(t, nv, no, end, self.contract)
                pose = kernels.motion(pose, before, t, applied, ratio)
                for store, value in zip(
                    (times, poses, active, v_due, o_due, nvs, nos),
                    (t, pose, live, vision, odor, nv, no),
                    strict=True,
                ):
                    store.append(value)
                continue
            live = t < end - 1e-10
            nxt = torch.minimum(
                torch.minimum(end, t + self.contract.neural_dt), torch.minimum(nv, no)
            )
            dt = torch.where(live, nxt - t, 0.0)
            yaw = pose[:, 2]
            # numpy 2 keeps float32 for float32 scalar x Python float: v1's omega * dt and
            # dt * v are float32 products, added to the float64 pose.
            dt32 = dt.to(torch.float32)
            turn = (applied[:, 1] * dt32).to(torch.float64)
            travel = (dt32 * applied[:, 0]).to(torch.float64)
            end_yaw = yaw + turn
            dx = torch.where(
                straight, travel * torch.cos(yaw), ratio * (torch.sin(end_yaw) - torch.sin(yaw))
            )
            dy = torch.where(
                straight, travel * torch.sin(yaw), ratio * (torch.cos(yaw) - torch.cos(end_yaw))
            )
            moved = torch.stack((pose[:, 0] + dx, pose[:, 1] + dy, wrap(yaw + turn)), -1)
            pose = torch.where(live[:, None], moved, pose)
            t = torch.where(live, nxt, t)
            vision = live & (nv <= t + 1e-9)
            nv = torch.where(vision, nv + 1 / self.contract.vision_hz, nv)
            odor = live & (no <= t + 1e-9)
            no = torch.where(odor, no + 1 / self.contract.odor_hz, no)
            for store, value in zip(
                (times, poses, active, v_due, o_due, nvs, nos),
                (t, pose, live, vision, odor, nv, no),
                strict=True,
            ):
                store.append(value)
        if self.check_substeps and bool((t < end - 1e-10).any()):
            raise RuntimeError("MAX_SUBSTEPS was not enough to finish the control step")
        times, poses, active = torch.stack(times, 1), torch.stack(poses, 1), torch.stack(active, 1)
        v_due, o_due = torch.stack(v_due, 1), torch.stack(o_due, 1)
        # Pass 2: the target at every substep, and the first collision: v1 stops there.
        self._rebase_wander(s)
        self._extend_wander(s, torch.ones_like(s["wander"]), end)
        target, velocity = self._target_state(s, times)
        hit = active & self._collides(poses, target)
        steps_index = self._substeps
        first = torch.where(hit.any(1), hit.long().argmax(1), MAX_SUBSTEPS)
        active = active & (steps_index[None] <= first[:, None])
        v_due, o_due = v_due & active, o_due & active
        last = (active.long() * (steps_index[None] + 1)).max(1).values - 1
        moved = last >= 0
        k = last.clamp(min=0)

        def at_last(values: Tensor, current: Tensor) -> Tensor:
            index = k.reshape(-1, *([1] * (values.ndim - 1))).expand(-1, 1, *values.shape[2:])
            picked = values.gather(1, index)[:, 0]
            return torch.where(moved.reshape(-1, *([1] * (picked.ndim - 1))), picked, current)

        # Pass 3: the plume, tick by tick; odor is read at its last valid sample's tick.
        p = self.plume
        desired = torch.floor((times + 1e-9) / p.integration_dt).long()
        final_tick = at_last(desired, s["tick"])
        if self.check_substeps and bool((final_tick - s["tick"] > MAX_TICKS).any()):
            raise RuntimeError("MAX_TICKS was not enough for one control step")
        has, slot = self._event_slots_of(o_due, ODOR_SLOTS)
        chosen, odor_time, points, odor_tick, odor_z = self._odor_events(
            s, has, times.gather(1, slot), poses.gather(1, slot[..., None].expand(-1, -1, 3)),
            desired.gather(1, slot),
        )  # fmt: skip
        if self.wind is not None:  # the antennae share the odor clock (wind_hz = odor_hz)
            wind_sample = self._wind_events(
                s, has, times.gather(1, slot), poses.gather(1, slot[..., None].expand(-1, -1, 3)),
                desired.gather(1, slot),
            )  # fmt: skip
            wind_tick = wind_sample[3]
            wind_at = s["wind"]  # replaced below once the plume reaches wind_tick
        concentration = torch.zeros(self.num_envs, 2, dtype=torch.float32, device=self.device)
        read = chosen & (odor_tick == s["tick"])
        concentration = torch.where(read[:, None], self._concentration(s, points), concentration)
        tick_times = (s["tick"][:, None] + 1 + self._tick_slots[None]).to(torch.float64)
        # Slots past this step's last tick are never used; clamp them inside the path.
        tick_times = torch.minimum(tick_times * p.integration_dt, end[:, None])
        sources = self._target_state(s, tick_times)[0][..., :2]  # [B,T,2]
        for j in range(MAX_TICKS):
            tick = s["tick"] < final_tick
            if isinstance(self.draws, V1Draws) and not bool(tick.any()):
                break
            self._plume_tick(s, tick, sources[:, j])
            read = tick & chosen & (odor_tick == s["tick"])
            concentration = torch.where(
                read[:, None], self._concentration(s, points), concentration
            )
            if self.wind is not None:
                wind_at = torch.where(
                    (tick & (wind_tick == s["tick"]))[:, None], s["wind"], wind_at
                )
        self._apply_odor(s, chosen, odor_time, concentration, odor_z)
        if self.wind is not None:
            self._apply_wind(s, *wind_sample, wind_at)
        # Vision: the step's samples in order; the last valid one per eye is rendered.
        has, slot = self._event_slots_of(v_due, VISION_SLOTS)
        self._vision_events(
            s, has, times.gather(1, slot), poses.gather(1, slot[..., None].expand(-1, -1, 3)),
            target.gather(1, slot[..., None].expand(-1, -1, 3))[..., :2],
        )  # fmt: skip
        s["time"] = at_last(times, s["time"])
        s["pose"] = at_last(poses, s["pose"])
        s["target"] = at_last(target, s["target"])
        s["target_velocity"] = at_last(velocity, s["target_velocity"])
        s["next_vision"] = at_last(torch.stack(nvs, 1), s["next_vision"])
        s["next_odor"] = at_last(torch.stack(nos, 1), s["next_odor"])
        s["collision"] = hit.any(1)
        self._render_vision(s)
        outputs = self._score(applied)
        self._commit(static)
        return outputs

    def _event_slots_of(self, due: Tensor, count: int) -> tuple[Tensor, Tensor]:
        """[B,count] (present, substep index) of the samples flagged in ``due`` [B,M]."""
        cumulative = due.long().cumsum(1)
        if self.check_substeps and bool((cumulative[:, -1] > count).any()):
            raise RuntimeError("more sensor samples in one control step than slots")
        present, index = [], []
        for e in range(count):
            here = due & (cumulative == e + 1)
            present.append(here.any(1))
            index.append(here.long().argmax(1))
        return torch.stack(present, 1), torch.stack(index, 1)

    def _distance_bearing(self, s: dict[str, Tensor]) -> tuple[Tensor, Tensor]:
        displacement = s["target"][:, :2] - s["pose"][:, :2]
        bearing = wrap(torch.atan2(displacement[:, 1], displacement[:, 0]) - s["pose"][:, 2])
        return displacement.norm(dim=-1), bearing

    def _score(self, applied: Tensor) -> tuple:
        """Reward, metrics and the ending state (FlyTrackEnv.step after its substep loop)."""
        s, w = self._state, self.weights
        distance, bearing = self._distance_bearing(s)
        change = applied - s["action"]
        penalty = ((distance - self.following).abs() - self.tolerance).clamp(min=0)
        if self.cap is not None:
            penalty = penalty.clamp(max=self.cap)
        collision = s["collision"]
        lost = distance > self.lost_distance
        near = distance <= SUCCESS_RULE.distance_m
        tracked = near & (torch.rad2deg(bearing).abs() <= SUCCESS_RULE.bearing_deg)
        if self.reward_kind == "v1":
            reward = (
                w["progress"] * (s["previous_distance"] - distance)
                - w["distance"] * penalty
                - w["smoothness"] * (change * change).sum(-1).to(torch.float64)
                - w["collision"] * collision.to(torch.float64)
                - w["lost"] * lost.to(torch.float64)
            )
            if self.survival:
                reward = torch.where(collision, reward, reward + self.survival)
        else:
            reward = track_v2_reward(tracked, distance, collision, change, self.following)
        speed = applied[:, 0].to(torch.float64)
        s["distance_sum"] += distance
        s["bearing_sum"] += bearing.abs()
        s["speed_sum"] += speed
        s["speed_sq_sum"] += speed.square()
        s["near"] += near
        s["tracked"] += tracked
        s["previous_distance"] = distance
        s["action"] = applied
        s["steps"] += 1
        s["return"] += reward
        occluded = self._occluded(s["time"])
        s["occluded_steps"] += occluded
        s["occluded_tracked"] += occluded & tracked
        ended_occlusion = s["previous_occluded"] & ~occluded
        s["reacquire_from"] = torch.where(ended_occlusion, self.occlusion_end, s["reacquire_from"])
        pending = ~torch.isnan(s["reacquire_from"])
        found = pending & (bearing.abs() < math.radians(15))
        s["reacquired"] = torch.where(
            found, (s["time"] - s["reacquire_from"]).clamp(min=0), s["reacquired"]
        )
        s["reacquire_from"] = torch.where(found, math.nan, s["reacquire_from"])
        s["previous_occluded"] = occluded
        terminated = collision.clone()
        truncated = (s["time"] >= self.episode_seconds - 1e-9) | (s["steps"] >= self.max_steps)
        info = self._info(tracked, collision, distance, bearing, occluded)
        final_obs = self._observation(s)
        final_privileged = self.privileged()
        ended = terminated | truncated
        reward = reward.to(torch.float32)
        return reward, terminated, truncated, info, final_obs, final_privileged, ended

    def _info(self, tracked, collision, distance, bearing, occluded) -> dict[str, Tensor]:
        s, f = self._state, self.dtype
        steps = s["steps"].clamp(min=1).to(torch.float64)
        mean_speed = s["speed_sum"] / steps
        occluded_steps = s["occluded_steps"]
        return {
            "tracked": tracked.clone(),
            "collision": collision.clone(),
            "distance_m": distance.to(f),
            "bearing_rad": bearing.to(f),
            "occluded": occluded,
            "episode_success": (s["tracked"] / steps >= SUCCESS_RULE.fraction) & ~collision,
            "episode_tracking_fraction": (s["tracked"] / steps).to(f),
            "episode_within_distance_fraction": (s["near"] / steps).to(f),
            "episode_collision": collision.clone(),
            "episode_steps": s["steps"].clone(),
            "episode_mean_distance_m": (s["distance_sum"] / steps).to(f),
            "episode_bearing_error_deg": torch.rad2deg(s["bearing_sum"] / steps).to(f),
            "episode_mean_speed_m_s": mean_speed.to(f),
            "episode_speed_std_m_s": (s["speed_sq_sum"] / steps - mean_speed.square())
            .clamp(min=0)
            .sqrt()
            .to(f),
            "episode_occlusion_tracking_fraction": torch.where(
                occluded_steps > 0, s["occluded_tracked"] / occluded_steps.clamp(min=1), math.nan
            ).to(f),
            "episode_reacquisition_time_s": s["reacquired"].to(f),
            "episode_reacquisition_censored": ~torch.isnan(s["reacquire_from"]),
            "episode_seed": s["seed"].clone(),
            "episode_return": s["return"].to(f),
        }
