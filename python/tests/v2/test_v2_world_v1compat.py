"""v1compat reproduces v1's FlyTrackEnv step by step, within tolerance (WP2a, #132).

Parity method (plan 3.3, #145): the world is fed v1's own random draws (draws="v1") and runs
in float64; both worlds receive the same actions; observations, reward, flags and episode
metrics are compared with tolerances. The hash draws used in production are checked for the
interface properties instead (seeded, independent of actions, automatic reset).
"""

from pathlib import Path

import numpy as np
import pytest
import torch
import yaml
from flytrack_core.contracts import SensorContract
from flytrack_sim.env import FlyTrackEnv
from flytrack_v2.contracts import PRIVILEGED_KEYS, SensorContractV2, validate_obs
from flytrack_v2.interfaces import EPISODE_INFO_KEYS, STEP_INFO_KEYS
from flytrack_v2.world.rng import episode_seed
from flytrack_v2.world.teacher import PrivilegedTeacher
from flytrack_v2.world.v1compat import V1CompatWorld, track_v2_reward
from flytrack_v2.world.vision import BatchedRetina

ROOT = Path(__file__).resolve().parents[3]
SCENARIOS = ("normal", "occlusion", "noise", "conflict")
# A vision value may differ by one subray's share where a ray grazes an edge (vision.py).
VISION_TOLERANCE, SUBRAY_SHARE = 1e-4, 1 / 7 + 1e-4
TOLERANCE = 1e-5


@pytest.fixture(scope="module")
def contracts():
    return (
        SensorContractV2.from_json(ROOT / "configs/v2/sensor_contract.json"),
        SensorContract.from_json(ROOT / "configs/sensor_contract.json"),
    )


def _scenario_configs(occlusion_start_s=None):
    configs = {}
    for name in SCENARIOS:
        configs[name] = yaml.safe_load((ROOT / f"configs/scenarios/{name}.yaml").read_text())
        configs[name].pop("name")
    if occlusion_start_s is not None:
        configs["occlusion"]["occlusion_start_s"] = occlusion_start_s
        configs["occlusion"]["occlusion_duration_s"] = 1.0
    return configs


def _compare(contracts, motion, steps, options, scenario_configs, teacher=True, seed0=1000):
    """Run v1 envs and one v1compat world side by side; return the worst differences."""
    c2, c1 = contracts
    seeds = [seed0 + 7 * i for i in range(len(SCENARIOS))]
    options = {"episode_seconds": 60, "target_motion": motion, **options}
    envs = [
        FlyTrackEnv(contract=c1, scenario=name, scenario_config=scenario_configs[name], **options)
        for name in SCENARIOS
    ]
    world = V1CompatWorld(
        c2,
        len(SCENARIOS),
        list(SCENARIOS),
        options,
        draws="v1",
        dtype=torch.float64,
        scenario_configs=scenario_configs,
        check_substeps=True,
        reward="v1",  # FlyTrackEnv's reward (#161 D3)
        occlusion_view="blackout",  # v1's black image during occlusion (#161 C2)
    )
    world.reset(episode_seeds=torch.tensor(seeds))
    teacher = PrivilegedTeacher(c1, options.get("following_distance_m", 1.0))
    state = teacher.initial_state(len(SCENARIOS), "cpu")
    infos = [env.reset(seed=seed)[1] for env, seed in zip(envs, seeds, strict=True)]
    rng = np.random.default_rng(5)
    alive = np.ones(len(envs), bool)
    worst = {"flips": 0, "values": 0, "episodes": 0}

    def record(name, value):
        worst[name] = max(worst.get(name, 0.0), float(value))

    for _ in range(steps):
        if teacher:
            action = np.stack([info["teacher_action"] for info in infos]).astype(np.float32)
        else:
            action = np.c_[rng.uniform(0, 0.5, 4), rng.uniform(-1.5, 1.5, 4)].astype(np.float32)
        result = world.step(torch.as_tensor(action))
        obs, privileged = result.final_obs, result.final_privileged
        label, _ = teacher.act(privileged, obs, privileged["agent_velocity"], state)
        for j, env in enumerate(envs):
            if not alive[j]:
                continue
            o1, reward, terminated, truncated, infos[j] = env.step(action[j])
            error = np.abs(obs["vision"][j].numpy() - o1["vision"])
            record("vision", error.max())
            worst["flips"] += int((error > VISION_TOLERANCE).sum())
            worst["values"] += error.size
            record("odor", np.abs(obs["odor"][j].numpy() - o1["odor"]).max())
            record("teacher", np.abs(label[j].numpy() - infos[j]["teacher_action"]).max())
            record("valid", np.abs(obs["valid"][j, :3].numpy() - o1["valid"]).max())
            record("reward", abs(float(result.reward[j]) - reward))
            record(
                "pose", np.abs(privileged["agent_pose"][j].numpy() - infos[j]["agent_pose"]).max()
            )
            record(
                "target",
                np.abs(
                    privileged["target_pose"][j, :2].numpy() - infos[j]["target_pose"][:2]
                ).max(),
            )
            flags = (
                bool(result.terminated[j]) != terminated,
                bool(result.truncated[j]) != truncated,
                bool(result.info["occluded"][j]) != infos[j]["occluded"],
                bool(result.info["collision"][j]) != infos[j]["collision"],
            )
            record("flags", sum(flags))
            if terminated or truncated:
                metrics = infos[j]["episode_metrics"]
                record("success", bool(result.info["episode_success"][j]) != metrics["success"])
                record(
                    "tracking",
                    abs(
                        float(result.info["episode_tracking_fraction"][j])
                        - metrics["tracking_fraction"]
                    ),
                )
                record(
                    "bearing_deg",
                    abs(
                        float(result.info["episode_bearing_error_deg"][j])
                        - metrics["bearing_error_deg"]
                    ),
                )
                record("steps", int(result.info["episode_steps"][j]) != metrics["episode_length"])
                worst["episodes"] += 1
                alive[j] = False
        if not alive.any():
            break
    return worst


def _assert_parity(worst):
    assert worst["vision"] <= SUBRAY_SHARE
    assert worst["flips"] <= 1e-5 * worst["values"]
    for name in ("odor", "valid", "pose", "target", "flags", "teacher"):
        assert worst[name] <= TOLERANCE, (name, worst[name])
    assert worst["reward"] <= 1e-6  # the interface returns float32 rewards
    for name in ("success", "tracking", "steps"):
        assert worst.get(name, 0) <= TOLERANCE, (name, worst)
    assert worst.get("bearing_deg", 0) <= 1e-6


@pytest.mark.parametrize("motion", ["curved", "wander"])
def test_short_parity_all_scenarios(contracts, motion):
    # Occlusion moved to 0.5-1.5 s so a short run crosses it, and the reacquisition logic runs.
    worst = _compare(contracts, motion, 40, {}, _scenario_configs(occlusion_start_s=0.5))
    _assert_parity(worst)


@pytest.mark.slow
@pytest.mark.parametrize("motion", ["curved", "wander"])
@pytest.mark.parametrize("teacher", [True, False], ids=["teacher", "random"])
def test_full_episode_parity_with_domain_randomization(contracts, motion, teacher):
    worst = _compare(
        contracts, motion, 1200, {"domain_randomization": True}, _scenario_configs(), teacher
    )
    _assert_parity(worst)
    assert worst["episodes"] == len(SCENARIOS)


def _world(contracts, num_envs=6, seed=3, **kwargs):
    return V1CompatWorld(
        contracts[0],
        num_envs,
        [SCENARIOS[i % 4] for i in range(num_envs)],
        {"episode_seconds": 2, "target_motion": ["curved", "wander"]},
        seed=seed,
        scenario_configs=_scenario_configs(occlusion_start_s=0.5),
        **kwargs,
    )


def test_interface_keys_and_shapes(contracts):
    world = _world(contracts)
    obs = world.reset()
    assert validate_obs(obs, contracts[0], check_values=True) == 6
    assert torch.all(obs["wind"] == 0) and torch.all(obs["valid"][:, 3] == 0)
    assert set(world.privileged()) == set(PRIVILEGED_KEYS)
    result = world.step(torch.zeros(6, 2))
    assert set(STEP_INFO_KEYS) | set(EPISODE_INFO_KEYS) <= set(result.info)
    assert set(result.final_privileged) == set(PRIVILEGED_KEYS)
    validate_obs(result.final_obs, contracts[0], check_values=True)


def test_hash_draws_are_seeded_and_independent_of_actions(contracts):
    a, b, c = _world(contracts), _world(contracts), _world(contracts, seed=4)
    for world in (a, b, c):
        world.reset()
    targets = {id(w): [] for w in (a, b, c)}
    for step in range(15):
        slow = torch.tensor([[0.1, 0.0]]).expand(6, 2)
        fast = torch.tensor([[0.5, 1.0 if step % 2 else -1.0]]).expand(6, 2)
        for world, action in ((a, slow), (b, fast), (c, slow)):
            targets[id(world)].append(world.step(action).final_privileged["target_pose"])
    # The target path is the world's randomness only: same seed, same path, whatever the agent does.
    torch.testing.assert_close(torch.stack(targets[id(a)]), torch.stack(targets[id(b)]))
    assert not torch.allclose(torch.stack(targets[id(a)]), torch.stack(targets[id(c)]))


def test_automatic_reset_keeps_the_final_state(contracts):
    world = _world(contracts)
    world.reset()
    seeds = world.privileged()["episode_seed"].clone()
    result = None
    for _ in range(40):  # 2 s episodes are 40 control steps
        result = world.step(torch.tensor([[0.2, 0.3]]).expand(6, 2))
    assert result is not None and bool(result.truncated.all())
    assert torch.all(result.info["episode_steps"] == 40)
    assert torch.all(result.final_privileged["time_s"] >= 2 - 1e-6)
    after = world.privileged()
    assert torch.all(after["time_s"] == 0) and not torch.any(after["episode_seed"] == seeds)
    assert torch.all(world.episode_index == 1)


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="the fused kernels and graphs need CUDA")
def test_fused_graph_path_matches_the_eager_reference_on_cuda(contracts):
    """Fused kernels + CUDA graphs against the eager torch code, same hash draws, through
    resets (2 s episodes), all scenarios and both motions."""
    worlds = [
        V1CompatWorld(
            contracts[0],
            96,
            [SCENARIOS[i % 4] for i in range(96)],
            {"episode_seconds": 2, "target_motion": ["curved", "wander"]},
            seed=11,
            device="cuda",
            scenario_configs=_scenario_configs(occlusion_start_s=0.5),
            **kw,
        )  # fmt: skip
        for kw in ({}, {"fused": False})
    ]
    assert worlds[0].graph and not worlds[1].fused
    assert worlds[0].reset_render_rows < 96  # synchronised truncation overflows the graph
    for world in worlds:
        world.reset()
    rng = np.random.default_rng(2)
    flips = values = 0
    for _ in range(130):  # three episodes and a bit
        action = torch.as_tensor(
            np.c_[rng.uniform(0, 0.5, 96), rng.uniform(-1.5, 1.5, 96)], dtype=torch.float32
        )
        fast, slow = (world.step(action) for world in worlds)
        for name in ("odor", "valid"):
            torch.testing.assert_close(
                fast.final_obs[name], slow.final_obs[name], rtol=1e-4, atol=1e-6
            )
        error = (fast.final_obs["vision"] - slow.final_obs["vision"]).abs()
        assert float(error.max()) <= SUBRAY_SHARE
        flips += int((error > VISION_TOLERANCE).sum())
        values += error.numel()
        torch.testing.assert_close(fast.reward, slow.reward, rtol=1e-5, atol=1e-5)
        for name in ("agent_pose", "target_pose", "time_s"):
            torch.testing.assert_close(
                fast.final_privileged[name], slow.final_privileged[name], rtol=0, atol=1e-5
            )
        assert torch.equal(fast.terminated, slow.terminated)
        assert torch.equal(fast.truncated, slow.truncated)
        assert torch.equal(fast.info["episode_seed"], slow.info["episode_seed"])
        torch.testing.assert_close(
            fast.obs["vision"], slow.obs["vision"], rtol=0, atol=SUBRAY_SHARE
        )
    assert flips <= 1e-5 * values
    assert int(worlds[0].episode_index.min()) >= 3


def _pair(contracts, view, seed=21):
    """An occlusion world (0.5-1.5 s) and a normal world with the same seeds."""
    configs = _scenario_configs(occlusion_start_s=0.5)
    return [
        V1CompatWorld(
            contracts[0],
            6,
            name,
            {"episode_seconds": 3, "target_motion": "curved"},
            seed=seed,
            scenario_configs=configs,
            occlusion_view=view,
            dtype=torch.float64,
        )  # fmt: skip
        for name in ("occlusion", "normal")
    ]


@pytest.mark.parametrize("view", ["hide_target", "blackout"])
def test_occlusion_keeps_valid_and_hides_only_the_target(contracts, view):
    """#161 C2: valid never encodes occlusion; hide_target shows the arena without the target."""
    occluded_world, normal_world = _pair(contracts, view)
    for world in (occluded_world, normal_world):
        world.reset()
    arena = BatchedRetina(contracts[1], occluded_world.walls, "cpu", torch.float64)
    occluded_steps = 0
    for _ in range(40):
        action = torch.tensor([[0.2, 0.3]]).expand(6, 2)
        a, b = occluded_world.step(action), normal_world.step(action)
        assert torch.equal(a.obs["valid"], b.obs["valid"])
        torch.testing.assert_close(
            a.final_privileged["agent_pose"], b.final_privileged["agent_pose"]
        )
        if bool(a.info["occluded"].all()):
            occluded_steps += 1
            s = occluded_world._state
            if view == "blackout":
                assert torch.all(a.obs["vision"] == 0)
            else:
                empty = torch.zeros(6, 2, 1, 5, dtype=torch.float64)
                empty[..., 3] = -1.0  # no disk
                expected = arena.render(s["eye_pose"], s["wall_intensity"], empty)
                torch.testing.assert_close(a.obs["vision"], expected, rtol=0, atol=1e-6)
                assert bool((a.obs["vision"] > 0).all(-1).all())  # walls and floor, not black
            assert not torch.equal(a.obs["vision"], b.obs["vision"])
    assert occluded_steps >= 15


def test_track_v2_reward(contracts):
    """#161 D3 / plan 3.8: +1 tracked, -0.3 min(|d - 1|, 3), -10 collision, -0.05 |delta a|."""
    reward = track_v2_reward(
        torch.tensor([True, False, True, False]),
        torch.tensor([1.0, 6.0, 0.3, 2.5], dtype=torch.float64),
        torch.tensor([False, False, True, False]),
        torch.tensor([[0.0, 0.0], [0.3, 0.4], [0.0, 0.0], [0.0, 0.0]]),
        1.0,
    )
    expected = torch.tensor([1.9, -0.025, 1 - 0.21 - 10, 0.45], dtype=torch.float64)
    torch.testing.assert_close(reward, expected)
    world = _world(contracts)
    assert world.reward_kind == "track_v2" and world.occlusion_view == "hide_target"
    world.reset()
    result = world.step(torch.zeros(6, 2))
    distance = result.info["distance_m"].to(torch.float64)
    manual = track_v2_reward(
        result.info["tracked"], distance, result.info["collision"], torch.zeros(6, 2), 1.0
    )
    torch.testing.assert_close(result.reward, manual.to(torch.float32))


# ------------------------------------------------------------------------------------------
# Episode numbering and seed options (#137)

NAMESPACE = 1 << 40


def _derived(world, rows, k):
    return episode_seed(world.seed, rows + world.env_index_offset, k) | world.episode_seed_namespace


def test_offset_and_namespace_shape_the_derived_seeds(contracts):
    world = _world(contracts, env_index_offset=12, episode_seed_namespace=NAMESPACE)
    rows = torch.arange(6)
    world.reset()
    seeds = world.privileged()["episode_seed"]
    assert torch.equal(seeds, _derived(world, rows, torch.zeros(6, dtype=torch.int64)))
    assert torch.all(seeds & NAMESPACE == NAMESPACE)
    # Row i of a world with offset 12 draws what row i + 12 of an offset-free world would.
    plain = _world(contracts, num_envs=18)
    plain.reset()
    assert torch.equal(seeds, plain.privileged()["episode_seed"][12:] | NAMESPACE)


def test_masked_reset_starts_the_next_episode_on_masked_rows_only(contracts):
    world = _world(contracts, env_index_offset=12, episode_seed_namespace=NAMESPACE)
    world.reset()
    before = world.privileged()["episode_seed"].clone()
    mask = torch.tensor([True, False, True, False, False, True])
    world.reset(mask)
    assert torch.equal(world.episode_index, mask.long())
    after = world.privileged()
    assert torch.equal(after["episode_seed"][~mask], before[~mask])
    assert torch.equal(after["episode_seed"], _derived(world, torch.arange(6), mask.long()))
    assert torch.all(after["time_s"][mask] == 0)


def test_full_reset_starts_a_new_episode_and_explicit_seeds_replay(contracts):
    world = _world(contracts)
    first = world.reset()
    seeds = world.privileged()["episode_seed"].clone()
    second = world.reset()
    assert torch.all(world.episode_index == 1)
    assert not torch.any(world.privileged()["episode_seed"] == seeds)
    assert not torch.equal(first["vision"], second["vision"])
    # Explicit seeds are used as given (no namespace) and leave the numbering alone.
    replay = world.reset(episode_seeds=seeds)
    assert torch.all(world.episode_index == 1)
    assert torch.equal(world.privileged()["episode_seed"], seeds)
    torch.testing.assert_close(replay["vision"], first["vision"], rtol=0, atol=0)
    named = _world(contracts, episode_seed_namespace=NAMESPACE)
    named.reset(episode_seeds=seeds)
    assert torch.equal(named.privileged()["episode_seed"], seeds)


def test_automatic_reset_uses_the_offset_and_namespace(contracts):
    world = _world(contracts, env_index_offset=12, episode_seed_namespace=NAMESPACE)
    world.reset()
    for _ in range(40):  # 2 s episodes
        result = world.step(torch.tensor([[0.2, 0.3]]).expand(6, 2))
    assert bool(result.truncated.all())
    assert torch.equal(
        world.privileged()["episode_seed"], _derived(world, torch.arange(6), torch.ones(6).long())
    )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"env_index_offset": -1},
        {"env_index_offset": 1.0},
        {"env_index_offset": True},
        {"episode_seed_namespace": -1},
        {"episode_seed_namespace": 1 << 52},
        {"episode_seed_namespace": True},
    ],
)
def test_seed_options_are_validated(contracts, kwargs):
    with pytest.raises(ValueError):
        _world(contracts, **kwargs)


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="the CUDA graphs need CUDA")
def test_graph_path_numbers_episodes_like_eager_and_recaptures(contracts):
    def make(**kw):
        return V1CompatWorld(
            contracts[0],
            64,
            [SCENARIOS[i % 4] for i in range(64)],
            {"episode_seconds": 2, "target_motion": ["curved", "wander"]},
            seed=5,
            device="cuda",
            scenario_configs=_scenario_configs(occlusion_start_s=0.5),
            env_index_offset=64,
            episode_seed_namespace=NAMESPACE,
            **kw,
        )

    fast, slow = make(), make(fused=False)
    assert fast.graph
    mask = (torch.arange(64, device="cuda") % 3) == 0
    action = torch.tensor([[0.2, 0.3]], device="cuda").expand(64, 2)
    for world in (fast, slow):
        world.reset()
    for step in range(100):
        if step == 7:
            for world in (fast, slow):
                world.reset(mask)
        if step == 50:
            fast.release_rollout_graphs()
            assert fast._graph is None
        a, b = fast.step(action), slow.step(action)
        assert torch.equal(a.info["episode_seed"], b.info["episode_seed"])
    assert torch.equal(fast.episode_index, slow.episode_index)
    assert torch.equal(fast.privileged()["episode_seed"], slow.privileged()["episode_seed"])
    assert fast._graph is not None
