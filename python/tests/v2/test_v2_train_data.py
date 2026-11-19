from collections import Counter
from copy import deepcopy

import pytest
import torch
from flytrack_v2.model import ModelConfigV2, build_policy
from flytrack_v2.train.config import CONFIG_DIR, load_config, plan_rounds, scenario_rows
from flytrack_v2.train.data import (
    EPISODE_SEED_BIT,
    Replay,
    ReplayShard,
    RolloutCollector,
    StepData,
    mix_actions,
)
from flytrack_v2.train.update import burn_in_state
from flytrack_v2.world.rng import episode_seed
from test_v2_model_core import small_graph


def test_scenarios_host_and_round_plan():
    path = CONFIG_DIR / "dagger.yaml"
    config = load_config(path, "4060")
    counts = Counter(scenario_rows(dict(reversed(list(config["scenarios"].items()))), 256))
    assert counts == {"normal": 103, "occlusion": 64, "noise": 51, "conflict": 38}
    assert config["micro_batch"] == 128
    assert config["desync_first_episode"] is True
    assert [plan.first_snapshot for plan in plan_rounds(config)] == [0, 3, 6, 9]
    assert [plan.updates for plan in plan_rounds(config)] == [184] * 4
    assert load_config(path, "2060")["micro_batch"] == 64
    assert load_config(path, "4070")["micro_batch"] == 128
    with pytest.raises(ValueError, match="divisible"):
        load_config(path, "4060", ["batch_sequences=200"])
    with pytest.raises(ValueError, match="desync_first_episode must be boolean"):
        load_config(path, "4060", ["desync_first_episode=1"])
    with pytest.raises(ValueError, match="label_margin must be <= 0.5"):
        load_config(path, "4060", ["label_margin=0.5001"])
    assert load_config(path, "4060", ["label_margin=0.5"])["label_margin"] == 0.5
    small = load_config(
        path, "4060", ["batch_sequences=8", "envs_per_instance=2", "rounds.env_steps_per_round=400"]
    )
    assert small["micro_batch"] == 8
    assert [plan.ticks for plan in plan_rounds(small)] == [80] * 4


def test_reserved_seeds_and_automatic_reset():
    config = load_config(
        CONFIG_DIR / "dagger.yaml",
        "4060",
        ["envs_per_instance=2", "episode_seconds=0.1", "desync_first_episode=false"],
    )
    policy = build_policy("VO2", ModelConfigV2(initial_state="bias"), graph=small_graph())
    collector = RolloutCollector(policy, config, seed=11, device=torch.device("cpu"))
    global_rows = torch.arange(collector.rows)
    initial = policy.initial_central_state(collector.rows, "cpu")
    for episode in range(2):
        seeds = torch.cat([world.privileged()["episode_seed"] for world in collector.worlds])
        expected = episode_seed(11, global_rows, torch.full_like(global_rows, episode))
        assert torch.equal(seeds, expected | EPISODE_SEED_BIT)
        assert seeds.min() >= EPISODE_SEED_BIT and seeds.unique().numel() == collector.rows
        assert not collector.step(0.5, initial).reset.any()
        assert collector.step(0.5, initial).reset.all()
        torch.testing.assert_close(collector.central, initial, rtol=0, atol=0)


def test_restore_releases_graphs_before_replacing_state(monkeypatch):
    config = load_config(
        CONFIG_DIR / "dagger.yaml", "4060", ["envs_per_instance=3", "episode_seconds=0.3"]
    )
    policy = build_policy("VO2", ModelConfigV2(initial_state="bias"), graph=small_graph())
    device = torch.device("cpu")
    collector = RolloutCollector(policy, config, 11, device)
    initial = policy.initial_central_state(collector.rows, device)
    for _ in range(2):
        collector.step(0.5, initial)
    # CPU checkpoints can share storage with the collector's live state.
    saved = deepcopy(collector.checkpoint())
    expected = RolloutCollector(policy, config, 11, device, restore=deepcopy(saved))
    for _ in range(3):
        collector.step(0.5, initial)

    old_states = {world: world._state for world in collector.worlds}
    world_type = type(collector.worlds[0])
    release = world_type.release_rollout_graphs
    released = []

    def record_release(world):
        assert world._state is old_states[world]
        released.append(world)
        release(world)

    monkeypatch.setattr(world_type, "release_rollout_graphs", record_release)
    collector.restore(saved)
    assert released == collector.worlds
    for _ in range(6):
        torch.testing.assert_close(
            collector.step(0.5, initial), expected.step(0.5, initial), rtol=0, atol=0
        )
    torch.testing.assert_close(
        collector.checkpoint(), expected.checkpoint(), rtol=0, atol=0, equal_nan=True
    )


def test_first_episode_reset_matches_automatic_reset():
    config = load_config(
        CONFIG_DIR / "dagger.yaml", "4060", ["envs_per_instance=3", "episode_seconds=0.3"]
    )
    policy = build_policy("VO2", ModelConfigV2(initial_state="bias"), graph=small_graph())
    device = torch.device("cpu")
    seed = 11
    collector = RolloutCollector(policy, config, seed, device)
    automatic = RolloutCollector(policy, {**config, "desync_first_episode": False}, seed, device)
    initial = policy.initial_central_state(collector.rows, device)
    sensory_initial = policy.initial_sensory_state(collector.rows, device)
    horizon = collector.worlds[0].max_steps
    offsets = collector.checkpoint()["first_episode_resets"]["steps"].flatten()
    assert (offsets == 0).any() and (offsets > 0).any()
    assert offsets.min() >= 0 and offsets.max() < horizon
    first_end = torch.where(offsets > 0, offsets, horizon)
    for tick in range(1, horizon + 1):
        assert torch.equal(
            automatic.step(1.0, initial).reset,
            torch.full_like(offsets, tick == horizon, dtype=torch.bool),
        )

    global_rows = torch.arange(collector.rows)
    seen = []
    for tick in range(2 * horizon + 1):
        first = tick == first_end
        second = tick == first_end + horizon
        if tick:
            reset = collector.step(1.0, initial).reset
            assert torch.equal(reset, first | second)
            torch.testing.assert_close(collector.central[reset], initial[reset], rtol=0, atol=0)
            columns = reset.repeat_interleave(2)
            assert torch.equal(
                collector.sensory.frontend[:, columns], sensory_initial.frontend[:, columns]
            )
            assert torch.equal(collector.sensory.odor_K[reset], sensory_initial.odor_K[reset])
            assert torch.equal(collector.teacher_state[reset], collector.teacher_initial[reset])
            for key, obs in collector.obs.items():
                assert torch.equal(obs[first], automatic.obs[key][first]), key
        seeds = torch.cat([world.privileged()["episode_seed"] for world in collector.worlds])
        # Match b1119c4: each row starts at episode 0 and advances once per reset.
        # The global row includes the instance offset, and bit 40 is set in all seeds.
        episode = (tick >= first_end).long() + (tick >= first_end + horizon).long()
        expected = episode_seed(seed, global_rows, episode) | (1 << 40)
        torch.testing.assert_close(
            seeds, expected, rtol=0, atol=0, msg=f"episode seeds at tick {tick}"
        )
        seen.append(seeds if tick == 0 else seeds[first | second])
    assert torch.cat(seen).unique().numel() == 3 * collector.rows
    saved = collector.checkpoint()["first_episode_resets"]
    assert torch.equal(saved["forced"].flatten(), offsets > 0)


def test_first_episode_natural_termination_takes_precedence():
    config = load_config(
        CONFIG_DIR / "dagger.yaml", "4060", ["envs_per_instance=3", "episode_seconds=0.3"]
    )
    policy = build_policy("VO2", ModelConfigV2(initial_state="bias"), graph=small_graph())
    collector = RolloutCollector(policy, config, 11, torch.device("cpu"))
    world = collector.worlds[0]
    offsets = collector.checkpoint()["first_episode_resets"]["steps"][0]
    # Force a wall collision on tick 1 for rows scheduled on that tick and later.
    assert offsets[0] == 1 and offsets[2] > 1
    world._state["pose"][[0, 2], 0] = world.walls[0] - 1
    initial = policy.initial_central_state(collector.rows, "cpu")
    assert collector.step(1.0, initial).reset[:3].all()
    for _ in range(1, int(offsets[2])):
        assert not collector.step(1.0, initial).reset[:3].any()
    assert torch.equal(world.episode_index, torch.ones_like(world.episode_index))
    assert collector.checkpoint()["first_episode_resets"]["forced"][0].tolist() == [
        False,
        True,
        False,
    ]


def test_beta_is_per_row_and_reproducible():
    teacher, student = torch.ones(40_000, 2), torch.zeros(40_000, 2)
    rng = torch.Generator().manual_seed(5)
    first = mix_actions(teacher, student, 0.5, rng)
    assert 0.49 <= first[:, 0].mean() <= 0.51
    assert torch.equal(first[:, 0], first[:, 1])
    assert torch.equal(first, mix_actions(teacher, student, 0.5, rng.manual_seed(5)))
    assert torch.equal(teacher, mix_actions(teacher, student, 1.0, rng))
    assert torch.equal(student, mix_actions(teacher, student, 0.0, rng))


def test_replay_starts_at_saved_state_and_resets_inside_sequence():
    policy = build_policy("VO2", ModelConfigV2(initial_state="bias"), graph=small_graph())
    ticks, rows, first_snapshot = 24, 4, 3
    code = torch.zeros(ticks, rows, policy.code_size)
    code[:, :, 0] = torch.arange(ticks)[:, None]
    reset = torch.zeros(ticks, rows, dtype=torch.bool)
    reset[4, 0] = reset[20, 0] = True
    state = torch.arange(2 * rows, dtype=code.dtype).reshape(2, rows, 1) + 1
    shard = ReplayShard(
        StepData(code, torch.zeros(ticks, rows, 2), reset, torch.zeros_like(reset)),
        state.expand(-1, -1, policy.central_state_size),
        torch.arange(rows)[:, None].expand(-1, 3),
        torch.tensor([True, False, False, False]),
        first_snapshot,
    )
    replay = Replay(burn_in=2, length=2)
    replay.shards.append(shard)
    rng = torch.Generator().manual_seed(3)
    held = replay.sample(8, rng, held_out=True)
    train = replay.sample(32, rng)
    assert set(held.code[0, :, 0].tolist()) == {3.0, 19.0}
    assert torch.equal(held.tags, torch.zeros_like(held.tags))
    assert (train.tags[:, 0] > 0).all()
    snapshot = (held.code[0, :, 0].long() - first_snapshot) // 16
    torch.testing.assert_close(held.initial, shard.states[snapshot, 0])
    initial = policy.initial_central_state(2, "cpu")
    state = burn_in_state(policy, held.code[:2], held.reset[:2], held.initial, initial)
    torch.testing.assert_close(state, initial, rtol=0, atol=0)
