import ctypes
import errno
from collections import Counter
from copy import deepcopy
from types import SimpleNamespace

import pytest
import torch
from flytrack_v2.model import ModelConfigV2, build_policy
from flytrack_v2.train import data
from flytrack_v2.train.config import (
    CONFIG_DIR,
    STATE_STRIDE,
    load_config,
    plan_rounds,
    scenario_rows,
)
from flytrack_v2.train.data import (
    EPISODE_SEED_BIT,
    Replay,
    ReplayShard,
    RolloutCollector,
    SequenceBatch,
    StepData,
    _env_major,
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
        StepData(
            _env_major(code, torch.float16),
            torch.zeros(rows, ticks, 2),
            _env_major(reset, torch.bool),
            torch.zeros(rows, ticks, dtype=torch.bool),
        ),
        _env_major(state.expand(-1, -1, policy.central_state_size), torch.float16),
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
    snapshot = (held.code[0, :, 0].long() - first_snapshot) // STATE_STRIDE
    torch.testing.assert_close(held.initial, shard.states[0, snapshot].float())
    initial = policy.initial_central_state(2, "cpu")
    state = burn_in_state(policy, held.code[:2], held.reset[:2], held.initial, initial)
    torch.testing.assert_close(state, initial, rtol=0, atol=0)


@pytest.mark.parametrize("limit", [True, False])
def test_labels_follow_the_teacher_acceleration_setting(limit):
    """#184: with the student driving, a limited label stays within one step's change of the
    student's applied action; an unlimited label is the action the teacher wants."""
    overrides = ["envs_per_instance=4", "desync_first_episode=false"]
    config = load_config(
        CONFIG_DIR / "dagger.yaml", "4060", [*overrides, f"teacher_limit_acceleration={limit}"]
    )
    assert config["teacher_limit_acceleration"] is limit
    policy = build_policy("VO2", ModelConfigV2(initial_state="bias"), graph=small_graph())
    collector = RolloutCollector(policy, config, seed=3, device=torch.device("cpu"))
    initial = policy.initial_central_state(collector.rows, "cpu")
    contract, dt = policy.contract, policy.contract.control_dt
    step_limit = torch.tensor([contract.linear_accel_max, contract.angular_accel_max]) * dt
    largest = torch.zeros(2)
    for _ in range(20):
        applied = torch.cat([world.privileged()["agent_velocity"] for world in collector.worlds])
        label = collector.step(0.0, initial).action
        largest = torch.maximum(largest, (label - applied).abs().amax(0))
    if limit:
        assert (largest <= step_limit + 1e-5).all()
    else:
        assert (largest > step_limit + 1e-3).any()


def test_configs_without_the_setting_keep_v1_labels(tmp_path):
    import yaml

    values = yaml.safe_load((CONFIG_DIR / "dagger.yaml").read_text())
    del values["teacher_limit_acceleration"]
    path = tmp_path / "old.yaml"
    path.write_text(yaml.safe_dump(values))
    assert load_config(path, "4060")["teacher_limit_acceleration"] is True
    assert load_config(CONFIG_DIR / "dagger.yaml", "4060")["teacher_limit_acceleration"] is False


def test_label_latent_bound_keeps_every_label_within_three():
    import math

    from flytrack_v2.interfaces import unsquash
    from flytrack_v2.train.update import label_margins, shrink_labels

    config = load_config(CONFIG_DIR / "dagger.yaml", "4060")
    assert config["label_latent_bound"] == 3.0
    margins = label_margins(config)
    assert margins == pytest.approx((1 / (1 + math.exp(3)), 1 - math.tanh(3)))
    extremes = torch.tensor([[0.0, -1.5], [0.5, 1.5], [0.25, 0.0]])
    latent = unsquash(shrink_labels(extremes, 0.5, 1.5, margins), 0.5, 1.5)
    assert latent[:2].abs().amax() == pytest.approx(3.0, abs=1e-4)
    assert latent[2].abs().amax() < 1e-5
    legacy = load_config(CONFIG_DIR / "dagger.yaml", "4060", ["label_latent_bound=null"])
    assert label_margins(legacy) == (1e-3, 1e-3)
    with pytest.raises(ValueError, match="label_latent_bound"):
        load_config(CONFIG_DIR / "dagger.yaml", "4060", ["label_latent_bound=0"])


def test_replay_round_trip_preserves_draw_order(tmp_path):
    rng = torch.Generator().manual_seed(42)
    replay = Replay(burn_in=2, length=4)
    originals = []
    for index, (rows, ticks, first) in enumerate(((67, 41, 3), (4, 27, 5))):
        n_snapshots = len(range(first, ticks, STATE_STRIDE))
        steps = StepData(
            torch.randn(ticks, rows, 7, generator=rng),
            torch.randn(ticks, rows, 2, generator=rng),
            torch.rand(ticks, rows, generator=rng) < 0.1,
            torch.rand(ticks, rows, generator=rng) < 0.3,
        )
        states = torch.randn(n_snapshots, rows, 9, generator=rng)
        tags = torch.stack((torch.full((rows,), index), torch.arange(rows)), dim=-1)
        held_out = tags[:, 1] % 3 == 0
        shard = ReplayShard(
            StepData(
                _env_major(steps.code, torch.float16),
                *(_env_major(value, value.dtype) for value in steps[1:]),
            ),
            _env_major(states, torch.float16),
            tags,
            held_out,
            first,
        )
        path = tmp_path / f"replay-{index}.pt"
        torch.save(shard.checkpoint(), path)
        saved = torch.load(path, weights_only=True)
        assert saved["layout"] == "env_major_f16"
        loaded = ReplayShard.load(path)
        assert all(tensor.is_contiguous() for tensor in (*loaded.steps, loaded.states))
        assert loaded.steps.code.dtype == loaded.states.dtype == torch.float16
        replay.shards.append(loaded)
        originals.append((steps, states, tags, held_out, first))

    for held_out in (False, True):
        # The old flat population enumerated starts first and rows second within each shard.
        population = [
            (steps, states, tags, row, snap, tick)
            for steps, states, tags, split, first in originals
            for snap, tick in enumerate(
                range(first, steps.code.size(0) - replay.window + 1, STATE_STRIDE)
            )
            for row in torch.where(split == held_out)[0].tolist()
        ]
        size, seed = 256, 19
        draw_rng = torch.Generator().manual_seed(seed)
        if held_out:
            indices = torch.randperm(len(population), generator=draw_rng)[:size]
        else:
            indices = torch.randint(len(population), (size,), generator=draw_rng)
            assert len(indices.unique()) < size
        picked = [population[i] for i in indices.tolist()]
        code, action, reset, occluded = (
            torch.stack(
                [
                    steps[field][tick : tick + replay.window, row]
                    for steps, _, _, row, _, tick in picked
                ],
                dim=1,
            )
            for field in range(len(StepData._fields))
        )
        initial = torch.stack([states[snap, row] for _, states, _, row, snap, _ in picked])
        tags = torch.stack([tags[row] for _, _, tags, row, _, _ in picked])
        expected = SequenceBatch(
            code.half().float(), action, reset, initial.half().float(), tags, occluded
        )
        sample_rng = torch.Generator().manual_seed(seed)
        batch = replay.sample(size, sample_rng, held_out=held_out)
        torch.testing.assert_close(batch, expected, rtol=0, atol=0)
        assert torch.equal(sample_rng.get_state(), draw_rng.get_state())


def test_madvise_failure_warns_once_and_keeps_replay_readable(tmp_path, monkeypatch):
    def madvise(*_args):
        ctypes.set_errno(errno.EINVAL)
        return -1

    monkeypatch.setattr(
        data.ctypes, "CDLL", lambda *_args, **_kwargs: SimpleNamespace(madvise=madvise)
    )
    monkeypatch.setattr(data, "_madvise_warned", False)
    code = torch.tensor([[[1.5]]], dtype=torch.float16)
    shard = ReplayShard(
        StepData(
            code,
            torch.zeros(1, 1, 2),
            torch.zeros(1, 1, dtype=torch.bool),
            torch.zeros(1, 1, dtype=torch.bool),
        ),
        code.clone(),
        torch.zeros(1, 3, dtype=torch.uint8),
        torch.tensor([False]),
        0,
    )
    paths = [tmp_path / f"replay-{i}.pt" for i in range(2)]
    for path in paths:
        torch.save(shard.checkpoint(), path)
    with pytest.warns(
        UserWarning, match=rf"madvise\(MADV_RANDOM\).*\[Errno {errno.EINVAL}\]"
    ) as caught:
        for path in paths:
            loaded = ReplayShard.load(path)
            assert torch.equal(loaded.steps.code, shard.steps.code)
            assert torch.equal(loaded.states, shard.states)
    assert len(caught) == 1


def test_restore_rejects_legacy_time_major_replay(tmp_path):
    code = torch.zeros(16, 2, 3)
    path = tmp_path / "legacy.pt"
    torch.save(
        {
            "steps": StepData(
                code,
                torch.zeros(16, 2, 2),
                torch.zeros(16, 2, dtype=torch.bool),
                torch.zeros(16, 2, dtype=torch.bool),
            )._asdict(),
            "states": torch.zeros(1, 2, 4),
            "tags": torch.zeros(2, 3, dtype=torch.uint8),
            "held_out": torch.tensor([False, True]),
            "first_snapshot": 0,
        },
        path,
    )
    with pytest.raises(ValueError, match=r"legacy time-major float32.*Start a new DAgger run"):
        ReplayShard.load(path)
