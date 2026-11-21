"""#138 A2: paired-observation DAgger (collection, replay and the paired turn loss)."""

import pytest
import torch
from flytrack_v2.eval.runner import rotate_wind, swap_odor
from flytrack_v2.interfaces import reset_rows
from flytrack_v2.model import ModelConfigV2, build_policy
from flytrack_v2.train import data, update
from flytrack_v2.train.config import CONFIG_DIR, load_config
from flytrack_v2.train.data import Replay, ReplayShard, RolloutCollector, pair_transform
from flytrack_v2.train.update import (
    PAIR_SUBSETS,
    PAIR_SUMS,
    LossSettings,
    batch_metrics,
    burn_in_state,
    finish_pair_metrics,
    sequence_loss,
)
from test_v2_model_core import small_graph

OCCLUDE_ALL = [
    "scenarios.normal=0.0", "scenarios.occlusion=1.0", "scenarios.noise=0.0",
    "scenarios.conflict=0.0", "desync_first_episode=false",
]  # fmt: skip


def policy():
    return build_policy("VO2", ModelConfigV2(initial_state="bias"), graph=small_graph())


def collector(monkeypatch, pair_weight, rows=4):
    """Every row in the occlusion scenario, occluded from 0.1 s."""
    read = data.read_yaml

    def early(path):
        values = read(path)
        return {**values, "occlusion_start_s": 0.1} if values.get("name") == "occlusion" else values

    monkeypatch.setattr(data, "read_yaml", early)
    overrides = [f"envs_per_instance={rows}", *OCCLUDE_ALL, f"pair_weight={pair_weight}"]
    config = load_config(CONFIG_DIR / "dagger.yaml", "4060", overrides)
    return RolloutCollector(policy(), config, seed=5, device=torch.device("cpu"))


def test_the_transform_is_e1s_swap_on_even_rows_and_rotation_on_odd_rows():
    obs = {
        "odor": torch.rand(6, 2), "wind": torch.randn(6, 2), "vision": torch.rand(6, 2, 3),
        "valid": torch.ones(6, 4),
    }  # fmt: skip
    shown = pair_transform(obs, 45.0)
    even, odd = slice(0, None, 2), slice(1, None, 2)
    assert torch.equal(shown["odor"][even], swap_odor(obs)["odor"][even])
    assert torch.equal(shown["wind"][even], obs["wind"][even])
    assert torch.equal(shown["odor"][odd], obs["odor"][odd])
    assert torch.equal(shown["wind"][odd], rotate_wind(90.0, 45.0)(obs)["wind"][odd])
    assert shown["vision"] is obs["vision"] and shown["valid"] is obs["valid"]
    twice = pair_transform(shown, 45.0)
    assert torch.equal(twice["odor"], obs["odor"])  # swapping twice restores the odor


def test_chemical_code_is_senses_columns_without_stepping_state():
    model = policy()
    obs = {
        "vision": torch.rand(5, 2, model.contract.n_ommatidia), "odor": torch.rand(5, 2) * 0.3,
        "wind": torch.randn(5, 2), "valid": torch.ones(5, 4),
    }  # fmt: skip
    obs["valid"][1, 2] = 0  # an invalid odor reading
    obs["valid"][2, 3] = 0  # an invalid wind reading
    state = model.initial_sensory_state(5, "cpu")
    before = state.odor_K.clone()
    code, _ = model.sense(obs, state)
    chem = model.encoder.chemical_code(obs, state)
    assert torch.equal(chem, code[:, model.encoder.chemical_columns])
    assert torch.equal(state.odor_K, before)


@pytest.mark.parametrize("weight", [0.0, 1.0])
def test_pairs_never_change_the_rollout(monkeypatch, weight):
    """The copies use the step's sensory and teacher states and advance neither."""
    plain, paired = collector(monkeypatch, 0.0), collector(monkeypatch, weight)
    initial = paired.policy.initial_central_state(paired.rows, "cpu")
    for _ in range(6):
        a, b = plain.step(0.5, initial), paired.step(0.5, initial)
        for x, y in zip(a[:4], b[:4], strict=True):
            assert torch.equal(x, y)
        assert (b.chem_pair is None) is (weight == 0)
    assert torch.equal(plain.teacher_state, paired.teacher_state)
    assert torch.equal(plain.sensory.odor_K, paired.sensory.odor_K)


def test_paired_labels_come_from_the_transformed_observation(monkeypatch):
    run = collector(monkeypatch, 1.0)
    initial = run.policy.initial_central_state(run.rows, "cpu")
    moved = torch.zeros(run.rows, dtype=torch.bool)
    for _ in range(50):  # the plume reaches the agents after about 2 s
        obs, sensory, teacher = run.obs, run.sensory, run.teacher_state
        privileged = data._concatenate([w.privileged() for w in run.worlds], data._TEACHER_KEYS)
        step = run.step(1.0, initial)
        shown = pair_transform(obs, run.policy.contract.antenna_angle_deg)
        assert torch.equal(step.chem_pair, run.policy.encoder.chemical_code(shown, sensory))
        expected, _ = run.teacher.act(
            privileged, shown, privileged["agent_velocity"], teacher, limit_acceleration=False
        )
        assert torch.equal(step.action_pair, expected)
        moved |= step.occluded & (step.action_pair[:, 1] != step.action[:, 1])
    # Once the odor is present, the occluded teacher turns on what it is shown (the bilateral
    # odor and, while the odor is present, the upwind surge): the copies' labels differ.
    assert moved[0::2].any() and moved[1::2].any()


def test_paired_replay_round_trip(monkeypatch, tmp_path):
    from flytrack_v2.train.config import RoundPlan

    run = collector(monkeypatch, 1.0)
    shard, _, _ = run.collect(RoundPlan(beta=1.0, ticks=40, first_snapshot=0, updates=1))
    assert shard.steps.chem_pair.dtype == torch.float16
    path = tmp_path / "replay.pt"
    torch.save(shard.checkpoint(), path)
    assert torch.load(path, weights_only=True)["layout"] == "env_major_f16_paired"
    loaded = ReplayShard.load(path)
    assert torch.equal(loaded.steps.chem_pair, shard.steps.chem_pair)
    replay = Replay(burn_in=4, length=8)
    replay.shards.append(loaded)
    batch = replay.sample(5, torch.Generator().manual_seed(0))
    assert batch.chem_pair.shape == (12, 5, 4) and batch.action_pair.shape == (12, 5, 2)


def batch_from(run, ticks=24):
    from flytrack_v2.train.config import RoundPlan

    shard, _, _ = run.collect(RoundPlan(beta=1.0, ticks=ticks, first_snapshot=0, updates=1))
    replay = Replay(burn_in=4, length=ticks - 4)
    replay.shards.append(shard)
    return replay.sample(8, torch.Generator().manual_seed(1))


def test_the_pair_loss_is_zero_weight_neutral_and_has_gradients(monkeypatch):
    run = collector(monkeypatch, 1.0)
    model, batch = run.policy, batch_from(run)
    initial = model.initial_central_state(1, "cpu")
    plain = sequence_loss(model, batch, initial, LossSettings(4, 1e-4, 1e-3))
    off = sequence_loss(model, batch, initial, LossSettings(4, 1e-4, 1e-3, pair_weight=0.0))
    on = sequence_loss(model, batch, initial, LossSettings(4, 1e-4, 1e-3, pair_weight=2.0))
    assert torch.equal(plain["loss"], off["loss"]) and "pair_loss" not in off
    assert on["pair_steps"] > 0 and on["pair_loss"] > 0
    torch.testing.assert_close(on["loss"], plain["loss"] + 2.0 * on["pair_loss"])
    on["loss"].backward()
    gains = [p for name, p in model.named_parameters() if "gain" in name]
    assert gains and any(p.grad is not None and p.grad.abs().sum() > 0 for p in gains)


def test_untransformed_copies_add_nothing(monkeypatch):
    run = collector(monkeypatch, 1.0)
    model, batch = run.policy, batch_from(run)
    same = batch._replace(
        chem_pair=batch.code[..., model.encoder.chemical_columns].clone(),
        action_pair=batch.action.clone(),
    )
    values = sequence_loss(
        model, same, model.initial_central_state(1, "cpu"), LossSettings(4, 1e-4, 1e-3, 1.0)
    )
    assert values["pair_steps"] == 0 and values["pair_loss"] == 0


def test_pair_slope_is_one_when_the_student_follows_the_teacher():
    sums = {"pair_loss": torch.tensor(0.0), "pair_steps": torch.tensor(0.5)}
    for kind in PAIR_SUMS:
        for subset in PAIR_SUBSETS:
            values = (8.0, 8.0, 8.0, 4.0) if kind == "odor" else (0.0, 2.0, 0.0, 2.0)
            names = ("sxy", "sxx", "syy", "n")
            for name, value in zip(names, values, strict=True):
                sums[f"pair_{kind}_{subset}_{name}"] = torch.tensor(value)
    out = finish_pair_metrics(sums)
    assert out["pair_odor_slope"] == 1 and out["pair_wind_slope"] == 0
    assert out["pair_odor_present_d9_16_slope"] == 1
    assert out["pair_odor_teacher_rms"] == out["pair_odor_student_rms"] == 2**0.5
    assert out["pair_wind_student_rms"] == 0 and "pair_odor_all_sxy" not in out


def test_the_branch_forks_every_window_and_carries_its_own_state(monkeypatch):
    run = collector(monkeypatch, 1.0)
    model, batch = run.policy, batch_from(run)
    settings = LossSettings(4, 1e-4, 1e-3, 1.0, pair_window=8)
    captured = {}
    real = update._pair_loss

    def spy(dist, latent, pair_means, rows, *rest):
        captured.update(means=torch.stack(pair_means), rows=rows)
        return real(dist, latent, pair_means, rows, *rest)

    monkeypatch.setattr(update, "_pair_loss", spy)
    initial = model.initial_central_state(1, "cpu")
    with torch.no_grad():
        values = sequence_loss(model, batch, initial, settings)
        rows, columns = captured["rows"], model.encoder.chemical_columns
        assert len(rows) and values["pair_case_present"] + values["pair_case_lost"] > 0
        state = burn_in_state(model, batch.code[:4], batch.reset[:4], batch.initial, initial)
        expected = []
        for tick in range(len(batch.code) - 4):
            code, ended = batch.code[4 + tick], batch.reset[4 + tick]
            if tick % 8 == 0:
                branch = state[rows]
            shown = code[rows].clone()
            held = batch.occluded[4 + tick, rows, None]
            shown[:, columns] = torch.where(
                held, batch.chem_pair[4 + tick, rows], shown[:, columns]
            )
            step = model.forward_code(shown, branch)
            expected.append(step.dist.mean)
            branch = reset_rows(step.central_state, initial, ended[rows])
            state = reset_rows(model.forward_code(code, state).central_state, initial, ended)
    torch.testing.assert_close(captured["means"], torch.stack(expected), rtol=0, atol=0)


def test_micro_batches_keep_the_pair_metrics(monkeypatch):
    run = collector(monkeypatch, 1.0)
    model, batch = run.policy, batch_from(run)
    settings = LossSettings(4, 1e-4, 1e-3, 1.0)
    whole = finish_pair_metrics(batch_metrics(model, batch, settings, 8, backward=False))
    split = finish_pair_metrics(batch_metrics(model, batch, settings, 4, backward=False))
    for key in ("pair_odor_slope", "pair_wind_slope", "pair_odor_teacher_rms"):
        torch.testing.assert_close(whole[key], split[key])
