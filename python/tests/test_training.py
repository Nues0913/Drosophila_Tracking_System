"""Algorithmic checks to run later; no downloaded data or trained artifact required."""

import importlib.util
import math
import random
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from flytrack_core.metrics import EpisodeMetrics
from flytrack_train.batches import LaneSchedule, reset_state, tensor_observation
from flytrack_train.distributions import BoundedGaussian, generalized_advantage
from flytrack_train.es import centered_ranks
from flytrack_train.evaluation import bootstrap_mean, seed_statistics
from flytrack_train.imitation import parameter_groups
from flytrack_train.recovery import Recovery, capture_rng, restore_rng


@pytest.mark.parametrize("exit_code", [0, 2])
def test_experiment_stage_dn_gate_propagates_diagnostic_failure(tmp_path, monkeypatch, exit_code):
    # Import only: never invoke the experiment runner or a training command.
    path = Path(__file__).resolve().parents[2] / "scripts/run_experiments.py"
    spec = importlib.util.spec_from_file_location("experiment_runner", path)
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    commands = []

    def run(command, **kwargs):
        commands.append(command)
        return SimpleNamespace(returncode=exit_code)

    monkeypatch.setattr(runner.subprocess, "run", run)
    assert runner.check_dn_health("python", "config.yaml", tmp_path, "bc", {}) == exit_code
    command = commands[0]
    assert Path(command[1]).name == "dn_health.py"
    assert "--require-healthy" in command
    assert command[command.index("--stage") + 1] == "bc"
    assert command[command.index("--artifact") + 1] == str(tmp_path / "actor")


def test_physical_mean_survives_sigmoid_and_tanh_mapping():
    mean = torch.tensor([[0.13, -0.4], [0.42, 1.2]], dtype=torch.float64)
    distribution = BoundedGaussian(mean, torch.zeros(2, dtype=torch.float64), 0.5, 1.5)
    assert torch.allclose(distribution.physical(distribution.normal.loc), mean, atol=1e-10)


def test_log_probability_contains_physical_jacobian():
    distribution = BoundedGaussian(
        torch.tensor([[0.25, 0.0]], dtype=torch.float64),
        torch.zeros(2, dtype=torch.float64),
        0.5,
        1.5,
    )
    latent = torch.tensor([[0.2, -0.8]], dtype=torch.float64)
    numerical_logdet = sum(
        math.log(scale * (1 - math.tanh(value) ** 2))
        for scale, value in zip((0.25, 1.5), latent[0].tolist(), strict=True)
    )
    expected = distribution.normal.log_prob(latent).sum(-1) - numerical_logdet
    assert torch.allclose(distribution.log_prob_latent(latent), expected)
    assert torch.isfinite(
        distribution.log_prob_latent(torch.tensor([[30.0, -30.0]], dtype=torch.float64))
    ).all()


def test_gae_bootstraps_truncation_without_crossing_reset():
    rewards = torch.tensor([[1.0], [100.0]])
    values = torch.tensor([[2.0], [3.0]])
    next_values = torch.tensor([[10.0], [5.0]])
    terminated = torch.tensor([[False], [True]])
    truncated = torch.tensor([[True], [False]])
    advantages, returns = generalized_advantage(
        rewards, values, next_values, terminated, truncated, 0.9, 0.95
    )
    assert advantages[:, 0].tolist() == pytest.approx([8.0, 97.0])
    assert returns[:, 0].tolist() == pytest.approx([10.0, 100.0])


def test_memory_and_actor_observation_isolation():
    state = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    assert reset_state(state, torch.tensor([True, False])).tolist() == [[0.0, 0.0], [3.0, 4.0]]
    initial = torch.full_like(state, 0.3)
    assert torch.allclose(
        reset_state(state, torch.tensor([True, False]), initial),
        torch.tensor([[0.3, 0.3], [3.0, 4.0]]),
    )
    observation = {
        "vision": np.ones((2, 7), dtype=np.float32),
        "odor": np.ones(2, dtype=np.float32),
        "valid": np.ones(3, dtype=np.float32),
        "privileged_state": np.arange(12),
        "teacher_action": np.ones(2),
    }
    assert set(tensor_observation([observation], torch.device("cpu"))) == {
        "vision",
        "odor",
        "valid",
    }


def test_seed_level_statistics_do_not_weight_large_seed_sample_more():
    def row(seed, success):
        return {
            "variant": "VO",
            "scenario": "normal",
            "seed": seed,
            "success": success,
            "mean_distance_m": 1,
            "mean_bearing_error_deg": 0,
            "reacquisition_time_s": float("nan"),
            "smoothness_rad_s": 0,
            "collision_rate": 0,
            "latency_p50_ms": 1,
            "latency_p95_ms": 2,
        }

    _, summary, _ = seed_statistics([row(0, 0)] + [row(1, 1)] * 100, np.random.default_rng(3), 100)
    success = next(record for record in summary if record["metric"] == "success")
    assert success["mean"] == 0.5
    assert success["n_seeds"] == 2
    assert bootstrap_mean([float("nan")], np.random.default_rng(1))["n_seeds"] == 0


def test_reacquisition_reports_unfinished_events_as_censored():
    metrics = EpisodeMetrics()
    metrics.append(1, 0.7, 0, False, True, 0)
    metrics.append(1, 0.7, 0.1, False, False, 1)
    metrics.append(1, 0.1, 0.2, False, False, 1.25)
    metrics.append(1, 0.7, 0.2, False, True, 2)
    result = metrics.result()
    assert result["reacquisition_time_s"] == 0.25
    assert result["censored_recaptures"] == 1
    assert result["success"] == 1


def test_es_tied_fitness_has_no_perturbation_preference():
    assert np.array_equal(centered_ranks(np.ones((4, 2))), np.zeros((4, 2)))


def test_lane_tbptt_preserves_chronology_and_covers_every_window_once():
    lengths = [9, 12, 4, 7, 5]
    schedule = LaneSchedule(lengths, [3, 0, 4, 1, 2], lanes=2, length=4)
    seen: dict[int, list[tuple[int, int]]] = {}
    lane_of: dict[int, int] = {}
    while (window := schedule.next()) is not None:
        for lane, item in enumerate(window):
            if item is None:
                continue
            episode, start, end, begins = item
            assert begins == (episode not in seen)
            # An episode stays on one lane, so its recurrent state is carried.
            assert lane_of.setdefault(episode, lane) == lane
            seen.setdefault(episode, []).append((start, end))
    for episode, length in enumerate(lengths):
        assert seen[episode] == [(start, min(start + 4, length)) for start in range(0, length, 4)]


def test_lockstep_collection_is_independent_of_slot_count_and_truncates_exactly():
    from flytrack_train.collection import collect_lockstep
    from flytrack_train.parallel import EnvPool

    def act(running, observations, infos):
        actions = {}
        for slot, entry in running.items():
            entry.observations.append({k: observations[slot][k].copy() for k in observations[slot]})
            entry.labels.append({"action": np.asarray([entry.rng.random(), 0.0], np.float32)})
            actions[slot] = entry.labels[-1]["action"]
        return actions

    def finish(entry):
        from flytrack_train.batches import pack_episode

        return pack_episode(entry.observations, entry.labels)

    datasets = []
    for slots in (1, 3):
        collected = []
        pool = EnvPool([_RecoveryEnv for _ in range(slots)], workers=0)
        steps = collect_lockstep(
            pool,
            seed=5,
            stream=1,
            first_index=0,
            committed_steps=0,
            step_budget=8,
            act=act,
            finish=finish,
            on_episode=lambda index, episode, steps, collected=collected: collected.append(
                (index, episode, steps)
            ),
        )
        pool.close()
        assert steps == 8
        datasets.append(collected)
    assert [(i, s, len(e)) for i, e, s in datasets[0]] == [(0, 3, 3), (1, 6, 3), (2, 8, 2)]
    for (_, left, _), (_, right, _) in zip(*datasets, strict=True):
        for key in left.observations:
            assert np.array_equal(left.observations[key], right.observations[key])
        assert np.array_equal(left.labels["action"], right.labels["action"])


def test_early_stopping_keeps_held_out_best_parameters():
    from flytrack_train.batches import SequenceModel
    from flytrack_train.recovery import fit_sequence_epochs

    class _RestingStatePolicy(_RecoveryPolicy):
        # Like the connectome, the initial state depends on a trainable parameter.
        def __init__(self):
            super().__init__()
            self.rest = torch.nn.Parameter(torch.full((2,), 0.1))

        def initial_state(self, batch, device):
            return self.rest.expand(batch, -1).clone()

    torch.manual_seed(0)
    policy = _RestingStatePolicy()
    model = SequenceModel(policy, "cpu")
    episodes = [
        {
            "vision": torch.zeros(6, 2, 3),
            "odor": torch.randn(6, 2),
            "valid": torch.ones(6, 3),
        }
        for _ in range(10)
    ]
    labels = [{"target": torch.zeros(6, 2)} for _ in episodes]
    optimizer = torch.optim.SGD(policy.parameters(), lr=5.0)  # deliberately unstable
    seen = []

    def window_loss(outputs, targets, mask):
        error = (outputs["action"] - targets["target"]).square().mean(-1)
        return (error * mask).sum(0) / mask.sum(0).clamp_min(1)

    reports = fit_sequence_epochs(
        model,
        episodes,
        labels,
        {"epochs": 8, "sequence_length": 3, "batch_episodes": 4, "validation_every": 5},
        optimizer,
        list(policy.parameters()),
        np.random.default_rng(0),
        window_loss,
        lambda state: seen.append(state["best"]),
        lambda **values: None,
    )
    best = min(reports, key=lambda report: report["validation_loss"])
    assert all(report["best_epoch"] <= best["epoch"] for report in reports)
    restored = seen[-1]["parameters"]
    for parameter, value in zip(policy.parameters(), restored, strict=True):
        assert torch.equal(parameter.detach(), value)
    assert len(reports) == 8 or reports[-1]["epoch"] - best["epoch"] == 3


def test_recovery_is_bound_to_the_parent_artifact(tmp_path):
    parent = tmp_path / "pretrain" / "actor"
    parent.mkdir(parents=True)
    (parent / "model_card.json").write_text('{"files": {"weights.pt": "aaa"}}')
    config = {"init_artifact": str(parent), "train": {}}
    output = tmp_path / "bc"
    Recovery(output, config, "bc", 0).save(np.random.default_rng(0), phase="fit")
    Recovery(output, {**config, "resume": True}, "bc", 0)
    # A rerun parent stage writes a new card at the same path: the old BC must not resume.
    (parent / "model_card.json").write_text('{"files": {"weights.pt": "bbb"}}')
    with pytest.raises(ValueError, match="source hashes differ"):
        Recovery(output, {**config, "resume": True}, "bc", 0)
    (parent / "model_card.json").unlink()
    with pytest.raises(FileNotFoundError, match="Parent actor of this stage is missing"):
        Recovery(output, {**config, "resume": True}, "bc", 0)


def test_recovery_restores_all_cpu_random_streams_without_cuda_init(tmp_path):
    rng = np.random.default_rng(31)
    state = capture_rng(rng)
    expected = (random.random(), np.random.random(), rng.random(), torch.rand(3))
    restore_rng(state, rng)
    observed = (random.random(), np.random.random(), rng.random(), torch.rand(3))
    assert expected[:3] == observed[:3]
    assert torch.equal(expected[3], observed[3])
    recovery = Recovery(tmp_path, {}, "unit", 31)
    recovery.save(rng, environment_steps=7)
    restored = Recovery(tmp_path, {"resume": True}, "unit", 31)
    assert restored.checkpoint["environment_steps"] == 7
    with pytest.raises(ValueError, match="configuration or source hashes differ"):
        Recovery(tmp_path, {"resume": True, "train": {"total_steps": 1}}, "unit", 31)
    with pytest.raises(FileExistsError, match="--resume"):
        Recovery(tmp_path, {}, "unit", 31)


class _RecoveryEnv:
    """Small stochastic world to prove reset/replay and training continuation."""

    def __init__(self):
        self._episode_id = -1

    def reset(self, *, seed):
        self.rng = np.random.default_rng(seed)
        self.position = self.rng.normal(size=2)
        self.steps = 0
        self._episode_id += 1
        return self._observe()

    def _observe(self):
        observation = {
            "vision": np.zeros((2, 3), np.float32),
            "odor": self.position.astype(np.float32),
            "valid": np.ones(3, np.float32),
        }
        info = {
            "privileged_state": np.pad(
                np.asarray([*self.position, self.steps], np.float32), (0, 9)
            ),
            "teacher_action": np.asarray([0.2, 0.0], np.float32),
            "target_bearing_rad": float(np.arctan2(self.position[0], self.position[1])),
            "target_visible": True,
        }
        return observation, info

    def step(self, action):
        self.position += np.asarray(action) * 0.01 + self.rng.normal(0, 0.01, 2)
        self.steps += 1
        observation, info = self._observe()
        return observation, float(-np.square(self.position).sum()), False, self.steps == 3, info

    def close(self):
        pass


class _RecoveryPolicy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(4, 2)
        self.log_std = torch.nn.Parameter(torch.full((2,), -1.0))
        self.population_masks = {
            key: torch.ones(2, dtype=torch.bool) for key in ("visual", "olfactory", "convergence")
        }

    def initial_state(self, batch, device):
        return torch.zeros(batch, 2, device=device)

    def forward(self, observation, state):
        hidden = torch.tanh(self.linear(torch.cat((observation["odor"], state), dim=1)))
        action = torch.stack(
            (torch.sigmoid(hidden[:, 0]) * 0.5, torch.tanh(hidden[:, 1]) * 1.5), dim=1
        )
        return action, hidden, hidden


def _assert_nested_equal(left, right):
    if isinstance(left, torch.Tensor):
        assert torch.equal(left, right)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            _assert_nested_equal(left[key], right[key])
    elif isinstance(left, (list, tuple)):
        assert len(left) == len(right)
        for a, b in zip(left, right, strict=True):
            _assert_nested_equal(a, b)
    else:
        assert left == right


@pytest.mark.parametrize("value_clip", ["omitted", None, 0.2, 1])
def test_ppo_value_clipping_is_independent_of_actor_clip(tmp_path, monkeypatch, value_clip):
    from flytrack_core.contracts import SensorContract
    from flytrack_train import ppo

    monkeypatch.setattr(ppo, "prepare_policy", lambda *args: (_RecoveryPolicy(), SensorContract()))
    monkeypatch.setattr(ppo, "make_env", lambda *args, **kwargs: _RecoveryEnv())
    monkeypatch.setattr(ppo, "save_actor", lambda *args, **kwargs: None)
    observed = []
    original_loss = ppo._critic_loss

    def record_loss(value, target, old_value, limit):
        observed.append(limit)
        return original_loss(value, target, old_value, limit)

    monkeypatch.setattr(ppo, "_critic_loss", record_loss)
    train = {
        "num_envs": 1,
        "total_steps": 2,
        "rollout_steps": 2,
        "sequence_length": 2,
        "sequence_batch_size": 1,
        "update_epochs": 1,
        "critic_hidden_size": 4,
        "clip_coef": 0.1,
    }
    if value_clip != "omitted":
        train["value_clip"] = value_clip
    ppo.train_ppo({"train": train}, tmp_path, 11, torch.device("cpu"))
    assert observed == [None if value_clip == "omitted" else value_clip]


def test_ppo_normalization_keeps_episode_metrics_in_raw_reward_units(tmp_path, monkeypatch):
    from flytrack_core.contracts import SensorContract
    from flytrack_train import ppo

    monkeypatch.setattr(ppo, "prepare_policy", lambda *args: (_RecoveryPolicy(), SensorContract()))
    monkeypatch.setattr(ppo, "make_env", lambda *args, **kwargs: _RecoveryEnv())
    monkeypatch.setattr(ppo, "save_actor", lambda *args, **kwargs: None)
    running = np.zeros(2)
    raw_returns, expected_reports, scaled_rewards = [], [], []
    original_normalize = ppo.RewardNormalizer.normalize

    def record_raw_rewards(self, rewards, terminated, truncated):
        running[:] += rewards
        for index, done in enumerate(np.asarray(terminated) | np.asarray(truncated)):
            if done:
                raw_returns.append(float(running[index]))
                running[index] = 0
        scaled = original_normalize(self, rewards, terminated, truncated)
        scaled_rewards.append(scaled)
        if len(scaled_rewards) % 2 == 0:
            expected_reports.append(float(np.mean(raw_returns[-20:])) if raw_returns else None)
        return scaled

    monkeypatch.setattr(ppo.RewardNormalizer, "normalize", record_raw_rewards)
    config = {
        "train": {
            "num_envs": 2,
            "total_steps": 12,
            "rollout_steps": 2,
            "sequence_length": 2,
            "sequence_batch_size": 2,
            "update_epochs": 1,
            "critic_hidden_size": 4,
        }
    }
    ppo.train_ppo(config, tmp_path, 11, torch.device("cpu"))
    checkpoint = torch.load(tmp_path / "training_checkpoint.pt", weights_only=True)
    assert checkpoint["episode_returns"] == raw_returns
    assert [row["mean_last_20_episode_return"] for row in checkpoint["reports"]] == expected_reports
    assert checkpoint["reward_normalizer"]["count"] == 12.0001
    for row in checkpoint["reports"]:
        assert row["actor_grad_norm_clipped"] <= 0.5 + 1e-7
        assert row["critic_grad_norm_clipped"] <= 0.5 + 1e-7


@pytest.mark.parametrize(
    "algorithm,interrupt_phase",
    [
        ("ppo", None),
        ("es", None),
        ("pretrain", "collect"),
        ("pretrain", "fit"),
        ("pretrain_population_rms", "collect"),
        ("pretrain_population_rms", "fit"),
        ("bc", "collect"),
        ("bc", "fit"),
    ],
)
def test_interrupted_training_matches_uninterrupted_weights_optimizer_and_rng(
    tmp_path, monkeypatch, algorithm, interrupt_phase
):
    from flytrack_core.contracts import SensorContract
    from flytrack_train import es, imitation, ppo, pretrain

    scaled = algorithm == "pretrain_population_rms"
    if scaled:
        algorithm = "pretrain"
    module = {"ppo": ppo, "es": es, "pretrain": pretrain, "bc": imitation}[algorithm]
    monkeypatch.setattr(
        module,
        "prepare_policy",
        lambda config, device: (_RecoveryPolicy().to(device), SensorContract()),
    )
    monkeypatch.setattr(module, "make_env", lambda *args, **kwargs: _RecoveryEnv())
    monkeypatch.setattr(module, "save_actor", lambda *args, **kwargs: None)
    # ES delegates environments through its own imported fixed-budget helper.
    config = {
        "train": {
            "num_envs": 2,
            "total_steps": 12,
            "rollout_steps": 2,
            "sequence_length": 2,
            "sequence_batch_size": 2,
            "update_epochs": 1,
            "critic_hidden_size": 4,
            "population_size": 1,
            "evaluation_steps": 2,
            "epochs": 2,
            "checkpoint_windows": 1,
            "modality": "vision",
            "teacher_steps": 6,
            "dagger_steps_per_round": 3,
            "dagger_rounds": 2,
        }
    }
    if scaled:
        config["train"]["decoder_input_scale"] = "population_rms"
    run = {
        "ppo": ppo.train_ppo,
        "es": es.train_es,
        "pretrain": pretrain.train_pretrain,
        "bc": imitation.train_bc,
    }[algorithm]
    full, interrupted = tmp_path / "full", tmp_path / "interrupted"
    run(config, full, 11, torch.device("cpu"))
    original_save = Recovery.save

    def stop_after_first_checkpoint(self, rng, **state):
        original_save(self, rng, **state)
        if (
            self.output == interrupted
            and (interrupt_phase is None or state.get("phase") == interrupt_phase)
            and (not scaled or interrupt_phase != "fit" or state.get("fit_state") is not None)
        ):
            raise InterruptedError("simulated process interruption after committed checkpoint")

    monkeypatch.setattr(Recovery, "save", stop_after_first_checkpoint)
    with pytest.raises(InterruptedError):
        run(config, interrupted, 11, torch.device("cpu"))
    monkeypatch.setattr(Recovery, "save", original_save)
    if scaled and interrupt_phase == "fit":
        original_initialize = pretrain.initialize_decoder_inputs

        def require_saved_statistics(decoder, *args):
            assert bool(decoder.input_initialized)
            original_initialize(decoder, *args)

        saved = torch.load(interrupted / "training_checkpoint.pt", weights_only=True)
        assert bool(saved["decoder"]["input_initialized"])
        monkeypatch.setattr(pretrain, "initialize_decoder_inputs", require_saved_statistics)
    run({**config, "resume": True}, interrupted, 11, torch.device("cpu"))
    expected = torch.load(full / "training_checkpoint.pt", weights_only=True)
    observed = torch.load(interrupted / "training_checkpoint.pt", weights_only=True)
    for key in ("actor", "optimizer", "rng", "environment_steps"):
        _assert_nested_equal(expected[key], observed[key])
    if algorithm == "ppo":
        for key in (
            "critic",
            "reward_normalizer",
            "recurrent_state",
            "observations",
            "privileged",
            "episode_start",
            "env_seeds",
            "action_histories",
        ):
            _assert_nested_equal(expected[key], observed[key])
    elif algorithm == "es":
        _assert_nested_equal(expected["perturbation_generator"], observed["perturbation_generator"])
    else:
        assert expected["dataset_count"] == observed["dataset_count"]
        if algorithm == "pretrain":
            _assert_nested_equal(expected["decoder"], observed["decoder"])
            if scaled:
                research = torch.load(interrupted / "pretraining_checkpoint.pt", weights_only=True)
                for key in ("input_rms", "input_convergence", "input_initialized"):
                    assert torch.equal(research["decoder"][key], observed["decoder"][key])
                    if interrupt_phase == "fit":
                        assert torch.equal(saved["decoder"][key], observed["decoder"][key])


def test_bc_parameter_groups_override_core_bias_and_readout_rates():
    policy = torch.nn.Module()
    policy.core = torch.nn.Module()
    policy.core.bias = torch.nn.Parameter(torch.zeros(3))
    policy.core.raw_alpha = torch.nn.Parameter(torch.zeros(3))
    policy.readout = torch.nn.Module()
    policy.readout.turn_weight = torch.nn.Parameter(torch.zeros(2))
    (single,) = parameter_groups(policy, {"learning_rate": 1e-3})
    assert single["lr"] == 1e-3 and len(single["params"]) == 3
    groups = parameter_groups(
        policy, {"learning_rate": 1e-3, "core_bias_learning_rate": 0, "readout_learning_rate": 1e-2}
    )
    by_rate = {group["lr"]: group["params"] for group in groups}
    assert by_rate[0.0] == [policy.core.bias]
    assert by_rate[1e-3] == [policy.core.raw_alpha]
    assert by_rate[1e-2] == [policy.readout.turn_weight]
    with pytest.raises(ValueError, match="nonnegative"):
        parameter_groups(policy, {"core_bias_learning_rate": -1})


def test_environment_audit_flags_the_blind_circling_world():
    from flytrack_core.contracts import SensorContract
    from flytrack_train.audit import audit_environment

    config = {"env": {"episode_seconds": 60, "target_motion": "curved"}}
    contract = SensorContract(eye_radius=2)
    # The pre-2026-10-17 world and success rule: a constant left turn "tracks".
    legacy = audit_environment(
        config, contract, episodes=4, layout_preset="legacy", success_bearing_deg=None
    )
    assert not legacy["passed"]
    assert legacy["blind_policies"]["left_0.10"]["success_rate"] == 1.0
    assert legacy["teacher"]["success_rate"] == 1.0
    varied = audit_environment(config, contract, episodes=4)
    assert varied["passed"], varied["failures"]
    assert max(row["success_rate"] for row in varied["blind_policies"].values()) == 0


def test_scenario_parameters_have_one_source():
    import yaml
    from flytrack_core.contracts import SensorContract
    from flytrack_core.world import ScenarioSpec
    from flytrack_train.worlds import make_env

    # The code defaults equal the YAML files, so no world can drift from them...
    for name in ("normal", "occlusion", "noise", "conflict"):
        values = yaml.safe_load(Path(f"configs/scenarios/{name}.yaml").read_text())
        spec = ScenarioSpec(name=name).to_dict()
        assert {k: spec[k] for k in values if k != "duration_s"} == {
            k: v for k, v in values.items() if k != "duration_s"
        }
    # ...and a corrupted scenario without its YAML is refused, not silently defaulted.
    contract = SensorContract(eye_radius=2)
    config = {"env": {"episode_seconds": 1}}
    make_env(config, contract, scenario="normal").close()
    with pytest.raises(ValueError, match="scenario_configs.noise"):
        make_env(config, contract, scenario="noise")
    loaded = {"scenario_configs": {"conflict": "configs/scenarios/conflict.yaml"}, **config}
    env = make_env(loaded, contract, scenario="conflict")
    assert (env.scenario_spec.decoy_x, env.scenario_spec.decoy_y) == (-2, 0)
    env.close()


@pytest.mark.parametrize("algorithm", ["ppo", "es"])
def test_zero_core_bias_rate_freezes_biases_after_bc(tmp_path, algorithm):
    """#4: a bias freeze set for pretraining and BC must hold in the final stage too."""
    import copy

    from flytrack_core.model import load_policy_artifact
    from flytrack_train import es, ppo
    from flytrack_train.config import prepare_policy, read_config

    config = copy.deepcopy(read_config("configs/train/smoke.yaml"))
    config.pop("init_artifact", None)
    if algorithm == "es":
        config["train"].update(population_size=2, evaluation_steps=8, total_steps=32)
    config["train"]["core_bias_learning_rate"] = 0.0
    torch.manual_seed(0)
    initial, _ = prepare_policy(config, torch.device("cpu"))
    torch.manual_seed(0)
    {"ppo": ppo.train_ppo, "es": es.train_es}[algorithm](config, tmp_path, 0, torch.device("cpu"))
    trained = load_policy_artifact(tmp_path / "actor", device="cpu")[0]
    assert torch.equal(trained.core.bias, initial.core.bias)
    assert not torch.equal(trained.core.raw_alpha, initial.core.raw_alpha)
    if algorithm == "es":
        with pytest.raises(ValueError, match="only 0"):
            es.train_es(
                {**config, "train": {**config["train"], "core_bias_learning_rate": 1e-4}},
                tmp_path / "rejected",
                0,
                torch.device("cpu"),
            )


@pytest.mark.parametrize("keep_going", [False, True])
def test_runner_failed_chain_stops_others_unless_keep_going(tmp_path, monkeypatch, keep_going):
    import sys

    import yaml

    path = Path(__file__).resolve().parents[2] / "scripts/run_experiments.py"
    spec = importlib.util.spec_from_file_location("experiment_runner", path)
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    budgets = {
        "pretrain_vision": {"total_steps": 1},
        "pretrain_odor": {"total_steps": 1},
        "bc": {"teacher_steps": 1, "dagger_rounds": 0, "dagger_steps_per_round": 1},
        "ppo": {"total_steps": 1},
    }
    for variant, first in (("O", "pretrain_odor"), ("V", "pretrain_vision")):
        for stage in (first, "bc", "ppo"):
            config = {
                "variant": variant,
                "seed": 0,
                "output": str(tmp_path / "runs" / variant / stage),
                "train": budgets[stage],
            }
            target = tmp_path / "configs" / variant / "seed_0" / f"{stage}.yaml"
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(yaml.safe_dump(config))
    finished, launched = set(), []

    def run(command, **kwargs):
        if "--config" in command and "validate-pretrain" not in command:
            output = yaml.safe_load(Path(command[command.index("--config") + 1]).read_text())
            launched.append(output["output"])
            if "/O/" in output["output"]:
                return SimpleNamespace(returncode=1)
            finished.add(output["output"])
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(runner.subprocess, "run", run)
    monkeypatch.setattr(
        runner, "completed", lambda config, command, device: config["output"] in finished
    )
    monkeypatch.setattr(runner, "check_dn_health", lambda *args: 0)
    argv = ["run_experiments.py", "--configs", str(tmp_path / "configs"), "--variants", "O", "V"]
    argv += ["--seeds", "0", "--jobs", "1", "--skip-environment-audit"]
    monkeypatch.setattr(sys, "argv", argv + (["--keep-going"] if keep_going else []))
    with pytest.raises(SystemExit, match="Stage failed"):
        runner.main()
    v_stages = [output for output in launched if "/V/" in output]
    assert len(v_stages) == (3 if keep_going else 0)


def _patch_small_pretrain(monkeypatch):
    from flytrack_core.contracts import SensorContract
    from flytrack_train import pretrain

    class BiasedPolicy(_RecoveryPolicy):
        def __init__(self):
            super().__init__()
            self.core = torch.nn.Module()
            self.core.bias = torch.nn.Parameter(torch.full((2,), 0.1))

        def initial_state(self, batch, device):
            return self.core.bias.expand(batch, -1).clone()

    monkeypatch.setattr(
        pretrain, "prepare_policy", lambda *args: (BiasedPolicy(), SensorContract())
    )
    monkeypatch.setattr(pretrain, "make_env", lambda *args: _RecoveryEnv())
    monkeypatch.setattr(pretrain, "save_actor", lambda *args, **kwargs: None)
    return pretrain, {
        "train": {
            "modality": "vision",
            "total_steps": 6,
            "epochs": 1,
            "sequence_length": 2,
            "batch_episodes": 2,
        }
    }


@pytest.mark.parametrize("value", ["standardize", "POPULATION_RMS", True, 1, None, []])
def test_pretrain_decoder_scale_rejects_invalid_options(tmp_path, monkeypatch, value):
    pretrain, config = _patch_small_pretrain(monkeypatch)
    config["train"]["decoder_input_scale"] = value
    with pytest.raises(ValueError, match="decoder_input_scale"):
        pretrain.train_pretrain(config, tmp_path, 0, torch.device("cpu"))


def test_pretrain_explicit_none_matches_default_bitwise(tmp_path, monkeypatch):
    pretrain, config = _patch_small_pretrain(monkeypatch)
    pretrain.train_pretrain(config, tmp_path / "default", 17, torch.device("cpu"))
    config["train"]["decoder_input_scale"] = "none"
    pretrain.train_pretrain(config, tmp_path / "none", 17, torch.device("cpu"))
    default = torch.load(tmp_path / "default/training_checkpoint.pt", weights_only=True)
    explicit = torch.load(tmp_path / "none/training_checkpoint.pt", weights_only=True)
    for key in ("actor", "decoder", "optimizer", "rng", "fit_state"):
        _assert_nested_equal(default[key], explicit[key])
    assert set(default["decoder"]) == {"weight", "bias"}


def test_population_rms_decoder_starts_at_zero_after_the_default_draw(tmp_path, monkeypatch):
    from flytrack_train.pretrain import PopulationRMSDecoder

    torch.manual_seed(5)
    reference = torch.nn.Linear(3, 4)
    after_reference = torch.get_rng_state()
    torch.manual_seed(5)
    decoder = PopulationRMSDecoder(3, 4, torch.tensor([False, True, True]))
    assert torch.equal(after_reference, torch.get_rng_state())
    assert reference.weight.abs().sum() > 0
    assert torch.count_nonzero(decoder.weight) == torch.count_nonzero(decoder.bias) == 0
    assert decoder.weight.requires_grad and decoder.bias.requires_grad

    pretrain, config = _patch_small_pretrain(monkeypatch)
    original_initialize = pretrain.initialize_decoder_inputs
    seen = []

    def record_start(decoder, *args):
        seen.append({key: value.clone() for key, value in decoder.state_dict().items()})
        original_initialize(decoder, *args)

    monkeypatch.setattr(pretrain, "initialize_decoder_inputs", record_start)
    config["train"]["decoder_input_scale"] = "population_rms"
    pretrain.train_pretrain(config, tmp_path, 3, torch.device("cpu"))
    (start,) = seen
    assert torch.count_nonzero(start["weight"]) == torch.count_nonzero(start["bias"]) == 0
    trained = torch.load(tmp_path / "training_checkpoint.pt", weights_only=True)["decoder"]
    assert torch.count_nonzero(trained["weight"]) > 0


@pytest.mark.parametrize("scale", ["none", "population_rms"])
def test_validate_pretrain_reads_plain_and_population_rms_decoders(tmp_path, monkeypatch, scale):
    from flytrack_core.contracts import SensorContract
    from flytrack_train import validation

    pretrain, config = _patch_small_pretrain(monkeypatch)
    config["train"]["decoder_input_scale"] = scale
    pretrain.train_pretrain(config, tmp_path / "train", 3, torch.device("cpu"))
    checkpoint_path = tmp_path / "train/pretraining_checkpoint.pt"
    checkpoint = torch.load(checkpoint_path, weights_only=True)
    assert ("input_rms" in checkpoint["decoder"]) == (scale == "population_rms")
    policy, _ = pretrain.prepare_policy(config, "cpu")
    policy.load_state_dict(checkpoint["actor"])
    monkeypatch.setattr(
        validation, "load_policy_artifact", lambda *args, **kwargs: (policy, SensorContract(), {})
    )
    monkeypatch.setattr(validation, "make_env", lambda *args: _RecoveryEnv())
    summary = validation.validate_pretraining(
        {**config, "validation": {"episodes": 2}},
        tmp_path / "validation",
        tmp_path / "actor",
        checkpoint_path,
        torch.device("cpu"),
    )
    assert summary["episodes"] == 2 and summary["present_steps"] == 6
    assert math.isfinite(summary["bearing_mae_deg"])

    # Uninitialized scales would silently divide by one; refuse them.
    if scale == "population_rms":
        checkpoint["decoder"]["input_initialized"] = torch.tensor(False)
        torch.save(checkpoint, tmp_path / "uninitialized.pt")
        with pytest.raises(ValueError, match="uninitialized"):
            validation.validate_pretraining(
                {**config, "validation": {"episodes": 2}},
                tmp_path / "rejected",
                tmp_path / "actor",
                tmp_path / "uninitialized.pt",
                torch.device("cpu"),
            )


def test_decoder_rms_ignores_padding_held_out_and_uses_whole_first_batch_once():
    from flytrack_train.batches import SequenceModel
    from flytrack_train.pretrain import PopulationRMSDecoder, initialize_decoder_inputs

    class ActivityPolicy(_RecoveryPolicy):
        def forward(self, observation, state):
            return observation["odor"], state, observation["odor"]

    inputs = [
        {"odor": torch.tensor([[1.0, 2.0, 4.0], [3.0, 4.0, 4.0]])},
        {"odor": torch.full((1, 3), 1000.0)},  # Held out.
        {"odor": torch.tensor([[5.0, 6.0, 4.0]])},
        {"odor": torch.full((1, 3), 1000.0)},  # Held out.
        {"odor": torch.full((1, 3), 1000.0)},  # Beyond first batch.
    ]
    decoder = PopulationRMSDecoder(3, 3, torch.tensor([False, False, True]))
    parameters = list(decoder.parameters())
    model = SequenceModel(ActivityPolicy(), "cpu")
    rng_before = torch.get_rng_state().clone()
    initialize_decoder_inputs(
        decoder,
        model,
        inputs,
        torch.ones(3, dtype=torch.bool),
        {"batch_episodes": 2, "sequence_length": 1, "validation_every": 2},
    )
    torch.testing.assert_close(decoder.input_rms, torch.tensor([math.sqrt(91 / 6), 4.0]))
    assert not decoder.input_rms.requires_grad
    assert torch.equal(rng_before, torch.get_rng_state())
    assert len(list(decoder.parameters())) == len(parameters) == 2
    assert all(parameter.grad is None for parameter in parameters)
    activity = torch.tensor([[2.0, 0.0, 4.0], [3.0, 1e-5, 4.0]], requires_grad=True)
    torch.testing.assert_close(
        decoder(activity),
        torch.nn.functional.linear(
            activity / decoder.input_rms[torch.tensor([0, 0, 1])], decoder.weight, decoder.bias
        ),
    )
    decoder(activity).sum().backward()
    assert activity.grad is not None and torch.isfinite(activity.grad).all()
    saved_rms = decoder.input_rms.clone()
    inputs[0]["odor"].fill_(999)
    initialize_decoder_inputs(
        decoder,
        SequenceModel(ActivityPolicy(), "cpu"),
        inputs,
        torch.ones(3, dtype=torch.bool),
        {},
    )
    assert torch.equal(decoder.input_rms, saved_rms)


@pytest.mark.parametrize("value", [0.0, float("inf"), float("nan")])
def test_decoder_population_rms_floor_and_nonfinite_inputs(value):
    from flytrack_train.batches import SequenceModel
    from flytrack_train.pretrain import PopulationRMSDecoder, initialize_decoder_inputs

    class ActivityPolicy(_RecoveryPolicy):
        def forward(self, observation, state):
            return observation["odor"], state, observation["odor"]

    decoder = PopulationRMSDecoder(2, 3, torch.tensor([False, True]))
    args = (
        decoder,
        SequenceModel(ActivityPolicy(), "cpu"),
        [{"odor": torch.full((2, 2), value)}],
        torch.ones(2, dtype=torch.bool),
        {"batch_episodes": 1},
    )
    if math.isfinite(value):
        initialize_decoder_inputs(*args)
        torch.testing.assert_close(decoder.input_rms, torch.full((2,), 1e-3))
    else:
        with pytest.raises(ValueError, match="finite"):
            initialize_decoder_inputs(*args)
        assert not bool(decoder.input_initialized)
