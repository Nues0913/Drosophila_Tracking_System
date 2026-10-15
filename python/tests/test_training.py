"""Algorithmic checks to run later; no downloaded data or trained artifact required."""

import math
import random

import numpy as np
import pytest
import torch
from flytrack_core.metrics import EpisodeMetrics
from flytrack_train.batches import Episode, reset_state, sequence_windows, tensor_observation
from flytrack_train.distributions import BoundedGaussian, generalized_advantage
from flytrack_train.es import centered_ranks
from flytrack_train.evaluation import bootstrap_mean, seed_statistics
from flytrack_train.recovery import Recovery, capture_rng, restore_rng


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


def test_supervised_tbptt_preserves_chronology_within_shuffled_episodes():
    episodes = [Episode({"odor": np.zeros((length, 2))}, {}) for length in (9, 12)]
    windows = sequence_windows(episodes, 4, np.random.default_rng(4))
    for index, episode in enumerate(episodes):
        sequence = [(start, end) for episode_id, start, end in windows if episode_id == index]
        assert sequence == [
            (start, min(start + 4, len(episode))) for start in range(0, len(episode), 4)
        ]


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


@pytest.mark.parametrize(
    "algorithm,interrupt_phase",
    [
        ("ppo", None),
        ("es", None),
        ("pretrain", "collect"),
        ("pretrain", "fit"),
        ("bc", "collect"),
        ("bc", "fit"),
    ],
)
def test_interrupted_training_matches_uninterrupted_weights_optimizer_and_rng(
    tmp_path, monkeypatch, algorithm, interrupt_phase
):
    from flytrack_core.contracts import SensorContract
    from flytrack_train import es, imitation, ppo, pretrain

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
        if self.output == interrupted and (
            interrupt_phase is None or state.get("phase") == interrupt_phase
        ):
            raise InterruptedError("simulated process interruption after committed checkpoint")

    monkeypatch.setattr(Recovery, "save", stop_after_first_checkpoint)
    with pytest.raises(InterruptedError):
        run(config, interrupted, 11, torch.device("cpu"))
    monkeypatch.setattr(Recovery, "save", original_save)
    run({**config, "resume": True}, interrupted, 11, torch.device("cpu"))
    expected = torch.load(full / "training_checkpoint.pt", weights_only=True)
    observed = torch.load(interrupted / "training_checkpoint.pt", weights_only=True)
    for key in ("actor", "optimizer", "rng", "environment_steps"):
        _assert_nested_equal(expected[key], observed[key])
    if algorithm == "ppo":
        for key in (
            "critic",
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
