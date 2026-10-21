"""Reward scaling, clipping isolation, and the pre-change CPU smoke baseline."""

import numpy as np
import pytest
import torch
from flytrack_train.config import read_config
from flytrack_train.ppo import RewardNormalizer, _clip_gradients, _critic_loss, train_ppo


def test_discounted_return_moments_match_numpy():
    normalizer = RewardNormalizer(3, gamma=0.9)
    samples = []
    running = np.zeros(3)
    rewards = [[1.0, -2.0, 3.0], [0.5, 4.0, -1.0], [2.0, 0.0, 0.1]]
    for row in rewards:
        running = 0.9 * running + row
        samples.extend(running)
        normalizer.normalize(row, [False] * 3, [False] * 3)
        # Include SB3's initial count=1e-4, mean=0, variance=1 prior.
        count = len(samples) + 1e-4
        mean = np.sum(samples) / count
        variance = (np.sum(np.square(samples)) + 1e-4) / count - mean**2
        assert normalizer.count == count
        assert normalizer.mean == pytest.approx(mean, rel=1e-13)
        assert normalizer.var == pytest.approx(variance, rel=1e-13)
        assert normalizer.std == pytest.approx(np.sqrt(variance + 1e-8), rel=1e-13)
        np.testing.assert_array_equal(normalizer.returns, running)


def test_terminal_and_truncated_returns_are_counted_then_reset_per_environment():
    normalizer = RewardNormalizer(3, gamma=0.9)
    normalizer.normalize([1.0, 2.0, 3.0], [True, False, False], [False, True, False])
    np.testing.assert_array_equal(normalizer.returns, [0.0, 0.0, 3.0])
    assert normalizer.mean == pytest.approx(6 / 3.0001)
    normalizer.normalize([4.0, 5.0, 6.0], [False] * 3, [False] * 3)
    np.testing.assert_array_equal(normalizer.returns, [4.0, 5.0, 8.7])


def test_normalized_reward_keeps_sign_and_only_divides_by_standard_deviation():
    normalizer = RewardNormalizer(5, gamma=0.99)
    rewards = np.asarray([-10.0, -0.1, 0.0, 0.1, 1.6])
    for _ in range(20):
        normalized = normalizer.normalize(rewards, [False] * 5, [False] * 5)
        np.testing.assert_array_equal(np.sign(normalized), np.sign(rewards))
        np.testing.assert_allclose(normalized, rewards / normalizer.std, rtol=1e-7)


def test_steady_survival_rewards_are_not_clipped_and_startup_is_bounded():
    normalizer = RewardNormalizer(1, gamma=0.99)
    first = normalizer.normalize([1.6], [False], [False])
    assert first[0] == 10.0  # SB3's small initial count can clip startup samples.
    for step in range(1, 240):
        result = normalizer.normalize([1.6], [False], [step % 40 == 39])
        if step >= 10:
            assert 0 < 1.6 / normalizer.std < normalizer.reward_clip
            np.testing.assert_allclose(result, [1.6 / normalizer.std], rtol=1e-7)


@pytest.mark.parametrize("reward_clip", [0, -1, float("inf"), float("nan")])
def test_reward_clip_requires_a_finite_positive_limit(reward_clip):
    with pytest.raises(ValueError, match="finite and positive"):
        RewardNormalizer(1, 0.99, reward_clip)


def test_reward_clip_configuration_is_applied_symmetrically():
    normalizer = RewardNormalizer(3, gamma=0.99, reward_clip=0.01)
    result = normalizer.normalize([-1, 0, 1], [False] * 3, [False] * 3)
    np.testing.assert_array_equal(result, np.asarray([-0.01, 0, 0.01], dtype=np.float32))


@pytest.mark.parametrize("separate", [False, True])
def test_actor_and_critic_clipping_limits_and_log_std(separate):
    actor = torch.nn.Linear(2, 1)
    log_std = torch.nn.Parameter(torch.zeros(2))
    actor.register_parameter("log_std", log_std)
    critic = torch.nn.Linear(2, 1)
    for parameter in actor.parameters():
        parameter.grad = torch.ones_like(parameter)
    for parameter in critic.parameters():
        parameter.grad = torch.full_like(parameter, 1000.0)
    actor_parameters, critic_parameters = list(actor.parameters()), list(critic.parameters())
    norms = _clip_gradients(actor_parameters, critic_parameters, 0.5, separate)
    actor_norm = torch.cat([p.grad.flatten() for p in actor_parameters]).norm().item()
    critic_norm = torch.cat([p.grad.flatten() for p in critic_parameters]).norm().item()
    assert actor_norm <= 0.5 + 1e-7
    assert critic_norm <= 0.5 + 1e-7
    assert norms[2:] == pytest.approx((actor_norm, critic_norm))
    assert norms[0] == pytest.approx(np.sqrt(5))
    assert norms[1] == pytest.approx(np.sqrt(3) * 1000)
    if separate:
        assert actor_norm == pytest.approx(0.5, abs=1e-6)
        assert log_std.grad.norm() > 0.1
    else:
        assert actor_norm < 0.001


@pytest.mark.parametrize("target", [20.0, 160.0])
@pytest.mark.parametrize("value_clip", [None, 0.2, 0.5])
def test_critic_can_correct_underestimates_unless_value_clipping_stalls_it(target, value_clip):
    value = torch.tensor([0.3], dtype=torch.float64, requires_grad=True)
    loss = _critic_loss(value, torch.tensor([target]), torch.zeros(1), value_clip)
    loss.backward()
    if value_clip == 0.2:
        assert loss.item() == pytest.approx(0.5 * (0.2 - target) ** 2)
        assert value.grad.item() == 0.0
    else:
        assert loss.item() == pytest.approx(0.5 * (0.3 - target) ** 2)
        assert value.grad.item() == pytest.approx(0.3 - target)
        assert value.grad.item() < 0  # Gradient descent increases an underestimated value.


@pytest.mark.parametrize(
    "value_clip",
    [0, -0.2, float("inf"), -float("inf"), float("nan"), True, False, "0.2", [], {}],
)
def test_value_clip_requires_null_or_a_finite_positive_number(tmp_path, value_clip):
    with pytest.raises(ValueError, match="train.value_clip must be null or a finite positive"):
        train_ppo({"train": {"value_clip": value_clip}}, tmp_path, 0, torch.device("cpu"))
    assert not (tmp_path / "training_checkpoint.pt").exists()


def test_legacy_settings_disable_normalization_and_split_clipping(tmp_path):
    # Bitwise equality with main under these settings was checked once on one host
    # (see PR #32); a stored hash fixture would fail on other CPUs (#34 review).
    config = read_config("configs/train/smoke.yaml")
    config["train"].update(
        value_clip=config["train"].get("clip_coef", 0.2),
        reward_normalization=False,
        separate_grad_clip=False,
    )
    train_ppo(config, tmp_path, 0, torch.device("cpu"))
    checkpoint = torch.load(tmp_path / "training_checkpoint.pt", weights_only=True)
    assert checkpoint["reward_normalizer"] is None
    assert all(report["return_std"] == 1.0 for report in checkpoint["reports"])
