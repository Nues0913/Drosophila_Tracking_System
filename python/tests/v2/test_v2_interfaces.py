"""Action distribution, state resets, policy step and protocol conformance (plan 5.3)."""

import math

import torch
from flytrack_v2.interfaces import (
    ACTION_EPS,
    ActionDistribution,
    BatchedWorld,
    Policy,
    PolicyStep,
    StepResult,
    Teacher,
    policy_step,
    reset_rows,
    unsquash,
)


def _dist(batch=64):
    generator = torch.Generator().manual_seed(0)
    mean = torch.randn(batch, 2, generator=generator) * 3
    log_std = torch.tensor([-0.5, 0.2])
    return ActionDistribution(mean, log_std, v_max=0.5, omega_max=1.5)


def test_samples_and_mode_respect_the_action_limits():
    dist = _dist()
    action, latent = dist.sample(torch.Generator().manual_seed(1))
    for value in (action, dist.mode()):
        assert value.shape == (64, 2)
        assert bool((value[:, 0] >= 0).all() and (value[:, 0] <= 0.5).all())
        assert bool((value[:, 1].abs() <= 1.5).all())
    assert torch.allclose(dist.squash(latent), action)


def test_log_prob_and_entropy_match_torch_normal_on_the_latent():
    dist = _dist()
    _, latent = dist.sample(torch.Generator().manual_seed(2))
    reference = torch.distributions.Normal(dist.mean, dist.log_std.exp().expand_as(dist.mean))
    assert torch.allclose(dist.log_prob(latent), reference.log_prob(latent).sum(-1), atol=1e-5)
    assert torch.allclose(dist.entropy(), reference.entropy().sum(-1), atol=1e-6)
    assert math.isclose(float(dist.entropy()[0]), float(dist.entropy()[-1]))


def test_unsquash_inverts_squash_and_stays_finite_at_the_limits():
    dist = _dist()
    action = dist.mode()
    assert torch.allclose(dist.squash(dist.unsquash(action)), action, atol=1e-5)
    limits = torch.tensor([[0.0, -1.5], [0.5, 1.5], [0.25, 0.0]])
    latent = unsquash(limits, 0.5, 1.5)
    assert bool(torch.isfinite(latent).all())
    assert torch.allclose(dist.squash(latent), limits, atol=1e-5)
    assert torch.allclose(torch.sigmoid(latent[1, 0]), torch.tensor(1 - ACTION_EPS))
    assert bool(torch.isfinite(dist.log_prob(dist.unsquash(limits[:1].expand(64, 2)))).all())


class _Policy:
    """Opaque sensory state of a different layout ([features, B]) than the central state."""

    variant, central_state_size, code_size = "VO2", 3, 4

    def initial_sensory_state(self, batch_size, device):
        return torch.zeros(5, batch_size, device=device)

    def reset_sensory_state(self, sensory_state, reset_mask):
        return torch.where(reset_mask[None, :], 0.0, sensory_state)

    def initial_central_state(self, batch_size, device):
        return torch.arange(3.0, device=device).expand(batch_size, -1).clone()

    def sense(self, obs, sensory_state):
        sensory_state = sensory_state + 1
        return sensory_state[:4].T.clone(), sensory_state

    def forward_code(self, code, central_state):
        dist = ActionDistribution(torch.zeros(len(code), 2), torch.zeros(2), 0.5, 1.5)
        return PolicyStep(dist, central_state + code[:, :3], {})


class _World:
    contract, num_envs, device = None, 2, torch.device("cpu")

    def reset(self, mask=None, episode_seeds=None):
        return {}

    def step(self, action):
        flags = torch.zeros(2, dtype=torch.bool)
        return StepResult({}, torch.zeros(2), flags, flags, {}, {}, {})

    def privileged(self):
        return {}


class _Teacher:
    def initial_state(self, batch_size, device):
        return torch.zeros(batch_size, 0, device=device)

    def act(self, privileged, obs, previous_action, state):
        return previous_action, state


def test_reset_rows_replaces_only_masked_rows():
    state = torch.full((3, 2, 2), -1.0)
    initial = torch.zeros(3, 2, 2)
    out = reset_rows(state, initial, torch.tensor([False, True, False]))
    assert torch.equal(out[1], initial[1]) and torch.equal(out[[0, 2]], state[[0, 2]])
    assert reset_rows(state, initial, None) is state


def test_policy_step_restarts_finished_rows_before_sensing():
    policy = _Policy()
    sensory = torch.full((5, 3), 7.0)
    central = torch.full((3, 3), -1.0)
    step, sensory, code = policy_step(policy, {}, sensory, central, torch.tensor([0, 1, 0]) > 0)
    assert torch.equal(sensory[:, 1], torch.ones(5)) and torch.equal(
        sensory[:, 0], torch.full((5,), 8.0)
    )
    assert torch.equal(step.central_state[1], torch.arange(3.0) + 1)
    assert torch.equal(step.central_state[0], torch.full((3,), 7.0))
    assert code.shape == (3, 4)


def test_reference_implementations_satisfy_the_protocols():
    assert isinstance(_Policy(), Policy)
    assert isinstance(_World(), BatchedWorld)
    assert isinstance(_Teacher(), Teacher)
    assert set(StepResult._fields) >= {"final_obs", "final_privileged"}
