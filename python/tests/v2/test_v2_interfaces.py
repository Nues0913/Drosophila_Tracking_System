"""Action distribution, state reset and protocol conformance (plan 5.3)."""

import math

import torch
from flytrack_v2.interfaces import (
    ActionDistribution,
    BatchedWorld,
    Policy,
    PolicyStep,
    StepResult,
    Teacher,
    reset_state,
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


class _Policy:
    variant, sensory_state_size, central_state_size, code_size = "VO2", 2, 3, 4

    def initial_state(self, batch_size, device):
        return torch.arange(5.0, device=device).expand(batch_size, -1).clone()

    def sense(self, obs, sensory_state):
        return torch.zeros(len(sensory_state), 4), sensory_state

    def forward_code(self, code, central_state):
        dist = ActionDistribution(torch.zeros(len(code), 2), torch.zeros(2), 0.5, 1.5)
        return PolicyStep(dist, central_state, {})

    def forward(self, obs, state, reset_mask=None):
        state = reset_state(self, state, reset_mask)
        return self.forward_code(torch.zeros(len(state), 4), state)


class _World:
    contract, num_envs, device = None, 2, torch.device("cpu")

    def reset(self, mask=None, episode_seeds=None):
        return {}

    def step(self, action):
        flags = torch.zeros(2, dtype=torch.bool)
        return StepResult({}, torch.zeros(2), flags, flags, {})

    def privileged(self):
        return {}


class _Teacher:
    def act(self, privileged, obs, previous_action):
        return previous_action


def test_reset_state_replaces_only_masked_rows():
    policy = _Policy()
    state = torch.full((3, 5), -1.0)
    out = reset_state(policy, state, torch.tensor([False, True, False]))
    assert torch.equal(out[1], torch.arange(5.0))
    assert torch.equal(out[[0, 2]], state[[0, 2]])
    assert reset_state(policy, state, None) is state


def test_reference_implementations_satisfy_the_protocols():
    assert isinstance(_Policy(), Policy)
    assert isinstance(_World(), BatchedWorld)
    assert isinstance(_Teacher(), Teacher)
