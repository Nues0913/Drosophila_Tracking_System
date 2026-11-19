"""PPO-v2 (#138): advantages, schedules, the three probe environments and an end-to-end run."""

import json
import math

import pytest
import torch
from flytrack_v2.interfaces import ActionDistribution, PolicyStep
from flytrack_v2.train import ppo
from flytrack_v2.train.ppo import (
    Critic,
    PPOSettings,
    Rollout,
    advantages_and_returns,
    gaussian_kl,
    ppo_update,
    schedule,
)
from torch import nn


class _ConstantCritic(nn.Module):
    """Predicts the first feature as the normalized value."""

    def forward(self, features):
        return features[..., 0]


def _rollout(reward, terminated, truncated, features, final, last, code_size=1, state=1):
    t, n = reward.shape
    return Rollout(
        code=torch.zeros(t, n, code_size),
        latent=torch.zeros(t, n, 2),
        logp=torch.zeros(t, n),
        features=features,
        reward=reward,
        terminated=terminated,
        truncated=truncated,
        final_features=final,
        last_features=last,
        central=torch.zeros(n, state),
        initial=torch.zeros(1, state),
    )


def test_advantages_bootstrap_truncation_and_stop_at_collision():
    gamma, lam = 0.9, 0.8
    scale = 1 / (1 - gamma)
    # Normalized values: state value = feature / (1 - gamma). Row 0 truncates after step 0,
    # row 1 collides at step 0; both continue with a new episode at step 1.
    features = torch.tensor([[[0.1], [0.2]], [[0.3], [0.4]]])
    final = torch.tensor([[[0.5], [0.0]], [[0.0], [0.0]]])
    last = torch.tensor([[0.6], [0.7]])
    reward = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    terminated = torch.tensor([[False, True], [False, False]])
    truncated = torch.tensor([[True, False], [False, False]])
    rollout = _rollout(reward, terminated, truncated, features, final, last)
    adv, ret, values = advantages_and_returns(_ConstantCritic(), rollout, gamma, lam)
    v = features[..., 0] * scale
    # step 1 continues into the last state
    d1 = reward[1] + gamma * last[:, 0] * scale - v[1]
    # step 0: row 0 bootstraps from its final state, row 1 stops; neither carries step 1's
    d0 = torch.tensor(
        [reward[0, 0] + gamma * final[0, 0, 0] * scale - v[0, 0], reward[0, 1] - v[0, 1]]
    )
    torch.testing.assert_close(adv[1], d1)
    torch.testing.assert_close(adv[0], d0)
    torch.testing.assert_close(ret, adv + values)


def test_advantages_accumulate_within_an_episode():
    gamma, lam = 0.9, 0.5
    features = torch.zeros(3, 1, 1)
    reward = torch.ones(3, 1)
    none = torch.zeros(3, 1, dtype=torch.bool)
    rollout = _rollout(reward, none, none, features, features, torch.zeros(1, 1))
    adv, _, _ = advantages_and_returns(_ConstantCritic(), rollout, gamma, lam)
    expected = [1 + gamma * lam * (1 + gamma * lam), 1 + gamma * lam, 1.0]
    torch.testing.assert_close(adv[:, 0], torch.tensor(expected))


def test_schedule_warms_up_the_critic_then_ramps_and_anneals_the_actor():
    config = {
        "updates": 100,
        "warmup": {"critic_only_updates": 10, "actor_ramp_updates": 10},
        "actor_lr": 3e-4,
        "critic_lr": 1e-3,
        "kickstart": {"initial_lambda": 1.0, "anneal_fraction": 0.3},
    }
    first = schedule(0, config)
    assert not first.actor_on and first.actor_lr == 0 and first.critic_lr == pytest.approx(1e-3)
    assert first.kick_lambda == 1.0
    assert schedule(10, config).actor_lr == pytest.approx(3e-5)
    assert schedule(19, config).actor_lr == pytest.approx(3e-4)
    assert schedule(20, config).actor_lr == pytest.approx(3e-4)
    assert schedule(99, config).actor_lr < 1e-5
    # 90 actor updates, annealed over the first 27
    assert schedule(10 + 27, config).kick_lambda == 0.0
    assert schedule(10 + 9, config).kick_lambda == pytest.approx(1 - 9 / 27)


def test_gaussian_kl_is_zero_for_equal_and_closed_form_otherwise():
    mean = torch.tensor([[0.3, -0.2]])
    log_std = torch.tensor([-0.5, 0.1])
    assert gaussian_kl(mean, log_std, mean, log_std).item() == pytest.approx(0.0, abs=1e-7)
    p = torch.distributions.Normal(mean, log_std.exp())
    q = torch.distributions.Normal(mean + 0.4, (log_std + 0.3).exp())
    expected = torch.distributions.kl_divergence(p, q).sum(-1)
    torch.testing.assert_close(gaussian_kl(mean, log_std, mean + 0.4, log_std + 0.3), expected)


# ---------------------------------------------------------------------------------------------
# Probe environments (plan 4.2): a tiny recurrent policy with v2's action distribution


class _ToyPolicy(nn.Module):
    def __init__(self, code_size: int, hidden: int = 16):
        super().__init__()
        self.cell = nn.GRUCell(code_size, hidden)
        self.head = nn.Linear(hidden, 2)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)
        self.log_std = nn.Parameter(torch.full((2,), -0.5))
        self.hidden = hidden

    def forward_code(self, code, state):
        h = self.cell(code, state)
        return PolicyStep(ActionDistribution(self.head(h), self.log_std, 1.0, 1.0), h, {})


def _toy_rollout(policy, env, n, steps, generator):
    """env(t, latent_or_None) -> code [n, C], features [n, F], reward [n], terminated [n]."""
    state = torch.zeros(n, policy.hidden)
    code, features, _, _ = env(0, None)
    buffers = {k: [] for k in ("code", "latent", "logp", "features", "reward", "terminated")}
    with torch.no_grad():
        for t in range(steps):
            step = policy.forward_code(code, state)
            _, latent = step.dist.sample(generator)
            buffers["code"].append(code)
            buffers["latent"].append(latent)
            buffers["logp"].append(step.dist.log_prob(latent))
            buffers["features"].append(features)
            code_next, features_next, reward, terminated = env(t + 1, latent)
            buffers["reward"].append(reward)
            buffers["terminated"].append(terminated)
            state = torch.where(terminated[:, None], 0.0, step.central_state)
            code, features = code_next, features_next
    stacked = {k: torch.stack(v) for k, v in buffers.items()}
    return Rollout(
        **stacked,
        truncated=torch.zeros_like(stacked["terminated"]),
        final_features=torch.zeros_like(stacked["features"]),
        last_features=features,
        central=torch.zeros(n, policy.hidden),
        initial=torch.zeros(1, policy.hidden),
    )


def _train(policy, critic, env, *, n, steps, updates, settings, actor_on, lr=3e-3, seed=0):
    generator = torch.Generator().manual_seed(seed)
    actor = torch.optim.Adam(policy.parameters(), lr=lr)
    value = torch.optim.Adam(critic.parameters(), lr=lr)
    for _ in range(updates):
        rollout = _toy_rollout(policy, env, n, steps, generator)
        ppo_update(
            policy, critic, actor, value, rollout, settings, actor_on=actor_on, kick_lambda=0.0,
            micro_batch=n, generator=generator,
        )  # fmt: skip
    return rollout


_PROBE = PPOSettings(gamma=0.9, gae_lambda=0.95, entropy_coef=0.0, activity_coef=0.0, minibatches=2)


def test_probe_constant_reward_value():
    """One-step episodes with reward 1: V = 1."""
    n = 64

    def env(t, latent):
        ones = torch.ones(n)
        return torch.ones(n, 1), torch.ones(n, 1), ones, torch.ones(n, dtype=torch.bool)

    critic = Critic(1, (16,))
    torch.manual_seed(0)
    _train(_ToyPolicy(1), critic, env, n=n, steps=4, updates=40, settings=_PROBE, actor_on=False)
    value = critic(torch.ones(1, 1)).item() / (1 - _PROBE.gamma)
    assert value == pytest.approx(1.0, abs=0.05)


def test_probe_reward_from_the_observation():
    """One-step episodes, reward = observation in {-1, 1}: V(obs) = obs."""
    n = 64
    generator = torch.Generator().manual_seed(1)
    current = {}

    def env(t, latent):
        obs = torch.randint(0, 2, (n, 1), generator=generator).float() * 2 - 1
        reward = current.get("obs", obs)[:, 0]
        current["obs"] = obs
        return obs, obs, reward, torch.ones(n, dtype=torch.bool)

    critic = Critic(1, (16,))
    torch.manual_seed(1)
    _train(_ToyPolicy(1), critic, env, n=n, steps=4, updates=60, settings=_PROBE, actor_on=False)
    scale = 1 / (1 - _PROBE.gamma)
    assert critic(torch.tensor([[1.0]])).item() * scale == pytest.approx(1.0, abs=0.1)
    assert critic(torch.tensor([[-1.0]])).item() * scale == pytest.approx(-1.0, abs=0.1)


def test_probe_truncation_bootstraps_from_the_state_before_the_reset():
    """Reward 1 per step, cut by a time limit every 5 steps and never terminated: V = 1/(1-g)."""
    n, steps = 32, 5
    critic = Critic(1, (16,))
    ones = torch.ones(steps, n)
    none = torch.zeros(steps, n, dtype=torch.bool)
    cut = none.clone()
    cut[-1] = True
    features = torch.ones(steps, n, 1)
    rollout = _rollout(ones, none, cut, features, features, torch.ones(n, 1))
    optimizer = torch.optim.Adam(critic.parameters(), lr=1e-2)
    generator = torch.Generator().manual_seed(2)
    for _ in range(200):
        ppo_update(
            _ToyPolicy(1), critic, torch.optim.Adam([nn.Parameter(torch.zeros(1))]), optimizer,
            rollout, _PROBE, actor_on=False, kick_lambda=0.0, micro_batch=n, generator=generator,
        )  # fmt: skip
    value = critic(torch.ones(1, 1)).item() / (1 - _PROBE.gamma)
    # treating the cut as a termination would give (1 - 0.9^5) / 0.1 = 4.1 at the first step
    assert value == pytest.approx(1 / (1 - _PROBE.gamma), rel=0.05)


def test_probe_delayed_reward_needs_memory():
    """Step 0 shows a cue in {-1, 1}; step 1 shows nothing and pays cue * turn. Only a policy
    that carries the cue in its recurrent state can earn the reward."""
    n = 256
    generator = torch.Generator().manual_seed(3)
    episode = {}

    def env(t, latent):
        if t % 2 == 0:
            cue = torch.randint(0, 2, (n,), generator=generator).float() * 2 - 1
            reward = (
                episode["cue"] * torch.tanh(latent[:, 1]) if latent is not None else torch.zeros(n)
            )
            episode["cue"] = cue
            code = torch.stack((cue, torch.ones(n), torch.zeros(n)), -1)
            terminated = torch.ones(n, dtype=torch.bool) if latent is not None else None
            features = torch.stack((cue, torch.ones(n)), -1)
            return code, features, reward, terminated
        code = torch.stack((torch.zeros(n), torch.zeros(n), torch.ones(n)), -1)
        features = torch.stack((episode["cue"], torch.zeros(n)), -1)
        return code, features, torch.zeros(n), torch.zeros(n, dtype=torch.bool)

    torch.manual_seed(3)
    policy, critic = _ToyPolicy(3), Critic(2, (16,))
    settings = PPOSettings(
        gamma=0.9, gae_lambda=0.95, entropy_coef=0.0, activity_coef=0.0, minibatches=4,
        target_kl=1.0,
    )  # fmt: skip
    rollout = _train(
        policy, critic, env, n=n, steps=2, updates=60, settings=settings, actor_on=True
    )
    paid = rollout.reward[1].mean().item()  # the second step of each episode
    assert paid > 0.6


# ---------------------------------------------------------------------------------------------
# End to end on the GPU worlds (CPU here), from an exported Stage 1 artifact


@pytest.mark.slow
def test_ppo_runs_stops_resumes_and_exports(tmp_path):
    from flytrack_v2.model import build_policy, export_policy, load_policy

    init = export_policy(
        build_policy("VO2", {"rest_settle_s": 0.05, "rest_refresh_s": 0.05}),
        tmp_path / "stage1",
        provenance={"git_sha": None, "dirty": None, "config_sha256": "c" * 64, "stages": []},
    )
    out = tmp_path / "ppo"
    overrides = [
        "envs_per_instance=2", "rollout_steps=3", "minibatches=2", "total_env_steps=96",
        "warmup.critic_only_updates=1", "warmup.actor_ramp_updates=1", "checkpoint_every=1",
    ]  # fmt: skip
    args = ["--init", str(init), "--seed", "5", "--host", "2060", "--device", "cpu"]
    args += ["--out", str(out), "--override", *overrides]
    out.mkdir()
    (out / "STOP").touch()
    assert ppo.main(args) == 0
    first = [json.loads(line) for line in (out / "metrics.jsonl").read_text().splitlines()]
    assert [row["update"] for row in first] == [1] and not (out / "final").exists()
    (out / "STOP").unlink()
    assert ppo.main([*args, "--resume"]) == 0
    rows = [json.loads(line) for line in (out / "metrics.jsonl").read_text().splitlines()]
    assert [row["update"] for row in rows] == [1, 2, 3, 4]  # 8 envs x 3 steps per update
    assert rows[0]["actor_on"] == 0.0 and rows[1]["actor_on"] == 1.0
    assert all(math.isfinite(row["value_loss"]) for row in rows)
    assert "kl_teacher" in rows[1]
    card = json.loads((out / "final" / "model_card.json").read_text())
    assert [stage["name"] for stage in card["provenance"]["stages"]] == ["ppo"]
    assert card["provenance"]["train_seed"] == 5
    policy = load_policy(out / "final")
    assert policy.variant == "VO2"
