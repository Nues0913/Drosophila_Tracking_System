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
from test_v2_model_core import small_graph
from torch import nn
from torch.nn.functional import softplus


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
    """Reward 1 per step, cut by a forced reset every 5 steps and never terminated: V = 1/(1-g)."""
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
    with pytest.raises(SystemExit):  # #188 review: another seed must not resume this run
        ppo.main([*args[:3], "6", *args[4:], "--resume"])
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


@pytest.mark.slow
def test_ppo_trains_plastic_gain_theta_and_resumes_with_frozen_delta(tmp_path):
    from flytrack_v2.model import build_policy, export_policy, load_policy

    labels = [("ORN", "Forward", 0, 2.0)]
    policy = build_policy(
        "VO2",
        {"rest_settle_s": 0.05, "rest_refresh_s": 0.05, "plastic_labels": labels},
        graph=small_graph(),
    )
    with torch.no_grad():
        policy.core.gain_delta.fill_(0.1)
    init = export_policy(
        policy,
        tmp_path / "stage1",
        provenance={"git_sha": None, "dirty": None, "config_sha256": "c" * 64, "stages": []},
    )
    overrides = [
        "envs_per_instance=2", "rollout_steps=3", "minibatches=2", "epochs=1",
        "total_env_steps=48", "warmup.critic_only_updates=0", "warmup.actor_ramp_updates=1",
    ]  # fmt: skip
    config = ppo.load_ppo_config(ppo.CONFIG_DIR / "ppo.yaml", "2060", overrides)
    device = torch.device("cpu")
    out = tmp_path / "ppo"
    torch.manual_seed(5)
    run = ppo.PPORun(config, init, 5, device, out)
    run.prepare(resume=False)
    (out / "STOP").touch()
    run.run()
    assert run.update == 1
    core = run.policy.core
    delta = core.gain_delta
    groups = run.actor_optimizer.param_groups
    assert all(param is not delta for group in groups for param in group["params"])
    assert any(param is core.gain_theta for param in groups[0]["params"])
    assert groups[0]["lr"] == config["actor_lr"]
    assert not delta.requires_grad and delta.grad is None
    assert torch.equal(delta, policy.core.gain_delta)
    index = core.plastic_index
    assert not torch.equal(core.gain_theta[index], policy.core.gain_theta[index])
    gains = core.gains()[index]
    assert bool(((core.plastic_lo <= gains) & (gains <= core.plastic_hi)).all())

    resumed = ppo.PPORun(config, init, 5, device, out)
    resumed.prepare(resume=True)
    assert resumed.update == 1
    torch.testing.assert_close(
        resumed.actor_optimizer.state_dict(), run.actor_optimizer.state_dict()
    )
    core = resumed.policy.core
    assert not core.gain_delta.requires_grad
    assert torch.equal(core.gain_delta, delta)
    # Force a bound violation to verify the projection hook also runs after resume.
    with torch.no_grad():
        core.gain_delta.fill_(100)
    (out / "STOP").unlink()
    resumed.run()
    raw = softplus(core.gain_theta[index] + core.gain_delta)
    assert bool(((core.plastic_lo <= raw) & (raw <= core.plastic_hi)).all())
    final = out / "final"
    exported = load_policy(final)
    assert torch.equal(exported.state_dict()["core.gain_delta"], core.gain_delta)
    assert torch.equal(exported.core.gains(), core.gains())
    card = json.loads((final / "model_card.json").read_text())
    assert card["model_config"]["plastic_labels"] == [list(label) for label in labels]


@pytest.mark.slow
def test_teacher_state_follows_the_student_through_the_warm_up(tmp_path):
    """#192 R7: the frozen teacher steps during the critic-only warm-up. With the actor's
    learning rate at 0 the student equals the teacher, so their recurrent states must agree."""
    from flytrack_v2.model import build_policy, export_policy

    init = export_policy(
        build_policy("VO2", {"rest_settle_s": 0.05, "rest_refresh_s": 0.05}),
        tmp_path / "stage1",
        provenance={"git_sha": None, "dirty": None, "config_sha256": "c" * 64, "stages": []},
    )
    overrides = [
        "envs_per_instance=2", "rollout_steps=4", "minibatches=2", "total_env_steps=320",
        "warmup.critic_only_updates=4", "warmup.actor_ramp_updates=1", "checkpoint_every=50",
        "actor_lr=0.0",
    ]  # fmt: skip
    config = ppo.load_ppo_config(ppo.CONFIG_DIR / "ppo.yaml", "2060", overrides)
    run = ppo.PPORun(config, init, 1, torch.device("cpu"), tmp_path / "ppo")
    run.prepare(resume=False)
    collector = run.collector
    for update in range(config["warmup"]["critic_only_updates"]):
        plan = schedule(update, config)
        assert not plan.actor_on and ppo.steps_teacher(plan)
        collector.collect_ppo(config["rollout_steps"], kickstart=ppo.steps_teacher(plan))
    torch.testing.assert_close(collector.teacher_central, collector.central, rtol=0, atol=1e-6)
    assert not torch.equal(collector.central, collector.teacher_initial)


def test_the_seed_must_be_the_stage1_training_seed(tmp_path):
    """#188 review: the card's train_seed picks the evaluation episodes."""
    card = {"provenance": {"train_seed": 3}}
    ppo.check_init_seed(tmp_path, card, 3)
    with pytest.raises(ValueError, match="differs from the Stage 1 training seed 3"):
        ppo.check_init_seed(tmp_path, card, 0)
    ppo.check_init_seed(tmp_path / "seed_3" / "final", {"provenance": {}}, 3)
    with pytest.raises(ValueError, match="contradicts the Stage 1 path"):
        ppo.check_init_seed(tmp_path / "seed_3" / "final", {"provenance": {}}, 0)
    ppo.check_init_seed(tmp_path / "stage1", {"provenance": {}}, 5)  # nothing to contradict


def test_the_time_limit_terminates_and_a_forced_reset_bootstraps():
    """#188 review: the critic sees time, so the 60 s end is terminal; a forced first-episode
    reset is not part of the task and bootstraps."""
    gamma = 0.9
    features = torch.full((1, 2, 1), 0.5)
    final = torch.full((1, 2, 1), 0.5)
    reward = torch.ones(1, 2)
    ended = torch.tensor([[True, False]])  # row 0: the time limit
    forced = torch.tensor([[False, True]])  # row 1: a forced reset
    rollout = _rollout(reward, ended, forced, features, final, torch.zeros(2, 1))
    adv, _, values = advantages_and_returns(_ConstantCritic(), rollout, gamma, 0.95)
    v = 0.5 / (1 - gamma)
    torch.testing.assert_close(adv[0], torch.tensor([1 - v, 1 + gamma * v - v]))


def test_the_warm_up_needs_one_actor_update_not_a_finished_ramp():
    """#188: 40 updates with a 20 + 20 warm-up train the actor during its whole ramp."""
    steps = lambda updates: f"total_env_steps={updates * 1024 * 32}"  # noqa: E731
    config = ppo.load_ppo_config(ppo.CONFIG_DIR / "ppo.yaml", "4070", [steps(40)])
    assert config["updates"] == 40
    assert schedule(39, config).actor_on and schedule(39, config).actor_lr > 0
    with pytest.raises(ValueError, match="no actor updates after the critic warm-up"):
        ppo.load_ppo_config(ppo.CONFIG_DIR / "ppo.yaml", "4070", [steps(20)])


def test_ppo_rejects_plastic_lr_scale_in_yaml_and_overrides(tmp_path):
    defaults = ppo.CONFIG_DIR / "ppo.yaml"
    config = tmp_path / "ppo.yaml"
    config.write_text(defaults.read_text() + "\nplastic_lr_scale: 3.0\n")
    with pytest.raises(ValueError, match="same keys as ppo.yaml"):
        ppo.load_ppo_config(config, "2060")
    with pytest.raises(ValueError, match="Expected an existing setting.*plastic_lr_scale=3.0"):
        ppo.load_ppo_config(defaults, "2060", ["plastic_lr_scale=3.0"])


# ---------------------------------------------------------------------------------------------
# #138 A2 third version: the occlusion speed term


def test_speed_groups_split_present_odor_by_the_teachers_slowdown():
    cases = torch.tensor([0, 1, 1, 2, 3, 1], dtype=torch.uint8)
    # mean odor / near (2.0): 0.25, 0.25, 0.05, -, -, exactly 0.1
    odor = torch.tensor([[0.5, 0.5], [0.4, 0.6], [0.1, 0.1], [0.0, 0.0], [0.9, 0.9], [0.2, 0.2]])
    groups = ppo.occlusion_speed_groups(cases, odor, near=2.0, strong_slowdown=0.1)
    assert groups.tolist() == [0, 1, 2, 3, 0, 1]


def test_speed_term_is_the_mean_over_groups_of_each_groups_mean_whatever_the_parts():
    generator = torch.Generator().manual_seed(3)
    speed = torch.rand(6, 10, generator=generator) * 0.4
    label = torch.rand(6, 10, generator=generator) * 0.4
    group = torch.randint(0, 3, (6, 10), generator=generator).to(torch.uint8)  # no group 3
    counts = ppo.speed_group_counts(group)
    assert counts[2] == 0
    scale = 0.05
    error = torch.nn.functional.huber_loss(
        (speed - label) / scale, torch.zeros_like(speed), reduction="none"
    )
    expected = sum(error[group == g].mean() for g in (1, 2)) / 2
    whole = ppo.occlusion_speed_term(speed, label, group, counts, scale)
    split = sum(
        ppo.occlusion_speed_term(speed[:, c], label[:, c], group[:, c], counts, scale)
        for c in (slice(0, 3), slice(3, 10))
    )
    torch.testing.assert_close(whole, expected)
    torch.testing.assert_close(split, expected)
    empty = torch.zeros_like(group)
    assert ppo.occlusion_speed_term(speed, label, empty, ppo.speed_group_counts(empty), scale) == 0


def test_the_speed_term_pulls_the_mode_speed_to_the_teachers_label():
    """Zero reward (no policy gradient beyond noise): only the speed term moves the speed, and
    only on the grouped steps' label. v_max is 1 in the toy policy, so the speed starts at 0.5."""
    n, steps = 32, 4

    def env(t, latent):
        return torch.ones(n, 1), torch.ones(n, 1), torch.zeros(n), torch.zeros(n, dtype=torch.bool)

    policy, critic = _ToyPolicy(1), Critic(1, (16,))
    actor = torch.optim.Adam(policy.parameters(), lr=1e-2)
    value = torch.optim.Adam(critic.parameters(), lr=1e-2)
    generator = torch.Generator().manual_seed(4)
    for _ in range(150):
        rollout = _toy_rollout(policy, env, n, steps, generator)
        rollout.speed_label = torch.full((steps, n), 0.3)
        rollout.speed_group = torch.full((steps, n), 3, dtype=torch.uint8)
        metrics = ppo_update(
            policy, critic, actor, value, rollout, _PROBE, actor_on=True, kick_lambda=0.0,
            micro_batch=n // 4, generator=generator, speed_coef=1.0,
        )  # fmt: skip
    speed = policy.forward_code(torch.ones(1, 1), torch.zeros(1, policy.hidden)).dist.mode()[0, 0]
    assert float(speed) == pytest.approx(0.3, abs=0.02)
    assert float(metrics["speed_loss"]) < 0.1


def test_the_occlusion_speed_is_off_by_default():
    config = ppo.load_ppo_config(ppo.CONFIG_DIR / "ppo.yaml", "2060")
    assert config["occlusion_speed"]["coef"] == config["occlusion_turn"]["coef"] == 0.0


@pytest.mark.slow
def test_ppo_with_the_occlusion_speed_labels_every_step_and_resumes(tmp_path):
    from flytrack_v2.model import build_policy, export_policy

    init = export_policy(
        build_policy("VO2", {"rest_settle_s": 0.05, "rest_refresh_s": 0.05}),
        tmp_path / "stage1",
        provenance={"git_sha": None, "dirty": None, "config_sha256": "c" * 64, "stages": []},
    )
    overrides = [
        "envs_per_instance=2", "rollout_steps=3", "minibatches=2", "total_env_steps=96",
        "warmup.critic_only_updates=1", "warmup.actor_ramp_updates=1", "checkpoint_every=1",
        "occlusion_speed.coef=1.0", "occlusion_turn.coef=1.0",
        "occlusion_speed.catchup_min_distance_m=1.0",
    ]  # fmt: skip
    config = ppo.load_ppo_config(ppo.CONFIG_DIR / "ppo.yaml", "2060", overrides)
    run = ppo.PPORun(config, init, 1, torch.device("cpu"), tmp_path / "ppo")
    run.prepare(resume=False)
    rollout, collected = run.collector.collect_ppo(3, kickstart=False)
    assert rollout.speed_label.shape == rollout.speed_group.shape == (3, config["rows"])
    assert torch.isfinite(rollout.speed_label).all()
    assert {"speed_bias_lost", "speed_share_strong", "turn_gap_weak"} <= set(collected)
    assert rollout.turn_label.shape == rollout.speed_label.shape
    # Nothing is occluded in the first seconds: no step is grouped, and the teacher's
    # occlusion flag (state column 4) is still 0.
    assert int(rollout.speed_group.sum()) == 0
    assert float(run.collector.teacher_state[:, 4].abs().sum()) == 0
    assert not run.collector.seen_occlusion.any()
    assert "seen_occlusion" in run.collector.checkpoint()
    args = ["--init", str(init), "--seed", "1", "--host", "2060", "--device", "cpu"]
    out = tmp_path / "main"
    args += ["--out", str(out), "--override", *overrides]
    out.mkdir()
    (out / "STOP").touch()
    assert ppo.main(args) == 0
    (out / "STOP").unlink()
    assert ppo.main([*args, "--resume"]) == 0
    rows = [json.loads(line) for line in (out / "metrics.jsonl").read_text().splitlines()]
    assert [row["update"] for row in rows] == [1, 2, 3, 4]
    assert "speed_loss" not in rows[0] and "speed_loss" in rows[1] and "turn_loss" in rows[1]
    assert all(math.isfinite(row["speed_bias_strong"]) for row in rows)


def test_lost_speed_scale_slows_only_the_labelling_teacher(tmp_path):
    """#138 A2 eleventh version: occlusion_speed.lost_speed_scale rebuilds the labelling teacher
    with that scale (odor lost: scale x the held speed); near, and so the groups, stay."""
    from flytrack_v2.model import build_policy, export_policy

    init = export_policy(
        build_policy("VO2", {"rest_settle_s": 0.05, "rest_refresh_s": 0.05}),
        tmp_path / "stage1",
        provenance={"git_sha": None, "dirty": None, "config_sha256": "c" * 64, "stages": []},
    )
    base = [
        "envs_per_instance=2", "rollout_steps=3", "minibatches=2", "total_env_steps=96",
        "warmup.critic_only_updates=1", "warmup.actor_ramp_updates=1",
        "occlusion_speed.coef=1.0", "occlusion_turn.coef=1.0",
    ]  # fmt: skip
    teachers = {}
    for scale in (1.0, 0.5):
        config = ppo.load_ppo_config(
            ppo.CONFIG_DIR / "ppo.yaml", "2060", [*base, f"occlusion_speed.lost_speed_scale={scale}"]
        )
        run = ppo.PPORun(config, init, 1, torch.device("cpu"), tmp_path / f"ppo_{scale}")
        run.prepare(resume=False)
        teachers[scale] = run.collector.teacher
    assert teachers[1.0].lost_speed_scale == 1.0 and teachers[0.5].lost_speed_scale == 0.5
    assert teachers[0.5].near == teachers[1.0].near and teachers[0.5].memory
    assert teachers[0.5].walls == teachers[1.0].walls
    for bad in (0, 1.5):
        config = ppo.load_ppo_config(
            ppo.CONFIG_DIR / "ppo.yaml", "2060", [*base, f"occlusion_speed.lost_speed_scale={bad}"]
        )
        with pytest.raises(ValueError, match="lost_speed_scale"):
            ppo.PPORun(config, init, 1, torch.device("cpu"), tmp_path / f"bad_{bad}").prepare(
                resume=False
            )


def test_the_turn_term_pulls_the_mode_turn_to_the_teachers_label():
    """#138 A2 fourth version: the turn term alone moves the turn rate (omega_max 1 in the toy
    policy, so it starts at 0) to the grouped steps' label, and leaves the speed alone."""
    n, steps = 32, 4

    def env(t, latent):
        return torch.ones(n, 1), torch.ones(n, 1), torch.zeros(n), torch.zeros(n, dtype=torch.bool)

    policy, critic = _ToyPolicy(1), Critic(1, (16,))
    actor = torch.optim.Adam(policy.parameters(), lr=1e-2)
    value = torch.optim.Adam(critic.parameters(), lr=1e-2)
    generator = torch.Generator().manual_seed(5)
    for _ in range(150):
        rollout = _toy_rollout(policy, env, n, steps, generator)
        rollout.speed_label = torch.full((steps, n), 0.3)
        rollout.turn_label = torch.full((steps, n), -0.4)
        rollout.speed_group = torch.full((steps, n), 1, dtype=torch.uint8)
        metrics = ppo_update(
            policy, critic, actor, value, rollout, _PROBE, actor_on=True, kick_lambda=0.0,
            micro_batch=n // 4, generator=generator, turn_coef=1.0,
        )  # fmt: skip
    with torch.no_grad():
        mode = policy.forward_code(torch.ones(1, 1), torch.zeros(1, policy.hidden)).dist.mode()[0]
    assert float(mode[1]) == pytest.approx(-0.4, abs=0.03)
    assert float(mode[0]) == pytest.approx(0.5, abs=0.1)  # no speed term: only PPO noise
    assert "turn_loss" in metrics and "speed_loss" not in metrics


def test_the_turn_term_leaves_the_catchup_group_out():
    """Catch-up steps (group 4) carry a speed label only: with every step in that group, the
    turn coefficient changes nothing (same seed, same updates), and the speed reaches its label."""
    n, steps = 32, 4

    def env(t, latent):
        return torch.ones(n, 1), torch.ones(n, 1), torch.zeros(n), torch.zeros(n, dtype=torch.bool)

    def train(turn_coef):
        torch.manual_seed(7)
        policy, critic = _ToyPolicy(1), Critic(1, (16,))
        actor = torch.optim.Adam(policy.parameters(), lr=1e-2)
        value = torch.optim.Adam(critic.parameters(), lr=1e-2)
        generator = torch.Generator().manual_seed(6)
        for _ in range(100):
            rollout = _toy_rollout(policy, env, n, steps, generator)
            rollout.speed_label = torch.full((steps, n), 0.3)
            rollout.turn_label = torch.full((steps, n), -0.4)
            rollout.speed_group = torch.full((steps, n), ppo.CATCHUP, dtype=torch.uint8)
            metrics = ppo_update(
                policy, critic, actor, value, rollout, _PROBE, actor_on=True, kick_lambda=0.0,
                micro_batch=n // 4, generator=generator, speed_coef=1.0, turn_coef=turn_coef,
            )  # fmt: skip
        return policy, metrics

    with_turn, metrics = train(1.0)
    without, _ = train(0.0)
    for a, b in zip(with_turn.parameters(), without.parameters(), strict=True):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    assert float(metrics["turn_loss"]) == 0.0
    with torch.no_grad():
        state = torch.zeros(1, with_turn.hidden)
        mode = with_turn.forward_code(torch.ones(1, 1), state).dist.mode()[0]
    assert float(mode[0]) == pytest.approx(0.3, abs=0.02)


@pytest.mark.slow
def test_the_speed_term_can_leave_the_speed_readout_weights_alone():
    """#138 A2 fifth version: the mode's speed is the same either way; without speed_readout its
    gradient reaches the speed bias (and the graph) but not the speed readout weights."""
    from flytrack_v2.model import build_policy

    torch.manual_seed(8)
    policy = build_policy("VO2", {"rest_settle_s": 0.05, "rest_refresh_s": 0.05})
    n, steps = 2, 2
    none = torch.zeros(steps, n, dtype=torch.bool)
    rollout = Rollout(
        code=torch.randn(steps, n, policy.code_size) * 0.1,
        latent=torch.zeros(steps, n, 2),
        logp=torch.zeros(steps, n),
        features=torch.zeros(steps, n, 1),
        reward=torch.zeros(steps, n),
        terminated=none,
        truncated=none,
        final_features=torch.zeros(steps, n, 1),
        last_features=torch.zeros(n, 1),
        central=policy.initial_central_state(n, "cpu"),
        initial=policy.initial_central_state(1, "cpu"),
    )
    found = {}
    for flag in (True, False):
        policy.zero_grad(set_to_none=True)
        *_, mode = ppo._actor_replay(policy, rollout, torch.arange(n), speed_readout=flag)
        mode[..., 0].sum().backward()
        weight = policy.readout.speed_weight.grad
        found[flag] = (mode.detach(), weight, policy.readout.speed_bias.grad.clone())
    torch.testing.assert_close(found[True][0], found[False][0])
    assert float(found[True][1].abs().sum()) > 0
    assert found[False][1] is None or float(found[False][1].abs().sum()) == 0
    torch.testing.assert_close(found[True][2], found[False][2])


def test_group_table_keeps_only_the_named_groups():
    """#138 A2 eighth version: the defaults keep every earlier version's groups."""
    assert ppo.group_table(ppo.SPEED_GROUPS, "cpu").tolist() == [0, 1, 2, 3, 4]
    assert ppo.group_table(ppo.TURN_GROUPS, "cpu").tolist() == [0, 1, 2, 3, 0]
    assert ppo.group_table(("strong", "weak"), "cpu").tolist() == [0, 1, 2, 0, 0]
    for bad in ((), ("near",), ("weak", "weak")):
        with pytest.raises(ValueError, match="groups"):
            ppo.group_table(bad, "cpu")


def test_training_on_some_groups_equals_dropping_the_others():
    """Speed and turn terms on (strong, weak) only train exactly as if the lost and catch-up
    steps had no group (same seed, same updates)."""
    n, steps = 32, 4

    def env(t, latent):
        return torch.ones(n, 1), torch.ones(n, 1), torch.zeros(n), torch.zeros(n, dtype=torch.bool)

    groups = torch.tensor([1, 2, 3, 4], dtype=torch.uint8).repeat(steps, n // 4)

    def train(group, **kwargs):
        torch.manual_seed(7)
        policy, critic = _ToyPolicy(1), Critic(1, (16,))
        actor = torch.optim.Adam(policy.parameters(), lr=1e-2)
        value = torch.optim.Adam(critic.parameters(), lr=1e-2)
        generator = torch.Generator().manual_seed(6)
        for _ in range(20):
            rollout = _toy_rollout(policy, env, n, steps, generator)
            rollout.speed_label = torch.linspace(0.1, 0.4, n).repeat(steps, 1)
            rollout.turn_label = torch.linspace(-0.5, 0.5, n).repeat(steps, 1)
            rollout.speed_group = group.clone()
            metrics = ppo_update(
                policy, critic, actor, value, rollout, _PROBE, actor_on=True, kick_lambda=0.0,
                micro_batch=n // 4, generator=generator, speed_coef=1.0, turn_coef=1.0, **kwargs
            )  # fmt: skip
        return policy, metrics

    chosen, metrics = train(groups, speed_groups=("strong", "weak"), turn_groups=("strong", "weak"))
    dropped, reference = train(torch.where(groups > 2, 0, groups).to(torch.uint8))
    for a, b in zip(chosen.parameters(), dropped.parameters(), strict=True):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    for key in ("speed_loss", "turn_loss"):
        torch.testing.assert_close(metrics[key], reference[key], rtol=0, atol=0)
    with pytest.raises(ValueError, match="catch-up"):
        train(groups, turn_groups=("strong", "catchup"))
