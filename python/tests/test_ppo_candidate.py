"""Candidate parameter rates, critic warmup, and bitwise recovery on CPU."""

import copy

import pytest
import torch
from flytrack_core.contracts import SensorContract
from flytrack_train import ppo
from flytrack_train.recovery import Recovery
from test_training import _assert_nested_equal, _RecoveryEnv


class CandidatePolicy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.core = torch.nn.Linear(4, 2)
        self.readout = torch.nn.Linear(2, 2)
        self.log_std = torch.nn.Parameter(torch.full((2,), -1.0))
        self.population_masks = {
            key: torch.ones(2, dtype=torch.bool) for key in ("visual", "olfactory", "convergence")
        }

    def initial_state(self, batch, device):
        return torch.zeros(batch, 2, device=device)

    def forward(self, observation, state):
        hidden = torch.tanh(self.core(torch.cat((observation["odor"], state), dim=1)))
        output = self.readout(hidden)
        action = torch.stack(
            (torch.sigmoid(output[:, 0]) * 0.5, torch.tanh(output[:, 1]) * 1.5), dim=1
        )
        return action, hidden, hidden


@pytest.fixture
def candidate(monkeypatch):
    monkeypatch.setattr(ppo, "prepare_policy", lambda *args: (CandidatePolicy(), SensorContract()))
    monkeypatch.setattr(ppo, "make_env", lambda *args, **kwargs: _RecoveryEnv())
    monkeypatch.setattr(ppo, "save_actor", lambda *args, **kwargs: None)
    return {
        "train": {
            "num_envs": 2,
            "total_steps": 16,
            "rollout_steps": 2,
            "sequence_length": 2,
            "sequence_batch_size": 2,
            "update_epochs": 2,
            "critic_hidden_size": 4,
        }
    }


def load(output):
    return torch.load(output / "training_checkpoint.pt", weights_only=True)


def test_default_two_updates_are_bitwise_equal_to_previous_optimizer(
    candidate, tmp_path, monkeypatch
):
    candidate["train"]["total_steps"] = 8
    ppo.train_ppo(candidate, tmp_path / "current", 11, torch.device("cpu"))
    # Previous code: a single actor group at base lr, a separate critic group,
    # no per-update lr scheduling, and no zero-rate gradient filtering.
    monkeypatch.setattr(
        ppo,
        "parameter_groups",
        lambda policy, train: [{"params": list(policy.parameters()), "lr": train["learning_rate"]}],
    )
    monkeypatch.setattr(ppo, "_set_update_learning_rates", lambda *args: False)
    monkeypatch.setattr(ppo, "_freeze_zero_rate_gradients", lambda *args: None)
    ppo.train_ppo(candidate, tmp_path / "previous", 11, torch.device("cpu"))
    current, previous = load(tmp_path / "current"), load(tmp_path / "previous")
    assert current["iteration"] == previous["iteration"] == 2
    for key in ("actor", "critic", "optimizer", "rng", "reward_normalizer"):
        _assert_nested_equal(current[key], previous[key])
    for left, right in zip(current["reports"], previous["reports"], strict=True):
        _assert_nested_equal(
            {k: v for k, v in left.items() if k != "elapsed_s"},
            {k: v for k, v in right.items() if k != "elapsed_s"},
        )


def test_bias_freeze_readout_rate_and_warmup(candidate, tmp_path, monkeypatch):
    candidate["train"].update(
        learning_rate=1e-3,
        core_bias_learning_rate=0,
        readout_learning_rate=1e-2,
        critic_warmup_updates=2,
    )
    snapshots = []
    original = Recovery.save

    def record(self, rng, **state):
        snapshots.append(copy.deepcopy(state))
        original(self, rng, **state)

    monkeypatch.setattr(Recovery, "save", record)
    ppo.seed_all(11)
    actor = CandidatePolicy().state_dict()
    critic = ppo.PrivilegedCritic(hidden_size=4).state_dict()
    ppo.train_ppo(candidate, tmp_path, 11, torch.device("cpu"))
    for index, snapshot in enumerate(snapshots):
        assert torch.equal(snapshot["actor"]["core.bias"], actor["core.bias"])
        if index < 2:
            _assert_nested_equal(snapshot["actor"], actor)
            assert any(not torch.equal(v, critic[k]) for k, v in snapshot["critic"].items())
            assert snapshot["reports"][-1]["critic_warmup"]
            assert snapshot["reports"][-1]["actor_learning_rates"] == [0, 0, 0]
            for group in (
                snapshot["optimizer"]["param_groups"][0],
                snapshot["optimizer"]["param_groups"][2],
            ):
                assert all(p in snapshot["optimizer"]["state"] for p in group["params"])
        else:
            assert not torch.equal(snapshot["actor"]["readout.weight"], actor["readout.weight"])
            assert snapshot["reports"][-1]["actor_learning_rates"] == [1e-3, 0, 1e-2]
            assert not snapshot["reports"][-1]["critic_warmup"]
        assert snapshot["optimizer"]["param_groups"][-1]["lr"] == 1e-3
    # Frozen parameters never acquire Adam state, even after many minibatches.
    frozen = snapshots[-1]["optimizer"]["param_groups"][1]["params"]
    assert all(index not in snapshots[-1]["optimizer"]["state"] for index in frozen)


def test_warmup_checkpoint_resumes_bitwise_and_restores_configured_rates(
    candidate, tmp_path, monkeypatch
):
    candidate["train"].update(
        learning_rate=1e-3,
        core_bias_learning_rate=0,
        readout_learning_rate=1e-2,
        critic_warmup_updates=2,
    )
    full, interrupted = tmp_path / "full", tmp_path / "interrupted"
    ppo.train_ppo(candidate, full, 11, torch.device("cpu"))
    original = Recovery.save

    def stop(self, rng, **state):
        original(self, rng, **state)
        raise InterruptedError("checkpoint during warmup")

    monkeypatch.setattr(Recovery, "save", stop)
    with pytest.raises(InterruptedError):
        ppo.train_ppo(candidate, interrupted, 11, torch.device("cpu"))
    saved = load(interrupted)
    assert saved["iteration"] == 1
    assert all(group["lr"] == 0 for group in saved["optimizer"]["param_groups"][:-1])
    monkeypatch.setattr(Recovery, "save", original)
    ppo.train_ppo({**candidate, "resume": True}, interrupted, 11, torch.device("cpu"))
    expected, observed = load(full), load(interrupted)
    for key in (
        "actor",
        "critic",
        "optimizer",
        "rng",
        "reward_normalizer",
        "recurrent_state",
        "observations",
        "privileged",
        "action_histories",
    ):
        _assert_nested_equal(expected[key], observed[key])
    assert observed["reports"][2]["actor_learning_rates"] == [1e-3, 0, 1e-2]


def test_zero_rate_groups_skip_weight_decay_and_existing_momentum():
    policy = CandidatePolicy()
    critic = torch.nn.Linear(2, 1)
    optimizer = torch.optim.Adam(
        [
            *ppo.parameter_groups(policy, {"core_bias_learning_rate": 0}),
            {"params": critic.parameters()},
        ],
        weight_decay=0.2,
    )
    # Populate moments first, as may happen in an optimizer loaded from disk.
    optimizer.param_groups[1]["lr"] = 1e-3
    for parameter in policy.parameters():
        parameter.grad = torch.ones_like(parameter)
    optimizer.step()
    optimizer.param_groups[1]["lr"] = 0
    before = policy.core.bias.detach().clone()
    moments = copy.deepcopy(optimizer.state[policy.core.bias])
    for _ in range(3):
        for parameter in policy.parameters():
            parameter.grad = torch.ones_like(parameter)
        ppo._freeze_zero_rate_gradients([policy.core.bias])
        optimizer.step()
    assert torch.equal(before, policy.core.bias)
    _assert_nested_equal(optimizer.state[policy.core.bias], moments)


def test_readout_rate_controls_actual_adam_step_after_restoring_zero_rates():
    policy = CandidatePolicy().double()
    train = {
        "learning_rate": 1e-3,
        "readout_learning_rate": 1e-2,
        "core_bias_learning_rate": 0,
        "critic_warmup_updates": 10,
    }
    optimizer = torch.optim.Adam(
        [*ppo.parameter_groups(policy, train), {"params": torch.nn.Linear(2, 1).parameters()}],
        eps=1e-5,
    )
    before = copy.deepcopy(policy.state_dict())
    for group in optimizer.param_groups:
        group["lr"] = 0  # Simulate saved warmup rates.
    assert not ppo._set_update_learning_rates(optimizer, policy, train, 11)
    for parameter in policy.parameters():
        parameter.grad = torch.ones_like(parameter)
    ppo._freeze_zero_rate_gradients([policy.core.bias])
    optimizer.step()
    for name, value in policy.state_dict().items():
        rate = 0 if name == "core.bias" else (1e-2 if name.startswith("readout.") else 1e-3)
        torch.testing.assert_close(before[name] - value, torch.full_like(value, rate / 1.00001))


@pytest.mark.parametrize("warmup", [-1, 1.5, True, None, "2"])
def test_invalid_warmup_is_rejected(tmp_path, warmup):
    with pytest.raises(ValueError, match="nonnegative integer"):
        ppo.train_ppo({"train": {"critic_warmup_updates": warmup}}, tmp_path, 0, "cpu")


@pytest.mark.parametrize("rate", [float("nan"), -1e-3, True, False])
def test_invalid_readout_bias_rate_is_rejected_at_start(tmp_path, monkeypatch, rate):
    def unexpected_seed(*args):
        pytest.fail("Invalid rates must be rejected before training setup")

    monkeypatch.setattr(ppo, "seed_all", unexpected_seed)
    with pytest.raises(
        ValueError, match=r"train\.readout_bias_learning_rate must be a finite nonnegative number"
    ):
        ppo.train_ppo({"train": {"readout_bias_learning_rate": rate}}, tmp_path, 0, "cpu")


def test_zero_readout_bias_rate_freezes_biases_through_ppo_update(candidate, tmp_path, monkeypatch):
    class Readout(torch.nn.Linear):
        def __init__(self):
            super().__init__(2, 2)
            self.forward_bias = torch.nn.Parameter(torch.tensor(0.1))
            self.turn_bias = torch.nn.Parameter(torch.tensor(-0.1))

        def forward(self, hidden):
            return super().forward(hidden) + torch.stack((self.forward_bias, self.turn_bias))

    policy = CandidatePolicy()
    policy.readout = Readout()
    before = copy.deepcopy(policy.state_dict())
    monkeypatch.setattr(ppo, "prepare_policy", lambda *args: (policy, SensorContract()))
    candidate["train"].update(total_steps=4, readout_bias_learning_rate=0)
    frozen_names = []
    original = ppo._freeze_zero_rate_gradients

    def record(parameters):
        frozen_ids = {id(parameter) for parameter in parameters}
        frozen_names.append(
            {name for name, parameter in policy.named_parameters() if id(parameter) in frozen_ids}
        )
        original(parameters)

    monkeypatch.setattr(ppo, "_freeze_zero_rate_gradients", record)
    ppo.train_ppo(candidate, tmp_path, 11, torch.device("cpu"))
    checkpoint = load(tmp_path)
    assert checkpoint["iteration"] == 1
    expected = {"readout.turn_bias", "readout.forward_bias"}
    assert frozen_names
    assert all(expected <= names for names in frozen_names)
    for name in expected:
        assert torch.equal(
            checkpoint["actor"][name].reshape(-1).view(torch.uint8),
            before[name].reshape(-1).view(torch.uint8),
        )
    assert not torch.equal(checkpoint["actor"]["readout.weight"], before["readout.weight"])
