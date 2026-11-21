"""Structure-selected gain plasticity, optimizer projection and artifact compatibility."""

import json
from dataclasses import replace

import pytest
import torch
from flytrack_core.contracts import file_sha256
from flytrack_v2.contracts import SensorContractV2
from flytrack_v2.model import (
    ModelConfigV2,
    build_policy,
    export_policy,
    load_policy,
    parameter_groups,
    parameter_table,
)
from flytrack_v2.model.core import ConnectomeCoreV2, resolve_plastic_indices
from flytrack_v2.model.policy import attach_plastic_projection
from flytrack_v2.train import dagger, ppo
from flytrack_v2.train.config import CONFIG_DIR, ROOT, load_config
from test_v2_model_core import small_graph

LABELS = (("ORN", "Turn", 0, 1.0), ("ORN", "Forward", 0, 2.0))
CONFIG = ModelConfigV2(rest_settle_s=0.05, rest_refresh_s=0.05)
PLASTIC = replace(CONFIG, plastic_labels=LABELS, plastic_lr_scale=10, plastic_floor=0.3)
CPU = torch.device("cpu")
PROVENANCE = {"git_sha": None, "dirty": None, "config_sha256": "a" * 64, "stages": []}


def _build(config=PLASTIC):
    return build_policy("VO2", config, graph=small_graph())


@torch.no_grad()
def test_zero_delta_preserves_output_and_state_dict():
    plain, plastic = _build(CONFIG), _build()
    assert not hasattr(plain.core, "gain_delta")
    state = dict(plastic.state_dict())
    assert torch.equal(state.pop("core.gain_delta"), torch.zeros(len(LABELS)))
    torch.testing.assert_close(state, dict(plain.state_dict()), atol=0, rtol=0)
    code = torch.rand(2, plain.code_size, generator=torch.Generator().manual_seed(100))
    first, second = [
        policy.forward_code(code, policy.initial_central_state(2, CPU))
        for policy in (plain, plastic)
    ]
    assert torch.equal(first.central_state, second.central_state)
    assert torch.equal(first.dist.mean, second.dist.mean)
    assert torch.equal(first.dist.log_std, second.dist.log_std)
    for key, value in first.aux.items():
        assert torch.equal(value, second.aux[key])
    assert parameter_table(plastic)["gain_delta"] == len(LABELS)
    assert parameter_table(plastic)["total"] == parameter_table(plain)["total"] + len(LABELS)


@pytest.mark.parametrize("dtype", (torch.float32, torch.float64))
def test_bounds_and_inward_gradient(dtype):
    core = _build().core.to(dtype)
    with torch.no_grad():
        core.gain_delta.copy_(torch.tensor([-100.0, 100.0]))
    limits = torch.stack((core.plastic_lo[0], core.plastic_hi[1]))
    assert torch.equal(core.gains()[core.plastic_index], limits)
    core.project_plastic_()
    raw = torch.nn.functional.softplus(core.gain_theta[core.plastic_index] + core.gain_delta)
    assert bool(((raw >= core.plastic_lo) & (raw <= core.plastic_hi)).all())
    torch.testing.assert_close(raw, limits, atol=1e-6, rtol=0)

    direction = raw.new_tensor([-1.0, 1.0])
    (core.gains()[core.plastic_index] * direction).sum().backward()
    assert bool((core.gain_delta.grad * direction > 0).all())
    with torch.no_grad():
        core.gain_delta.add_(core.gain_delta.grad, alpha=-0.1)
    gains = core.gains()[core.plastic_index]
    assert bool(((gains > core.plastic_lo) & (gains < core.plastic_hi)).all())


def test_clamp_gradient_at_exact_boundary():
    value = torch.tensor([-1.0, 1.0], requires_grad=True)
    value.clamp(-1.0, 1.0).sum().backward()
    assert torch.equal(value.grad, torch.ones_like(value))


def _assert_projected_step(policy, optimizer):
    core = policy.core
    with torch.no_grad():
        core.gain_delta.fill_(100)
    core.gain_delta.grad = torch.zeros_like(core.gain_delta)
    optimizer.step()
    raw = torch.nn.functional.softplus(core.gain_theta[core.plastic_index] + core.gain_delta)
    assert bool(((raw >= core.plastic_lo) & (raw <= core.plastic_hi)).all())
    assert core.gain_delta in optimizer.state


class _Collector:
    def __init__(self, *_args, **_kwargs):
        pass

    def checkpoint(self):
        return {}

    def restore(self, state):
        assert state == {}


def test_dagger_groups_schedule_and_resume_hook(tmp_path, monkeypatch):
    config = load_config(
        CONFIG_DIR / "dagger.yaml", "2060", ["readout_normalization=false", "rounds.beta=[1.0]"]
    )
    monkeypatch.setattr(dagger, "build_training_policy", lambda *_args, **_kwargs: _build())
    monkeypatch.setattr(dagger, "RolloutCollector", _Collector)
    fresh = dagger.DAggerRun(config, "VO2", 100, CPU, tmp_path)
    fresh.prepare(None)
    opt = fresh.optimizer
    main, plastic = opt.param_groups[0], opt.param_groups[-1]
    assert all(param is not fresh.policy.core.gain_delta for param in main["params"])
    assert len(plastic["params"]) == 1 and plastic["params"][0] is fresh.policy.core.gain_delta
    assert plastic["lr_scale"] == 10
    gamma = config["optimizer"]["lr"]
    assert plastic["lr"] == pytest.approx(10 * gamma)
    for _ in range(2):
        _assert_projected_step(fresh.policy, opt)
        fresh.scheduler.step()
        assert plastic["lr"] == pytest.approx(10 * main["lr"])
    fresh.save(tmp_path / "initial.pt")

    resumed = dagger.DAggerRun(config, "VO2", 100, CPU, tmp_path)
    resumed.prepare((tmp_path / "config.json").read_text())
    assert resumed.scheduler.last_epoch == fresh.scheduler.last_epoch
    assert resumed.optimizer.state[resumed.policy.core.gain_delta]["step"].item() == 2
    _assert_projected_step(resumed.policy, resumed.optimizer)


@pytest.mark.parametrize("scale", (None, 3))
def test_ppo_groups_schedule_and_resume_hook(tmp_path, monkeypatch, scale):
    overrides = [] if scale is None else [f"plastic_lr_scale={scale}"]
    config = ppo.load_ppo_config(CONFIG_DIR / "ppo.yaml", "2060", overrides)
    init = export_policy(_build(), tmp_path / "init", provenance=PROVENANCE)
    monkeypatch.setattr(ppo, "PPOCollector", _Collector)
    out = tmp_path / "ppo"
    fresh = ppo.PPORun(config, init, 100, CPU, out)
    fresh.prepare(resume=False)
    fresh._set_rates(ppo.Schedule(True, 0.001, 0.002, 0.0))
    opt = fresh.actor_optimizer
    scale = 1.0 if scale is None else scale
    assert opt.param_groups[-1]["lr_scale"] == scale
    assert opt.param_groups[-1]["lr"] == pytest.approx(scale * opt.param_groups[0]["lr"])
    _assert_projected_step(fresh.policy, opt)
    fresh.save()

    resumed = ppo.PPORun(config, init, 100, CPU, out)
    resumed.prepare(resume=True)
    assert resumed.actor_optimizer.state[resumed.policy.core.gain_delta]["step"].item() == 1
    _assert_projected_step(resumed.policy, resumed.actor_optimizer)


@pytest.mark.parametrize("variant", ("VO2", "GRU2"))
def test_disabled_projection_does_not_register_hook(variant):
    config = replace(CONFIG, budget_target=12_000) if variant == "GRU2" else CONFIG
    policy = build_policy(variant, config, graph=small_graph())
    optimizer = torch.optim.Adam(parameter_groups(policy, 0.01))
    attach_plastic_projection(optimizer, policy)
    assert not optimizer._optimizer_step_post_hooks
    optimizer.step()


@torch.no_grad()
def test_official_list_bilateral_export_and_cli_override(tmp_path, monkeypatch):
    card = json.loads((ROOT / "scripts/v2/explore/a3_plastic_labels.json").read_text())
    labels = [[row[k] for k in ("pre", "post", "code", "cap")] for row in card["labels"]]
    parsed = {}

    class CaptureRun:
        def __init__(self, config, *_args):
            parsed.update(config["model"])

        def prepare(self, _saved_text):
            pass

        def run(self):
            pass

    monkeypatch.setattr(dagger, "DAggerRun", CaptureRun)
    arguments = [
        "--variant=VO2",
        "--seed=100",
        "--host=2060",
        "--device=cpu",
        f"--out={tmp_path / 'capture'}",
        "--override",
        "model.graph_path=data/subgraphs/v2_cx_lat/graph.npz",
        "model.speed_types=bilateral",
        f"model.plastic_labels={json.dumps(labels)}",
        "model.plastic_lr_scale=10",
        "model.plastic_floor=0.3",
    ]
    assert dagger.main(arguments) == 0
    config = replace(ModelConfigV2.from_dict(parsed), rest_settle_s=0.05, rest_refresh_s=0.05)
    assert len(config.plastic_labels) == 52
    assert config.plastic_labels == tuple(map(tuple, labels))
    assert ModelConfigV2.from_dict(json.loads(json.dumps(config.to_dict()))) == config
    policy = build_policy("VO2", config)
    assert policy.core.plastic_index.tolist() == [row["label"] for row in card["labels"]]
    policy.core.gain_delta.fill_(-0.123)
    artifact = export_policy(policy, tmp_path / "final", provenance=PROVENANCE)
    restored = load_policy(artifact)
    assert len(restored.readout.speed_types) > 1
    assert restored.config.plastic_labels == config.plastic_labels
    assert restored.config.plastic_lr_scale == 10
    assert restored.config.plastic_floor == 0.3
    assert torch.equal(restored.core.gain_delta, policy.core.gain_delta)
    assert torch.equal(restored.core.gains(), policy.core.gains())
    code = torch.rand(2, policy.code_size, generator=torch.Generator().manual_seed(100))
    first, second = [
        actor.forward_code(code, actor.initial_central_state(2, CPU))
        for actor in (policy, restored)
    ]
    assert torch.equal(first.dist.mean, second.dist.mean)
    assert torch.equal(first.central_state, second.central_state)


@torch.no_grad()
def test_legacy_card_uses_plastic_defaults(tmp_path):
    policy = _build(CONFIG)
    artifact = export_policy(policy, tmp_path / "legacy", provenance=PROVENANCE)
    path = artifact / "model_card.json"
    card = json.loads(path.read_text())
    for key in ("plastic_labels", "plastic_floor", "plastic_lr_scale"):
        del card["model_config"][key]
    path.write_text(json.dumps(card))
    checksums = {**card["files"], path.name: file_sha256(path)}
    (artifact / "SHA256SUMS").write_text(
        "".join(f"{digest}  {name}\n" for name, digest in checksums.items())
    )
    restored = load_policy(artifact)
    assert restored.config == replace(
        CONFIG,
        steer_types=policy.readout.steer_types,
        speed_types=policy.readout.speed_types,
    )
    assert not hasattr(restored.core, "gain_delta")
    torch.testing.assert_close(restored.state_dict(), policy.state_dict(), atol=0, rtol=0)


def test_missing_label_lists_all_entries_and_gru_rejects_labels():
    missing = (("ORN", "Turn", 1, 2.0), ("unknown", "Turn", 0, 1.0))
    match = r"'ORN', 'Turn', 1.*'unknown', 'Turn', 0"
    with pytest.raises(ValueError, match=match):
        resolve_plastic_indices(small_graph(), missing)
    with pytest.raises(ValueError, match=match):
        ConnectomeCoreV2(small_graph(), replace(CONFIG, plastic_labels=missing), SensorContractV2())
    for labels in (LABELS, ()):
        with pytest.raises(ValueError, match="GRU2 does not support plastic_labels"):
            build_policy("GRU2", replace(CONFIG, plastic_labels=labels))


@pytest.mark.parametrize(
    ("changes", "match"),
    [
        ({"plastic_labels": (LABELS[0], LABELS[0])}, "Duplicate"),
        ({"plastic_labels": (("ORN", "Turn", 0, float("inf")),)}, "cap"),
        ({"plastic_labels": (("ORN", "Turn", 0, 0.9),)}, "cap"),
        ({"plastic_floor": 0.0}, "floor"),
        ({"plastic_floor": 1.1}, "floor"),
        ({"plastic_lr_scale": float("nan")}, "lr_scale"),
        ({"plastic_lr_scale": 0}, "lr_scale"),
        ({"gain_parameterization": "bounded"}, "softplus"),
    ],
)
def test_plastic_config_validation(changes, match):
    with pytest.raises(ValueError, match=match):
        replace(PLASTIC, **changes)
