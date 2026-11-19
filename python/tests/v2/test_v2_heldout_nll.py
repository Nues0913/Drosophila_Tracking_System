"""CPU coverage of held-out replay sampling, alignment and NLL reporting."""

import importlib.util
import json
import subprocess
import sys
from argparse import Namespace
from dataclasses import replace
from pathlib import Path

import pytest
import torch
from flytrack_v2.model import ModelConfigV2, build_policy, export_policy, load_policy
from flytrack_v2.train.config import STATE_STRIDE
from flytrack_v2.train.data import Replay, ReplayShard, StepData, _env_major
from flytrack_v2.train.update import LossSettings, batch_metrics, label_margins
from test_v2_model_core import small_graph

SCRIPT = Path(__file__).resolve().parents[3] / "scripts/v2/heldout_nll.py"
_spec = importlib.util.spec_from_file_location("heldout_nll", SCRIPT)
heldout = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(heldout)

BURN_INS = [0, STATE_STRIDE, 2 * STATE_STRIDE]
LENGTH = 4
SEED = 19
SEQUENCES = 7
BATCH_SIZE = 2


@pytest.fixture
def runs(tmp_path):
    policy = build_policy(
        "VO2", ModelConfigV2(rest_settle_s=0.2, rest_refresh_s=0.1), graph=small_graph()
    )
    policy.readout.rate_mean.fill_(0.06)
    policy.readout.rate_scale.fill_(0.04)
    policy.readout.rate_norm_fitted.fill_(True)
    rng = torch.Generator().manual_seed(13)
    run_a, run_b = tmp_path / "A" / "run", tmp_path / "B" / "run"
    config = {
        "sequence": {"length": LENGTH, "burn_in": STATE_STRIDE},
        "micro_batch": BATCH_SIZE,
        "activity_coef": 0.0001,
        "label_latent_bound": 3.0,
        "state_stride": STATE_STRIDE,
    }
    provenance = {"git_sha": None, "dirty": None, "config_sha256": None, "stages": []}
    for run in (run_a, run_b):
        export_policy(policy, run / "final", provenance=provenance)
        (run / "config.json").write_text(json.dumps(config), encoding="utf-8")

    for number, ticks, first in ((10, 65, 5), (2, 57, 3)):
        rows = 4
        code = torch.randn(ticks, rows, policy.code_size, generator=rng)
        code[..., 0] = torch.arange(ticks)[:, None] / 64
        code[..., -4:] = 1
        action = torch.rand(ticks, rows, 2, generator=rng)
        action[..., 0] *= policy.contract.v_max
        action[..., 1] = (2 * action[..., 1] - 1) * policy.contract.omega_max
        reset = torch.rand(ticks, rows, generator=rng) < 0.15
        snapshots = len(range(first, ticks, STATE_STRIDE))
        states = torch.rand(snapshots, rows, policy.central_state_size, generator=rng)
        shard = ReplayShard(
            StepData(
                _env_major(code, torch.float16),
                _env_major(action, torch.float32),
                _env_major(reset, torch.bool),
                torch.zeros(rows, ticks, dtype=torch.bool),
            ),
            _env_major(states, torch.float16),
            torch.arange(rows)[:, None].expand(-1, 3).clone(),
            torch.tensor([True, False, True, False]),
            first,
        )
        for run in (run_a, run_b):
            directory = run / f"round_{number}"
            directory.mkdir()
            if run == run_b:
                # Give the second replay distinct labels even though its weights are identical.
                other_action = torch.full_like(shard.steps.action, policy.contract.v_max)
                shard = replace(shard, steps=shard.steps._replace(action=other_action))
            torch.save(shard.checkpoint(), directory / "replay.pt")
    return run_a, run_b


def test_cli_compares_identical_weights_on_shared_held_out(runs, tmp_path):
    out = tmp_path / "report"
    subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--runs",
            *(str(run) for run in runs),
            "--burn-in",
            *(str(b) for b in BURN_INS),
            "--sequences",
            str(SEQUENCES),
            "--batch-size",
            str(BATCH_SIZE),
            "--device",
            "cpu",
            "--seed",
            str(SEED),
            "--out",
            str(out),
        ],
        check=True,
        stdin=subprocess.DEVNULL,
    )
    report = json.loads((out / "heldout_nll.json").read_text(encoding="utf-8"))
    assert report["shared_replay"] == "A"
    assert report["runs"] == dict(zip(("A", "B"), map(str, runs), strict=True))
    cells = {(r["weights"], r["data"], r["burn_in"]): r for r in report["results"]}
    assert len(cells) == 3 * len(BURN_INS)
    for b in BURN_INS:
        a, shared, own = (cells[w, d, b] for w, d in (("A", "A"), ("B", "A"), ("B", "B")))
        for key in heldout.NLL_KEYS:
            assert a[key] == pytest.approx(shared[key], abs=1e-5, rel=0)
        assert own["nll_speed"] != pytest.approx(shared["nll_speed"])
        for cell in (a, shared, own):
            assert cell["samples"] == SEQUENCES
            assert cell["scored_steps"] == SEQUENCES * LENGTH
    markdown = (out / "heldout_nll.md").read_text(encoding="utf-8")
    assert "Both policies scored the same held-out sequences from replay A." in markdown
    assert f"| B | A | {STATE_STRIDE} | {SEQUENCES} |" in markdown


def test_burn_in_scores_identical_codes_and_uses_aligned_snapshots(runs):
    longest = max(BURN_INS)
    paths = heldout.replay_paths(runs[0])
    assert [path.parent.name for path in paths] == ["round_2", "round_10"]
    draws = list(heldout.held_out_draws(paths, longest + LENGTH, 100, SEED))
    assert sum(len(picks) for _, picks in draws) == 8
    policy = load_policy(runs[0] / "final")
    offsets = [longest - b for b in BURN_INS]
    steps, initials, tags = heldout.read_batch(draws, 0, 8, longest + LENGTH, offsets, policy)
    for path, picks in draws:
        shard = ReplayShard.load(path)
        positions, rows, snaps = picks.unbind(dim=1)
        assert shard.held_out[rows].all()
        scored_ticks = shard.first_snapshot + snaps * STATE_STRIDE + longest
        for burn_in in BURN_INS:
            offset = longest - burn_in
            batch = heldout.aligned_batch(steps, initials[offset], tags, offset)
            assert batch.code.size(0) == burn_in + LENGTH
            torch.testing.assert_close(
                batch.code[burn_in:, positions, 0] * 64,
                (scored_ticks + torch.arange(LENGTH)[:, None]).float(),
                rtol=0,
                atol=0,
            )
            for source, value in zip(steps[:3], batch[:3], strict=True):
                assert torch.equal(value[burn_in:], source[longest:])
            expected_state = shard.states[rows, snaps + offset // STATE_STRIDE].float()
            assert torch.equal(batch.initial[positions], expected_state)


@torch.no_grad()
def test_nll_matches_trainer_on_same_draw_for_each_burn_in(runs):
    report = heldout.probe(
        Namespace(
            runs=runs,
            burn_in=BURN_INS,
            sequences=SEQUENCES,
            batch_size=None,
            device="cpu",
            seed=SEED,
        )
    )
    run = runs[0]
    config = json.loads((run / "config.json").read_text(encoding="utf-8"))
    policy = load_policy(run / "final")
    shards = [ReplayShard.load(path) for path in heldout.replay_paths(run)]
    for result in report["results"]:
        if (result["weights"], result["data"]) != ("A", "A"):
            continue
        burn_in = result["burn_in"]
        delta = max(BURN_INS) - burn_in
        replay = Replay(burn_in, LENGTH)
        # Drop delta ticks and their snapshots so the trainer draws the same
        # (row, scored interval) pairs when configured with the shorter burn-in.
        replay.shards = [
            replace(
                shard,
                steps=StepData(*(tensor[:, delta:] for tensor in shard.steps)),
                states=shard.states[:, delta // STATE_STRIDE :],
            )
            for shard in shards
        ]
        rng = torch.Generator().manual_seed(SEED)
        batch = replay.sample(SEQUENCES, rng, held_out=True)
        settings = LossSettings(burn_in, config["activity_coef"], label_margins(config))
        expected = batch_metrics(policy, batch, settings, BATCH_SIZE, backward=False)
        for key in heldout.NLL_KEYS:
            assert result[key] == pytest.approx(expected[key].item(), abs=1e-5, rel=0)
