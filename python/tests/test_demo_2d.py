"""Synthetic demo diagnostics: no real actors, F3 data or frontend artifacts."""

from __future__ import annotations

import copy
import gzip
import importlib.util
import json
import sys
from pathlib import Path
from threading import Barrier
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from flytrack_core.contracts import SensorContract
from flytrack_core.model.graph import Graph
from flytrack_core.model.policy import MotorReadout
from PIL import Image

DIAGNOSTICS = Path(__file__).resolve().parents[2] / "scripts/diagnostics"


def load_script(name):
    spec = importlib.util.spec_from_file_location(name, DIAGNOSTICS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(DIAGNOSTICS))
    try:
        spec.loader.exec_module(module)
    finally:
        sys.path.pop(0)
    return module


demo = load_script("demo_2d")
trace = load_script("episode_trace")
summary = load_script("a1_summary")


def synthetic_readout(branches=1, extra=True, mode="structured"):
    # Unequal left/right node counts distinguish all-node from side-mean pooling.
    # Unknown-side nodes also contribute to all; an extra forward type must be
    # recorded even though the demo only plots the four required DN types.
    graph = Graph(
        node_ids=np.arange(20),
        type_index=np.repeat(np.arange(5), 4),
        type_names=np.array(["DNp09", "DNa01", "DNa02", "DNb05", "DNextra_forward"]),
        pre=np.arange(20),
        post=np.roll(np.arange(20), 1),
        synapse_count=np.ones(20, dtype=np.float32),
        sign=np.ones(20, dtype=np.int8),
        side=np.tile([-1, -1, 1, 0], 5),
        group_mask=np.repeat([16, 8, 8, 2, 16], 4),
    )
    readout = MotorReadout(graph, SensorContract(), mode, branches, turn_bias=True)
    if mode == "structured":
        if extra:
            readout.set_extra_turns(["DNb05"], [0.2], [1.3])
        with torch.no_grad():
            readout.turn_weight.copy_(torch.linspace(0.2, 0.4, 2 * branches))
            readout.turn_bias.fill_(0.13)
            readout.forward_bias.fill_(-0.3)
            if extra:
                readout.extra_turn_weight.fill_(0.7)
    return readout


class SyntheticPolicy(torch.nn.Module):
    def __init__(self, branches=1, **kwargs):
        super().__init__()
        self.readout = synthetic_readout(branches, **kwargs)
        self.inputs = [
            torch.arange(20, dtype=torch.float32).reshape(1, -1) / (b + 3) + b + 0.1
            for b in range(branches)
        ]
        self.actions = []

    def initial_state(self, batch, device):
        return 0

    def forward(self, observations, state):
        action = self.readout(self.inputs)
        self.actions.append(action.clone())
        # Deliberately return branch-averaged activity, as the real policy does.
        return action, state + 1, torch.stack(self.inputs).mean(0)

    def convergence_current_contributions(self, activity):
        return {name: torch.tensor([float(i + 1)]) for i, name in enumerate(trace.MASS_NAMES)}

    def set_lesion(self, lesion):
        pass


class SyntheticEnv:
    def __init__(self, fail=False):
        self.unwrapped = SimpleNamespace(
            walls=np.array([-5.0, -5.0, 5.0, 5.0]),
            obstacles=[{"x": 2.0, "y": 0.0, "radius": 0.3}],
            agent_radius_m=0.25,
            target_radius_m=0.5,
        )
        self.fail = fail

    def info(self):
        t = self.t
        return {
            "time_s": float(t),
            "agent_pose": [float(t), 0, 0],
            "target_pose": [4.0, 0, 0],
            "teacher_action": [10.0 + t, 20.0 + t],
            "target_distance_m": 4.0 - t,
            "target_bearing_rad": 0.1 * t,
            "target_visible": t != 0,
            "occluded": t == 0,
            "applied_action": [0.5, 0.25],
            "collision": t == 2,
        }

    def reset(self, seed):
        self.t = 0
        return {"odor": np.zeros(2, np.float32)}, self.info()

    def step(self, action):
        if self.fail:
            raise RuntimeError("synthetic step failure")
        self.t += 1
        return {"odor": np.zeros(2, np.float32)}, 0, self.t == 2, False, self.info()

    def close(self):
        pass


def replay(tmp_path, policy, record_dn=False, variant="VO-late", lesion="none", fail=False):
    args = SimpleNamespace(
        device="cpu", output=tmp_path, scenario="occlusion", lesion=lesion, record_dn=record_dn
    )
    path = tmp_path / "trace.jsonl.gz"
    row = trace.replay_episode(
        policy,
        SimpleNamespace(control_dt=1),
        {"variant": variant},
        {},
        SyntheticEnv(fail),
        args,
        0,
        0,
        path,
    )
    with gzip.open(path, "rt") as stream:
        lines = stream.readlines()
    return row, [json.loads(line) for line in lines], lines


def test_disabled_record_keys_and_fixed_json_bytes(tmp_path):
    class LegacyPolicy:
        @property
        def readout(self):
            raise AssertionError("disabled recording must not inspect or hook readout")

        def initial_state(self, batch, device):
            return None

        def __call__(self, observations, state):
            return torch.tensor([[1.0, 0.5]]), None, {}

        convergence_current_contributions = SyntheticPolicy.convergence_current_contributions

    row, records, lines = replay(tmp_path, LegacyPolicy())
    expected = (
        '{"step": 0, "time_s": 0.0, "agent_pose": [0.0, 0.0, 0.0], '
        '"target_pose": [4.0, 0.0, 0.0], "teacher_action": [10.0, 20.0], '
        '"target_distance_m": 4.0, "target_bearing_rad": 0.0, '
        '"target_visible": false, "occluded": true, "action": [1.0, 0.5], '
        '"applied_action": [0.5, 0.25], "collision": false, "post_time_s": 1.0, '
        '"post_agent_pose": [1.0, 0.0, 0.0], "post_target_pose": [4.0, 0.0, 0.0], '
        '"current_mass": {"visual_exclusive": 1.0, "odor_exclusive": 2.0, "shared": 3.0}}\n'
    )
    assert lines[0].encode() == expected.encode()
    assert all(record.keys() == json.loads(expected).keys() for record in records)
    assert "world" not in row


@pytest.mark.parametrize("branches,extra", [(1, True), (2, True), (1, False)])
def test_dn_actual_branch_pools_latents_and_bitwise_actions(tmp_path, branches, extra):
    policy = SyntheticPolicy(branches, extra=extra)
    variant = "VO-late" if extra else "V"
    inputs = [a.clone() for a in policy.inputs]
    with torch.no_grad():
        baseline = policy.readout(policy.inputs)
    assert not policy.readout._forward_hooks
    row, records, _ = replay(tmp_path, policy, record_dn=True, variant=variant)
    assert not policy.readout._forward_hooks
    assert all(torch.equal(baseline, action) for action in policy.actions)
    assert all(torch.equal(a, b) for a, b in zip(inputs, policy.inputs, strict=True))
    assert row["world"]["walls"] == [-5, -5, 5, 5]
    assert row["world"]["obstacles"][0]["radius"] == 0.3
    for record in records:
        dn = record["dn"]
        assert [b["name"] for b in dn["branches"]] == (
            ["central"] if branches == 1 else ["visual", "odor"]
        )
        for i, branch in enumerate(dn["branches"]):
            graph = policy.readout._graph
            assert set(branch["types"]) == set(graph.type_names)
            for j, kind in enumerate(graph.type_names):
                values = policy.inputs[i][0, j * 4 : (j + 1) * 4]
                rates = branch["types"][kind]
                assert rates["in_readout"] is (extra or kind != "DNb05")
                assert {side: rates[side] for side in ("left", "right", "all")} == pytest.approx(
                    {
                        "left": values[:2].mean().item(),
                        "right": values[2].item(),
                        "all": values.mean().item(),
                    }
                )
        action = torch.stack(
            (
                policy.readout.v_max * torch.sigmoid(torch.tensor(dn["forward_latent"])),
                policy.readout.omega_max * torch.tanh(torch.tensor(dn["omega_latent"])),
            )
        )
        torch.testing.assert_close(action, baseline[0], rtol=1e-6, atol=0)
    with trace.DNRecorder(policy, {"variant": variant}, "none") as recorder:
        assert recorder.required_types_not_in_readout == ([] if extra else ["DNb05"])
        policy.readout(policy.inputs)
        recorder.pop()
        assert recorder.pending is None
    assert not policy.readout._forward_hooks


def test_hook_removed_on_replay_failure(tmp_path):
    policy = SyntheticPolicy()
    with pytest.raises(RuntimeError, match="synthetic step failure"):
        replay(tmp_path, policy, record_dn=True, fail=True)
    assert not policy.readout._forward_hooks


@pytest.mark.parametrize(
    "variant,mode,extra,lesion,match",
    [
        ("GRU", "structured", True, "none", "GRU/linear"),
        ("V", "linear", False, "none", "GRU/linear"),
        ("VO-late", "structured", True, "vision", "lesion none"),
    ],
)
def test_record_dn_rejects_unsupported(tmp_path, variant, mode, extra, lesion, match):
    policy = SyntheticPolicy(mode=mode, extra=extra)
    with pytest.raises(ValueError, match=match):
        replay(tmp_path, policy, record_dn=True, variant=variant, lesion=lesion)
    assert not policy.readout._forward_hooks
    assert not (tmp_path / "trace.jsonl.gz").exists()


@pytest.mark.parametrize("missing", ["type", "left", "right"])
def test_record_dn_rejects_required_type_or_side_missing_from_graph(tmp_path, missing):
    policy = SyntheticPolicy(extra=False)
    graph = policy.readout._graph
    nodes = graph.type_index == 3
    if missing == "type":
        # A stale type_names entry alone does not establish that nodes exist.
        graph.type_index[nodes] = 4
        match = "missing required DN types in graph: \\['DNb05'\\]"
    else:
        graph.side[nodes & (graph.side == (-1 if missing == "left" else 1))] = 0
        match = f"missing required DN sides:.*DNb05.*{missing}"
    with pytest.raises(ValueError, match=match) as error:
        replay(tmp_path, policy, record_dn=True, variant="V")
    actual_types = sorted(set(map(str, graph.type_names[graph.type_index])))
    assert f"graph types: {actual_types}" in str(error.value)
    assert not policy.readout._forward_hooks
    assert not (tmp_path / "trace.jsonl.gz").exists()


@pytest.mark.parametrize("enabled,extra", [(False, True), (True, True), (True, False)])
def test_replay_cli_opt_in_notes_and_world(tmp_path, monkeypatch, enabled, extra):
    import flytrack_core.model
    import flytrack_train.config

    policy = SyntheticPolicy(2 if extra else 1, extra=extra)
    variant = "VO-late" if extra else "V"
    contract = SensorContract()
    artifact = tmp_path / "actor"
    artifact.mkdir()
    (artifact / "model_card.json").write_text("{}")
    config = tmp_path / "config.yaml"
    config.write_text("{}")
    output = tmp_path / "out"
    monkeypatch.setattr(
        flytrack_core.model,
        "load_policy_artifact",
        lambda *a, **k: (
            policy,
            contract,
            {"variant": variant, "metadata": {"seed": 0}},
        ),
    )
    monkeypatch.setattr(flytrack_train.config, "contract_from_config", lambda c: contract)
    monkeypatch.setattr(flytrack_train.config, "make_env", lambda *a, **k: SyntheticEnv())
    monkeypatch.setattr(
        trace,
        "provenance",
        lambda paths: {"inputs_sha256": {str(p): trace.sha256(p) for p in paths}},
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "episode_trace",
            "--artifact",
            str(artifact),
            "--config",
            str(config),
            "--scenario",
            "occlusion",
            "--episodes",
            "0",
            "--output",
            str(output),
            *(["--record-dn"] if enabled else []),
        ],
    )
    trace.main()
    notes = demo.read_json(output / "analysis_notes.json")
    assert ("record_dn" in notes) == enabled
    assert ("dn_definition" in notes) == enabled
    assert ("required_dn_types_not_in_readout" in notes) == enabled
    if enabled:
        assert "in_readout" in notes["dn_definition"]
        assert notes["required_dn_types_not_in_readout"] == ([] if extra else ["DNb05"])
    assert ("world" in demo.read_json(output / "episodes.json")[0]) == enabled
    assert demo.card_sha(notes) == trace.sha256(artifact / "model_card.json")
    assert not policy.readout._forward_hooks


def episode(ep=0, kind="success", **extra):
    return {
        "variant": "VO-late",
        "scenario": "occlusion",
        "seed": 0,
        "episode": ep,
        "environment_seed": 1700000000 + ep,
        "success": kind == "success",
        "collision_rate": float(kind == "collision"),
        "within_distance_fraction": 0.79 if kind == "distance" else 0.9,
        "mean_distance_m": 1.5,
        "model_card_sha256": "card-sha",
        **extra,
    }


def selection(tmp_path, rows, variants=None, scenarios=None):
    f3 = tmp_path / "f3.json"
    f3.write_text(json.dumps(rows))
    manifest = demo.select(f3, variants or ["VO-late"], 0, scenarios or ["occlusion"], tmp_path)
    return manifest, tmp_path / "selection/manifest.json"


def test_selection_classes_ties_minimum_and_relative_paths(tmp_path):
    rows = [
        episode(9),
        episode(2),
        episode(7, "collision"),
        episode(3, "collision"),
        episode(0, "distance"),
        episode(4, "distance"),
        episode(1, "bearing"),
    ]
    manifest, path = selection(tmp_path, list(reversed(rows)))
    cell = manifest["cells"][0]
    assert cell["counts"] == {"success": 2, "collision": 2, "distance": 2, "bearing": 1}
    assert cell["success"]["episode"] == 2
    assert cell["failure"]["episode"] == 3
    assert cell["failure"]["failure_class"] == "collision"
    assert "collision > distance > bearing" in manifest["selection_rule"]
    assert manifest["input"]["sha256"] == trace.sha256(tmp_path / "f3.json")
    assert not Path(manifest["input"]["path"]).is_absolute()
    assert str(tmp_path) not in path.read_text()
    assert (path.parent / "manifest.csv").read_text().count("\n") == 3
    for row in rows:
        assert demo.category(row) == summary.failure_category(row)
    assert demo.DISTANCE_FRACTION_THRESHOLD == 0.8
    assert demo.category(episode(kind="distance", within_distance_fraction=0.8)) == "bearing"
    assert demo.category(episode(collision_rate=1)) == "success"


@pytest.mark.parametrize(
    "rows,success,failure",
    [
        ([episode()], True, False),
        ([episode(kind="distance")], False, True),
        ([], False, False),
    ],
)
def test_selection_null_slots_and_empty_cells(tmp_path, rows, success, failure):
    manifest, _ = selection(tmp_path, rows)
    cell = manifest["cells"][0]
    for role, exists in (("success", success), ("failure", failure)):
        assert (cell[role] is not None) == exists
        if not exists:
            assert cell[f"{role}_reason"] == "none"


def test_selection_distance_bearing_tie_and_dominant_bearing(tmp_path):
    rows = [episode(8, "distance"), episode(2, "bearing")]
    manifest, _ = selection(tmp_path, rows)
    assert manifest["cells"][0]["failure"]["episode"] == 8
    manifest, _ = selection(tmp_path, [*rows, episode(5, "bearing")])
    assert manifest["cells"][0]["failure"]["episode"] == 2


def test_selection_rejects_multiple_cards_and_duplicates(tmp_path):
    with pytest.raises(ValueError, match="model_card_sha256"):
        selection(tmp_path, [episode(), episode(1, model_card_sha256="other")])
    with pytest.raises(ValueError, match="Duplicate episode"):
        selection(tmp_path, [episode(), episode()])


@pytest.mark.parametrize("jobs", [None, 2])
def test_replay_cli_calls_once_per_nonempty_cell_with_logs(tmp_path, monkeypatch, jobs):
    _, manifest_path = selection(
        tmp_path,
        [
            episode(5, variant="V", scenario="normal"),
            episode(2, "collision", variant="V", scenario="normal"),
            episode(7),
            episode(3, "distance"),
        ],
        variants=["V", "VO-late"],
        scenarios=["normal", "occlusion"],
    )
    output = tmp_path / "out"
    actors = {"V": tmp_path / "actor=v", "VO-late": tmp_path / "actor-vo"}
    config = Path("configs/train/evaluate.yaml")
    calls = []
    barrier = Barrier(2) if jobs == 2 else None

    def fake_run(command, *, stdin, stdout, stderr, check):
        assert command[:2] == [sys.executable, str(DIAGNOSTICS / "episode_trace.py")]
        assert stdin == demo.subprocess.DEVNULL
        assert stderr == demo.subprocess.STDOUT
        assert check is True
        assert command.count("--record-dn") == 1
        args = command[2:]
        args.remove("--record-dn")
        options = dict(zip(args[::2], args[1::2], strict=True))
        calls.append(options)
        if barrier is not None:
            barrier.wait(timeout=10)
        directory = Path(options["--output"])
        directory.mkdir()
        (directory / "episodes.json").write_text(
            json.dumps([{"episode": int(ep)} for ep in options["--episodes"].split(",")])
        )
        stdout.write(b"fake stdout and stderr\n")

    monkeypatch.setattr(demo.subprocess, "run", fake_run)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "demo_2d",
            "replay",
            "--manifest",
            str(manifest_path),
            "--artifact",
            f"V={actors['V']}",
            "--artifact",
            f"VO-late={actors['VO-late']}",
            "--config",
            str(config),
            "-o",
            str(output),
            *([] if jobs is None else ["--jobs", str(jobs)]),
        ],
    )
    demo.main()
    expected = [("V", "normal", "2,5"), ("VO-late", "occlusion", "3,7")]
    assert len(calls) == len(expected)
    for variant, scenario, episodes in expected:
        directory = output / "replay" / f"{variant}_{scenario}"
        assert {
            "--artifact": str(actors[variant]),
            "--config": str(config),
            "--scenario": scenario,
            "--lesion": "none",
            "--device": "cpu",
            "--episodes": episodes,
            "--output": str(directory),
        } in calls
        assert directory.with_suffix(".log").read_text() == "fake stdout and stderr\n"
    assert not (output / "replay/V_occlusion").exists()
    assert not (output / "replay/VO-late_normal.log").exists()

    # Identical episode sets are complete, independently of row order.
    def unexpected_run(*args, **kwargs):
        pytest.fail("complete cells must not run again")

    monkeypatch.setattr(demo.subprocess, "run", unexpected_run)
    demo.main()
    assert not list((output / "replay").glob("*.partial_*"))


@pytest.mark.parametrize("previous", [None, "[", "{}", "[{}]", '[{"episode": 1}]', "[]"])
def test_replay_preserves_incomplete_directory_before_rerunning(tmp_path, monkeypatch, previous):
    _, path = selection(tmp_path, [episode()])
    directory = tmp_path / "replay/VO-late_occlusion"
    directory.mkdir(parents=True)
    (directory / "keep.txt").write_text("partial trace")
    if previous is not None:
        (directory / "episodes.json").write_text(previous)
    calls = []

    def fake_run(command, **kwargs):
        calls.append(command)
        assert not directory.exists()
        backups = list(directory.parent.glob("VO-late_occlusion.partial_*"))
        assert len(backups) == 1
        assert (backups[0] / "keep.txt").read_text() == "partial trace"
        if previous is not None:
            assert (backups[0] / "episodes.json").read_text() == previous
        directory.mkdir()
        (directory / "episodes.json").write_text('[{"episode": 0}]')

    monkeypatch.setattr(demo.subprocess, "run", fake_run)
    demo.replay(path, {"VO-late": Path("actor")}, Path("config.yaml"), tmp_path)
    assert len(calls) == 1
    assert not (directory / "keep.txt").exists()
    assert demo.replay_complete(directory, {0})


def test_replay_missing_artifact_even_for_empty_cell_fails_before_start(tmp_path, monkeypatch):
    _, path = selection(tmp_path, [episode()], variants=["VO-late", "V"])

    def unexpected_run(*args, **kwargs):
        pytest.fail("missing artifacts must be validated before starting any cell")

    monkeypatch.setattr(demo.subprocess, "run", unexpected_run)
    with pytest.raises(ValueError, match="Missing --artifact.*V"):
        demo.replay(path, {"VO-late": Path("actor")}, Path("config.yaml"), tmp_path)
    assert not (tmp_path / "replay").exists()


def test_replay_propagates_subprocess_failure_and_keeps_log(tmp_path, monkeypatch):
    _, path = selection(tmp_path, [episode()])

    def fake_run(command, **kwargs):
        kwargs["stdout"].write(b"synthetic replay failure\n")
        raise demo.subprocess.CalledProcessError(1, command)

    monkeypatch.setattr(demo.subprocess, "run", fake_run)
    with pytest.raises(demo.subprocess.CalledProcessError):
        demo.replay(path, {"VO-late": Path("actor")}, Path("config.yaml"), tmp_path)
    assert (tmp_path / "replay/VO-late_occlusion.log").read_text() == "synthetic replay failure\n"


@pytest.mark.parametrize("jobs", [0, -1])
def test_replay_rejects_nonpositive_jobs(tmp_path, jobs):
    with pytest.raises(ValueError, match="--jobs must be at least 1"):
        demo.replay(tmp_path / "missing.json", {}, Path("config.yaml"), tmp_path, jobs)


@pytest.mark.parametrize("argument", ["V", "=actor", "V="])
def test_replay_rejects_invalid_artifact_argument(argument):
    with pytest.raises(demo.argparse.ArgumentTypeError, match="VARIANT=directory"):
        demo.artifact_argument(argument)


def write_replay(directory, row, card="card-sha"):
    directory.mkdir(exist_ok=True)
    (directory / "episodes.json").write_text(json.dumps([row]))
    (directory / "analysis_notes.json").write_text(
        json.dumps(
            {
                "provenance": {"inputs_sha256": {"relative/actor/model_card.json": card}},
            }
        )
    )


@pytest.mark.parametrize(
    "field,value",
    [
        ("environment_seed", 123),
        ("success", False),
        ("collision_rate", 0.1),
        ("model_card_sha256", "other-card"),
    ],
)
def test_verify_rejects_each_mismatch(tmp_path, field, value):
    row = episode()
    _, path = selection(tmp_path, [row])
    replay_row = {**row, field: value}
    directory = tmp_path / "replay"
    write_replay(directory, replay_row, value if field == "model_card_sha256" else "card-sha")
    with pytest.raises(
        ValueError,
        match="Replay mismatch.*" + ("collision" if field == "collision_rate" else field),
    ):
        demo.verify(path, [directory])
    assert not (path.parent / "verify.json").exists()


def test_verify_deltas_missing_duplicates_and_card_provenance(tmp_path):
    row = episode()
    manifest, path = selection(tmp_path, [row])
    directory = tmp_path / "replay"
    write_replay(directory, {**row, "within_distance_fraction": 0.7, "mean_distance_m": 3})
    checks = demo.verify(path, [directory])
    assert checks[0]["within_distance_fraction_delta"] == pytest.approx(-0.2)
    assert checks[0]["mean_distance_m_delta"] == 1.5
    assert demo.read_json(path.parent / "verify.json")["matched"]
    with pytest.raises(ValueError, match="Missing replay"):
        demo.verify_cases(manifest, {})
    with pytest.raises(ValueError, match="Duplicate replay"):
        demo.replay_index([directory, directory])
    with pytest.raises(ValueError, match="exactly one"):
        demo.card_sha({"provenance": {"inputs_sha256": {}}})
    assert demo.card_sha({"model_card_sha256": "sha"}) == "sha"


@pytest.fixture(params=[True, False], ids=["all-in-readout", "dn-not-in-readout"])
def plot_inputs(tmp_path, request):
    policy = SyntheticPolicy(2, extra=request.param)
    row, records, _ = replay(tmp_path, policy, record_dn=True)
    f3 = episode(
        kind="collision",
        **{
            k: row[k]
            for k in (
                "success",
                "collision_rate",
                "within_distance_fraction",
                "mean_distance_m",
            )
        },
    )
    _, manifest_path = selection(tmp_path, [f3], variants=["VO-late", "V"])
    directory = tmp_path / "replay"
    write_replay(directory, row)
    (directory / row["trace"]).write_bytes((tmp_path / row["trace"]).read_bytes())
    return manifest_path, directory, records


def test_plot_png_gif_overview_and_index_smoke(tmp_path, plot_inputs, monkeypatch):
    path, directory, records = plot_inputs
    output = tmp_path / "plots"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "demo_2d",
            "plot",
            "--manifest",
            str(path),
            "--replay",
            str(directory),
            "-o",
            str(output),
            "--animate",
            "--dpi",
            "45",
        ],
    )
    demo.main()
    png = output / "figures/VO-late_occlusion_failure_ep0.png"
    overview = output / "figures/overview_trajectories.png"
    gif = output / "animations/VO-late_occlusion_failure_ep0.gif"
    for image_path in (png, overview, gif):
        assert image_path.stat().st_size > 0
        with Image.open(image_path) as image:
            assert min(image.size) > 0
            if image_path.suffix == ".gif":
                assert 1 < image.n_frames <= 300
            image.verify()
    index = demo.read_json(output / "figures/index.json")
    assert len(index["figures"]) == 3
    entry = index["figures"][0]
    assert entry["episode"]["role"] == "failure"
    assert entry["source_trace"]["sha256"] == trace.sha256(directory / "trace.jsonl.gz")
    assert not Path(entry["source_trace"]["path"]).is_absolute()
    assert "agg" == sys.modules["matplotlib"].get_backend().lower()
    import matplotlib.pyplot as plt

    fig = plt.figure()
    layout = fig.add_gridspec(6, 2)
    axes = []
    for k in range(6):
        axes.append(fig.add_subplot(layout[k, 1], sharex=axes[0] if axes else None))
    try:
        demo.time_panels(axes, records)
        assert axes[2].lines[0].get_ydata().tolist() == [0.5, 0.5]
        assert axes[3].lines[0].get_ydata().tolist() == [0.25, 0.25]
        assert all(len(ax.patches) == 1 for ax in axes)
        assert len(axes[4].lines) == len(axes[5].lines) == 8
        assert not np.array_equal(axes[4].lines[0].get_ydata(), axes[5].lines[0].get_ydata())
        for ax, branch in zip(axes[4:], records[0]["dn"]["branches"], strict=True):
            assert ax.get_title() == f"DN rates — {branch['name']} branch"
            in_readout = branch["types"]["DNb05"]["in_readout"]
            for line, side in zip(ax.lines[-2:], ("left", "right"), strict=True):
                suffix = "" if in_readout else " (not in readout)"
                assert line.get_label() == f"DNb05 {side}{suffix}"
                assert line.get_color() == ("C3" if in_readout else "0.5")
                assert line.get_linewidth() == (1.5 if in_readout else 1.0)
            assert ax.lines[-2].get_linestyle() == "-"
            assert ax.lines[-1].get_linestyle() == "--"
        assert all(not ax.get_xticklabels() and not ax.get_xlabel() for ax in axes[:-1])
        assert axes[-1].get_xticklabels()
        assert axes[-1].get_xlabel() == "decision time (s)"
        assert [ax.get_ylabel() for ax in axes] == [
            "distance (m)",
            "bearing (deg)",
            "v (m/s)",
            "omega (rad/s)",
            "DN rate",
            "DN rate",
        ]
        assert all(ax.get_yticklabels() for ax in axes)
        assert all(axes[0].get_shared_x_axes().joined(axes[0], ax) for ax in axes[1:])
        assert axes[1].get_ylim() == (-180, 180)
        assert axes[1].get_yticks().tolist() == [-180, -90, 0, 90, 180]
    finally:
        plt.close(fig)
    fig, ax = plt.subplots()
    try:
        row = demo.read_json(directory / "episodes.json")[0]
        demo.trajectory(ax, records, row)
        assert ax.collections[-1].get_offsets().tolist() == [records[-1]["post_agent_pose"][:2]]
        assert any(text.get_text() == "collision: obstacle" for text in ax.texts)
        # The first orange segment extends through the post-step pose, while
        # the next segment follows the visible decision-time flag.
        assert ax.collections[0].get_segments()[0].tolist() == [[0, 0], [1, 0]]
        from matplotlib.colors import to_rgba

        assert ax.collections[0].get_colors().tolist() == [
            list(to_rgba("tab:orange")),
            list(to_rgba("tab:blue")),
        ]
    finally:
        plt.close(fig)


def test_animation_downsamples_and_caps_frames():
    records = [{"post_time_s": i / 100} for i in range(6001)]
    frames = demo.animation_frames(records)
    assert len(frames) == 300
    assert frames[0] == 0 and frames[-1] == 6000
    short = demo.animation_frames(records[:501])
    assert 49 <= len(short) <= 52


@pytest.mark.parametrize(
    "degrees,expected",
    [
        ([170, -170], [170, np.nan, -170]),
        ([-170, 170], [-170, np.nan, 170]),
        ([170, -170, 175], [170, np.nan, -170, np.nan, 175]),
        ([90, -90], [90, -90]),
    ],
)
def test_bearing_line_breaks_wrap_jumps_without_changing_data(degrees, expected):
    values = np.array(degrees, dtype=float)
    time = np.arange(len(values), dtype=float)
    x, y = demo.bearing_line(time, values)
    np.testing.assert_equal(y, expected)
    np.testing.assert_equal(x[np.isfinite(y)], time)
    np.testing.assert_equal(np.isnan(x), np.isnan(y))
    np.testing.assert_equal(values, degrees)
    np.testing.assert_equal(time, np.arange(len(values)))


def test_select_and_verify_cli(tmp_path, monkeypatch):
    row = episode()
    f3 = tmp_path / "f3.json"
    f3.write_text(json.dumps([row]))
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "demo_2d",
            "select",
            "--f3",
            str(f3),
            "--variants",
            "VO-late",
            "--seed",
            "0",
            "--scenarios",
            "occlusion",
            "-o",
            str(tmp_path),
        ],
    )
    demo.main()
    directory = tmp_path / "replay"
    write_replay(directory, row)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "demo_2d",
            "verify",
            "--manifest",
            str(tmp_path / "selection/manifest.json"),
            "--replay",
            str(directory),
        ],
    )
    demo.main()
    assert (tmp_path / "selection/verify.json").is_file()


def test_plot_rejects_missing_dn_instead_of_filling(tmp_path, plot_inputs):
    path, directory, records = plot_inputs
    records = copy.deepcopy(records)
    del records[0]["dn"]["branches"][0]["types"]["DNb05"]
    with gzip.open(directory / "trace.jsonl.gz", "wt") as stream:
        stream.writelines(json.dumps(row) + "\n" for row in records)
    with pytest.raises(ValueError, match="Missing left/right DN rates for DNb05"):
        demo.plot(path, [directory], tmp_path / "plots", dpi=40)
