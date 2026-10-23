"""C11 completion checks with the configured recovery seed."""

import math

from c11_common import read
from protocol100 import finite


def ppo_complete(path, config, smoke):
    # Keep check.ppo_complete's checks; its fixed seed 0 cannot validate C11-s1.
    import torch
    from flytrack_core.model import load_policy_artifact
    from flytrack_train.recovery import Recovery

    checkpoint = Recovery(
        path, {**config, "resume": True}, "ppo", config["seed"], "cpu" if smoke else "cuda"
    ).checkpoint
    assert checkpoint["environment_steps"] == config["train"]["total_steps"]
    assert checkpoint["iteration"] == math.ceil(
        config["train"]["total_steps"]
        / (config["train"]["rollout_steps"] * config["train"]["num_envs"])
    )
    progress = read(path / "progress.json")
    finite(progress)
    assert progress["phase"] == "complete"
    reports = read(path / "training_metrics.json")
    finite(reports)
    assert len(reports) == checkpoint["iteration"]
    assert reports == checkpoint["reports"]
    assert reports[-1]["environment_steps"] == checkpoint["environment_steps"]
    for section in ("actor", "critic", "optimizer"):

        def tensors(value, section=section):
            if isinstance(value, torch.Tensor):
                assert bool(torch.isfinite(value).all()), f"Nonfinite {section}"
            elif isinstance(value, dict):
                for item in value.values():
                    tensors(item)
            elif isinstance(value, (list, tuple)):
                for item in value:
                    tensors(item)
            else:
                finite(value)

        tensors(checkpoint[section])
    actor, _, _ = load_policy_artifact(path / "actor", device="cpu")
    for name, value in actor.state_dict().items():
        assert torch.equal(value, checkpoint["actor"][name]), "Actor differs from final checkpoint"
