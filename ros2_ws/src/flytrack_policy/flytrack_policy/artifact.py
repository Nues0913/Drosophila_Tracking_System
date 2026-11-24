"""Load a v1 or v2 policy artifact for the ROS policy node (#140).

The card's ``spec_version`` decides the loader: a schema 4 card (``spec_version: "v2"``) goes
through ``flytrack_v2.model.load_policy``, the single v2 loading entry (docs/v2/interfaces.md
section 7); a card without it goes through the v1 ``load_policy_artifact``. Both return a
policy that ``PolicyRuntime`` drives the same way: ``initial_state(1, device)`` and
``policy(obs, state) -> (action, next_state, activity)`` with a ``node_ids`` tensor matching
``activity``.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch

V2_SPEC = "v2"


def artifact_spec(artifact_dir: str | Path) -> str:
    """``"v2"`` for a schema 4 card, ``"v1"`` for a card without ``spec_version``."""
    card = json.loads((Path(artifact_dir) / "model_card.json").read_text(encoding="utf-8"))
    if not isinstance(card, dict):
        raise ValueError("model_card.json must be an object")
    spec = card.get("spec_version")
    if spec is None:
        return "v1"
    if spec != V2_SPEC:
        raise ValueError(f"Unsupported model card spec_version: {spec!r}")
    return V2_SPEC


class V2PolicyAdapter(torch.nn.Module):
    """Runs a v2 policy one control step at a time with the v1 runtime's call convention.

    The state is ``(sensory_state, central_state)``. The action is the distribution's mode
    (deterministic deployment). Until the world publishes the antenna reading, ``wind`` is
    zero and its valid flag 0, which is exactly what the v1compat world gives the policy
    (docs/v2/interfaces.md section 2).
    """

    def __init__(self, policy):
        super().__init__()
        self.policy = policy
        graph = getattr(policy, "graph", None)
        if graph is not None:
            node_ids = torch.as_tensor(graph.node_ids, dtype=torch.int64)
        else:  # GRU2: the activity is the hidden state, one id per unit
            node_ids = torch.arange(policy.central_state_size, dtype=torch.int64)
        self.register_buffer("node_ids", node_ids, persistent=False)

    def initial_state(self, batch_size: int, device):
        return (
            self.policy.initial_sensory_state(batch_size, device),
            self.policy.initial_central_state(batch_size, device),
        )

    def forward(self, obs: dict, state):
        sensory, central = state
        vision = obs["vision"]
        valid = obs["valid"]
        if valid.shape[-1] == 3:
            valid = torch.cat((valid, valid.new_zeros(*valid.shape[:-1], 1)), dim=-1)
        wind = obs.get("wind")
        if wind is None:
            wind = vision.new_zeros(*vision.shape[:-2], 2)
        observation = {"vision": vision, "odor": obs["odor"], "wind": wind, "valid": valid}
        code, sensory = self.policy.sense(observation, sensory)
        step = self.policy.forward_code(code, central)
        core = getattr(self.policy, "core", None)
        activity = core.rates(step.central_state) if core is not None else step.central_state
        return step.dist.mode(), (sensory, step.central_state), activity


def load_deployment_policy(artifact_dir: str | Path, device: str = "cpu"):
    """(policy, contract, card, spec) for the node; ``policy`` follows the runtime convention."""
    spec = artifact_spec(artifact_dir)
    if spec == V2_SPEC:
        from flytrack_v2.contracts import SensorContractV2
        from flytrack_v2.model import load_policy

        source = Path(artifact_dir)
        policy = load_policy(source, device=device)
        contract = SensorContractV2.from_json(source / "sensor_contract.json")
        card = json.loads((source / "model_card.json").read_text(encoding="utf-8"))
        return V2PolicyAdapter(policy).to(device), contract, card, spec
    from flytrack_core.model import load_policy_artifact

    policy, contract, card = load_policy_artifact(artifact_dir, device=device)
    return policy, contract, card, spec
