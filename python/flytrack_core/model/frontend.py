"""Frozen, torch-native flyvis rate network; no flyvis dependency at deployment."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import torch
from torch import Tensor, nn

from flytrack_core.contracts import SensorContract
from flytrack_core.sensing.hex_eye import axial_coordinates


class FrozenFlyvis(nn.Module):
    """Run one authentic exported network per eye and fixed spatial pooling.

    An export is accepted only with checkpoint provenance and an upstream-output
    parity record. This is an artifact contract, not a cryptographic attestation
    of authorship. No random, identity, or untrained fallback is available.
    """

    def __init__(self, path: str | Path | None, contract: SensorContract, visual_bins: int = 5):
        super().__init__()
        if path is None or not Path(path).is_file():
            raise FileNotFoundError(
                "A real exported flyvis frontend is required; run scripts/export_flyvis.py "
                "on an official pretrained checkpoint"
            )
        if visual_bins < 1:
            raise ValueError("visual_bins must be positive")
        with np.load(path, allow_pickle=False) as archive:
            data = {key: archive[key].copy() for key in archive.files}
        required = {
            "source_metadata",
            "schema_version",
            "bias",
            "tau",
            "pre",
            "post",
            "weight",
            "input_node",
            "input_ommatidium",
            "output_node",
            "output_type",
            "output_type_names",
            "eye_radius",
            "n_ommatidia",
            "node_u",
            "node_v",
        }
        if not required.issubset(data):
            raise ValueError(
                f"Incomplete official frontend export: missing {sorted(required - set(data))}"
            )
        self.metadata = json.loads(str(data["source_metadata"].item()))
        if (
            int(data["schema_version"].item()) != 1
            or self.metadata.get("source") != "TuragaLab/flyvis"
        ):
            raise ValueError("Unsupported frontend format or source")
        checksum = self.metadata.get("checkpoint_sha256", "")
        if len(checksum) != 64 or any(char not in "0123456789abcdef" for char in checksum):
            raise ValueError("Frontend checkpoint provenance is missing")
        parity = self.metadata.get("parity", {})
        if parity.get("passed") is not True or not math.isclose(
            float(parity.get("dt", -1)), contract.neural_dt
        ):
            raise ValueError(
                "Frontend requires successful upstream parity validation "
                "at the deployment neural_dt"
            )
        if not math.isfinite(float(parity.get("max_abs_error", math.inf))):
            raise ValueError("Frontend parity record is invalid")
        self.n_nodes = len(data["bias"])
        self.n_ommatidia = int(data["n_ommatidia"].item())
        self.output_types = data["output_type_names"].astype(str).tolist()
        if self.n_ommatidia != contract.n_ommatidia:
            raise ValueError("Frontend ommatidium count does not match sensor contract")
        if int(data["eye_radius"].item()) != contract.eye_radius:
            raise ValueError("Frontend hex radius does not match sensor contract")
        input_node = data["input_node"].astype(np.int64)
        input_ommatidium = data["input_ommatidium"].astype(np.int64)
        pre, post = data["pre"].astype(np.int64), data["post"].astype(np.int64)
        output_node = data["output_node"].astype(np.int64)
        output_type = data["output_type"].astype(np.int64)
        if (
            self.n_nodes < 1
            or not self.output_types
            or pre.shape != post.shape
            or pre.shape != data["weight"].shape
        ):
            raise ValueError("Malformed frontend graph")
        for indices in (pre, post, input_node, output_node):
            if indices.ndim != 1 or np.any(indices < 0) or np.any(indices >= self.n_nodes):
                raise ValueError("Invalid frontend node indices")
        if (
            input_node.shape != input_ommatidium.shape
            or np.any(input_ommatidium < 0)
            or np.any(input_ommatidium >= self.n_ommatidia)
        ):
            raise ValueError("Invalid frontend retina input map")
        coordinates = axial_coordinates(contract.eye_radius)
        if not np.array_equal(
            np.stack((data["node_u"][input_node], data["node_v"][input_node]), axis=1),
            coordinates[input_ommatidium],
        ):
            raise ValueError(
                "Frontend input map is not in the canonical sensor hex coordinate order"
            )
        if (
            output_node.shape != output_type.shape
            or np.any(output_type < 0)
            or np.any(output_type >= len(self.output_types))
        ):
            raise ValueError("Invalid frontend output groups")
        for name in ("bias", "tau", "node_u", "node_v"):
            if data[name].shape != (self.n_nodes,) or not np.isfinite(data[name]).all():
                raise ValueError(f"Invalid frontend {name}")
        if np.any(data["tau"] <= 0) or not np.isfinite(data["weight"]).all():
            raise ValueError("Frontend time constants and weights must be finite")
        self.register_buffer(
            "bias", torch.as_tensor(data["bias"], dtype=torch.float32), persistent=False
        )
        self.register_buffer(
            "tau", torch.as_tensor(data["tau"], dtype=torch.float32), persistent=False
        )
        self.register_buffer("input_node", torch.as_tensor(input_node), persistent=False)
        self.register_buffer(
            "input_ommatidium", torch.as_tensor(input_ommatidium), persistent=False
        )
        matrix = torch.sparse_coo_tensor(
            torch.as_tensor(np.stack((post, pre))),
            torch.as_tensor(data["weight"], dtype=torch.float32),
            (self.n_nodes, self.n_nodes),
        ).coalesce()
        self.register_buffer("weight", matrix.to_sparse_csr(), persistent=False)
        self.visual_bins = visual_bins
        self.features_per_eye = len(self.output_types) * visual_bins
        x = data["node_u"][output_node] + data["node_v"][output_node] / 2
        radius = max(1, contract.eye_radius)
        bins = np.clip(
            np.floor((x + radius) / (2 * radius + 1e-6) * visual_bins), 0, visual_bins - 1
        ).astype(np.int64)
        feature_index = output_type * visual_bins + bins
        counts = np.bincount(feature_index, minlength=self.features_per_eye)
        if np.any(counts == 0):
            raise ValueError(
                "Chosen flyvis output types have empty spatial bins; choose fewer visual_bins"
            )
        pooling = torch.sparse_coo_tensor(
            torch.as_tensor(np.stack((feature_index, output_node))),
            torch.as_tensor(1.0 / counts[feature_index], dtype=torch.float32),
            (self.features_per_eye, self.n_nodes),
        ).coalesce()
        self.register_buffer("pooling", pooling, persistent=False)
        self.control_dt, self.neural_dt = contract.control_dt, contract.neural_dt
        self.substeps = round(self.control_dt / self.neural_dt)
        if self.substeps < 1 or not math.isclose(self.substeps * self.neural_dt, self.control_dt):
            raise ValueError("control_dt must be an integer multiple of neural_dt")

    def initial_state(self, batch_size: int, device: torch.device | str) -> Tensor:
        return self.bias.to(device).expand(batch_size, 2, -1).clone().flatten(1)

    def forward(self, vision: Tensor, state: Tensor, valid: Tensor) -> tuple[Tensor, Tensor]:
        batch = len(vision)
        if vision.shape != (batch, 2, self.n_ommatidia) or state.shape != (batch, 2 * self.n_nodes):
            raise ValueError("Incorrect frontend observation/state shape")
        available = (valid > 0).reshape(batch * 2, 1)
        image = torch.where(available, vision.reshape(batch * 2, -1), 0.0)
        # The policy concatenates frozen sensory memory and trainable central
        # memory. Its slice may inherit requires_grad from that concatenation;
        # the retina has no dependency on any trainable actor parameter.
        # Integrate in node-major [N, eyes] layout: CSR spmm with a contiguous
        # dense operand is several times faster than with a transposed view.
        voltage = state.detach().reshape(batch * 2, self.n_nodes).T.contiguous()
        current = voltage.new_zeros(voltage.shape)
        current.index_add_(0, self.input_node, image[:, self.input_ommatidium].T)
        # Upstream flyvis clamps frozen tau at dt. Central trainable tau uses 2dt.
        ratio = (self.neural_dt / self.tau.clamp_min(self.neural_dt)).unsqueeze(1)
        bias = self.bias.unsqueeze(1)
        columns = available.reshape(1, batch * 2)
        for _ in range(self.substeps):
            recurrent = torch.sparse.mm(self.weight, torch.relu(voltage))
            evolved = voltage + ratio * (-voltage + bias + recurrent + current)
            voltage = torch.where(columns, evolved, voltage)
        features = torch.sparse.mm(self.pooling, torch.relu(voltage)).T
        features = torch.where(available, features, 0.0)
        return features.reshape(batch, 2, -1), voltage.T.reshape(batch, -1)
