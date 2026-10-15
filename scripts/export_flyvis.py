#!/usr/bin/env python3
"""Export authentic pretrained flyvis parameters without requiring flyvis in ROS.

Run in the Python 3.12 ML environment after obtaining official pretrained models.
Conversion is published only after both the expanded equations and the actual
deployment frontend match the upstream network on the same stimuli.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from importlib.metadata import version
from pathlib import Path


def checksum(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--network-dir",
        required=True,
        type=Path,
        help="Official flyvis results/flow/0000/000 model directory",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        help="Explicit official checkpoint; otherwise resolve the model's best checkpoint",
    )
    parser.add_argument("--sensor-contract", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--output-types",
        nargs="+",
        default=["T4a", "T4b", "T4c", "T4d", "T5a", "T5b", "T5c", "T5d"],
    )
    parser.add_argument("--parity-atol", type=float, default=2e-5)
    parser.add_argument("--parity-rtol", type=float, default=2e-5)
    parser.add_argument(
        "--visual-bins",
        type=int,
        default=5,
        help="Deployment pooling bins; checked against upstream responses",
    )
    args = parser.parse_args()
    if args.parity_atol <= 0 or args.parity_rtol < 0:
        parser.error("Parity tolerances must be positive/nonnegative")
    # Upstream selects a module-global CUDA generator at import time. This
    # converter deliberately runs on CPU; constrain only this subprocess before
    # importing torch/flyvis, without changing the host or parent environment.
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    # Importing flyvis otherwise searches and loads .env files from ancestors.
    # Public pretrained conversion requires no account or local credentials.
    os.environ["PYTHON_DOTENV_DISABLED"] = "1"
    # Optional training-only dependency; importing core never imports flyvis.
    import numpy as np
    import torch
    from flytrack_core.contracts import SensorContract
    from flytrack_core.model.frontend import FrozenFlyvis
    from flytrack_core.sensing.hex_eye import axial_coordinates
    from flyvis.network import Network, NetworkView
    from flyvis.network.dynamics import PPNeuronIGRSynapses

    contract = SensorContract.from_dict(
        json.loads(args.sensor_contract.read_text(encoding="utf-8"))
    )
    with torch.device("cpu"):
        view = NetworkView(args.network_dir)
        checkpoint_value = args.checkpoint or view.get_checkpoint("best")
        if checkpoint_value is None:
            raise FileNotFoundError("No pretrained checkpoint was found")
        checkpoint_path = Path(checkpoint_value)
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        if not isinstance(checkpoint, dict) or "network" not in checkpoint:
            raise ValueError("Official checkpoint must contain a network state_dict")
        config = view.dir.config.network.to_dict()
        network = Network(**config).to("cpu")
        network.load_state_dict(checkpoint["network"], strict=True)
        network.eval().clamp()
        if not isinstance(network.dynamics, PPNeuronIGRSynapses) or not isinstance(
            network.dynamics.activation, torch.nn.ReLU
        ):
            raise ValueError(
                "Only the official passive-rate/ReLU dynamics are supported; "
                "do not silently approximate another model"
            )
        graph = network.connectome

        def expanded(parameter, route: str):
            values = parameter.semantic_values.detach().cpu().numpy()
            indices = parameter.readers[route].detach().cpu().numpy()
            return values[indices].astype(np.float32)

        bias = expanded(network.node_params["bias"], "nodes")
        tau = expanded(network.node_params["time_const"], "nodes")
        pre = network._source_indices.detach().cpu().numpy().astype(np.int64)
        post = network._target_indices.detach().cpu().numpy().astype(np.int64)
        weight = (
            expanded(network.edge_params["sign"], "edges")
            * expanded(network.edge_params["syn_count"], "edges")
            * expanded(network.edge_params["syn_strength"], "edges")
        )
        u, v = np.asarray(graph.nodes.u[:]), np.asarray(graph.nodes.v[:])
        node_types = np.asarray(graph.nodes.type[:]).astype(str)
        coordinates = [
            tuple(map(int, coordinate)) for coordinate in axial_coordinates(contract.eye_radius)
        ]
        if len(coordinates) != contract.n_ommatidia:
            raise ValueError("Sensor hex radius and ommatidium count differ")
        coordinate_index = {coordinate: index for index, coordinate in enumerate(coordinates)}
        input_node = np.asarray(network.stimulus.input_index).reshape(-1).astype(np.int64)
        try:
            input_ommatidium = np.array(
                [coordinate_index[(int(u[node]), int(v[node]))] for node in input_node],
                dtype=np.int64,
            )
        except KeyError as error:
            raise ValueError(
                "Upstream retina radius differs from requested sensor contract"
            ) from error
        for indices in np.asarray(network.stimulus.input_index):
            if len(indices) != contract.n_ommatidia or set(
                zip(u[indices].tolist(), v[indices].tolist(), strict=True)
            ) != set(coordinates):
                raise ValueError(
                    "Every upstream input cell type must cover the complete requested hex lattice"
                )
        if len(set(args.output_types)) != len(args.output_types):
            raise ValueError("Output cell types must be unique")
        absent = set(args.output_types) - set(node_types.tolist())
        if absent:
            raise ValueError(f"Requested output cell types are absent: {sorted(absent)}")
        output_node = np.flatnonzero(np.isin(node_types, args.output_types)).astype(np.int64)
        output_type = np.array(
            [args.output_types.index(node_types[node]) for node in output_node], dtype=np.int64
        )

        # Compare independent expanded equations to the loaded upstream network.
        # Input values repeat for five substeps, exactly as in control-time runtime.
        substeps = round(contract.control_dt / contract.neural_dt)
        if substeps < 1 or abs(substeps * contract.neural_dt - contract.control_dt) > 1e-9:
            raise ValueError("Control period must be a multiple of neural period")
        count, frames = contract.n_ommatidia, 24
        phase = torch.arange(count, dtype=torch.float32) / max(count - 1, 1)
        stimuli = []
        for time in range(frames):
            grey = torch.full((count,), 0.5)
            impulse = torch.zeros(count)
            if time == 4:
                impulse[count // 2] = 1
            moving = (0.5 + 0.4 * torch.sin(2 * torch.pi * (phase - time / frames))).clamp(0, 1)
            stimuli.append(torch.stack((grey, impulse, moving)))
        control_images = torch.stack(stimuli, dim=1)
        images = control_images.repeat_interleave(substeps, dim=1)
        buffer = torch.zeros(3, images.shape[1], len(bias))
        # Export mapping explicitly reorders canonical (u,v) into upstream indices.
        buffer[:, :, torch.as_tensor(input_node)] = images[:, :, torch.as_tensor(input_ommatidium)]
        with torch.no_grad():
            reference = network(buffer, dt=contract.neural_dt)
            voltage = torch.as_tensor(bias).expand(3, -1).clone()
            matrix = torch.sparse_coo_tensor(
                torch.as_tensor(np.stack((post, pre))),
                torch.as_tensor(weight),
                (len(bias), len(bias)),
            ).coalesce()
            step = contract.neural_dt / torch.as_tensor(tau).clamp_min(contract.neural_dt)
            converted = []
            for frame in range(images.shape[1]):
                recurrent = torch.sparse.mm(matrix, torch.relu(voltage).T).T
                voltage = voltage + step * (
                    -voltage + torch.as_tensor(bias) + recurrent + buffer[:, frame]
                )
                converted.append(voltage)
            converted = torch.stack(converted, dim=1)
            max_abs_error = float((reference - converted).abs().max())
            passed = bool(
                torch.isfinite(reference).all()
                and torch.isfinite(converted).all()
                and torch.allclose(
                    reference, converted, atol=args.parity_atol, rtol=args.parity_rtol
                )
            )
        if not passed:
            raise ValueError(
                f"Expanded frontend did not match upstream outputs: max_abs_error={max_abs_error}"
            )
        metadata = {
            "source": "TuragaLab/flyvis",
            "package_version": version("flyvis"),
            "checkpoint_sha256": checksum(checkpoint_path),
            "checkpoint_name": checkpoint_path.name,
            "network_config": config,
            "connectome_file": str(config.get("connectome", {}).get("file", "")),
            "source_urls": [
                "https://github.com/TuragaLab/flyvis",
                "https://turagalab.github.io/flyvis/reference/network/",
                "https://raw.githubusercontent.com/TuragaLab/flyvis/v1.2.0/flyvis/network/dynamics.py",
                "https://raw.githubusercontent.com/TuragaLab/flyvis/v1.2.0/flyvis/network/stimulus.py",
            ],
            "input_coordinate_order": (
                "sensing.hex_eye.axial_coordinates: r(v) ascending, "
                "then q(u) ascending inside valid hex bounds"
            ),
            "parity": {
                "passed": passed,
                "dt": contract.neural_dt,
                "max_abs_error": max_abs_error,
                "atol": args.parity_atol,
                "rtol": args.parity_rtol,
                "stimuli": ["constant grey", "central flash", "moving sinusoidal pattern"],
                "n_neural_steps": int(images.shape[1]),
            },
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)

        def save_export(path: Path) -> None:
            # A file handle prevents numpy appending an extra .npz extension.
            with path.open("wb") as stream:
                np.savez_compressed(
                    stream,
                    schema_version=np.int64(1),
                    source_metadata=np.array(json.dumps(metadata, sort_keys=True)),
                    eye_radius=np.int64(contract.eye_radius),
                    n_ommatidia=np.int64(contract.n_ommatidia),
                    bias=bias,
                    tau=tau,
                    pre=pre,
                    post=post,
                    weight=weight,
                    node_u=u.astype(np.int32),
                    node_v=v.astype(np.int32),
                    input_node=input_node,
                    input_ommatidium=input_ommatidium,
                    output_node=output_node,
                    output_type=output_type,
                    output_type_names=np.asarray(args.output_types),
                )

        # Verify the CSR recurrence, state layout, binocular independence and
        # spatial pooling actually used during training/deployment. Keep failed
        # exports out of the destination, even when an older valid export exists.
        with tempfile.TemporaryDirectory(
            prefix=".flyvis-export-", dir=args.output.parent
        ) as temporary:
            staged = Path(temporary) / "frontend.npz"
            save_export(staged)
            frontend = FrozenFlyvis(staged, contract, visual_bins=args.visual_bins)
            state = frontend.initial_state(3, "cpu")
            state_error = feature_error = 0.0
            runtime_passed = True
            x = u[output_node] + v[output_node] / 2
            bins = np.clip(
                np.floor(
                    (x + contract.eye_radius) / (2 * contract.eye_radius + 1e-6) * args.visual_bins
                ),
                0,
                args.visual_bins - 1,
            ).astype(np.int64)
            groups = output_type * args.visual_bins + bins
            with torch.no_grad():
                for frame in range(frames):
                    image = control_images[:, frame]
                    vision = torch.stack((image, image.roll(1, dims=0)), dim=1)
                    features, state = frontend(vision, state, torch.ones(3, 2))
                    upstream = reference[:, (frame + 1) * substeps - 1]
                    expected_state = torch.stack((upstream, upstream.roll(1, dims=0)), dim=1)
                    expected_features = torch.stack(
                        [
                            torch.relu(expected_state[:, :, output_node[groups == group]]).mean(
                                dim=-1
                            )
                            for group in range(frontend.features_per_eye)
                        ],
                        dim=-1,
                    )
                    observed_state = state.reshape_as(expected_state)
                    state_error = max(
                        state_error, float((observed_state - expected_state).abs().max())
                    )
                    feature_error = max(
                        feature_error, float((features - expected_features).abs().max())
                    )
                    runtime_passed = runtime_passed and bool(
                        torch.allclose(
                            observed_state,
                            expected_state,
                            atol=args.parity_atol,
                            rtol=args.parity_rtol,
                        )
                        and torch.allclose(
                            features,
                            expected_features,
                            atol=args.parity_atol,
                            rtol=args.parity_rtol,
                        )
                    )
            if not runtime_passed:
                raise ValueError(
                    f"Deployment frontend did not match upstream: state error={state_error}, "
                    f"feature error={feature_error}"
                )
            metadata["runtime_parity"] = {
                "passed": True,
                "backend": "torch-cpu-sparse-csr",
                "visual_bins": args.visual_bins,
                "max_state_abs_error": state_error,
                "max_feature_abs_error": feature_error,
                "n_control_steps": frames,
                "binocular_independence": True,
            }
            save_export(staged)
            staged.replace(args.output)
        print(
            f"Exported authentic frozen frontend to {args.output}; "
            f"expanded error={max_abs_error:.3g}, "
            f"deployment state error={state_error:.3g}, pooled feature error={feature_error:.3g}"
        )


if __name__ == "__main__":
    main()
