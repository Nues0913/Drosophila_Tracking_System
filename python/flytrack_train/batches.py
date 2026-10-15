"""Recurrent sequence storage and explicit reset semantics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch


def tensor_observation(
    observations: list[dict[str, np.ndarray]], device: torch.device
) -> dict[str, torch.Tensor]:
    # Whitelist actor inputs. In particular, never merge environment info into observations.
    return {
        key: torch.as_tensor(
            np.stack([obs[key] for obs in observations]), device=device, dtype=torch.float32
        )
        for key in ("vision", "odor", "valid")
    }


def reset_state(
    state: torch.Tensor, episode_start: torch.Tensor, initial_state: torch.Tensor | None = None
) -> torch.Tensor:
    # ORN adaptation thresholds and resting voltages may be nonzero. Reset to the
    # model-defined state rather than multiplying the entire memory by zero.
    initial = torch.zeros_like(state) if initial_state is None else initial_state
    return torch.where(episode_start.bool().unsqueeze(-1), initial, state)


@dataclass
class Episode:
    observations: dict[str, np.ndarray]
    labels: dict[str, np.ndarray]

    def __len__(self) -> int:
        return len(self.observations["odor"])


def pack_episode(
    observations: list[dict[str, np.ndarray]], labels: list[dict[str, Any]]
) -> Episode:
    return Episode(
        {
            k: np.stack([o[k] for o in observations]).astype(np.float32)
            for k in ("vision", "odor", "valid")
        },
        {k: np.stack([v[k] for v in labels]).astype(np.float32) for k in labels[0]},
    )


def sequence_windows(episodes: list[Episode], length: int, rng: np.random.Generator):
    # Shuffle independent episodes, preserve chronology inside each episode.
    # This carries full recurrent memory with O(T), rather than replaying every
    # prefix for randomly shuffled windows (O(T^2 / sequence_length)).
    order = np.arange(len(episodes))
    rng.shuffle(order)
    windows = [
        (int(i), start, min(start + length, len(episodes[i])))
        for i in order
        for start in range(0, len(episodes[i]), length)
    ]
    return windows
