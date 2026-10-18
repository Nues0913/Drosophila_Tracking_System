"""Recurrent sequence storage and explicit reset semantics."""

from __future__ import annotations

import contextlib
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


class LaneSchedule:
    """Chronological TBPTT windows for several episodes advancing in parallel.

    Each lane walks one episode from its first step in consecutive windows and
    carries that episode's recurrent state; a finished lane takes the next
    episode of the shuffled order. One optimizer step covers one window per lane,
    so memory remains O(T) per episode as in single-sequence TBPTT.
    """

    def __init__(
        self,
        lengths: list[int],
        order: list[int],
        lanes: int,
        length: int,
        cursor: int = 0,
        positions: list[list[int] | None] | None = None,
    ):
        if lanes <= 0 or length <= 0:
            raise ValueError("Lane count and sequence length must be positive")
        self.lengths, self.order, self.lanes, self.length = lengths, list(order), lanes, length
        self.cursor = cursor
        self.positions = positions if positions is not None else [None] * lanes

    def state_dict(self) -> dict[str, Any]:
        return {
            "order": self.order,
            "cursor": self.cursor,
            "positions": [None if p is None else list(p) for p in self.positions],
        }

    @classmethod
    def from_state(cls, lengths, lanes, length, state):
        return cls(lengths, state["order"], lanes, length, state["cursor"], state["positions"])

    def total_windows(self) -> int:
        return sum(-(-self.lengths[i] // self.length) for i in self.order)

    def next(self) -> list[tuple[int, int, int, bool] | None] | None:
        """Per lane (episode, start, end, starts_episode), or None when exhausted."""
        window: list[tuple[int, int, int, bool] | None] = []
        for lane in range(self.lanes):
            position = self.positions[lane]
            begins = False
            if position is None or position[1] >= self.lengths[position[0]]:
                position = None
                if self.cursor < len(self.order):
                    position = [self.order[self.cursor], 0]
                    self.cursor += 1
                    begins = True
            self.positions[lane] = position
            if position is None:
                window.append(None)
                continue
            episode, start = position
            end = min(start + self.length, self.lengths[episode])
            window.append((episode, start, end, begins))
            position[1] = end
        return window if any(item is not None for item in window) else None


class SequenceModel:
    """Runs an actor over stored or live sequences.

    Connectome and GRU actors separate frozen sensing (sense) from trainable
    dynamics (forward_code). Their sensory code is computed once per recorded
    step and reused by every epoch and PPO update; only trainable central state
    is carried through TBPTT. Other actors run unsplit on raw observations.
    """

    def __init__(self, policy, device: torch.device | str):
        self.policy, self.device = policy, torch.device(device)
        self.split = hasattr(policy, "forward_code") and hasattr(policy, "sense")
        self.sensory_size = policy.encoder.state_size if self.split else 0

    def initial(self, batch_size: int) -> tuple[torch.Tensor | None, torch.Tensor]:
        state = self.policy.initial_state(batch_size, self.device)
        if not self.split:
            return None, state
        return state[:, : self.sensory_size], state[:, self.sensory_size :]

    def observe(
        self, obs: dict[str, torch.Tensor], sensory: torch.Tensor | None
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor | None]:
        if not self.split:
            return dict(obs), None
        code, sensory = self.policy.sense(obs, sensory)
        return {"code": code}, sensory

    def step(self, inputs: dict[str, torch.Tensor], central: torch.Tensor):
        if self.split:
            return self.policy.forward_code(inputs["code"], central)
        return self.policy(inputs, central)

    def cache(self):
        factory = getattr(self.policy, "adjacency_cache", None)
        return factory() if factory is not None else contextlib.nullcontext()

    @torch.no_grad()
    def encode(self, episodes: list[Episode], lanes: int = 64) -> list[dict[str, torch.Tensor]]:
        """Per-episode actor inputs on device, [T, ...] per key."""
        if not self.split:
            return [
                {
                    key: torch.as_tensor(
                        episode.observations[key], device=self.device, dtype=torch.float32
                    )
                    for key in ("vision", "odor", "valid")
                }
                for episode in episodes
            ]
        result: list[dict[str, torch.Tensor] | None] = [None] * len(episodes)
        by_length: dict[int, list[int]] = {}
        for index, episode in enumerate(episodes):
            by_length.setdefault(len(episode), []).append(index)
        for length, indices in by_length.items():
            for first in range(0, len(indices), lanes):
                group = indices[first : first + lanes]
                sensory, _ = self.initial(len(group))
                codes = []
                for step in range(length):
                    obs = {
                        key: torch.as_tensor(
                            np.stack([episodes[i].observations[key][step] for i in group]),
                            device=self.device,
                            dtype=torch.float32,
                        )
                        for key in ("vision", "odor", "valid")
                    }
                    inputs, sensory = self.observe(obs, sensory)
                    codes.append(inputs["code"])
                stacked = torch.stack(codes, dim=1)
                for row, index in enumerate(group):
                    result[index] = {"code": stacked[row].contiguous()}
        return result  # type: ignore[return-value]
