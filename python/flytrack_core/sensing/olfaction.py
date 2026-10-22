"""Bilateral concentration sensing and adaptive Hill-rate ORN encoding."""

import numpy as np


def odor_sensor_positions(
    agent_pose: np.ndarray, baseline_m: float = 0.12, forward_m: float = 0.15
) -> np.ndarray:
    """Return [...,2,2] world xy, left then right, from [...,3] poses."""
    if baseline_m <= 0:
        raise ValueError("odor sensor baseline must be positive")
    pose = np.asarray(agent_pose, dtype=np.float64)
    forward = np.stack((np.cos(pose[..., 2]), np.sin(pose[..., 2])), axis=-1)
    left = np.stack((-np.sin(pose[..., 2]), np.cos(pose[..., 2])), axis=-1)
    center = pose[..., :2] + forward_m * forward
    return np.stack((center + baseline_m / 2 * left, center - baseline_m / 2 * left), axis=-2)


class ORNAdapter:
    """K state is [..., left/right, type]; invalid observations freeze it.

    r uses the old K, then K=max(K_floor,K+min(dt/tau,1)*(c-K)). This ordering
    is part of the sensing contract and must match the policy's Torch adapter.
    shared_adaptation (#4 P2, opt-in diagnostic) replaces c in the K update with the
    mean over the present sides, as odor_shared_adaptation does in the policy.
    """

    def __init__(
        self,
        num_types: int = 1,
        K=0.3,
        hill_n: float = 2,
        adaptation_tau_s: float = 2,
        K_floor: float = 1e-4,
        shared_adaptation: bool = False,
    ):
        if num_types < 1 or hill_n <= 0 or adaptation_tau_s <= 0 or K_floor <= 0:
            raise ValueError("ORN parameters must be positive")
        baseline = np.broadcast_to(np.asarray(K, dtype=np.float64), (num_types,)).copy()
        if np.any(baseline < K_floor) or not np.all(np.isfinite(baseline)):
            raise ValueError("K must be finite and at least K_floor")
        self.baseline = baseline
        self.num_types, self.hill_n = num_types, hill_n
        self.adaptation_tau_s, self.K_floor = adaptation_tau_s, K_floor
        self.shared_adaptation = bool(shared_adaptation)
        self.reset()

    def reset(self, batch_shape: tuple[int, ...] = ()) -> None:
        self.K = np.broadcast_to(self.baseline, (*batch_shape, 2, self.num_types)).copy()

    def update(
        self, concentration: np.ndarray, dt: float, valid: np.ndarray | None = None
    ) -> np.ndarray:
        if dt <= 0:
            raise ValueError("ORN dt must be positive")
        concentration = np.asarray(concentration, dtype=np.float64)
        if concentration.shape[-1:] != (2,):
            raise ValueError("concentration must end with bilateral dimension 2")
        expected = (*concentration.shape[:-1], 2, self.num_types)
        if self.K.shape != expected:
            self.reset(concentration.shape[:-1])
        present = np.isfinite(concentration)
        if valid is not None:
            supplied = np.asarray(valid, dtype=bool)
            if supplied.shape == concentration.shape[:-1]:
                supplied = supplied[..., None]
            present &= np.broadcast_to(supplied, concentration.shape)
        c = np.maximum(np.nan_to_num(concentration, nan=0, posinf=0, neginf=0), 0)[..., None]
        power = c**self.hill_n
        response = power / (power + self.K**self.hill_n)
        response = np.where(present[..., None], response, 0)
        target = c
        if self.shared_adaptation:
            weight = present[..., None].astype(np.float64)
            target = (c * weight).sum(-2, keepdims=True) / np.maximum(
                weight.sum(-2, keepdims=True), 1
            )
        updated = np.maximum(
            self.K_floor, self.K + min(dt / self.adaptation_tau_s, 1) * (target - self.K)
        )
        self.K = np.where(present[..., None], updated, self.K)
        return response.astype(np.float32)
