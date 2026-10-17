"""Pure numerical episode metrics, shared by training, simulator and ROS."""

from __future__ import annotations

import numpy as np


class EpisodeMetrics:
    def __init__(
        self,
        dt: float = 0.05,
        distance_threshold: float = 2.0,
        success_fraction: float = 0.8,
        bearing_threshold_deg: float | None = 45.0,
    ):
        """A tracked sample is within ``distance_threshold`` and, unless the bearing
        threshold is None (legacy distance-only definition), faces the target within
        ``bearing_threshold_deg``; success needs ``success_fraction`` tracked samples."""
        if dt <= 0 or distance_threshold <= 0 or not 0 <= success_fraction <= 1:
            raise ValueError("Invalid sampling interval or tracking success thresholds")
        if bearing_threshold_deg is not None and not 0 < bearing_threshold_deg <= 180:
            raise ValueError("bearing_threshold_deg must be in (0,180] or None")
        self.dt, self.distance_threshold, self.success_fraction = (
            dt,
            distance_threshold,
            success_fraction,
        )
        self.bearing_threshold_deg = bearing_threshold_deg
        self.distances, self.bearings, self.omegas, self.latencies = [], [], [], []
        self.collision = False
        self.previous_occluded = False
        self.capture_started = None
        self.recaptures: list[float] = []
        self.occlusion_events = 0

    def append(
        self,
        distance: float,
        bearing: float,
        omega: float,
        collision: bool,
        occluded: bool = False,
        timestamp: float | None = None,
        latency_ms: float | None = None,
    ):
        if timestamp is None:
            timestamp = len(self.distances) * self.dt
        self.distances.append(float(distance))
        self.bearings.append(abs(float(np.rad2deg(bearing))))
        self.omegas.append(float(omega))
        self.collision |= bool(collision)
        if latency_ms is not None and np.isfinite(latency_ms) and latency_ms >= 0:
            self.latencies.append(float(latency_ms))
        if occluded and not self.previous_occluded:
            self.occlusion_events += 1
            self.capture_started = None
        if self.previous_occluded and not occluded:
            self.capture_started = timestamp
        if self.capture_started is not None and not occluded and abs(np.rad2deg(bearing)) < 15:
            self.recaptures.append(float(timestamp - self.capture_started))
            self.capture_started = None
        self.previous_occluded = bool(occluded)

    def result(self) -> dict[str, float]:
        if not self.distances:
            raise ValueError("Cannot report an episode without samples")
        near = np.asarray(self.distances) <= self.distance_threshold
        facing = (
            np.ones_like(near)
            if self.bearing_threshold_deg is None
            else np.asarray(self.bearings) <= self.bearing_threshold_deg
        )
        tracked_fraction = float(np.mean(near & facing))
        return {
            "success": float(tracked_fraction >= self.success_fraction and not self.collision),
            "mean_distance_m": float(np.mean(self.distances)),
            "mean_bearing_error_deg": float(np.mean(self.bearings)),
            "reacquisition_time_s": float(np.mean(self.recaptures))
            if self.recaptures
            else float("nan"),
            "smoothness_rad_s": float(np.mean(np.abs(np.diff(self.omegas))))
            if len(self.omegas) > 1
            else 0.0,
            "collision_rate": float(self.collision),
            "latency_p50_ms": float(np.percentile(self.latencies, 50))
            if self.latencies
            else float("nan"),
            "latency_p95_ms": float(np.percentile(self.latencies, 95))
            if self.latencies
            else float("nan"),
            "tracked_fraction": tracked_fraction,
            "within_distance_fraction": float(np.mean(near)),
            "samples": len(self.distances),
            "occlusion_events": self.occlusion_events,
            "completed_recaptures": len(self.recaptures),
            "censored_recaptures": self.occlusion_events - len(self.recaptures),
        }
