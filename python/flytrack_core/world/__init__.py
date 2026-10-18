"""Planar world physics; these modules have no Gymnasium or ROS imports."""

from .layout import LayoutConfig, motion_choices, sample_episode_layout
from .plume import GaussianPlume, PlumeConfig, PuffPlume, make_plume
from .scenarios import ScenarioSpec
from .trajectories import MOTIONS, TargetTrajectory, TrajectoryConfig, wrap_angle

__all__ = [
    "LayoutConfig",
    "MOTIONS",
    "motion_choices",
    "sample_episode_layout",
    "GaussianPlume",
    "PlumeConfig",
    "PuffPlume",
    "make_plume",
    "ScenarioSpec",
    "TargetTrajectory",
    "TrajectoryConfig",
    "wrap_angle",
]
