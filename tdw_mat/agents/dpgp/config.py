"""Validated method-only configuration for the TDW method pipeline."""

from dataclasses import asdict, dataclass
import math
from typing import Dict, Tuple



# Method-side arrival regions and the unchanged public TDW forward stride.
OBJECT_APPROACH_RADIUS = 1.0
GOAL_APPROACH_RADIUS = 1.5
EXPLORE_CENTER_RADIUS = 1.5
FORWARD_DISTANCE = 0.5


@dataclass(frozen=True)
class MethodConfig:
    """Settings owned by the method, not by the Benchmark environment."""

    map_size: Tuple[int, int] = (240, 120)
    scene_x_min: float = -15.0
    scene_x_max: float = 15.0
    scene_z_min: float = -7.5
    scene_z_max: float = 7.5
    cell_size: float = 0.125
    map_update_frequency: int = 10
    detection_distance: float = 5.0
    navigation_distance: float = 5.0
    wall_height: float = 2.0
    true_wall_height: float = 2.9
    turn_angle_degrees: float = 15.0
    message_limit: int = 12000
    action_horizon: int = 5
    edge_belief_threshold: float = 0.5
    prediction_attempts: int = 3
    prediction_cooldown_steps: int = 25
    dependency_failure_rounds: int = 3
    idle_replan_steps: int = 5
    target_position_wait_steps: int = 5
    coop_pair_retry_steps: int = 5
    discovery_candidate_count: int = 6
    discovery_world_count: int = 8
    discovery_seed: int = 1
    discovery_rollout_steps: int = 160
    discovery_astar_expansions: int = 2000

    def __post_init__(self) -> None:
        if (
            not isinstance(self.map_size, tuple)
            or len(self.map_size) != 2
            or any(type(value) is not int or value <= 0 for value in self.map_size)
        ):
            raise ValueError("map_size must contain two positive integers")
        integer_fields = (
            "map_update_frequency",
            "message_limit",
            "action_horizon",
            "prediction_attempts",
            "prediction_cooldown_steps",
            "dependency_failure_rounds",
            "idle_replan_steps",
            "target_position_wait_steps",
            "coop_pair_retry_steps",
            "discovery_candidate_count",
            "discovery_world_count",
            "discovery_rollout_steps",
            "discovery_astar_expansions",
        )
        for name in integer_fields:
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise ValueError("{} must be a positive integer".format(name))
        float_fields = (
            "scene_x_min",
            "scene_x_max",
            "scene_z_min",
            "scene_z_max",
            "cell_size",
            "detection_distance",
            "navigation_distance",
            "wall_height",
            "true_wall_height",
            "turn_angle_degrees",
            "edge_belief_threshold",
        )
        for name in float_fields:
            value = getattr(self, name)
            if not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError("{} must be finite".format(name))
        if self.scene_x_min >= self.scene_x_max or self.scene_z_min >= self.scene_z_max:
            raise ValueError("scene bounds must have positive extent")
        if min(
            self.cell_size,
            self.detection_distance,
            self.navigation_distance,
            self.wall_height,
            self.true_wall_height,
            self.turn_angle_degrees,
        ) <= 0:
            raise ValueError("spatial configuration values must be positive")
        if not 0.0 <= self.edge_belief_threshold <= 1.0:
            raise ValueError("edge_belief_threshold must be in [0, 1]")
        if type(self.discovery_seed) is not int or self.discovery_seed < 0:
            raise ValueError("discovery_seed must be a non-negative integer")

    @property
    def scene_bounds(self) -> Dict[str, float]:
        return {
            "x_min": float(self.scene_x_min),
            "x_max": float(self.scene_x_max),
            "z_min": float(self.scene_z_min),
            "z_max": float(self.scene_z_max),
        }

    def to_dict(self) -> dict:
        return asdict(self)
