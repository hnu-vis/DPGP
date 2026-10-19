"""Method-only settings; benchmark execution timing is not changed here."""

from dataclasses import asdict, dataclass
import math



@dataclass(frozen=True)
class MethodConfig:
    action_horizon: int = 8
    prediction_attempts: int = 2
    message_limit: int = 12000
    movement_speed: float = 1.0
    interaction_cost: float = 1.0
    completion_latency_weight: float = 0.1
    boundary_tolerance: float = 0.6
    pickup_candidates_per_class: int = 2
    navigation_step_distance: float = 1.0
    discovery_candidates: int = 6
    discovery_samples: int = 8
    discovery_time_horizon: float = 120.0
    discovery_rollout_steps: int = 160
    discovery_astar_expansions: int = 2000
    discovery_hidden_checks: float = 2.0

    def __post_init__(self):
        for name in ("action_horizon", "prediction_attempts", "message_limit", "pickup_candidates_per_class",
                     "discovery_candidates", "discovery_samples", "discovery_rollout_steps",
                     "discovery_astar_expansions"):
            if type(getattr(self, name)) is not int or getattr(self, name) <= 0:
                raise ValueError(name + " must be a positive integer")
        if self.prediction_attempts > 2:
            raise ValueError("at most two prediction attempts are supported")
        for name in ("movement_speed", "interaction_cost", "completion_latency_weight", "boundary_tolerance",
                     "navigation_step_distance", "discovery_time_horizon", "discovery_hidden_checks"):
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0:
                raise ValueError(name + " must be finite and nonnegative")
        if min(self.movement_speed, self.interaction_cost, self.navigation_step_distance,
               self.discovery_time_horizon, self.discovery_hidden_checks) == 0:
            raise ValueError("action costs must be positive")

    def to_dict(self, prediction_mode="discovery_probability"):
        result = dict(asdict(self),
                    cost_unit="normalized_planning_proxy",
                    action_selection="per_class_candidates_ready_first_nearest_delivery",
                    optimality_scope="task_directed_action_domain",
                    heuristic="independent_delivery_lower_bound",
                    prefix_boundary="shared_observation_and_horizon")
        if prediction_mode == "discovery_probability":
            result.update(cost_unit="estimated_calibrated_seconds",
                          optimality_scope="finite_sample_candidate_comparison",
                          timing="synchronous_action_opportunities_step_max",
                          internal_astar_cost="unit_interaction_proposal_only",
                          structure_tail="hidden_room_search_cost_proxy", lookahead_depth=1,
                          discovery_selection="task_capped_joint_candidate",
                          discovery_prior="task_capped_storage_hypotheses",
                          synchronization="fact_change_decision_events")
        return result
