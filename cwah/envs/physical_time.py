"""Standardized physical-action time for the C-WAH benchmark.

The formal benchmark executes VirtualHome scripts with animations skipped for
stability.  Physical time is therefore estimated from an offline-calibrated,
method-independent action timing instead of pose files produced at runtime.
"""

from dataclasses import asdict, dataclass
import math

from envs import action_protocol


@dataclass(frozen=True)
class PhysicalTimeSettings:
    frame_rate: int = 5
    movement_speed: float = 1.0
    turn_time: float = 1.0
    grab_time: float = 1.6
    open_time: float = 3.2
    put_time: float = 2.8
    distance_unit: str = "virtualhome_scene_unit"

    def to_dict(self):
        return asdict(self)


PHYSICAL_TIME_SETTINGS = PhysicalTimeSettings()


def character_positions(graph, num_agents):
    """Return agent-indexed VirtualHome character positions from a graph."""
    positions = {}
    for node in (graph or {}).get("nodes", []):
        node_id = node.get("id")
        if node_id is None or not 1 <= int(node_id) <= int(num_agents):
            continue
        if str(node.get("class_name", "")).lower() != "character":
            continue
        position = (node.get("obj_transform") or {}).get("position")
        if isinstance(position, (list, tuple)) and len(position) >= 3:
            positions[int(node_id) - 1] = [float(value) for value in position[:3]]
    return positions


def planar_distance(before, after):
    if before is None or after is None:
        return 0.0
    return math.hypot(float(after[0]) - float(before[0]), float(after[2]) - float(before[2]))


def estimate_action_time(action, before_position, after_position, timing=PHYSICAL_TIME_SETTINGS):
    """Estimate one agent's unquantized physical attempt time."""
    parsed, reason = action_protocol.parse_physical_action(action)
    if reason is not None or parsed.kind == action_protocol.WAIT:
        return {
            "action_kind": None,
            "distance": 0.0,
            "unquantized_time": 0.0,
        }

    distance = 0.0
    if parsed.kind == "walktowards":
        distance = planar_distance(before_position, after_position)
        seconds = distance / timing.movement_speed
    elif parsed.kind == "turnleft":
        seconds = timing.turn_time
    elif parsed.kind == "open":
        seconds = timing.open_time
    elif parsed.kind in {"grab", action_protocol.COOP_GRAB}:
        seconds = timing.grab_time
    elif parsed.kind in {"putin", "putback"} | action_protocol.COOP_PUT_ACTIONS:
        seconds = timing.put_time
    else:
        seconds = 0.0

    return {
        "action_kind": parsed.kind,
        "distance": float(distance),
        "unquantized_time": float(seconds),
    }


def seconds_to_equivalent_frames(seconds, frame_rate):
    seconds = max(float(seconds or 0.0), 0.0)
    if seconds == 0.0:
        return 0
    return int(math.ceil(seconds * float(frame_rate) - 1e-12))


def build_step_physical_time_info(
    *,
    step,
    num_agents,
    submitted_actions,
    charged_actions,
    charge_reasons,
    before_graph,
    after_graph,
    render_success,
    enabled=True,
    timing=PHYSICAL_TIME_SETTINGS,
):
    """Build physical-time evidence using standardized equivalent frames."""
    before_positions = character_positions(before_graph, num_agents)
    after_positions = character_positions(after_graph, num_agents)
    estimates = {}
    for agent_id in range(int(num_agents)):
        estimate = estimate_action_time(
            charged_actions.get(agent_id),
            before_positions.get(agent_id),
            after_positions.get(agent_id),
            timing,
        )
        estimate["individual_equivalent_frames"] = seconds_to_equivalent_frames(
            estimate["unquantized_time"], timing.frame_rate
        )
        estimate["individual_time"] = (
            float(estimate["individual_equivalent_frames"]) / float(timing.frame_rate)
        )
        estimates[agent_id] = estimate

    joint_frames = max(
        (item["individual_equivalent_frames"] for item in estimates.values()),
        default=0,
    ) if enabled else 0
    joint_time = float(joint_frames) / float(timing.frame_rate) if timing.frame_rate else 0.0

    per_agent = {}
    for agent_id in range(int(num_agents)):
        estimate = estimates[agent_id]
        per_agent[agent_id] = {
            "script": charged_actions.get(agent_id),
            "submitted_script": submitted_actions.get(agent_id),
            "submitted_physical_action": submitted_actions.get(agent_id) is not None,
            "charged_physical_attempt": charged_actions.get(agent_id) is not None,
            "charge_reason": charge_reasons.get(agent_id),
            "action_kind": estimate["action_kind"],
            "distance": estimate["distance"],
            "distance_unit": timing.distance_unit,
            "unquantized_time": estimate["unquantized_time"],
            "individual_equivalent_frames": estimate["individual_equivalent_frames"],
            "individual_time": estimate["individual_time"],
            "shared_timeline": True,
            "frames": joint_frames,
            "time": joint_time,
        }

    return {
        "step": int(step),
        "enabled": bool(enabled),
        "time_source": "calibrated_action_model",
        "physical_time_parameters": timing.to_dict(),
        "frame_rate": timing.frame_rate,
        "render_mode": "joint",
        "skip_animation": True,
        "aggregation": "max_concurrent_calibrated_action_time",
        "unit": "standardized_equivalent_frames",
        "joint": {
            "script": dict(charged_actions),
            "submitted_script": dict(submitted_actions),
            "frames": joint_frames,
            "time": joint_time,
            "render_success": render_success,
        },
        "per_agent": per_agent,
        "step_physical_action_frames": joint_frames,
        "step_physical_action_time": joint_time,
    }
