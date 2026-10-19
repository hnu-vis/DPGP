"""Formal action parsing and benchmark-level validation for C-WAH.

The protocol deliberately separates deterministic checks from Unity execution.
It validates only information available to the environment graph; target
visibility is not a precondition because a method may have learned an id through
an earlier observation or communication.
"""

from dataclasses import dataclass
import re


WAIT = "wait"
MOVE_ACTIONS = {"walktowards", "turnleft"}
COOP_GRAB = "coop_grab"
COOP_PUT_ACTIONS = {"coop_putin", "coop_putback"}
ACTIVE_CARRY_ALLOWED_ACTIONS = MOVE_ACTIONS | COOP_PUT_ACTIONS | {"open", WAIT}
IMPLICIT_CARRY_ALLOWED_ACTIONS = {"putin", "putback"}

ONE_OBJECT_ACTIONS = {"walktowards", "open", "grab", COOP_GRAB}
TWO_OBJECT_ACTIONS = {"putin", "putback", "coop_putin", "coop_putback"}
SUPPORTED_ACTIONS = ONE_OBJECT_ACTIONS | TWO_OBJECT_ACTIONS | {"turnleft"}

_ACTION_NAME_PATTERN = re.compile(r"^\s*\[([^\]]+)\]")
_ONE_OBJECT_PATTERN = re.compile(
    r"^\s*\[([^\]]+)\]\s+<([^<>]+)>\s+\((-?\d+)\)\s*$"
)
_TWO_OBJECT_PATTERN = re.compile(
    r"^\s*\[([^\]]+)\]\s+<([^<>]+)>\s+\((-?\d+)\)"
    r"\s+<([^<>]+)>\s+\((-?\d+)\)\s*$"
)
_NO_OBJECT_PATTERN = re.compile(r"^\s*\[([^\]]+)\]\s*$")


@dataclass(frozen=True)
class ParsedAction:
    kind: str
    raw: object
    object_class: str = None
    object_id: int = None
    target_class: str = None
    target_id: int = None


@dataclass(frozen=True)
class EnvelopeResult:
    raw: object
    physical_action: object
    message: object
    error: str = None


def _canonical_kind(value):
    value = str(value).strip().lower()
    aliases = {
        "turn_left": "turnleft",
        "turnleft": "turnleft",
    }
    return aliases.get(value, value)


def normalize_action_envelopes(action_dict, num_agents):
    """Require one exact physical_action/message envelope per agent."""
    source = action_dict if isinstance(action_dict, dict) else {}
    results = {}
    for agent_id in range(int(num_agents)):
        raw = source.get(agent_id, source.get(str(agent_id)))
        if not isinstance(raw, dict):
            results[agent_id] = EnvelopeResult(
                raw=raw,
                physical_action=None,
                message=None,
                error="invalid_action_envelope",
            )
            continue
        keys = set(raw.keys())
        if keys != {"physical_action", "message"}:
            missing = {"physical_action", "message"} - keys
            if "physical_action" in missing:
                reason = "missing_physical_action_field"
            elif "message" in missing:
                reason = "missing_message_field"
            else:
                reason = "invalid_action_envelope"
            results[agent_id] = EnvelopeResult(raw, None, None, reason)
            continue
        physical_action = raw.get("physical_action")
        message = raw.get("message")
        if physical_action is not None and not isinstance(physical_action, str):
            results[agent_id] = EnvelopeResult(
                raw, None, None, "invalid_physical_action_type"
            )
            continue
        if message is not None and not isinstance(message, str):
            results[agent_id] = EnvelopeResult(raw, None, None, "invalid_message_type")
            continue
        results[agent_id] = EnvelopeResult(raw, physical_action, message, None)
    return results


def validate_communication_envelope(envelope, communication_mode):
    """Validate whether communication and physical action may share a step."""
    if envelope.error is not None:
        return envelope.error
    if communication_mode == "exclusive_action":
        has_physical_action = envelope.physical_action is not None
        has_message = envelope.message not in (None, "")
        if has_physical_action and has_message:
            return "physical_message_mutually_exclusive"
    return None


def parse_physical_action(action):
    """Parse one canonical formal action without raising on malformed input."""
    if action is None:
        return ParsedAction(kind=WAIT, raw=None), None
    if not isinstance(action, str):
        return None, "invalid_physical_action_type"
    if not action.strip():
        return None, "malformed_action"

    name_match = _ACTION_NAME_PATTERN.match(action)
    if name_match is None:
        return None, "malformed_action"
    kind = _canonical_kind(name_match.group(1))
    if kind not in SUPPORTED_ACTIONS:
        return None, "unsupported_action"

    if kind == "turnleft":
        match = _NO_OBJECT_PATTERN.fullmatch(action)
        if match is None:
            return None, "malformed_action"
        return ParsedAction(kind=kind, raw=action), None

    if kind in ONE_OBJECT_ACTIONS:
        match = _ONE_OBJECT_PATTERN.fullmatch(action)
        if match is None:
            return None, "missing_object" if "(" not in action else "malformed_action"
        return ParsedAction(
            kind=kind,
            raw=action,
            object_class=match.group(2).strip(),
            object_id=int(match.group(3)),
        ), None

    match = _TWO_OBJECT_PATTERN.fullmatch(action)
    if match is None:
        object_matches = re.findall(r"<([^<>]+)>\s+\((-?\d+)\)", action)
        if len(object_matches) == 0:
            return None, "missing_object"
        if len(object_matches) == 1:
            return None, "missing_target"
        return None, "malformed_action"
    return ParsedAction(
        kind=kind,
        raw=action,
        object_class=match.group(2).strip(),
        object_id=int(match.group(3)),
        target_class=match.group(4).strip(),
        target_id=int(match.group(5)),
    ), None


def format_physical_action(action):
    """Return the canonical VirtualHome script spelling for a parsed action."""
    if action is None or action.kind == WAIT:
        return None
    if action.kind == "turnleft":
        return "[TurnLeft]"
    if action.kind in ONE_OBJECT_ACTIONS:
        return "[{}] <{}> ({})".format(
            action.kind, action.object_class, action.object_id
        )
    return "[{}] <{}> ({}) <{}> ({})".format(
        action.kind,
        action.object_class,
        action.object_id,
        action.target_class,
        action.target_id,
    )


def validate_class_ids(env, action):
    """Validate referenced node ids and their declared classes."""
    if action is None or action.kind in {WAIT, "turnleft"}:
        return None, None
    node = env._get_node_from_current_graph(action.object_id)
    if node is None:
        return "object_not_found", {"object_id": action.object_id}
    actual_class = str(node.get("class_name", ""))
    if actual_class.lower() != str(action.object_class).lower():
        return "class_id_mismatch", {
            "role": "object",
            "object_id": action.object_id,
            "declared_class": action.object_class,
        }
    if action.kind in TWO_OBJECT_ACTIONS:
        target = env._get_node_from_current_graph(action.target_id)
        if target is None:
            return "target_not_found", {"target_id": action.target_id}
        actual_target_class = str(target.get("class_name", ""))
        if actual_target_class.lower() != str(action.target_class).lower():
            return "class_id_mismatch", {
                "role": "target",
                "target_id": action.target_id,
                "declared_class": action.target_class,
            }
    return None, None


def active_carry_allowed_actions(env):
    allowed_actions = set(ACTIVE_CARRY_ALLOWED_ACTIONS)
    if getattr(env, "coop_action_interface", "implicit_pairing") == "implicit_pairing":
        allowed_actions.update(IMPLICIT_CARRY_ALLOWED_ACTIONS)
    return allowed_actions


def validate_active_carrying_action(env, action):
    carrying = getattr(env, "coop_carrying", None)
    if not isinstance(carrying, dict) or not carrying.get("active", False):
        return None
    allowed_actions = active_carry_allowed_actions(env)
    if action.kind not in allowed_actions:
        return "in_coop_carrying"
    return None


def _graph(env):
    return env.get_graph()


def _node_properties(node):
    return {str(value).upper() for value in node.get("properties", [])}


def _node_states(node):
    return {str(value).upper() for value in node.get("states", [])}


def _held_by_agents(env, object_id):
    holders = []
    for agent_id in range(int(env.num_agents)):
        if env._agent_holds_object(agent_id, object_id):
            holders.append(agent_id)
    return holders


def _inside_closed_container(env, object_id):
    graph = _graph(env)
    nodes = {int(node.get("id")): node for node in graph.get("nodes", [])}
    parent_by_child = {
        int(edge.get("from_id")): int(edge.get("to_id"))
        for edge in graph.get("edges", [])
        if str(edge.get("relation_type", "")).upper() == "INSIDE"
    }
    current = int(object_id)
    visited = set()
    while current in parent_by_child and current not in visited:
        visited.add(current)
        current = parent_by_child[current]
        parent = nodes.get(current)
        if parent is None:
            break
        if str(parent.get("category", "")) == "Rooms":
            return False
        if "CLOSED" in _node_states(parent):
            return True
    return False


def _goal_predicates(env):
    goal_spec = getattr(env, "goal_spec", {}) or {}
    if isinstance(goal_spec, dict) and 0 in goal_spec:
        goal_spec = goal_spec.get(0) or {}
    result = []
    for predicate, value in (goal_spec or {}).items():
        try:
            needed = int(value[0] if isinstance(value, (list, tuple)) else value)
        except Exception:
            needed = 0
        parts = str(predicate).split("_")
        if needed <= 0 or len(parts) < 3 or parts[0].lower() not in {"on", "inside"}:
            continue
        destination_text = parts[-1]
        destination_match = re.search(r"\((-?\d+)\)", destination_text)
        if destination_match is not None:
            destination_id = int(destination_match.group(1))
        else:
            try:
                destination_id = int(destination_text)
            except ValueError:
                continue
        result.append((parts[0].upper(), parts[1].lower(), destination_id))
    return result


def _goal_related(env, object_id):
    node = env._get_node_from_current_graph(object_id)
    if node is None:
        return False
    target_classes = {item[1] for item in _goal_predicates(env)}
    if not target_classes:
        return True
    return str(node.get("class_name", "")).lower() in target_classes


def _already_satisfies_goal(env, object_id):
    node = env._get_node_from_current_graph(object_id)
    if node is None:
        return False
    object_class = str(node.get("class_name", "")).lower()
    required = {
        (relation, destination_id)
        for relation, target_class, destination_id in _goal_predicates(env)
        if target_class == object_class
    }
    if not required:
        return False
    for edge in _graph(env).get("edges", []):
        relation = str(edge.get("relation_type", "")).upper()
        if (
            int(edge.get("from_id", -1)) == int(object_id)
            and (relation, int(edge.get("to_id", -1))) in required
        ):
            return True
    return False


def validate_walktowards(env, agent_id, action):
    del env, agent_id, action
    return None, None


def validate_open(env, agent_id, action):
    node = env._get_node_from_current_graph(action.object_id)
    properties = _node_properties(node)
    states = _node_states(node)
    if "OPEN" in states:
        return "target_already_open", None
    if "CAN_OPEN" not in properties:
        return "target_not_openable", None
    if "CLOSED" not in states:
        return "target_not_closed", None
    if not env._agent_close_to_object(agent_id, action.object_id):
        return "object_not_close", None
    if len(env._agent_held_objects(agent_id)) >= 2:
        return "no_free_hand", None
    return None, None


def validate_grab(env, agent_id, action):
    node = env._get_node_from_current_graph(action.object_id)
    if not env._agent_close_to_object(agent_id, action.object_id):
        return "object_not_close", None
    if "GRABBABLE" not in _node_properties(node):
        return "object_not_grabbable", None
    if _inside_closed_container(env, action.object_id):
        return "object_inside_closed_container", None
    occupancy = env._agent_grab_occupancy(agent_id)
    if occupancy.get("blocked"):
        return occupancy.get("reason"), {"held_objects": occupancy.get("held_objects", [])}
    holders = _held_by_agents(env, action.object_id)
    if holders:
        return "object_already_held", {"holder_agents": holders}
    if not _goal_related(env, action.object_id):
        return "grasp_non_goal_object", None
    if _already_satisfies_goal(env, action.object_id):
        return "grasp_target_already_satisfied", None
    return None, None


def validate_coop_grab(env, agent_ids, action):
    assignment, reason, details = env._coop_grab_assignment(action.object_id)
    if assignment is None:
        return reason, details

    # Only the physical holder must satisfy the ordinary object's CLOSE guard.
    # The supporter may instead be CLOSE to that holder. All non-spatial object
    # and occupancy guards still run before the hidden property is discovered.
    holder_agent = int(assignment["holder_agent"])
    reason, details = validate_grab(env, holder_agent, action)
    if reason is not None:
        if reason == "object_not_close":
            reason = "coop_grab_too_far"
        return reason, details
    for agent_id in agent_ids:
        if int(agent_id) == holder_agent:
            continue
        occupancy = env._agent_grab_occupancy(agent_id)
        if occupancy.get("blocked"):
            return occupancy.get("reason"), {
                "held_objects": occupancy.get("held_objects", []),
            }
    return None, None


def _validate_put_target(env, action):
    if int(action.object_id) == int(action.target_id):
        return "object_target_same", None
    target = env._get_node_from_current_graph(action.target_id)
    properties = _node_properties(target)
    states = _node_states(target)
    if action.kind in {"putin", "coop_putin"}:
        if not ({"CAN_OPEN", "CONTAINERS"} & properties):
            return "target_not_container", None
        if "OPEN" not in states:
            return "target_closed", None
    else:
        if "SURFACES" not in properties:
            return "target_not_surface", None
    return None, None


def validate_put(env, agent_id, action):
    if not env._agent_holds_object(agent_id, action.object_id):
        return "object_not_held", None
    if not env._agent_close_to_object(agent_id, action.target_id):
        return "target_not_close", None
    return _validate_put_target(env, action)


def validate_coop_put(env, action):
    carrying = getattr(env, "coop_carrying", None)
    if not isinstance(carrying, dict) or not carrying.get("active", False):
        return "no_active_coop_carrying", None
    if int(carrying.get("object_id", -1)) != int(action.object_id):
        return "coop_carrying_object_mismatch", None
    if carrying.get("status") == "paused_distance":
        return "coop_carrying_paused_distance", None

    holder_agent, partner_agent = carrying.get("holder_agent"), carrying.get("partner_agent")
    if (type(holder_agent) is not int or type(partner_agent) is not int
            or {holder_agent, partner_agent} != {0, 1}):
        return "invalid_coop_carrying_state", None
    holders = _held_by_agents(env, action.object_id)
    if holders != [holder_agent]:
        return "coop_object_not_held", {
            "expected_holder_agent": holder_agent,
            "actual_holder_agents": holders,
        }
    holder_close_to_target = env._agent_close_to_object(holder_agent, action.target_id)
    partner_close_to_target = env._agent_close_to_object(partner_agent, action.target_id)
    partner_close_to_holder = env._agents_close_to_each_other(
        holder_agent, partner_agent
    )
    if not holder_close_to_target:
        return "holder_not_close_to_target", {
            "holder_agent": holder_agent,
            "partner_agent": partner_agent,
            "holder_close_to_target": False,
            "partner_close_to_target": partner_close_to_target,
            "partner_close_to_holder": partner_close_to_holder,
        }
    if not (partner_close_to_target or partner_close_to_holder):
        return "supporter_not_close_for_coop_put", {
            "holder_agent": holder_agent,
            "partner_agent": partner_agent,
            "holder_close_to_target": True,
            "partner_close_to_target": False,
            "partner_close_to_holder": False,
        }

    return _validate_put_target(env, action)
