"""Method-independent TDW action-envelope and cooperation normalization."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any, Dict, Iterable, Optional, Tuple


SUPPORTED_ACTION_TYPES = {0, 1, 2, 3, 4, 5, 6, 7, "ongoing"}

# The Benchmark uses the right hand for grasp and resolves the holding arm
# for drop. Validate the optional arm label at the public boundary.
ACTION_ALLOWED_FIELDS = {
    0: {"type"},
    1: {"type"},
    2: {"type"},
    3: {"type", "object", "arm"},
    4: {"type", "object", "arm"},
    5: {"type", "object", "arm"},
    6: {"type", "object", "arm"},
    7: {"type"},
    "ongoing": {"type"},
}
ACTION_REQUIRED_FIELDS = {
    3: {"object"},
    4: {"object"},
}


def canonical_action_status_code(status: Any) -> Optional[str]:
    """Return a normalized code for a low-level action status."""
    if status is None:
        return None
    value = str(status)
    prefix = "ActionStatus."
    return value[len(prefix):] if value.startswith(prefix) else value


@dataclass(frozen=True)
class EnvelopeResult:
    raw: Any
    requested_physical_action: Any
    physical_action: Dict[str, Any]
    message: Optional[str]
    error: Optional[str] = None


def validate_physical_action_payload(
    action: Any,
) -> Tuple[Dict[str, Any], Optional[str]]:
    """Validate and canonicalize one method-facing physical action."""
    if not isinstance(action, dict):
        return {"type": 7}, "invalid_physical_action_type"
    if "type" not in action:
        return {"type": 7}, "missing_action_type"

    action_type = action.get("type")
    if isinstance(action_type, bool) or not (
        type(action_type) is int or action_type == "ongoing"
    ):
        return {"type": 7}, "unsupported_action_type"
    if action_type not in SUPPORTED_ACTION_TYPES:
        return {"type": 7}, "unsupported_action_type"

    allowed_fields = ACTION_ALLOWED_FIELDS[action_type]
    if set(action) - allowed_fields:
        return {"type": 7}, "invalid_physical_action_fields"
    if ACTION_REQUIRED_FIELDS.get(action_type, set()) - set(action):
        return {"type": 7}, "missing_object"

    canonical = copy.deepcopy(action)
    if "object" in canonical:
        object_id = canonical["object"]
        if isinstance(object_id, bool):
            return {"type": 7}, "invalid_object_id"
        try:
            canonical["object"] = int(object_id)
        except (TypeError, ValueError):
            return {"type": 7}, "invalid_object_id"
    if "arm" in canonical and canonical["arm"] not in {"left", "right"}:
        return {"type": 7}, "invalid_arm"
    return canonical, None


def normalize_action_envelopes(
    actions: Any,
    agent_ids: Iterable[int],
) -> Dict[int, EnvelopeResult]:
    """Normalize one strict physical-action/message envelope per agent."""
    payloads = actions if isinstance(actions, dict) else {}
    results: Dict[int, EnvelopeResult] = {}
    for agent_id in agent_ids:
        raw = copy.deepcopy(payloads.get(str(agent_id), payloads.get(agent_id)))
        error = None
        requested_physical_action = None
        message = None

        if not isinstance(raw, dict):
            error = "invalid_action_envelope"
        elif set(raw.keys()) != {"physical_action", "message"}:
            missing = {"physical_action", "message"} - set(raw.keys())
            if "physical_action" in missing:
                error = "missing_physical_action_field"
            elif "message" in missing:
                error = "missing_message_field"
            else:
                error = "invalid_action_envelope"
        else:
            requested_physical_action = copy.deepcopy(raw.get("physical_action"))
            message = raw.get("message")

        if error is None and requested_physical_action is None:
            physical_action = {"type": 7}
        elif error is None:
            physical_action, error = validate_physical_action_payload(
                requested_physical_action
            )
        else:
            physical_action = {"type": 7}
        if error is None and message is not None and not isinstance(message, str):
            error = "invalid_message_type"
            physical_action = {"type": 7}
            message = None

        results[int(agent_id)] = EnvelopeResult(
            raw=raw,
            requested_physical_action=requested_physical_action,
            physical_action=physical_action,
            message=message,
            error=error,
        )
    return results


def apply_communication_mode(
    envelope: EnvelopeResult,
    communication_mode: str,
) -> Tuple[Dict[str, Any], Optional[str], Optional[str]]:
    """Return effective physical action, deliverable message, and protocol reason."""
    if communication_mode != "exclusive_action":
        raise ValueError("communication_mode must be exclusive_action")
    physical_action = copy.deepcopy(envelope.physical_action)
    message = envelope.message if envelope.message not in (None, "") else None
    if envelope.error is not None or message is None:
        return physical_action, message, envelope.error

    action_type = physical_action.get("type")
    if action_type == "ongoing":
        # An already-dispatched Replicant action can't be cancelled safely.
        return physical_action, None, "message_blocked_by_ongoing_action"
    if action_type != 7:
        return {"type": 7}, message, "physical_message_mutually_exclusive"
    return physical_action, message, None


def apply_implicit_coop_conversion(
    actions: Dict[str, Dict[str, Any]],
    agent_ids: Iterable[int],
    *,
    coop_action_interface: str,
    active_coop_carrying: Optional[dict],
) -> Dict[int, str]:
    """Convert matching ordinary grasp/drop intents to internal coop actions."""
    ids = [int(agent_id) for agent_id in agent_ids]
    if coop_action_interface != "implicit_pairing":
        raise ValueError("coop_action_interface must be implicit_pairing")
    if len(ids) != 2:
        raise ValueError("cooperation requires two agents")
    pair = [actions.get(str(agent_id), {}) for agent_id in ids]

    if all(action.get("type") == 3 for action in pair):
        try:
            object_ids = [int(action["object"]) for action in pair]
        except (KeyError, TypeError, ValueError):
            return {}
        if object_ids[0] != object_ids[1]:
            return {}
        for agent_id in ids:
            actions[str(agent_id)]["type"] = 4
        return {agent_id: "paired_grasp_to_coop_grasp" for agent_id in ids}

    if active_coop_carrying is not None and all(
        action.get("type") == 5 for action in pair
    ):
        for agent_id in ids:
            actions[str(agent_id)]["type"] = 6
        return {agent_id: "paired_drop_to_coop_drop" for agent_id in ids}
    return {}
