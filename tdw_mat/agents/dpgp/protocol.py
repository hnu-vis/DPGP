"""Compact fact exchange and synchronized plan activation.

The graph payload is always :class:`FactGraph`.  Probabilistic runs may add a
validated belief seed and a minimal execution commitment to the envelope;
temporary deterministic assumptions never cross the boundary.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
import json
from typing import Optional, Tuple

from .config import MethodConfig
from .models import DeterministicGraph, FactGraph
from .planning_domain import MOVE, OBJECT, PUT, ROOM, HighLevelAction, JointHighLevelAction


GRAPH_MESSAGE = "planning_graph"
ACTION_MESSAGE = "planning_action"
EXCLUSIVE_ACTION = "exclusive_action"

COOP_GRAB = "coop_grab"
COOP_PUT = "coop_put"

_DEFAULTS = MethodConfig()
DEFAULT_MAX_MESSAGE_CHARS = _DEFAULTS.message_limit
DEFAULT_PLANNING_ACTION_HORIZON = _DEFAULTS.action_horizon
DEFAULT_EDGE_BELIEF_THRESHOLD = _DEFAULTS.edge_belief_threshold


class CommunicationProtocolError(ValueError):
    """Raised when a method message cannot be trusted or reconstructed."""


@dataclass(frozen=True)
class PlanningGraphMessage:
    source_agent: int
    source_step: int
    plan_epoch: int
    graph: FactGraph
    is_full_snapshot: bool
    base_epoch: Optional[int] = None
    belief_seed: Optional[dict] = None
    commitment: Optional[dict] = None


@dataclass(frozen=True)
class PlanningActionMessage:
    source_agent: int
    source_step: int
    plan_epoch: int
    activation_step: int
    actions: Tuple[dict, ...]
    belief_seed: Optional[dict] = None


def _belief_seed(value):
    if value is None:
        return None
    try:
        from .discovery_belief import DiscoveryBelief
        return DiscoveryBelief.from_seed(value).to_seed()
    except (TypeError, ValueError) as error:
        raise CommunicationProtocolError("invalid discovery belief seed") from error


def _commitment(value):
    if value is None:
        return None
    expected = {"agent_id", "busy", "action"}
    if not isinstance(value, dict) or set(value) != expected:
        raise CommunicationProtocolError("invalid execution commitment")
    agent_id = _agent_id(value["agent_id"], "commitment agent_id")
    busy = value["busy"]
    action = value["action"]
    if type(busy) is not bool:
        raise CommunicationProtocolError("commitment busy must be boolean")
    if action is not None:
        fields = {"kind", "object_id", "target_kind", "target_room_node_id"}
        if (not isinstance(action, dict) or not fields <= set(action)
                or set(action) - fields - {"required_revisit_count", "scan_started"}):
            raise CommunicationProtocolError("invalid committed action fields")
        if "required_revisit_count" in action and (
                action["kind"] != MOVE or action["target_kind"] != ROOM
                or type(action["required_revisit_count"]) is not int
                or action["required_revisit_count"] < 0):
            raise CommunicationProtocolError("invalid committed scan requirement")
        if "scan_started" in action and (action["kind"] != MOVE or action["target_kind"] != ROOM
                                         or type(action["scan_started"]) is not bool):
            raise CommunicationProtocolError("invalid committed scan phase")
        try:
            # Wire identity is stable across fused graph slot renumbering.
            HighLevelAction(action["kind"], action["object_id"],
                            action["target_kind"], action["target_room_node_id"])
        except (TypeError, ValueError) as error:
            raise CommunicationProtocolError("invalid committed action") from error
    return {"agent_id": agent_id, "busy": busy,
            "action": None if action is None else dict(action)}


def _compact_json(data: dict) -> str:
    return json.dumps(
        data, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )


def _check_length(message: str, max_chars: int) -> None:
    if type(max_chars) is not int or max_chars <= 0:
        raise ValueError("max_chars must be a positive integer")
    if len(message) > max_chars:
        raise CommunicationProtocolError(
            "message_too_long: {} > {} characters".format(len(message), max_chars)
        )


def _load_message(message: str, max_chars: int) -> dict:
    if not isinstance(message, str) or not message:
        raise CommunicationProtocolError("message must be a non-empty string")
    _check_length(message, max_chars)
    try:
        data = json.loads(message)
    except json.JSONDecodeError as error:
        raise CommunicationProtocolError("invalid JSON message") from error
    if not isinstance(data, dict):
        raise CommunicationProtocolError("message root must be an object")
    return data


def _agent_id(value, name: str = "source_agent") -> int:
    if type(value) is not int or value not in (0, 1):
        raise CommunicationProtocolError("{} must be 0 or 1".format(name))
    return value


def _step(value, name: str) -> int:
    if type(value) is not int or value < 0:
        raise CommunicationProtocolError(
            "{} must be a non-negative integer".format(name)
        )
    return value


def peek_message_type(
    message: str,
    max_chars: int = DEFAULT_MAX_MESSAGE_CHARS,
) -> str:
    message_type = _load_message(message, max_chars).get("message_type")
    if message_type not in (GRAPH_MESSAGE, ACTION_MESSAGE):
        raise CommunicationProtocolError("unknown message type")
    return message_type


def _peek_graph(message: str, max_chars: int) -> tuple:
    data = _load_message(message, max_chars)
    if data.get("message_type") != GRAPH_MESSAGE:
        raise CommunicationProtocolError("not a planning graph message")
    return _step(data.get("round"), "round"), "full" in data


_LIST_KEYS = {
    "rooms": "node_id",
    "edges": ("source", "target"),
    "objects": "graph_index",
    "agents": "agent_id",
    "inspections": "inspection_id",
}


def _record_key(item: dict, key_spec):
    if isinstance(key_spec, tuple):
        return tuple(item[name] for name in key_spec)
    return item[key_spec]


def _transport_graph(graph: FactGraph) -> dict:
    data = graph.to_dict()
    data.pop("frame")
    return data


def _list_delta(old_items, new_items, key_spec) -> Optional[dict]:
    old = {_record_key(item, key_spec): item for item in old_items}
    new = {_record_key(item, key_spec): item for item in new_items}
    upsert = [copy.deepcopy(new[key]) for key in sorted(new)
              if old.get(key) != new[key]]
    remove = [list(key) if isinstance(key, tuple) else key
              for key in sorted(old) if key not in new]
    if not upsert and not remove:
        return None
    return dict((name, value) for name, value in (
        ("upsert", upsert), ("remove", remove)
    ) if value)


def _edge_delta(old_items, new_items, threshold: float) -> Optional[dict]:
    key_spec = _LIST_KEYS["edges"]
    old = {_record_key(item, key_spec): item for item in old_items}
    new = {_record_key(item, key_spec): item for item in new_items}
    upsert = []
    for key in sorted(new):
        old_belief = float(old[key]["belief"]) if key in old else float("-inf")
        if old_belief < threshold <= float(new[key]["belief"]):
            upsert.append(copy.deepcopy(new[key]))
    return {"upsert": upsert} if upsert else None


def _graph_delta(previous: FactGraph, current: FactGraph, threshold: float) -> dict:
    old, new = _transport_graph(previous), _transport_graph(current)
    delta = {}
    for name, key_spec in _LIST_KEYS.items():
        change = (_edge_delta(old[name], new[name], threshold)
                  if name == "edges" else
                  _list_delta(old[name], new[name], key_spec))
        if change is not None:
            delta[name] = change
    for name in ("goal", "cooperation"):
        if old[name] != new[name]:
            delta[name] = copy.deepcopy(new[name])
    return delta


def _apply_graph_delta(previous: FactGraph, delta: dict, frame: int) -> FactGraph:
    if not isinstance(delta, dict) or not set(delta) <= set(_LIST_KEYS) | {"goal", "cooperation"}:
        raise CommunicationProtocolError("invalid graph delta fields")
    result = previous.to_dict()
    result["frame"] = frame
    for name, key_spec in _LIST_KEYS.items():
        if name not in delta:
            continue
        change = delta[name]
        if not isinstance(change, dict) or not set(change) <= {"upsert", "remove"}:
            raise CommunicationProtocolError("invalid {} delta".format(name))
        upsert, remove = change.get("upsert", []), change.get("remove", [])
        if not isinstance(upsert, list) or not isinstance(remove, list):
            raise CommunicationProtocolError("invalid {} delta".format(name))
        by_key = {_record_key(item, key_spec): item for item in result[name]}
        for raw_key in remove:
            key = tuple(raw_key) if isinstance(key_spec, tuple) else raw_key
            by_key.pop(key, None)
        for item in upsert:
            if not isinstance(item, dict):
                raise CommunicationProtocolError("invalid {} upsert".format(name))
            by_key[_record_key(item, key_spec)] = copy.deepcopy(item)
        result[name] = [by_key[key] for key in sorted(by_key)]
    if "goal" in delta:
        if not isinstance(delta["goal"], dict):
            raise CommunicationProtocolError("invalid goal delta")
        result["goal"] = copy.deepcopy(delta["goal"])
    if "cooperation" in delta:
        result["cooperation"] = copy.deepcopy(delta["cooperation"])
    try:
        return FactGraph.from_dict(result)
    except (KeyError, TypeError, ValueError) as error:
        raise CommunicationProtocolError("invalid reconstructed graph") from error


def encode_planning_graph(
    graph: FactGraph,
    step: int,
    source_agent: int = 0,
    max_chars: int = DEFAULT_MAX_MESSAGE_CHARS,
    previous_graph: Optional[FactGraph] = None,
    previous_round: Optional[int] = None,
    edge_belief_threshold: float = DEFAULT_EDGE_BELIEF_THRESHOLD,
    belief_seed: Optional[dict] = None,
    commitment: Optional[dict] = None,
) -> str:
    """Encode an initial full fact graph or a later fact-only delta."""
    if not isinstance(graph, FactGraph):
        raise TypeError("graph communication accepts FactGraph only")
    graph.validate()
    source_agent, step = _agent_id(source_agent), _step(step, "step")
    threshold = float(edge_belief_threshold)
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("edge_belief_threshold must be in [0, 1]")
    data = {
        "message_type": GRAPH_MESSAGE,
        "source_agent": source_agent,
        "round": step,
        "frame": graph.frame,
    }
    if previous_graph is None:
        if previous_round is not None:
            raise CommunicationProtocolError("initial graph cannot have a base")
        data["full"] = _transport_graph(graph)
    else:
        if not isinstance(previous_graph, FactGraph):
            raise TypeError("previous_graph must be FactGraph")
        previous_graph.validate()
        previous_round = _step(previous_round, "previous_round")
        if previous_round >= step:
            raise CommunicationProtocolError("graph round must increase")
        data["base"] = previous_round
        data["delta"] = _graph_delta(previous_graph, graph, threshold)
    if belief_seed is not None:
        data["belief"] = _belief_seed(belief_seed)
    if commitment is not None:
        data["commitment"] = _commitment(commitment)
        if data["commitment"]["agent_id"] != source_agent:
            raise CommunicationProtocolError("commitment must come from its owner")
    message = _compact_json(data)
    _check_length(message, max_chars)
    return message


def decode_planning_graph(
    message: str,
    receiver_agent: Optional[int] = None,
    current_step: Optional[int] = None,
    max_chars: int = DEFAULT_MAX_MESSAGE_CHARS,
    previous_graph: Optional[FactGraph] = None,
    previous_round: Optional[int] = None,
) -> PlanningGraphMessage:
    data = _load_message(message, max_chars)
    if data.get("message_type") != GRAPH_MESSAGE:
        raise CommunicationProtocolError("not a planning graph message")
    source_agent = _agent_id(data.get("source_agent"))
    if receiver_agent is not None and source_agent == _agent_id(
        receiver_agent, "receiver_agent"
    ):
        raise CommunicationProtocolError("planning graph must come from the peer")
    round_id = _step(data.get("round"), "round")
    if current_step is not None and round_id > int(current_step):
        raise CommunicationProtocolError("graph round is in the future")
    frame = _step(data.get("frame"), "frame")
    if "full" in data:
        expected = {"message_type", "source_agent", "round",
                    "frame", "full"}
        if (not expected <= set(data)
                or set(data) - expected - {"belief", "commitment"}
                or previous_graph is not None):
            raise CommunicationProtocolError("invalid full graph packet")
        raw = copy.deepcopy(data["full"])
        if not isinstance(raw, dict):
            raise CommunicationProtocolError("invalid full planning graph")
        raw.update(frame=frame)
        try:
            graph = FactGraph.from_dict(raw)
        except (KeyError, TypeError, ValueError) as error:
            raise CommunicationProtocolError("invalid full planning graph") from error
        base_epoch, is_full = None, True
    else:
        expected = {"message_type", "source_agent", "round",
                    "frame", "base", "delta"}
        if (not expected <= set(data)
                or set(data) - expected - {"belief", "commitment"}
                or previous_graph is None or previous_round is None):
            raise CommunicationProtocolError("invalid graph delta packet")
        base_epoch = _step(data["base"], "base")
        if base_epoch != int(previous_round):
            raise CommunicationProtocolError("graph delta baseline mismatch")
        if round_id <= base_epoch:
            raise CommunicationProtocolError("graph round must increase")
        graph = _apply_graph_delta(previous_graph, data["delta"], frame)
        is_full = False
    commitment = _commitment(data.get("commitment"))
    if commitment is not None and commitment["agent_id"] != source_agent:
        raise CommunicationProtocolError("commitment must come from its owner")
    return PlanningGraphMessage(
        source_agent, round_id, round_id, graph, is_full, base_epoch,
        _belief_seed(data.get("belief")), commitment,
    )


def _action_dict(action, peer_action, graph: DeterministicGraph) -> dict:
    data = action.to_dict()
    index = action.object_index
    node = next((item for item in graph.objects
                 if item.graph_index == index), None)
    data["object_id"] = node.object_id if node is not None else None
    paired = (peer_action is not None and action.kind in ("grab", PUT)
              and peer_action.kind == action.kind
              and peer_action.object_index == index)
    if action.kind in ("grab", PUT) and node is not None:
        if node.property == 2 or (node.property is None and paired):
            data["kind"] = COOP_GRAB if action.kind == "grab" else COOP_PUT
    if (action.kind == MOVE and action.target_kind == ROOM
            and action.target_room_node_id in graph.reopened_room_ids):
        room = next(item for item in graph.facts.rooms
                    if item.node_id == action.target_room_node_id)
        data["required_revisit_count"] = room.revisit_count + 1
    return data


def _requires_observation_barrier(joint, graph: DeterministicGraph) -> bool:
    unexplored = {item.node_id for item in graph.rooms if not item.explored}
    reopened = set(graph.reopened_room_ids)
    objects = {item.graph_index: item for item in graph.objects}
    for action in (joint.agent_0, joint.agent_1):
        if action is None or action.kind != MOVE:
            continue
        if (action.target_kind == ROOM
                and (action.target_room_node_id in unexplored
                     or action.target_room_node_id in reopened)):
            return True
        if (action.target_kind == OBJECT and action.object_index in objects
                and objects[action.object_index].location_assumed):
            return True
    return False


def project_planning_actions(
    result,
    graph: DeterministicGraph,
    agent_id: int,
    max_actions: int = DEFAULT_PLANNING_ACTION_HORIZON,
) -> Tuple[dict, ...]:
    """Project a bounded executable prefix for exactly one recipient."""
    if not isinstance(graph, DeterministicGraph):
        raise TypeError("action projection requires DeterministicGraph")
    graph.validate()
    agent_id = _agent_id(agent_id, "agent_id")
    if type(max_actions) is not int or max_actions <= 0:
        raise ValueError("max_actions must be a positive integer")
    projected = []
    continuing = {(agent, room): count for agent, room, count
                  in getattr(result, "continued_scans", ())}
    for joint in tuple(getattr(result, "joint_actions", ()) or ()):
        if not isinstance(joint, JointHighLevelAction):
            raise CommunicationProtocolError("invalid joint planning trajectory")
        own = joint.agent_0 if agent_id == 0 else joint.agent_1
        peer = joint.agent_1 if agent_id == 0 else joint.agent_0
        if own is not None:
            data = _action_dict(own, peer, graph)
            key = (agent_id, own.target_room_node_id)
            if own.kind == MOVE and own.target_kind == ROOM and key in continuing:
                data["required_revisit_count"] = continuing.pop(key)
            projected.append(data)
        if _requires_observation_barrier(joint, graph):
            break
        if len(projected) >= max_actions:
            break
    return tuple(projected)


def build_planning_action_message(
    result,
    graph: DeterministicGraph,
    source_agent: int,
    source_step: int,
    plan_epoch: int,
    planned_agent: int,
    max_actions: int = DEFAULT_PLANNING_ACTION_HORIZON,
    belief_seed: Optional[dict] = None,
) -> PlanningActionMessage:
    source_agent = _agent_id(source_agent)
    source_step = _step(source_step, "source_step")
    plan_epoch = _step(plan_epoch, "plan_epoch")
    if plan_epoch > source_step:
        raise CommunicationProtocolError("plan epoch is from the future")
    return PlanningActionMessage(
        source_agent, source_step, plan_epoch, source_step + 1,
        project_planning_actions(result, graph, planned_agent, max_actions),
        _belief_seed(belief_seed),
    )


def encode_planning_action(
    result,
    graph: DeterministicGraph,
    step: int,
    source_agent: int = 0,
    plan_epoch: Optional[int] = None,
    planned_agent: Optional[int] = None,
    max_actions: int = DEFAULT_PLANNING_ACTION_HORIZON,
    max_chars: int = DEFAULT_MAX_MESSAGE_CHARS,
    belief_seed: Optional[dict] = None,
) -> str:
    source_agent, step = _agent_id(source_agent), _step(step, "step")
    packet = build_planning_action_message(
        result, graph, source_agent, step,
        step if plan_epoch is None else plan_epoch,
        1 - source_agent if planned_agent is None else planned_agent,
        max_actions, belief_seed,
    )
    data = {
        "message_type": ACTION_MESSAGE,
        "source_agent": packet.source_agent,
        "source_step": packet.source_step,
        "plan_epoch": packet.plan_epoch,
        "activation_step": packet.activation_step,
        "actions": list(packet.actions),
    }
    if packet.belief_seed is not None:
        data["belief"] = packet.belief_seed
    message = _compact_json(data)
    _check_length(message, max_chars)
    return message


def decode_planning_action(
    message: str,
    receiver_agent: Optional[int] = None,
    current_step: Optional[int] = None,
    max_chars: int = DEFAULT_MAX_MESSAGE_CHARS,
) -> PlanningActionMessage:
    data = _load_message(message, max_chars)
    expected = {"message_type", "source_agent", "source_step", "plan_epoch",
                "activation_step", "actions"}
    if (not expected <= set(data) or set(data) - expected - {"belief"}
            or data.get("message_type") != ACTION_MESSAGE):
        raise CommunicationProtocolError("invalid planning action message")
    source_agent = _agent_id(data["source_agent"])
    if receiver_agent is not None and source_agent == _agent_id(
        receiver_agent, "receiver_agent"
    ):
        raise CommunicationProtocolError("planning action must come from peer")
    source_step = _step(data["source_step"], "source_step")
    plan_epoch = _step(data["plan_epoch"], "plan_epoch")
    activation_step = _step(data["activation_step"], "activation_step")
    if activation_step != source_step + 1:
        raise CommunicationProtocolError("invalid activation step")
    if current_step is not None and source_step > int(current_step):
        raise CommunicationProtocolError("planning action is from the future")
    if plan_epoch > source_step:
        raise CommunicationProtocolError("plan epoch is from the future")
    actions = data["actions"]
    if not isinstance(actions, list) or not all(isinstance(item, dict) for item in actions):
        raise CommunicationProtocolError("actions must be a list of objects")
    return PlanningActionMessage(
        source_agent, source_step, plan_epoch, activation_step,
        tuple(copy.deepcopy(actions)), _belief_seed(data.get("belief")),
    )


class CommunicationCoordinator:
    """Own graph baselines, plan epochs, activation, and outgoing envelopes."""

    def __init__(self, agent_id: int, communication_mode: str = EXCLUSIVE_ACTION,
                 config: Optional[MethodConfig] = None):
        self.agent_id = _agent_id(agent_id, "agent_id")
        if communication_mode != EXCLUSIVE_ACTION:
            raise ValueError("unsupported communication_mode")
        self.communication_mode = communication_mode
        self.config = config or MethodConfig()
        self.reset()

    def reset(self) -> None:
        self.current_plan_epoch = None
        self.latest_peer_graph: Optional[PlanningGraphMessage] = None
        self.last_sent_graph: Optional[FactGraph] = None
        self.last_sent_graph_step = None
        self.selected_plan: Optional[PlanningActionMessage] = None
        self.prepared_own_plan: Optional[PlanningActionMessage] = None
        self.pending_graph_message = None
        self.pending_graph_baseline = None
        self.pending_action_message = None
        self.last_activated_key = None
        self.last_protocol_error = None

    def prepare_graph(self, graph: FactGraph, current_step: int,
                      belief_seed=None, commitment=None) -> str:
        message = encode_planning_graph(
            graph, current_step, self.agent_id, self.config.message_limit,
            self.last_sent_graph, self.last_sent_graph_step,
            self.config.edge_belief_threshold,
            belief_seed, commitment,
        )
        self.current_plan_epoch = current_step
        self.selected_plan = self.prepared_own_plan = None
        self.pending_action_message = self.last_activated_key = None
        self.pending_graph_message = message
        self.pending_graph_baseline = (graph, current_step)
        return message

    def prepare_actions(self, result, graph: DeterministicGraph,
                        current_step: int, plan_epoch: int,
                        belief_seed=None) -> None:
        if self.is_stale_epoch(plan_epoch):
            raise CommunicationProtocolError(
                "cannot send a planning action for a stale epoch"
            )
        self.pending_action_message = encode_planning_action(
            result, graph, current_step, self.agent_id, plan_epoch,
            1 - self.agent_id, self.config.action_horizon,
            self.config.message_limit, belief_seed,
        )
        self.prepared_own_plan = build_planning_action_message(
            result, graph, self.agent_id, current_step, plan_epoch,
            self.agent_id, self.config.action_horizon,
            belief_seed,
        )

    def take_prepared_own_plan(self):
        plan, self.prepared_own_plan = self.prepared_own_plan, None
        return plan

    def select_plan(self, plan: PlanningActionMessage) -> None:
        self._store_plan(plan)

    def has_pending_work(self) -> bool:
        return any((self.selected_plan, self.pending_graph_message,
                    self.pending_action_message))

    def receive(self, message: Optional[str], current_step: int):
        self.last_protocol_error = None
        if message in (None, ""):
            return None
        try:
            message_type = peek_message_type(message, self.config.message_limit)
            if message_type == GRAPH_MESSAGE:
                graph_round, is_full = _peek_graph(message, self.config.message_limit)
                if (self.latest_peer_graph is not None
                        and graph_round <= self.latest_peer_graph.plan_epoch):
                    return None
                decoded = decode_planning_graph(
                    message, self.agent_id, current_step, self.config.message_limit,
                    None if is_full or self.latest_peer_graph is None
                    else self.latest_peer_graph.graph,
                    None if is_full or self.latest_peer_graph is None
                    else self.latest_peer_graph.plan_epoch,
                )
                self.latest_peer_graph = decoded
                if self.current_plan_epoch is None or decoded.plan_epoch >= self.current_plan_epoch:
                    self.current_plan_epoch = decoded.plan_epoch
                    self.selected_plan = self.pending_action_message = None
                    self.last_activated_key = None
                return decoded
            decoded = decode_planning_action(
                message, self.agent_id, current_step, self.config.message_limit
            )
            if decoded.activation_step < current_step:
                raise CommunicationProtocolError("planning action arrived late")
            self._store_plan(decoded)
            return decoded
        except CommunicationProtocolError as error:
            self.last_protocol_error = str(error)
            return None

    def _store_plan(self, candidate: PlanningActionMessage) -> None:
        if (self.last_sent_graph_step == candidate.plan_epoch
                and candidate.source_step <= candidate.plan_epoch):
            return
        if self.is_stale_epoch(candidate.plan_epoch):
            return
        if (self.selected_plan is not None
                and self.selected_plan.plan_epoch == candidate.plan_epoch
                and self.selected_plan.activation_step != candidate.activation_step):
            raise CommunicationProtocolError(
                "same-epoch plans disagree on activation_step"
            )
        if (self.selected_plan is None
                or candidate.plan_epoch > self.selected_plan.plan_epoch
                or (candidate.plan_epoch == self.selected_plan.plan_epoch
                    and candidate.source_step >= self.selected_plan.source_step)):
            self.selected_plan = candidate
            self.current_plan_epoch = candidate.plan_epoch
            self.last_activated_key = None

    def is_stale_epoch(self, plan_epoch: int) -> bool:
        return (self.current_plan_epoch is not None
                and int(plan_epoch) < self.current_plan_epoch)

    def activate(self, current_step: int):
        candidate = self.selected_plan
        if candidate is None:
            return None
        key = (candidate.plan_epoch, candidate.activation_step,
               repr(candidate.actions))
        if candidate.activation_step < current_step:
            self.last_protocol_error = "planning action missed activation_step"
            self.selected_plan = None
            return None
        if candidate.activation_step > current_step or key == self.last_activated_key:
            return None
        self.last_activated_key, self.selected_plan = key, None
        return candidate

    def choose_outgoing(self, can_send: bool = True):
        if not can_send:
            return None
        if self.pending_graph_message is not None:
            message, self.pending_graph_message = self.pending_graph_message, None
            if self.pending_graph_baseline is not None:
                self.last_sent_graph, self.last_sent_graph_step = \
                    self.pending_graph_baseline
            self.pending_graph_baseline = None
            self.pending_action_message = None
            return message
        message, self.pending_action_message = self.pending_action_message, None
        return message

    def schedule_envelope(self, physical_action: dict, outgoing_message):
        can_send = physical_action.get("type") != "ongoing"
        message = outgoing_message if can_send else None
        action = physical_action
        if message is not None and action.get("type") != 7:
            action = {"type": 7}
        return {"physical_action": action, "message": message}
