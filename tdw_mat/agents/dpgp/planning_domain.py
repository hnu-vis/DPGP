"""Deterministic event-driven TDW planning domain.

Planning exposes only Move, Grab, and Put. Move is the sole positive-cost
action; navigation to an object, goal anchor, or unexplored room center is
explicit instead of being hidden inside an interaction.
"""

from dataclasses import dataclass, replace
from itertools import product
import math
from typing import Any, Dict, Optional, Tuple

from .config import GOAL_APPROACH_RADIUS, OBJECT_APPROACH_RADIUS, EXPLORE_CENTER_RADIUS
from .models import DeterministicGraph, Position


MOVE = "move"
GRAB = "grab"
PUT = "put"
ROOM = "room"
OBJECT = "object"
GOAL = "goal"
ACTION_KINDS = (MOVE, GRAB, PUT)
MOVE_TARGET_KINDS = (ROOM, OBJECT, GOAL)
TIME_EPSILON = 1e-12
POSITION_EPSILON = 1e-6
LATENCY_WEIGHT = 0.1
PICKUP_CANDIDATE_COUNT = 2


class PlanningDomainError(ValueError):
    pass


@dataclass(frozen=True)
class HighLevelAction:
    kind: str
    object_index: Optional[int] = None
    target_kind: Optional[str] = None
    target_room_node_id: Optional[int] = None

    def __post_init__(self) -> None:
        if self.kind not in ACTION_KINDS:
            raise ValueError("unknown high-level action kind: {}".format(self.kind))
        if self.kind in (GRAB, PUT):
            if self.object_index is None:
                raise ValueError("{} requires object_index".format(self.kind))
            if self.target_kind is not None or self.target_room_node_id is not None:
                raise ValueError("{} cannot contain a move target".format(self.kind))
        elif self.target_kind not in MOVE_TARGET_KINDS:
            raise ValueError("move requires target_kind")
        elif self.target_room_node_id is None:
            raise ValueError("move requires target_room_node_id")
        elif self.target_kind in (OBJECT, GOAL) and self.object_index is None:
            raise ValueError("{} move requires object_index".format(self.target_kind))
        for name, value in (("object_index", self.object_index),
                            ("target_room_node_id", self.target_room_node_id)):
            if value is not None and (type(value) is not int or value < 0):
                raise ValueError("{} must be a non-negative integer".format(name))

    @classmethod
    def move_room(cls, room_node_id: int,
                  carried_object_index: Optional[int] = None) -> "HighLevelAction":
        return cls(MOVE, carried_object_index, ROOM, room_node_id)

    @classmethod
    def move_object(cls, object_index: int,
                    room_node_id: int) -> "HighLevelAction":
        return cls(MOVE, object_index, OBJECT, room_node_id)

    @classmethod
    def move_goal(cls, object_index: int,
                  room_node_id: int) -> "HighLevelAction":
        return cls(MOVE, object_index, GOAL, room_node_id)

    @classmethod
    def grab(cls, object_index: int) -> "HighLevelAction":
        return cls(GRAB, object_index=object_index)

    @classmethod
    def put(cls, object_index: int) -> "HighLevelAction":
        return cls(PUT, object_index=object_index)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "kind": self.kind,
            "object_index": self.object_index,
            "target_kind": self.target_kind,
            "target_room_node_id": self.target_room_node_id,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "HighLevelAction":
        return cls(
            str(data["kind"]),
            (None if data.get("object_index") is None
             else int(data["object_index"])),
            data.get("target_kind"),
            (None if data.get("target_room_node_id") is None
             else int(data["target_room_node_id"])),
        )


@dataclass(frozen=True)
class JointHighLevelAction:
    agent_0: Optional[HighLevelAction]
    agent_1: Optional[HighLevelAction]

    def __post_init__(self) -> None:
        if any(item is not None and not isinstance(item, HighLevelAction)
               for item in (self.agent_0, self.agent_1)):
            raise TypeError("joint action components must be HighLevelAction or None")

    def to_dict(self) -> dict:
        return {
            "agent_0": self.agent_0.to_dict() if self.agent_0 is not None else None,
            "agent_1": self.agent_1.to_dict() if self.agent_1 is not None else None,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "JointHighLevelAction":
        def action(value):
            return None if value is None else HighLevelAction.from_dict(value)
        return cls(action(data["agent_0"]), action(data["agent_1"]))


@dataclass(frozen=True)
class AgentExecutionState:
    agent_id: int
    room_node_id: int
    position: Position
    held_object_index: Optional[int]
    current_action: Optional[HighLevelAction]
    remaining_time: float
    request: Optional[HighLevelAction] = None


@dataclass(frozen=True)
class ObjectPlanningState:
    graph_index: int
    room_node_id: int
    position: Optional[Position]
    property: int
    holder_agent_ids: Tuple[int, ...]
    finished: bool
    location_assumed: bool


@dataclass(frozen=True)
class PlannerState:
    agents: Tuple[AgentExecutionState, ...]
    objects: Tuple[ObjectPlanningState, ...]
    explored_room_ids: Tuple[int, ...]
    holder: Optional[int] = None

    def is_goal(self) -> bool:
        return all(item.finished for item in self.objects)

    def agent(self, agent_id: int) -> AgentExecutionState:
        return self.agents[agent_id]

    def object(self, graph_index: int) -> ObjectPlanningState:
        for item in self.objects:
            if item.graph_index == graph_index:
                return item
        raise KeyError("unknown object graph index: {}".format(graph_index))


class PlanningDomain:
    """Legal actions, navigation-only costs, transitions, and lower bounds."""

    def __init__(self, graph: DeterministicGraph, navigation=None, commitments=()):
        if not isinstance(graph, DeterministicGraph):
            raise TypeError("PlanningDomain requires DeterministicGraph")
        graph.validate()
        self.graph = graph
        self.navigation = navigation
        self._active_object_indices = (
            {item.graph_index for item in graph.objects}
            if graph.planning_object_indices is None
            else set(graph.planning_object_indices)
        )
        self._object_classes = {o.graph_index: o.class_name for o in graph.objects}
        self.goal_room_node_id = self._validate_graph()
        self._rooms = {item.node_id: item for item in graph.rooms}
        self._room_distances = self._build_room_distances()
        self._route_cache = {}
        self._heuristic_cache = {}
        self.initial_state = self._build_initial_state()
        self.initial_actions = (None, None)
        self.continued_scans = ()
        self._restore_commitments(commitments)

    def _validate_graph(self) -> int:
        goal = self.graph.goal.target_room_node_id
        if goal is None:
            raise PlanningDomainError("goal room must be known")
        if any(item.room_node_id is None for item in self.graph.agents):
            raise PlanningDomainError("both agent rooms must be known")
        rooms = {item.agent_id: item.room_node_id for item in self.graph.agents}
        for item in self.graph.objects:
            if item.finished or item.graph_index not in self._active_object_indices:
                continue
            if item.room_node_id is None or item.property is None:
                raise PlanningDomainError(
                    "unfinished object {} must be deterministic".format(
                        item.graph_index
                    )
                )
            for holder in item.holder_agent_ids:
                if rooms[holder] != item.room_node_id:
                    raise PlanningDomainError(
                        "holder and object {} must share a room".format(
                            item.graph_index
                    )
                )
        held = {index for agent in self.graph.agents
                for index in agent.held_object_indices}
        if not held <= self._active_object_indices:
            raise PlanningDomainError(
                "planning scope must retain every currently held object"
            )
        return goal

    def _build_room_distances(self):
        room_ids = tuple(item.node_id for item in self.graph.rooms)
        distances = {(source, target): (0.0 if source == target else math.inf)
                     for source in room_ids for target in room_ids}
        for edge in self.graph.edges:
            distances[(edge.source, edge.target)] = min(
                distances[(edge.source, edge.target)], float(edge.path_len)
            )
            distances[(edge.target, edge.source)] = min(
                distances[(edge.target, edge.source)], float(edge.path_len)
            )
        for middle in room_ids:
            for source in room_ids:
                for target in room_ids:
                    distances[(source, target)] = min(
                        distances[(source, target)],
                        distances[(source, middle)] + distances[(middle, target)],
                    )
        return distances

    def _build_initial_state(self) -> PlannerState:
        reopened = set(self.graph.reopened_room_ids)
        state = PlannerState(
            agents=tuple(AgentExecutionState(
                item.agent_id,
                item.room_node_id,
                (item.position if item.position is not None
                 else self._rooms[item.room_node_id].center),
                item.held_object_indices[0] if item.held_object_indices else None,
                None,
                0.0,
            ) for item in self.graph.agents),
            objects=tuple(ObjectPlanningState(
                item.graph_index,
                item.room_node_id,
                item.position,
                item.property,
                item.holder_agent_ids,
                item.finished,
                item.location_assumed,
            ) for item in self.graph.objects
              if item.graph_index in self._active_object_indices),
            explored_room_ids=tuple(
                item.node_id for item in self.graph.rooms
                if item.explored and item.node_id not in reopened
            ),
        )
        coop = self.graph.facts.cooperation
        if coop is not None:
            index = next(o.graph_index for o in self.graph.objects if o.object_id == coop.object_id)
            actors = tuple(replace(a, held_object_index=index) for a in state.agents)
            state = replace(state, agents=actors, holder=coop.holder_id)
        return state

    def _restore_commitments(self, commitments):
        """Owner-issued work has identical semantics in both prediction modes."""
        state = self.initial_state
        actors, initial, scans = list(state.agents), [None, None], []
        by_id = {o.object_id: o.graph_index for o in self.graph.objects if o.object_id is not None}
        for item in commitments:
            if item is None or not item["busy"]:
                continue
            i, raw = item["agent_id"], item["action"]
            if raw is None:
                raise PlanningDomainError("ongoing action has no semantic commitment")
            action = HighLevelAction(raw["kind"], by_id.get(raw["object_id"]),
                                     raw["target_kind"], raw["target_room_node_id"])
            if action.target_kind == ROOM:
                room = self._rooms[action.target_room_node_id]
                required = raw.get("required_revisit_count", 0)
                if room.explored and room.revisit_count >= required:
                    continue
                if (not raw.get("scan_started", False)
                        and self.graph.goal.goal_anchor.known
                        and not any(not o.finished and not o.holder_agent_ids and o.room_node_id is None
                                    for o in self.graph.facts.objects)):
                    continue
                scans.append((i, room.node_id, required))
            elif action.object_index is None:
                raise PlanningDomainError("commitment references unknown object")
            if action.target_kind == OBJECT:
                obj = state.object(action.object_index)
                if obj.finished or obj.holder_agent_ids:
                    continue
            if action.target_kind == GOAL and actors[i].held_object_index != action.object_index:
                continue
            if action.kind == GRAB and actors[i].held_object_index == action.object_index:
                continue
            if action.kind == PUT and actors[i].held_object_index != action.object_index:
                continue
            initial[i] = action
            if action.kind == MOVE:
                actors[i] = replace(actors[i], current_action=action,
                                    remaining_time=self.action_duration(state, i, action))
            else:
                # A submitted interaction is a request, not a fictitious
                # completed grab. Joint legality still requires the real pair.
                actors[i] = replace(actors[i], request=action)
                # A committed interaction is not evidence of spatial readiness.
                # Price its remaining approach from the latest fused facts.
                actor = state.agent(state.holder if action.kind == PUT
                                    and state.holder is not None else i)
                target = (self._object_target(state.object(action.object_index))
                          if action.kind == GRAB else self._goal_target())
                radius = OBJECT_APPROACH_RADIUS if action.kind == GRAB else GOAL_APPROACH_RADIUS
                room = None if action.kind == GRAB else self.goal_room_node_id
                if not self._arrived(actor, target, radius, room):
                    move = (HighLevelAction.move_object(action.object_index,
                            state.object(action.object_index).room_node_id)
                            if action.kind == GRAB else HighLevelAction.move_goal(
                                action.object_index, self.goal_room_node_id))
                    initial[i] = move
                    actors[i] = replace(actors[i], current_action=move,
                                        remaining_time=self.action_duration(state, i, move))
        if state.holder is not None and any(initial):
            holder = state.holder
            index = state.agent(holder).held_object_index
            if any(a.object_index != index or (a.kind == MOVE and a.target_kind != GOAL)
                   for a in initial if a is not None):
                raise PlanningDomainError("conflicting active carry commitments")
            # Real phases can differ: a supporter may already request Put
            # while the holder is still approaching. The holder owns the joint
            # semantic continuation; never turn that valid lag into no-plan.
            action = initial[holder] or next(
                (a for a in self._individual_actions(state, holder) if a is not None), None)
            if action is None:
                raise PlanningDomainError("active holder has no executable continuation")
            initial = [action, action]
            if action.kind == MOVE:
                duration = self.action_duration(state, holder, action)
                actors = [replace(a, current_action=action, remaining_time=duration,
                                  request=None) for a in actors]
            else:
                actors = [replace(a, current_action=None, remaining_time=0.0,
                                  request=action) for a in actors]
        self.initial_state = replace(state, agents=tuple(actors))
        self.initial_actions, self.continued_scans = tuple(initial), tuple(scans)

    def _object_target(self, item: ObjectPlanningState) -> Position:
        return (item.position if item.position is not None
                else self._rooms[item.room_node_id].center)

    def _goal_target(self) -> Position:
        anchor = self.graph.goal.goal_anchor
        return (anchor.position if anchor.known and anchor.position is not None
                else self._rooms[self.goal_room_node_id].center)

    def _move_target(self, state: PlannerState,
                     action: HighLevelAction) -> Position:
        if action.target_kind == ROOM:
            return self._rooms[action.target_room_node_id].center
        if action.target_kind == OBJECT:
            return self._object_target(state.object(action.object_index))
        if action.target_kind == GOAL:
            return self._goal_target()
        raise PlanningDomainError("invalid move target")

    @staticmethod
    def _at(left: Position, right: Position) -> bool:
        return math.hypot(left[0] - right[0], left[2] - right[2]) <= POSITION_EPSILON

    def _path_len(self, start: Position, goal: Position,
                  source_room: int, target_room: int) -> float:
        if self._at(start, goal):
            return 0.0
        key = (tuple(start), tuple(goal), source_room, target_room)
        if key in self._route_cache:
            return self._route_cache[key]
        if self.navigation is not None:
            getter = getattr(self.navigation, "get_navigation_path_info", None)
            if not callable(getter):
                raise PlanningDomainError(
                    "navigation must provide get_navigation_path_info"
                )
            info = getter(start, goal) or {}
            value = float(info.get("path_len", -1.0))
            distance = value if math.isfinite(value) and value >= 0 else math.inf
        elif source_room == target_room:
            distance = math.hypot(start[0] - goal[0], start[2] - goal[2])
        else:
            distance = self._room_distances[(source_room, target_room)]
        self._route_cache[key] = distance
        return distance

    def _move_route(self, state, agent_id, action):
        """Use the executable arrival region, including its actual endpoint."""
        agent = state.agent(state.holder if state.holder is not None
                            and action.target_kind == GOAL else agent_id)
        target = self._move_target(state, action)
        if self.navigation is None:
            return {"path_len": self._path_len(agent.position, target, agent.room_node_id,
                                                action.target_room_node_id), "endpoint": target}
        radius = {ROOM: EXPLORE_CENTER_RADIUS, OBJECT: OBJECT_APPROACH_RADIUS,
                  GOAL: GOAL_APPROACH_RADIUS}[action.target_kind]
        room = None if action.target_kind == OBJECT else self._rooms[action.target_room_node_id].room_name
        return self._navigation_route(agent.position, target, radius, room)

    def _navigation_route(self, start, target, radius, room):
        key = ("region", tuple(start), tuple(target), radius, room)
        if key not in self._route_cache:
            self._route_cache[key] = self.navigation.get_navigation_path_info(
                start, target, radius=radius, room_name=room)
        return self._route_cache[key]

    def _arrived(self, agent, target, radius, room=None):
        if room is not None and agent.room_node_id != room:
            return False
        if self.navigation is None:
            return self._at(agent.position, target)
        return math.hypot(agent.position[0] - target[0], agent.position[2] - target[2]) < radius

    def action_duration(self, state: PlannerState, agent_id: int,
                        action: HighLevelAction) -> float:
        if action.kind in (GRAB, PUT):
            return 0.0
        duration = float(self._move_route(state, agent_id, action).get("path_len", -1))
        if not math.isfinite(duration) or duration < 0:
            raise PlanningDomainError("move target is unreachable")
        return duration

    def legal_joint_actions(self, state: PlannerState) -> Tuple[JointHighLevelAction, ...]:
        if state.holder is not None:
            if any(a.current_action is not None for a in state.agents):
                return (JointHighLevelAction(None, None),)
            return tuple(JointHighLevelAction(a, a)
                         for a in self._individual_actions(state, state.holder)
                         if a is not None)
        individual = tuple(self._individual_actions(state, agent_id)
                           for agent_id in (0, 1))
        actions = tuple(
            JointHighLevelAction(pair[0], pair[1]) for pair in product(*individual)
            if self._is_joint_legal(state, pair)
        )
        return tuple(sorted(actions, key=lambda item: self._joint_key(item)))

    def _reachable(self, state: PlannerState, agent_id: int,
                   action: HighLevelAction) -> bool:
        try:
            return math.isfinite(self.action_duration(state, agent_id, action))
        except PlanningDomainError:
            return False

    def _individual_actions(self, state: PlannerState,
                            agent_id: int) -> Tuple[Optional[HighLevelAction], ...]:
        agent = state.agent(agent_id)
        if agent.current_action is not None:
            return (None,)
        if state.holder is not None:
            agent = state.agent(state.holder)
        actions = []
        if agent.request is not None:
            return (agent.request, None)
        if agent.held_object_index is not None:
            item = state.object(agent.held_object_index)
            ready = (item.property == 1 and item.holder_agent_ids == (agent_id,)) or (
                item.property == 2 and state.holder is not None
            )
            if ready:
                target = self._goal_target()
                if self._arrived(agent, target, GOAL_APPROACH_RADIUS, self.goal_room_node_id):
                    return (HighLevelAction.put(item.graph_index),)
                else:
                    move = HighLevelAction.move_goal(
                        item.graph_index, self.goal_room_node_id
                    )
                    if self._reachable(state, agent_id, move):
                        actions.append(move)
        else:
            explored = set(state.explored_room_ids)
            candidates = self._pickup_candidate_indices(state)
            peer = state.agent(1 - agent_id)
            reserved = peer.current_action or peer.request
            requests = [a.request for a in state.agents if a.request is not None
                        and a.request.kind == GRAB]
            if requests:
                candidates = {requests[0].object_index}
            for item in state.objects:
                if item.graph_index not in candidates:
                    continue
                if item.finished or (item.holder_agent_ids and item.property == 1):
                    continue
                if (item.property == 1 and reserved is not None
                        and reserved.object_index == item.graph_index
                        and (reserved.kind == GRAB or reserved.target_kind == OBJECT)):
                    continue
                if item.property == 2 and agent_id in item.holder_agent_ids:
                    continue
                needs_observation = self._needs_observation(item, explored)
                if needs_observation:
                    move = HighLevelAction.move_room(item.room_node_id)
                elif self._arrived(agent, self._object_target(item), OBJECT_APPROACH_RADIUS):
                    if not item.holder_agent_ids:
                        actions.append(HighLevelAction.grab(item.graph_index))
                    continue
                else:
                    move = HighLevelAction.move_object(
                        item.graph_index, item.room_node_id
                    )
                if self._reachable(state, agent_id, move):
                    actions.append(move)
        actions.append(None)
        return tuple(sorted(set(actions), key=self._optional_action_key))

    def _pickup_candidate_indices(self, state: PlannerState) -> set:
        """Keep two rolling task candidates per class, shared by both agents."""
        explored = set(state.explored_room_ids)
        available_agents = tuple(
            agent for agent in state.agents
            if agent.held_object_index is None and agent.current_action is None
        )
        ranked = {}
        for item in state.objects:
            if item.finished or (item.holder_agent_ids and item.property == 1):
                continue
            target = (self._rooms[item.room_node_id].center
                      if self._needs_observation(item, explored)
                      else self._object_target(item))
            if self.navigation is not None:
                move = (HighLevelAction.move_room(item.room_node_id)
                        if self._needs_observation(item, explored) else
                        HighLevelAction.move_object(item.graph_index, item.room_node_id))
                distances = []
                for agent in available_agents:
                    approach = self._move_route(state, agent.agent_id, move)
                    endpoint = approach.get("endpoint")
                    if endpoint is None:
                        continue
                    carry = self._navigation_route(endpoint, self._goal_target(),
                        GOAL_APPROACH_RADIUS, self._rooms[self.goal_room_node_id].room_name)
                    if carry.get("endpoint") is not None:
                        distances.append(approach["path_len"] + carry["path_len"])
            else:
                distances = [self._path_len(
                    agent.position, target, agent.room_node_id, item.room_node_id
                ) + self._path_len(target, self._goal_target(), item.room_node_id,
                                  self.goal_room_node_id) for agent in available_agents]
            rank = (
                0 if item.holder_agent_ids else 1,
                min(distances, default=math.inf),
                item.graph_index,
            )
            name = self._object_classes[item.graph_index]
            ranked.setdefault(name, []).append((rank, item.graph_index))
        return {index for rows in ranked.values()
                for _, index in sorted(rows)[:PICKUP_CANDIDATE_COUNT]}

    def _needs_observation(self, item, explored) -> bool:
        return bool(item.location_assumed
                    and item.room_node_id not in explored)

    def _requires_joint_pair(self, state: PlannerState,
                             action: Optional[HighLevelAction]) -> bool:
        if action is None or action.object_index is None:
            return False
        item = state.object(action.object_index)
        return item.property == 2 and (
            action.kind in (GRAB, PUT)
            or (action.kind == MOVE and action.target_kind == GOAL)
        )

    def _is_joint_legal(self, state: PlannerState, actions) -> bool:
        if any(a is not None and (actor.current_action is not None
               or (actor.request is not None and a != actor.request))
               for actor, a in zip(state.agents, actions)):
            return False
        if state.holder is not None and actions[0] != actions[1]:
            return False
        if actions == (None, None):
            return any(item.current_action is not None for item in state.agents)
        for action in actions:
            if self._requires_joint_pair(state, action) and actions[0] != actions[1]:
                return False
        effective = [actor.current_action or actor.request or action
                     for actor, action in zip(state.agents, actions)]
        scans = [a.target_room_node_id for a in effective if a is not None and a.target_kind == ROOM]
        if len(scans) != len(set(scans)):
            return False
        ordinary = [a.object_index for a in effective if a is not None
                    and (a.kind == GRAB or a.target_kind == OBJECT)
                    and state.object(a.object_index).property == 1]
        if len(ordinary) != len(set(ordinary)):
            return False
        return True

    @staticmethod
    def _optional_action_key(action: Optional[HighLevelAction]) -> tuple:
        if action is None:
            return (4, -1, "", -1)
        category = {PUT: 0, GRAB: 1, MOVE: 2}[action.kind]
        return (category,
                -1 if action.object_index is None else action.object_index,
                action.target_kind or "",
                -1 if action.target_room_node_id is None
                else action.target_room_node_id)

    def _joint_key(self, action: JointHighLevelAction) -> tuple:
        return (self._optional_action_key(action.agent_0),
                self._optional_action_key(action.agent_1))

    def transition(self, state: PlannerState,
                   joint_action: JointHighLevelAction) -> Tuple[PlannerState, float]:
        if joint_action not in self.legal_joint_actions(state):
            raise PlanningDomainError("joint action is illegal in current state")
        return self.advance(state, joint_action)

    def advance(self, state: PlannerState,
                joint_action: JointHighLevelAction) -> Tuple[PlannerState, float]:
        actions = (joint_action.agent_0, joint_action.agent_1)
        agents = list(state.agents)
        for agent_id, action in enumerate(actions):
            agent = agents[agent_id]
            if agent.current_action is not None or action is None:
                continue
            agents[agent_id] = replace(
                agent,
                current_action=action,
                remaining_time=self.action_duration(state, agent_id, action),
            )
        started = replace(state, agents=tuple(agents))
        if any(item.current_action is not None
               and item.remaining_time <= TIME_EPSILON for item in started.agents):
            return self._complete_ready(started), 0.0
        times = tuple(item.remaining_time for item in started.agents
                      if item.current_action is not None)
        if not times:
            raise PlanningDomainError("joint action makes no progress")
        elapsed = min(times)
        advanced = replace(started, agents=tuple(
            replace(item, remaining_time=self._normalized(
                item.remaining_time - elapsed
            )) if item.current_action is not None else item
            for item in started.agents
        ))
        return self._complete_ready(advanced), elapsed

    @staticmethod
    def _normalized(value: float) -> float:
        return 0.0 if abs(value) <= TIME_EPSILON else float(value)

    def _complete_ready(self, state: PlannerState) -> PlannerState:
        agents = {item.agent_id: item for item in state.agents}
        objects = {item.graph_index: item for item in state.objects}
        explored = set(state.explored_room_ids)
        holder = state.holder
        put_objects = set()
        ready = tuple(item for item in state.agents
                      if item.current_action is not None
                      and item.remaining_time <= TIME_EPSILON)
        for original in ready:
            agent_id, action = original.agent_id, original.current_action
            agents[agent_id] = replace(agents[agent_id], current_action=None,
                                       remaining_time=0.0,
                                       request=original.request if action.kind == MOVE else None)
            if action.kind == MOVE:
                endpoint = self._move_route(state, agent_id, action).get("endpoint")
                if endpoint is None:
                    raise PlanningDomainError("navigation route has no endpoint")
                endpoint = tuple(endpoint)
                agents[agent_id] = replace(
                    agents[agent_id], room_node_id=action.target_room_node_id,
                    position=endpoint,
                )
                explored.add(action.target_room_node_id)
                if original.held_object_index is not None and holder in (None, agent_id):
                    item = objects[original.held_object_index]
                    objects[item.graph_index] = replace(
                        item, room_node_id=action.target_room_node_id,
                        position=endpoint,
                    )
            elif action.kind == GRAB:
                item = objects[action.object_index]
                objects[item.graph_index] = replace(
                    item,
                    position=agents[holder if holder is not None else agent_id].position,
                    holder_agent_ids=(agent_id,),
                )
                agents[agent_id] = replace(
                    agents[agent_id], held_object_index=item.graph_index
                )
            elif action.kind == PUT:
                item = objects[action.object_index]
                objects[item.graph_index] = replace(
                    item,
                    holder_agent_ids=tuple(
                        value for value in item.holder_agent_ids
                        if value != agent_id
                    ),
                    room_node_id=self.goal_room_node_id,
                    position=agents[holder if holder is not None else agent_id].position,
                )
                agents[agent_id] = replace(
                    agents[agent_id], held_object_index=None
                )
                put_objects.add(item.graph_index)
        grabs = {a.current_action.object_index for a in ready
                 if a.current_action.kind == GRAB
                 and objects[a.current_action.object_index].property == 2}
        for index in grabs:
            target = self._object_target(state.object(index))
            holder = min((0, 1), key=lambda i: (
                math.hypot(state.agent(i).position[0] - target[0],
                           state.agent(i).position[2] - target[2]), i))
            objects[index] = replace(objects[index], holder_agent_ids=(holder,),
                                     position=agents[holder].position,
                                     room_node_id=agents[holder].room_node_id)
        for index in put_objects:
            if not objects[index].holder_agent_ids:
                objects[index] = replace(objects[index], finished=True)
        if put_objects and holder is not None:
            holder = None
        return PlannerState(
            tuple(agents[index] for index in (0, 1)),
            tuple(objects[index] for index in sorted(objects)),
            tuple(sorted(explored)),
            holder,
        )

    def transition_cost(self, state: PlannerState, elapsed: float) -> float:
        """Navigation makespan plus a small unfinished-object latency term."""
        if elapsed <= 0:
            return 0.0
        active = {item.graph_index for item in state.objects}
        unfinished = sum(not item.finished for item in state.objects) + sum(
            not item.finished and item.graph_index not in active
            for item in self.graph.objects
        )
        total = max(1, len(self.graph.objects))
        return elapsed * (1.0 + LATENCY_WEIGHT * unfinished / total)

    def heuristic(self, state: PlannerState) -> float:
        key = self.state_key(state)
        if key not in self._heuristic_cache:
            self._heuristic_cache[key] = self._estimate_navigation_bound(state)
        return self._heuristic_cache[key]

    @staticmethod
    def state_key(state: PlannerState) -> tuple:
        return (
            tuple((item.room_node_id, item.position, item.held_object_index,
                   item.current_action, item.remaining_time, item.request)
                  for item in state.agents),
            tuple((item.room_node_id, item.position, item.holder_agent_ids,
                   item.finished) for item in state.objects),
            state.explored_room_ids,
            state.holder,
        )

    def _region_lower_bound(self, start, goal, radius):
        # Point-path costs can overestimate a region goal and invalidate A*
        # pruning. Relax all obstacles and quantization for a geometric bound.
        config = getattr(self.navigation, "config", None)
        if config is None:
            return 0.0
        distance = math.hypot(start[0] - goal[0], start[2] - goal[2])
        return max(0.0, distance - radius - 2 * config.cell_size) / config.cell_size

    def _estimate_navigation_bound(self, state: PlannerState) -> float:
        """Optimistic per-object completion bound.

        Capacity, reservations, other jobs, and current commitments are relaxed.
        This keeps the value finite when both agents currently hold objects and
        makes it safe for A* branch-and-bound pruning.
        """
        bounds = []
        transport_bounds = []
        goal = self._goal_target()
        for item in state.objects:
            if item.finished:
                continue
            if item.holder_agent_ids:
                holder_bounds = []
                for holder in item.holder_agent_ids:
                    agent = state.agent(holder)
                    holder_bounds.append(
                        agent.remaining_time if agent.current_action is not None
                        else (self._region_lower_bound(agent.position, goal, GOAL_APPROACH_RADIUS)
                              if self.navigation is not None else self._path_len(
                                  agent.position, goal, agent.room_node_id, self.goal_room_node_id))
                    )
                bound = max(holder_bounds, default=0.0)
                bounds.append(bound)
                transport_bounds.append(bound)
                continue
            target = self._object_target(item)
            approaches = []
            for agent in state.agents:
                if agent.current_action is not None:
                    # Its stored position is the departure point, not its
                    # current progress. Relax it to an immediately free agent.
                    approaches.append(0.0)
                    continue
                approaches.append(
                    self._region_lower_bound(agent.position, target, EXPLORE_CENTER_RADIUS)
                    if self.navigation is not None else self._path_len(
                        agent.position, target, agent.room_node_id, item.room_node_id))
            carry = (self._region_lower_bound(target, goal, EXPLORE_CENTER_RADIUS + GOAL_APPROACH_RADIUS)
                     if self.navigation is not None else self._path_len(
                         target, goal, item.room_node_id, self.goal_room_node_id))
            bounds.append(min(approaches, default=math.inf) + carry)
            transport_bounds.append(carry)
        # Transport work is disjoint across single-capacity jobs; initial
        # approach distances can overlap and must not be summed as work.
        return max(max(bounds, default=0.0), sum(transport_bounds) / 2.0)
