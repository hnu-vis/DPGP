"""Immutable factual values shared by TDW DPGP components.

``FactGraph`` contains only reset facts, local public observations, public
feedback, navigation measurements, or facts received through the formal
communication channel.  LLM assumptions are deliberately not representable
as a separate field in this stage.
"""

from collections import Counter
from dataclasses import dataclass
import math
from typing import Any, Dict, Optional, Tuple



Position = Tuple[float, float, float]
ASSUMPTION_SOURCES = frozenset(("llm_room_prediction", "temporary_assumption"))


def position(value: Any) -> Optional[Position]:
    if value is None:
        return None
    if isinstance(value, (str, bytes)) or not hasattr(value, "__len__") or len(value) != 3:
        raise ValueError("position must contain exactly three coordinates")
    result = tuple(float(item) for item in value)
    if not all(math.isfinite(item) for item in result):
        raise ValueError("position coordinates must be finite")
    return result  # type: ignore[return-value]


def _sorted_unique(values: tuple, name: str) -> None:
    if values != tuple(sorted(values)) or len(values) != len(set(values)):
        raise ValueError("{} must be sorted and unique".format(name))


@dataclass(frozen=True)
class RoomFact:
    node_id: int
    room_name: str
    center: Position
    explored: bool = False
    revisit_count: int = 0


@dataclass(frozen=True)
class NavigationEdgeFact:
    source: int
    target: int
    path_len: float
    known_grid_count: int
    total_grid_count: int
    belief: float
    cost: float


@dataclass(frozen=True)
class GoalAnchorFact:
    known: bool = False
    object_id: Optional[int] = None
    position: Optional[Position] = None
    room_node_id: Optional[int] = None
    last_seen_frame: Optional[int] = None


@dataclass(frozen=True)
class GoalFact:
    target_room_node_id: Optional[int]
    required_objects: Tuple[Tuple[str, int], ...]
    goal_anchor: GoalAnchorFact


@dataclass(frozen=True)
class ObjectFact:
    graph_index: int
    class_name: str
    object_id: Optional[int] = None
    room_node_id: Optional[int] = None
    position: Optional[Position] = None
    property: Optional[int] = None
    holder_agent_ids: Tuple[int, ...] = ()
    finished: bool = False
    last_seen_frame: Optional[int] = None
    information_source: Optional[str] = None


@dataclass(frozen=True)
class AgentFact:
    agent_id: int
    room_node_id: Optional[int] = None
    held_object_indices: Tuple[int, ...] = ()
    last_seen_frame: Optional[int] = None
    position: Optional[Position] = None


@dataclass(frozen=True)
class InspectionFact:
    """One completed TDW scan program, not a hard whole-room exclusion."""

    inspection_id: str
    scope_key: str
    room_node_id: int
    viewpoint: Position
    scan_pass: int
    start_frame: int
    end_frame: int
    observed_target_ids: Tuple[int, ...]
    scan_completed: bool
    source_agent: int


@dataclass(frozen=True)
class CooperationFact:
    object_id: int
    holder_id: int
    supporter_id: int


@dataclass(frozen=True)
class FactGraph:
    frame: int
    rooms: Tuple[RoomFact, ...]
    edges: Tuple[NavigationEdgeFact, ...]
    goal: GoalFact
    objects: Tuple[ObjectFact, ...]
    agents: Tuple[AgentFact, ...]
    inspections: Tuple[InspectionFact, ...] = ()
    cooperation: Optional[CooperationFact] = None

    def validate(self) -> None:
        if type(self.frame) is not int or self.frame < 0:
            raise ValueError("frame must be a non-negative integer")

        room_ids = tuple(item.node_id for item in self.rooms)
        _sorted_unique(room_ids, "room ids")
        room_set = set(room_ids)
        for room in self.rooms:
            if type(room.node_id) is not int or not room.room_name:
                raise ValueError("rooms require integer ids and names")
            position(room.center)
            if type(room.explored) is not bool:
                raise ValueError("room explored must be boolean")
            if type(room.revisit_count) is not int or room.revisit_count < 0:
                raise ValueError("room revisit_count must be a non-negative integer")

        edge_keys = tuple((item.source, item.target) for item in self.edges)
        _sorted_unique(edge_keys, "navigation edges")
        for edge in self.edges:
            if edge.source >= edge.target or not {edge.source, edge.target} <= room_set:
                raise ValueError("navigation edge references invalid rooms")
            if not all(math.isfinite(value) for value in (edge.path_len, edge.belief, edge.cost)):
                raise ValueError("navigation edge values must be finite")
            if edge.path_len < 0 or edge.cost < 0 or edge.total_grid_count <= 0:
                raise ValueError("FactGraph cannot contain unavailable edges")
            if not 0 <= edge.known_grid_count <= edge.total_grid_count:
                raise ValueError("invalid navigation grid counts")
            expected = float(edge.known_grid_count) / edge.total_grid_count
            if not math.isclose(edge.belief, expected, rel_tol=1e-9, abs_tol=1e-12):
                raise ValueError("edge belief must match grid counts")
            if not math.isclose(edge.cost, edge.path_len, rel_tol=1e-9, abs_tol=1e-12):
                raise ValueError("edge cost must equal path_len")

        required_names = tuple(name for name, _ in self.goal.required_objects)
        _sorted_unique(required_names, "required object classes")
        if any(not name or type(count) is not int or count <= 0
               for name, count in self.goal.required_objects):
            raise ValueError("goal counts must be positive integers")
        if (self.goal.target_room_node_id is not None
                and self.goal.target_room_node_id not in room_set):
            raise ValueError("goal target room is unknown")
        anchor = self.goal.goal_anchor
        position(anchor.position)
        if anchor.known and (anchor.object_id is None or anchor.position is None):
            raise ValueError("known goal anchor requires id and position")
        if anchor.room_node_id is not None and anchor.room_node_id not in room_set:
            raise ValueError("goal anchor room is unknown")
        if anchor.last_seen_frame is not None and anchor.last_seen_frame < 0:
            raise ValueError("goal anchor frame must be non-negative")

        indices = tuple(item.graph_index for item in self.objects)
        _sorted_unique(indices, "object graph indices")
        index_set = set(indices)
        object_ids = [item.object_id for item in self.objects if item.object_id is not None]
        if len(object_ids) != len(set(object_ids)):
            raise ValueError("runtime object ids must be unique")
        if Counter(item.class_name for item in self.objects) != Counter(dict(self.goal.required_objects)):
            raise ValueError("object slots must exactly match the task goal")

        agent_ids = tuple(item.agent_id for item in self.agents)
        if agent_ids != (0, 1):
            raise ValueError("TDW FactGraph requires sorted agents 0 and 1")
        held = {0: set(), 1: set()}
        class_properties: Dict[str, int] = {}
        for item in self.objects:
            if not item.class_name or type(item.graph_index) is not int:
                raise ValueError("invalid object identity")
            if item.room_node_id is not None and item.room_node_id not in room_set:
                raise ValueError("object room is unknown")
            position(item.position)
            if item.property not in (None, 1, 2):
                raise ValueError("object property must be None, 1, or 2")
            if item.property is not None:
                known = class_properties.setdefault(item.class_name, item.property)
                if known != item.property:
                    raise ValueError("one class cannot have conflicting properties")
            _sorted_unique(item.holder_agent_ids, "object holders")
            if any(agent_id not in (0, 1) for agent_id in item.holder_agent_ids):
                raise ValueError("object holder is unknown")
            if len(item.holder_agent_ids) > 1:
                raise ValueError("cooperative objects also have only one physical holder")
            if item.last_seen_frame is not None and item.last_seen_frame < 0:
                raise ValueError("object frame must be non-negative")
            if item.information_source in ASSUMPTION_SOURCES:
                raise ValueError("FactGraph cannot contain temporary assumptions")
            for agent_id in item.holder_agent_ids:
                held[agent_id].add(item.graph_index)

        for agent in self.agents:
            if agent.room_node_id is not None and agent.room_node_id not in room_set:
                raise ValueError("agent room is unknown")
            _sorted_unique(agent.held_object_indices, "agent held objects")
            if len(agent.held_object_indices) > 1:
                raise ValueError("an agent cannot hold more than one target")
            if any(index not in index_set for index in agent.held_object_indices):
                raise ValueError("agent references an unknown object")
            if set(agent.held_object_indices) != held[agent.agent_id]:
                raise ValueError("object holders and agent held objects disagree")
            if agent.last_seen_frame is not None and agent.last_seen_frame < 0:
                raise ValueError("agent frame must be non-negative")
            position(agent.position)

        if self.cooperation is not None:
            coop = self.cooperation
            if (any(type(value) is not int for value in (coop.object_id, coop.holder_id, coop.supporter_id))
                    or {coop.holder_id, coop.supporter_id} != {0, 1}):
                raise ValueError("cooperation requires distinct agents 0 and 1")
            obj = next((o for o in self.objects if o.object_id == coop.object_id), None)
            if obj is None or obj.property != 2 or obj.holder_agent_ids != (coop.holder_id,):
                raise ValueError("cooperation must reference its physical holder")
            if self.agents[coop.supporter_id].held_object_indices:
                raise ValueError("cooperative supporter cannot physically hold an object")

        inspection_ids = tuple(item.inspection_id for item in self.inspections)
        _sorted_unique(inspection_ids, "inspection ids")
        known_object_ids = set(object_ids)
        for item in self.inspections:
            if not item.inspection_id or not item.scope_key:
                raise ValueError("inspection identifiers cannot be empty")
            if item.room_node_id not in room_set:
                raise ValueError("inspection room is unknown")
            position(item.viewpoint)
            if type(item.scan_pass) is not int or item.scan_pass < 0:
                raise ValueError("inspection scan_pass must be non-negative")
            if (type(item.start_frame) is not int or item.start_frame < 0
                    or type(item.end_frame) is not int
                    or item.end_frame < item.start_frame):
                raise ValueError("inspection frames are invalid")
            _sorted_unique(item.observed_target_ids,
                           "inspection observed target ids")
            if not set(item.observed_target_ids) <= known_object_ids:
                raise ValueError("inspection references an unknown target")
            if type(item.scan_completed) is not bool:
                raise ValueError("inspection scan_completed must be boolean")
            if item.source_agent not in (0, 1):
                raise ValueError("inspection source agent is unknown")

    def to_dict(self) -> dict:
        self.validate()
        return {
            "frame": self.frame,
            "cooperation": (None if self.cooperation is None else {
                "object_id": self.cooperation.object_id,
                "holder_id": self.cooperation.holder_id,
                "supporter_id": self.cooperation.supporter_id}),
            "rooms": [
                {"node_id": item.node_id, "room_name": item.room_name,
                 "center": list(item.center), "explored": item.explored,
                 "revisit_count": item.revisit_count}
                for item in self.rooms
            ],
            "edges": [
                {"source": item.source, "target": item.target,
                 "path_len": item.path_len,
                 "known_grid_count": item.known_grid_count,
                 "total_grid_count": item.total_grid_count,
                 "belief": item.belief, "cost": item.cost}
                for item in self.edges
            ],
            "goal": {
                "target_room_node_id": self.goal.target_room_node_id,
                "required_objects": dict(self.goal.required_objects),
                "goal_anchor": {
                    "known": self.goal.goal_anchor.known,
                    "object_id": self.goal.goal_anchor.object_id,
                    "position": (list(self.goal.goal_anchor.position)
                                 if self.goal.goal_anchor.position is not None else None),
                    "room_node_id": self.goal.goal_anchor.room_node_id,
                    "last_seen_frame": self.goal.goal_anchor.last_seen_frame,
                },
            },
            "objects": [
                {"graph_index": item.graph_index, "class_name": item.class_name,
                 "object_id": item.object_id, "room_node_id": item.room_node_id,
                 "position": list(item.position) if item.position is not None else None,
                 "property": item.property,
                 "holder_agent_ids": list(item.holder_agent_ids),
                 "finished": item.finished,
                 "last_seen_frame": item.last_seen_frame,
                 "information_source": item.information_source}
                for item in self.objects
            ],
            "agents": [
                {"agent_id": item.agent_id, "room_node_id": item.room_node_id,
                 "held_object_indices": list(item.held_object_indices),
                 "last_seen_frame": item.last_seen_frame,
                 "position": (list(item.position)
                              if item.position is not None else None)}
                for item in self.agents
            ],
            "inspections": [
                {"inspection_id": item.inspection_id,
                 "scope_key": item.scope_key,
                 "room_node_id": item.room_node_id,
                 "viewpoint": list(item.viewpoint),
                 "scan_pass": item.scan_pass,
                 "start_frame": item.start_frame,
                 "end_frame": item.end_frame,
                 "observed_target_ids": list(item.observed_target_ids),
                 "scan_completed": item.scan_completed,
                 "source_agent": item.source_agent}
                for item in self.inspections
            ],
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "FactGraph":
        rooms = tuple(sorted((
            RoomFact(int(item["node_id"]), str(item["room_name"]),
                     position(item["center"]), bool(item["explored"]),
                     int(item.get("revisit_count", 0)))
            for item in data["rooms"]
        ), key=lambda item: item.node_id))
        edges = tuple(sorted((
            NavigationEdgeFact(
                int(item["source"]), int(item["target"]), float(item["path_len"]),
                int(item["known_grid_count"]), int(item["total_grid_count"]),
                float(item["belief"]), float(item["cost"]),
            ) for item in data["edges"]
        ), key=lambda item: (item.source, item.target)))
        raw_goal = data["goal"]
        raw_anchor = raw_goal["goal_anchor"]
        goal = GoalFact(
            int(raw_goal["target_room_node_id"])
            if raw_goal.get("target_room_node_id") is not None else None,
            tuple(sorted((str(name), int(count))
                         for name, count in raw_goal["required_objects"].items())),
            GoalAnchorFact(
                bool(raw_anchor["known"]),
                int(raw_anchor["object_id"])
                if raw_anchor.get("object_id") is not None else None,
                position(raw_anchor.get("position")),
                int(raw_anchor["room_node_id"])
                if raw_anchor.get("room_node_id") is not None else None,
                int(raw_anchor["last_seen_frame"])
                if raw_anchor.get("last_seen_frame") is not None else None,
            ),
        )
        objects = tuple(sorted((
            ObjectFact(
                int(item["graph_index"]), str(item["class_name"]),
                int(item["object_id"]) if item.get("object_id") is not None else None,
                int(item["room_node_id"]) if item.get("room_node_id") is not None else None,
                position(item.get("position")),
                int(item["property"]) if item.get("property") is not None else None,
                tuple(sorted(int(value) for value in item["holder_agent_ids"])),
                bool(item["finished"]),
                int(item["last_seen_frame"])
                if item.get("last_seen_frame") is not None else None,
                item.get("information_source"),
            ) for item in data["objects"]
        ), key=lambda item: item.graph_index))
        agents = tuple(sorted((
            AgentFact(
                int(item["agent_id"]),
                int(item["room_node_id"]) if item.get("room_node_id") is not None else None,
                tuple(sorted(int(value) for value in item["held_object_indices"])),
                int(item["last_seen_frame"])
                if item.get("last_seen_frame") is not None else None,
                position(item.get("position")),
            ) for item in data["agents"]
        ), key=lambda item: item.agent_id))
        inspections = tuple(sorted((
            InspectionFact(
                str(item["inspection_id"]), str(item["scope_key"]),
                int(item["room_node_id"]), position(item["viewpoint"]),
                int(item["scan_pass"]), int(item["start_frame"]),
                int(item["end_frame"]),
                tuple(sorted(int(value)
                             for value in item["observed_target_ids"])),
                bool(item["scan_completed"]), int(item["source_agent"]),
            ) for item in data.get("inspections", [])
        ), key=lambda item: item.inspection_id))
        coop = data["cooperation"]
        cooperation = None if coop is None else CooperationFact(**coop)
        graph = cls(int(data["frame"]), rooms,
                    edges, goal, objects, agents, inspections, cooperation)
        graph.validate()
        return graph


@dataclass(frozen=True)
class ObjectAssumption:
    """Temporary values needed to make one object deterministic for planning."""

    graph_index: int
    room_node_id: Optional[int] = None
    ordinary_property: bool = False


@dataclass(frozen=True)
class DeterministicObject:
    graph_index: int
    class_name: str
    object_id: Optional[int]
    room_node_id: Optional[int]
    position: Optional[Position]
    property: Optional[int]
    holder_agent_ids: Tuple[int, ...]
    finished: bool
    last_seen_frame: Optional[int]
    information_source: Optional[str]
    location_assumed: bool
    property_assumed: bool


@dataclass(frozen=True)
class DeterministicGraph:
    """One temporary, fully specified planning view over immutable facts.

    The factual graph is retained verbatim. LLM room predictions and the
    ordinary-property default live only in ``assumptions`` and therefore
    cannot be accepted by ``MethodMemory.apply_fused_facts``.
    """

    facts: FactGraph
    assumptions: Tuple[ObjectAssumption, ...] = ()
    reopened_room_ids: Tuple[int, ...] = ()
    planning_object_indices: Optional[Tuple[int, ...]] = None

    @property
    def frame(self) -> int:
        return self.facts.frame

    @property
    def rooms(self) -> Tuple[RoomFact, ...]:
        return self.facts.rooms

    @property
    def edges(self) -> Tuple[NavigationEdgeFact, ...]:
        return self.facts.edges

    @property
    def goal(self) -> GoalFact:
        return self.facts.goal

    @property
    def agents(self) -> Tuple[AgentFact, ...]:
        return self.facts.agents

    @property
    def objects(self) -> Tuple[DeterministicObject, ...]:
        assumptions = {item.graph_index: item for item in self.assumptions}
        result = []
        for item in self.facts.objects:
            assumption = assumptions.get(item.graph_index)
            location_assumed = (
                assumption is not None and assumption.room_node_id is not None
            )
            property_assumed = bool(
                assumption is not None and assumption.ordinary_property
            )
            result.append(DeterministicObject(
                item.graph_index,
                item.class_name,
                item.object_id,
                assumption.room_node_id if location_assumed else item.room_node_id,
                item.position,
                1 if property_assumed else item.property,
                item.holder_agent_ids,
                item.finished,
                item.last_seen_frame,
                item.information_source,
                location_assumed,
                property_assumed,
            ))
        return tuple(result)

    def validate(self) -> None:
        self.facts.validate()
        room_ids = tuple(item.node_id for item in self.facts.rooms)
        _sorted_unique(self.reopened_room_ids, "reopened room ids")
        if not set(self.reopened_room_ids) <= set(room_ids):
            raise ValueError("reopened rooms must exist in the factual graph")
        indices = tuple(item.graph_index for item in self.assumptions)
        _sorted_unique(indices, "assumption object indices")
        facts = {item.graph_index: item for item in self.facts.objects}
        if self.planning_object_indices is not None:
            _sorted_unique(self.planning_object_indices,
                           "planning object indices")
            if not set(self.planning_object_indices) <= set(facts):
                raise ValueError("planning scope references an unknown object")
        planned = (set(facts) if self.planning_object_indices is None
                   else set(self.planning_object_indices))
        for assumption in self.assumptions:
            if assumption.graph_index not in facts:
                raise ValueError("assumption references an unknown object")
            if type(assumption.ordinary_property) is not bool:
                raise ValueError("ordinary_property must be boolean")
            if assumption.room_node_id is None and not assumption.ordinary_property:
                raise ValueError("empty object assumption")
            item = facts[assumption.graph_index]
            if assumption.room_node_id is not None:
                if assumption.room_node_id not in room_ids:
                    raise ValueError("room prediction references an unknown room")
                if item.finished or item.holder_agent_ids or item.position is not None:
                    raise ValueError("only unresolved object locations may be predicted")
            if assumption.ordinary_property and (item.finished or item.property is not None):
                raise ValueError("only unresolved object properties may be defaulted")
        for item in self.objects:
            if item.finished or item.graph_index not in planned:
                continue
            if item.room_node_id is None:
                raise ValueError("deterministic graph has an unresolved object location")
            if item.property not in (1, 2):
                raise ValueError("deterministic graph has an unresolved object property")
            if len(item.holder_agent_ids) > 1 and item.property != 2:
                raise ValueError("cooperatively held object must remain cooperative")

    @classmethod
    def from_facts(cls, facts: FactGraph, room_predictions: Dict[int, int],
                   reopen_all_rooms: bool = False) -> "DeterministicGraph":
        facts.validate()
        expected = {
            item.graph_index for item in facts.objects
            if not item.finished and not item.holder_agent_ids
            and item.position is None
        }
        if set(room_predictions) != expected:
            raise ValueError("room predictions must cover all unresolved locations")
        assumptions = []
        for item in facts.objects:
            room_node_id = room_predictions.get(item.graph_index)
            ordinary = not item.finished and item.property is None
            if room_node_id is not None or ordinary:
                assumptions.append(ObjectAssumption(
                    item.graph_index, room_node_id, ordinary
                ))
        graph = cls(
            facts=facts,
            assumptions=tuple(assumptions),
            reopened_room_ids=(
                tuple(sorted(set(room_predictions.values())))
                if reopen_all_rooms else ()
            ),
        )
        graph.validate()
        return graph

    @classmethod
    def for_known_facts(cls, facts: FactGraph,
                        revealed_rooms: Optional[Dict[int, int]] = None,
                        reopened_room_ids=()) -> "DeterministicGraph":
        """Build a scoped view that cannot plan unrevealed task slots."""
        facts.validate()
        revealed = dict(revealed_rooms or {})
        indices = {
            item.graph_index for item in facts.objects
            if item.holder_agent_ids
            or (not item.finished and item.room_node_id is not None)
        } | set(revealed)
        assumptions = []
        for item in facts.objects:
            if item.graph_index not in indices:
                continue
            room_node_id = revealed.get(item.graph_index)
            ordinary = item.property is None
            if room_node_id is not None or ordinary:
                assumptions.append(ObjectAssumption(
                    item.graph_index, room_node_id, ordinary
                ))
        graph = cls(
            facts=facts,
            assumptions=tuple(assumptions),
            reopened_room_ids=tuple(sorted(set(reopened_room_ids))),
            planning_object_indices=tuple(sorted(indices)),
        )
        graph.validate()
        return graph

    def to_dict(self) -> dict:
        self.validate()
        return {
            "graph_kind": "deterministic",
            "facts": self.facts.to_dict(),
            "assumptions": [
                {"graph_index": item.graph_index,
                 "room_node_id": item.room_node_id,
                 "ordinary_property": item.ordinary_property}
                for item in self.assumptions
            ],
            "reopened_room_ids": list(self.reopened_room_ids),
            "planning_object_indices": (
                list(self.planning_object_indices)
                if self.planning_object_indices is not None else None
            ),
        }
