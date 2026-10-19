"""Immutable public facts and separate, temporary deterministic assumptions."""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields, replace
import math
from typing import Optional, Tuple, Union, get_args, get_origin, get_type_hints

Position = Tuple[float, float, float]


@dataclass(frozen=True, order=True)
class Stamp:
    step: int
    source: int


@dataclass(frozen=True)
class Room:
    id: int
    name: str
    center: Position
    size: Position
    explored: bool = False


@dataclass(frozen=True)
class Portal:
    source: int
    target: int
    position: Position
    door_id: Optional[int] = None
    state: Optional[str] = None
    stamp: Stamp = Stamp(-1, 0)


@dataclass(frozen=True)
class Action:
    kind: str
    target_id: Optional[int] = None
    object_id: Optional[int] = None
    cooperative: bool = False

    def validate(self):
        if type(self.cooperative) is not bool:
            raise ValueError("invalid cooperative flag")
        if any(v is not None and type(v) is not int for v in (self.target_id, self.object_id)):
            raise ValueError("action identities must be integers")
        if self.kind not in ("Move", "Grab", "Put", "CheckContainer", "Wait", "Continue"):
            raise ValueError("unknown action kind")
        if self.kind in ("Wait", "Continue"):
            if self.target_id is not None or self.object_id is not None:
                raise ValueError("wait/continue cannot name an entity")
        elif self.target_id is None:
            raise ValueError("action needs target_id")
        if self.kind == "Put" and self.object_id is None:
            raise ValueError("Put needs object_id")
        if self.kind != "Put" and self.object_id is not None:
            raise ValueError("only Put carries object_id")


@dataclass(frozen=True)
class Commitment:
    action: Action
    phase: str
    plan_epoch: int
    remaining: Optional[float] = None


@dataclass(frozen=True)
class Agent:
    index: int
    room_id: Optional[int] = None
    position: Optional[Position] = None
    held: Tuple[int, ...] = ()
    close: Tuple[int, ...] = ()
    facing: Tuple[int, ...] = ()
    stamp: Stamp = Stamp(-1, 0)
    commitment: Optional[Commitment] = None


@dataclass(frozen=True)
class Entity:
    id: int
    name: str
    room_id: Optional[int] = None
    position: Optional[Position] = None
    inside: Optional[int] = None
    on: Tuple[int, ...] = ()
    holders: Tuple[int, ...] = ()
    container: bool = False
    searchable: bool = False
    open_state: Optional[str] = None
    explored: bool = False
    coop: Optional[int] = None
    stamp: Stamp = Stamp(-1, 0)


@dataclass(frozen=True)
class Goal:
    key: str
    relation: str
    object_class: str
    target_id: int
    count: int


@dataclass(frozen=True)
class Cooperation:
    object_id: Optional[int] = None
    holder: Optional[int] = None
    supporter: Optional[int] = None
    stamp: Stamp = Stamp(-1, 0)


# ==================== Coop: Compact Discovery Evidence ====================
@dataclass(frozen=True)
class Inspection:
    region: str
    object_ids: Tuple[int, ...]
    container_ids: Tuple[int, ...]
    semantics: Tuple[str, ...]
    complete: bool
    stamp: Stamp


@dataclass(frozen=True)
class Origin:
    object_id: int
    region: str  # Empty means a held/moved object without a verified source.
    stamp: Stamp


@dataclass(frozen=True)
class NavigationEvidence:
    index: int
    count: int
    distance_sum: float
    stamp: Stamp
# ==================== End Coop: Compact Discovery Evidence ====================


def distance(a: Position, b: Position) -> float:
    return abs(a[0] - b[0]) + abs(a[2] - b[2])


def _finite_position(value):
    return value is not None and len(value) == 3 and all(math.isfinite(v) for v in value)


def _unique(values, label):
    if tuple(sorted(set(values))) != tuple(values):
        raise ValueError(label + " must be sorted and unique")


def restore_record(cls, data):
    """Strict JSON decoder shared by snapshots and the positional wire schema."""
    if not isinstance(data, dict) or set(data) != {f.name for f in fields(cls)}:
        raise ValueError("invalid fields for " + cls.__name__)

    def decode(annotation, value):
        origin, args = get_origin(annotation), get_args(annotation)
        if origin is Union:
            if value is None and type(None) in args:
                return None
            return decode(next(t for t in args if t is not type(None)), value)
        if origin is tuple:
            if not isinstance(value, (list, tuple)):
                raise ValueError("expected array")
            if len(args) == 2 and args[1] is Ellipsis:
                return tuple(decode(args[0], v) for v in value)
            if len(value) != len(args):
                raise ValueError("wrong tuple length")
            return tuple(decode(t, v) for t, v in zip(args, value))
        if hasattr(annotation, "__dataclass_fields__"):
            return restore_record(annotation, value)
        if annotation is float:
            if type(value) not in (int, float) or not math.isfinite(value):
                raise ValueError("expected finite number")
            return float(value)
        if type(value) is not annotation:
            raise ValueError("invalid scalar type")
        return value

    types = get_type_hints(cls)
    return cls(**{name: decode(types[name], value) for name, value in data.items()})


@dataclass(frozen=True)
class FactGraph:
    step: int
    source: int
    rooms: Tuple[Room, ...]
    portals: Tuple[Portal, ...]
    agents: Tuple[Agent, ...]
    entities: Tuple[Entity, ...]
    goals: Tuple[Goal, ...]
    class_coop: Tuple[Tuple[str, int], ...] = ()
    cooperation: Cooperation = Cooperation()
    inspections: Tuple[Inspection, ...] = ()
    origins: Tuple[Origin, ...] = ()
    navigation: Tuple[NavigationEvidence, ...] = ()

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, data):
        result = restore_record(cls, data)
        result.validate()
        return result

    def satisfied(self, goal):
        return tuple(e.id for e in self.entities if e.name == goal.object_class
                     and not e.holders and (goal.target_id in e.on if goal.relation == "ON"
                                            else e.inside == goal.target_id))

    def remaining(self, goal):
        return max(0, goal.count - len(self.satisfied(goal)))

    def validate(self, *, allow_virtual=False):
        if type(self.step) is not int or type(self.source) is not int or self.step < 0 or self.source not in (0, 1):
            raise ValueError("invalid graph clock/source")
        rooms = {r.id: r for r in self.rooms}
        entities = {e.id: e for e in self.entities}
        _unique([r.id for r in self.rooms], "rooms")
        _unique([e.id for e in self.entities], "entities")
        _unique([g.key for g in self.goals], "goals")
        _unique([(p.source, p.target) for p in self.portals], "portals")
        _unique(self.class_coop, "class properties")
        if not rooms or set(rooms) & set(entities) or set(rooms) & {1, 2} or set(entities) & {1, 2}:
            raise ValueError("invalid/overlapping entity identities")
        if not allow_virtual and any(i < 0 for i in entities):
            raise ValueError("virtual entities cannot enter facts")
        for room in self.rooms:
            if not room.name or not _finite_position(room.center) or not _finite_position(room.size) or min(room.size) <= 0:
                raise ValueError("invalid public room geometry")
        for record in (self.portals + self.agents + self.entities + (self.cooperation,)
                       + self.inspections + self.origins + self.navigation):
            if record.stamp.source not in (0, 1) or not -1 <= record.stamp.step <= self.step:
                raise ValueError("invalid evidence stamp")
        _unique([e.region for e in self.inspections], "inspection regions")
        _unique([e.object_id for e in self.origins], "origin identities")
        _unique([e.index for e in self.navigation], "navigation sources")
        def valid_region(region):
            kind, sep, key = region.partition(":")
            return (sep and key.isdigit() and ((kind == "v" and int(key) in rooms)
                    or (kind == "c" and int(key) in entities and entities[int(key)].container)))
        for item in self.inspections:
            if not valid_region(item.region) or type(item.complete) is not bool:
                raise ValueError("invalid inspection")
            for values in (item.object_ids, item.container_ids, item.semantics):
                _unique(values, "inspection values")
            if any(i not in entities for i in item.object_ids + item.container_ids):
                raise ValueError("inspection references unknown identity")
        for item in self.origins:
            if item.object_id not in entities or (item.region and not valid_region(item.region)):
                raise ValueError("invalid discovery origin")
        for item in self.navigation:
            if (item.index not in (0, 1) or item.stamp.source != item.index or type(item.count) is not int
                    or item.count <= 0 or not math.isfinite(item.distance_sum) or item.distance_sum <= 0):
                raise ValueError("invalid navigation evidence")
        reached = {self.rooms[0].id}
        for portal in self.portals:
            if portal.source not in rooms or portal.target not in rooms or portal.source >= portal.target:
                raise ValueError("invalid room connection")
            if not _finite_position(portal.position) or portal.state not in (None, "open", "closed"):
                raise ValueError("invalid portal")
        for _ in self.rooms:
            for p in self.portals:
                if p.source in reached or p.target in reached:
                    reached.update((p.source, p.target))
        if reached != set(rooms):
            raise ValueError("disconnected room topology")
        if tuple(a.index for a in self.agents) != (0, 1):
            raise ValueError("exactly agents 0 and 1 required")
        for agent in self.agents:
            if agent.room_id is not None and agent.room_id not in rooms:
                raise ValueError("unknown agent room")
            if agent.position is not None and not _finite_position(agent.position):
                raise ValueError("invalid agent position")
            for values in (agent.held, agent.close, agent.facing):
                _unique(values, "agent relations")
            if len(agent.held) > 1 or any(i not in entities for i in agent.held):
                raise ValueError("invalid held object")
            if agent.commitment:
                agent.commitment.action.validate()
                c = agent.commitment
                if c.plan_epoch < 0 or not c.phase or (c.remaining is not None and (not math.isfinite(c.remaining) or c.remaining < 0)):
                    raise ValueError("invalid execution commitment")
        for entity in self.entities:
            if not entity.name or entity.coop not in (None, 1, 2):
                raise ValueError("invalid entity class/property")
            if entity.room_id is not None and entity.room_id not in rooms:
                raise ValueError("unknown entity room")
            if entity.position is not None and not _finite_position(entity.position):
                raise ValueError("invalid entity position")
            if entity.open_state not in (None, "open", "closed") or (entity.searchable and not entity.container):
                raise ValueError("invalid container state")
            if not entity.container and (entity.open_state is not None or entity.explored):
                raise ValueError("non-container has container state")
            _unique(entity.on, "ON relations")
            _unique(entity.holders, "holders")
            if len(entity.holders) > 1 or any(i not in (0, 1) for i in entity.holders):
                raise ValueError("invalid physical holder")
            if entity.inside is not None and entity.inside not in rooms and entity.inside not in entities:
                raise ValueError("unknown INSIDE parent")
            if any(i not in entities for i in entity.on):
                raise ValueError("unknown ON target")
            seen, current = {entity.id}, entity.inside
            while current in entities:
                if current in seen:
                    raise ValueError("containment cycle")
                seen.add(current)
                current = entities[current].inside
            if entity.holders and (entity.inside is not None or entity.on):
                raise ValueError("held entity still has placement relations")
            for agent in self.agents:
                if (agent.index in entity.holders) != (entity.id in agent.held):
                    raise ValueError("holder records disagree")
        for goal in self.goals:
            if goal.relation not in ("ON", "INSIDE") or type(goal.count) is not int or goal.count <= 0 or goal.target_id not in entities:
                raise ValueError("invalid goal")
        if len({(g.relation, g.object_class, g.target_id) for g in self.goals}) != len(self.goals):
            raise ValueError("duplicate semantic goal")
        if len(dict(self.class_coop)) != len(self.class_coop) or any(v not in (1, 2) for _, v in self.class_coop):
            raise ValueError("invalid class cooperative properties")
        coop = self.cooperation
        if coop.object_id is None:
            if coop.holder is not None or coop.supporter is not None:
                raise ValueError("inactive cooperation has roles")
        elif (coop.object_id not in entities or {coop.holder, coop.supporter} != {0, 1}
              or entities[coop.object_id].holders != (coop.holder,)
              or entities[coop.object_id].coop != 2 or self.agents[coop.supporter].held):
            raise ValueError("inconsistent active cooperation")


@dataclass(frozen=True)
class Assumption:
    entity_id: int
    field: str
    value: int


@dataclass(frozen=True)
class PlanningGraph(FactGraph):
    """Same node schema, distinct type: never accepted by fact I/O."""


@dataclass(frozen=True)
class DeterministicGraph:
    """Separate wrapper prevents temporary assumptions entering fact fusion/wire."""
    graph: PlanningGraph
    assumptions: Tuple[Assumption, ...]

    def validate(self):
        self.graph.validate(allow_virtual=True)
        completed = {i for g in self.graph.goals for i in self.graph.satisfied(g)}
        demand = {}
        for goal in self.graph.goals:
            demand[goal.object_class] = demand.get(goal.object_class, 0) + self.graph.remaining(goal)
            if self.graph.remaining(goal):
                target = next(e for e in self.graph.entities if e.id == goal.target_id)
                if target.room_id is None or target.position is None:
                    raise ValueError("unresolved active goal target")
        for name, count in demand.items():
            available = [e for e in self.graph.entities if e.name == name and e.id not in completed
                         and e.room_id is not None and e.position is not None and e.coop in (1, 2)]
            if len(available) < count:
                raise ValueError("insufficient resolved supply for " + name)
        for agent in self.graph.agents:
            if any(self.graph.remaining(g) for g in self.graph.goals) and (agent.room_id is None or agent.position is None):
                raise ValueError("both agents must have known planning positions")


@dataclass(frozen=True)
class PlanningResult:
    status: str
    prefixes: Tuple[Tuple[Action, ...], Tuple[Action, ...]] = ((), ())
    objective: Optional[float] = None
    reason: Optional[str] = None
