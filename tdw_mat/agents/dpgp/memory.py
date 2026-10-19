"""Single factual task memory and immutable ``FactGraph`` export."""

import copy
from dataclasses import asdict, dataclass, field
from itertools import combinations
from typing import Any, Dict, Iterable, List, Optional, Tuple

from .models import (
    AgentFact,
    CooperationFact,
    FactGraph,
    GoalAnchorFact,
    GoalFact,
    InspectionFact,
    NavigationEdgeFact,
    ObjectFact,
    Position,
    RoomFact,
    position,
)


def _class_name(value: Any) -> str:
    return str(value).strip().lower()


@dataclass
class _Room:
    node_id: int
    room_name: str
    center: Position
    explored: bool = False
    revisit_count: int = 0


@dataclass
class _Goal:
    required_objects: Dict[str, int] = field(default_factory=dict)
    target_room_node_id: Optional[int] = None


@dataclass
class _GoalAnchor:
    known: bool = False
    object_id: Optional[int] = None
    position: Optional[Position] = None
    room_node_id: Optional[int] = None
    last_seen_frame: Optional[int] = None


@dataclass
class _Object:
    graph_index: int
    class_name: str
    object_id: Optional[int] = None
    room_node_id: Optional[int] = None
    position: Optional[Position] = None
    property: Optional[int] = None
    holder_agent_ids: List[int] = field(default_factory=list)
    finished: bool = False
    last_seen_frame: Optional[int] = None
    information_source: Optional[str] = None


@dataclass
class _Agent:
    agent_id: int
    room_node_id: Optional[int] = None
    held_object_indices: List[int] = field(default_factory=list)
    last_seen_frame: Optional[int] = None
    position: Optional[Position] = None


@dataclass
class _Edge:
    source: int
    target: int
    path_len: float = -1.0
    known_grid_count: int = 0
    total_grid_count: int = 0
    belief: float = 0.0
    cost: float = -1.0
    updated_frame: Optional[int] = None


@dataclass
class ActionFeedback:
    valid: Optional[bool] = None
    action_status: Optional[str] = None
    reason_code: Optional[str] = None
    coop_action_feedback: List[dict] = field(default_factory=list)
    effective_physical_action: Optional[dict] = None
    frame: Optional[int] = None


class MethodMemory:
    """Per-agent mutable facts; assumptions and planning state are forbidden."""

    def __init__(self, agent_id: int):
        if agent_id not in (0, 1):
            raise ValueError("agent_id must be 0 or 1")
        self.agent_id = int(agent_id)
        self.frame = 0
        self.goal = _Goal()
        self.rooms: Dict[int, _Room] = {}
        self.objects: Dict[int, _Object] = {}
        self.agents = {0: _Agent(0), 1: _Agent(1)}
        self.goal_anchor = _GoalAnchor()
        self.cooperation = None
        self.navigation_edges: Dict[Tuple[int, int], _Edge] = {}
        self.inspections: Dict[str, InspectionFact] = {}
        self.current_visible_target_ids: Tuple[int, ...] = ()
        self.last_messages: List[Optional[str]] = []
        self.last_action_feedback = ActionFeedback()
        self._env_api: Optional[dict] = None
        self._room_name_to_id: Dict[str, int] = {}
        self._object_id_to_index: Dict[int, int] = {}

    def reset(self, goal_objects: Dict[str, int], rooms_name: Dict[int, Optional[str]],
              env_api: dict) -> None:
        if not isinstance(goal_objects, dict) or not isinstance(rooms_name, dict):
            raise TypeError("goal_objects and rooms_name must be dictionaries")
        if not isinstance(env_api, dict) or not callable(env_api.get("center_of_room")):
            raise ValueError("env_api must provide center_of_room")
        if not callable(env_api.get("belongs_to_which_room")):
            raise ValueError("env_api must provide belongs_to_which_room")
        self.frame = 0
        self._env_api = env_api
        self.cooperation = None
        self.rooms = {}
        self._room_name_to_id = {}
        target_room = None
        for raw_id, room_name in sorted(rooms_name.items(), key=lambda item: int(item[0])):
            if room_name is None:
                continue
            node_id = int(raw_id)
            center = position(env_api["center_of_room"](room_name))
            if center is None:
                raise ValueError("room centers must be known")
            self.rooms[node_id] = _Room(node_id, str(room_name), center)
            self._room_name_to_id[str(room_name)] = node_id
            if "Bedroom" in str(room_name):
                target_room = node_id
        if not self.rooms:
            raise ValueError("at least one public room is required")

        required = {}
        for raw_name, raw_count in goal_objects.items():
            name, count = _class_name(raw_name), int(raw_count)
            if count > 0:
                required[name] = required.get(name, 0) + count
        if not required:
            raise ValueError("at least one target object is required")
        self.goal = _Goal(dict(sorted(required.items())), target_room)
        self.objects = {}
        index = 0
        for name, count in self.goal.required_objects.items():
            for _ in range(count):
                self.objects[index] = _Object(index, name)
                index += 1
        self.agents = {0: _Agent(0), 1: _Agent(1)}
        self.goal_anchor = _GoalAnchor()
        self.navigation_edges = {}
        self.inspections = {}
        self.current_visible_target_ids = ()
        self.last_messages = []
        self.last_action_feedback = ActionFeedback()
        self._object_id_to_index = {}

    def update(self, obs: dict, agent_memory=None,
               previous_effective_action: Optional[dict] = None) -> None:
        """Update facts from one agent's public Observation and feedback."""
        self.frame = int(obs.get("current_frames", self.frame))
        self.last_messages = list(obs.get("messages") or [])
        action_feedback = obs.get("action_feedback") or {}
        self.last_action_feedback = ActionFeedback(
            valid=obs.get("valid"),
            action_status=obs.get("action_status"),
            reason_code=action_feedback.get("reason_code"),
            coop_action_feedback=copy.deepcopy(obs.get("coop_action_feedback") or []),
            effective_physical_action=copy.deepcopy(
                action_feedback.get("effective_physical_action")
            ),
            frame=self.frame,
        )
        own_position = position((obs.get("agent") or ())[:3])
        own = self.agents[self.agent_id]
        own.room_node_id = self._room_id(own_position)
        own.last_seen_frame = self.frame
        own.position = own_position

        visible = [item for item in (obs.get("visible_objects") or [])
                   if isinstance(item, dict) and item.get("id") is not None]
        object_info = getattr(agent_memory, "object_info", {}) if agent_memory else {}
        visible_target_ids = []
        for raw in sorted(visible, key=lambda item: (_class_name(item.get("name") or ""),
                                                     int(item.get("id", -1)))):
            object_id = int(raw["id"])
            object_position = position((object_info.get(object_id) or {}).get("position"))
            if raw.get("type") == 0:
                if self._observe_target(
                    object_id, raw.get("name"), object_position,
                    "direct_observation"
                ) is not None:
                    visible_target_ids.append(object_id)
            elif raw.get("type") == 2:
                self.goal_anchor = _GoalAnchor(
                    known=object_position is not None,
                    object_id=object_id,
                    position=object_position,
                    room_node_id=self._room_id(object_position),
                    last_seen_frame=self.frame,
                )
        self.current_visible_target_ids = tuple(sorted(set(visible_target_ids)))

        self._update_opponent(visible, object_info)
        self._update_held(obs)
        self._update_public_properties(obs)
        self._update_cooperation(obs.get("coop_info") or {},
                                 obs.get("coop_action_feedback") or [])
        effective = self.last_action_feedback.effective_physical_action
        self._infer_property(effective if effective is not None
                             else previous_effective_action)
        self._infer_finished()

    def refresh_navigation(self, agent_memory) -> None:
        edges = {}
        for source, target in combinations(sorted(self.rooms), 2):
            info = agent_memory.get_navigation_path_info(
                self.rooms[source].center, self.rooms[target].center
            )
            path_len = float(info.get("path_len", -1))
            known = int(info.get("known_grid_count", 0))
            total = int(info.get("total_grid_count", 0))
            edges[(source, target)] = _Edge(
                source, target, path_len, known, total,
                float(known) / total if total > 0 else 0.0,
                path_len if path_len >= 0 else -1.0,
                self.frame,
            )
        self.navigation_edges = edges

    def snapshot(self) -> FactGraph:
        """Return a detached immutable graph containing factual data only."""
        graph = FactGraph(
            frame=self.frame,
            rooms=tuple(RoomFact(item.node_id, item.room_name, tuple(item.center),
                                 item.explored, item.revisit_count)
                        for _, item in sorted(self.rooms.items())),
            edges=tuple(
                NavigationEdgeFact(item.source, item.target, float(item.path_len),
                                   int(item.known_grid_count), int(item.total_grid_count),
                                   float(item.belief), float(item.cost))
                for _, item in sorted(self.navigation_edges.items()) if item.cost >= 0
            ),
            goal=GoalFact(
                self.goal.target_room_node_id,
                tuple(sorted(self.goal.required_objects.items())),
                GoalAnchorFact(
                    self.goal_anchor.known, self.goal_anchor.object_id,
                    tuple(self.goal_anchor.position)
                    if self.goal_anchor.position is not None else None,
                    self.goal_anchor.room_node_id,
                    self.goal_anchor.last_seen_frame,
                ),
            ),
            objects=tuple(
                ObjectFact(
                    item.graph_index, item.class_name, item.object_id,
                    item.room_node_id,
                    tuple(item.position) if item.position is not None else None,
                    item.property, tuple(sorted(item.holder_agent_ids)),
                    item.finished, item.last_seen_frame, item.information_source,
                ) for _, item in sorted(self.objects.items())
            ),
            agents=tuple(
                AgentFact(item.agent_id, item.room_node_id,
                          tuple(sorted(item.held_object_indices)),
                          item.last_seen_frame,
                          tuple(item.position) if item.position is not None else None)
                for _, item in sorted(self.agents.items())
            ),
            inspections=tuple(self.inspections[key]
                              for key in sorted(self.inspections)),
            cooperation=self.cooperation,
        )
        graph.validate()
        return graph

    def apply_fused_facts(self, facts: FactGraph) -> None:
        if not isinstance(facts, FactGraph):
            raise TypeError("apply_fused_facts accepts FactGraph only")
        facts.validate()
        self.frame = facts.frame
        self.cooperation = facts.cooperation
        self.rooms = {item.node_id: _Room(item.node_id, item.room_name,
                                         tuple(item.center), item.explored,
                                         item.revisit_count)
                      for item in facts.rooms}
        self._room_name_to_id = {item.room_name: item.node_id for item in facts.rooms}
        self.goal = _Goal(dict(facts.goal.required_objects),
                          facts.goal.target_room_node_id)
        anchor = facts.goal.goal_anchor
        self.goal_anchor = _GoalAnchor(
            anchor.known, anchor.object_id,
            tuple(anchor.position) if anchor.position is not None else None,
            anchor.room_node_id, anchor.last_seen_frame,
        )
        self.objects = {
            item.graph_index: _Object(
                item.graph_index, item.class_name, item.object_id,
                item.room_node_id,
                tuple(item.position) if item.position is not None else None,
                item.property, list(item.holder_agent_ids), item.finished,
                item.last_seen_frame, item.information_source,
            ) for item in facts.objects
        }
        self.agents = {
            item.agent_id: _Agent(item.agent_id, item.room_node_id,
                                  list(item.held_object_indices),
                                  item.last_seen_frame,
                                  tuple(item.position)
                                  if item.position is not None else None)
            for item in facts.agents
        }
        self.navigation_edges = {
            (item.source, item.target): _Edge(
                item.source, item.target, item.path_len,
                item.known_grid_count, item.total_grid_count,
                item.belief, item.cost, facts.frame,
            ) for item in facts.edges
        }
        self.inspections = {
            item.inspection_id: item for item in facts.inspections
        }
        self._object_id_to_index = {
            item.object_id: item.graph_index for item in facts.objects
            if item.object_id is not None
        }

    def mark_room_explored(self, node_id: int, explored: bool = True) -> None:
        if node_id not in self.rooms:
            raise KeyError("unknown room {}".format(node_id))
        self.rooms[node_id].explored = bool(explored)

    def mark_room_revisited(self, node_id: int, required_count: int) -> None:
        if node_id not in self.rooms:
            raise KeyError("unknown room {}".format(node_id))
        if type(required_count) is not int or required_count <= 0:
            raise ValueError("required_count must be a positive integer")
        room = self.rooms[node_id]
        room.revisit_count = max(room.revisit_count, required_count)

    def record_inspection(self, room_node_id: int, viewpoint: Position,
                          scan_pass: int, start_frame: int, end_frame: int,
                          observed_target_ids: Iterable[int]) -> InspectionFact:
        if room_node_id not in self.rooms:
            raise KeyError("unknown room {}".format(room_node_id))
        viewpoint = position(viewpoint)
        if viewpoint is None:
            raise ValueError("inspection viewpoint is required")
        inspection_id = "{}:{}:{}".format(
            self.agent_id, room_node_id, scan_pass
        )
        scope_key = "{}:{:.1f}:{:.1f}:turn360".format(
            room_node_id,
            round(viewpoint[0] * 2.0) / 2.0,
            round(viewpoint[2] * 2.0) / 2.0,
        )
        evidence = InspectionFact(
            inspection_id=inspection_id,
            scope_key=scope_key,
            room_node_id=int(room_node_id),
            viewpoint=viewpoint,
            scan_pass=int(scan_pass),
            start_frame=int(start_frame),
            end_frame=int(end_frame),
            observed_target_ids=tuple(sorted(set(
                int(value) for value in observed_target_ids
            ))),
            scan_completed=True,
            source_agent=self.agent_id,
        )
        existing = self.inspections.get(inspection_id)
        if existing is not None and existing != evidence:
            raise ValueError("inspection id conflicts with existing evidence")
        self.inspections[inspection_id] = evidence
        return evidence

    def inspection_viewpoints(self, room_node_id: int) -> Tuple[Position, ...]:
        return tuple(
            item.viewpoint for item in self.inspections.values()
            if item.room_node_id == room_node_id and item.scan_completed
        )

    def mark_object_finished(self, object_id: int, finished: bool = True) -> None:
        item = self._object_by_id(object_id)
        if item is None:
            raise KeyError("unknown target object {}".format(object_id))
        item.finished = bool(finished)

    def mark_object_property(self, object_id: int, property_value: int,
                             source: str) -> None:
        item = self._object_by_id(object_id)
        if item is None:
            raise KeyError("unknown target object {}".format(object_id))
        self._set_class_property(item, property_value, source)

    def to_dict(self) -> dict:
        return {
            "frame": self.frame,
            "agent_id": self.agent_id,
            "goal": asdict(self.goal),
            "rooms": [asdict(self.rooms[key]) for key in sorted(self.rooms)],
            "objects": [asdict(self.objects[key]) for key in sorted(self.objects)],
            "agents": [asdict(self.agents[key]) for key in sorted(self.agents)],
            "goal_anchor": asdict(self.goal_anchor),
            "navigation_edges": [asdict(self.navigation_edges[key])
                                 for key in sorted(self.navigation_edges)],
            "inspections": [asdict(self.inspections[key])
                            for key in sorted(self.inspections)],
            "last_messages": list(self.last_messages),
            "last_action_feedback": asdict(self.last_action_feedback),
        }

    def _room_id(self, object_position: Optional[Position]) -> Optional[int]:
        if object_position is None or self._env_api is None:
            return None
        room_name = self._env_api["belongs_to_which_room"](object_position)
        return self._room_name_to_id.get(room_name)

    def _observe_target(self, object_id: int, raw_name: Any,
                        object_position: Optional[Position], source: str):
        item = self._object_by_id(object_id)
        if item is None:
            candidates = [candidate for candidate in self.objects.values()
                          if candidate.object_id is None
                          and candidate.class_name == _class_name(raw_name)]
            if not candidates:
                return None
            item = min(candidates, key=lambda candidate: candidate.graph_index)
            item.object_id = int(object_id)
            self._object_id_to_index[int(object_id)] = item.graph_index
        if object_position is not None:
            item.position = object_position
            item.room_node_id = self._room_id(object_position)
        item.last_seen_frame = self.frame
        item.information_source = source
        return item

    def _object_by_id(self, object_id: Any):
        if object_id is None:
            return None
        index = self._object_id_to_index.get(int(object_id))
        return self.objects.get(index) if index is not None else None

    def _set_holders(self, item: _Object, holder_ids: Iterable[int]) -> None:
        holders = sorted(set(int(value) for value in holder_ids))
        if any(value not in self.agents for value in holders):
            raise ValueError("object references an unknown holder")
        item.holder_agent_ids = holders
        if holders:
            room = self.agents[holders[0]].room_node_id
            if room is not None:
                item.room_node_id = room
        for agent_id, agent in self.agents.items():
            values = set(agent.held_object_indices)
            if agent_id in holders:
                values.add(item.graph_index)
            else:
                values.discard(item.graph_index)
            agent.held_object_indices = sorted(values)

    def _set_class_property(self, item: _Object, property_value: int,
                            source: str) -> None:
        if property_value not in (1, 2):
            raise ValueError("property must be 1 or 2")
        for candidate in self.objects.values():
            if candidate.class_name == item.class_name:
                candidate.property = property_value
                candidate.information_source = source

    @staticmethod
    def _property(value):
        if value in (1, "ordinary") and value is not True:
            return 1
        if value in (2, "cooperative"):
            return 2
        return None

    def _update_public_properties(self, obs: dict) -> None:
        for raw_name, raw_value in (obs.get("class_coop_properties") or {}).items():
            value = self._property(raw_value)
            item = next((candidate for candidate in self.objects.values()
                         if candidate.class_name == _class_name(raw_name)), None)
            if value is not None and item is not None:
                self._set_class_property(item, value, "environment_property_feedback")
        for raw_id, raw_value in (obs.get("object_coop_properties") or {}).items():
            value, item = self._property(raw_value), self._object_by_id(raw_id)
            if value is not None and item is not None:
                self._set_class_property(item, value, "environment_property_feedback")

    def _update_opponent(self, visible: Iterable[dict], object_info: dict) -> None:
        opponent = 1 - self.agent_id
        seen = any(item.get("type") == 3 and int(item.get("id", -1)) == opponent
                   for item in visible)
        opponent_position = position((object_info.get(opponent) or {}).get("position"))
        if seen and opponent_position is not None:
            self.agents[opponent].room_node_id = self._room_id(opponent_position)
            self.agents[opponent].last_seen_frame = self.frame
            self.agents[opponent].position = opponent_position

    def _update_held(self, obs: dict) -> None:
        own = self.agents[self.agent_id]
        previous = set(own.held_object_indices)
        current = self._held_indices(obs.get("held_objects") or [], self.agent_id)
        own.held_object_indices = current
        for index in previous - set(current):
            item = self.objects[index]
            self._set_holders(item, (value for value in item.holder_agent_ids
                                     if value != self.agent_id))

        opponent = 1 - self.agent_id
        visible = any(item.get("type") == 3 and int(item.get("id", -1)) == opponent
                      for item in (obs.get("visible_objects") or [])
                      if isinstance(item, dict) and item.get("id") is not None)
        if not visible:
            return
        state = self.agents[opponent]
        previous = set(state.held_object_indices)
        current = self._held_indices(obs.get("oppo_held_objects") or [], opponent)
        state.held_object_indices = current
        state.last_seen_frame = self.frame
        for index in previous - set(current):
            item = self.objects[index]
            self._set_holders(item, (value for value in item.holder_agent_ids
                                     if value != opponent))

    def _held_indices(self, raw_items: Iterable[dict], holder: int) -> List[int]:
        result = []
        for raw in raw_items:
            if (not isinstance(raw, dict) or raw.get("id") is None
                    or raw.get("type") != 0):
                continue
            item = self._observe_target(int(raw["id"]), raw.get("name"), None,
                                        "held_object_observation")
            if item is not None:
                self._set_holders(item, (holder,))
                result.append(item.graph_index)
        return sorted(result)

    def _update_cooperation(self, coop_info: dict, events: Iterable[dict]) -> None:
        self.cooperation = None
        if coop_info.get("active"):
            holder = int(coop_info["holder_id"])
            supporter = int(coop_info["supporter_id"])
            room = self._room_id(position(coop_info.get("holder_pos")))
            if room is None:
                room = self.agents[holder].room_node_id
            if room is not None:
                self.agents[holder].room_node_id = room
                self.agents[holder].last_seen_frame = self.frame
            holder_position = position(coop_info.get("holder_pos"))
            if holder_position is not None:
                self.agents[holder].position = holder_position
            item = self._object_by_id(coop_info.get("object_id"))
            if item is not None:
                self._set_class_property(item, 2, "coop_info")
                self._set_holders(item, (holder,))
                self.cooperation = CooperationFact(item.object_id, holder, supporter)
                if room is not None:
                    item.room_node_id = room
        coop_events = {"coop_grasp_started", "coop_carrying_started",
                       "coop_drop_started", "coop_carrying_released"}
        for event in events:
            if not isinstance(event, dict):
                continue
            item = self._object_by_id(event.get("object_id"))
            if item is None or event.get("type") not in coop_events:
                continue
            self._set_class_property(item, 2, "coop_action_feedback")
            if event.get("type") == "coop_carrying_released":
                self._set_holders(item, ())

    def _infer_property(self, raw_action: Optional[dict]) -> None:
        action = raw_action or {}
        if "physical_action" in action:
            action = action.get("physical_action") or {}
        item = self._object_by_id(action.get("object"))
        if item is None:
            return
        reason = self.last_action_feedback.reason_code
        if reason == "requires_two_agent_cooperation":
            self._set_class_property(item, 2, "ordinary_grasp_feedback")
        elif reason == "coop_grasp_non_coop_target":
            self._set_class_property(item, 1, "coop_grasp_feedback")
        elif (action.get("type") == 3 and self.last_action_feedback.valid
              and (self.last_action_feedback.action_status == "accepted"
                   or str(self.last_action_feedback.action_status).lower().endswith("success"))):
            self._set_class_property(item, 1, "ordinary_grasp_feedback")

    def _infer_finished(self) -> None:
        anchor, target_room = self.goal_anchor, self.goal.target_room_node_id
        if not anchor.known or anchor.position is None or target_room is None:
            return
        for item in self.objects.values():
            if (item.finished or item.last_seen_frame != self.frame
                    or item.position is None or item.room_node_id != target_room
                    or item.holder_agent_ids):
                continue
            squared = ((item.position[0] - anchor.position[0]) ** 2
                       + (item.position[2] - anchor.position[2]) ** 2)
            if squared < 9.0:
                item.finished = True
