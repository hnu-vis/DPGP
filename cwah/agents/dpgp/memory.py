"""Single public-fact memory: reset arguments and this agent's observations only."""

from copy import deepcopy
from dataclasses import replace
from itertools import combinations
import re
import math

from .config import MethodConfig
from .coordination import merge_graphs, merge_grounding, normalize_graph
from .models import (Agent, Cooperation, Entity, FactGraph, Goal, Portal, Room, Stamp, distance,
                     Inspection, Origin, NavigationEvidence)


def _position(value):
    if value is None:
        return None
    if not isinstance(value, (tuple, list)) or len(value) != 3:
        raise ValueError("position must have three coordinates")
    return tuple(float(v) for v in value)


def _node_position(node):
    return _position((node.get("obj_transform") or {}).get("position")) or _position(
        (node.get("bounding_box") or {}).get("center"))


def _property(value):
    if value in (1, "ordinary") and value is not True:
        return 1
    if value in (2, "cooperative"):
        return 2
    if value is None or value == "unknown":
        return None
    raise ValueError("invalid public cooperation property")


def initialize_topology(rooms, tolerance=0.6):
    """Retain the backup's boundary-contact MST approximation at reset only.

    Geometry does not prove a doorway exists. Observed BETWEEN edges refine or
    add connections later; this method never inspects hidden scene connectivity.
    """
    parents = {r.id: r.id for r in rooms}

    def root(key):
        while parents[key] != key:
            key = parents[key]
        return key

    candidates = []
    for a, b in combinations(rooms, 2):
        contacts = []
        for axis, tangent in ((0, 2), (2, 0)):
            direction = 1 if b.center[axis] >= a.center[axis] else -1
            fa = a.center[axis] + direction * a.size[axis] / 2
            fb = b.center[axis] - direction * b.size[axis] / 2
            low = max(a.center[tangent] - a.size[tangent] / 2, b.center[tangent] - b.size[tangent] / 2)
            high = min(a.center[tangent] + a.size[tangent] / 2, b.center[tangent] + b.size[tangent] / 2)
            if abs(fa - fb) <= tolerance and high - low > 1e-6:
                contacts.append((abs(fa - fb), axis, tangent, (fa + fb) / 2, low, high))
        if not contacts:
            continue
        _, axis, tangent, plane, low, high = min(contacts)
        delta = b.center[axis] - a.center[axis]
        ratio = (plane - a.center[axis]) / delta if abs(delta) > 1e-9 else .5
        position = [(x + y) / 2 for x, y in zip(a.center, b.center)]
        position[axis] = plane
        position[tangent] = max(low, min(high, a.center[tangent] + ratio * (b.center[tangent] - a.center[tangent])))
        candidates.append((distance(a.center, b.center), a.id, b.id, tuple(position)))
    result = []
    for _, a, b, position in sorted(candidates):
        if root(a) != root(b):
            parents[root(a)] = root(b)
            result.append(Portal(a, b, position))
    if len(result) != len(rooms) - 1:
        raise ValueError("public boundary-contact topology disconnected or missing room bounds")
    return tuple(sorted(result, key=lambda p: (p.source, p.target)))


def _preserve_completed(previous, updated):
    """Keep factual goal placements immutable for the rest of this episode."""
    completed = {key for goal in previous.goals for key in previous.satisfied(goal)}
    if not completed:
        return updated
    old = {e.id: e for e in previous.entities}
    entities = {e.id: e for e in updated.entities}
    for key in completed:
        placement = old[key]
        current = entities.get(key, placement)
        if not current.holders and current.inside == placement.inside and current.on == placement.on:
            continue
        entities[key] = replace(current, room_id=placement.room_id, position=placement.position,
                                inside=placement.inside, on=placement.on, holders=(),
                                stamp=placement.stamp)
    agents = tuple(replace(agent, held=tuple(i for i in agent.held if i not in completed))
                   for agent in updated.agents)
    cooperation = updated.cooperation
    if cooperation.object_id in completed:
        cooperation = Cooperation(stamp=cooperation.stamp)
    return normalize_graph(replace(updated, agents=agents,
                                   entities=tuple(entities[i] for i in sorted(entities)),
                                   cooperation=cooperation))


def _confirm_placement(entities, agents, cooperation, goals, key, target, relation, stamp):
    """Commit one environment-confirmed goal placement; repeated commits are harmless."""
    if key not in entities or target not in entities or relation not in ("ON", "INSIDE"):
        raise ValueError("confirmed placement references unknown entities/relation")
    entity, destination = entities[key], entities[target]
    if not any(g.object_class == entity.name and g.target_id == target and g.relation == relation for g in goals):
        raise ValueError("confirmed placement does not match a public goal")
    satisfied = not entity.holders and (target in entity.on if relation == "ON" else entity.inside == target)
    if not satisfied:
        entities[key] = replace(entity, holders=(),
                                inside=target if relation == "INSIDE" else destination.room_id,
                                on=(target,) if relation == "ON" else (), room_id=destination.room_id,
                                position=destination.position, stamp=stamp)
    agents = [replace(agent, held=tuple(i for i in agent.held if i != key)) for agent in agents]
    if cooperation.object_id == key:
        cooperation = Cooperation(stamp=stamp)
    return entities, agents, cooperation


class MethodMemory:
    def __init__(self, agent_id, char_index, config=None):
        if type(char_index) is not int or char_index not in (0, 1) or agent_id != char_index + 1:
            raise ValueError("agent_id must equal char_index + 1")
        self.agent_id, self.index = agent_id, char_index
        self.config = config or MethodConfig()
        self.graph = None
        self.step = 0
        self.visible_ids = ()
        self.last_messages = ()
        self.last_feedback = {}
        self.last_message_feedback = {}
        self.last_coop_feedback = ()
        self.last_coop_info = {}
        self._container_classes = set()

    def reset(self, observation, containers_name, goal_objects_name, rooms_name, room_info, goal_spec):
        self.step = 0
        self._container_classes = {str(v).lower() for v in containers_name}
        goals, targets = [], {}
        for key, raw_count in sorted(goal_spec.items()):
            count = raw_count[0] if isinstance(raw_count, (list, tuple)) else raw_count
            if type(count) is not int or count < 0:
                raise ValueError("invalid goal count")
            if not count:
                continue
            match = re.fullmatch(r"(on|inside)_(.+)_<([^<>]+)>\s*\((-?\d+)\)", key, re.I)
            if match:
                relation, name, target_class, target_id = match.groups()
            else:
                match = re.fullmatch(r"(on|inside)_(.+)_(-?\d+)", key, re.I)
                if not match:
                    raise ValueError("unsupported goal " + key)
                relation, name, target_id = match.groups()
                target_class = "unknown_target"
            target_id = int(target_id)
            goals.append(Goal(key, relation.upper(), name.lower(), target_id, count))
            target = Entity(target_id, target_class.lower(), container=relation.upper() == "INSIDE")
            if target_id in targets and targets[target_id].name != target.name:
                raise ValueError("inconsistent public target class")
            targets[target_id] = replace(target, container=target.container or (
                target_id in targets and targets[target_id].container))
        inferred = {str(n).lower() for n in goal_objects_name} - {g.object_class for g in goals}
        if len(inferred) == 1:
            targets = {i: replace(t, name=next(iter(inferred))) if t.name == "unknown_target" else t
                       for i, t in targets.items()}
        rooms = {}
        for node in list(room_info) + list(observation.get("nodes", ())):
            if str(node.get("category", "")).lower() == "rooms":
                box = node.get("bounding_box") or {}
                room = Room(int(node["id"]), node["class_name"].lower(),
                            _position(box.get("center")) or _node_position(node), _position(box.get("size")))
                if room.center is None or room.size is None or min(room.size) <= 0:
                    raise ValueError("public room center and size are required")
                if room.id in rooms and rooms[room.id] != room:
                    raise ValueError("inconsistent public room geometry")
                rooms[room.id] = room
        rooms = tuple(rooms[i] for i in sorted(rooms))
        self.graph = FactGraph(0, self.index, rooms, initialize_topology(rooms, self.config.boundary_tolerance),
                               (Agent(0), Agent(1)), tuple(targets[i] for i in sorted(targets)), tuple(goals))
        self.update_from_observation(observation, step=0)

    def snapshot(self):
        if self.graph is None:
            raise RuntimeError("memory has not been reset")
        return self.graph

    def merge_received(self, graph):
        """Called only after a message was actually delivered and decoded."""
        if graph.step > self.step:
            raise ValueError("peer message contains future facts")
        previous = self.snapshot()
        self.graph = _preserve_completed(previous, merge_graphs(previous, graph, source=self.index))

    def merge_grounding(self, entities):
        """Absorb delivered action facts without reopening completed goals."""
        previous = self.snapshot()
        self.graph = _preserve_completed(previous, merge_grounding(previous, entities))

    def confirm_placement(self, action):
        """Commit a Put whose public environment feedback passed its postcondition."""
        if action.kind != "Put":
            raise ValueError("only Put can confirm a goal placement")
        graph = self.snapshot()
        entities = {e.id: e for e in graph.entities}
        if action.object_id not in entities:
            raise ValueError("confirmed Put references an unknown object")
        matches = [g for g in graph.goals if g.object_class == entities[action.object_id].name
                   and g.target_id == action.target_id]
        if len(matches) != 1:
            raise ValueError("confirmed Put does not identify one public goal")
        stamp = Stamp(self.step, self.index)
        entities, agents, cooperation = _confirm_placement(
            entities, list(graph.agents), graph.cooperation, graph.goals,
            action.object_id, action.target_id, matches[0].relation, stamp)
        self.graph = normalize_graph(replace(
            graph, agents=tuple(agents), entities=tuple(entities[i] for i in sorted(entities)),
            cooperation=cooperation))

    def update_from_observation(self, observation, previous_action=None, *, step=None, commitment=None):
        graph = self.snapshot()
        new_step = self.step + 1 if step is None else step
        if type(new_step) is not int or new_step < self.step:
            raise ValueError("observation clock must not go backwards")
        stamp = Stamp(new_step, self.index)
        nodes = {int(n["id"]): n for n in observation.get("nodes", ())}
        edges = [(int(e["from_id"]), str(e["relation_type"]).upper(), int(e["to_id"]))
                 for e in observation.get("edges", ())]
        rooms = {r.id: r for r in graph.rooms}
        entities = {e.id: e for e in graph.entities}
        agents = list(graph.agents)

        def relations(key, kind):
            return tuple(sorted({b for a, rel, b in edges if a == key and rel == kind}))

        for index in (0, 1):
            if index != self.index and index + 1 not in nodes:
                continue
            old = agents[index]
            room_ids = [r for r in relations(index + 1, "INSIDE") if r in rooms]
            close = tuple(sorted({b if a == index + 1 else a for a, rel, b in edges
                                  if rel == "CLOSE" and index + 1 in (a, b)}))
            held = tuple(sorted({b for a, rel, b in edges if a == index + 1 and rel.startswith("HOLDS")}))
            position = _node_position(nodes.get(index + 1, {}))
            if index == self.index:
                position = _position(observation.get("location")) or position
            agents[index] = Agent(index, room_ids[0] if room_ids else old.room_id,
                                  position if position is not None else old.position, held, close,
                                  relations(index + 1, "FACING"), stamp,
                                  commitment if index == self.index else old.commitment)
        own_room = agents[self.index].room_id
        if own_room in rooms:
            rooms[own_room] = replace(rooms[own_room], explored=True)
        # Select task entities and all their visible placement ancestors.
        relevant = {e.id for e in entities.values()} | {i for a in agents for i in a.held}
        classes = {g.object_class for g in graph.goals} | self._container_classes
        relevant.update(i for i, n in nodes.items() if n.get("class_name", "").lower() in classes)
        for _ in nodes:
            additions = {b for a, rel, b in edges if a in relevant and rel in ("INSIDE", "ON")
                         and b in nodes and b not in rooms and b not in (1, 2)}
            if additions <= relevant:
                break
            relevant |= additions
        visible_entities = relevant & set(nodes) - set(rooms) - {1, 2}
        entity_ids = set(entities) | visible_entities
        for key in sorted(visible_entities):
            node = nodes[key]
            name = node["class_name"].lower()
            old = entities.get(key, Entity(key, name))
            if old.name not in (name, "unknown_target"):
                raise ValueError("observed identity changed class")
            states = {str(s).upper() for s in node.get("states", ())}
            properties = {str(p).upper() for p in node.get("properties", ())}
            container = old.container or name in self._container_classes or "CAN_OPEN" in properties
            state = "open" if "OPEN" in states else "closed" if "CLOSED" in states else None
            parents = relations(key, "INSIDE")
            if len(parents) > 1:
                # VH can expose both the immediate container and its room.
                nonroom = [p for p in parents if p not in rooms]
                parents = tuple(nonroom) if nonroom else parents
            if len(parents) > 1:
                raise ValueError("multiple immediate INSIDE parents")
            # A visible entity's current relations override remembered remote holds.
            holders = tuple(sorted({a - 1 for a, rel, b in edges
                                    if b == key and a in (1, 2) and rel.startswith("HOLDS")}))
            entities[key] = Entity(key, name, old.room_id, _node_position(node),
                                   None if holders else next(iter(parents), None),
                                   () if holders else tuple(i for i in relations(key, "ON") if i in entity_ids), holders,
                                   container, name in self._container_classes, state if container else None,
                                   old.explored or (container and (state == "open" or (
                                       state is None and "CAN_OPEN" not in properties))), old.coop, stamp)
        # A complete visible region can disprove a remembered location, not identity.
        visible_regions = {own_room} | {e.id for e in entities.values() if e.id in nodes
                                       and e.container and e.open_state == "open"}
        for key, entity in tuple(entities.items()):
            if key in nodes or entity.holders or entity.stamp.step < 0:
                continue
            if entity.inside in visible_regions:
                entities[key] = replace(entity, room_id=None, position=None, inside=None, on=(), stamp=stamp)
        properties = dict(graph.class_coop)
        for name, raw in (observation.get("class_coop_properties") or {}).items():
            name, value = name.lower(), _property(raw)
            if value is None or name not in {g.object_class for g in graph.goals}:
                continue
            if name in properties and properties[name] != value:
                raise ValueError("conflicting class cooperation property")
            properties[name] = value
        for key, raw in (observation.get("object_coop_properties") or {}).items():
            key, value = int(key), _property(raw)
            if key in entities and value is not None:
                old = entities[key]
                if old.coop is not None and old.coop != value:
                    raise ValueError("conflicting instance cooperation property")
                entities[key] = replace(old, coop=value)
        public_coop = observation.get("coop_info") or {}
        coop = Cooperation(stamp=stamp)
        if public_coop.get("active"):
            key, holder, supporter = (int(public_coop[f]) for f in ("object_id", "holder_agent", "supporter_agent"))
            if {holder, supporter} != {0, 1}:
                raise ValueError("invalid cooperation roles")
            old = entities.get(key, Entity(key, public_coop["object_class"].lower()))
            entities[key] = replace(old, holders=(holder,), inside=None, on=(), coop=2, stamp=stamp)
            coop = Cooperation(key, holder, supporter, stamp)
            for index, field in ((holder, "holder_pos"), (supporter, "supporter_pos")):
                position = _position(public_coop.get(field))
                agents[index] = replace(agents[index], position=position or agents[index].position,
                                        held=(key,) if index == holder else (), stamp=stamp)
        feedback = deepcopy(observation.get("action_feedback") or {})
        for field in ("valid", "action_status", "invalid_reason"):
            if field not in feedback and field in observation:
                feedback[field] = observation[field]
        if "requested_envelope" not in feedback and previous_action is not None:
            feedback["requested_envelope"] = deepcopy(previous_action)
        events = deepcopy(observation.get("coop_action_feedback") or [])
        for event in events:
            if event.get("type") != "coop_put_relation_committed":
                continue
            key, target = int(event["object_id"]), int(event["target_id"])
            relation = str(event["relation"]).upper()
            entities, agents, coop = _confirm_placement(
                entities, agents, coop, graph.goals, key, target, relation, stamp)
        portals = {(p.source, p.target): p for p in graph.portals}
        for key, node in nodes.items():
            if node.get("class_name", "").lower() not in ("door", "doorjamb"):
                continue
            pair = tuple(r for r in relations(key, "BETWEEN") if r in rooms)
            if len(pair) != 2:
                continue
            old = portals.get(pair)
            if old and old.door_id is not None and old.door_id < key:
                continue
            position = _node_position(node) or (old.position if old else None)
            if position is None:
                a, b = (rooms[r].center for r in pair)
                position = tuple((x + y) / 2 for x, y in zip(a, b))
            states = {str(s).upper() for s in node.get("states", ())}
            portals[pair] = Portal(*pair, position, key,
                                   "open" if "OPEN" in states else "closed" if "CLOSED" in states else None, stamp)
        # Transactional commit: malformed observations never partially update memory.
        updated = normalize_graph(FactGraph(new_step, self.index, tuple(rooms[i] for i in sorted(rooms)),
                                           tuple(portals[k] for k in sorted(portals)), tuple(agents),
                                           tuple(entities[i] for i in sorted(entities)), graph.goals,
                                           tuple(sorted(properties.items())), coop,
                                           graph.inspections, graph.origins, graph.navigation))
        # ==================== Coop: Public Inspection Evidence ====================
        records = {e.region: e for e in graph.inspections}
        origins = {e.object_id: e for e in graph.origins}
        current = {e.id: e for e in updated.entities}
        targets = {g.object_class for g in graph.goals}
        observed = [e for e in updated.entities if e.id in nodes]
        for entity in observed:
            if entity.name not in targets or entity.id in origins:
                continue
            parent = current.get(entity.inside)
            region = ""
            if not entity.holders:
                if entity.inside in rooms:
                    region = "v:" + str(entity.inside)
                elif parent and parent.container and parent.id in nodes and parent.open_state == "open":
                    region = "c:" + str(parent.id)
            origins[entity.id] = Origin(entity.id, region, stamp)
        areas = [("v:" + str(own_room), own_room)] if own_room in rooms else []
        areas += [("c:" + str(e.id), e.id) for e in observed
                  if e.container and e.open_state == "open" and e.room_id == own_room]
        for region, parent_id in areas:
            if region in records and records[region].complete:
                continue
            members = [e for e in observed if e.inside == parent_id and not e.holders]
            # A direct coverage statement never asserts arbitrary nested visibility.
            complete = (region.startswith("v:") or current[parent_id].inside in rooms)
            records[region] = Inspection(region,
                tuple(sorted(e.id for e in members if e.name in targets)),
                tuple(sorted(e.id for e in members if e.container)),
                tuple(sorted({n.get("class_name", "").lower() for key, n in nodes.items()
                              if key not in rooms and key not in (1, 2)
                              and parent_id in relations(key, "INSIDE")})), complete, stamp)
        navigation = {e.index: e for e in graph.navigation}
        submitted = (previous_action or {}).get("physical_action") if isinstance(previous_action, dict) else previous_action
        old_position, new_position = graph.agents[self.index].position, updated.agents[self.index].position
        if (isinstance(submitted, str) and submitted.startswith("[walktowards]")
                and feedback.get("valid") is not False and feedback.get("action_status") in ("success", "accepted")
                and old_position is not None and new_position is not None):
            travelled = math.hypot(new_position[0] - old_position[0], new_position[2] - old_position[2])
            old = navigation.get(self.index)
            if travelled > 0 and (old is None or old.stamp.step < new_step):
                navigation[self.index] = NavigationEvidence(self.index, (old.count if old else 0) + 1,
                    (old.distance_sum if old else 0.) + travelled, stamp)
        updated = replace(updated, inspections=tuple(records[k] for k in sorted(records)),
                          origins=tuple(origins[k] for k in sorted(origins)),
                          navigation=tuple(navigation[k] for k in sorted(navigation)))
        updated.validate()
        # ==================== End Coop: Public Inspection Evidence ====================
        self.graph, self.step = _preserve_completed(graph, updated), new_step
        self.visible_ids = tuple(sorted(nodes))
        self.last_messages = tuple(deepcopy(observation.get("messages") or []))
        self.last_message_feedback = deepcopy(observation.get("message_feedback") or {})
        self.last_coop_info = deepcopy(public_coop)
        self.last_feedback, self.last_coop_feedback = feedback, tuple(events)
