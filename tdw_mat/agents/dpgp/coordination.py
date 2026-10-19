"""Fact-only coordination: change classification and deterministic fusion."""

from dataclasses import dataclass, replace
from typing import Dict, Optional, Tuple

from .models import AgentFact, CooperationFact, FactGraph, GoalFact, ObjectFact


@dataclass(frozen=True)
class CoordinationNeeds:
    all_events: Tuple[str, ...]
    communication_events: Tuple[str, ...]
    local_events: Tuple[str, ...]


def _objects(graph: FactGraph) -> Dict[int, ObjectFact]:
    return {item.graph_index: item for item in graph.objects}


def _agent(graph: FactGraph, agent_id: int) -> AgentFact:
    return next(item for item in graph.agents if item.agent_id == agent_id)


def _coop_signature(graph: FactGraph):
    return graph.cooperation


def _cooperative_holding_changed(previous: FactGraph, current: FactGraph,
                                 agent_id: int) -> bool:
    before = _agent(previous, agent_id)
    after = _agent(current, agent_id)
    changed = set(before.held_object_indices) ^ set(after.held_object_indices)
    objects = _objects(current)
    objects.update(_objects(previous))
    return any(objects[index].property == 2 for index in changed if index in objects)


def detect_new_situations(previous: FactGraph, current: FactGraph, agent_id: int,
                          move_target_room_node_id: Optional[int] = None,
                          planning_action_completed: bool = False) -> Tuple[str, ...]:
    """Return planning-relevant changes in stable order."""
    events = []
    before_agent, after_agent = _agent(previous, agent_id), _agent(current, agent_id)
    if (move_target_room_node_id is not None
            and before_agent.room_node_id != move_target_room_node_id
            and after_agent.room_node_id == move_target_room_node_id):
        events.append("move_target_room_reached")

    before_objects = _objects(previous)
    if any(before_objects[item.graph_index].room_node_id is None
           and item.room_node_id is not None for item in current.objects):
        events.append("new_target_object_found")
    before_properties = {
        item.class_name: item.property for item in previous.objects
        if item.property is not None
    }
    after_properties = {
        item.class_name: item.property for item in current.objects
        if item.property is not None
    }
    if any(name not in before_properties for name in after_properties):
        events.append("class_property_revealed")
    if any(not before_objects[item.graph_index].finished and item.finished
           for item in current.objects):
        events.append("target_finished")

    before_rooms = {item.node_id: item for item in previous.rooms}
    if any(not before_rooms[item.node_id].explored and item.explored
           for item in current.rooms):
        events.append("room_explored")
    if any(item.revisit_count > before_rooms[item.node_id].revisit_count
           for item in current.rooms):
        events.append("room_revisited")
    if ({item.inspection_id for item in current.inspections}
            - {item.inspection_id for item in previous.inspections}):
        events.append("room_inspected")
    if (not previous.goal.goal_anchor.known
            and current.goal.goal_anchor.known):
        events.append("goal_anchor_found")
    if before_agent.held_object_indices != after_agent.held_object_indices:
        events.append("self_target_holding_changed")
    if _coop_signature(previous) != _coop_signature(current):
        events.append("active_coop_changed")
    if planning_action_completed:
        events.append("planning_action_completed")
    return tuple(events)


def evaluate_coordination_needs(
    previous: Optional[FactGraph],
    current: FactGraph,
    agent_id: int,
    move_target_room_node_id: Optional[int] = None,
    planning_action_completed: bool = False,
    execution_action_failed: bool = False,
) -> CoordinationNeeds:
    events = (("initial_graph",) if previous is None else detect_new_situations(
        previous, current, agent_id, move_target_room_node_id,
        planning_action_completed,
    ))
    if execution_action_failed:
        events += ("execution_action_failed",)
    global_events = {
        "initial_graph", "new_target_object_found", "class_property_revealed",
        "target_finished", "room_explored", "room_revisited",
        "room_inspected",
        "goal_anchor_found",
        "active_coop_changed", "execution_action_failed",
    }
    if (previous is not None and "self_target_holding_changed" in events
            and _cooperative_holding_changed(previous, current, agent_id)):
        global_events.add("self_target_holding_changed")
    communication = tuple(item for item in events if item in global_events)
    local = tuple(item for item in events if item not in global_events)
    return CoordinationNeeds(events, communication, local)


def _last_seen(item) -> int:
    return item.last_seen_frame if item.last_seen_frame is not None else -1


def _choose_recent(values, preferred_source: int):
    return max(values, key=lambda value: (
        _last_seen(value[1]), value[0] == preferred_source,
    ))


def _merge_rooms(graphs: Dict[int, FactGraph]):
    left = {item.node_id: item for item in graphs[0].rooms}
    right = {item.node_id: item for item in graphs[1].rooms}
    if set(left) != set(right):
        raise ValueError("input graphs disagree on room nodes")
    rooms = []
    for node_id in sorted(left):
        if (left[node_id].room_name != right[node_id].room_name
                or left[node_id].center != right[node_id].center):
            raise ValueError("input graphs disagree on shared room facts")
        rooms.append(replace(
            left[node_id],
            explored=left[node_id].explored or right[node_id].explored,
            revisit_count=max(
                left[node_id].revisit_count, right[node_id].revisit_count
            ),
        ))
    return tuple(rooms)


def _merge_goal(graphs: Dict[int, FactGraph], peer_id: int) -> GoalFact:
    left, right = graphs[0].goal, graphs[1].goal
    if left.required_objects != right.required_objects:
        raise ValueError("input graphs disagree on required objects")
    anchors = [(0, left.goal_anchor), (1, right.goal_anchor)]
    known = [item for item in anchors if item[1].known]
    _, anchor = _choose_recent(known or anchors, peer_id)
    targets = (left.target_room_node_id, right.target_room_node_id)
    if targets[0] is None or targets[1] is None or targets[0] == targets[1]:
        target = targets[1] if targets[0] is None else targets[0]
    else:
        source, _ = _choose_recent(anchors, peer_id)
        target = graphs[source].goal.target_room_node_id
    return GoalFact(target, left.required_objects, anchor)


def _merge_edges(graphs: Dict[int, FactGraph], peer_id: int):
    sets = {
        source: {(item.source, item.target): item for item in graph.edges}
        for source, graph in graphs.items()
    }
    merged = []
    for key in sorted(set(sets[0]) | set(sets[1])):
        left, right = sets[0].get(key), sets[1].get(key)
        if left is None or right is None:
            selected = right if left is None else left
        elif left.belief != right.belief:
            selected = left if left.belief > right.belief else right
        elif left.cost != right.cost:
            selected = left if left.cost < right.cost else right
        else:
            selected = sets[peer_id][key]
        merged.append(selected)
    return tuple(merged)


def _merge_inspections(graphs: Dict[int, FactGraph]):
    merged = {}
    for graph in graphs.values():
        for item in graph.inspections:
            previous = merged.get(item.inspection_id)
            if previous is not None and previous != item:
                raise ValueError("input graphs disagree on inspection evidence")
            merged[item.inspection_id] = item
    return tuple(merged[key] for key in sorted(merged))


def _active_cooperation(coop_info: Optional[dict]):
    if not coop_info or not coop_info.get("active"):
        return None
    holder, supporter = int(coop_info["holder_id"]), int(coop_info["supporter_id"])
    if {holder, supporter} != {0, 1}:
        raise ValueError("active cooperation requires agents 0 and 1")
    return {
        "object_id": (int(coop_info["object_id"])
                      if coop_info.get("object_id") is not None else None),
        "holder_id": holder,
        "supporter_id": supporter,
    }


def _object_by_id(graph: FactGraph, object_id: int):
    return next((item for item in graph.objects if item.object_id == object_id), None)


def _held_ids(graph: FactGraph, agent_id: int) -> set:
    objects = _objects(graph)
    held = set()
    for index in _agent(graph, agent_id).held_object_indices:
        object_id = objects[index].object_id
        if object_id is None:
            raise ValueError("an authoritative agent cannot hold an anonymous slot")
        held.add(object_id)
    return held


def _merge_objects_and_agents(graphs, steps, peer_id, goal, cooperation):
    required = dict(goal.required_objects)
    all_objects = [
        (source, item) for source, graph in graphs.items() for item in graph.objects
    ]
    properties = {}
    for class_name in required:
        known = {item.property for _, item in all_objects
                 if item.class_name == class_name and item.property is not None}
        if len(known) > 1:
            raise ValueError("input graphs disagree on class property")
        properties[class_name] = next(iter(known)) if known else None

    known_ids = sorted({item.object_id for _, item in all_objects
                        if item.object_id is not None})
    facts_by_id = {
        object_id: [value for value in all_objects if value[1].object_id == object_id]
        for object_id in known_ids
    }
    for facts in facts_by_id.values():
        if len({item.class_name for _, item in facts}) != 1:
            raise ValueError("input graphs disagree on object class")

    held = {agent_id: _held_ids(graphs[agent_id], agent_id) for agent_id in (0, 1)}
    agent_facts = {
        agent_id: _choose_recent([
            (source, _agent(graph, agent_id))
            for source, graph in graphs.items()
        ], agent_id)[1]
        for agent_id in (0, 1)
    }
    agent_rooms = {
        agent_id: agent_facts[agent_id].room_node_id for agent_id in (0, 1)
    }
    if cooperation is not None and cooperation["object_id"] is not None:
        object_id = cooperation["object_id"]
        if object_id not in facts_by_id:
            raise ValueError("active cooperation references an unknown target")
        class_name = facts_by_id[object_id][0][1].class_name
        if properties[class_name] == 1:
            raise ValueError("active cooperation conflicts with ordinary property")
        properties[class_name] = 2
        held[cooperation["holder_id"]].add(object_id)
        held[cooperation["supporter_id"]].discard(object_id)

    merged_known = []
    for object_id in known_ids:
        facts = facts_by_id[object_id]
        class_name = facts[0][1].class_name
        _, metadata = _choose_recent(facts, peer_id)
        finished = any(item.finished for _, item in facts)
        holders = tuple(agent_id for agent_id in (0, 1) if object_id in held[agent_id])
        location = None
        if holders:
            if cooperation is not None and cooperation["object_id"] == object_id:
                source = cooperation["holder_id"]
            else:
                source = max(holders, key=lambda value: (
                    steps[value], value == peer_id,
                ))
            located = [value for value in facts if value[1].position is not None]
            _, location = _choose_recent(located or facts, source)
            room = agent_rooms[source]
            object_position = location.position
        else:
            located = [value for value in facts
                       if value[1].position is not None
                       or value[1].room_node_id is not None]
            if located:
                _, location = _choose_recent(located, peer_id)
                room, object_position = location.room_node_id, location.position
            else:
                room, object_position = None, None
        last_seen = [item.last_seen_frame for _, item in facts
                     if item.last_seen_frame is not None]
        merged_known.append(ObjectFact(
            -1, class_name, object_id, room, object_position,
            properties[class_name], holders, finished,
            max(last_seen) if last_seen else None,
            location.information_source if location is not None
            else metadata.information_source,
        ))

    objects = []
    for class_name in sorted(required):
        known = sorted(
            (item for item in merged_known if item.class_name == class_name),
            key=lambda item: item.object_id,
        )
        missing = required[class_name] - len(known)
        if missing < 0:
            raise ValueError("known objects exceed required class count")
        objects.extend(known)
        objects.extend(ObjectFact(-1, class_name, property=properties[class_name])
                       for _ in range(missing))
    objects = tuple(replace(item, graph_index=index)
                    for index, item in enumerate(objects))
    index_by_id = {item.object_id: item.graph_index for item in objects
                   if item.object_id is not None}
    agents = tuple(AgentFact(
        agent_id,
        agent_rooms[agent_id],
        tuple(sorted(index_by_id[value] for value in held[agent_id])),
        agent_facts[agent_id].last_seen_frame,
        agent_facts[agent_id].position,
    ) for agent_id in (0, 1))
    objects = tuple(replace(
        item,
        room_node_id=(agent_rooms[item.holder_agent_ids[0]]
                      if item.holder_agent_ids else item.room_node_id),
    ) for item in objects)
    return objects, agents


def merge_fact_graphs(local: FactGraph, received: FactGraph, local_agent_id: int,
                      local_step: int, received_step: int,
                      coop_info: Optional[dict] = None) -> FactGraph:
    """Fuse only observed/public facts; no LLM is involved."""
    if local_agent_id not in (0, 1):
        raise ValueError("local_agent_id must be 0 or 1")
    local.validate()
    received.validate()
    peer_id = 1 - local_agent_id
    graphs = {local_agent_id: local, peer_id: received}
    steps = {local_agent_id: int(local_step), peer_id: int(received_step)}
    cooperation = _active_cooperation(coop_info)
    if coop_info is None:
        newest = max(graphs.values(), key=lambda graph: graph.frame).cooperation
        if newest is not None:
            cooperation = dict(object_id=newest.object_id, holder_id=newest.holder_id,
                               supporter_id=newest.supporter_id)
    goal = _merge_goal(graphs, peer_id)
    objects, agents = _merge_objects_and_agents(
        graphs, steps, peer_id, goal, cooperation
    )
    result = FactGraph(
        max(item.frame for item in graphs.values()),
        _merge_rooms(graphs),
        _merge_edges(graphs, peer_id),
        goal,
        objects,
        agents,
        _merge_inspections(graphs),
        None if cooperation is None else CooperationFact(**cooperation),
    )
    result.validate()
    return result
