"""Pure fact reconciliation and task-relevant event classification."""

from dataclasses import dataclass, replace

from .models import Cooperation, FactGraph, Stamp


def normalize_graph(graph):
    """Reconcile placement and physical ownership using their evidence clocks."""
    entities = {e.id: e for e in graph.entities}
    agents = list(graph.agents)
    coop = graph.cooperation
    for entity in tuple(entities.values()):
        claims = [a for a in agents if entity.id in a.held]
        claims = [a for a in claims if a.stamp.step > entity.stamp.step
                  or (a.stamp.step == entity.stamp.step and a.index in entity.holders)]
        if claims:
            newest = max(claims, key=lambda a: (a.stamp.step, a.stamp.source == a.index, a.index))
            if len(claims) > 1 and claims[0].stamp.step == claims[1].stamp.step:
                raise ValueError("simultaneous physical holders")
            entity = replace(entity, holders=(newest.index,), inside=None, on=(),
                             room_id=newest.room_id, position=newest.position,
                             stamp=max(entity.stamp, newest.stamp))
        elif entity.holders:
            holder = agents[entity.holders[0]]
            if holder.stamp.step >= entity.stamp.step and entity.id not in holder.held:
                entity = replace(entity, holders=(), room_id=None, position=None,
                                 stamp=holder.stamp)
        entities[entity.id] = entity
    if coop.object_id is not None:
        entity = entities.get(coop.object_id)
        if entity is None:
            raise ValueError("cooperation refers to missing public entity")
        holder_state = agents[coop.holder]
        released = (entity.stamp.step > coop.stamp.step and not entity.holders) or (
            holder_state.stamp.step > coop.stamp.step and coop.object_id not in holder_state.held)
        if released:
            coop = Cooperation(stamp=max(entity.stamp, holder_state.stamp))
        else:
            entities[entity.id] = replace(entity, holders=(coop.holder,), inside=None, on=(), coop=2,
                                          stamp=max(entity.stamp, coop.stamp))
    # Entities are the single final authority; derive agent holding lists once.
    for i, agent in enumerate(agents):
        agents[i] = replace(agent, held=tuple(sorted(e.id for e in entities.values() if i in e.holders)))

    def locate(entity, visited):
        if entity.id in visited:
            raise ValueError("containment cycle")
        if entity.holders:
            holder = agents[entity.holders[0]]
            return holder.room_id, holder.position
        if entity.inside in entities:
            parent = entities[entity.inside]
            room, position = locate(parent, visited | {entity.id})
            return room, entity.position if entity.position is not None else position
        if entity.inside in {r.id for r in graph.rooms}:
            return entity.inside, entity.position
        return entity.room_id, entity.position

    for entity in tuple(entities.values()):
        room, position = locate(entity, set())
        entities[entity.id] = replace(entity, room_id=room, position=position)
    result = replace(graph, agents=tuple(agents), entities=tuple(entities[i] for i in sorted(entities)),
                     cooperation=coop)
    result.validate()
    return result


def _latest(first, second, owner=None):
    def key(item):
        return (item.stamp.step, item.stamp.source == owner, item.stamp.source)
    if key(first) == key(second) and first != second:
        if owner is not None and replace(first, held=second.held) == second:
            # Normalisation may already have removed a stale hold in one copy.
            # Merge claims here; the newer entity placement will resolve them.
            return replace(first, held=tuple(sorted(set(first.held) | set(second.held))))
        raise ValueError("different facts with identical evidence stamp")
    return max((first, second), key=key)


def _merge_agent(first, second, first_source, second_source):
    """Merge observed motion by time, but execution intent by its author."""
    physical = _latest(replace(first, commitment=None), replace(second, commitment=None),
                       owner=first.index)
    first_owned = first_source == first.index
    second_owned = second_source == second.index
    if first_owned != second_owned:
        commitment = first.commitment if first_owned else second.commitment
    else:
        commitment = _latest(first, second, owner=first.index).commitment
    return replace(physical, commitment=commitment)


def _completed_placements(entity, goals):
    if entity.holders:
        return frozenset()
    return frozenset((goal.relation, goal.target_id) for goal in goals
                     if goal.object_class == entity.name and
                     (goal.target_id in entity.on if goal.relation == "ON"
                      else entity.inside == goal.target_id))


def merge_graphs(first, second, *, source=None):
    """Only delivered FactGraph inputs are accepted; assumptions cannot be fused."""
    if type(first) is not FactGraph or type(second) is not FactGraph:
        raise TypeError("fusion accepts facts only")
    first.validate()
    second.validate()
    if first.goals != second.goals:
        raise ValueError("public task definitions differ")
    if tuple((r.id, r.name, r.center, r.size) for r in first.rooms) != tuple(
            (r.id, r.name, r.center, r.size) for r in second.rooms):
        raise ValueError("public room definitions differ")
    rooms = tuple(replace(a, explored=a.explored or b.explored) for a, b in zip(first.rooms, second.rooms))
    entities = {e.id: e for e in first.entities}
    for other in second.entities:
        old = entities.get(other.id)
        if old is None:
            entities[other.id] = other
            continue
        if old.name != other.name and "unknown_target" not in (old.name, other.name):
            raise ValueError("entity class mismatch")
        old_completed = _completed_placements(old, first.goals)
        other_completed = _completed_placements(other, first.goals)
        if old_completed and other_completed and old_completed != other_completed:
            raise ValueError("conflicting completed goal placements")
        # Identical timestamps can contain independently learned static facts.
        if bool(old_completed) != bool(other_completed):
            newest = old if old_completed else other
        elif old.stamp == other.stamp and old != other:
            dynamic = ("room_id", "position", "inside", "on", "holders", "open_state")
            if any(getattr(old, f) != getattr(other, f) for f in dynamic):
                raise ValueError("contradictory placement with identical stamp")
            newest = old
        else:
            newest = _latest(old, other)
        if old.coop is not None and other.coop is not None and old.coop != other.coop:
            raise ValueError("conflicting instance cooperation property")
        name = other.name if old.name == "unknown_target" else old.name
        entities[other.id] = replace(newest, name=name, container=old.container or other.container,
                                    searchable=old.searchable or other.searchable,
                                    explored=old.explored or other.explored,
                                    coop=old.coop if old.coop is not None else other.coop)
    properties = dict(first.class_coop)
    for name, value in second.class_coop:
        if name in properties and properties[name] != value:
            raise ValueError("conflicting class cooperation property")
        properties[name] = value
    portals = {(p.source, p.target): p for p in first.portals}
    for portal in second.portals:
        key = (portal.source, portal.target)
        old = portals.get(key)
        if old is None:
            portals[key] = portal
        elif old.door_id is None and portal.door_id is not None:
            portals[key] = portal
        elif portal.door_id is None and old.door_id is not None:
            pass
        elif old.door_id != portal.door_id:
            portals[key] = min((old, portal), key=lambda p: p.door_id)
        else:
            portals[key] = _latest(old, portal)
    coop = _latest(first.cooperation, second.cooperation)
    agents = tuple(_merge_agent(a, b, first.source, second.source)
                   for a, b in zip(first.agents, second.agents))
    completed = {key for key, entity in entities.items() if _completed_placements(entity, first.goals)}
    if completed:
        agents = tuple(replace(agent, held=tuple(i for i in agent.held if i not in completed)) for agent in agents)
        if coop.object_id in completed:
            coop = Cooperation(stamp=coop.stamp)
    def evidence_union(left, right, key, rank):
        records = {key(item): item for item in left}
        for item in right:
            ident = key(item)
            if ident not in records or rank(item) < rank(records[ident]):
                records[ident] = item
            elif rank(item) == rank(records[ident]) and item != records[ident]:
                raise ValueError("contradictory discovery evidence")
        return tuple(records[k] for k in sorted(records))
    inspections = evidence_union(first.inspections, second.inspections, lambda e: e.region,
                                 lambda e: (not e.complete, e.stamp))
    origins = evidence_union(first.origins, second.origins, lambda e: e.object_id, lambda e: e.stamp)
    navigation = evidence_union(first.navigation, second.navigation, lambda e: e.index,
                                lambda e: (-e.stamp.step, e.stamp.source))
    result = FactGraph(max(first.step, second.step), min(first.source, second.source) if source is None else source,
                       rooms, tuple(portals[k] for k in sorted(portals)),
                       agents,
                       tuple(entities[k] for k in sorted(entities)), first.goals,
                       tuple(sorted(properties.items())), coop, inspections, origins, navigation)
    return normalize_graph(result)


@dataclass(frozen=True)
class GraphChanges:
    categories: tuple
    affected_ids: tuple

    @property
    def planning_relevant(self):
        return bool(set(self.categories) - {"agent_progress", "recency"})


def classify_changes(before, after):
    after.validate()
    if before is None:
        return GraphChanges(("initial",), ())
    categories, ids = set(), set()
    if before.inspections != after.inspections or before.origins != after.origins:
        categories.add("discovery_evidence")
    if before.rooms != after.rooms:
        categories.add("room_exploration")
    if tuple((p.source, p.target, p.position, p.state, p.door_id) for p in before.portals) != tuple(
            (p.source, p.target, p.position, p.state, p.door_id) for p in after.portals):
        categories.add("topology")
    old = {e.id: e for e in before.entities}
    relevant_classes = {g.object_class for g in after.goals}
    targets = {g.target_id for g in after.goals}
    for entity in after.entities:
        previous = old.get(entity.id)
        if previous is None:
            ids.add(entity.id)
            if entity.name in relevant_classes or entity.id in targets or entity.searchable:
                categories.add("discovery")
        elif replace(previous, stamp=entity.stamp) != entity:
            ids.add(entity.id)
            if previous.coop != entity.coop:
                categories.add("cooperation_property")
            if (previous.explored, previous.open_state) != (entity.explored, entity.open_state):
                categories.add("container_state")
            if (previous.room_id, previous.inside, previous.on, previous.holders) != (
                    entity.room_id, entity.inside, entity.on, entity.holders):
                categories.add("placement")
            elif previous.position != entity.position and not entity.holders:
                categories.add("entity_geometry")
    removed = set(old) - {e.id for e in after.entities}
    if removed:
        categories.add("entity_removed")
        ids.update(removed)
    if before.class_coop != after.class_coop:
        categories.add("cooperation_property")
    if replace(before.cooperation, stamp=after.cooperation.stamp) != after.cooperation:
        categories.add("cooperation_state")
    if tuple(before.satisfied(g) for g in before.goals) != tuple(after.satisfied(g) for g in after.goals):
        categories.add("goal_progress")
    if any(replace(a, stamp=b.stamp) != b for a, b in zip(before.agents, after.agents)):
        categories.add("agent_progress")
    if not categories and before != after:
        categories.add("recency")
    return GraphChanges(tuple(sorted(categories)), tuple(sorted(ids)))


def inspection_requires_sync(regions, targets):
    """A newly inspected search region changes the next joint decision."""
    return bool(set(regions) - {"c:" + str(i) for i in targets})


def merge_grounding(graph, entities):
    """Absorb only the factual fragment explicitly delivered with a plan."""
    records = {e.id: e for e in graph.entities}
    for item in entities:
        if item.stamp.step > graph.step:
            raise ValueError("future action grounding")
        old = records.get(item.id)
        if old is None or item.stamp > old.stamp:
            records[item.id] = item
        elif old.name == "unknown_target":
            records[item.id] = replace(old, name=item.name)
        if old is not None:
            records[item.id] = replace(records[item.id], explored=old.explored or item.explored,
                                       coop=old.coop if old.coop is not None else item.coop)
    return normalize_graph(replace(graph, entities=tuple(records[i] for i in sorted(records))))
