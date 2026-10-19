"""Shared-world event evaluation. Only reveal() reads sampled locations.

Candidates share a public continuation policy and finite navigation horizon.
Empty scans preserve priors; coverage schedules revisits. Future scans use an
ideal-discovery proxy, not a calibrated detection model.
"""

from dataclasses import dataclass, replace
from itertools import product
import math
import random
import time
from typing import Optional

from .astar_planner import STATUS_ACTION, STATUS_FAILED, PlanningResult, plan_deterministic_graph
from .config import MethodConfig
from .discovery_belief import DiscoveryBelief
from .models import DeterministicGraph, FactGraph
from .planning_domain import (
    GRAB, MOVE, ROOM, HighLevelAction, JointHighLevelAction,
    PlannerState, PlanningDomain, PlanningDomainError,
)


@dataclass(frozen=True)
class DiscoveryPlan:
    result: PlanningResult
    graph: DeterministicGraph
    diagnostics: dict
    wall_time: float


@dataclass(frozen=True)
class _Candidate:
    schedule: tuple
    scans: tuple = ()
    value: float = 0.0
    inspection_cost: float = 0.0

    @property
    def ratio(self):
        return self.value / max(1.0, self.inspection_cost)


def _joint(left=None, right=None):
    return JointHighLevelAction(left, right)


def _parts(joint):
    return joint.agent_0, joint.agent_1


def _sample_worlds(facts, belief, config):
    rng = random.Random(config.discovery_seed + 1000003 * belief.revision)
    unresolved = [o for o in facts.objects if not o.finished
                  and not o.holder_agent_ids and o.room_node_id is None]
    worlds = []
    for _ in range(config.discovery_world_count):
        world = {}
        for obj in unresolved:
            weights = (belief.instance_weights.get(obj.object_id)
                       if obj.object_id is not None else
                       belief.class_weights.get(obj.class_name))
            if not weights:
                raise ValueError("missing discovery belief for target {}".format(obj.graph_index))
            rooms, probabilities = zip(*sorted(weights.items()))
            world[obj.graph_index] = rng.choices(rooms, probabilities, k=1)[0]
        worlds.append(world)
    return tuple(worlds)


class _NavigationSnapshot:
    """Cache only within one planning round; the public map is then fixed."""
    def __init__(self, navigation):
        self.navigation, self.cache = navigation, {}
        self.config = getattr(navigation, "config", None)

    def get_navigation_path_info(self, start, goal, *, radius=0.0, room_name=None):
        key = tuple(start), tuple(goal), radius, room_name
        if key not in self.cache:
            self.cache[key] = self.navigation.get_navigation_path_info(
                start, goal, radius=radius, room_name=room_name)
        return self.cache[key]


class _EventPlanner:
    def __init__(self, facts, belief, navigation, config, commitments):
        self.facts, self.belief, self.config = facts, belief, config
        self.navigation = _NavigationSnapshot(navigation) if navigation is not None else None
        self.graph = DeterministicGraph.for_known_facts(facts)
        self.unresolved = {o.graph_index: o for o in facts.objects if not o.finished
                           and not o.holder_agent_ids and o.room_node_id is None}
        self.domain = PlanningDomain(self.graph, self.navigation, commitments)
        self.continued_scans = self.domain.continued_scans
        self.cache = {}
        self.stats = dict(astar_calls=0, astar_expanded=0, astar_bounded=0,
                          cache_hits=0, rollout_steps=0, incomplete_rollouts=0)
        initial = _Simulation(self)
        d = initial.domain
        # A public conservative serial reference: a complete coverage sweep,
        # then at most two navigation legs per remaining mandatory object.
        # It bounds the evaluation scale, not real TDW time or feasibility.
        endpoints = [(r.center, r.node_id) for r in facts.rooms]
        endpoints += [(a.position, a.room_node_id) for a in initial.state.agents]
        endpoints += [(o.position, o.room_node_id) for o in facts.objects
                      if o.position is not None and o.room_node_id is not None]
        endpoints.append((d._goal_target(), d.goal_room_node_id))
        legs = [d._path_len(p, q, r, s) for p, r in endpoints for q, s in endpoints]
        unit = max((x for x in legs if math.isfinite(x)), default=1.0)
        passes = initial.passes
        sweep = sum(max(passes.values()) + 1 - n for n in passes.values())
        remaining = sum(not o.finished for o in facts.objects)
        self.horizon = max(1.0, unit) * (sweep + 2 * remaining + 2)

    def known_plan(self, sim):
        key = (sim.domain.graph, sim.state)
        if key in self.cache:
            cached = self.cache[key]
            if cached.status != STATUS_ACTION or sim.executable(cached.selected_action):
                self.stats["cache_hits"] += 1
                return cached
        allowed = tuple(a for a in sim.domain.legal_joint_actions(sim.state)
                        if sim.executable(a))
        result = plan_deterministic_graph(
            sim.domain.graph, navigation=self.navigation, root_state=sim.state,
            allowed_root_actions=allowed,
            max_expansions=self.config.discovery_astar_expansions if self.unresolved else None,
        )
        self.stats["astar_calls"] += 1
        self.stats["astar_expanded"] += result.expanded_states
        self.stats["astar_bounded"] += int(bool(result.reason and "search_budget" in result.reason))
        self.cache[key] = result
        # Cache legal continuations of the same public plan, rather than solve
        # the identical known subproblem again after each zero-cost interaction.
        if result.status == STATUS_ACTION:
            state, cost = sim.state, 0.0
            for index, action in enumerate(result.joint_actions[:-1]):
                following, elapsed = sim.domain.advance(state, action)
                cost += sim.domain.transition_cost(state, elapsed)
                state = following
                suffix = result.joint_actions[index + 1:]
                future_key = (sim.domain.graph, state)
                self.cache.setdefault(future_key, replace(
                    result, selected_action=suffix[0], joint_actions=suffix,
                    total_cost=max(0.0, result.total_cost - cost), expanded_states=0))
        return result

    def room_scores(self, sim, agent_id):
        if len(sim.revealed) == len(self.unresolved):
            return []
        actor = sim.state.agent(agent_id)
        if (actor.current_action is not None or actor.held_object_index is not None
                or agent_id in sim.requests):
            return []
        if any(sim.state.object(a.object_index).property == 2 for a in sim.requests.values()):
            return []
        reserved = {a.current_action.target_room_node_id for a in sim.state.agents
                    if a.current_action is not None and a.current_action.target_kind == ROOM}
        d, rows = sim.domain, []
        for room_id in sim.eligible_rooms():
            if room_id in reserved:
                continue
            action = HighLevelAction.move_room(room_id)
            try:
                approach = d.action_duration(sim.state, agent_id, action)
            except PlanningDomainError:
                continue
            room = d._rooms[room_id]
            carry = d._path_len(room.center, d._goal_target(), room_id, d.goal_room_node_id)
            if not math.isfinite(carry):
                continue
            value = mass = 0.0
            for index, obj in self.unresolved.items():
                if index in sim.revealed:
                    continue
                weights = (self.belief.instance_weights[obj.object_id] if obj.object_id is not None
                           else self.belief.class_weights[obj.class_name])
                p = weights[room_id]
                delivery = approach + carry
                if obj.property == 2:
                    other = sim.state.agent(1 - agent_id)
                    delivery = max(approach, d._path_len(
                        other.position, room.center, other.room_node_id, room_id)) + carry
                # Mandatory instances and distinct rooms make this joint
                # expected reduction additive; no replacement-supply matching.
                value += p * max(0.0, self.horizon - delivery)
                mass += p
            rows.append((value / max(1.0, approach), value, approach, room_id, mass))
        return sorted(rows, key=lambda row: (-row[0], row[2], row[3]))

    def candidates(self):
        sim = _Simulation(self)
        known = self.known_plan(sim)
        candidates = [_Candidate(known.joint_actions)] if known.status == STATUS_ACTION else []
        choices = [self.room_scores(sim, i)[:3] for i in (0, 1)]
        proposals = []
        for left, right in product([None] + choices[0], [None] + choices[1]):
            rows = (left, right)
            scans = tuple((i, row[3]) for i, row in enumerate(rows) if row is not None)
            if not scans or (left and right and left[3] == right[3]):
                continue
            actions = tuple(HighLevelAction.move_room(row[3]) if row else None for row in rows)
            value = sum(row[1] for row in rows if row)
            cost = max(row[2] for row in rows if row)
            variants = [actions]
            if known.status == STATUS_ACTION:
                mixed = tuple(a if a is not None else _parts(known.selected_action)[i]
                              for i, a in enumerate(actions))
                if mixed != actions:
                    variants.append(mixed)
            for pair in variants:
                joint = _joint(*pair)
                if sim.executable(joint):
                    proposals.append(_Candidate((joint,), scans, value, cost))
        proposals.sort(key=lambda c: (-c.ratio, c.inspection_cost, len(c.scans), c.scans,
                                      sim.domain._joint_key(c.schedule[0])))
        seen = {c.schedule for c in candidates}
        for candidate in proposals:
            if len(candidates) >= self.config.discovery_candidate_count:
                break
            if candidate.schedule not in seen:
                candidates.append(candidate)
                seen.add(candidate.schedule)
        if not candidates and any(a.current_action for a in sim.state.agents):
            candidates.append(_Candidate((_joint(),)))
        return candidates

    def prefix(self, candidate, world=None):
        sim = _Simulation(self, world)
        continuing = self.domain.initial_actions
        issued, used = [], [0, 0]
        for joint in candidate.schedule:
            if not sim.executable(joint):
                break
            if any(used[i] >= self.config.action_horizon and action is not None
                   for i, action in enumerate(_parts(joint))):
                break
            # Reinstall the same public semantic commitment, not just its Put
            # suffix, when an online replan occurs between physical substeps.
            issued.append(_joint(*(a or old for a, old in zip(_parts(joint), continuing)))
                          if not issued else joint)
            for i, action in enumerate(_parts(joint)):
                used[i] += int(action is not None)
            event = sim.advance(joint)
            if event or sim.complete or sim.elapsed >= self.horizon:
                break
        else:
            # Wait only to the first reveal. A longer peer action stays active.
            for _ in range(self.config.discovery_rollout_steps - sim.steps):
                if not any(a.current_action for a in sim.state.agents):
                    break
                if sim.advance(_joint()) or sim.elapsed >= self.horizon:
                    break
        return tuple(issued), sim

    def tail_action(self, sim):
        known = self.known_plan(sim)
        if known.status == STATUS_ACTION and sim.executable(known.selected_action):
            pair = list(_parts(known.selected_action))
        else:
            # A bounded failure is not proof of infeasibility.
            legal = [a for a in sim.domain.legal_joint_actions(sim.state) if sim.executable(a)]
            def rank(action):
                state, cost = sim.domain.advance(sim.state, action)
                return cost + sim.domain.heuristic(state), sim.domain._joint_key(action)
            pair = list(_parts(min(legal, key=rank))) if legal else [None, None]
        assigned = {a.target_room_node_id for a in pair if a and a.target_kind == ROOM}
        for i in (0, 1):
            if pair[i] is not None:
                continue
            # C-WAH continuation: known work first, then available discovery.
            if any(sim.domain._individual_actions(sim.state, i)):
                continue
            rows = [row for row in self.room_scores(sim, i) if row[3] not in assigned]
            if rows:
                room = rows[0][3]
                pair[i] = HighLevelAction.move_room(room)
                assigned.add(room)
        joint = _joint(*pair)
        return joint if sim.executable(joint) else None

    def rollout(self, sim):
        seen = set()
        while (not sim.complete and sim.elapsed < self.horizon
               and sim.steps < self.config.discovery_rollout_steps):
            key = (sim.state, tuple(sorted(sim.revealed.items())),
                   tuple(sorted(sim.passes.items())))
            if key in seen:
                break
            seen.add(key)
            action = self.tail_action(sim)
            if action is None:
                break
            sim.advance(action)
        if not sim.complete:
            self.stats["incomplete_rollouts"] += 1
            sim.charge(self.horizon - sim.elapsed)
        return sim.objective


class _Simulation:
    """Private state; sampled truth is accessible only at a completed scan."""
    def __init__(self, planner, world=None):
        self.planner, self.world = planner, world
        self.revealed = {}
        self.domain = planner.domain
        self.state = self.domain.initial_state
        self.passes = {r.node_id: r.revisit_count + 1 if r.explored else 0
                       for r in planner.facts.rooms}
        self.probed = set()
        self.elapsed = self.objective = 0.0
        self.steps = 0

    @property
    def requests(self):
        return {a.agent_id: a.request for a in self.state.agents if a.request is not None}

    @property
    def complete(self):
        return (len(self.revealed) == len(self.planner.unresolved)
                and self.state.is_goal())

    def eligible_rooms(self):
        tier = min(self.passes.values())
        return tuple(r for r, n in sorted(self.passes.items()) if n == tier)

    def executable(self, joint):
        pair = _parts(joint)
        for i, action in enumerate(pair):
            actor = self.state.agent(i)
            if action is None:
                continue
            if actor.current_action is not None:
                return False
            if i in self.requests and action != self.requests[i]:
                return False
            if action.kind == MOVE and action.target_kind == ROOM:
                if actor.held_object_index is not None or action.target_room_node_id not in self.eligible_rooms():
                    return False
            elif action not in self.domain._individual_actions(self.state, i):
                return False
        return self.domain._is_joint_legal(self.state, pair)

    def charge(self, cost):
        cost = min(cost, max(0.0, self.planner.horizon - self.elapsed))
        self.objective += self.domain.transition_cost(self.state, cost)
        self.elapsed += cost

    def advance(self, joint):
        before = self.state
        actions = tuple(a.current_action or submitted
                        for a, submitted in zip(before.agents, _parts(joint)))
        after, cost = self.domain.advance(before, joint)
        remaining = self.planner.horizon - self.elapsed
        self.charge(cost)
        self.steps += 1
        self.planner.stats["rollout_steps"] += 1
        if cost > remaining + 1e-9:
            return False
        self.state = after
        scans, boundary = [], False
        for i, action in enumerate(actions):
            if action is None or after.agent(i).current_action is not None:
                continue
            if action.target_kind == ROOM:
                scans.append(action.target_room_node_id)
            if action.kind == GRAB and action.object_index not in self.probed:
                original = next(o for o in self.planner.facts.objects if o.graph_index == action.object_index)
                boundary |= original.property is None
                self.probed.add(action.object_index)
        if scans:
            self.reveal(scans)
        return bool(scans) or boundary

    def reveal(self, rooms):
        for room in set(rooms):
            self.passes[room] += 1
            if self.world is not None:
                self.revealed.update((index, r) for index, r in self.world.items() if r == room)
        graph = DeterministicGraph.for_known_facts(self.planner.facts, self.revealed)
        domain = PlanningDomain(graph, self.planner.navigation)
        old = {o.graph_index: o for o in self.state.objects}
        self.state = PlannerState(
            self.state.agents,
            tuple(old.get(o.graph_index, o) for o in domain.initial_state.objects),
            tuple(sorted(set(self.state.explored_room_ids) | set(rooms))),
            self.state.holder,
        )
        self.domain = domain


def _stderr(values):
    mean = sum(values) / len(values)
    return math.sqrt(sum((x - mean) ** 2 for x in values) / max(1, len(values) - 1) / len(values))


def _select_candidate(rows):
    """Compare full public continuations, never pretend unknown work is done."""
    best = min(range(len(rows)), key=lambda i: (rows[i]["score"], i))
    baseline = next((i for i, row in enumerate(rows) if row["transport_baseline"]), None)
    for row in rows:
        delta = [a - b for a, b in zip(row["scores"], rows[best]["scores"])]
        row.update(paired_delta=sum(delta) / len(delta), paired_stderr=_stderr(delta))
        if baseline is not None:
            delta = [a - b for a, b in zip(row["scores"], rows[baseline]["scores"])]
            row.update(baseline_delta=sum(delta) / len(delta), baseline_stderr=_stderr(delta))
    if baseline is not None:
        eligible = [i for i, row in enumerate(rows) if i == baseline
                    or row["baseline_delta"] + 2 * row["baseline_stderr"] < -1e-9]
        return min(eligible, key=lambda i: (rows[i]["score"], i))
    tied = [i for i, row in enumerate(rows)
            if row["paired_delta"] <= row["paired_stderr"] + 1e-9]
    return min(tied, key=lambda i: (-rows[i]["value_per_cost"], rows[i]["inspection_cost"], i))


def plan_discovery(facts: FactGraph, belief: DiscoveryBelief, navigation=None,
                   config: Optional[MethodConfig] = None, commitments=()) -> DiscoveryPlan:
    started = time.perf_counter()
    config = config or MethodConfig()
    facts.validate()
    graph = DeterministicGraph.for_known_facts(facts)
    if not any(not o.finished and not o.holder_agent_ids and o.room_node_id is None
               for o in facts.objects):
        result = plan_deterministic_graph(graph, navigation=navigation, commitments=commitments)
        return DiscoveryPlan(result, graph, {
            "cost_unit": "penalized_navigation_path_len", "candidate_count": 0,
            "world_count": 0, "selected_scan_rooms": [],
        }, time.perf_counter() - started)
    try:
        planner = _EventPlanner(facts, belief, navigation, config, commitments)
        candidates = planner.candidates()
        if not candidates:
            raise PlanningDomainError("no executable discovery candidate")
        worlds = _sample_worlds(facts, belief, config)
        rows, prefixes = [], []
        for candidate in candidates:
            prefix, _ = planner.prefix(candidate)
            if not prefix:
                continue
            scores, covered = [], []
            for world in worlds:
                same, sim = planner.prefix(candidate, world)
                if same != prefix:
                    raise PlanningDomainError("sample truth changed the common prefix")
                covered.append(len(sim.revealed))
                scores.append(planner.rollout(sim))
            prefixes.append(prefix)
            rows.append(dict(score=sum(scores) / len(scores), scores=scores,
                             sample_stderr=_stderr(scores), scans=list(candidate.scans),
                             screening_value=candidate.value, value_per_cost=candidate.ratio,
                             inspection_cost=candidate.inspection_cost,
                             transport_baseline=not candidate.scans and any(
                                 a is not None and a.target_kind != ROOM
                                 for j in candidate.schedule for a in _parts(j)),
                             expected_discoveries=sum(covered) / len(covered)))
        if not rows:
            raise PlanningDomainError("no executable observation prefix")
        selected = _select_candidate(rows)
        row, prefix = rows[selected], prefixes[selected]
        rooms = {a.target_room_node_id for joint in prefix for a in _parts(joint)
                 if a is not None and a.target_kind == ROOM}
        graph = DeterministicGraph.for_known_facts(facts, reopened_room_ids=(
            r.node_id for r in facts.rooms if r.explored and r.node_id in rooms))
        result = PlanningResult(STATUS_ACTION, prefix[0], row["score"], prefix,
                                planner.stats["astar_expanded"], graph.assumptions,
                                graph.reopened_room_ids, None, tuple(planner.continued_scans))
        initial = _Simulation(planner)
        diagnostics = dict(
            cost_unit="penalized_navigation_path_len",
            candidate_count=len(rows), world_count=len(worlds), shared_worlds=True,
            selected_candidate=selected, selected_scan_rooms=[r for _, r in row["scans"]],
            expected_total_cost=row["score"], expected_covered_target_mass=row["expected_discoveries"],
            eligible_scan_rooms=list(initial.eligible_rooms()), candidates=rows,
            cost_horizon=planner.horizon, rollout_step_limit=config.discovery_rollout_steps,
            astar_expansion_limit=config.discovery_astar_expansions,
            negative_scan_update="none_non_conclusive", scan_model="ideal_future_discovery_proxy",
            hidden_locations_exposed_to_astar=False, **planner.stats,
        )
    except (PlanningDomainError, ValueError, KeyError) as error:
        result = PlanningResult(STATUS_FAILED, None, None, (), 0, graph.assumptions,
                                graph.reopened_room_ids, str(error))
        diagnostics = {"error": str(error)}
    return DiscoveryPlan(result, graph, diagnostics, time.perf_counter() - started)
