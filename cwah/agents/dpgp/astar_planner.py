"""One A* decision module for a deterministic graph; no discovery event scorer."""

from dataclasses import dataclass
import heapq
from itertools import count
import math
import time

from .planning_domain import EPS, PlanningDomain, action_key


class PlanningError(RuntimeError):
    pass


@dataclass(frozen=True)
class PlanStep:
    time: float
    actions: tuple
    durations: tuple


@dataclass(frozen=True)
class SearchResult:
    schedule: tuple
    objective: float
    makespan: float
    expanded: int
    wall_time: float
    optimal: bool = True


def _step(domain, state, actions, elapsed):
    return PlanStep(elapsed, actions, tuple(domain.duration(state, i, a) if a and a.kind != "Wait" else None
                                           for i, a in enumerate(actions)))


def solve(resolved, config=None, *, use_heuristic=True, root_state=None, max_expansions=None, partial=False):
    started = time.monotonic()
    domain = PlanningDomain(resolved, config, partial=partial)
    root = domain.initial if root_state is None else root_state
    if root.complete:
        return SearchResult((), 0., 0., 0, time.monotonic() - started)
    initial_steps = ()
    initial_actions = domain.initial_actions if root_state is None else tuple(a.action for a in root.actors)
    if any(initial_actions):
        initial_steps = (PlanStep(0., initial_actions, tuple(
            a.remaining if initial and a.action else 0. if initial else None
            for initial, a in zip(initial_actions, root.actors))),)
    serial = count()
    heuristic = domain.lower_bound if use_heuristic else lambda state: 0.
    h_cache = {}

    def h(state):
        if state not in h_cache:
            h_cache[state] = heuristic(state)
        return h_cache[state]

    # A cheap feasible rollout supplies an upper bound; failure never replaces A*.
    def greedy():
        state, seen, g, elapsed, steps = root, {root}, 0., 0., list(initial_steps)
        for _ in range(256):
            if state.complete:
                return g, elapsed, tuple(steps)
            candidates = []
            for joint in domain.joint_actions(state):
                nxt, dt = domain.transition(state, joint, check=False)
                if nxt in seen:
                    continue
                edge = dt * (1 + domain.config.completion_latency_weight * sum(state.remaining) / domain.total_goals)
                candidates.append((edge + h(nxt), sum(nxt.remaining), tuple(map(action_key, joint)), joint, nxt, dt, edge))
            if not candidates:
                return None
            _, _, _, joint, nxt, dt, edge = min(candidates, key=lambda row: row[:3])
            steps.append(_step(domain, state, joint, elapsed))
            state, g, elapsed = nxt, g + edge, elapsed + dt
            seen.add(state)
        return None

    incumbent = greedy() if use_heuristic else None
    upper = incumbent[0] if incumbent else math.inf
    # Node records retain physical states even if later search compression is added.
    nodes = [(root, 0., 0., None, None)]
    pending = [(h(root), 0., next(serial), 0)]
    best = {root: 0.}
    expanded = 0
    bounded = False
    while pending:
        if max_expansions is not None and expanded >= max_expansions:
            bounded = True
            break
        estimate, _, _, node_id = heapq.heappop(pending)
        if estimate >= upper - EPS:
            break
        state, g, elapsed, parent, incoming = nodes[node_id]
        if g > best.get(state, math.inf) + EPS:
            continue
        if state.complete:
            steps, current = [], node_id
            while nodes[current][3] is not None:
                steps.append(nodes[current][4])
                current = nodes[current][3]
            incumbent = (g, elapsed, initial_steps + tuple(reversed(steps)))
            upper = g
            continue
        expanded += 1
        for joint in domain.joint_actions(state):
            nxt, dt = domain.transition(state, joint, check=False)
            new_g = g + dt * (1 + domain.config.completion_latency_weight * sum(state.remaining) / domain.total_goals)
            if new_g >= best.get(nxt, math.inf) - EPS or new_g + h(nxt) >= upper - EPS:
                continue
            best[nxt] = new_g
            new_id = len(nodes)
            nodes.append((nxt, new_g, elapsed + dt, node_id, _step(domain, state, joint, elapsed)))
            heapq.heappush(pending, (new_g + h(nxt), -new_g, next(serial), new_id))
    if incumbent is None:
        if bounded:
            raise PlanningError("search_budget_exhausted_without_incumbent")
        raise PlanningError("no plan in the task-directed action domain (expanded={})".format(expanded))
    objective, makespan, schedule = incumbent
    return SearchResult(schedule, objective, makespan, expanded, time.monotonic() - started, not bounded)


def project_prefixes(result, resolved, facts, horizon=8):
    """A common observation/horizon boundary prevents mismatched carry tails."""
    if horizon < 1:
        raise ValueError("horizon must be positive")
    entities, rooms = {e.id: e for e in facts.entities}, {r.id: r for r in facts.rooms}
    defaulted = {a.entity_id for a in resolved.assumptions if a.field == "coop"}
    predicted = {a.entity_id for a in resolved.assumptions if a.field == "location"}
    barrier = math.inf
    for step in result.schedule:
        for action, duration in zip(step.actions, step.durations):
            if action is None or action.kind in ("Wait", "Continue"):
                continue
            keys = [k for k in (action.target_id, action.object_id) if k is not None]
            if any(k in predicted or (k not in rooms and (k not in entities or entities[k].stamp.step < 0
                                                         or entities[k].room_id is None)) for k in keys):
                barrier = min(barrier, step.time)
                continue
            observes = ((action.kind == "Move" and action.target_id in rooms and not rooms[action.target_id].explored)
                        or (action.kind == "CheckContainer" and not entities[action.target_id].explored)
                        or (action.kind == "Grab" and action.target_id in defaulted))
            if observes:
                barrier = min(barrier, step.time + (duration or 0.))
    prefixes = [[], []]
    for step in result.schedule:
        if step.time >= barrier - EPS:
            # The triggering zero-duration room crossing is still executable.
            if step.time > barrier + EPS:
                break
            allowed_zero = any(a and a.kind == "Move" and a.target_id in rooms
                               and not rooms[a.target_id].explored and d == 0.
                               for a, d in zip(step.actions, step.durations))
            if not allowed_zero:
                break
        additions = [(i, a) for i, a in enumerate(step.actions) if a and a.kind not in ("Wait", "Continue")]
        if any(len(prefixes[i]) >= horizon for i, _ in additions):
            break
        for i, action in additions:
            keys = [k for k in (action.target_id, action.object_id) if k is not None]
            if any(k in predicted or (k not in rooms and k not in entities) for k in keys):
                raise PlanningError("prediction boundary would expose an ungrounded action")
            prefixes[i].append(action)
        if step.time >= barrier - EPS:
            break
    if not any(prefixes) and any(facts.remaining(g) for g in facts.goals):
        raise PlanningError("nonterminal plan projects to two empty prefixes")
    return tuple(tuple(p) for p in prefixes)
