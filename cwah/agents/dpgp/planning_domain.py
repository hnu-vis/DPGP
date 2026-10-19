"""Deterministic, non-preemptive joint actions at finite completion events."""

from collections import Counter
from dataclasses import dataclass, replace
from functools import lru_cache
import heapq
from itertools import product
import math

from .config import MethodConfig
from .models import Action, distance

EPS = 1e-9


@dataclass(frozen=True)
class Actor:
    room: int
    position: tuple
    held: object = None
    action: object = None
    remaining: float = 0.0
    request: object = None
    close: tuple = ()


@dataclass(frozen=True)
class SearchState:
    actors: tuple
    opened: tuple
    delivered: tuple
    remaining: tuple
    holder: object = None

    @property
    def complete(self):
        return not any(self.remaining)


def action_key(action):
    return ("", 0, 0, False) if action is None else (
        action.kind, action.target_id or 0, action.object_id or 0, action.cooperative)


class PlanningDomain:
    def __init__(self, resolved, config=None, *, partial=False):
        if partial:
            resolved.graph.validate(allow_virtual=True)
        else:
            resolved.validate()
        self.resolved, self.graph = resolved, resolved.graph
        self.config = config or MethodConfig()
        self.rooms = {r.id: r for r in self.graph.rooms}
        self.entities = {e.id: e for e in self.graph.entities}
        self.portals = {(p.source, p.target): p.position for p in self.graph.portals}
        self.neighbors = {key: [] for key in self.rooms}
        for a, b in self.portals:
            self.neighbors[a].append(b)
            self.neighbors[b].append(a)
        self.goals = self.graph.goals
        for target in {g.target_id for g in self.goals}:
            if len({g.relation for g in self.goals if g.target_id == target}) != 1:
                raise ValueError("ambiguous ON/INSIDE role for one target")
        satisfied = {i for g in self.goals for i in self.graph.satisfied(g)}
        active = {g.object_class for g in self.goals if self.graph.remaining(g)}
        self.objects = {e.id: e for e in self.graph.entities if e.name in active and e.id not in satisfied
                        and e.room_id is not None and e.position is not None}
        self.total_goals = max(1, sum(g.count for g in self.goals))
        actors = tuple(Actor(a.room_id, a.position, next(iter(a.held), None), close=a.close) for a in self.graph.agents)
        coop = self.graph.cooperation
        if coop.object_id is not None:
            actors = tuple(replace(a, held=coop.object_id) for a in actors)
        state = SearchState(actors, tuple(e.id for e in self.graph.entities if e.container and e.open_state == "open"),
                            (), tuple(self.graph.remaining(g) for g in self.goals), coop.holder)
        # Commitments are already-issued semantic work; include them in the plan
        # prefix so the real executor can keep, rather than cancel, its current work.
        initial = [None, None]
        actors = list(state.actors)
        for i, metadata in enumerate(self.graph.agents):
            commitment = metadata.commitment
            if commitment is None or state.complete:
                continue
            action = commitment.action
            if action.kind in ("Wait", "Continue") or self.action_done(state, i, action):
                continue
            valid_move = (action.kind == "Move" and ((action.target_id in self.neighbors.get(actors[i].room, ()))
                          or (action.target_id in self.entities and self.entities[action.target_id].room_id == actors[i].room)))
            locked_request = (action.kind == "Grab" and action.cooperative and commitment.phase == "coop_wait"
                              and action.target_id in self.objects and actors[i].held is None)
            if not valid_move and not locked_request and action not in self.individual_actions(state, i):
                # Stale completed/invalid plans do not create fictitious progress.
                continue
            initial[i] = action
            if action.kind == "Grab" and action.cooperative and commitment.phase == "coop_wait":
                actors[i] = replace(actors[i], request=action.target_id)
            else:
                duration = commitment.remaining
                if duration is None:
                    duration = self.duration(state, i, action)
                actors[i] = replace(actors[i], action=action, remaining=duration)
        if state.holder is not None and initial[0] != initial[1]:
            # Carrying is one synchronous resource. Facts/holder position define
            # the next common action, not mismatched physical execution phases.
            initial = [None, None]
            actors = list(state.actors)
        elif state.holder is not None and any(initial):
            remaining = max(a.remaining for a in actors)
            actors = [replace(a, remaining=remaining) for a in actors]
        self.initial = self.settle(replace(state, actors=tuple(actors)))
        self.initial_actions = tuple(initial)

    @lru_cache(maxsize=8192)
    def route(self, position, source, target, endpoint):
        """Shortest portal route; Move always selects just its next room."""
        pending = [(0., (source,), position)]
        best, answer = {}, (math.inf, ())
        while pending:
            cost, path, point = heapq.heappop(pending)
            room = path[-1]
            if cost > best.get((room, point), math.inf) + EPS:
                continue
            if room == target:
                answer = min(answer, (cost + distance(point, endpoint) / self.config.movement_speed, path))
                continue
            for neighbor in sorted(self.neighbors[room]):
                if neighbor in path:
                    continue
                portal = self.portals[tuple(sorted((room, neighbor)))]
                new = cost + distance(point, portal) / self.config.movement_speed
                if new < best.get((neighbor, portal), math.inf) - EPS:
                    best[neighbor, portal] = new
                    heapq.heappush(pending, (new, path + (neighbor,), portal))
        return answer

    def point(self, key):
        entity = self.entities[key]
        # A contained object's interaction point is its container, even after open.
        if key in self.objects and entity.inside in self.entities:
            return self.entities[entity.inside].position
        return entity.position

    def at(self, actor, key):
        return key in actor.close or distance(actor.position, self.point(key)) <= EPS

    def toward(self, actor, key, cooperative=False):
        entity = self.entities[key]
        if actor.room != entity.room_id:
            _, path = self.route(actor.position, actor.room, entity.room_id, self.point(key))
            if len(path) < 2:
                raise ValueError("no route to task entity")
            return Action("Move", path[1], cooperative=cooperative)
        if not self.at(actor, key):
            return Action("Move", key, cooperative=cooperative)
        return None

    def pickup(self, state, index, key):
        actor, obj = state.actors[index], self.objects[key]
        parent = self.entities.get(obj.inside)
        target = parent.id if parent and parent.container else key
        move = self.toward(actor, target)
        if move:
            return move
        if parent and parent.container and parent.id not in state.opened:
            return Action("CheckContainer", parent.id)
        return Action("Grab", key, cooperative=obj.coop == 2)

    def goal_index(self, action):
        name = self.objects[action.object_id].name
        return next(i for i, g in enumerate(self.goals) if g.target_id == action.target_id and g.object_class == name)

    def deliveries(self, state, index):
        actor = state.actors[index]
        if actor.held not in self.objects:
            raise ValueError("held object cannot be allocated to an unfinished goal")
        choices = []
        for i, goal in enumerate(self.goals):
            if not state.remaining[i] or goal.object_class != self.objects[actor.held].name:
                continue
            if self.entities[goal.target_id].room_id is None:
                continue
            reserved = {a.action.object_id for a in state.actors if a.action and a.action.kind == "Put"
                        and self.goal_index(a.action) == i and a.action.object_id != actor.held}
            if len(reserved) >= state.remaining[i]:
                continue
            coop = state.holder is not None
            action = self.toward(actor, goal.target_id, coop)
            if action is None:
                action = (Action("CheckContainer", goal.target_id, cooperative=coop)
                          if goal.relation == "INSIDE" and goal.target_id not in state.opened else
                          Action("Put", goal.target_id, actor.held, coop))
            rank, _ = self.route(actor.position, actor.room, self.entities[goal.target_id].room_id,
                                 self.point(goal.target_id))
            choices.append((rank, goal.key, action))
        return (min(choices, key=lambda v: v[:2])[2],) if choices else (None,)

    def requests(self, state):
        return tuple((i, a.request if a.request is not None else a.action.target_id)
                     for i, a in enumerate(state.actors) if a.request is not None or
                     (a.action and a.action.kind == "Grab" and a.action.cooperative))

    def individual_actions(self, state, index):
        actor = state.actors[index]
        if actor.action:
            return (None,)
        if actor.request is not None:
            return (Action("Wait"),)
        if actor.held is not None:
            return self.deliveries(state, index)
        requests = self.requests(state)
        if requests:
            return (self.pickup(state, index, requests[0][1]),)
        reserved = {a.held for a in state.actors if a.held is not None}
        reserved.update(a.action.target_id for a in state.actors if a.action and a.action.kind == "Grab")
        counts = Counter(self.objects[k].name for k in reserved)
        demand = Counter()
        for goal, remaining in zip(self.goals, state.remaining):
            demand[goal.object_class] += remaining
        peer = state.actors[1 - index]
        candidates = {}
        for key, obj in self.objects.items():
            if key in state.delivered or key in reserved or demand[obj.name] <= counts[obj.name]:
                continue
            if obj.coop == 1 and peer.action and peer.action.kind == "Move" and peer.action.target_id == key:
                continue
            rank, _ = self.route(actor.position, actor.room, obj.room_id, self.point(key))
            candidates.setdefault(obj.name, []).append((rank, key))
        actions = {self.pickup(state, index, key) for rows in candidates.values()
                   for _, key in sorted(rows)[:self.config.pickup_candidates_per_class]}
        ready = {a for a in actions if a.kind != "Move"}
        return tuple(sorted(ready or actions, key=action_key)) or (None,)

    def legal(self, state, joint):
        effective = []
        for actor, action in zip(state.actors, joint):
            if actor.action and action is not None:
                return False
            if actor.request is not None and action != Action("Wait"):
                return False
            if actor.request is None and action is not None and action.kind in ("Wait", "Continue"):
                return False
            effective.append(actor.action or action)
        if state.holder is not None and effective[0] != effective[1]:
            return False
        if not any(a.action for a in state.actors) and not any(a and a.kind != "Wait" for a in joint):
            return False
        requests = {key for _, key in self.requests(state)} | {
            a.target_id for a in effective if a and a.kind == "Grab" and a.cooperative}
        if len(requests) > 1:
            return False
        ordinary = [a.target_id for a in effective if a and a.kind in ("Grab", "Move")
                    and a.target_id in self.objects and self.objects[a.target_id].coop == 1]
        if len(ordinary) != len(set(ordinary)):
            return False
        reserved = {a.held for a in state.actors if a.held is not None}
        reserved |= {a.target_id for a in effective if a and a.kind == "Grab"}
        counts = Counter(self.objects[key].name for key in reserved)
        demand = Counter()
        for goal, remaining in zip(self.goals, state.remaining):
            demand[goal.object_class] += remaining
        if any(counts[name] > demand[name] for name in counts):
            return False
        puts = {(a.object_id, self.goal_index(a)) for a in effective if a and a.kind == "Put"}
        return all(n <= state.remaining[g] for g, n in Counter(g for _, g in puts).items())

    def joint_actions(self, state):
        if state.complete:
            return ()
        if state.holder is not None:
            choices = ((None, None),) if any(a.action for a in state.actors) else tuple(
                (a, a) for a in self.deliveries(state, state.holder))
        else:
            individual = [self.individual_actions(state, i) for i in (0, 1)]
            choices = tuple(product(*individual))
        result = [pair for pair in choices if self.legal(state, pair)]
        if not result and state.holder is None:
            # Forced inactivity is not a selectable ordinary Wait. Only use it
            # when scarce work prevents every simultaneous assignment.
            forced = []
            for i in (0, 1):
                if state.actors[1 - i].action or state.actors[1 - i].request is not None:
                    continue
                for action in individual[i]:
                    pair = (action, None) if i == 0 else (None, action)
                    if action and action.kind != "Wait" and self.legal(state, pair):
                        forced.append(((action.kind == "Move", self.duration(state, i, action), i),
                                       tuple(map(action_key, pair)), pair))
            if forced:
                result = [min(forced, key=lambda row: row[:2])[2]]
        return tuple(sorted(set(result), key=lambda p: tuple(map(action_key, p))))

    def endpoint(self, actor, action):
        if action.kind == "Move" and action.target_id in self.rooms:
            return action.target_id, self.portals[tuple(sorted((actor.room, action.target_id)))], ()
        key = action.target_id
        return self.entities[key].room_id, self.point(key), (key,)

    def duration(self, state, index, action):
        if action.kind == "Move":
            actor = state.actors[state.holder if state.holder is not None else index]
            return distance(actor.position, self.endpoint(actor, action)[1]) / self.config.movement_speed
        if action.kind in ("Grab", "Put", "CheckContainer"):
            return self.config.interaction_cost
        raise ValueError("Wait/Continue have no finite action duration")

    def action_done(self, state, index, action):
        actor = state.actors[index]
        if action.kind == "Move":
            return actor.room == action.target_id if action.target_id in self.rooms else self.at(actor, action.target_id)
        if action.kind == "CheckContainer":
            return action.target_id in state.opened
        if action.kind == "Grab":
            return actor.held == action.target_id
        if action.kind == "Put":
            return action.object_id in state.delivered or action.object_id not in self.objects
        return False

    def transition(self, state, joint, *, check=True):
        if check and joint not in self.joint_actions(state):
            raise ValueError("illegal joint action")
        actors = list(state.actors)
        for i, action in enumerate(joint):
            if action and action.kind != "Wait":
                actors[i] = replace(actors[i], action=action, remaining=self.duration(state, i, action))
        started = replace(state, actors=tuple(actors))
        settled = self.settle(started)
        if settled != started:
            return settled, 0.
        finite = [a.remaining for a in actors if a.action and a.remaining > EPS]
        if not finite:
            raise ValueError("no finite completion event")
        elapsed = min(finite)
        actors = tuple(replace(a, remaining=max(0., round(a.remaining - elapsed, 12))) if a.action else a for a in actors)
        return self.settle(replace(started, actors=actors)), elapsed

    def settle(self, state):
        actors, opened, delivered, remaining = list(state.actors), set(state.opened), set(state.delivered), list(state.remaining)
        holder = state.holder
        completed = [(i, a.action) for i, a in enumerate(actors) if a.action and a.remaining <= EPS]
        for i, action in completed:
            actor = actors[i]
            actors[i] = replace(actor, action=None, remaining=0.)
            if action.kind == "Move":
                room, point, close = self.endpoint(actor, action)
                actors[i] = replace(actors[i], room=room, position=point, close=close)
            elif action.kind == "CheckContainer":
                opened.add(action.target_id)
            elif action.kind == "Grab":
                actors[i] = replace(actors[i], **({"request": action.target_id} if action.cooperative else {"held": action.target_id}))
            elif action.kind == "Put":
                if action.object_id not in delivered:
                    remaining[self.goal_index(action)] -= 1
                    delivered.add(action.object_id)
                actors[i] = replace(actors[i], held=None)
                if action.cooperative:
                    holder = None
        if actors[0].request is not None and actors[0].request == actors[1].request:
            key = actors[0].request
            holder = min((0, 1), key=lambda i: (distance(actors[i].position, self.point(key)), i))
            leader = actors[holder]
            actors = [replace(a, held=key, request=None, room=leader.room, position=leader.position, close=leader.close)
                      for a in actors]
        return SearchState(tuple(actors), tuple(sorted(opened)), tuple(sorted(delivered)), tuple(remaining), holder)

    # ==================== Coop: Synchronous Candidate Evaluation ====================
    def executable(self, state, joint):
        """Validate factual exploration as well as the restricted A* action set."""
        if not self.legal(state, joint):
            return False
        for i, action in enumerate(joint):
            if action is None or action.kind == "Wait":
                continue
            actor, key = state.actors[i], action.target_id
            if action.kind == "Move":
                if key in self.rooms:
                    if key not in self.neighbors[actor.room]:
                        return False
                elif key not in self.entities or self.entities[key].room_id != actor.room:
                    return False
            elif action.kind == "CheckContainer":
                entity = self.entities.get(key)
                if (not entity or not entity.container or entity.room_id != actor.room
                        or key in state.opened or not self.at(actor, key)):
                    return False
            elif action.kind == "Grab":
                if (key not in self.objects or actor.held is not None or key in state.delivered
                        or self.pickup(state, i, key) != action):
                    return False
            elif action.kind == "Put":
                if actor.held != action.object_id or action not in self.deliveries(state, i):
                    return False
            else:
                return False
        return True

    def exploration_action(self, state, index, region):
        """Return one grounded action towards a room or an observed container."""
        actor = state.actors[index]
        if actor.action or actor.request is not None or actor.held is not None:
            return None
        kind, raw = region.split(":")
        key = int(raw)
        if kind == "v":
            if actor.room == key:
                return None
            _, path = self.route(actor.position, actor.room, key, self.rooms[key].center)
            return Action("Move", path[1]) if len(path) > 1 else None
        if kind == "c" and key not in state.opened:
            return self.toward(actor, key) or Action("CheckContainer", key)
        return None

    def synchronous_step(self, state, joint, *, step_distance=None, message_times=(0., 0.),
                         communication_mode="parallel_side_channel"):
        """One environment action opportunity; wall time never advances extra moves.

        Actor.remaining stays in the original movement proxy units. Interactions
        commit once at this step boundary regardless of their calibrated charge.
        All action effects and cooperative pairing are shared with settle().
        """
        from envs.physical_time import PHYSICAL_TIME_SETTINGS, seconds_to_equivalent_frames
        if communication_mode not in ("exclusive_action", "parallel_side_channel"):
            raise ValueError("invalid communication mode")
        if len(message_times) != 2 or any(not math.isfinite(t) or t < 0 for t in message_times):
            raise ValueError("invalid message durations")
        active = any(a.action for a in state.actors) or any(a and a.kind != "Wait" for a in joint)
        if active and not self.executable(state, joint):
            raise ValueError("illegal synchronous candidate")
        if not active and any(a not in (None, Action("Wait")) for a in joint):
            raise ValueError("invalid idle candidate")
        stride = self.config.navigation_step_distance if step_distance is None else step_distance
        if not math.isfinite(stride) or stride <= 0:
            raise ValueError("step distance must be positive")
        actors, charges, attempts = list(state.actors), [0., 0.], [False, False]
        timing = PHYSICAL_TIME_SETTINGS
        for i, choice in enumerate(joint):
            if communication_mode == "exclusive_action" and message_times[i] > 0:
                continue
            actor = actors[i]
            action = actor.action or choice
            if action is None or action.kind == "Wait":
                continue
            attempts[i] = True
            if actor.action is None:
                actor = replace(actor, action=action, remaining=self.duration(state, i, action))
            if action.kind == "Move":
                travel = min(stride, actor.remaining * self.config.movement_speed)
                charges[i] = travel / timing.movement_speed
                left = max(0., actor.remaining - travel / self.config.movement_speed)
                point = list(actor.position)
                endpoint = self.endpoint(actor, action)[1]
                budget = travel
                for axis in (0, 2):
                    delta = endpoint[axis] - point[axis]
                    moved = min(abs(delta), budget)
                    point[axis] += math.copysign(moved, delta)
                    budget -= moved
                actor = replace(actor, position=tuple(point), close=(), remaining=left)
            else:
                charges[i] = {"Grab": timing.grab_time, "Put": timing.put_time,
                              "CheckContainer": timing.open_time}[action.kind]
                actor = replace(actor, remaining=0.)
            actors[i] = actor
        seconds = max(*(seconds_to_equivalent_frames(t, timing.frame_rate) / timing.frame_rate
                        for t in charges), *message_times,
                      1. / timing.frame_rate if not any(attempts) and not any(message_times) else 0.)
        return self.settle(replace(state, actors=tuple(actors))), seconds
    # ==================== End Coop: Synchronous Candidate Evaluation ====================

    def lower_bound(self, state):
        """Independent deliveries with free future origins and no opening cost.

        Ignore reservations, rendezvous, obstacles and other jobs. Reuse candidates
        across goal targets as a relaxation. No travel/interaction can be counted
        more than once in the work bound; parallel capacity is at most two.
        """
        values_by_class = {}
        for key, obj in self.objects.items():
            if key in state.delivered:
                continue
            held = [a for a in state.actors if a.held == key]
            grabs = [a for a in state.actors if a.request == key or (a.action and a.action.kind == "Grab" and a.action.target_id == key)]
            puts = [a for a in state.actors if a.action and a.action.kind == "Put" and a.action.object_id == key]
            options = []
            for goal, count in zip(self.goals, state.remaining):
                if not count or goal.object_class != obj.name:
                    continue
                if puts:
                    options.append(min(a.remaining for a in puts))
                    continue
                points = [self.point(key)]
                if held:
                    points = [a.position for a in held]
                    points += [self.endpoint(a, a.action)[1] for a in held if a.action and a.action.kind == "Move"]
                else:
                    # CLOSE permits interaction at the actor's actual position,
                    # not necessarily at the object's centre. Preserve that
                    # relaxation or the geometric bound can overestimate.
                    parent = self.entities.get(obj.inside)
                    anchor = parent.id if parent and parent.container else key
                    points += [a.position for a in state.actors if self.at(a, anchor)]
                travel = min(distance(p, self.point(goal.target_id)) for p in points) / self.config.movement_speed
                if any(goal.target_id in a.close for a in state.actors):
                    travel = 0.  # Optimistic shared access, not a physical transfer.
                pickup = 0. if held or any(a.request == key for a in grabs) else (
                    min(a.remaining for a in grabs) if grabs else self.config.interaction_cost)
                options.append(travel + pickup + self.config.interaction_cost)
            if options:
                values_by_class.setdefault(obj.name, []).append(min(options))
        demand = Counter()
        for goal, remaining in zip(self.goals, state.remaining):
            demand[goal.object_class] += remaining
        bounds = []
        for name, count in demand.items():
            available = sorted(values_by_class.get(name, []))
            if len(available) < count:
                return math.inf
            bounds += available[:count]
        return max(max(bounds, default=0.), sum(bounds) / 2) + self.config.completion_latency_weight * sum(bounds) / self.total_goals
