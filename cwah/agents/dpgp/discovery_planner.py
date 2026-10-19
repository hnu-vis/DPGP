"""One observation-event lookahead with public-history policy rollouts.

World samples answer inspections only. The acting policy and deterministic A*
receive revealed graphs, never the samples. The same fixed-horizon synchronous
cost is used for every candidate, including misses and unsupported supply.
"""

from collections import Counter, deque
from dataclasses import dataclass, replace
from itertools import product
import math
import random
import time

from .astar_planner import PlanningError, solve
from .config import MethodConfig
from .discovery_belief import DiscoveryBelief, partition, region_group
from .coordination import inspection_requires_sync
from .models import Action, Entity, PlanningGraph, Stamp
from .planning_domain import EPS, PlanningDomain, action_key
from .prediction import factual_planning_graph
from envs.physical_time import PHYSICAL_TIME_SETTINGS


@dataclass(frozen=True)
class DiscoveryResult:
    prefixes: tuple
    objective: float
    makespan: float
    expanded: int
    wall_time: float
    diagnostics: dict


def _draw(rng, probabilities):
    value = rng.random()
    for i, p in enumerate(probabilities):
        value -= p
        if value < 0:
            return i
    return len(probabilities) - 1


def sample_worlds(facts, belief, count, seed):
    """Stable IDs depend on class/region/slot, not on earlier hidden draws."""
    rng = random.Random(seed)
    result = []
    regions = sorted({k for row in belief.roots.values() for k in row["weights"]}
                     | {k for row in belief.instances.values() for k in row})
    width = max((len(row["pmf"]) for row in belief.roots.values()), default=1) + 1
    for _ in range(count):
        world = {}
        for ci, (name, row) in enumerate(sorted(belief.roots.items())):
            if not row["weights"]:
                continue
            keys, weights = zip(*sorted(row["weights"].items()))
            allocation = Counter(keys[_draw(rng, weights)] for _ in range(_draw(rng, row["pmf"])))
            for region, n in allocation.items():
                for slot in range(n):
                    key = -1000 - (ci * max(1, len(regions)) + regions.index(region)) * width - slot
                    world.setdefault(region, []).append((key, name))
        entities = {e.id: e for e in facts.entities}
        for key, row in sorted(belief.instances.items()):
            if row:
                regions_i, weights = zip(*sorted(row.items()))
                region = regions_i[_draw(rng, weights)]
                world.setdefault(region, []).append((int(key), entities[int(key)].name))
        result.append(world)
    return result


def navigation_stride(facts, config):
    count = sum(e.count for e in facts.navigation)
    return sum(e.distance_sum for e in facts.navigation) / count if count else config.navigation_step_distance


class EventPlanner:
    # ==================== Coop: Bounded Discovery Evaluation ====================
    def __init__(self, facts, belief, config=None, *, seed=1, communication_mode="exclusive_action",
                 send_rate=2500., template=None):
        from .protocol import WireTemplate
        self.template = template or WireTemplate(facts.rooms, facts.goals)
        self.facts, self.belief = facts, belief
        self.config = config or MethodConfig()
        self.seed, self.communication_mode, self.send_rate = seed, communication_mode, send_rate
        if not math.isfinite(send_rate) or send_rate <= 0:
            raise ValueError("invalid public communication rate")
        self.stride = navigation_stride(facts, self.config)
        self.cache = {}
        self.screening = []
        self.joint_screening = []
        self._joint_screening = {}
        self.stats = dict(astar_calls=0, astar_expanded=0, astar_bounded=0, cache_hits=0,
                          rollout_steps=0, structure_proxy_events=0, unresolved=0, incomplete_rollouts=0,
                          incomplete_shortage=0, incomplete_known_supply=0, synchronization_events=0)

    def known_ready(self, d, state):
        needs = Counter()
        for goal, n in zip(d.goals, state.remaining):
            if n and d.entities[goal.target_id].room_id is None:
                return False
            needs[goal.object_class] += n
        supply = Counter(e.name for k, e in d.objects.items() if k not in state.delivered)
        return all(supply[k] >= n for k, n in needs.items())

    def known_plan(self, sim):
        d, state = sim.domain, sim.state
        if not self.known_ready(d, state):
            return None
        key = (d.graph, state)
        if key in self.cache:
            self.stats["cache_hits"] += 1
            return self.cache[key]
        self.stats["astar_calls"] += 1
        try:
            plan = solve(d.resolved, self.config, root_state=state, partial=True,
                         max_expansions=self.config.discovery_astar_expansions)
        except PlanningError:
            # A budget/domain failure does not prove the environment unsolvable.
            self.stats["astar_bounded"] += 1
            result = None
        else:
            self.stats["astar_expanded"] += plan.expanded
            self.stats["astar_bounded"] += int(not plan.optimal)
            result = tuple(tuple(step.actions[i] for step in plan.schedule if step.actions[i]
                                 and step.actions[i].kind not in ("Wait", "Continue")) for i in (0, 1))
        self.cache[key] = result
        return result

    @staticmethod
    def allocation_cost(slots, rows, missing):
        """Small public-demand assignment; each known identity can serve one slot."""
        costs = {0: 0.}
        for row in rows:
            updated = dict(costs)
            for mask, value in costs.items():
                for i in range(slots):
                    if not mask & (1 << i):
                        new = mask | (1 << i)
                        updated[new] = min(updated.get(new, math.inf), value + row[i])
            costs = updated
        return min(value + (slots - bin(mask).count("1")) * missing for mask, value in costs.items())

    def delivery_cost(self, sim, room, point, target_id, *, obj=None):
        """Screening proxy only: public geometry, held state and calibrated actions."""
        d, state = sim.domain, sim.state
        target = d.entities[target_id]
        if target.room_id is None or target.position is None:
            weights = sim.belief.instances.get(str(target_id), {})
            locations = [(float(p), int(k[2:]), d.rooms[int(k[2:])].center)
                         for k, p in weights.items() if k.startswith("v:") and p > 0]
            if not locations:
                return self.config.discovery_time_horizon
        else:
            locations = [(1., target.room_id, target.position)]
        travel = sum(p * d.route(point, room, r, pos)[0] for p, r, pos in locations)
        interaction = PHYSICAL_TIME_SETTINGS.put_time
        holders = [a for a in state.actors if obj is not None and a.held == obj.id]
        if not holders:
            interaction += PHYSICAL_TIME_SETTINGS.grab_time
            if obj is not None:
                parent = d.entities.get(obj.inside)
                if parent and parent.container and parent.id not in state.opened:
                    interaction += PHYSICAL_TIME_SETTINGS.open_time
                travel += min(d.route(a.position, a.room, room, point)[0] for a in state.actors)
        if target.container and target.id not in state.opened:
            interaction += PHYSICAL_TIME_SETTINGS.open_time
        return travel + interaction

    def supply_summary(self, sim, *, exclude=()):
        d, state = sim.domain, sim.state
        target_belief = tuple((str(g.target_id), tuple(sorted(sim.belief.instances.get(str(g.target_id), {}).items())))
                              for g in d.goals if d.entities[g.target_id].room_id is None)
        key = ("supply", d.graph, state, target_belief, tuple(sorted(exclude)))
        if key in self.cache:
            return self.cache[key]
        summary = {}
        for name in sorted({g.object_class for g in d.goals}):
            targets = tuple(g.target_id for g, n in zip(d.goals, state.remaining)
                            if g.object_class == name for _ in range(n))
            if not targets:
                continue
            rows = []
            for obj in d.objects.values():
                if obj.name != name or obj.id in state.delivered or obj.id in exclude:
                    continue
                holder = next((a for a in state.actors if a.held == obj.id), None)
                room, point = (holder.room, holder.position) if holder else (obj.room_id, d.point(obj.id))
                rows.append(tuple(self.delivery_cost(sim, room, point, target, obj=obj) for target in targets))
            # Missing slots use the existing evaluation horizon, not a learned scarcity penalty.
            base = self.allocation_cost(len(targets), rows, self.config.discovery_time_horizon)
            summary[name] = (targets, rows, base, max(0, len(targets) - len(rows)))
        self.cache[key] = summary
        return summary

    def discoveries_improve_supply(self, identities, *, sim=None):
        """Compare new supply at the same current poses; never credit elapsed travel."""
        sim = sim if sim is not None else Simulation(self, {})
        if identities & {g.target_id for g, n in zip(sim.domain.goals, sim.state.remaining) if n}:
            return True
        # Different cooperation requirements cannot be safely screened out by
        # the relaxed single-object delivery proxy.
        if any(e.id in identities and other.id not in identities and e.name == other.name
               and e.coop != other.coop for e in sim.domain.objects.values()
               for other in sim.domain.objects.values()):
            return True
        before = self.supply_summary(sim, exclude=identities)
        after = self.supply_summary(sim)
        return any(after[name][3] < row[3] or after[name][2] < row[2] - EPS
                   for name, row in before.items())

    def inspection_communication_cost(self, sim, region, index):
        """Normal request/reply estimate; future evidence size remains unknown."""
        from .protocol import estimate_exchange_chars, _digest
        seed = sim.belief.to_seed()
        key = ("inspection_wire", sim.domain.graph, _digest(seed), region, index)
        if key not in self.cache:
            kind, raw = region.split(":")
            action = Action("CheckContainer" if kind == "c" else "Move", int(raw))
            chars = estimate_exchange_chars(sim.domain.graph, seed, base_graph=sim.domain.graph,
                base_seed=seed, actions=(action,), recipient=index, template=self.template)
            self.cache[key] = chars / self.send_rate
        return self.cache[key]

    def region_scores(self, sim, index, *, tail=False):
        d, state, b = sim.domain, sim.state, sim.belief
        actor = state.actors[index]
        if actor.held is not None or actor.request is not None:
            return []
        summary = self.supply_summary(sim)
        weights = {k for row in b.roots.values() for k in row["weights"]}
        weights |= {k for row in b.instances.values() for k in row}
        candidates = set()
        for key in weights:
            kind, raw = key.split(":")
            ident = int(raw)
            if kind == "h":
                if not d.rooms[ident].explored:
                    candidates.add("v:" + raw)
                elif tail:
                    candidates.add(key)
            elif kind in ("v", "c"):
                candidates.add(key)
        result = []
        for region in sorted(candidates - sim.checked):
            kind, raw = region.split(":")
            ident = int(raw)
            if kind == "c":
                if ident not in d.entities or ident in state.opened:
                    continue
                room, point = d.entities[ident].room_id, d.point(ident)
            else:
                if kind == "v" and d.rooms[ident].explored:
                    continue
                room, point = ident, d.rooms[ident].center
            if room is None or point is None:
                continue
            cost, _ = d.route(actor.position, actor.room, room, point)
            cost += (PHYSICAL_TIME_SETTINGS.open_time if kind == "c" else
                     self.config.discovery_hidden_checks * PHYSICAL_TIME_SETTINGS.open_time if kind == "h" else 1.)
            regions = [region] + (["h:" + raw] if kind == "v" else [])
            value, coverage, no_hidden = 0., set(), 1.
            for name, (targets, rows, base, shortage) in summary.items():
                pmf = b.region_count(name, regions, d.entities, len(targets))
                if shortage and sum(pmf[1:]) > 0:
                    coverage.add(name)
                delivery = tuple(self.delivery_cost(sim, room, point, target) for target in targets)
                if kind == "v":
                    no_hidden *= b.region_count(name, ["h:" + raw], d.entities, 1)[0]
                for n, probability in enumerate(pmf[1:], 1):
                    after = self.allocation_cost(len(targets), rows + [delivery] * n,
                                                 self.config.discovery_time_horizon)
                    value += probability * max(0., base - after)
            for key, row in b.instances.items():
                if any(n and g.target_id == int(key) for g, n in zip(d.goals, state.remaining)):
                    probability = sum(row.get(loc, 0.) for loc in regions)
                    value += probability * self.config.discovery_time_horizon
                    if probability > 0:
                        coverage.add("target:" + key)
            # Hidden contents require later checks, shared across classes/instances.
            if kind == "v":
                cost += (1 - no_hidden) * self.config.discovery_hidden_checks * PHYSICAL_TIME_SETTINGS.open_time
            cost += self.inspection_communication_cost(sim, region, index)
            if value <= EPS and not coverage and self.known_ready(d, state):
                continue
            # Only a proposal score. Rollout charges physical/transport costs once.
            result.append((-(value + 1e-9) / max(.2, cost), cost, region, frozenset(coverage)))
        return sorted(result)

    def joint_screen(self, sim, directives, lookup):
        """Task-capped proposal value for the selected regions together."""
        d, state, b = sim.domain, sim.state, sim.belief
        selected = []
        for index, region in enumerate(directives):
            if region is None:
                continue
            kind, raw = region.split(":")
            ident = int(raw)
            if kind == "c":
                room, point = d.entities[ident].room_id, d.point(ident)
            else:
                room, point = ident, d.rooms[ident].center
            effective = frozenset((region, "h:" + raw)) if kind == "v" else frozenset((region,))
            selected.append((index, region, room, point, effective))
        if not selected:
            return dict(value=0., value_per_cost=0., inspection_cost=0., shortage_coverage=[])

        value, coverage = 0., set()
        summary = self.supply_summary(sim)
        groups = [row[4] for row in selected]
        for name, (targets, rows, base, shortage) in summary.items():
            allocations = b.region_allocation(name, groups, d.entities, len(targets))
            deliveries = [tuple(self.delivery_cost(sim, room, point, target) for target in targets)
                          for _, _, room, point, _ in selected]
            if shortage and any(sum(counts) and probability > 0 for counts, probability in allocations):
                coverage.add(name)
            for counts, probability in allocations:
                discovered = rows + [deliveries[i] for i, count in enumerate(counts) for _ in range(count)]
                after = self.allocation_cost(len(targets), discovered, self.config.discovery_time_horizon)
                value += probability * max(0., base - after)
        for key, row in b.instances.items():
            if any(n and goal.target_id == int(key) for goal, n in zip(d.goals, state.remaining)):
                probability = min(1., sum(row.get(region, 0.) for group in groups for region in group))
                value += probability * self.config.discovery_time_horizon
                if probability > 0:
                    coverage.add("target:" + key)
        # This only screens concurrent proposals; the rollout below charges the
        # actual synchronous trajectory and every communication boundary.
        cost = max(lookup[index][region][1] for index, region, *_ in selected)
        return dict(value=value, value_per_cost=(value + 1e-9) / max(.2, cost),
                    inspection_cost=cost, shortage_coverage=sorted(coverage))

    def joint(self, sim, directives):
        d, state = sim.domain, sim.state
        if state.holder is not None:
            candidates = d.joint_actions(state)
            return candidates[0] if candidates else (None, None)
        options = []
        for i, actor in enumerate(state.actors):
            if actor.action:
                options.append((None,))
            elif actor.request is not None:
                options.append((Action("Wait"),))
            elif directives[i] is not None:
                action = d.exploration_action(state, i, directives[i])
                options.append((action,) if action else (None,))
            else:
                actions = d.individual_actions(state, i)
                options.append(tuple(sorted(actions, key=lambda a: (a is None, a.kind == "Move" if a else True,
                                           d.duration(state, i, a) if a and a.kind != "Wait" else 0., action_key(a)))))
        combinations = list(product(*options))
        valid = [pair for pair in combinations if d.executable(state, pair)]
        if valid:
            return valid[0]
        for pair in combinations:
            for i in (0, 1):
                if not state.actors[1 - i].action and state.actors[1 - i].request is None:
                    single = (pair[0], None) if i == 0 else (None, pair[1])
                    if d.executable(state, single):
                        return single
        return (None, None)

    def prefix(self, directives, world=None, *, queues=None):
        sim = Simulation(self, world)
        chosen = [list(), list()]
        # Already-issued work is retained on both sides of the event boundary.
        for i, action in enumerate(sim.domain.initial_actions):
            if action:
                chosen[i].append(action)
        frozen = [deque(q) for q in queues] if queues else None
        for _ in range(self.config.discovery_rollout_steps):
            if sim.state.complete:
                break
            if frozen is not None:
                joint = sim.queue_joint(frozen)
                if joint is None:
                    break
            else:
                joint = self.joint(sim, directives)
            if not any(a.action for a in sim.state.actors) and not any(a and a.kind != "Wait" for a in joint):
                break
            additions = [(i, a) for i, a in enumerate(joint) if a and a.kind != "Wait"]
            if any(len(chosen[i]) >= self.config.action_horizon for i, _ in additions):
                break
            for i, action in additions:
                if not chosen[i] or chosen[i][-1] != action or sim.domain.action_done(sim.state, i, action):
                    chosen[i].append(action)
            changed = sim.advance(joint)
            if changed or sim.observation_boundary or sim.elapsed >= self.config.discovery_time_horizon:
                break
        return tuple(tuple(q) for q in chosen), sim

    def candidates(self):
        sim = Simulation(self, {})
        candidates, seen = [], set()
        known = self.known_plan(sim)
        if known:
            prefix, _ = self.prefix((None, None), queues=known)
            if any(prefix):
                candidates.append((prefix, (None, None), known))
                seen.add(prefix)
        if len(candidates) >= self.config.discovery_candidates:
            return candidates
        scores = [self.region_scores(sim, i) for i in (0, 1)]
        self.screening = [[dict(region=r[2], value_per_cost=-r[0], inspection_cost=r[1],
                                shortage_coverage=sorted(r[3])) for r in rows] for rows in scores]
        options = []
        catalog = partition(self.facts)
        for rows in scores:
            selected, groups = [], set()
            for row in rows:
                group = region_group(row[2], catalog)
                if group not in groups:
                    selected.append(row)
                    groups.add(group)
                if len(selected) == 3:
                    break
            selected += [row for row in rows if row not in selected][:3 - len(selected)]
            covered = set().union(*(row[3] for row in selected))
            for row in rows:
                if row[3] - covered:
                    if row not in selected:
                        selected.append(row)
                    covered.update(row[3])
            options.append([None] + [r[2] for r in selected])
        lookup = [{r[2]: r for r in rows} for rows in scores]
        assignments = [pair for pair in product(*options)
                       if pair[0] is None or pair[0] != pair[1]]
        self._joint_screening = {pair: self.joint_screen(sim, pair, lookup) for pair in assignments}
        self.joint_screening = [dict(directives=list(pair), **row)
                                for pair, row in sorted(self._joint_screening.items(), key=lambda item: str(item[0]))]
        covered, covered_groups = set(), set()
        while assignments and len(candidates) < self.config.discovery_candidates:
            def rank(pair):
                rows = [lookup[i][r] for i, r in enumerate(pair) if r is not None]
                joint = self._joint_screening[pair]
                coverage = set(joint["shortage_coverage"])
                groups = {region_group(r[2], catalog) for r in rows}
                return (pair != (None, None), -joint["value_per_cost"],
                        -len(coverage - covered), -len(groups - covered_groups),
                        joint["inspection_cost"], str(pair))
            directives = min(assignments, key=rank)
            assignments.remove(directives)
            prefix, _ = self.prefix(directives)
            if not any(prefix) or prefix in seen:
                continue
            seen.add(prefix)
            candidates.append((prefix, directives, None))
            for i, region in enumerate(directives):
                if region is not None:
                    covered.update(lookup[i][region][3])
                    covered_groups.add(region_group(region, catalog))
        return candidates

    def rollout(self, sim):
        frozen = None
        assigned, used = (False, False), [0, 0]
        abstracts = {}  # A declared hidden-room search cost, never executable IDs.
        for _ in range(self.config.discovery_rollout_steps):
            if sim.state.complete or sim.elapsed >= self.config.discovery_time_horizon:
                break
            if frozen is None and not abstracts:
                plan = self.known_plan(sim)
                frozen = [deque(q) for q in plan] if plan else None
                assigned = tuple(bool(q) for q in plan) if plan else (False, False)
            joint = sim.queue_joint(frozen) if frozen else None
            if joint is None:
                frozen = None
                directives = [None, None]
                d, s = sim.domain, sim.state
                if s.holder is None and not d.requests(s):
                    assigned = set(region for region, _ in abstracts.values())
                    for i, actor in enumerate(s.actors):
                        if actor.action or actor.held is not None or i in abstracts:
                            continue
                        # Fixed tail policy first uses known work, then searches.
                        available = d.individual_actions(s, i)
                        if any(a for a in available):
                            continue
                        rows = [r for r in self.region_scores(sim, i, tail=True) if r[2] not in assigned]
                        if rows:
                            region = rows[0][2]
                            assigned.add(region)
                            if region.startswith("h:") and actor.room == int(region[2:]):
                                abstracts[i] = (region, max(1, math.ceil(self.config.discovery_hidden_checks)))
                            else:
                                directives[i] = "v:" + region[2:] if region.startswith("h:") else region
                joint = self.joint(sim, directives)
                if abstracts:
                    joint = tuple(None if i in abstracts else a for i, a in enumerate(joint))
                if not any(a.action for a in s.actors) and not any(a and a.kind != "Wait" for a in joint) and not abstracts:
                    self.stats["unresolved"] += 1
                    break
            extra = PHYSICAL_TIME_SETTINGS.open_time if abstracts else 0.
            for i, action in enumerate(joint):
                used[i] += int(action is not None and action.kind not in ("Wait", "Continue"))
            event = sim.advance(joint, extra_physical=extra)
            for i, (region, left) in list(abstracts.items()):
                if left <= 1:
                    sim.reveal((region,))
                    self.stats["structure_proxy_events"] += 1
                    del abstracts[i]
                    event = True
                else:
                    abstracts[i] = (region, left - 1)
            event |= any(n >= self.config.action_horizon for n in used)
            if frozen is not None:
                event |= any(assigned[i] and not frozen[i] and actor.action is None and actor.request is None
                             for i, actor in enumerate(sim.state.actors))
            if event:
                frozen = None
                used = [0, 0]
                sim.synchronization_charge()
        # Every incomplete branch pays the same remaining finite time horizon.
        # This includes impossible sampled counts and unexecutable residual mass.
        self.stats["incomplete_rollouts"] += int(not sim.state.complete)
        if not sim.state.complete:
            reason = "incomplete_known_supply" if self.known_ready(sim.domain, sim.state) else "incomplete_shortage"
            self.stats[reason] += 1
        if not sim.state.complete and sim.elapsed < self.config.discovery_time_horizon:
            sim.charge(self.config.discovery_time_horizon - sim.elapsed)
        return sim.objective

    def plan(self):
        started = time.monotonic()
        if not any(self.facts.remaining(g) for g in self.facts.goals):
            return DiscoveryResult(((), ()), 0., 0., 0, 0., dict(candidates=[]))
        if any(a.room_id is None or a.position is None for a in self.facts.agents):
            raise PlanningError("discovery requires communicated agent positions")
        candidates = self.candidates()
        if not candidates:
            raise PlanningError("no grounded discovery or transport prefix; residual is not an action")
        worlds = sample_worlds(self.facts, self.belief, self.config.discovery_samples, self.seed)
        rows = []
        for prefix, directives, queues in candidates:
            values, times = [], []
            for world in worlds:
                same, sim = self.prefix(directives, world, queues=queues)
                if same != prefix:
                    raise PlanningError("sample truth changed the common prefix")
                if not sim.state.complete:
                    sim.synchronization_charge()
                values.append(self.rollout(sim))
                times.append(sim.elapsed)
            mean = sum(values) / len(values)
            stderr = math.sqrt(sum((v - mean) ** 2 for v in values) / max(1, len(values) - 1) / len(values))
            rows.append(dict(score=mean, sample_stderr=stderr, scores=values,
                             prefix_lengths=[len(q) for q in prefix], directives=list(directives),
                             mean_time=sum(times) / len(times)))
        selected = min(range(len(rows)), key=lambda i: (rows[i]["score"], i))
        baseline = next((i for i, candidate in enumerate(candidates) if candidate[2] is not None), None)
        reference = baseline if baseline is not None else selected
        for row in rows:
            differences = [a - b for a, b in zip(row["scores"], rows[reference]["scores"])]
            mean = sum(differences) / len(differences)
            stderr = math.sqrt(sum((x - mean) ** 2 for x in differences)
                               / max(1, len(differences) - 1) / len(differences))
            row.update(paired_delta=mean, paired_stderr=stderr)
        if baseline is not None:
            # A conservative two-standard-error rule, not a calibrated guarantee.
            if rows[selected]["paired_delta"] + 2 * rows[selected]["paired_stderr"] >= -EPS:
                selected = baseline
        else:
            # Truncated/indistinguishable rollouts must not fall back to ID order.
            # Reuse analytical discovery value; no new reward or extra samples.
            lookup = [{row["region"]: row for row in side} for side in self.screening]
            tied = [i for i, row in enumerate(rows)
                    if row["paired_delta"] - row["paired_stderr"] <= EPS]
            def rank(i):
                joint = self._joint_screening.get(tuple(candidates[i][1]))
                if joint is not None:
                    return (-joint["value_per_cost"], joint["inspection_cost"], i)
                selected_rows = [lookup[a][r] for a, r in enumerate(candidates[i][1])
                                 if a < len(lookup) and r in lookup[a]]
                return (-sum(r["value_per_cost"] for r in selected_rows),
                        sum(r["inspection_cost"] for r in selected_rows), i)
            selected = min(tied, key=rank)
        diagnostics = dict(candidates=rows, selected=selected,
                           samples=len(worlds), time_horizon=self.config.discovery_time_horizon,
                           navigation_step_distance=self.stride, **self.stats,
                           tail="public_policy_rollout_hidden_structure_proxy",
                           screening="task_capped_joint_candidate", screening_candidates=self.screening,
                           joint_screening=self.joint_screening, baseline=baseline,
                           paired_reference=reference,
                           unknown_cooperation="ordinary_until_public_feedback",
                           communication="delta_exchange_estimate", belief_revision=self.belief.revision)
        return DiscoveryResult(candidates[selected][0], rows[selected]["score"], rows[selected]["mean_time"],
                               self.stats["astar_expanded"], time.monotonic() - started, diagnostics)
    # ==================== End Coop: Bounded Discovery Evaluation ====================


class Simulation:
    """A private evaluation state; only reveal() is allowed to read world samples."""
    def __init__(self, planner, world):
        self.planner, self.world = planner, world
        self.belief = DiscoveryBelief.from_seed(planner.belief.to_seed())
        self.domain = PlanningDomain(factual_planning_graph(planner.facts), planner.config, partial=True)
        self.state = self.domain.initial
        self.checked = set(self.belief.processed)
        self.elapsed = self.objective = 0.
        self.steps = 0
        self.observation_boundary = False
        self.shared_graph = self.domain.graph
        self.shared_seed = self.belief.to_seed()

    def charge(self, seconds):
        seconds = min(seconds, max(0., self.planner.config.discovery_time_horizon - self.elapsed))
        self.objective += seconds * (1 + self.planner.config.completion_latency_weight *
                                      sum(self.state.remaining) / self.domain.total_goals)
        self.elapsed += seconds

    def synchronization_charge(self):
        if self.state.complete:
            return
        self.planner.stats["synchronization_events"] += 1
        from .protocol import estimate_exchange_chars
        step = self.planner.facts.step + self.steps
        graph = replace(self.domain.graph, step=step, agents=tuple(
            replace(metadata, room_id=actor.room, position=actor.position,
                    held=() if actor.held is None else (actor.held,), close=actor.close,
                    stamp=Stamp(step, metadata.index))
            for metadata, actor in zip(self.domain.graph.agents, self.state.actors)))
        plan = self.planner.known_plan(self)
        recipient = 1 - self.planner.facts.source
        if plan:
            actions = plan[recipient][:self.planner.config.action_horizon]
        else:
            action = self.planner.joint(self, (None, None))[recipient]
            actions = (action,) if action and action.kind not in ("Wait", "Continue") else ()
        seed = self.belief.to_seed()
        chars = estimate_exchange_chars(graph, seed, base_graph=self.shared_graph,
                                        base_seed=self.shared_seed, actions=actions, recipient=recipient, template=self.planner.template)
        self.charge(chars / self.planner.send_rate)
        self.shared_graph, self.shared_seed = graph, seed

    def queue_joint(self, queues):
        if queues is None:
            return None
        choices = []
        for i, actor in enumerate(self.state.actors):
            if actor.action:
                if queues[i] and queues[i][0] == actor.action:
                    queues[i].popleft()
                choices.append(None)
            elif actor.request is not None:
                while queues[i] and queues[i][0].kind == "Grab" and queues[i][0].target_id == actor.request:
                    queues[i].popleft()
                choices.append(Action("Wait"))
            else:
                while queues[i] and self.domain.action_done(self.state, i, queues[i][0]):
                    queues[i].popleft()
                choices.append(queues[i][0] if queues[i] else None)
        pair = tuple(choices)
        if not any(a.action for a in self.state.actors) and not any(a and a.kind != "Wait" for a in pair):
            return None
        if not self.domain.executable(self.state, pair):
            return None
        for i, action in enumerate(pair):
            if action and action.kind != "Wait":
                queues[i].popleft()
        return pair

    def advance(self, joint, *, extra_physical=0.):
        previous = self.state
        self.observation_boundary = False
        located = {e.id for e in self.domain.entities.values() if e.room_id is not None}
        following, seconds = self.domain.synchronous_step(previous, joint, step_distance=self.planner.stride)
        duration = max(seconds, extra_physical)
        remaining_horizon = self.planner.config.discovery_time_horizon - self.elapsed
        self.charge(duration)
        # A boundary beyond the common horizon cannot earn partial-step success.
        if duration > remaining_horizon + EPS:
            return False
        self.state = following
        self.steps += 1
        self.planner.stats["rollout_steps"] += 1
        regions = {"v:" + str(a.room) for a in following.actors
                   if a.room not in {r.id for r in self.domain.graph.rooms if r.explored}}
        regions |= {"c:" + str(k) for k in set(following.opened) - set(previous.opened)}
        unknown_probe = any(a and a.kind == "Grab" and any(x.entity_id == a.target_id and x.field == "coop"
                           for x in self.domain.resolved.assumptions) for a in joint)
        # Issued prefixes stop before choosing anything from an unseen outcome,
        # even when a tail policy can defer communicating an incidental miss.
        self.observation_boundary = bool(regions - {"c:" + str(g.target_id) for g in self.domain.goals}
                                         or unknown_probe)
        if regions or unknown_probe:
            # Preserve still-unknown properties across graph reconstruction; a
            # sampled room inspection must not silently resolve unrelated coop.
            self.reveal(regions, probed={a.target_id for a in joint if a and a.kind == "Grab"})
        discovered = {e.id for e in self.domain.entities.values() if e.room_id is not None} - located
        new_supply = bool(discovered) and self.planner.discoveries_improve_supply(discovered, sim=self)
        # Real execution also replans at every non-target inspection boundary.
        return bool(unknown_probe or new_supply or inspection_requires_sync(
            regions, {g.target_id for g in self.domain.goals}))

    def reveal(self, regions, *, probed=()):
        graph = self.domain.graph
        entities = {e.id: e for e in graph.entities}
        for assumption in self.domain.resolved.assumptions:
            if assumption.field == "coop" and assumption.entity_id not in probed:
                entities[assumption.entity_id] = replace(entities[assumption.entity_id], coop=None)
        rooms = {r.id: r for r in graph.rooms}
        counts_by_region = {}
        for region in sorted(set(regions) - self.checked):
            kind, raw = region.split(":")
            key = int(raw)
            if kind == "c" and key not in entities:
                continue
            room = entities[key].room_id if kind == "c" else key
            if kind == "v":
                rooms[room] = replace(rooms[room], explored=True)
            if kind == "c":
                entities[key] = replace(entities[key], open_state="open", explored=True)
            point = entities[key].position if kind == "c" else rooms[room].center
            counts = Counter()
            for identity, name in (self.world or {}).get(region, ()):
                old = entities.get(identity, Entity(identity, name))
                entities[identity] = replace(old, room_id=room, position=point,
                    inside=key if kind == "c" else room, on=(), holders=(), stamp=Stamp(graph.step, graph.source))
                if identity < 0:
                    counts[name] += 1
                self.belief.instances.pop(str(identity), None)
            counts_by_region[region] = counts
            self.checked.add(region)
        if self.world is not None:
            for region, counts in counts_by_region.items():
                self.belief._inspect(region, counts)
        # The planning view is synthetic and intentionally never converted to facts.
        values = dict(graph.__dict__)
        values.update(rooms=tuple(rooms[k] for k in sorted(rooms)), entities=tuple(entities[k] for k in sorted(entities)))
        self.domain = PlanningDomain(factual_planning_graph(PlanningGraph(**values)), self.planner.config, partial=True)
