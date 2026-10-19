"""Sparse finite-count discovery model, conditioned only on delivered evidence."""

from copy import deepcopy
from collections import Counter
import hashlib
import json
import math

from .config import MethodConfig



def normalized(values):
    if (not values or any(type(x) not in (int, float) or not math.isfinite(x) or x < 0 for x in values)
            or sum(values) <= 0):
        raise ValueError("weights must be finite, nonnegative and nonzero")
    total = sum(values)
    return [float(x) / total for x in values]


def condition_count(pmf, fraction, observed):
    """Condition task-useful representatives, saturating observations at the cap."""
    if not 0 <= fraction <= 1 or type(observed) is not int or observed < 0:
        raise ValueError("invalid inspection likelihood")
    if observed and len(pmf) == 1:
        raise ValueError("observation outside model support")
    observed = min(observed, len(pmf) - 1)
    likelihood = [p * math.comb(n, observed) * fraction ** observed * (1 - fraction) ** (n - observed)
                  if n >= observed else 0. for n, p in enumerate(pmf)]
    if sum(likelihood) <= 0:
        raise ValueError("observation outside model support")
    posterior = normalized(likelihood)
    return posterior[observed:]


def partition(facts, *, include_checked=False):
    """No imagined container IDs: hidden/residual nodes have no physical action."""
    rooms = {r.id: r for r in facts.rooms}
    targets = {g.target_id for g in facts.goals}
    result = {}
    for r in facts.rooms:
        if not r.explored:
            result["v:" + str(r.id)] = dict(kind="visible", room_id=r.id, name=r.name)
            result["h:" + str(r.id)] = dict(kind="hidden", room_id=r.id, name=r.name)
        else:
            result["u:" + str(r.id)] = dict(kind="unmapped", room_id=r.id, name=r.name)
    for e in facts.entities:
        if (e.searchable and e.id not in targets and e.room_id in rooms and rooms[e.room_id].explored
                and (include_checked or not e.explored)):
            result["c:" + str(e.id)] = dict(kind="contents", room_id=e.room_id,
                                            name=e.name, room=rooms[e.room_id].name)
    return result


def region_group(region, catalog):
    """Public structural siblings share one prior mass, independent of multiplicity."""
    item = catalog.get(region)
    if item and item["kind"] == "contents":
        return ("contents", item["room_id"], item["name"])
    return (region,)


def _group_factors(facts, class_name, catalog, processed):
    """Beta(1,1) reliability posterior from this episode's complete checks."""
    entities = {entity.id: entity for entity in facts.entities}
    evidence = {}
    for inspection in facts.inspections:
        item = catalog.get(inspection.region)
        if (not inspection.complete or inspection.region not in processed or not item
                or item["kind"] != "contents"):
            continue
        group = region_group(inspection.region, catalog)
        hit = any(object_id in entities and entities[object_id].name == class_name
                  for object_id in inspection.object_ids)
        row = evidence.setdefault(group, [0, 0])
        row[0 if hit else 1] += 1
    return {group: 2. * (1 + hits) / (2 + hits + misses)
            for group, (hits, misses) in evidence.items()}


def _condition_groups(weights, facts, class_name, catalog, before, after):
    """Apply each newly available semantic-group observation exactly once."""
    if not weights or set(before) == set(after):
        return weights
    old = _group_factors(facts, class_name, catalog, set(before))
    new = _group_factors(facts, class_name, catalog, set(after))
    adjusted = {region: value * new.get(region_group(region, catalog), 1.)
                / old.get(region_group(region, catalog), 1.)
                for region, value in weights.items()}
    values = normalized(list(adjusted.values()))
    return dict(zip(adjusted, values))


def task_key(facts):
    data = [(g.key, g.count) for g in facts.goals], [(r.id, r.name, r.center) for r in facts.rooms]
    return hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()[:20]


class DiscoveryBelief:
    # ==================== Coop: Synchronized Probability State ====================
    def __init__(self):
        self.task = ""
        self.revision = 0
        self.roots = {}
        self.instances = {}
        self.known = []
        self.processed = []
        self.repairs = 0
        self.last_diagnostics = {}

    def to_seed(self):
        return dict(task=self.task, revision=self.revision,
                    roots=deepcopy(self.roots), instances=deepcopy(self.instances),
                    known=list(self.known), processed=list(self.processed), repairs=self.repairs)

    @classmethod
    def from_seed(cls, seed, facts=None):
        keys = {"task", "revision", "roots", "instances", "known", "processed", "repairs"}
        if not isinstance(seed, dict) or set(seed) != keys:
            raise ValueError("invalid belief fields")
        if (not isinstance(seed["task"], str) or type(seed["revision"]) is not int or seed["revision"] < 0
                or type(seed["repairs"]) is not int or seed["repairs"] < 0):
            raise ValueError("invalid belief revision")
        for key, typ in (("known", int), ("processed", str)):
            rows = seed[key]
            if not isinstance(rows, list) or any(type(x) is not typ for x in rows) or rows != sorted(set(rows)):
                raise ValueError("invalid evidence baseline")
        if any(key <= 0 for key in seed["known"]):
            raise ValueError("hypothetical identities cannot enter a belief seed")
        if not isinstance(seed["roots"], dict) or not isinstance(seed["instances"], dict):
            raise ValueError("invalid belief rows")
        def check_weights(weights):
            if not isinstance(weights, dict) or any(not isinstance(k, str) for k in weights):
                raise ValueError("invalid region weights")
            if weights:
                values = list(weights.values())
                normalized(values)
                if abs(sum(values) - 1.) > 1e-6:
                    raise ValueError("unnormalized belief weights")
            for key in weights:
                kind, sep, number = key.partition(":")
                if not sep or kind not in ("v", "c", "h", "u") or not number.isdigit():
                    raise ValueError("invalid region identity")
        for name, row in seed["roots"].items():
            if not isinstance(name, str) or not isinstance(row, dict) or set(row) != {"pmf", "weights"}:
                raise ValueError("invalid count row")
            if not isinstance(row["pmf"], list):
                raise ValueError("invalid count probabilities")
            normalized(row["pmf"])
            if abs(sum(row["pmf"]) - 1.) > 1e-6:
                raise ValueError("unnormalized count probabilities")
            check_weights(row["weights"])
        for key, weights in seed["instances"].items():
            if not isinstance(key, str) or not key.isdigit():
                raise ValueError("invalid instance identity")
            check_weights(weights)
        if facts is not None and seed["task"] and seed["task"] != task_key(facts):
            raise ValueError("belief seed belongs to another task")
        result = cls()
        for k in ("task", "revision", "roots", "instances", "known", "processed", "repairs"):
            setattr(result, k, deepcopy(seed[k]))
        return result

    def probability(self, name, region):
        row = self.roots.get(name)
        if not row:
            return 0.
        p = row["weights"].get(region, 0.)
        return max(0., min(1., 1 - sum(x * (1 - p) ** n for n, x in enumerate(row["pmf"]))))

    def expected_useful(self, name, region, demand):
        row = self.roots.get(name)
        if not row:
            return 0.
        p = row["weights"].get(region, 0.)
        return sum(pn * sum(min(k, demand) * math.comb(n, k) * p ** k * (1 - p) ** (n - k)
                            for k in range(n + 1)) for n, pn in enumerate(row["pmf"]))

    def region_count(self, name, regions, entities, cap):
        """Exact marginal count, capped at demand, including unlocated identities."""
        row = self.roots.get(name)
        result = [1.] + [0.] * cap
        if row:
            fraction = min(1., sum(row["weights"].get(k, 0.) for k in regions))
            result = [0.] * (cap + 1)
            for n, pn in enumerate(row["pmf"]):
                for k in range(n + 1):
                    result[min(k, cap)] += pn * math.comb(n, k) * fraction ** k * (1 - fraction) ** (n - k)
        for key, weights in self.instances.items():
            entity = entities[int(key)]
            if entity.name != name or entity.room_id is not None:
                continue
            p = min(1., sum(weights.get(k, 0.) for k in regions))
            updated = [0.] * (cap + 1)
            for k, value in enumerate(result):
                updated[k] += value * (1 - p)
                updated[min(cap, k + 1)] += value * p
            result = updated
        return result

    def region_allocation(self, name, region_groups, entities, cap):
        """Joint count distribution over disjoint candidate regions."""
        groups = tuple(frozenset(group) for group in region_groups)
        if len(set().union(*groups)) != sum(len(group) for group in groups):
            raise ValueError("candidate regions overlap")
        zero = (0,) * len(groups)

        def add_objects(distribution, probabilities, count=1):
            total = sum(probabilities)
            if total > 1.:
                probabilities = [probability / total for probability in probabilities]
                total = 1.
            outside = 1. - total
            for _ in range(count):
                updated = {}
                for counts, mass in distribution.items():
                    updated[counts] = updated.get(counts, 0.) + mass * outside
                    for index, probability in enumerate(probabilities):
                        placed = list(counts)
                        placed[index] = min(cap, placed[index] + 1)
                        placed = tuple(placed)
                        updated[placed] = updated.get(placed, 0.) + mass * probability
                distribution = updated
            return distribution

        row = self.roots.get(name)
        if row:
            probabilities = [sum(row["weights"].get(region, 0.) for region in group) for group in groups]
            result = {}
            for count, mass in enumerate(row["pmf"]):
                for counts, probability in add_objects({zero: 1.}, probabilities, count).items():
                    result[counts] = result.get(counts, 0.) + mass * probability
        else:
            result = {zero: 1.}
        for key, weights in self.instances.items():
            entity = entities[int(key)]
            if entity.name == name and entity.room_id is None:
                probabilities = [sum(weights.get(region, 0.) for region in group) for group in groups]
                result = add_objects(result, probabilities)
        return sorted((counts, probability) for counts, probability in result.items() if probability > 0)

    def _inspect(self, region, counts):
        roots, instances = deepcopy(self.roots), deepcopy(self.instances)
        for name, row in roots.items():
            weights = row["weights"]
            if region not in weights:
                if counts.get(name, 0):
                    raise ValueError("new identity has no anonymous source")
                continue
            row["pmf"] = condition_count(row["pmf"], weights[region], counts.get(name, 0))
            rest = {k: v for k, v in weights.items() if k != region}
            total = sum(rest.values())
            row["weights"] = {k: v / total for k, v in rest.items()} if total > 0 else {}
        for weights in instances.values():
            if region in weights:
                del weights[region]
                if sum(weights.values()) > 0:
                    total = sum(weights.values())
                    weights.update({k: v / total for k, v in weights.items()})
        self.roots, self.instances = roots, instances

    def synchronize(self, facts, client, *, config=None, agent_index=None):
        """Transactional prior/refinement/observation; the caller only sees a commit."""
        from .prediction import predict_discovery, aggregate_usage
        config = config or MethodConfig()
        facts.validate()
        work = self.from_seed(self.to_seed(), facts)
        work.task = task_key(facts)
        entities = {e.id: e for e in facts.entities}
        demand = Counter()
        for goal in facts.goals:
            demand[goal.object_class] += facts.remaining(goal)
        active = {name for name, n in demand.items() if n}
        completed = {key for goal in facts.goals for key in facts.satisfied(goal)}
        known_supply = Counter(e.name for e in facts.entities
                               if e.name in active and e.stamp.step >= 0 and e.id not in completed)
        shortage = {name: max(0, demand[name] - known_supply[name]) for name in active}
        work.roots = {name: row for name, row in work.roots.items() if shortage.get(name, 0) > 0}
        target_ids = {g.target_id for g in facts.goals if facts.remaining(g)}
        work.instances = {key: row for key, row in work.instances.items()
                          if int(key) in entities and entities[int(key)].room_id is None
                          and (int(key) in target_ids or entities[int(key)].name in active)}
        current_ids = {e.id for e in facts.entities if e.name in demand and e.stamp.step >= 0}
        new_ids = current_ids - set(work.known)
        attempts, repairs, refinements = [], [], 0
        catalog = partition(facts)
        evidence_catalog = partition(facts, include_checked=True)
        # Located identities are facts, never resampled after a negative region update.
        work.instances = {key: row for key, row in work.instances.items()
                          if int(key) in entities and entities[int(key)].room_id is None}
        consumed = set()
        origins = {e.object_id: e for e in facts.origins}
        pending = [ev for ev in sorted(facts.inspections, key=lambda e: (e.stamp, e.region))
                   if ev.complete and ev.region not in work.processed]

        def represented(ev):
            return (any(ev.region in row["weights"] for row in work.roots.values())
                    or any(ev.region in row for row in work.instances.values()))

        def inspect(events):
            nonlocal consumed
            for ev in events:
                before = set(work.processed)
                discovered = {key for key in set(ev.object_ids) & new_ids - consumed
                              if key in origins and origins[key].region == ev.region}
                counts = Counter(entities[k].name for k in discovered)
                try:
                    work._inspect(ev.region, counts)
                except ValueError:
                    # Real evidence wins. Reinitialize only affected anonymous models
                    # from current facts; never label a zero-likelihood event as empty.
                    affected = {name for name in work.roots
                                if counts.get(name) or ev.region in work.roots[name]["weights"]}
                    for name in affected:
                        work.roots.pop(name, None)
                    repairs.append("support:" + ev.region)
                else:
                    after = before | {ev.region}
                    for name, row in work.roots.items():
                        row["weights"] = _condition_groups(
                            row["weights"], facts, name, evidence_catalog, before, after)
                work.processed.append(ev.region)
                consumed |= discovered

        # First condition observations represented by the old model. If public
        # structure then changes, semantic location weights are refreshed across
        # every remaining legal region while the count posterior stays fixed.
        first = [ev for ev in pending if represented(ev)]
        inspect(first)
        locations = sorted(catalog)
        refresh = []
        if locations:
            for name, row in sorted(work.roots.items()):
                if set(row["weights"]) != set(locations):
                    refresh.append(dict(key="place:" + name, kind="weights",
                                        class_name=name, candidates=locations))
        for key, weights in sorted(work.instances.items()):
            candidates = sorted(k for k in catalog if int(key) not in target_ids or k.startswith("v:"))
            if candidates and set(weights) != set(candidates):
                refresh.append(dict(key="instance:" + key, kind="weights",
                                    class_name=entities[int(key)].name,
                                    role="goal_target" if int(key) in target_ids else "known_object",
                                    candidates=candidates))
        if refresh:
            answers, usage = predict_discovery(facts, refresh, catalog, client, config=config,
                                               agent_index=agent_index)
            attempts += usage["llm_attempts"]
            for query in refresh:
                weights = dict(zip(query["candidates"], answers[query["key"]]))
                if query["key"].startswith("place:"):
                    name = query["class_name"]
                    work.roots[name]["weights"] = _condition_groups(
                        weights, facts, name, evidence_catalog, (), work.processed)
                else:
                    work.instances[query["key"].split(":")[1]] = weights
                refinements += 1
        inspect([ev for ev in pending if ev not in first])
        if self.task:
            for key in new_ids - consumed:
                name = entities[key].name
                if name in work.roots:
                    work.roots.pop(name)
                    repairs.append("unattributed:" + name)
        work.known = sorted(set(work.known) | current_ids)
        work.processed = sorted(set(work.processed))
        queries = []
        for name in sorted(active):
            if shortage[name] and name not in work.roots and catalog:
                queries += [dict(key="count:" + name, kind="pmf", class_name=name,
                                 size=shortage[name] + 1),
                            dict(key="place:" + name, kind="weights", class_name=name, candidates=sorted(catalog))]
        for key, entity in sorted(entities.items()):
            if entity.room_id is not None or (key not in target_ids and entity.name not in active):
                continue
            if key not in target_ids and entity.stamp.step < 0:
                continue
            if str(key) in work.instances and sum(work.instances[str(key)].values()) > 0:
                continue
            candidates = sorted(k for k in catalog if key not in target_ids or k.startswith("v:"))
            if not candidates:
                from .prediction import PredictionError
                raise PredictionError("no legal location for known identity " + str(key), aggregate_usage(attempts))
            queries.append(dict(key="instance:" + str(key), kind="weights", class_name=entity.name,
                                role="goal_target" if key in target_ids else "known_object", candidates=candidates))
        if queries:
            try:
                answers, usage = predict_discovery(facts, queries, catalog, client, config=config,
                                                   agent_index=agent_index)
            except Exception as error:
                if hasattr(error, "usage"):
                    error.usage = aggregate_usage(attempts + error.usage.get("llm_attempts", []))
                raise
            attempts += usage["llm_attempts"]
            for query in queries:
                key = query["key"]
                if key.startswith("count:"):
                    name = query["class_name"]
                    values = answers["place:" + name]
                    weights = dict(zip(sorted(catalog), values))
                    work.roots[name] = dict(
                        pmf=answers[key],
                        weights=_condition_groups(
                            weights, facts, name, evidence_catalog, (), work.processed))
                elif key.startswith("instance:"):
                    values = answers[key]
                    work.instances[key.split(":")[1]] = dict(zip(query["candidates"], values))
        work.repairs += len(repairs)
        previous = self.to_seed()
        current = work.to_seed()
        current["revision"] = previous["revision"]
        if current != previous:
            work.revision += 1
        work.last_diagnostics = dict(refinements=refinements, repairs=repairs,
                                     initialized=sum(q["kind"] == "pmf" for q in queries))
        validated = self.from_seed(work.to_seed(), facts)
        self.__dict__.update(validated.__dict__)
        self.last_diagnostics = work.last_diagnostics
        usage = aggregate_usage(attempts)
        return usage
    # ==================== End Coop: Synchronized Probability State ====================
