"""One LLM location per missing item, resolved into one temporary planning graph."""

from collections import Counter
from dataclasses import dataclass, fields, replace
import json
import time

from .config import MethodConfig
from .models import Assumption, DeterministicGraph, Entity, FactGraph, PlanningGraph

SYSTEM_PROMPT = """Predict missing task-object locations in a partially observed household.
Use only the supplied public facts and household commonsense. Each query needs
one candidate location_id. Room candidates mean unexplored visible room areas;
container candidates mean the interiors of observed, unopened storage containers.
Exploring a room excludes its visible area, not its still-unchecked containers.
A target query locates the target furniture itself, not its contents, and accepts
room candidates only. Multiple objects may share a location; do not distribute
them artificially. IDs are labels, not clues. Predictions are temporary assumptions.
Return exactly this json object: {"predictions":[{"query_id":"...","location_id":123},...]}.
Include each query exactly once, in any order. Use integer IDs from that query's
candidate_ids. Do not predict object identities, properties, probabilities or actions.
"""


class PredictionError(ValueError):
    def __init__(self, message, usage=None):
        super().__init__(message)
        self.usage = usage or aggregate_usage(())


def aggregate_usage(attempts):
    result = {"llm_called": bool(attempts), "llm_attempts": list(attempts)}
    for field in ("prompt_tokens", "completion_tokens", "total_tokens", "wall_time"):
        result[field] = sum(a.get(field, 0) for a in attempts)
    return result


@dataclass(frozen=True)
class PredictionRequest:
    facts: FactGraph
    draft_entities: tuple
    payload: dict


def build_request(graph):
    if type(graph) is not FactGraph:
        raise TypeError("prediction input must be facts")
    graph.validate()
    if any(graph.remaining(g) for g in graph.goals) and any(a.position is None or a.room_id is None for a in graph.agents):
        raise PredictionError("both agent positions must be communicated before prediction")
    rooms = {r.id: r for r in graph.rooms}
    target_ids = {g.target_id for g in graph.goals}
    needed = Counter()
    completed = set()
    for goal in graph.goals:
        needed[goal.object_class] += graph.remaining(goal)
        completed.update(graph.satisfied(goal))
    entities = {e.id: e for e in graph.entities}
    room_candidates = [{"id": r.id, "kind": "room_area", "name": r.name}
                       for r in graph.rooms if not r.explored]
    containers = [{"id": e.id, "kind": "container", "name": e.name,
                   "room": rooms[e.room_id].name} for e in graph.entities
                  if e.searchable and not e.explored and e.open_state == "closed"
                  and e.id not in target_ids and e.room_id in rooms and rooms[e.room_id].explored]
    object_ids = [c["id"] for c in room_candidates + containers]
    room_ids = [c["id"] for c in room_candidates]
    queries = []
    virtual_id = -1
    for name, count in sorted(needed.items()):
        if count == 0:
            continue
        available = [e for e in graph.entities if e.name == name and e.id not in completed]
        located = [e for e in available if e.room_id is not None]
        missing = max(0, count - len(located))
        unlocated = sorted((e for e in available if e.room_id is None), key=lambda e: e.id)
        for slot in range(missing):
            if slot < len(unlocated):
                entity = unlocated[slot]
            else:
                entity = Entity(virtual_id, name)
                entities[virtual_id] = entity
                virtual_id -= 1
            queries.append({"query_id": "object:" + str(entity.id), "class_name": name,
                            "role": "task_object", "candidate_ids": object_ids})
    for target_id in sorted({g.target_id for g in graph.goals if graph.remaining(g)}):
        entity = entities[target_id]
        if entity.room_id is None:
            queries.append({"query_id": "target:" + str(target_id), "class_name": entity.name,
                            "role": "goal_target", "candidate_ids": room_ids})
    for query in queries:
        if not query["candidate_ids"]:
            raise PredictionError("no legal candidate for " + query["query_id"])
    payload = {
        "task": [{"relation": g.relation, "object_class": g.object_class,
                  "target": entities[g.target_id].name, "remaining": graph.remaining(g)} for g in graph.goals],
        "candidates": room_candidates + containers,
        "known_locations": [{"class_name": e.name, "room": rooms[e.room_id].name,
                             "inside_container": entities[e.inside].name if e.inside in entities else None,
                             "held": bool(e.holders)} for e in graph.entities if e.room_id in rooms],
        "negative_evidence": {
            "explored_visible_room_areas": [r.name for r in graph.rooms if r.explored],
            "checked_containers": [{"class_name": e.name, "room": rooms[e.room_id].name}
                                   for e in graph.entities if e.container and e.explored and e.room_id in rooms]},
        "queries": queries,
    }
    return PredictionRequest(graph, tuple(entities[k] for k in sorted(entities)), payload)


def parse_predictions(request, text):
    def unique_pairs(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("duplicate JSON key")
            result[key] = value
        return result
    response = json.loads(text, object_pairs_hook=unique_pairs)
    if not isinstance(response, dict) or set(response) != {"predictions"} or not isinstance(response["predictions"], list):
        raise ValueError("expected exactly predictions array")
    queries = {q["query_id"]: q for q in request.payload["queries"]}
    result, seen, errors = {}, set(), []
    for index, row in enumerate(response["predictions"]):
        if not isinstance(row, dict) or set(row) != {"query_id", "location_id"}:
            errors.append("row {} fields differ".format(index))
            continue
        key, location = row["query_id"], row["location_id"]
        if not isinstance(key, str) or key not in queries:
            errors.append("row {} has unknown query_id".format(index))
            continue
        if key in seen:
            errors.append("repeated query " + key)
            continue
        seen.add(key)
        if type(location) is not int or location not in queries[key]["candidate_ids"]:
            errors.append("{} location_id={} is not in candidate_ids={}".format(
                key, repr(location), queries[key]["candidate_ids"]))
            continue
        result[key] = location
    missing = sorted(set(queries) - seen)
    if missing:
        errors.append("missing query_ids=" + repr(missing))
    if errors:
        raise ValueError("; ".join(errors))
    return result


def resolve_request(request, predictions, *, partial=False):
    facts = request.facts
    entities = {e.id: e for e in request.draft_entities}
    rooms = {r.id: r for r in facts.rooms}
    assumptions = []
    # Validate even when callers bypass the LLM parser.
    predictions = parse_predictions(request, json.dumps({"predictions": [
        {"query_id": k, "location_id": v} for k, v in predictions.items()]}))
    for query in request.payload["queries"]:
        key = query["query_id"]
        entity_id = int(key.split(":", 1)[1])
        location = predictions[key]
        if location in rooms:
            room_id, position, parent = location, rooms[location].center, location
        else:
            container = entities[location]
            room_id, parent = container.room_id, location
            position = container.position or rooms[room_id].center
        entities[entity_id] = replace(entities[entity_id], room_id=room_id, position=position, inside=parent, on=())
        assumptions.append(Assumption(entity_id, "location", location))
    active_classes = {g.object_class for g in facts.goals if facts.remaining(g)}
    class_properties = dict(facts.class_coop)
    for key, entity in tuple(entities.items()):
        if entity.room_id is not None and entity.position is None:
            parent = entities.get(entity.inside)
            position = parent.position if parent and parent.position is not None else rooms[entity.room_id].center
            entity = replace(entity, position=position)
            assumptions.append(Assumption(key, "position_at_parent_or_room", entity.inside or entity.room_id))
        if entity.name in active_classes and entity.coop is None:
            value = class_properties.get(entity.name, 1)
            entity = replace(entity, coop=value)
            if entity.name not in class_properties:
                assumptions.append(Assumption(key, "coop", value))
        # A goal container of unknown state is not silently assumed accessible.
        if entity.container and entity.open_state is None:
            entity = replace(entity, open_state="closed")
            assumptions.append(Assumption(key, "closed", 1))
        entities[key] = entity
    agents = facts.agents
    coop = facts.cooperation
    if coop.object_id is not None:
        holder = agents[coop.holder]
        agents = tuple(replace(a, room_id=holder.room_id, position=holder.position,
                               close=holder.close) if a.index == coop.supporter else a for a in agents)
    values = {f.name: getattr(facts, f.name) for f in fields(FactGraph)}
    values.update(entities=tuple(entities[k] for k in sorted(entities)), agents=agents)
    result = DeterministicGraph(PlanningGraph(**values), tuple(assumptions))
    if partial:
        result.graph.validate(allow_virtual=True)
    else:
        result.validate()
    return result


def factual_planning_graph(facts):
    """A partial planning view never invents supply or target locations."""
    return resolve_request(PredictionRequest(facts, facts.entities, {"queries": []}), {}, partial=True)


def resolve_with_llm(graph, llm_client=None, *, agent_index=None, config=None):
    """Client contract: generate(messages, agent_id) -> (text, usage_dict).

    No network SDK is imported here. Usage is local to this call, including
    unsuccessful attempts; an API failure is reported without invented answers.
    """
    config = config or MethodConfig()
    request = build_request(graph)
    if not request.payload["queries"]:
        return resolve_request(request, {}), aggregate_usage(())
    if llm_client is None:
        raise PredictionError("location prediction requires an LLM client")
    messages = [{"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": json.dumps(request.payload, ensure_ascii=False)}]
    attempts = []
    for attempt in range(config.prediction_attempts):
        started = time.monotonic()
        try:
            text, usage = llm_client.generate(deepcopy_messages(messages),
                                             agent_id=graph.source if agent_index is None else agent_index)
        except Exception as error:
            usage = dict(getattr(error, "usage", {}) or {})
            attempts.append(_attempt(usage, "api_failed", None, str(error), time.monotonic() - started))
            raise PredictionError("LLM request failed: " + str(error), aggregate_usage(attempts)) from error
        try:
            predictions = parse_predictions(request, text)
        except (ValueError, TypeError) as error:
            attempts.append(_attempt(usage, "invalid", str(text)[:12000], str(error), time.monotonic() - started))
            if attempt + 1 == config.prediction_attempts:
                raise PredictionError("invalid location prediction: " + str(error), aggregate_usage(attempts)) from error
            allowed = [{"query_id": q["query_id"], "candidate_ids": q["candidate_ids"]}
                       for q in request.payload["queries"]]
            messages.extend([{"role": "assistant", "content": str(text)},
                             {"role": "user", "content":
                              "Correct the JSON. Return exactly these query_ids once each, with no extra rows: "
                              + json.dumps(allowed, ensure_ascii=False) + ". Validation error: " + str(error)}])
            continue
        attempts.append(_attempt(usage, "applied", str(text)[:12000], None, time.monotonic() - started))
        try:
            resolved = resolve_request(request, predictions)
        except ValueError as error:
            attempts[-1]["status"] = "resolution_failed"
            attempts[-1]["error"] = str(error)
            raise PredictionError(str(error), aggregate_usage(attempts)) from error
        return resolved, aggregate_usage(attempts)


def deepcopy_messages(messages):
    return [dict(m) for m in messages]


# ==================== Coop: Discovery Probability Protocol ====================
DISCOVERY_PROMPT = """Estimate a finite discovery model from public household semantics.
OUTPUT FORMAT:
Return only a json object with exactly one field, "rows", containing an array.
Each row has EXACTLY "key" and "levels". Copy its key and every candidate_id
from required_rows exactly once. Every value in "levels" must be one of the JSON
numbers 0, 1, 2, 3 or 4. Never return probabilities, counts, arrays, input metadata
or explanations in a row.
Fictional format example (NOT this request):
{"rows":[{"key":"count:fictional_item","levels":{"n0":1,"n1":3,"n2":2,"n3":0}},
{"key":"place:fictional_item","levels":{"g0":1,"g1":4}}]}.
Notice that n3 identifies the capped count hypothesis 3, while its value 0 is a support
level. Never copy the example keys, candidate IDs or levels into your answer.
Return all required rows exactly once in the supplied order. No markdown or extra fields.

MEANING OF LEVELS:
Levels mean relative support: 0=very unlikely but possible, 1=unlikely,
2=neutral/insufficient evidence, 3=plausible, 4=strongly plausible.
These are NOT probabilities. Code maps each level L to 2**L and normalizes the row.
Equal levels are valid when evidence does not distinguish alternatives. Do not
force differences or assign high support simply because a task needs an object.
Judge whether a useful object is physically stored there now and would be found by
inspection. This is storage/discovery likelihood, not a broad association between
the object and activities performed near that location. Consider the object class,
whether the location normally stores or contains such loose objects, and its room
together. Room context may adjust but must not override an unsuitable location
function. An unexplored location means missing evidence, not a semantic match.
Level 2 means the supplied evidence truly does not distinguish the alternatives;
use lower support when the location's normal storage purpose clearly differs.
Generic example: clothing is usually more discoverable in bedroom storage than in
a kitchen refrigerator, although an atypical location remains possible.
For kind="count_support", useful_limit is the number of unfinished task slots not
already covered by known identities. Each candidate explicitly gives an id, count
and relation. relation="exact" means exactly that many useful anonymous objects;
relation="at_least" is the final saturated hypothesis and means enough anonymous
objects to cover useful_limit, possibly with irrelevant extras. For useful_limit=1,
judge only none versus at least one. For useful_limit=3, judge 0, 1, 2 and at least
3. Extra objects cannot fill extra task slots. Copy the n-prefixed id; never use the
count itself as the support level. Do NOT sum levels to one.
For kind="weights", candidates explicitly describe semantic location groups. Return
one levels entry for each candidate id. Containers of the same class in the same
room share a group: rate the TOTAL allocation mass of that group. Code normalizes
group weights, then divides each group's mass equally among its real members.
More copies of a container are not, by themselves, stronger evidence.
Visible areas, hidden areas and unmapped residuals remain separate alternatives.
Distinguish groups when household storage semantics supports a difference; use
neutral levels when it does not. Do not give high support to a fixture merely
because an object is used in the same room; ask whether inspecting it would
normally reveal the object inside. A location can be atypical without being
impossible. Rate relative discovery plausibility. All-zero levels are valid neutral relative
support. Different objects may occupy the same region. Do not return a location ID.
When newly observed containers change the candidates, compare every supplied group
again; a new container's semantics may change its room's support relative to other
rooms. Do not preserve an earlier room total merely because it was predicted first.
For role="goal_target", locate the furniture itself, not the objects to deliver.

PUBLIC EVIDENCE:
Exclude every known identity from anonymous counts, including held or unlocated
known objects. Task demand is not a count of hidden supply; known objects are not
new discoveries. Task destinations are not evidence of initial object locations.
The candidate fields room, location and kind state the relevant public semantics.
A checked visible area does not mean that closed containers there are empty. Use
the supplied public semantics only; group IDs are labels, not semantic clues. Household semantics
are a soft prior: atypical locations remain possible, and repeated containers do
not imply proportionally more hidden supply. Estimate anonymous supply separately
from task demand; do not assume every inspected container yields a useful object. Do not predict cooperative
labels, navigation costs or actions. Before responding, check the row count, exact
field names, every candidate ID, levels 0..4, and absence of extra fields.
"""


def predict_discovery(facts, queries, catalog, client, *, config=None, agent_index=None):
    from .discovery_belief import normalized, region_group
    config = config or MethodConfig()
    original, queries, group_members = queries, [], {}
    room_names = {room.id: room.name for room in facts.rooms}
    for query in original:
        if query["kind"] == "pmf":
            limit = query["size"] - 1
            queries.append(dict(key=query["key"], kind="count_support",
                                class_name=query.get("class_name", query["key"].split(":", 1)[-1]),
                                useful_limit=limit,
                                candidates=[dict(id="n" + str(i), count=i,
                                                 relation="at_least" if i == limit else "exact")
                                            for i in range(query["size"])]))
            continue
        groups = {}
        for region in query["candidates"]:
            groups.setdefault(region_group(region, catalog), []).append(region)
        candidates, members = [], {}
        for index, regions in enumerate(groups.values()):
            group_id = "g" + str(index)
            representative = regions[0]
            item = catalog.get(representative, {})
            kind = item.get("kind", "unknown")
            room = room_names.get(item.get("room_id"))
            location = item.get("name")
            if kind == "visible":
                location = "visible area"
            elif kind == "hidden":
                location = "hidden contents"
            elif kind == "unmapped":
                location = "unmapped residual"
            candidates.append(dict(id=group_id, room=room, location=location or representative, kind=kind))
            members[group_id] = regions
        group_members[query["key"]] = members
        queries.append(dict(query, candidates=candidates))

    def expand(answers):
        result = dict(answers)
        for query, grouped in zip(original, queries):
            if query["kind"] == "weights":
                members = group_members[query["key"]]
                weights = {r: p / len(members[candidate["id"]])
                           for candidate, p in zip(grouped["candidates"], answers[query["key"]])
                           for r in members[candidate["id"]]}
                result[query["key"]] = [weights[r] for r in query["candidates"]]
        return result

    automatic = {q["key"]: [1.] for q in queries if q["kind"] == "weights" and len(q["candidates"]) == 1}
    pending = [q for q in queries if q["key"] not in automatic]
    if not pending:
        return expand(automatic), aggregate_usage(())
    if client is None:
        raise PredictionError("discovery prediction requires an LLM client")
    names = {g.object_class for g in facts.goals}
    required_rows = [dict(key=q["key"], candidate_ids=[candidate["id"]
                                                       for candidate in q["candidates"]])
                     for q in pending]
    payload = dict(required_rows=required_rows,
        queries=pending,
        task=[dict(class_name=g.object_class, remaining=facts.remaining(g), target=next(e.name for e in facts.entities
                                                                                      if e.id == g.target_id)) for g in facts.goals],
        semantics=[dict(region=e.region, classes=list(e.semantics)) for e in facts.inspections],
        known=[dict(class_name=e.name, held=bool(e.holders), located=e.room_id is not None)
               for e in facts.entities if e.name in names and e.stamp.step >= 0],
        checked=[e.region for e in facts.inspections if e.complete])
    messages = [dict(role="system", content=DISCOVERY_PROMPT),
                dict(role="user", content=json.dumps(payload, ensure_ascii=False))]
    attempts = []
    def parse(text):
        def unique(pairs):
            out = {}
            for k, v in pairs:
                if k in out:
                    raise ValueError("duplicate JSON key")
                out[k] = v
            return out
        data = json.loads(text, object_pairs_hook=unique)
        if not isinstance(data, dict) or set(data) != {"rows"} or not isinstance(data["rows"], list):
            raise ValueError("expected exactly rows")
        expected, result, errors, seen = {q["key"]: q for q in pending}, dict(automatic), [], set()
        for index, row in enumerate(data["rows"]):
            if not isinstance(row, dict) or not isinstance(row.get("key"), str):
                errors.append("rows[{}]: expected an object with a string key".format(index))
                continue
            key = row["key"]
            if key not in expected or key in seen:
                errors.append("unknown/repeated key " + key)
                continue
            seen.add(key)
            q = expected[key]
            try:
                if set(row) != {"key", "levels"}:
                    raise ValueError('expected exactly "key" and "levels"; got fields ' + repr(sorted(row)))
                levels = row["levels"]
                candidate_ids = [candidate["id"] for candidate in q["candidates"]]
                if not isinstance(levels, dict) or set(levels) != set(candidate_ids):
                    raise ValueError("expected levels for candidate_ids {}; got {}".format(
                        candidate_ids, sorted(levels) if isinstance(levels, dict) else type(levels).__name__))
                invalid = ["{}={}".format(candidate_id, repr(levels[candidate_id]))
                           for candidate_id in candidate_ids
                           if type(levels[candidate_id]) not in (int, float)
                           or levels[candidate_id] not in (0, 1, 2, 3, 4)]
                if invalid:
                    raise ValueError("expected support levels 0..4; invalid " + ", ".join(invalid))
                values = [int(levels[candidate_id]) for candidate_id in candidate_ids]
                result[key] = normalized([2 ** v for v in values])
            except ValueError as error:
                errors.append("rows[{}] {}: {}".format(index, key, error))
        errors += ["missing " + k for k in expected if k not in seen]
        if errors:
            raise ValueError("; ".join(errors))
        return result
    for attempt in range(config.prediction_attempts):
        started = time.monotonic()
        try:
            text, usage = client.generate(deepcopy_messages(messages), agent_id=facts.source if agent_index is None else agent_index)
        except Exception as error:
            attempts.append(_attempt(getattr(error, "usage", {}), "api_failed", None, type(error).__name__, time.monotonic() - started))
            raise PredictionError("discovery backend failed (" + type(error).__name__ + ")", aggregate_usage(attempts)) from error
        try:
            result = parse(text)
        except (ValueError, TypeError) as error:
            attempts.append(_attempt(usage, "invalid", str(text)[:12000], str(error), time.monotonic() - started))
            if attempt + 1 == config.prediction_attempts:
                raise PredictionError(str(error), aggregate_usage(attempts)) from error
            messages.extend([dict(role="assistant", content=str(text)), dict(role="user", content=
                'Rewrite the COMPLETE json answer. Every row has only "key" and "levels". '
                'Map every candidate_id exactly once to one of 0, 1, 2, 3 or 4. For n-prefixed IDs, '
                'the number in the ID is the capped count hypothesis, never the level. Do not return arrays, '
                'counts or probabilities. Required rows: ' + json.dumps(required_rows) + '.'
                + ". Validation errors: " + str(error))])
            continue
        attempts.append(_attempt(usage, "applied", str(text)[:12000], None, time.monotonic() - started))
        return expand(result), aggregate_usage(attempts)
# ==================== End Coop: Discovery Probability Protocol ====================


def _attempt(usage, status, output, error, elapsed):
    result = dict(usage or {})
    for name in ("prompt_tokens", "completion_tokens"):
        result[name] = int(result.get(name, 0) or 0)
    result["total_tokens"] = int(result.get("total_tokens", result["prompt_tokens"] + result["completion_tokens"]) or 0)
    result.update(wall_time=elapsed, status=status, output=output, error=error)
    return result


class PredictionClientError(RuntimeError):
    def __init__(self, message, usage):
        super().__init__(message)
        self.usage = usage


class OpenAIPredictionClient:
    """Lazy SDK adapter for official and OpenAI-compatible chat endpoints.

    OPENAI_BASE_URL takes precedence over OPENAI_API_BASE; omitting both uses
    the official endpoint. SDK interfaces are selected by capability.
    """

    def __init__(self, model, temperature=0., top_p=1., max_tokens=4096, client=None):
        self.model, self.temperature, self.top_p, self.max_tokens = model, temperature, top_p, max_tokens
        self.client = client
        self._request_options = {}

    def generate(self, messages, agent_id=None):
        started = time.monotonic()
        try:
            if self.client is None:
                import os
                import openai
                key = os.getenv("OPENAI_API_KEY") or os.getenv("OPENAI_KEY")
                base_url = (os.getenv("OPENAI_BASE_URL") or os.getenv("OPENAI_API_BASE")
                            or "https://api.openai.com/v1")
                if callable(getattr(openai, "OpenAI", None)):
                    self.client = openai.OpenAI(api_key=key, base_url=base_url,
                                                max_retries=0, timeout=60.)
                else:
                    self.client = openai
                    # Per-request settings avoid changing SDK globals shared by agents.
                    self._request_options = {"api_key": key, "api_base": base_url}
            params = dict(model=self.model, messages=messages, temperature=self.temperature,
                          top_p=self.top_p, max_tokens=self.max_tokens, n=1,
                          response_format={"type": "json_object"})
            if hasattr(self.client, "chat"):
                response = self.client.chat.completions.create(**params)
            else:
                response = self.client.ChatCompletion.create(
                    request_timeout=60., **self._request_options, **params)
            def value(obj, key, default=None):
                return obj.get(key, default) if isinstance(obj, dict) else getattr(obj, key, default)
            choices = value(response, "choices", ())
            text = value(value(choices[0], "message"), "content") if choices else None
            if not isinstance(text, str) or not text.strip():
                raise ValueError("model returned no textual prediction")
            raw = value(response, "usage", {})
            usage = {k: int(value(raw, k, 0) or 0) for k in ("prompt_tokens", "completion_tokens")}
            usage.update(total_tokens=int(value(raw, "total_tokens", sum(usage.values())) or 0),
                         model=self.model, agent_id=agent_id, wall_time=time.monotonic() - started)
            return text, usage
        except Exception as error:
            # Do not interpolate SDK exception text, request headers or credentials.
            raise PredictionClientError("prediction backend failed ({})".format(type(error).__name__),
                                        dict(model=self.model, agent_id=agent_id,
                                             wall_time=time.monotonic() - started)) from error
