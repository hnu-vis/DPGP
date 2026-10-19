"""Simple LLM room prediction and temporary deterministic graph creation."""

from dataclasses import dataclass
import json
import os
import time
from typing import Any, Dict, List, Optional, Tuple

from eval.llm_usage import complete_sum

from .config import MethodConfig
from .models import DeterministicGraph, FactGraph


SYSTEM_PROMPT = """You and another agent are searching rooms in a home. Your shared goal is to
find the required target objects and transport them to the designated goal
location as efficiently as possible. The reliable facts observed by both agents
have already been combined. Some required objects still have unknown locations.

For every object in unknown_objects, predict the single candidate room where it
is most likely to be found. Candidate rooms are either unexplored rooms or, when
all rooms have already been explored, the least-revisited rooms selected for a
new visual search. Use everyday household commonsense: consider the object's
usual function, each room's function, and where related targets were observed.
Treat every prediction as a temporary search hypothesis, never as an observed
fact. If evidence is weak, still select the most plausible listed candidate.

Rules:
1. Return exactly one prediction for every graph_index in unknown_objects.
2. room_node_id must be selected from candidate_rooms.
   known_target_object_locations is context, not a list of allowed rooms.
3. Do not invent objects, rooms, coordinates, properties, observations, or task facts.
4. Do not merge graphs, choose actions, make a plan, or explain the answer.
5. Return only JSON in this exact shape:
   {"predictions":[{"graph_index":0,"room_node_id":1}]}
"""


DISCOVERY_SYSTEM_PROMPT = """Estimate where still-unlocated household task objects are
physically likely to be stored and discovered, using household commonsense and
the supplied public facts. Judge each concrete object class and each room's
function together. Both a relevant functional room and a shared living space
can be plausible. A broad association with an activity is weaker evidence than
a room's normal role in storing or keeping that object. Personal belongings
need not have the same distribution as specialized work items.
For every query, rate EVERY supplied room candidate:
0=very unlikely but possible, 1=unlikely, 2=neutral/insufficient evidence,
3=plausible, 4=strongly plausible. These are relative support levels, NOT
probabilities; code normalizes 2**level. Distinguish rooms when semantics or
public discoveries justify it. Equal levels are appropriate when evidence
does not distinguish alternatives, including equally plausible rooms with the
same function. Room IDs and candidate keys are labels, not semantic clues.
First-discovery locations are positive evidence, not exclusive class locations.
Do not infer initial storage from the task destination or transported objects.
Unexplored means missing evidence, not high support. A visual scan can miss an
object: no empty scan proves absence. Atypical rooms remain possible. Do not
invent observations, objects, rooms, coordinates or cooperative properties.
Do not choose actions or explain the answer. Return each query_key and every
candidate_key exactly once, with integer levels 0..4 and no extra fields.
Return only JSON in this exact shape:
{"rows":[{"query_key":"place:fictional_item","levels":{"g0":2,"g1":4}}]}
The example is structural only; never copy its labels or levels.
"""


DEPENDENCY_ERROR_TYPES = frozenset({
    "APIConnectionError", "APITimeoutError", "ConnectError",
    "ConnectionError", "ConnectTimeout", "NewConnectionError",
    "ProxyError", "ReadTimeout", "Timeout", "TimeoutError",
})


@dataclass(frozen=True)
class PredictionResult:
    graph: Optional[DeterministicGraph]
    usage: dict
    failed: bool


@dataclass(frozen=True)
class DiscoveryPredictionResult:
    weights: Optional[Dict[str, Dict[int, float]]]
    usage: dict
    failed: bool


def is_llm_dependency_failure(usage: dict) -> bool:
    error_type = usage.get("prediction_error_type")
    error_text = str(usage.get("prediction_error") or "").lower()
    return (error_type in DEPENDENCY_ERROR_TYPES
            or "connection error" in error_text
            or "connect to proxy" in error_text)


def _payload(graph: FactGraph, reopen_all_rooms: bool = False) -> dict:
    rooms = {item.node_id: item for item in graph.rooms}
    if reopen_all_rooms:
        minimum = min(item.revisit_count for item in graph.rooms)
        candidates = [
            item for item in graph.rooms if item.revisit_count == minimum
        ]
    else:
        candidates = [item for item in graph.rooms if not item.explored]
    return {
        "goal_required_objects": dict(graph.goal.required_objects),
        "target_room_node_id": graph.goal.target_room_node_id,
        "unknown_objects": [
            {"graph_index": item.graph_index,
             "class_name": item.class_name,
             "object_id": item.object_id}
            for item in graph.objects
            if not item.finished and not item.holder_agent_ids
            and item.position is None
        ],
        "candidate_rooms": [
            {"room_node_id": item.node_id, "room_name": item.room_name}
            for item in candidates
        ],
        "known_target_object_locations": [
            {"class_name": item.class_name,
             "object_id": item.object_id,
             "room_node_id": item.room_node_id,
             "room_name": (rooms[item.room_node_id].room_name
                           if item.room_node_id is not None else None)}
            for item in graph.objects
            if item.position is not None and not item.finished
        ],
    }


def _response_json(text: str) -> dict:
    text = text.strip()
    fence = "```"
    if text.startswith(fence):
        text = text.split("\n", 1)[1].rsplit(fence, 1)[0]
        if text.lstrip().startswith("json"):
            text = text.lstrip()[4:].lstrip()
    response = json.loads(text)
    if not isinstance(response, dict):
        raise ValueError("LLM response must be a JSON object")
    return response


def _room_predictions(payload: dict, response: dict) -> Dict[int, int]:
    predictions = response.get("predictions")
    if not isinstance(predictions, list):
        raise ValueError("LLM response requires a predictions list")
    expected = {item["graph_index"] for item in payload["unknown_objects"]}
    candidates = {
        item["room_node_id"] for item in payload["candidate_rooms"]
    }
    selected = {}
    for item in predictions:
        if not isinstance(item, dict):
            raise ValueError("each room prediction must be an object")
        graph_index = int(item["graph_index"])
        room_node_id = int(item["room_node_id"])
        if graph_index in selected:
            raise ValueError("duplicate graph_index in room predictions")
        if room_node_id not in candidates:
            raise ValueError("room prediction must use a candidate room")
        selected[graph_index] = room_node_id
    if set(selected) != expected:
        raise ValueError("room predictions must cover exactly all unknown objects")
    return selected


def _discovery_payload(graph: FactGraph, class_names, first_locations=None) -> dict:
    rooms = tuple(sorted(graph.rooms, key=lambda item: item.node_id))
    names = {room.node_id: room.room_name for room in rooms}
    first_locations = first_locations or {}
    room_keys = {room.node_id: "g{}".format(index)
                 for index, room in enumerate(rooms)}
    return {
        "query_kind": "discovery_support",
        "ordinal_scale": {"minimum": 0, "maximum": 4,
                          "weight": "2^level"},
        "room_candidates": [
            {"candidate_key": room_keys[room.node_id],
             "room_node_id": room.node_id, "room_name": room.room_name}
            for room in rooms
        ],
        "queries": [
            {"query_key": "place:{}".format(name), "class_name": name}
            for name in sorted(set(class_names))
        ],
        "first_discovery_locations": [
            {"class_name": item.class_name, "object_id": item.object_id,
             "room_node_id": first_locations[item.object_id],
             "room_name": names[first_locations[item.object_id]]}
            for item in graph.objects
            if first_locations.get(item.object_id) is not None
        ],
    }


def _discovery_weights(payload: dict, response: dict):
    if set(response) != {"rows"} or not isinstance(response["rows"], list):
        raise ValueError("discovery response requires only a rows list")
    room_by_key = {
        item["candidate_key"]: item["room_node_id"]
        for item in payload["room_candidates"]
    }
    expected = {item["query_key"]: item["class_name"]
                for item in payload["queries"]}
    selected = {}
    for row in response["rows"]:
        if not isinstance(row, dict) or set(row) != {"query_key", "levels"}:
            raise ValueError("each discovery row requires query_key and levels")
        query_key, levels = row["query_key"], row["levels"]
        if query_key not in expected or query_key in selected:
            raise ValueError("discovery rows must cover each query exactly once")
        if not isinstance(levels, dict) or set(levels) != set(room_by_key):
            raise ValueError("discovery levels must cover every room candidate")
        raw = {}
        for key, level in levels.items():
            if type(level) is not int or not 0 <= level <= 4:
                raise ValueError("discovery levels must be integers in [0, 4]")
            raw[room_by_key[key]] = float(2 ** level)
        total = sum(raw.values())
        selected[expected[query_key]] = {
            room_id: value / total for room_id, value in raw.items()
        }
    if set(selected) != set(expected.values()):
        raise ValueError("discovery rows must cover each query exactly once")
    return selected


def _usage(details: List[dict], status: str, rooms_reopened: bool,
           error: Optional[Exception] = None) -> dict:
    result = {
        "llm_called": True,
        "llm_call_count": len(details),
        "llm_attempt_count": len(details),
        "llm_attempt_details": details,
        "prompt_tokens": complete_sum(item.get("prompt_tokens") for item in details),
        "completion_tokens": complete_sum(
            item.get("completion_tokens") for item in details
        ),
        "total_tokens": complete_sum(item.get("total_tokens") for item in details),
        "wall_time": sum(float(item.get("wall_time", 0.0) or 0.0)
                         for item in details),
        "prediction_status": status,
        "rooms_reopened": rooms_reopened,
    }
    if error is not None:
        result["prediction_error"] = str(error)
        result["prediction_error_type"] = type(error).__name__
    return result


def predict_deterministic_graph(
    facts: FactGraph,
    llm_client,
    agent_id: int,
    config: Optional[MethodConfig] = None,
    llm_enabled: bool = True,
) -> PredictionResult:
    """Resolve one factual graph without mutating it or creating alternatives."""
    facts.validate()
    if agent_id not in (0, 1):
        raise ValueError("agent_id must be 0 or 1")
    config = config or MethodConfig()
    payload = _payload(facts)
    if not payload["unknown_objects"]:
        graph = DeterministicGraph.from_facts(facts, {})
        return PredictionResult(
            graph, {"llm_called": False, "prediction_status": "not_needed"}, False
        )
    if not llm_enabled:
        return PredictionResult(None, {
            "llm_called": False,
            "prediction_status": "deferred",
            "prediction_error": "llm_circuit_open",
            "rooms_reopened": False,
        }, True)

    rooms_reopened = not payload["candidate_rooms"]
    if rooms_reopened:
        payload = _payload(facts, reopen_all_rooms=True)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": json.dumps(payload, separators=(",", ":"))},
    ]
    details = []
    last_error = None
    for attempt in range(1, config.prediction_attempts + 1):
        attempt_usage = {}
        started = time.perf_counter()
        try:
            text, attempt_usage = llm_client.generate(messages, agent_id=agent_id)
            predictions = _room_predictions(payload, _response_json(text))
            graph = DeterministicGraph.from_facts(
                facts, predictions, reopen_all_rooms=rooms_reopened
            )
        except Exception as error:
            last_error = error
            detail = dict(attempt_usage or {})
            detail.update(
                attempt=attempt,
                prediction_status="failed",
                prediction_error=str(error),
                prediction_error_type=type(error).__name__,
                wall_time=float(detail.get(
                    "wall_time", time.perf_counter() - started
                )),
            )
            details.append(detail)
            if not is_llm_dependency_failure(detail):
                allowed = [item["room_node_id"]
                           for item in payload["candidate_rooms"]]
                messages = messages[:2] + [{
                    "role": "user",
                    "content": (
                        "The previous prediction was invalid: {}. "
                        "Allowed room_node_id values are {}. "
                        "Choose only from these values for every unknown "
                        "graph_index and return only the required JSON."
                    ).format(error, allowed),
                }]
            continue
        detail = dict(attempt_usage or {})
        detail.update(
            attempt=attempt,
            prediction_status="applied",
            wall_time=float(detail.get("wall_time", time.perf_counter() - started)),
        )
        details.append(detail)
        return PredictionResult(
            graph, _usage(details, "applied", rooms_reopened), False
        )
    return PredictionResult(
        None, _usage(details, "failed", rooms_reopened, last_error), True
    )


def predict_discovery_weights(
    facts: FactGraph,
    class_names,
    llm_client,
    agent_id: int,
    config: Optional[MethodConfig] = None,
    llm_enabled: bool = True,
    first_locations=None,
) -> DiscoveryPredictionResult:
    """Query one shared room-support row per unresolved object class."""
    facts.validate()
    if agent_id not in (0, 1):
        raise ValueError("agent_id must be 0 or 1")
    names = tuple(sorted(set(str(name).strip().lower()
                             for name in class_names if str(name).strip())))
    required = {name for name, _ in facts.goal.required_objects}
    if not set(names) <= required:
        raise ValueError("discovery classes must belong to the task goal")
    if not names:
        return DiscoveryPredictionResult(
            {}, {"llm_called": False, "prediction_status": "not_needed",
                 "prediction_kind": "discovery_support"}, False
        )
    if not llm_enabled:
        return DiscoveryPredictionResult(None, {
            "llm_called": False, "prediction_status": "deferred",
            "prediction_kind": "discovery_support",
            "prediction_error": "llm_circuit_open",
        }, True)
    config = config or MethodConfig()
    payload = _discovery_payload(facts, names, first_locations)
    messages = [
        {"role": "system", "content": DISCOVERY_SYSTEM_PROMPT},
        {"role": "user", "content": json.dumps(payload, separators=(",", ":"))},
    ]
    details, last_error = [], None
    for attempt in range(1, config.prediction_attempts + 1):
        attempt_usage = {}
        started = time.perf_counter()
        try:
            text, attempt_usage = llm_client.generate(messages, agent_id=agent_id)
            weights = _discovery_weights(payload, _response_json(text))
        except Exception as error:
            last_error = error
            detail = dict(attempt_usage or {})
            detail.update(
                attempt=attempt, prediction_status="failed",
                prediction_error=str(error),
                prediction_error_type=type(error).__name__,
                wall_time=float(detail.get(
                    "wall_time", time.perf_counter() - started
                )),
            )
            details.append(detail)
            continue
        detail = dict(attempt_usage or {})
        detail.update(
            attempt=attempt, prediction_status="applied",
            wall_time=float(detail.get("wall_time", time.perf_counter() - started)),
        )
        details.append(detail)
        usage = _usage(details, "applied", False)
        usage["prediction_kind"] = "discovery_support"
        return DiscoveryPredictionResult(weights, usage, False)
    usage = _usage(details, "failed", False, last_error)
    usage["prediction_kind"] = "discovery_support"
    return DiscoveryPredictionResult(None, usage, True)


class SimpleLLMClient:
    """Lazy SDK adapter for official and OpenAI-compatible chat endpoints."""

    def __init__(self, model: str, temperature: float,
                 top_p: float, max_tokens: int, n: int = 1):
        self.source, self.model = "openai", model
        self.temperature, self.top_p = temperature, top_p
        self.max_tokens, self.n = max_tokens, n
        self.client = None
        self._request_options = {}
        self._api_key = os.getenv("OPENAI_API_KEY") or os.getenv("OPENAI_KEY")
        self._api_base = (os.getenv("OPENAI_BASE_URL") or os.getenv("OPENAI_API_BASE")
                          or "https://api.openai.com/v1")
        if not self._api_key:
            raise ValueError("OPENAI_API_KEY or OPENAI_KEY is required")

    def _get_client(self):
        if self.client is None:
            import openai
            if callable(getattr(openai, "OpenAI", None)):
                self.client = openai.OpenAI(api_key=self._api_key, base_url=self._api_base)
            else:
                self.client = openai
                # Keep settings per request so the two agents do not mutate SDK globals.
                self._request_options = {"api_key": self._api_key, "api_base": self._api_base}
        return self.client

    def generate(self, messages: List[Dict[str, str]],
                 agent_id: Optional[int] = None) -> Tuple[str, Dict[str, Any]]:
        started = time.time()
        client = self._get_client()
        params = dict(model=self.model, messages=messages, temperature=self.temperature,
                      top_p=self.top_p, max_tokens=self.max_tokens, n=self.n)
        if hasattr(client, "chat"):
            response = client.chat.completions.create(**params)
        else:
            response = client.ChatCompletion.create(**self._request_options, **params)

        def value(obj, key):
            return obj.get(key) if isinstance(obj, dict) else getattr(obj, key, None)

        usage = value(response, "usage")
        prompt = value(usage, "prompt_tokens")
        completion = value(usage, "completion_tokens")
        total = value(usage, "total_tokens")
        if total is None and prompt is not None and completion is not None:
            total = int(prompt) + int(completion)
        return value(value(value(response, "choices")[0], "message"), "content"), {
            "prompt_tokens": prompt,
            "completion_tokens": completion,
            "total_tokens": total,
            "wall_time": time.time() - started,
            "model": self.model,
            "source": self.source,
        }
