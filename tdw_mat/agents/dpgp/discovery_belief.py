"""Room-level discovery belief for the optional probabilistic method path."""

from __future__ import annotations

import math
from typing import Dict, Optional

from .config import MethodConfig
from .models import FactGraph
from .prediction import predict_discovery_weights




class DiscoveryBeliefError(RuntimeError):
    """Raised when a belief update cannot be applied without guessing."""

    def __init__(self, message: str, usage: Optional[dict] = None):
        super().__init__(message)
        self.usage = dict(usage or {})


def _normalized(values: Dict[int, float]) -> Dict[int, float]:
    if not values or any(not math.isfinite(value) or value < 0
                         for value in values.values()):
        raise ValueError("room weights must be finite and non-negative")
    total = sum(values.values())
    if total <= 0:
        raise ValueError("room weights require positive support")
    return {room_id: value / total for room_id, value in sorted(values.items())}


def _task_signature(facts: FactGraph) -> dict:
    return {
        "rooms": [item.node_id for item in facts.rooms],
        "required_objects": [[name, count]
                             for name, count in facts.goal.required_objects],
    }


def _unresolved(facts: FactGraph):
    anonymous = {}
    instances = {}
    for item in facts.objects:
        if (item.finished or item.holder_agent_ids
                or item.room_node_id is not None):
            continue
        if item.object_id is None:
            anonymous[item.class_name] = anonymous.get(item.class_name, 0) + 1
        else:
            instances[item.object_id] = item.class_name
    return anonymous, instances


class DiscoveryBelief:
    """Positive-only room belief; incomplete empty scans never exclude rooms."""

    def __init__(self):
        self.task: Optional[dict] = None
        self.revision = 0
        self.anonymous_counts: Dict[str, int] = {}
        self.class_weights: Dict[str, Dict[int, float]] = {}
        self.instance_weights: Dict[int, Dict[int, float]] = {}
        # Each identity is consumed once. None means its original room was not
        # certified (e.g. first learned while held); later transport is no prior.
        self.first_locations: Dict[int, Optional[int]] = {}

    @property
    def diagnostics(self) -> dict:
        return {
            "revision": self.revision,
            "anonymous_target_count": sum(self.anonymous_counts.values()),
            "class_count": len(self.class_weights),
            "instance_count": len(self.instance_weights),
        }

    def to_seed(self) -> dict:
        task = None if self.task is None else {
            "rooms": list(self.task["rooms"]),
            "required_objects": [list(item)
                                 for item in self.task["required_objects"]],
        }
        return {
            "task": task,
            "revision": self.revision,
            "anonymous_counts": dict(sorted(self.anonymous_counts.items())),
            "class_weights": {
                name: {str(room_id): value for room_id, value in weights.items()}
                for name, weights in sorted(self.class_weights.items())
            },
            "instance_weights": {
                str(object_id): {
                    str(room_id): value for room_id, value in weights.items()
                }
                for object_id, weights in sorted(self.instance_weights.items())
            },
            "first_locations": {str(key): room for key, room
                                in sorted(self.first_locations.items())},
        }

    @classmethod
    def from_seed(cls, seed: dict) -> "DiscoveryBelief":
        expected = {
            "task", "revision", "anonymous_counts", "class_weights",
            "instance_weights", "first_locations",
        }
        if not isinstance(seed, dict) or set(seed) != expected:
            raise ValueError("invalid discovery belief seed fields")
        task = seed["task"]
        if task is not None:
            if (not isinstance(task, dict)
                    or set(task) != {"rooms", "required_objects"}
                    or not isinstance(task["rooms"], list)
                    or any(type(value) is not int for value in task["rooms"])
                    or task["rooms"] != sorted(set(task["rooms"]))
                    or not isinstance(task["required_objects"], list)):
                raise ValueError("invalid discovery belief task")
            required = task["required_objects"]
            if (any(not isinstance(item, list) or len(item) != 2
                    or not isinstance(item[0], str) or not item[0]
                    or type(item[1]) is not int or item[1] <= 0
                    for item in required)
                    or [item[0] for item in required]
                    != sorted(set(item[0] for item in required))):
                raise ValueError("invalid discovery belief task")
        if type(seed["revision"]) is not int or seed["revision"] < 0:
            raise ValueError("belief revision must be a non-negative integer")
        if (not isinstance(seed["class_weights"], dict)
                or not isinstance(seed["anonymous_counts"], dict)
                or not isinstance(seed["instance_weights"], dict)):
            raise ValueError("invalid discovery belief collections")
        if any(not isinstance(name, str) or not name
               or type(count) is not int or count <= 0
               for name, count in seed["anonymous_counts"].items()):
            raise ValueError("invalid anonymous target counts")
        if task is not None:
            required_counts = dict(task["required_objects"])
            if any(name not in required_counts or count > required_counts[name]
                   for name, count in seed["anonymous_counts"].items()):
                raise ValueError("anonymous target counts exceed task goal")

        result = cls()
        result.task = None if task is None else {
            "rooms": list(task["rooms"]),
            "required_objects": [list(item) for item in task["required_objects"]],
        }
        result.revision = seed["revision"]
        result.anonymous_counts = dict(seed["anonymous_counts"])
        locations = seed["first_locations"]
        if (not isinstance(locations, dict)
                or any(not isinstance(key, str) or not key.isdigit()
                       or str(int(key)) != key
                       or (room is not None and
                           (type(room) is not int or task is None
                            or room not in task["rooms"]))
                       for key, room in locations.items())):
            raise ValueError("invalid first-discovery locations")
        result.first_locations = {int(key): room for key, room in locations.items()}
        for name, weights in seed["class_weights"].items():
            if not isinstance(name, str) or not name or not isinstance(weights, dict):
                raise ValueError("invalid class belief")
            parsed = {int(room_id): float(value)
                      for room_id, value in weights.items()}
            if task is not None and set(parsed) != set(task["rooms"]):
                raise ValueError("class belief rooms do not match task")
            normalized = _normalized(parsed)
            if any(not math.isclose(parsed[key], normalized[key], abs_tol=1e-9)
                   for key in parsed):
                raise ValueError("class belief must be normalized")
            result.class_weights[name] = parsed
        for raw_id, weights in seed["instance_weights"].items():
            if not isinstance(weights, dict):
                raise ValueError("invalid instance belief")
            object_id = int(raw_id)
            parsed = {int(room_id): float(value)
                      for room_id, value in weights.items()}
            if task is not None and set(parsed) != set(task["rooms"]):
                raise ValueError("instance belief rooms do not match task")
            normalized = _normalized(parsed)
            if any(not math.isclose(parsed[key], normalized[key], abs_tol=1e-9)
                   for key in parsed):
                raise ValueError("instance belief must be normalized")
            result.instance_weights[object_id] = parsed
        return result

    def synchronize(self, facts: FactGraph, llm_client, agent_id: int,
                    config: Optional[MethodConfig] = None,
                    llm_enabled: bool = True) -> dict:
        """Align priors with facts; empty TDW scans are non-conclusive."""
        facts.validate()
        config = config or MethodConfig()
        task = _task_signature(facts)
        if self.task is not None and self.task != task:
            raise DiscoveryBeliefError("belief task does not match fact graph")

        work = DiscoveryBelief.from_seed(self.to_seed())
        before = work.to_seed()
        work.task = task
        discovered_classes = set()
        for item in facts.objects:
            if item.object_id is None or item.object_id in work.first_locations:
                continue
            room = (item.room_node_id if item.information_source == "direct_observation"
                    and not item.finished and not item.holder_agent_ids else None)
            work.first_locations[item.object_id] = room
            if room is not None:
                discovered_classes.add(item.class_name)
        anonymous, instances = _unresolved(facts)
        work.anonymous_counts = dict(sorted(anonymous.items()))
        active_classes = set(anonymous) | set(instances.values())
        work.class_weights = {
            name: weights for name, weights in work.class_weights.items()
            if name in active_classes
        }
        work.instance_weights = {
            object_id: weights
            for object_id, weights in work.instance_weights.items()
            if object_id in instances
        }
        refresh = sorted((active_classes - set(work.class_weights))
                         | (active_classes & discovered_classes))
        if refresh:
            prediction = predict_discovery_weights(
                facts, refresh, llm_client, agent_id, config, llm_enabled,
                first_locations=work.first_locations,
            )
            if prediction.failed or prediction.weights is None:
                raise DiscoveryBeliefError(
                    prediction.usage.get("prediction_error")
                    or "discovery support prediction failed",
                    prediction.usage,
                )
            work.class_weights.update(prediction.weights)
            usage = prediction.usage
        else:
            usage = {
                "llm_called": False, "prediction_status": "not_needed",
                "prediction_kind": "discovery_support",
            }
        for object_id, class_name in instances.items():
            if object_id not in work.instance_weights or class_name in refresh:
                work.instance_weights[object_id] = dict(work.class_weights[class_name])

        room_ids = {item.node_id for item in facts.rooms}
        for weights in tuple(work.class_weights.values()) + tuple(
                work.instance_weights.values()):
            if set(weights) != room_ids:
                raise DiscoveryBeliefError("belief rooms do not match fact graph")

        after = work.to_seed()
        if after != before:
            work.revision += 1
        self.task = work.task
        self.revision = work.revision
        self.anonymous_counts = work.anonymous_counts
        self.class_weights = work.class_weights
        self.instance_weights = work.instance_weights
        self.first_locations = work.first_locations
        usage["anonymous_counts"] = dict(sorted(anonymous.items()))
        usage["known_unlocated_instance_count"] = len(instances)
        usage["completed_scan_count"] = sum(
            item.scan_completed for item in facts.inspections
        )
        usage["negative_scan_update"] = "none_non_conclusive"
        usage["belief_revision"] = self.revision
        usage["refreshed_classes"] = refresh
        return usage
