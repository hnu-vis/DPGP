"""Positional JSON: factual snapshots/deltas and grounded prefixes."""

from dataclasses import dataclass, fields, is_dataclass, replace
from copy import deepcopy
from collections import Counter
from functools import cached_property
import hashlib
import json
from typing import Union, get_args, get_origin, get_type_hints

from .config import MethodConfig
from .models import Action, Entity, FactGraph, Room, restore_record


def _pack(value):
    if is_dataclass(value):
        return [_pack(getattr(value, f.name)) for f in fields(value)]
    if isinstance(value, tuple):
        return [_pack(v) for v in value]
    return value


def _unpack(annotation, value):
    origin, args = get_origin(annotation), get_args(annotation)
    if origin is Union:
        return None if value is None else _unpack(next(a for a in args if a is not type(None)), value)
    if origin is tuple:
        if not isinstance(value, list):
            raise ValueError("expected wire array")
        if len(args) == 2 and args[1] is Ellipsis:
            return [_unpack(args[0], v) for v in value]
        if len(value) != len(args):
            raise ValueError("wrong tuple size")
        return [_unpack(t, v) for t, v in zip(args, value)]
    if hasattr(annotation, "__dataclass_fields__"):
        names, types = fields(annotation), get_type_hints(annotation)
        if not isinstance(value, list) or len(value) != len(names):
            raise ValueError("invalid wire record for " + annotation.__name__)
        return {f.name: _unpack(types[f.name], v) for f, v in zip(names, value)}
    return value


def _json(value):
    return json.dumps(value, separators=(",", ":"), ensure_ascii=False, allow_nan=False, sort_keys=True)


def _digest(packed):
    return hashlib.sha256(_json(packed).encode("utf-8")).hexdigest()[:24]


@dataclass(frozen=True)
class WireTemplate:
    """Only shared public geometry/goals/vocabulary; never observed door state."""
    rooms: tuple
    goals: tuple
    classes: tuple = ()
    tolerance: float = .6

    def __post_init__(self):
        object.__setattr__(self, "rooms", tuple(replace(r, explored=False,
            center=tuple(float(v) for v in r.center), size=tuple(float(v) for v in r.size))
            for r in sorted(self.rooms, key=lambda r: r.id)))
        object.__setattr__(self, "goals", tuple(sorted(self.goals, key=lambda g: g.key)))
        object.__setattr__(self, "classes", tuple(sorted(set(self.classes) |
            {g.object_class for g in self.goals})))

    @cached_property
    def digest(self):
        return _digest([_pack(self.rooms), _pack(self.goals), list(self.classes), self.tolerance])

    @cached_property
    def topology(self):
        from .memory import initialize_topology
        return {str(p.source) + ":" + str(p.target): _pack(p)[2:]
                for p in initialize_topology(self.rooms, self.tolerance)}

    def class_code(self, name):
        return self.classes.index(name) if name in self.classes else name

    def class_name(self, code):
        if type(code) is int and 0 <= code < len(self.classes):
            return self.classes[code]
        if isinstance(code, str) and code:
            return code
        raise ValueError("invalid class reference")

    def region_code(self, region):
        if not region:
            return ""
        kind, key = region.split(":")
        index = int(key) if kind == "c" else [r.id for r in self.rooms].index(int(key))
        return kind + str(index)

    def region_name(self, code):
        if not code:
            return ""
        if not isinstance(code, str) or code[0] not in "vchu" or not code[1:].isdigit():
            raise ValueError("invalid region code")
        index = int(code[1:])
        if code[0] != "c":
            if index >= len(self.rooms):
                raise ValueError("invalid room index")
            index = self.rooms[index].id
        return code[0] + ":" + str(index)


def _wire_text(message):
    """Lossless packet-local directory for repeated strings/vectors, when smaller."""
    counts, values = Counter(), {}
    def collect(value):
        if isinstance(value, (str, list, dict, float)):
            key = _json(value)
            if len(key) > 12:
                counts[key] += 1
                values[key] = value
        if isinstance(value, list):
            for item in value:
                collect(item)
        elif isinstance(value, dict):
            for item in value.values():
                collect(item)
    collect(message)
    directory = [values[k] for k in sorted(counts) if counts[k] > 1 and
                 (counts[k] - 1) * len(k) > counts[k] * 10 + 1]
    plain = _json(message)
    if not directory:
        return plain
    indices = {_json(v): i for i, v in enumerate(directory)}
    def shrink(value):
        key = _json(value)
        if key in indices:
            return {"@": indices[key]}
        if isinstance(value, list):
            return [shrink(v) for v in value]
        if isinstance(value, dict):
            return {k: shrink(v) for k, v in value.items()}
        return value
    compact = _json(["T", message[1], directory, shrink(message)])
    return compact if len(compact) < len(plain) else plain


def _dump(message, limit):
    text = _wire_text(message)
    if len(text) > limit:
        raise ValueError("message exceeds character limit")
    return text


def _load(text, limit, template=None):
    if not isinstance(text, str) or len(text) > limit:
        raise ValueError("invalid/oversize message")
    def pairs(items):
        result = dict(items)
        if len(result) != len(items):
            raise ValueError("duplicate wire key")
        return result
    value = json.loads(text, object_pairs_hook=pairs,
                       parse_constant=lambda v: (_ for _ in ()).throw(ValueError("nonfinite JSON")))
    if isinstance(value, list) and len(value) == 4 and value[0] == "T":
        _, epoch, directory, body = value
        if not isinstance(directory, list) or len({_json(v) for v in directory}) != len(directory):
            raise ValueError("invalid value directory")
        def expand(v):
            if isinstance(v, dict):
                if "@" in v:
                    i = v["@"]
                    if set(v) != {"@"} or type(i) is not int or not 0 <= i < len(directory):
                        raise ValueError("invalid directory reference")
                    return deepcopy(directory[i])
                return {k: expand(x) for k, x in v.items()}
            return [expand(x) for x in v] if isinstance(v, list) else v
        value = expand(body)
        if not isinstance(value, list) or len(value) != 6 or value[1] != epoch:
            raise ValueError("directory envelope mismatch")
    if (not isinstance(value, list) or len(value) != 6 or value[0] not in ("F", "D", "A", "S")
            or type(value[1]) is not int or value[1] < 0 or type(value[3]) is not int
            or value[3] not in (0, 1) or not isinstance(value[5], list) or len(value[5]) > 2):
        raise ValueError("invalid protocol header")
    if template is not None and value[2] != template.digest:
        raise ValueError("public template mismatch")
    return value


class BaselineMismatch(ValueError):
    """Repair through an explicitly marked existing graph transaction."""


def _diff(before, after):
    """One keyed-table / field-mask delta, shared by facts, grounding and belief.

    Identical row patches (including recency-only updates) share one key group.
    List order and scalar types are exact; tables use stable identity keys.
    """
    full = [0, after]
    if isinstance(before, dict) and isinstance(after, dict):
        groups = {}
        for key in sorted(after):
            if key in before and _json(before[key]) == _json(after[key]):
                continue
            patch = _diff(before[key], after[key]) if key in before else [0, after[key]]
            groups.setdefault(_json(patch), [patch, []])[1].append(key)
        delta = [2, sorted(set(before) - set(after)), list(groups.values())]
    elif isinstance(before, list) and isinstance(after, list) and len(before) == len(after):
        changed = [i for i in range(len(after)) if _json(before[i]) != _json(after[i])]
        delta = [1, sum(1 << i for i in changed)] + [_diff(before[i], after[i]) for i in changed]
    else:
        return full
    return delta if len(_json(delta)) < len(_json(full)) else full


def _patched(base, patch):
    if not isinstance(patch, list) or not patch or type(patch[0]) is not int:
        raise ValueError("invalid patch")
    if patch[0] == 0 and len(patch) == 2:
        return deepcopy(patch[1])
    result = deepcopy(base)
    if patch[0] == 1 and len(patch) >= 2 and isinstance(base, list):
        mask = patch[1]
        if type(mask) is not int or not 0 <= mask < (1 << len(base)):
            raise ValueError("invalid field mask")
        indices = [i for i in range(len(base)) if mask & (1 << i)]
        if len(indices) != len(patch) - 2:
            raise ValueError("field mask length mismatch")
        for i, p in zip(indices, patch[2:]):
            result[i] = _patched(base[i], p)
    elif patch[0] == 2 and len(patch) == 3 and isinstance(base, dict):
        removed, groups = patch[1:]
        if not isinstance(removed, list) or not isinstance(groups, list):
            raise ValueError("invalid table update")
        seen = set()
        for key in removed:
            if not isinstance(key, str) or key not in result or key in seen:
                raise ValueError("invalid table deletion")
            seen.add(key)
            del result[key]
        for group in groups:
            if not isinstance(group, list) or len(group) != 2 or not isinstance(group[1], list):
                raise ValueError("invalid row group")
            p, keys = group
            for key in keys:
                if not isinstance(key, str) or key in seen:
                    raise ValueError("duplicate table change")
                seen.add(key)
                result[key] = _patched(base.get(key), p)
    else:
        raise ValueError("patch type/shape mismatch")
    return result


def _update(value, base=None):
    digest = _digest(value)
    full = ["f", digest, value]
    if base is None:
        return full
    previous = _digest(base)
    if previous == digest:
        return ["r", digest]
    delta = ["d", previous, digest, _diff(base, value)]
    return delta if len(_json(delta)) < len(_json(full)) else full


def _restore_update(value, base=None):
    if not isinstance(value, list) or not value:
        raise ValueError("invalid update")
    if value[0] == "f" and len(value) == 3:
        digest, result = value[1:]
    elif value[0] in ("r", "d") and len(value) == (2 if value[0] == "r" else 4):
        if base is None or _digest(base) != value[1]:
            raise BaselineMismatch("missing or mismatched update baseline")
        digest, result = (value[1], deepcopy(base)) if value[0] == "r" else (value[2], _patched(base, value[3]))
    else:
        raise ValueError("invalid update fields")
    if _digest(result) != digest:
        raise ValueError("update digest mismatch")
    return result


def _entity_pack(entity, template):
    row = _pack(entity)
    if row[3] is not None:
        row[3] = [float(v) for v in row[3]]
    # Preserve all facts and exact coordinates; only bool/state representation changes.
    flags = int(row[7]) | int(row[8]) << 1 | int(row[10]) << 2
    return [template.class_code(row[1]), *row[2:7], flags,
            (None, "open", "closed").index(row[9]), row[11], row[12]]


def _entity_unpack(key, row, template):
    if not isinstance(row, list) or len(row) != 10 or type(row[6]) is not int or not 0 <= row[6] < 8:
        raise ValueError("invalid entity flags/row")
    if type(row[7]) is not int or row[7] not in (0, 1, 2):
        raise ValueError("invalid open state")
    flags = row[6]
    return [int(key), template.class_name(row[0]), *row[1:6], bool(flags & 1), bool(flags & 2),
            (None, "open", "closed")[row[7]], bool(flags & 4), row[8], row[9]]


def _graph_pack(graph, template):
    if type(graph) is FactGraph:
        graph = FactGraph.from_dict(graph.to_dict())
    if tuple(replace(r, explored=False) for r in graph.rooms) != template.rooms or graph.goals != template.goals:
        raise ValueError("graph differs from public template")
    # Portals are a delta against public geometry, never against private reset observations.
    topology = template.topology
    portals = {str(p.source) + ":" + str(p.target): _pack(p)[2:] for p in graph.portals}
    return [graph.step, graph.source, sum(int(r.explored) << i for i, r in enumerate(graph.rooms)),
            _diff(topology, portals), {str(a.index): _pack(a)[1:] for a in graph.agents},
            {str(e.id): _entity_pack(e, template) for e in graph.entities},
            {str(template.classes.index(k)): v for k, v in graph.class_coop}, _pack(graph.cooperation),
            {template.region_code(e.region): [_pack(e)[1], _pack(e)[2], [template.class_code(c) for c in e.semantics],
                        e.complete, _pack(e.stamp)] for e in graph.inspections},
            {str(e.object_id): [template.region_code(e.region), _pack(e.stamp)] for e in graph.origins},
            {str(e.index): _pack(e)[1:] for e in graph.navigation}]


def _graph_unpack(value, template):
    if not isinstance(value, list) or len(value) != 11:
        raise ValueError("invalid graph record")
    step, source, bits, portals, agents, entities, classes, coop, checks, origins, navigation = value
    if type(bits) is not int or not 0 <= bits < 1 << len(template.rooms):
        raise ValueError("invalid explored mask")
    topology = template.topology
    topology = _patched(topology, portals)
    rows = [step, source, [_pack(replace(r, explored=bool(bits & 1 << i))) for i, r in enumerate(template.rooms)],
            [list(map(int, k.split(":"))) + topology[k] for k in sorted(topology, key=lambda k: tuple(map(int, k.split(":"))))],
            [[int(k)] + agents[k] for k in sorted(agents, key=int)],
            [_entity_unpack(k, entities[k], template) for k in sorted(entities, key=int)], _pack(template.goals),
            sorted([[template.class_name(int(k)), v] for k, v in classes.items()]), coop,
            sorted([[template.region_name(k), *v[:2], [template.class_name(c) for c in v[2]], *v[3:]] for k, v in checks.items()]),
            [[int(k), template.region_name(origins[k][0]), origins[k][1]] for k in sorted(origins, key=int)],
            [[int(k)] + navigation[k] for k in sorted(navigation, key=int)]]
    return FactGraph.from_dict(_unpack(FactGraph, rows))


def _packet(kind, epoch, template, payload, recovery=False):
    if type(epoch) is not int or epoch < 0 or type(recovery) is not bool:
        raise ValueError("invalid round/recovery")
    return [kind, epoch, template.digest, int(recovery), payload, []]


def _graph_message(graph, round_id, template, base=None, recovery=False):
    payload = _update(_graph_pack(graph, template), _graph_pack(base, template) if base is not None else None)
    return _packet("F" if payload[0] == "f" else "D", round_id, template, payload, recovery)


def encode_graph(graph, *, round_id, template, base=None, recovery=False, config=None):
    if type(graph) is not FactGraph or (base is not None and type(base) is not FactGraph):
        raise TypeError("wire accepts facts only")
    graph.validate()
    if base is not None:
        base.validate()
        if base.source != graph.source or base.step > graph.step:
            raise ValueError("delta baseline source/clock mismatch")
    return _dump(_graph_message(graph, round_id, template, base, recovery), (config or MethodConfig()).message_limit)


def decode_graph(text, *, template, base=None, expected_source=None, config=None):
    if base is not None and type(base) is not FactGraph:
        raise TypeError("graph baseline requires facts")
    value = _load(text, (config or MethodConfig()).message_limit, template)
    if value[0] not in ("F", "D"):
        raise ValueError("not a factual graph message")
    packed = _restore_update(value[4], _graph_pack(base, template) if base is not None else None)
    graph = _graph_unpack(packed, template)
    if expected_source is not None and graph.source != expected_source:
        raise ValueError("unexpected graph sender")
    if base is not None and (base.source != graph.source or base.step > graph.step):
        raise ValueError("graph source/clock regressed")
    return value[1], graph


@dataclass(frozen=True)
class ActionPacket:
    round_id: int
    recipient: int
    actions: tuple
    entities: tuple
    rooms: tuple


def _action_pack(action, template, entities):
    code = {"Move": "M", "Grab": "G", "Put": "P", "CheckContainer": "C"}[action.kind]
    if code == "P":
        name = entities[action.object_id].name
        goal = next(i for i, g in enumerate(template.goals) if g.target_id == action.target_id and g.object_class == name)
        row = [code, action.object_id, goal]
    else:
        row = [code, action.target_id]
    return row + [1] if action.cooperative else row


def _action_unpack(row, template):
    if not isinstance(row, list) or not row or row[0] not in ("M", "G", "P", "C"):
        raise ValueError("invalid action opcode")
    size = 3 if row[0] == "P" else 2
    if len(row) not in (size, size + 1) or (len(row) > size and (type(row[-1]) is not int or row[-1] != 1)):
        raise ValueError("invalid action size/coop")
    coop = len(row) > size
    if row[0] == "P":
        if type(row[2]) is not int or not 0 <= row[2] < len(template.goals):
            raise ValueError("invalid goal reference")
        return Action("Put", template.goals[row[2]].target_id, row[1], coop)
    return Action({"M": "Move", "G": "Grab", "C": "CheckContainer"}[row[0]], row[1], cooperative=coop)


def _action_message(actions, facts, round_id, recipient, template, base=None):
    entities, rooms = {e.id: e for e in facts.entities}, {r.id: r for r in facts.rooms}
    keys = {k for a in actions for k in (a.target_id, a.object_id) if k is not None}
    entity_ids, room_ids = keys & entities.keys(), keys & rooms.keys()
    pending = list(entity_ids)
    while pending:
        node = entities[pending.pop()]
        for parent in (node.inside,) + node.on:
            if parent in entities and parent not in entity_ids:
                entity_ids.add(parent)
                pending.append(parent)
    rows = {str(i): _entity_pack(entities[i], template) for i in sorted(entity_ids)}
    old = {str(e.id): _entity_pack(e, template) for e in base.entities if e.id in entity_ids} if base is not None else None
    room_rows = [[i, int(rooms[r.id].explored)] for i, r in enumerate(template.rooms) if r.id in room_ids]
    payload = [recipient, [_action_pack(a, template, entities) for a in actions],
               sorted(entity_ids), _update(rows, old), room_rows]
    return _packet("A", round_id, template, payload)


def encode_actions(actions, facts, *, round_id, recipient, template, base=None, config=None):
    config = config or MethodConfig()
    if type(facts) is not FactGraph or (base is not None and type(base) is not FactGraph):
        raise TypeError("action grounding requires facts")
    facts.validate()
    if type(recipient) is not int or recipient not in (0, 1):
        raise ValueError("invalid recipient")
    actions = tuple(actions)
    for action in actions:
        action.validate()
    actions = tuple(a for a in actions if a.kind not in ("Wait", "Continue"))[:config.action_horizon]
    if not actions:
        raise ValueError("empty executable prefix must not be sent")
    entities, rooms = {e.id: e for e in facts.entities}, {r.id for r in facts.rooms}
    for action in actions:
        for key in (action.target_id, action.object_id):
            if key is None or (action.kind == "Move" and key in rooms):
                continue
            if key not in entities or entities[key].stamp.step < 0 or entities[key].room_id is None:
                raise ValueError("action depends on unobserved/unknown entity")
    return _dump(_action_message(actions, facts, round_id, recipient, template, base), config.message_limit)


def decode_actions(text, *, template, expected_recipient=None, base=None, config=None):
    if base is not None and type(base) is not FactGraph:
        raise TypeError("grounding baseline requires facts")
    config = config or MethodConfig()
    value = _load(text, config.message_limit, template)
    if value[0] != "A" or not isinstance(value[4], list) or len(value[4]) != 5:
        raise ValueError("not an action message")
    recipient, raw_actions, ids, update, raw_rooms = value[4]
    if type(recipient) is not int or recipient not in (0, 1) or (expected_recipient is not None and recipient != expected_recipient):
        raise ValueError("unexpected recipient")
    if not isinstance(raw_actions, list) or not 1 <= len(raw_actions) <= config.action_horizon:
        raise ValueError("invalid action horizon")
    actions = tuple(_action_unpack(a, template) for a in raw_actions)
    if not isinstance(ids, list) or any(type(i) is not int or i <= 0 for i in ids) or sorted(set(ids)) != ids:
        raise ValueError("invalid grounding identities")
    old = {str(e.id): _entity_pack(e, template) for e in base.entities if e.id in ids} if base is not None else None
    rows = _restore_update(update, old)
    if sorted(rows, key=int) != [str(i) for i in ids]:
        raise ValueError("grounding identity mismatch")
    entities = tuple(restore_record(Entity, _unpack(Entity, _entity_unpack(str(i), rows[str(i)], template))) for i in ids)
    rooms = []
    for index, explored in raw_rooms:
        if type(index) is not int or not 0 <= index < len(template.rooms) or type(explored) is not int or explored not in (0, 1):
            raise ValueError("invalid room reference")
        rooms.append(replace(template.rooms[index], explored=bool(explored)))
    room_ids = {r.id for r in rooms}
    if len(room_ids) != len(rooms) or room_ids & set(ids) or any(e.stamp.step < 0 or e.room_id is None for e in entities):
        raise ValueError("invalid grounding")
    for action in actions:
        action.validate()
        if action.target_id not in ids and not (action.kind == "Move" and action.target_id in room_ids):
            raise ValueError("missing target grounding")
        if action.object_id is not None and action.object_id not in ids:
            raise ValueError("missing object grounding")
    return ActionPacket(value[1], recipient, actions, entities, tuple(rooms))


def encode_status(status, *, round_id, recipient, template, config=None):
    if status not in ("idle", "complete") or type(recipient) is not int or recipient not in (0, 1):
        raise ValueError("invalid plan status")
    return _dump(_packet("S", round_id, template, [recipient, int(status == "complete")]),
                 (config or MethodConfig()).message_limit)


def decode_status(text, *, recipient, template, config=None):
    value = _load(text, (config or MethodConfig()).message_limit, template)
    if value[0] != "S" or value[4] not in ([recipient, 0], [recipient, 1]):
        raise ValueError("invalid plan status")
    return value[1], ("idle", "complete")[value[4][1]]


def message_kind(text, config=None):
    return _load(text, (config or MethodConfig()).message_limit)[0]


def _seed_pack(seed, template):
    def weights(row):
        return {template.region_code(k): v for k, v in sorted(row.items())}
    return [seed["task"], seed["revision"],
            {str(template.classes.index(k)): [r["pmf"], weights(r["weights"])] for k, r in sorted(seed["roots"].items())},
            {k: weights(v) for k, v in sorted(seed["instances"].items())},
            {str(i): 0 for i in seed["known"]}, {template.region_code(k): 0 for k in seed["processed"]}, seed["repairs"]]


def _seed_unpack(row, template):
    if not isinstance(row, list) or len(row) != 7:
        raise ValueError("invalid belief record")
    task, revision, roots, instances, known, processed, repairs = row
    def weights(values):
        return {template.region_name(k): v for k, v in values.items()}
    result = dict(task=task, revision=revision,
                roots={template.class_name(int(k)): dict(pmf=r[0], weights=weights(r[1])) for k, r in roots.items()},
                instances={k: weights(v) for k, v in instances.items()}, known=sorted(map(int, known)),
                processed=sorted(template.region_name(k) for k in processed), repairs=repairs)
    if _json(_seed_pack(result, template)) != _json(row):
        raise ValueError("noncanonical belief identities/set values")
    return result


def _extra(message, index, value):
    while len(message[5]) <= index:
        message[5].append(None)
    message[5][index] = value
    return message


def attach_belief(text, seed, *, model_epoch, template, base=None, config=None):
    from .discovery_belief import DiscoveryBelief
    config = config or MethodConfig()
    value = _load(text, config.message_limit, template)
    DiscoveryBelief.from_seed(seed)
    if type(model_epoch) is not int or not -1 <= model_epoch <= value[1]:
        raise ValueError("invalid probability epoch")
    update = _update(_seed_pack(seed, template), _seed_pack(base, template) if base is not None else None)
    return _dump(_extra(value, 0, [model_epoch, update]), config.message_limit)


def split_belief(text, config=None, *, template, base=None):
    from .discovery_belief import DiscoveryBelief
    config = config or MethodConfig()
    value = _load(text, config.message_limit, template)
    if not value[5] or value[5][0] is None:
        return text, None, -1
    epoch, update = value[5][0]
    if type(epoch) is not int or not -1 <= epoch <= value[1]:
        raise ValueError("invalid probability epoch")
    seed = _seed_unpack(_restore_update(update, _seed_pack(base, template) if base is not None else None), template)
    DiscoveryBelief.from_seed(seed)
    value[5][0] = None
    while value[5] and value[5][-1] is None:
        value[5].pop()
    return _dump(value, config.message_limit), seed, epoch


def attach_snapshot(text, graph, *, template, base=None, config=None):
    """Reply piggyback establishes a real graph baseline, including idle replies."""
    if type(graph) is not FactGraph:
        raise TypeError("snapshot requires facts")
    graph.validate()
    config = config or MethodConfig()
    value = _load(text, config.message_limit, template)
    if value[0] not in ("A", "S"):
        raise ValueError("snapshot requires a reply")
    return _dump(_extra(value, 1, _update(_graph_pack(graph, template),
        _graph_pack(base, template) if base is not None else None)), config.message_limit)


def split_snapshot(text, *, template, base=None, expected_source=None, config=None):
    config = config or MethodConfig()
    value = _load(text, config.message_limit, template)
    if len(value[5]) < 2 or value[5][1] is None:
        return None
    if value[0] not in ("A", "S"):
        raise ValueError("snapshot requires a reply")
    graph = _graph_unpack(_restore_update(value[5][1], _graph_pack(base, template) if base is not None else None), template)
    if expected_source is not None and graph.source != expected_source:
        raise ValueError("unexpected snapshot sender")
    return graph


def recovery_requested(text, config=None):
    return bool(_load(text, (config or MethodConfig()).message_limit)[3])


def estimate_exchange_chars(graph, seed, *, base_graph, base_seed, actions, recipient, template):
    """Same codec, but synthetic rollout identities remain size-only and never sent."""
    request = _graph_message(graph, graph.step, template, base_graph)
    request = _extra(request, 0, [graph.step, _update(_seed_pack(base_seed, template), _seed_pack(base_seed, template))])
    reply = (_action_message(actions, graph, graph.step, recipient, template, base_graph) if actions else
             _packet("S", graph.step, template, [recipient, 0]))
    reply = _extra(reply, 0, [graph.step, _update(_seed_pack(seed, template), _seed_pack(base_seed, template))])
    return len(_wire_text(request)) + len(_wire_text(reply))
