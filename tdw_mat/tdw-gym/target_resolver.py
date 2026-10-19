"""Runtime target resolver for the TDW-MAT forced-cooperation benchmark.

The dataset stores cooperation constraints at the object-class level. TDW object
ids are only reliable after a scene reset, so this module expands class-level
fields into concrete object ids at runtime.
"""

from collections import Counter


def _names_for_ids(object_ids, object_names):
    return [object_names[object_id] for object_id in object_ids]


def _select_ids_by_classes(candidate_ids, object_names, target_classes):
    target_classes = set(target_classes or [])
    return [object_id for object_id in candidate_ids if object_names.get(object_id) in target_classes]


def build_goal_description(object_ids, object_names):
    """Count target instances by class/name for the active goal description."""
    return dict(Counter(_names_for_ids(object_ids, object_names)))


def resolve_forced_coop_targets(scene_info, target_object_ids, object_names, excluded_target_ids=None):
    """Resolve forced-coop class fields into runtime object id sets.

    Returns a dictionary that preserves the original target ids and adds active,
    cooperative, and ordinary subsets. If the episode has no forced-coop fields,
    active targets fall back to all original target ids and coop targets are empty.
    """
    scene_info = scene_info or {}
    target_filter_classes = scene_info.get("target_filter_classes")
    coop_constraints = scene_info.get("coop_constraints") or {}
    coop_target_classes = coop_constraints.get("target_classes") or []

    original_target_ids = list(target_object_ids)
    excluded_target_id_set = set(excluded_target_ids or [])
    if target_filter_classes:
        active_target_ids = _select_ids_by_classes(original_target_ids, object_names, target_filter_classes)
    else:
        active_target_ids = list(original_target_ids)

    excluded_active_target_ids = [object_id for object_id in active_target_ids if object_id in excluded_target_id_set]
    active_target_ids = [object_id for object_id in active_target_ids if object_id not in excluded_target_id_set]

    coop_target_ids = _select_ids_by_classes(active_target_ids, object_names, coop_target_classes)
    coop_target_id_set = set(coop_target_ids)
    ordinary_target_ids = [object_id for object_id in active_target_ids if object_id not in coop_target_id_set]

    resolved = {
        "enabled": bool(target_filter_classes or coop_target_classes or scene_info.get("task_variant")),
        "task_variant": scene_info.get("task_variant", "original"),
        "matched_pair_id": scene_info.get("matched_pair_id"),
        "source_episode_id": scene_info.get("source_episode_id"),
        "target_filter_classes": list(target_filter_classes or []),
        "coop_target_classes": list(coop_target_classes),
        "original_target_ids": original_target_ids,
        "excluded_target_ids": excluded_active_target_ids,
        "excluded_goal_description": build_goal_description(excluded_active_target_ids, object_names),
        "active_target_ids": active_target_ids,
        "coop_target_ids": coop_target_ids,
        "ordinary_target_ids": ordinary_target_ids,
        "active_goal_description": build_goal_description(active_target_ids, object_names),
        "coop_goal_description": build_goal_description(coop_target_ids, object_names),
        "ordinary_goal_description": build_goal_description(ordinary_target_ids, object_names),
        "expected_active_target_instances": scene_info.get("expected_active_target_instances"),
        "expected_coop_target_instances": scene_info.get("expected_coop_target_instances"),
    }
    resolved["active_target_count_matches_expected"] = (
        resolved["expected_active_target_instances"] is None
        or len(active_target_ids) == resolved["expected_active_target_instances"]
    )
    resolved["coop_target_count_matches_expected"] = (
        resolved["expected_coop_target_instances"] is None
        or len(coop_target_ids) == resolved["expected_coop_target_instances"]
    )
    return resolved
