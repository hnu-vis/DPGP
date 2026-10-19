"""Evaluation metrics for TDW forced-cooperation benchmark runs.

The metrics are method-independent: they read environment state and per-step
trace records, while method-specific code can optionally pass LLM usage stats.
"""

from __future__ import annotations

import copy
import json
import math
from collections import Counter
from typing import Any, Dict, Iterable, List, Optional
if __package__:
    from .llm_usage import combine_llm_stats, normalize_llm_stats
    from .evaluation_records import validate_idle_step_timing
else:
    from llm_usage import combine_llm_stats, normalize_llm_stats
    from evaluation_records import validate_idle_step_timing

IDLE_TIME_METRICS = ('communication_time_step_sum', 'communication_extra_time',
                     'nonidle_execution_time', 'idle_execution_time', 'idle_step_count')
TASK_GROUPS = ("all", "ordinary", "coop")
OUTCOME_GROUPS = ("all", "success", "failure")
LLM_TOKEN_METRICS = ("llm_prompt_tokens", "llm_completion_tokens", "llm_total_tokens")


SUCCESS_RATE_KEYS = [
    "task_success_rate",
    "ordinary_task_success_rate",
    "coop_task_success_rate",
    "subgoal_success_rate",
    "ordinary_subgoal_success_rate",
    "coop_subgoal_success_rate",
]

NUMERIC_METRIC_PATHS = {
    **{key: (("execution_time", key),) for key in IDLE_TIME_METRICS},
    "num_steps": (("num_steps",),),
    "physical_action_frames": (("execution_time", "physical_action_frames"),),
    "physical_execution_time": (("execution_time", "physical_execution_time"),),
    "embodied_execution_time": (("execution_time", "embodied_execution_time"),),
    "message_send_time_agent_sum": (("execution_time", "message_send_time_agent_sum"),),
    "communication_total_time_agent_sum": (("execution_time", "communication_total_time_agent_sum"),),
    "nonphysical_extra_time": (("execution_time", "nonphysical_extra_time"),),
    "communication_critical_path_overhead": (
        ("execution_time", "communication_critical_path_overhead"),
    ),
    "max_embodied_execution_time": (("max_embodied_execution_time",),),
    "tgc_auc": (("tgc_auc",),),
    "communication_chars": (("execution_time", "communication_chars"),),
    "message_count": (("execution_time", "message_count"),),
    "message_truncated_count": (("execution_time", "message_truncated_count"),),
    "invalid_action_count": (("failure_summary", "invalid_action_count"),),
    "physical_execution_failure_count": (
        ("failure_summary", "physical_execution_failure_count"),
    ),
    "action_failure_count": (("failure_summary", "action_failure_count"),),
    "coop_action_event_count": (("coop_summary", "coop_action_event_count"),),
    "coop_feedback_event_count": (("coop_summary", "coop_feedback_event_count"),),
    "llm_call_count": (("execution_time", "llm_call_count"),),
    "llm_usage_reported_call_count": (("execution_time", "llm_usage_reported_call_count"),),
    "llm_usage_missing_call_count": (("execution_time", "llm_usage_missing_call_count"),),
    "llm_prompt_tokens": (("execution_time", "llm_prompt_tokens"),),
    "llm_completion_tokens": (("execution_time", "llm_completion_tokens"),),
    "llm_total_tokens": (("execution_time", "llm_total_tokens"),),
    "llm_inference_time_estimated": (("execution_time", "llm_inference_time_estimated"),),
    "llm_wall_time": (("execution_time", "llm_wall_time"),),
}


def metric_rate(success_count: int, total_count: int) -> Optional[float]:
    return float(success_count) / float(total_count) if total_count > 0 else None


def metric_entry(success_count: int, total_count: int) -> Dict[str, Any]:
    return {
        "success_count": int(success_count),
        "total_count": int(total_count),
        "rate": metric_rate(int(success_count), int(total_count)),
    }


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [json_safe(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    if hasattr(value, "tolist"):
        try:
            return json_safe(value.tolist())
        except Exception:
            pass
    return str(value)


def unwrap_env(env: Any) -> Any:
    return getattr(env, "unwrapped", env)


def _forced_info(env: Any, info: Optional[dict]) -> Dict[str, Any]:
    raw_env = unwrap_env(env)
    forced = getattr(raw_env, "forced_coop_info", None)
    if isinstance(forced, dict):
        return forced
    if isinstance(info, dict):
        forced = info.get("forced_coop")
        if isinstance(forced, dict):
            return forced
    return {}


def _int_list(values: Iterable[Any]) -> List[int]:
    output = []
    for value in values or []:
        try:
            output.append(int(value))
        except Exception:
            continue
    return output


def _active_target_ids(env: Any, info: Optional[dict]) -> List[int]:
    raw_env = unwrap_env(env)
    forced = _forced_info(raw_env, info)
    ids = _int_list(forced.get("active_target_ids", []))
    if ids:
        return ids
    return _int_list(getattr(raw_env, "active_target_object_ids", []) or getattr(raw_env, "target_object_ids", []))


def _coop_target_ids(env: Any, info: Optional[dict]) -> List[int]:
    raw_env = unwrap_env(env)
    forced = _forced_info(raw_env, info)
    ids = _int_list(forced.get("coop_target_ids", []))
    if ids:
        return ids
    return _int_list(getattr(raw_env, "coop_target_object_ids", []) or [])


def _ordinary_target_ids(env: Any, info: Optional[dict], active_ids: List[int], coop_ids: List[int]) -> List[int]:
    raw_env = unwrap_env(env)
    forced = _forced_info(raw_env, info)
    ids = _int_list(forced.get("ordinary_target_ids", []))
    if ids:
        return ids
    ordinary = getattr(raw_env, "ordinary_target_object_ids", None)
    if ordinary is not None:
        return _int_list(ordinary)
    coop_set = set(coop_ids)
    return [object_id for object_id in active_ids if object_id not in coop_set]


def _normalize_llm_stats(llm_stats: Optional[dict]) -> Dict[str, Any]:
    return normalize_llm_stats(llm_stats)


def _summarize_trace(trace: Optional[List[dict]]) -> Dict[str, Any]:
    if not isinstance(trace, list) or not trace:
        raise ValueError("metrics require a nonempty step trace")
    summary = dict.fromkeys((
        "embodied_execution_time", "message_send_time_agent_sum",
        "communication_critical_path_overhead", "communication_chars",
        "message_count", "message_truncated_count", "invalid_action_count",
        "physical_execution_failure_count", "action_failure_count",
        "coop_action_event_count", "coop_feedback_event_count", *IDLE_TIME_METRICS,
    ), 0)
    action_status_counts = Counter()
    invalid_reason_counts = Counter()
    physical_failure_reason_counts = Counter()
    action_failure_reason_counts = Counter()
    coop_action_event_counts = Counter()
    coop_feedback_event_counts = Counter()
    for record in trace:
        execution = record["execution_time_step"]
        validate_idle_step_timing(execution, record["physical_action_frame_info"])
        summary["embodied_execution_time"] += execution["step_embodied_execution_time"]
        for key in ("message_send_time_agent_sum", "communication_critical_path_overhead",
                    "communication_chars", *IDLE_TIME_METRICS):
            summary[key] += execution[key]
        messages = execution["messages"]
        summary["message_count"] += len(messages)
        summary["message_truncated_count"] += sum(bool(m.get("truncated")) for m in messages)
        for field, count_key, reasons in (
            ("invalid_action_events", "invalid_action_count", invalid_reason_counts),
            ("physical_execution_failure_events", "physical_execution_failure_count", physical_failure_reason_counts),
        ):
            events = record[field]
            if not isinstance(events, list):
                raise ValueError("missing typed action failure evidence: " + field)
            for event in events:
                reason = str(event["reason"])
                summary[count_key] += 1
                summary["action_failure_count"] += 1
                reasons[reason] += 1
                action_failure_reason_counts[reason] += 1
        for status in record.get("action_status") or []:
            if status is not None:
                action_status_counts[str(status)] += 1
        for event in record.get("coop_action_events") or []:
            if isinstance(event, dict):
                summary["coop_action_event_count"] += 1
                coop_action_event_counts[str(event.get("type", "unknown"))] += 1
        for event in record.get("coop_feedback_events") or []:
            if isinstance(event, dict):
                summary["coop_feedback_event_count"] += 1
                coop_feedback_event_counts[str(event.get("type", "unknown"))] += 1

    summary["action_status_counts"] = dict(sorted(action_status_counts.items()))
    summary["invalid_reason_counts"] = dict(sorted(invalid_reason_counts.items()))
    summary["physical_execution_failure_reason_counts"] = dict(
        sorted(physical_failure_reason_counts.items())
    )
    summary["action_failure_reason_counts"] = dict(
        sorted(action_failure_reason_counts.items())
    )
    summary["coop_action_event_counts"] = dict(sorted(coop_action_event_counts.items()))
    summary["coop_feedback_event_counts"] = dict(sorted(coop_feedback_event_counts.items()))
    for key in ("communication_count_unit", "communication_send_chars_per_second"):
        values = {row["execution_time_step"][key] for row in trace}
        if len(values) != 1:
            raise ValueError("mixed communication settings within one episode: " + key)
        summary[key] = values.pop()
    return summary


def _completion_fraction(finish: Any) -> float:
    if not isinstance(finish, (list, tuple)) or len(finish) < 2:
        return 0.0
    completed = int(finish[0] or 0)
    total = int(finish[1] or 0)
    if total <= 0:
        return 1.0
    return min(1.0, max(0.0, float(completed) / float(total)))


def summarize_tgc_auc(
    trace: Optional[List[dict]],
    initial_finish: Any,
    time_budget: Optional[float],
) -> Dict[str, Any]:
    """Integrate latched TDW target completion over embodied execution time."""
    if time_budget is None or not math.isfinite(float(time_budget)) or float(time_budget) <= 0.0 or trace is None:
        return {
            "tgc_auc": None,
            "time_budget": (float(time_budget) if time_budget is not None
                            and math.isfinite(float(time_budget)) and float(time_budget) > 0 else None),
            "completion_fraction_at_budget": None,
            "final_completion_fraction": _completion_fraction((trace or [{}])[-1].get("finish", initial_finish)),
        }
    budget = float(time_budget)
    current_fraction = _completion_fraction(initial_finish)
    elapsed = 0.0
    area = 0.0
    records = list(trace or [])
    consumed_records = 0
    for index, record in enumerate(records):
        execution = record.get("execution_time_step") or {}
        step_time = float(execution["step_embodied_execution_time"])
        step_end = elapsed + max(0.0, step_time)
        interval_end = min(budget, step_end)
        if interval_end > elapsed:
            area += current_fraction * (interval_end - elapsed)
        elapsed = interval_end
        consumed_records = index + 1
        if step_end <= budget:
            current_fraction = _completion_fraction(record.get("finish"))
        if elapsed >= budget:
            break

    if elapsed < budget:
        if consumed_records < len(records):
            current_fraction = _completion_fraction(
                records[consumed_records].get("finish")
            )
        area += current_fraction * (budget - elapsed)

    final_finish = records[-1].get("finish") if records else initial_finish
    return {
        "tgc_auc": min(1.0, max(0.0, area / budget)),
        "time_budget": budget,
        "completion_fraction_at_budget": current_fraction,
        "final_completion_fraction": _completion_fraction(final_finish),
    }


def make_episode_metadata(env, episode, info=None, effective_config=None):
    """Evaluation-only facts; never feed these hidden target labels to agents."""
    raw_env = unwrap_env(env)
    active_ids = _active_target_ids(raw_env, info)
    coop_ids = _coop_target_ids(raw_env, info)
    ordinary_ids = _ordinary_target_ids(raw_env, info, active_ids, coop_ids)
    return json_safe({
        "episode": int(episode),
        "target_ids": {
            "active": active_ids, "ordinary": ordinary_ids, "coop": coop_ids,
            "excluded": _int_list(_forced_info(raw_env, info).get("excluded_target_ids", [])),
        },
        "initial_finish": getattr(raw_env, "initial_goal_progress", (0, len(active_ids), False)),
        "initial_completed_ids": sorted(int(value) for value in (getattr(raw_env, "satisfied", {}) or {})),
        "initial_frames": int(getattr(raw_env, "num_frames", 0) or 0),
        "physical_action_frame_rate": getattr(raw_env, "physical_action_frame_rate", None),
        "max_embodied_execution_time": (info or {}).get("max_embodied_execution_time", getattr(raw_env, "max_embodied_execution_time", None)),
        "max_steps": getattr(raw_env, "max_steps", None),
        "max_frames": getattr(raw_env, "max_frame", None),
        "effective_config": effective_config,
        "communication_count_unit": getattr(raw_env, "communication_count_unit", None),
        "communication_send_chars_per_second": getattr(raw_env, "communication_send_chars_per_second", None),

    })


def build_episode_metrics(
    env: Any,
    episode: int,
    info: Optional[dict],
    trace: Optional[List[dict]],
    step_count: Optional[int] = None,
    llm_stats: Optional[dict] = None,
) -> Dict[str, Any]:
    """Build one episode's metrics from final env state and per-step trace.

    Subgoal denominators use active_target_ids, which are resolved after
    container-inside targets have been filtered out at reset time.
    """
    raw_env = unwrap_env(env)
    finish = raw_env.check_goal()
    metadata = make_episode_metadata(raw_env, episode, info)
    # The live adapter may be called after execution; AUC uses reset progress.
    return build_record_metrics({
        "metadata": metadata, "trace": trace,
        "final": {
            "finish": finish,
            "task_success_within_time": (info or {}).get("task_success_within_time", finish[2]),
            "termination_reason": (info or {}).get("termination_reason"),
            "completed_ids": sorted(int(value) for value in (getattr(raw_env, "satisfied", {}) or {})),
            "frames": int(getattr(raw_env, "num_frames", 0) or 0),
            "num_steps": step_count if step_count is not None else len(trace or []),
        },
        "llm_stats": llm_stats,
    })


def build_record_metrics(record):
    """Shared online/offline calculation from recorded facts, without TDW imports."""
    metadata, final = record["metadata"], record["final"]
    trace = record.get("trace")
    episode = metadata["episode"]
    finish_count, finish_total, raw_task_success = final["finish"]
    task_success = bool(
        final.get("task_success_within_time", raw_task_success)
    )
    active_ids = metadata["target_ids"]["active"]
    coop_ids = metadata["target_ids"]["coop"]
    ordinary_ids = metadata["target_ids"]["ordinary"]
    completed_ids = set(_int_list(final["completed_ids"]))
    active_set = set(active_ids)
    coop_set = set(coop_ids) & active_set
    ordinary_set = set(ordinary_ids) & active_set

    all_success = len(active_set & completed_ids)
    ordinary_success = len(ordinary_set & completed_ids)
    coop_success = len(coop_set & completed_ids)
    has_joint_task = len(coop_set) > 0

    trace_summary = _summarize_trace(trace)
    if "llm_stats" in record:
        llm_summary = _normalize_llm_stats(record.get("llm_stats"))
    else:
        llm_parts = []
        if trace:
            llm_parts.extend((trace[0].get("initial_llm_stats") or {}).values())
        llm_parts.extend(
            (row.get("llm_stats") or {}).get(agent_id)
            for row in trace or [] for agent_id in ("0", "1")
        )
        llm_summary = combine_llm_stats(llm_parts)
    physical_frames = final.get("frames")
    frame_rate = metadata.get("physical_action_frame_rate")
    physical_time = (float(physical_frames) / frame_rate
                     if physical_frames is not None and frame_rate and frame_rate > 0 else None)
    total_execution_time = trace_summary["embodied_execution_time"]
    time_budget = metadata.get("max_embodied_execution_time")
    initial_finish = metadata.get("initial_finish")
    tgc_summary = summarize_tgc_auc(
        trace if total_execution_time is not None and initial_finish is not None else None,
        initial_finish, time_budget,
    )

    success_rates = {
        "task_success_rate": metric_entry(int(bool(task_success)), 1),
        "ordinary_task_success_rate": metric_entry(int(bool(task_success)) if not has_joint_task else 0, 0 if has_joint_task else 1),
        "coop_task_success_rate": metric_entry(int(bool(task_success)) if has_joint_task else 0, 1 if has_joint_task else 0),
        "subgoal_success_rate": metric_entry(all_success, len(active_set)),
        "ordinary_subgoal_success_rate": metric_entry(ordinary_success, len(ordinary_set)),
        "coop_subgoal_success_rate": metric_entry(coop_success, len(coop_set)),
    }

    return json_safe({
        "episode": episode,
        "task_success": bool(task_success),
        "task_success_within_time": bool(task_success),
        "raw_goal_success": bool(raw_task_success),
        "termination_reason": final.get("termination_reason"),
        "task_type": "coop" if has_joint_task else "ordinary",
        "has_joint_task": has_joint_task,
        "num_steps": int(final.get("num_steps", len(trace or []))),
        "finish": [int(finish_count), int(finish_total), bool(task_success)],
        "target_counts": {
            "active": len(active_set),
            "ordinary": len(ordinary_set),
            "coop": len(coop_set),
            "completed": all_success,
            "completed_ordinary": ordinary_success,
            "completed_coop": coop_success,
            "excluded": len(metadata["target_ids"].get("excluded", [])),
        },
        "target_ids": {
            "active": sorted(active_set),
            "ordinary": sorted(ordinary_set),
            "coop": sorted(coop_set),
            "completed": sorted(active_set & completed_ids),
            "excluded": metadata["target_ids"].get("excluded", []),
        },
        "success_rates": success_rates,
        "embodied_execution_time": total_execution_time,
        "max_embodied_execution_time": tgc_summary["time_budget"],
        "tgc_auc": tgc_summary["tgc_auc"],
        "time_normalized_goal_completion": tgc_summary,
        "execution_time": {
            **{key: trace_summary[key] for key in IDLE_TIME_METRICS},
            "physical_action_frames": physical_frames,
            "physical_action_frame_rate": frame_rate,
            "physical_execution_time": physical_time,
            "embodied_execution_time": total_execution_time,
            "message_send_time_agent_sum": trace_summary["message_send_time_agent_sum"],
            "communication_total_time_agent_sum": trace_summary["message_send_time_agent_sum"],
            "nonphysical_extra_time": total_execution_time - physical_time,
            "nonidle_execution_time": total_execution_time - trace_summary["idle_execution_time"],
            "communication_extra_time": max(
                0.0, total_execution_time - physical_time - trace_summary["idle_execution_time"]
            ),
            "communication_critical_path_overhead": trace_summary["communication_critical_path_overhead"],
            "max_embodied_execution_time": tgc_summary["time_budget"],
            "communication_chars": trace_summary["communication_chars"],
            "communication_count_unit": trace_summary["communication_count_unit"],
            "communication_send_chars_per_second": trace_summary["communication_send_chars_per_second"],
            "message_count": trace_summary["message_count"],
            "message_truncated_count": trace_summary["message_truncated_count"],
            **llm_summary,
        },
        "failure_summary": {
            "invalid_action_count": trace_summary["invalid_action_count"],
            "invalid_reason_counts": trace_summary["invalid_reason_counts"],
            "physical_execution_failure_count": trace_summary[
                "physical_execution_failure_count"
            ],
            "physical_execution_failure_reason_counts": trace_summary[
                "physical_execution_failure_reason_counts"
            ],
            "action_failure_count": trace_summary["action_failure_count"],
            "action_failure_reason_counts": trace_summary[
                "action_failure_reason_counts"
            ],
            "action_status_counts": trace_summary["action_status_counts"],
        },
        "coop_summary": {
            "coop_action_event_count": trace_summary["coop_action_event_count"],
            "coop_feedback_event_count": trace_summary["coop_feedback_event_count"],
            "coop_action_event_counts": trace_summary["coop_action_event_counts"],
            "coop_feedback_event_counts": trace_summary["coop_feedback_event_counts"],
        },
    })


_MISSING = object()


def _lookup_numeric_metric(metrics: dict, candidate_paths: Iterable[tuple]):
    for path in candidate_paths:
        current = metrics
        for key in path:
            if not isinstance(current, dict) or key not in current:
                current = _MISSING
                break
            current = current[key]
        if current is _MISSING or current is None or isinstance(current, bool):
            continue
        try:
            value = float(current)
        except (TypeError, ValueError):
            continue
        if math.isfinite(value):
            return value
    return None


def _numeric_summary(values: Iterable[float]) -> Dict[str, Any]:
    values = [float(value) for value in values]
    if not values:
        return {
            "count": 0,
            "sum": 0.0,
            "mean": None,
            "population_std": None,
            "min": None,
            "max": None,
        }
    mean = sum(values) / float(len(values))
    variance = sum((value - mean) ** 2 for value in values) / float(len(values))
    return {
        "count": len(values),
        "sum": sum(values),
        "mean": mean,
        "population_std": math.sqrt(variance),
        "min": min(values),
        "max": max(values),
    }


def _merge_count_map(counter: Counter, values: Optional[dict]) -> None:
    for key, value in (values or {}).items():
        try:
            counter[str(key)] += int(value)
        except (TypeError, ValueError):
            continue


def _summarize_core_group(records):
    rates = {
        key: metric_entry(
            sum(int((item.get("success_rates", {}).get(key) or {}).get("success_count") or 0) for item in records),
            sum(int((item.get("success_rates", {}).get(key) or {}).get("total_count") or 0) for item in records),
        ) for key in SUCCESS_RATE_KEYS
    }
    summaries = {
        key: _numeric_summary(
            (0 if value is None and key in LLM_TOKEN_METRICS else value)
            for item in records
            for value in [_lookup_numeric_metric(item, paths)]
            if value is not None or key in LLM_TOKEN_METRICS
        ) for key, paths in NUMERIC_METRIC_PATHS.items()
    }
    return {
        "episode_count": len(records), "success_rates": rates,
        "numeric_summaries": summaries,
        "averages": {"avg_" + key: value["mean"] for key, value in summaries.items()},
        "totals": {"total_" + key: value["sum"] for key, value in summaries.items()},
    }


def _build_nine_groups(records):
    return {
        task_type: {
            outcome: _summarize_core_group([
                record for record in records
                if (task_type == "all" or record.get("task_type") == task_type)
                and (outcome == "all" or (
                    record.get("task_success_within_time", record.get("task_success")) is not None
                    and bool(record.get("task_success_within_time", record.get("task_success")))
                    == (outcome == "success")
                ))
            ]) for outcome in OUTCOME_GROUPS
        } for task_type in TASK_GROUPS
    }


def aggregate_experiment_metrics(metrics_list: List[dict]) -> Dict[str, Any]:
    records = list(metrics_list or [])
    valid_records = [
        item for item in records
        if isinstance(item, dict) and not item.get("error") and isinstance(item.get("success_rates"), dict)
    ]
    identities = {tuple((item.get('execution_time') or {}).get(key) for key in
                        ('communication_count_unit', 'communication_send_chars_per_second'))
                  for item in valid_records}
    if len(identities) > 1:
        raise ValueError('mixed communication settings')
    groups = _build_nine_groups(valid_records)
    core = groups["all"]["all"]

    task_type_counts = Counter()
    termination_reason_counts = Counter()
    invalid_reason_counts = Counter()
    physical_failure_reason_counts = Counter()
    action_failure_reason_counts = Counter()
    action_status_counts = Counter()
    coop_action_event_counts = Counter()
    coop_feedback_event_counts = Counter()
    for metrics in valid_records:
        task_type_counts[str(metrics.get("task_type") or "unknown")] += 1
        termination_reason_counts[
            str(metrics.get("termination_reason") or "unknown")
        ] += 1
        failure = metrics.get("failure_summary") or {}
        coop = metrics.get("coop_summary") or {}
        _merge_count_map(invalid_reason_counts, failure.get("invalid_reason_counts"))
        _merge_count_map(
            physical_failure_reason_counts,
            failure.get("physical_execution_failure_reason_counts"),
        )
        _merge_count_map(
            action_failure_reason_counts,
            failure.get("action_failure_reason_counts"),
        )
        _merge_count_map(action_status_counts, failure.get("action_status_counts"))
        _merge_count_map(coop_action_event_counts, coop.get("coop_action_event_counts"))
        _merge_count_map(coop_feedback_event_counts, coop.get("coop_feedback_event_counts"))

    return json_safe({
        "communication_model": dict(zip(
            ('count_unit', 'send_chars_per_second'),
            next(iter(identities), (None, None)))),
        "groups": groups,
        "metric_notes": {
            "sample_unit": "episode-run",
            "communication_total": "end-to-end send agent-sum workload, not joint elapsed time",
            "nonphysical_extra": "embodied - physical, including minimum control-cycle overhead",
            "communication_step_sum": "sum of per-step maximum sender durations",
            "communication_extra": "embodied - physical - joint idle time; excludes communication overlapped by physical execution",
            "nonidle_execution": "embodied - joint idle time; includes physical execution",
            "llm_inference_time_estimated": "total tokens / 50 tokens per second; excluded from episode budget",
        },
        "episode_record_count": len(records),
        "valid_episode_count": len(valid_records),
        "invalid_metric_record_count": len(records) - len(valid_records),
        "task_type_counts": dict(sorted(task_type_counts.items())),
        "success_rates": core["success_rates"],
        "averages": core["averages"],
        "totals": core["totals"],
        "numeric_summaries": core["numeric_summaries"],
        "termination_reason_counts": dict(sorted(termination_reason_counts.items())),
        "event_counts": {
            "invalid_reason_counts": dict(sorted(invalid_reason_counts.items())),
            "physical_execution_failure_reason_counts": dict(
                sorted(physical_failure_reason_counts.items())
            ),
            "action_failure_reason_counts": dict(
                sorted(action_failure_reason_counts.items())
            ),
            "action_status_counts": dict(sorted(action_status_counts.items())),
            "coop_action_event_counts": dict(sorted(coop_action_event_counts.items())),
            "coop_feedback_event_counts": dict(sorted(coop_feedback_event_counts.items())),
        },
    })


def _format_rate(entry: dict) -> str:
    success = int(entry.get("success_count", 0) or 0)
    total = int(entry.get("total_count", 0) or 0)
    if total <= 0:
        return "N/A (0 / 0)"
    rate = float(entry.get("rate", 0.0) or 0.0) * 100.0
    return f"{success}/{total} ({rate:.1f}%)"


def _format_counts(values: Optional[dict]) -> str:
    values = values or {}
    if not values:
        return "none"
    return ", ".join(
        f"{key}={values[key]}" for key in sorted(values)
    )


def _emit_metric_lines(lines: List[str]) -> None:
    for line in lines:
        print(line)


def _format_value(value, digits=3):
    return "N/A" if value is None else f"{value:.{digits}f}"


def print_episode_metrics(metrics, logger=None, prefix="Episode metrics"):
    execution = metrics.get("execution_time") or {}
    rates, targets = metrics.get("success_rates") or {}, metrics.get("target_counts") or {}
    status = "SUCCESS" if metrics.get("task_success_within_time", metrics.get("task_success")) else "FAILED"
    value = lambda key: _format_value(_lookup_numeric_metric(metrics, NUMERIC_METRIC_PATHS[key]))
    _emit_metric_lines([
        f"=== {prefix} | episode {metrics.get('episode')} | {metrics.get('task_type')} | {status} ===",
        "  Communication is charged once end-to-end, by delivered Unicode character count.",
        f"  Progress : total {targets.get('completed', 0)}/{targets.get('active', 0)} | "
        f"ordinary {targets.get('completed_ordinary', 0)}/{targets.get('ordinary', 0)} | "
        f"coop {targets.get('completed_coop', 0)}/{targets.get('coop', 0)}",
        "  SR       : task " + _format_rate(rates.get("task_success_rate", {}))
        + " | subgoals " + _format_rate(rates.get("subgoal_success_rate", {}))
        + " | ordinary " + _format_rate(rates.get("ordinary_subgoal_success_rate", {}))
        + " | coop " + _format_rate(rates.get("coop_subgoal_success_rate", {})),
        f"  Efficiency: embodied {value('embodied_execution_time')} s | physical {value('physical_execution_time')} s | "
        f"TGC-AUC {value('tgc_auc')} | steps {metrics.get('num_steps')}",
        f"  Comm     : total(agent-sum) {value('communication_total_time_agent_sum')} s | "
        f"step-sum {value('communication_time_step_sum')} s | extra {value('communication_extra_time')} s | "
        f"messages {value('message_count')}",
        f"  LLM Token: input {value('llm_prompt_tokens')} | output {value('llm_completion_tokens')} | total {value('llm_total_tokens')}",
        f"  Auxiliary: calls {value('llm_call_count')} | estimated {value('llm_inference_time_estimated')} s | "
        f"wall {value('llm_wall_time')} s | idle steps {value('idle_step_count')} | idle time {value('idle_execution_time')} s | "
        f"communication chars {value('communication_chars')} | "
        f"invalid {value('invalid_action_count')} | physical failures {value('physical_execution_failure_count')} | "
        f"termination {metrics.get('termination_reason')}",
    ])


def _print_table(title, headers, rows):
    print(title)
    widths = [max(len(str(row[index])) for row in [headers] + rows) for index in range(len(headers))]
    for row in [headers] + rows:
        print("  " + " | ".join(str(cell).ljust(width) for cell, width in zip(row, widths)))


def print_experiment_metrics(metrics, logger=None, prefix="Experiment metrics"):
    groups = metrics.get("groups") or {}
    rows = [(task, outcome, (groups.get(task) or {}).get(outcome) or (
        metrics if task == outcome == "all" else {}
    )) for task in TASK_GROUPS for outcome in OUTCOME_GROUPS]
    print(f"=== {prefix} | {metrics.get('episode_record_count', 0)} records ===")
    print(f"  Comm model: {metrics.get('communication_model', {})}")
    print(f"  Coverage : valid {metrics.get('valid_episode_count', 'N/A')} | "
          f"invalid {metrics.get('invalid_metric_record_count', 'N/A')} | task types [{_format_counts(metrics.get('task_type_counts'))}]")
    sr_keys = ("task_success_rate", "subgoal_success_rate", "ordinary_subgoal_success_rate", "coop_subgoal_success_rate")
    _print_table("Success rates", ["Task", "Outcome", "N", "Task SR", "Subgoal SR", "Ordinary subgoal SR", "Coop subgoal SR"], [
        [task, outcome, group.get("episode_count", 0)] +
        [_format_rate((group.get("success_rates") or {}).get(key, {})) for key in sr_keys]
        for task, outcome, group in rows
    ])
    for title, keys in (
        ("Execution efficiency (episode-run mean)", ("embodied_execution_time", "physical_execution_time", "tgc_auc", "num_steps")),
        ("Communication (episode-run mean; time in seconds)", ("communication_total_time_agent_sum", "communication_time_step_sum", "communication_extra_time", "message_count")),
    ):
        _print_table(title, ["Task", "Outcome", "N"] + list(keys), [
            [task, outcome, group.get("episode_count", 0)] +
            [_format_value((group.get("averages") or {}).get("avg_" + key)) for key in keys]
            for task, outcome, group in rows
        ])
    keys = ("llm_prompt_tokens", "llm_completion_tokens", "llm_total_tokens")
    _print_table("LLM Token (sum / episode-run mean)", ["Task", "Outcome", "N", "Input", "Output", "Total"], [
        [task, outcome, group.get("episode_count", 0)] + [
            "N/A" if not ((group.get("numeric_summaries") or {}).get(key) or {}).get("count") else
            _format_value(group["numeric_summaries"][key]["sum"], 0) + " / " +
            _format_value(group["numeric_summaries"][key]["mean"])
            for key in keys
        ] for task, outcome, group in rows
    ])
    print("  Note: communication total is agent-sum workload; "
          "step-sum takes the maximum send time per step; extra=execution-physical-idle.")
    for key in ("llm_call_count", "llm_usage_reported_call_count", "llm_usage_missing_call_count",
                "llm_wall_time", "llm_inference_time_estimated", "physical_action_frames",
                "communication_chars", "message_count", "message_truncated_count", "communication_critical_path_overhead",
                "idle_step_count", "idle_execution_time", "nonidle_execution_time", "nonphysical_extra_time",
                "invalid_action_count", "physical_execution_failure_count", "action_failure_count",
                "coop_action_event_count", "coop_feedback_event_count"):
        summary = (metrics.get("numeric_summaries") or {}).get(key) or {}
        print(f"  Auxiliary {key}: n={summary.get('count', 0)} mean={_format_value(summary.get('mean'))}")
    print("  Termination: " + _format_counts(metrics.get("termination_reason_counts")))
    for key, counts in (metrics.get("event_counts") or {}).items():
        print(f"  {key}: {_format_counts(counts)}")
