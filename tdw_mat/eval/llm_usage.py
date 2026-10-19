"""Shared LLM telemetry accounting; wall time is diagnostic, not budget."""

import math

INFERENCE_TOKENS_PER_SECOND = 50.0
TOKEN_FIELDS = ("prompt_tokens", "completion_tokens", "total_tokens")
STAT_FIELDS = tuple("llm_" + name for name in TOKEN_FIELDS) + (
    "llm_call_count", "llm_usage_reported_call_count",
    "llm_usage_missing_call_count", "llm_wall_time",
    "llm_inference_time_estimated",
)


def finite_number(value):
    if value is None or isinstance(value, bool):
        return None
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) and value >= 0 else None


def complete_sum(values):
    values = [finite_number(value) for value in values]
    return None if any(value is None for value in values) else sum(values)


def summarize_llm_calls(calls):
    calls = list(calls)
    complete_usage = [
        all(finite_number(call.get(field)) is not None for field in TOKEN_FIELDS)
        for call in calls
    ]
    stats = {
        "llm_call_count": len(calls),
        "llm_usage_reported_call_count": sum(complete_usage),
        "llm_usage_missing_call_count": len(calls) - sum(complete_usage),
        "llm_call_details": calls,
    }
    for field in TOKEN_FIELDS:
        values = []
        for call in calls:
            value = finite_number(call.get(field))
            if field == "total_tokens" and value is None:
                prompt = finite_number(call.get("prompt_tokens"))
                completion = finite_number(call.get("completion_tokens"))
                if prompt is not None and completion is not None:
                    value = prompt + completion
            values.append(0 if value is None else value)
        stats["llm_" + field] = int(sum(values))
    stats["llm_wall_time"] = complete_sum(call.get("wall_time") for call in calls)
    tokens = stats["llm_total_tokens"]
    stats["llm_inference_time_estimated"] = (
        None if tokens is None else tokens / INFERENCE_TOKENS_PER_SECOND
    )
    return stats


def normalize_llm_stats(stats):
    if not isinstance(stats, dict):
        result = dict.fromkeys(STAT_FIELDS, None)
        result["llm_call_details"] = []
        return result
    count = finite_number(stats.get("llm_call_count"))
    details = list(stats.get("llm_call_details") or [])
    if count is not None and len(details) == count:
        return summarize_llm_calls(details)
    if count == 0:
        return summarize_llm_calls([])
    result = {field: finite_number(stats.get(field)) for field in STAT_FIELDS}
    result["llm_call_details"] = details
    tokens = result["llm_total_tokens"]
    result["llm_inference_time_estimated"] = (
        None if tokens is None else tokens / INFERENCE_TOKENS_PER_SECOND
    )
    return result


def combine_llm_stats(stats):
    stats = [normalize_llm_stats(item) for item in stats]
    details = [
        call for item in stats for call in item["llm_call_details"]
    ]
    if all(item["llm_call_count"] == len(item["llm_call_details"]) for item in stats):
        return summarize_llm_calls(details)
    result = {field: complete_sum(item[field] for item in stats) for field in STAT_FIELDS}
    result["llm_call_details"] = details
    return result


def delta_llm_stats(before, after):
    """Return calls added since a snapshot, including ledgers cleared by reset."""
    before = normalize_llm_stats(before)
    after = normalize_llm_stats(after)
    before_details = before["llm_call_details"]
    after_details = after["llm_call_details"]
    if (before["llm_call_count"] == len(before_details)
            and after["llm_call_count"] == len(after_details)):
        start = len(before_details)
        if after_details[:start] != before_details:
            start = 0
        return summarize_llm_calls(after_details[start:])
    return normalize_llm_stats(None)


def step_llm_stats(stats, previous_count):
    """Slice actual call details, avoiding differences of unknown cumulative tokens."""
    count = finite_number((stats or {}).get("llm_call_count"))
    details = (stats or {}).get("llm_call_details")
    if count is not None and isinstance(details, list) and len(details) == count:
        if previous_count is not None and 0 <= previous_count <= count:
            return summarize_llm_calls(details[int(previous_count):])
    return normalize_llm_stats(None)
