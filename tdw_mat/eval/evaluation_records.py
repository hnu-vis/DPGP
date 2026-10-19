"""Load and validate TDW raw JSON evidence without importing the simulator."""

import json
import math
from pathlib import Path
import sys

TDW_GYM_DIR = Path(__file__).resolve().parents[1] / "tdw-gym"
if str(TDW_GYM_DIR) not in sys.path:
    sys.path.append(str(TDW_GYM_DIR))

from episode_limits import optional_positive_int, positive_finite, resolve_episode_termination



def _ids(value, field):
    if not isinstance(value, list) or any(isinstance(item, bool) or not isinstance(item, int) for item in value):
        raise ValueError("invalid target ID evidence: " + field)
    if len(set(value)) != len(value):
        raise ValueError("duplicate target IDs: " + field)
    return set(value)


def validate_idle_step_timing(execution, physical):
    """Check the timing model against actual TDW activity/frame evidence."""
    frame_rate = positive_finite(physical.get('frame_rate'))
    minimum_frames = optional_positive_int(execution.get('minimum_step_frames'))
    if minimum_frames is None:
        raise ValueError('missing minimum control cycle')
    minimum = minimum_frames / frame_rate
    count = physical.get('step_physical_action_frames')
    if isinstance(count, bool) or not isinstance(count, int) or count < 0:
        raise ValueError('invalid physical frame evidence')
    agents = physical.get('per_agent') or {}
    active = False
    for agent_id in (0, 1):
        agent = agents.get(str(agent_id), agents.get(agent_id))
        if not isinstance(agent, dict) or type(agent.get('physical_action_active')) is not bool:
            raise ValueError('missing native physical activity evidence')
        active |= agent['physical_action_active']
        frames = count if agent['physical_action_active'] else 0
        if agent.get('frames') != frames or not math.isclose(agent.get('time', -1), frames / frame_rate, abs_tol=1e-9):
            raise ValueError('inconsistent per-agent physical activity/frames')
        timing = (execution.get('per_agent') or {}).get(str(agent_id), (execution.get('per_agent') or {}).get(agent_id))
        if not isinstance(timing, dict) or not math.isclose(timing.get('physical_action_time', -1), frames / frame_rate, abs_tol=1e-9):
            raise ValueError('inconsistent per-agent physical execution time')
    has_physical = active or count > 0
    messages = execution.get('messages')
    if not isinstance(messages, list):
        raise ValueError('missing delivered communication evidence')
    idle = not has_physical and not messages
    if execution.get('has_physical_action') is not has_physical or execution.get('is_idle_step') is not idle:
        raise ValueError('inconsistent physical/idle classification')
    if type(execution.get('idle_step_count')) is not int or execution['idle_step_count'] != int(idle):
        raise ValueError('inconsistent joint idle step count')
    if type(execution.get('message_count')) is not int or execution['message_count'] != len(messages):
        raise ValueError('inconsistent delivered message count')
    speed = positive_finite(execution.get('communication_send_chars_per_second'))
    sending = [len(message['content']) / speed for message in messages]
    communication = max(sending, default=0.0)
    physical_time = count / frame_rate
    idle_time = minimum if idle else 0.0
    total = max(physical_time, communication, idle_time)
    expected = {
        'minimum_step_time': minimum,
        'physical_execution_time': physical_time,
        'idle_execution_time': idle_time,
        'communication_time_step_sum': communication,
        'step_embodied_execution_time': total,
        'total_execution_time': total,
        'communication_extra_time': max(0.0, total - physical_time - idle_time),
        'communication_critical_path_overhead': max(0.0, total - physical_time - idle_time),
        'nonidle_execution_time': total - idle_time,
    }
    for key, value in expected.items():
        recorded = execution.get(key)
        if not isinstance(recorded, (int, float)) or isinstance(recorded, bool) or not math.isfinite(recorded) or recorded < 0 or not math.isclose(recorded, value, abs_tol=1e-9):
            raise ValueError('inconsistent idle-only timing evidence: ' + key)


def validate_record(record):
    metadata, trace = record["metadata"], record["trace"]
    required = {"episode", "target_ids", "initial_finish", "initial_completed_ids", "initial_frames",
                "physical_action_frame_rate", "max_embodied_execution_time", "max_steps", "max_frames",
                "effective_config", "communication_count_unit", "communication_send_chars_per_second"}
    if not required <= set(metadata):
        raise ValueError("invalid raw metadata fields")
    budget = positive_finite(metadata.get("max_embodied_execution_time"))
    for key in ("max_steps", "max_frames"):
        if key not in metadata:
            raise ValueError("missing collection limit evidence: " + key)
        optional_positive_int(metadata[key])
    frame_rate = positive_finite(metadata.get("physical_action_frame_rate"))
    benchmark = (metadata.get("effective_config") or {}).get("benchmark")
    if not isinstance(benchmark, dict) or any(
        key not in benchmark or benchmark[key] != metadata[key]
        for key in ("max_embodied_execution_time", "max_steps", "max_frames", "physical_action_frame_rate")
    ):
        raise ValueError("raw collection settings differ from effective configuration")
    optional_positive_int(benchmark.get("minimum_step_frames"))
    for key in ('communication_count_unit', 'communication_send_chars_per_second'):
        if metadata[key] != benchmark.get(key):
            raise ValueError('inconsistent communication configuration: ' + key)
    if metadata['communication_count_unit'] != 'unicode_codepoint':
        raise ValueError('unsupported communication unit')
    positive_finite(metadata['communication_send_chars_per_second'])
    targets = metadata["target_ids"]
    active, ordinary, coop = (_ids(targets.get(key), key) for key in ("active", "ordinary", "coop"))
    if not active or ordinary & coop or ordinary | coop != active:
        raise ValueError("invalid active/ordinary/coop partition")
    initial = _ids(metadata.get("initial_completed_ids"), "initial_completed_ids")
    initial_progress = metadata.get("initial_finish")
    if not isinstance(initial_progress, list) or initial_progress[:2] != [len(initial & active), len(active)]:
        raise ValueError("missing or inconsistent initial progress")
    if not isinstance(trace, list) or not trace:
        raise ValueError("raw evaluation requires a nonempty step trace")
    elapsed, frames = 0.0, metadata.get("initial_frames")
    if not isinstance(frames, int) or frames != 0:
        raise ValueError("physical time requires explicit zero task-start frames")
    previous_completed = initial
    for index, row in enumerate(trace, 1):
        if row.get("runner_step") != index or row.get("env_step") != index or row.get("env_step_index") != index - 1:
            raise ValueError("misaligned step evidence at {}".format(index))
        execution = row.get("execution_time_step")
        physical = row.get("physical_action_frame_info")
        if not isinstance(execution, dict) or not isinstance(physical, dict):
            raise ValueError("missing physical/execution time evidence")
        fields = ('message_send_time_agent_sum', 'physical_execution_time', 'communication_critical_path_overhead')
        fields += ('communication_chars',)
        for field in fields:
            value = execution.get(field)
            if value is None or isinstance(value, bool) or not math.isfinite(float(value)) or float(value) < 0:
                raise ValueError("missing or invalid execution evidence: " + field)
        if not isinstance(execution.get("messages"), list):
            raise ValueError("missing explicit per-step communication evidence")
        for key in ('communication_count_unit', 'communication_send_chars_per_second'):
            if execution.get(key) != metadata[key]:
                raise ValueError('step communication settings differ from metadata: ' + key)
        messages = execution['messages']
        senders = set()
        for message in messages:
            sender = message.get('sender_agent')
            content = message.get('content')
            if sender not in (0, 1) or sender in senders or not isinstance(content, str) or not content:
                raise ValueError('invalid delivered message evidence')
            senders.add(sender)
            if message.get('message_chars') != len(content):
                raise ValueError('inconsistent message character count')
        chars = execution.get('communication_chars')
        if isinstance(chars, bool) or not isinstance(chars, int) or chars != sum(len(m['content']) for m in messages):
            raise ValueError('inconsistent communication character total')
        send_time = chars / metadata['communication_send_chars_per_second']
        if not math.isclose(execution['message_send_time_agent_sum'], send_time, abs_tol=1e-9):
            raise ValueError('inconsistent end-to-end communication workload')
        agents = execution.get('per_agent') or {}
        totals = []
        for agent_id in (0, 1):
            agent = agents.get(str(agent_id), agents.get(agent_id))
            if not isinstance(agent, dict):
                raise ValueError('missing per-agent time evidence')
            content_chars = sum(len(m['content']) for m in messages if m['sender_agent'] == agent_id)
            expected_send = content_chars / metadata['communication_send_chars_per_second']
            if agent.get('send_chars') != content_chars or not math.isclose(agent.get('message_send_time', -1), expected_send, abs_tol=1e-9):
                raise ValueError('inconsistent per-agent communication evidence')
            physical_time = agent.get('physical_action_time')
            if physical_time is None or not math.isfinite(physical_time) or physical_time < 0:
                raise ValueError('invalid per-agent physical time')
            total = max(physical_time, expected_send)
            if not math.isclose(agent.get('total_execution_time', -1), total, abs_tol=1e-9):
                raise ValueError('inconsistent per-agent execution time')
            totals.append(total)
        if execution.get('minimum_step_frames') != benchmark.get('minimum_step_frames'):
            raise ValueError('step control cycle differs from collection configuration')
        validate_idle_step_timing(execution, physical)
        duration = execution['step_embodied_execution_time']
        elapsed += duration
        count = physical.get("step_physical_action_frames")
        if isinstance(count, bool) or not isinstance(count, int) or count < 0:
            raise ValueError("invalid per-step physical frames")
        frames += count
        if physical.get("frame_rate") != frame_rate or not math.isclose(
                physical.get("step_physical_action_time", -1), count / frame_rate, abs_tol=1e-9):
            raise ValueError("inconsistent physical frame/time evidence")
        if not math.isclose(execution["physical_execution_time"], count / frame_rate, abs_tol=1e-9):
            raise ValueError("inconsistent execution physical time")
        if row.get("frame") != frames or not math.isclose(
                row.get("elapsed_embodied_execution_time", -1), elapsed, rel_tol=1e-9, abs_tol=1e-9):
            raise ValueError("inconsistent cumulative time/frame evidence")
        completed = _ids(row.get("satisfied_target_ids"), "satisfied_target_ids")
        if not previous_completed <= completed:
            raise ValueError("TDW latched completion cannot regress")
        progress = row.get("finish")
        if not isinstance(progress, list) or len(progress) != 3 or progress[:2] != [len(completed & active), len(active)] or progress[2] != (progress[0] == progress[1]):
            raise ValueError("inconsistent per-step goal progress")
        previous_completed = completed
        for field in ("actions", "llm_stats"):
            if not isinstance(row.get(field), dict) or set(row[field]) != {"0", "1"}:
                raise ValueError("missing both-agent step evidence: " + field)
        initial_llm_stats = row.get("initial_llm_stats")
        if index == 1:
            if not isinstance(initial_llm_stats, dict) or set(initial_llm_stats) != {"0", "1"}:
                raise ValueError("missing both-agent reset LLM evidence")
        elif initial_llm_stats is not None:
            raise ValueError("reset LLM evidence must appear only on the first step")
        llm_stats = list(row["llm_stats"].values())
        if index == 1:
            llm_stats.extend(initial_llm_stats.values())
        for stats in llm_stats:
            calls = stats.get("llm_call_count") if isinstance(stats, dict) else None
            if isinstance(calls, bool) or not isinstance(calls, (int, float)) or not math.isfinite(calls) or calls < 0 or int(calls) != calls:
                raise ValueError("missing explicit per-step LLM call evidence")
        done, success, reason = resolve_episode_termination(
            progress[0] == progress[1], elapsed, budget, index, metadata["max_steps"], frames, metadata["max_frames"],
        )
        if row.get("task_success_within_time") is not success:
            raise ValueError("inconsistent budget-aware task result")
        if row.get("termination_reason") != reason:
            raise ValueError("inconsistent termination evidence")
        if done and index != len(trace):
            raise ValueError("trace continues after task termination")
    final = record["final"]
    if final.get("frames") != frames or final.get("num_steps") != len(trace):
        raise ValueError("inconsistent final step/frame evidence")
    if final.get("finish") != trace[-1]["finish"] or set(final.get("completed_ids", [])) != previous_completed:
        raise ValueError("inconsistent final target evidence")
    if final.get("task_success_within_time") is not success or final.get("termination_reason") != reason:
        raise ValueError("inconsistent final termination evidence")
    if not done:
        raise ValueError("source trajectory is incomplete and lacks terminal evidence")
    return record


def load_episode_record(directory):
    directory = Path(directory)
    metadata = json.loads((directory / "episode_metadata.json").read_text())
    trace = json.loads((directory / "episode_trace.json").read_text())
    result = json.loads((directory / "result_episode.json").read_text())
    if (result.get("metrics") or {}).get("error"):
        raise ValueError("source episode is an infrastructure error record")
    if result.get("effective_config") != metadata.get("effective_config"):
        raise ValueError("result and raw metadata configurations differ")
    if not trace:
        raise ValueError("empty source trace")
    record = {
        "metadata": metadata, "trace": trace,
        "final": {
            "finish": trace[-1]["finish"], "completed_ids": trace[-1].get("satisfied_target_ids"),
            "frames": result.get("frames"), "num_steps": result.get("steps"),
            "task_success_within_time": result.get("done"),
            "termination_reason": result.get("termination_reason"),
        },
    }
    return validate_record(record)
