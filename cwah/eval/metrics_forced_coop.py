"""Method-independent episode metrics for forced-cooperation runs."""

import copy
import math

from collections import Counter

from envs.physical_time import PHYSICAL_TIME_SETTINGS


TASK_GROUPS = ('all', 'ordinary', 'coop')
OUTCOME_GROUPS = ('all', 'success', 'failure')

SUCCESS_RATE_KEYS = (
    'task_success_rate',
    'ordinary_task_success_rate',
    'coop_task_success_rate',
    'subgoal_success_rate',
    'ordinary_subgoal_success_rate',
    'coop_subgoal_success_rate',
)

NUMERIC_METRIC_PATHS = {
    'num_steps': (('num_steps',),),
    'physical_action_frames': (('execution_time', 'physical_action_frames'),),
    'physical_execution_time': (
        ('execution_time', 'physical_execution_time'),
    ),
    'embodied_execution_time': (
        ('execution_time', 'embodied_execution_time'),
    ),
    'message_send_time_agent_sum': (
        ('execution_time', 'message_send_time_agent_sum'),
    ),
    'communication_critical_path_overhead': (
        ('execution_time', 'communication_critical_path_overhead'),
    ),
    'communication_total_time_agent_sum': (
        ('execution_time', 'communication_total_time_agent_sum'),
    ),
    'nonphysical_extra_time': (('execution_time', 'nonphysical_extra_time'),),
    'communication_time_step_sum': (('execution_time', 'communication_time_step_sum'),),
    'communication_extra_time': (('execution_time', 'communication_extra_time'),),
    'nonidle_execution_time': (('execution_time', 'nonidle_execution_time'),),
    'idle_step_count': (('execution_time', 'idle_step_count'),),
    'idle_execution_time': (('execution_time', 'idle_execution_time'),),
    'max_embodied_execution_time': (
        ('max_embodied_execution_time',),
        ('execution_time', 'max_embodied_execution_time'),
    ),
    'tgc_auc': (('tgc_auc',),),
    'communication_chars': (('execution_time', 'communication_chars'),),
    'message_count': (('execution_time', 'message_count'),),
    'message_truncated_count': (('execution_time', 'message_truncated_count'),),
    'failed_exec_count': (('failed_exec_count',),),
    'invalid_action_count': (('failure_summary', 'invalid_action_count'),),
    'coop_failure_event_total': (('failure_summary', 'coop_failure_event_total'),),
    'coop_event_total': (('coop_event_total',),),
    'llm_call_count': (('execution_time', 'llm_call_count'),),
    'llm_prompt_tokens': (('execution_time', 'llm_prompt_tokens'),),
    'llm_completion_tokens': (('execution_time', 'llm_completion_tokens'),),
    'llm_total_tokens': (('execution_time', 'llm_total_tokens'),),
    'llm_inference_time_estimated': (
        ('execution_time', 'llm_inference_time_estimated'),
    ),
    'llm_wall_time': (('execution_time', 'llm_wall_time'),),
}


def metric_rate(numerator, denominator):
    if denominator <= 0:
        return None
    return float(numerator) / float(denominator)


def metric_entry(success_count, total_count):
    return {
        'success_count': int(success_count),
        'total_count': int(total_count),
        'rate': metric_rate(success_count, total_count),
    }


def _iter_coop_step_infos(saved_info):
    for step_info in saved_info.get('coop', []) or []:
        if isinstance(step_info, dict):
            yield step_info


def _iter_coop_events(saved_info):
    for step_info in _iter_coop_step_infos(saved_info):
        for event in step_info['coop_step_events']:
            if not isinstance(event, dict):
                raise ValueError('invalid cooperative event')
            yield event

def has_coop_activity(saved_info):
    for step_info in _iter_coop_step_infos(saved_info):
        if step_info.get('coop_step_events'):
            return True
        if step_info.get('coop_carrying') is not None:
            return True
    return False


def summarize_coop_events(saved_info):
    event_counts = Counter()
    last_event = None
    for event in _iter_coop_events(saved_info):
        event_type = event.get('type', 'unknown')
        event_counts[event_type] += 1
        last_event = event

    return {
        'event_counts': dict(sorted(event_counts.items())),
        'total_events': int(sum(event_counts.values())),
        'last_event': copy.deepcopy(last_event),
    }




def _failed_exec_count(saved_info):
    return sum(1 for item in saved_info.get('failed_exec', []) if bool(item))


# ==================== Metrics: Task and Subgoal Success Rates ====================
# Purpose: Compute method-independent task/subgoal success from saved episode info.
def _latest_coop_step_info(saved_info):
    latest = {}
    for step_info in _iter_coop_step_infos(saved_info):
        latest = step_info
    return latest


def _get_latest_coop_constraints(saved_info):
    latest = _latest_coop_step_info(saved_info)
    constraints = latest.get('coop_constraints', {}) if isinstance(latest, dict) else {}
    return constraints if isinstance(constraints, dict) else {}


def _has_joint_task(saved_info):
    coop_constraints = _get_latest_coop_constraints(saved_info)
    for constraint in coop_constraints.values():
        if isinstance(constraint, dict) and constraint.get('requires_two_agents', False):
            return True
    return bool(coop_constraints)


def _build_id_to_class_lookup(saved_info):
    graph = saved_info.get('init_unity_graph', {}) or {}
    return {
        node.get('id'): node.get('class_name')
        for node in graph.get('nodes', [])
        if isinstance(node, dict) and 'id' in node
    }


def _goal_key_is_coop_subgoal(goal_key, coop_constraints, id_to_class):
    if not coop_constraints:
        return False

    elements = str(goal_key).split('_')
    if len(elements) < 2:
        return False

    object_token = elements[1]
    coop_key_strings = {str(key) for key in coop_constraints.keys()}
    if object_token in coop_key_strings:
        return True

    if object_token.isdigit():
        object_id = int(object_token)
        class_name = id_to_class.get(object_id)
        return class_name in coop_key_strings

    return False


def _goal_key_signature(goal_key):
    elements = str(goal_key).split('_', 2)
    if len(elements) < 3:
        return None

    relation, object_token, target_token = elements
    target_id = None
    if target_token.isdigit():
        target_id = target_token
    elif '(' in target_token and ')' in target_token:
        try:
            target_id = target_token.split('(')[-1].split(')')[0]
        except Exception:
            target_id = None

    if target_id is None:
        target_id = target_token
    return relation, object_token, str(target_id)


def _build_unsatisfied_lookup(unsatisfied):
    exact = {}
    by_signature = {}
    for key, value in (unsatisfied or {}).items():
        exact[key] = value
        signature = _goal_key_signature(key)
        if signature is not None:
            by_signature[signature] = (key, value)
    return exact, by_signature


def _lookup_unmet_count(goal_key, required_count, unsatisfied_exact, unsatisfied_by_signature):
    if goal_key in unsatisfied_exact:
        return int(unsatisfied_exact[goal_key]), goal_key

    signature = _goal_key_signature(goal_key)
    if signature is not None and signature in unsatisfied_by_signature:
        matched_key, value = unsatisfied_by_signature[signature]
        return int(value), matched_key

    return required_count, None


def _get_goal_counts(saved_info):
    gt_goals = saved_info.get('gt_goals', {}) or {}
    goal_counts = {}
    for goal_key, value in gt_goals.items():
        if isinstance(value, (list, tuple)):
            if len(value) == 0:
                continue
            goal_counts[goal_key] = int(value[0])
        else:
            goal_counts[goal_key] = int(value)
    return goal_counts


def _compute_subgoal_metric_counts(saved_info):
    final_progress = saved_info.get('progress', [])[-1] if saved_info.get('progress') else {}
    unsatisfied = final_progress.get('unsatisfied', {}) if isinstance(final_progress, dict) else {}
    goal_counts = _get_goal_counts(saved_info)
    coop_constraints = _get_latest_coop_constraints(saved_info)
    id_to_class = _build_id_to_class_lookup(saved_info)
    unsatisfied_exact, unsatisfied_by_signature = _build_unsatisfied_lookup(unsatisfied)

    counts = {
        'all': {'success': 0, 'total': 0},
        'ordinary': {'success': 0, 'total': 0},
        'coop': {'success': 0, 'total': 0},
    }
    per_subgoal = {}

    for goal_key, required_count in goal_counts.items():
        unmet_count, progress_key = _lookup_unmet_count(
            goal_key,
            required_count,
            unsatisfied_exact,
            unsatisfied_by_signature,
        )
        unmet_count = min(required_count, max(0, unmet_count))
        achieved_count = required_count - unmet_count
        is_coop = _goal_key_is_coop_subgoal(goal_key, coop_constraints, id_to_class)
        group = 'coop' if is_coop else 'ordinary'

        counts['all']['success'] += achieved_count
        counts['all']['total'] += required_count
        counts[group]['success'] += achieved_count
        counts[group]['total'] += required_count
        per_subgoal[goal_key] = {
            'required': required_count,
            'achieved': achieved_count,
            'unsatisfied': unmet_count,
            'is_coop_subgoal': is_coop,
            'progress_key': progress_key,
        }

    return counts, per_subgoal


def _completion_fraction(saved_info, progress):
    goal_counts = _get_goal_counts(saved_info)
    total_count = int(sum(max(0, value) for value in goal_counts.values()))
    if total_count <= 0:
        return 1.0
    unsatisfied = progress.get('unsatisfied', {}) if isinstance(progress, dict) else {}
    unsatisfied_exact, unsatisfied_by_signature = _build_unsatisfied_lookup(unsatisfied)
    completed_count = 0
    for goal_key, required_count in goal_counts.items():
        unmet_count, _ = _lookup_unmet_count(
            goal_key,
            required_count,
            unsatisfied_exact,
            unsatisfied_by_signature,
        )
        unmet_count = min(required_count, max(0, unmet_count))
        completed_count += required_count - unmet_count
    return min(1.0, max(0.0, float(completed_count) / float(total_count)))


def _build_success_rates(saved_info):
    task_success = bool(saved_info['task_success_within_time'])
    has_joint_task = _has_joint_task(saved_info)
    subgoal_counts, per_subgoal = _compute_subgoal_metric_counts(saved_info)
    success_rates = {
        'task_success_rate': metric_entry(int(task_success), 1),
        'ordinary_task_success_rate': metric_entry(
            int(task_success) if not has_joint_task else 0,
            0 if has_joint_task else 1,
        ),
        'coop_task_success_rate': metric_entry(
            int(task_success) if has_joint_task else 0,
            1 if has_joint_task else 0,
        ),
        'subgoal_success_rate': metric_entry(
            subgoal_counts['all']['success'],
            subgoal_counts['all']['total'],
        ),
        'ordinary_subgoal_success_rate': metric_entry(
            subgoal_counts['ordinary']['success'],
            subgoal_counts['ordinary']['total'],
        ),
        'coop_subgoal_success_rate': metric_entry(
            subgoal_counts['coop']['success'],
            subgoal_counts['coop']['total'],
        ),
    }
    return has_joint_task, success_rates, per_subgoal
# ==================== End Metrics: Task and Subgoal Success Rates ====================


DEFAULT_LLM_INFERENCE_TOKENS_PER_SECOND = 50.0

# ==================== Metrics: LLM Call Tokens and Time ====================
# Purpose: Summarize per-agent LLM metadata saved by arena without changing run logic.
def _iter_llm_records(saved_info):
    llm_info = saved_info.get('LLM', {}) or {}
    for agent_records in llm_info.values():
        for record in agent_records or []:
            if isinstance(record, dict):
                yield record


def _summarize_llm_records(saved_info):
    summary = {
        'llm_call_count': 0,
        'llm_wall_time': 0.0,
        'llm_inference_time_estimated': 0.0,
        'llm_prompt_tokens': 0,
        'llm_completion_tokens': 0,
        'llm_total_tokens': 0,
        'llm_call_details': [],
    }

    for record in _iter_llm_records(saved_info):
        llm_calls = record.get('llm_calls')
        if isinstance(llm_calls, list):
            summary['llm_call_details'].extend(copy.deepcopy(llm_calls))

        summary['llm_call_count'] += int(record.get('llm_call_count') or 0)
        summary['llm_wall_time'] += float(record.get('llm_wall_time') or 0.0)
        summary['llm_prompt_tokens'] += int(record.get('llm_prompt_tokens') or 0)
        summary['llm_completion_tokens'] += int(record.get('llm_completion_tokens') or 0)
        summary['llm_total_tokens'] += int(record.get('llm_total_tokens') or 0)
        summary['llm_inference_time_estimated'] += float(record.get('llm_inference_time_estimated') or 0.0)

    if summary['llm_inference_time_estimated'] <= 0.0 and DEFAULT_LLM_INFERENCE_TOKENS_PER_SECOND > 0:
        summary['llm_inference_time_estimated'] = (
            float(summary['llm_total_tokens']) / DEFAULT_LLM_INFERENCE_TOKENS_PER_SECOND
        )

    return summary
# ==================== End Metrics: LLM Thinking Time and Usage ====================


# ==================== Metrics: Failure and Blocked Action Summary ====================
# Purpose: Summarize env execution failures and coop-rule blocked events from saved logs.
COOP_FAILURE_EVENT_TYPES = {
    'coop_carrying_distance_violation',
    'coop_grab_blocked',
    'coop_grab_cancelled',
    'solo_grab_blocked',
    'coop_put_blocked',
    'solo_put_blocked',
    'coop_pending_update_cancelled',
}


# ==================== Metrics: Standard Action Feedback ====================
# Purpose: Aggregate per-agent action status and invalid reasons like TDW-MAT.
def _iter_agent_step_values(saved_info, field_name):
    for step_values in saved_info.get(field_name, []) or []:
        if isinstance(step_values, dict):
            for value in step_values.values():
                yield value
        elif isinstance(step_values, (list, tuple)):
            for value in step_values:
                yield value


def _summarize_action_feedback(saved_info):
    invalid_reasons = Counter(str(value) for value in _iter_agent_step_values(saved_info, 'invalid_reasons') if value is not None)
    action_statuses = Counter(str(value) for value in _iter_agent_step_values(saved_info, 'action_status') if value is not None)
    return {
        'invalid_action_count': int(sum(invalid_reasons.values())),
        'invalid_reason_counts': dict(sorted(invalid_reasons.items())),
        'action_status_counts': dict(sorted(action_statuses.items())),
    }
# ==================== End Metrics: Standard Action Feedback ====================


def _iter_failed_exec_steps(saved_info):
    for step_index, failed in enumerate(saved_info.get('failed_exec', []) or []):
        if bool(failed):
            yield step_index


def _is_failure_like_coop_event(event):
    return event.get('type') in COOP_FAILURE_EVENT_TYPES


def _summarize_failure_events(saved_info):
    failed_exec_steps = list(_iter_failed_exec_steps(saved_info))
    action_feedback_summary = _summarize_action_feedback(saved_info)
    coop_failure_counts = Counter()
    last_coop_failure_event = None

    for event in _iter_coop_events(saved_info):
        if not _is_failure_like_coop_event(event):
            continue
        event_type = event.get('type', 'unknown')
        coop_failure_counts[event_type] += 1
        last_coop_failure_event = event

    coop_failure_total = int(sum(coop_failure_counts.values()))
    return {
        'has_failure': bool(failed_exec_steps) or coop_failure_total > 0 or action_feedback_summary['invalid_action_count'] > 0,
        'failed_exec_count': int(len(failed_exec_steps)),
        'failed_exec_steps': failed_exec_steps,
        'first_failed_exec_step': failed_exec_steps[0] if failed_exec_steps else None,
        'last_failed_exec_step': failed_exec_steps[-1] if failed_exec_steps else None,
        'coop_failure_event_counts': dict(sorted(coop_failure_counts.items())),
        'coop_failure_event_total': coop_failure_total,
        'last_coop_failure_event': copy.deepcopy(last_coop_failure_event),
        **action_feedback_summary,
    }
# ==================== End Metrics: Failure and Blocked Action Summary ====================


# ==================== Metrics: Physical Action Execution Time ====================
# Purpose: Summarize calibrated PAET from per-step physical evidence.
def _iter_physical_action_frame_infos(saved_info):
    for frame_info in saved_info.get('physical_action_frames', []) or []:
        if isinstance(frame_info, dict):
            yield frame_info


def _summarize_physical_action_time(saved_info):
    summary = {
        'physical_action_enabled': False,
        'physical_action_frame_rate': PHYSICAL_TIME_SETTINGS.frame_rate,
        'physical_action_render_mode': None,
        'physical_time_source': 'calibrated_action_model',
        'physical_time_parameters': PHYSICAL_TIME_SETTINGS.to_dict(),
        'physical_action_frames': 0,
        'physical_action_time': 0.0,
        'physical_action_step_count': 0,
        'physical_action_details': [],
    }

    for frame_info in _iter_physical_action_frame_infos(saved_info):
        summary['physical_action_enabled'] = summary['physical_action_enabled'] or bool(frame_info.get('enabled'))
        if summary['physical_action_frame_rate'] is None and frame_info.get('frame_rate') is not None:
            summary['physical_action_frame_rate'] = frame_info.get('frame_rate')
        if summary['physical_action_render_mode'] is None and frame_info.get('render_mode') is not None:
            summary['physical_action_render_mode'] = frame_info.get('render_mode')
        if summary['physical_time_source'] is None and frame_info.get('time_source') is not None:
            summary['physical_time_source'] = frame_info.get('time_source')
        parameters = frame_info.get('physical_time_parameters')
        if (parameters != PHYSICAL_TIME_SETTINGS.to_dict()
                or frame_info.get('time_source') != 'calibrated_action_model'
                or frame_info.get('frame_rate') != PHYSICAL_TIME_SETTINGS.frame_rate):
            raise ValueError('physical time settings do not match benchmark calibration')
        step_frames = frame_info.get('step_physical_action_frames')
        step_time = frame_info.get('step_physical_action_time')
        if (type(step_frames) is not int or step_frames < 0
                or not isinstance(step_time, (int, float)) or isinstance(step_time, bool)
                or not math.isfinite(step_time) or step_time < 0
                or not math.isclose(step_time, step_frames / PHYSICAL_TIME_SETTINGS.frame_rate, abs_tol=1e-9)):
            raise ValueError('invalid physical execution time evidence')

        summary['physical_action_frames'] += step_frames
        summary['physical_action_time'] += float(step_time or 0.0)
        summary['physical_action_step_count'] += 1
        summary['physical_action_details'].append({
            'step': frame_info.get('step'),
            'enabled': bool(frame_info.get('enabled')),
            'frame_rate': frame_info.get('frame_rate'),
            'render_mode': frame_info.get('render_mode'),
            'time_source': frame_info.get('time_source'),
            'physical_time_parameters': copy.deepcopy(frame_info.get('physical_time_parameters')),
            'skip_animation': frame_info.get('skip_animation'),
            'aggregation': frame_info.get('aggregation'),
            'unit': frame_info.get('unit'),
            'step_physical_action_frames': step_frames,
            'step_physical_action_time': float(step_time or 0.0),
            'joint': copy.deepcopy(frame_info.get('joint')),
            'per_agent': copy.deepcopy(frame_info.get('per_agent', {})),
        })

    return summary
# ==================== End Metrics: Physical Action Execution Time ====================


# ==================== Metrics: Step Execution Time ====================
# Purpose: Summarize per-step physical and communication execution-time estimates.
def _iter_execution_time_step_infos(saved_info):
    for step_info in saved_info.get('execution_time_steps', []) or []:
        if isinstance(step_info, dict):
            yield step_info


def _summarize_step_execution_time(saved_info, physical_action_summary):
    steps = saved_info['execution_time_steps']
    totals = ('embodied_execution_time', 'message_send_time_agent_sum',
              'communication_critical_path_overhead', 'communication_time_step_sum',
              'idle_step_count', 'idle_execution_time', 'nonidle_execution_time',
              'communication_extra_time', 'communication_chars')
    summary = dict.fromkeys(totals, 0)
    summary.update(communication_details=[], execution_time_step_details=[])
    settings = set()
    for step in steps:
        speed = step.get('communication_send_chars_per_second')
        if (not isinstance(speed, (int, float)) or isinstance(speed, bool)
                or not math.isfinite(speed) or speed <= 0
                or step.get('communication_count_unit') != 'unicode_codepoint'):
            raise ValueError('invalid character communication speed/unit')
        settings.add((step['communication_count_unit'], speed))
        messages = step.get('messages')
        if not isinstance(messages, list):
            raise ValueError('missing delivered-message evidence')
        senders = set()
        for message in messages:
            if not isinstance(message, dict):
                raise ValueError('invalid delivered-message evidence')
            content, sender = message.get('content'), message.get('sender_agent')
            if (not isinstance(content, str) or not content or type(sender) is not int
                    or sender not in (0, 1) or sender in senders
                    or type(message.get('message_chars')) is not int
                    or message['message_chars'] != len(content)):
                raise ValueError('invalid delivered-message character evidence')
            senders.add(sender)
        chars = sum(len(message['content']) for message in messages)
        if type(step.get('communication_chars')) is not int or step['communication_chars'] != chars:
            raise ValueError('inconsistent communication character total')
        for key in ('physical_execution_time', 'minimum_step_time', 'message_send_time_agent_sum',
                    'communication_time_step_sum', 'idle_execution_time', 'nonidle_execution_time',
                    'communication_extra_time', 'step_embodied_execution_time',
                    'communication_critical_path_overhead'):
            value = step.get(key)
            if (not isinstance(value, (int, float)) or isinstance(value, bool)
                    or not math.isfinite(value) or value < 0):
                raise ValueError('invalid execution time evidence: ' + key)
        idle = step.get('idle_step_count')
        if (type(idle) is not int or idle not in (0, 1)
                or type(step.get('is_idle_step')) is not bool or step['is_idle_step'] != bool(idle)
                or type(step.get('has_physical_action')) is not bool
                or bool(idle) != (not step['has_physical_action'] and not messages)):
            raise ValueError('inconsistent idle action evidence')
        physical = step['physical_execution_time']
        if physical > 0 and not step['has_physical_action']:
            raise ValueError('physical time without physical action evidence')
        comm = max((len(message['content']) / speed for message in messages), default=0.)
        idle_time = step['minimum_step_time'] if idle else 0.
        total = max(physical, comm, idle_time)
        expected = {'message_send_time_agent_sum': chars / speed,
                    'communication_time_step_sum': comm, 'idle_execution_time': idle_time,
                    'nonidle_execution_time': total - idle_time,
                    'communication_extra_time': max(0., total - physical - idle_time),
                    'step_embodied_execution_time': total,
                    'communication_critical_path_overhead': max(0., total - physical - idle_time)}
        if (type(step.get('message_count')) is not int or step['message_count'] != len(messages)
                or any(not math.isclose(step[key], value, abs_tol=1e-9) for key, value in expected.items())):
            raise ValueError('inconsistent execution time decomposition')
        for key in totals:
            summary[key] += step['step_embodied_execution_time'] if key == 'embodied_execution_time' else step[key]
        summary['communication_details'].extend(copy.deepcopy(messages))
        summary['execution_time_step_details'].append(copy.deepcopy(step))
    if len(settings) > 1:
        raise ValueError('mixed communication settings within one episode')
    unit, speed = next(iter(settings), (saved_info['communication_count_unit'],
                                       saved_info['communication_send_chars_per_second']))
    if (unit != saved_info['communication_count_unit']
            or speed != saved_info['communication_send_chars_per_second']):
        raise ValueError('record and step communication settings disagree')
    summary.update(communication_count_unit=unit, communication_send_chars_per_second=speed)
    return summary# ==================== End Metrics: Step Execution Time ====================


# ==================== Metrics: Time-Normalized Goal Completion AUC ====================
# Purpose: Reward early, sustained goal completion on a shared embodied-time axis.
def _summarize_tgc_auc(saved_info, execution_time_summary):
    step_infos = list(_iter_execution_time_step_infos(saved_info))
    progress_steps = list(saved_info.get('progress', []) or [])
    budget = float(saved_info['max_embodied_execution_time'])
    initial_progress = saved_info['initial_progress']
    current_fraction = _completion_fraction(saved_info, initial_progress)
    elapsed = 0.0
    area = 0.0
    consumed_steps = 0
    for step_index, step_info in enumerate(step_infos):
        step_time = float(
            step_info['step_embodied_execution_time']
        )
        step_end = elapsed + max(0.0, step_time)
        interval_end = min(budget, step_end)
        if interval_end > elapsed:
            area += current_fraction * (interval_end - elapsed)
        elapsed = interval_end
        consumed_steps = step_index + 1
        # Completion at exactly the budget is valid; completion beyond it is
        # terminal progress only and must not change the fraction at the budget.
        if step_end <= budget and step_index < len(progress_steps):
            current_fraction = _completion_fraction(saved_info, progress_steps[step_index])
        if elapsed >= budget:
            break

    if elapsed < budget:
        if consumed_steps < len(progress_steps):
            current_fraction = _completion_fraction(
                saved_info,
                progress_steps[min(consumed_steps, len(progress_steps) - 1)],
            )
        area += current_fraction * (budget - elapsed)

    final_progress = progress_steps[-1] if progress_steps else initial_progress
    return {
        'tgc_auc': min(1.0, max(0.0, area / budget)),
        'time_budget': budget,
        'completion_fraction_at_budget': current_fraction,
        'final_completion_fraction': _completion_fraction(saved_info, final_progress),
    }
# ==================== End Metrics: Time-Normalized Goal Completion AUC ====================


def validate_episode_record(saved_info):
    """Require the actual evidence written by the C-WAH runner."""
    if not isinstance(saved_info, dict):
        raise ValueError('episode record must be a dictionary')
    required = ('gt_goals', 'initial_progress', 'progress', 'action', 'coop', 'LLM',
                'physical_action_frames', 'execution_time_steps', 'task_success_within_time',
                'max_embodied_execution_time', 'max_episode_length', 'communication_count_unit',
                'communication_send_chars_per_second', 'communication_mode',
                'communication_max_message_chars', 'minimum_step_frames')
    missing = [key for key in required if key not in saved_info]
    if missing:
        raise ValueError('missing episode evidence: ' + ', '.join(missing))
    if type(saved_info['task_success_within_time']) is not bool:
        raise ValueError('invalid task success evidence')
    if not isinstance(saved_info['initial_progress'], dict) or 'unsatisfied' not in saved_info['initial_progress']:
        raise ValueError('missing initial goal progress')
    budget = saved_info['max_embodied_execution_time']
    if (not isinstance(budget, (int, float)) or isinstance(budget, bool)
            or not math.isfinite(budget) or budget <= 0):
        raise ValueError('invalid collection time budget')
    steps = saved_info['execution_time_steps']
    if not isinstance(steps, list) or any(not isinstance(step, dict) for step in steps):
        raise ValueError('invalid execution time steps')
    for key in ('progress', 'coop', 'physical_action_frames'):
        rows = saved_info[key]
        if (not isinstance(rows, list) or len(rows) != len(steps)
                or any(not isinstance(row, dict) for row in rows)):
            raise ValueError('missing or misaligned episode evidence: ' + key)
    for key in ('action', 'LLM'):
        rows = saved_info[key]
        if (not isinstance(rows, dict) or set(rows) != {0, 1}
                or any(not isinstance(row, list) or len(row) != len(steps) for row in rows.values())):
            raise ValueError('missing or misaligned agent evidence: ' + key)
    for row in saved_info['progress']:
        if 'unsatisfied' not in row:
            raise ValueError('missing goal progress')
    for row in saved_info['coop']:
        if 'coop_constraints' not in row or not isinstance(row.get('coop_step_events'), list):
            raise ValueError('missing cooperative task/event evidence')
    if any(not isinstance(row, dict) for rows in saved_info['LLM'].values() for row in rows):
        raise ValueError('invalid LLM step evidence')
    for row in _iter_llm_records(saved_info):
        if type(row.get('llm_call_count')) is not int or row['llm_call_count'] < 0:
            raise ValueError('missing LLM call evidence')


def build_episode_metrics(saved_info):
    validate_episode_record(saved_info)
    coop_summary = summarize_coop_events(saved_info)
    has_joint_task, success_rates, per_subgoal = _build_success_rates(saved_info)
    llm_summary = _summarize_llm_records(saved_info)
    failure_summary = _summarize_failure_events(saved_info)
    physical_action_summary = _summarize_physical_action_time(saved_info)
    step_execution_summary = _summarize_step_execution_time(saved_info, physical_action_summary)
    tgc_summary = _summarize_tgc_auc(saved_info, step_execution_summary)
    num_steps = max((len(actions) for actions in (saved_info.get('action') or {}).values()), default=0)
    communication_details = step_execution_summary['communication_details']
    message_count = len(communication_details)
    message_truncated_count = sum(
        1 for message in communication_details
        if isinstance(message, dict) and bool(message.get('truncated'))
    )
    metrics = {
        'task_success': bool(saved_info.get('finished', False)),
        'task_success_within_time': bool(
            saved_info['task_success_within_time']
        ),
        'termination_reason': saved_info.get('termination_reason'),
        'task_type': 'coop' if has_joint_task else 'ordinary',
        'has_joint_task': has_joint_task,
        'num_steps': int(num_steps),
        'success_rates': success_rates,
        'per_subgoal': per_subgoal,
        'has_coop_activity': has_coop_activity(saved_info),
        'coop_event_counts': coop_summary['event_counts'],
        'coop_event_total': coop_summary['total_events'],
        'last_coop_event': coop_summary['last_event'],
        'failed_exec_count': int(_failed_exec_count(saved_info)),
        'failure_summary': failure_summary,
        'embodied_execution_time': step_execution_summary['embodied_execution_time'],
        'max_embodied_execution_time': tgc_summary['time_budget'],
        'tgc_auc': tgc_summary['tgc_auc'],
        'time_normalized_goal_completion': tgc_summary,
        'execution_time': {
            'embodied_execution_time': step_execution_summary['embodied_execution_time'],
            'physical_execution_time': physical_action_summary['physical_action_time'],
            'physical_action_render_mode': physical_action_summary['physical_action_render_mode'],
            'physical_time_source': physical_action_summary['physical_time_source'],
            'physical_time_parameters': physical_action_summary['physical_time_parameters'],
            'message_send_time_agent_sum': step_execution_summary['message_send_time_agent_sum'],
            'communication_critical_path_overhead': step_execution_summary['communication_critical_path_overhead'],
            'max_embodied_execution_time': tgc_summary['time_budget'],
            'communication_chars': step_execution_summary['communication_chars'],
            'communication_time_step_sum': step_execution_summary['communication_time_step_sum'],
            'communication_extra_time': step_execution_summary['communication_extra_time'],
            'nonidle_execution_time': step_execution_summary['nonidle_execution_time'],
            'idle_step_count': step_execution_summary['idle_step_count'],
            'idle_execution_time': step_execution_summary['idle_execution_time'],
            'communication_count_unit': step_execution_summary['communication_count_unit'],
            'communication_send_chars_per_second': step_execution_summary['communication_send_chars_per_second'],
            'message_count': int(message_count),
            'message_truncated_count': int(message_truncated_count),
            'communication_details': communication_details,
            'execution_time_step_details': step_execution_summary['execution_time_step_details'],
            'llm_call_count': llm_summary['llm_call_count'],
            'llm_prompt_tokens': llm_summary['llm_prompt_tokens'],
            'llm_completion_tokens': llm_summary['llm_completion_tokens'],
            'llm_total_tokens': llm_summary['llm_total_tokens'],
            'llm_inference_time_estimated': llm_summary['llm_inference_time_estimated'],
            'llm_wall_time': llm_summary['llm_wall_time'],
            'llm_call_details': llm_summary['llm_call_details'],
            'physical_action_enabled': physical_action_summary['physical_action_enabled'],
            'physical_action_frame_rate': physical_action_summary['physical_action_frame_rate'],
            'physical_action_frames': physical_action_summary['physical_action_frames'],
            'physical_action_step_count': physical_action_summary['physical_action_step_count'],
            'physical_action_details': physical_action_summary['physical_action_details'],
        },
    }
    execution = metrics['execution_time']
    for key in ('communication_mode', 'communication_max_message_chars', 'minimum_step_frames'):
        execution[key] = saved_info[key]
    for key in ('llm_prompt_tokens', 'llm_completion_tokens', 'llm_total_tokens'):
        if any(record['llm_call_count'] > 0 and record.get(key) is None
               for record in _iter_llm_records(saved_info)):
            execution[key] = None
    return finalize_episode_metrics(metrics)


# ==================== Metrics: Derived Metric Validation ====================
# Purpose: Validate current evidence and derive time totals.
def finalize_episode_metrics(metrics):
    """Validate current metrics and compute their derived time quantities."""
    result = dict(metrics)
    execution = dict(result.get('execution_time') or {})
    result['execution_time'] = execution
    required = ('embodied_execution_time', 'physical_execution_time', 'message_send_time_agent_sum',
                'communication_time_step_sum', 'idle_step_count', 'idle_execution_time',
                'communication_count_unit', 'communication_send_chars_per_second',
                'physical_time_source', 'physical_time_parameters')
    missing = [key for key in required if key not in execution]
    if missing or type(result.get('task_success_within_time')) is not bool:
        raise ValueError('missing current metric evidence: ' + ', '.join(missing))
    if (execution['physical_time_source'] != 'calibrated_action_model'
            or execution['physical_time_parameters'] != PHYSICAL_TIME_SETTINGS.to_dict()):
        raise ValueError('metric calibration does not match benchmark settings')
    speed = execution['communication_send_chars_per_second']
    if (execution['communication_count_unit'] != 'unicode_codepoint'
            or not isinstance(speed, (int, float)) or isinstance(speed, bool)
            or not math.isfinite(speed) or speed <= 0
            or type(execution['idle_step_count']) is not int):
        raise ValueError('invalid communication/idle metric settings')
    for key in required[:6]:
        value = execution[key]
        if (not isinstance(value, (int, float)) or isinstance(value, bool)
                or not math.isfinite(value) or value < 0):
            raise ValueError('invalid metric value: ' + key)
    total, physical, idle = (execution[key] for key in
                             ('embodied_execution_time', 'physical_execution_time', 'idle_execution_time'))
    if total + 1e-9 < physical + idle:
        raise ValueError('inconsistent physical/idle metric totals')
    execution.update(communication_total_time_agent_sum=execution['message_send_time_agent_sum'],
                     nonphysical_extra_time=total - physical,
                     nonidle_execution_time=total - idle,
                     communication_extra_time=max(0., total - physical - idle))
    rates = dict(result.get('success_rates') or {})
    result['success_rates'] = rates
    for key in SUCCESS_RATE_KEYS:
        if key not in rates:
            raise ValueError('missing success metric: ' + key)
        entry = rates[key]
        if not isinstance(entry, dict) or not {'total_count', 'success_count'} <= entry.keys():
            raise ValueError('missing success counts: ' + key)
        total_count, achieved = entry['total_count'], entry['success_count']
        if (type(total_count) is not int or type(achieved) is not int or not 0 <= achieved <= total_count):
            raise ValueError('invalid success counts: ' + key)
        if key.endswith('task_success_rate'):
            achieved = int(result['task_success_within_time']) * total_count
        rates[key] = metric_entry(achieved, total_count)
    return result# ==================== End Metrics: Derived Metric Validation ====================


_MISSING = object()


def _lookup_path(value, path):
    current = value
    for key in path:
        if not isinstance(current, dict) or key not in current:
            return _MISSING
        current = current[key]
    return current


def _lookup_numeric_metric(metrics, candidate_paths):
    for path in candidate_paths:
        value = _lookup_path(metrics, path)
        if value is _MISSING or value is None or isinstance(value, bool):
            continue
        try:
            numeric_value = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(numeric_value):
            return numeric_value
    return None


def _numeric_summary(values):
    values = [float(value) for value in values]
    if not values:
        return {
            'count': 0,
            'sum': 0.0,
            'mean': None,
            'population_std': None,
            'min': None,
            'max': None,
        }
    mean = sum(values) / float(len(values))
    variance = sum((value - mean) ** 2 for value in values) / float(len(values))
    return {
        'count': len(values),
        'sum': sum(values),
        'mean': mean,
        'population_std': math.sqrt(variance),
        'min': min(values),
        'max': max(values),
    }


def _merge_count_map(counter, values):
    for key, value in (values or {}).items():
        try:
            counter[str(key)] += int(value)
        except (TypeError, ValueError):
            continue


# ==================== Metrics: Experiment-Level Aggregation ====================
# Purpose: Pool rates by counts and summarize every valid episode-run, including failures.
# Purpose: Apply the same task/outcome strata to all core metric families.
def _summarize_core_group(records):
    rates = {}
    for key in SUCCESS_RATE_KEYS:
        entries = [record.get('success_rates', {}).get(key) or {} for record in records]
        rates[key] = metric_entry(
            sum(int(entry.get('success_count') or 0) for entry in entries),
            sum(int(entry.get('total_count') or 0) for entry in entries),
        )
    summaries = {
        key: _numeric_summary(
            value for record in records
            for value in [_lookup_numeric_metric(record, paths)] if value is not None
        ) for key, paths in NUMERIC_METRIC_PATHS.items()
    }
    return {
        'episode_count': len(records), 'success_rates': rates,
        'numeric_summaries': summaries,
        'averages': {'avg_' + key: value['mean'] for key, value in summaries.items()},
        'totals': {'total_' + key: value['sum'] for key, value in summaries.items()},
    }


def _build_nine_groups(records):
    groups = {}
    for task_type in TASK_GROUPS:
        groups[task_type] = {}
        for outcome in OUTCOME_GROUPS:
            selected = [
                record for record in records
                if (task_type == 'all' or record.get('task_type') == task_type)
                and (outcome == 'all' or
                     bool(record['task_success_within_time'])
                     == (outcome == 'success'))
            ]
            groups[task_type][outcome] = _summarize_core_group(selected)
    return groups


def aggregate_experiment_metrics(metrics_list):
    records = list(metrics_list or [])
    valid_records = [
        finalize_episode_metrics(metrics) for metrics in records
        if isinstance(metrics, dict) and not metrics.get('error')
        and isinstance(metrics.get('success_rates'), dict)
    ]

    identities = {tuple((item.get('execution_time') or {}).get(key) for key in
                        ('communication_count_unit', 'communication_send_chars_per_second'))
                  for item in valid_records}
    if len(identities) > 1:
        raise ValueError('mixed communication settings: evaluate experiments separately')
    groups = _build_nine_groups(valid_records)
    success_rates = groups['all']['all']['success_rates']
    numeric_summaries = groups['all']['all']['numeric_summaries']

    task_type_counts = Counter()
    termination_reason_counts = Counter()
    invalid_reason_counts = Counter()
    action_status_counts = Counter()
    coop_failure_event_counts = Counter()
    coop_event_counts = Counter()
    physical_time_source_counts = Counter()
    for metrics in valid_records:
        task_type_counts[str(metrics.get('task_type') or 'unknown')] += 1
        termination_reason_counts[
            str(metrics.get('termination_reason') or 'unknown')
        ] += 1
        failure_summary = metrics.get('failure_summary') or {}
        _merge_count_map(
            invalid_reason_counts,
            failure_summary.get('invalid_reason_counts'),
        )
        _merge_count_map(
            action_status_counts,
            failure_summary.get('action_status_counts'),
        )
        _merge_count_map(
            coop_failure_event_counts,
            failure_summary.get('coop_failure_event_counts'),
        )
        _merge_count_map(coop_event_counts, metrics.get('coop_event_counts'))
        execution = metrics.get('execution_time') or {}
        physical_time_source_counts[
            str(execution.get('physical_time_source') or 'unknown')
        ] += 1


    averages = {
        'avg_{}'.format(key): summary['mean']
        for key, summary in numeric_summaries.items()
    }
    totals = {
        'total_{}'.format(key): summary['sum']
        for key, summary in numeric_summaries.items()
    }
    return {
        'communication_model': dict(zip(
            ('count_unit', 'send_chars_per_second'),
            next(iter(identities), (None, None)))),
        'aggregation_unit': 'episode-run',
        'aggregation_semantics': {
            'success_rates': 'pooled numerators / pooled denominators (micro)',
            'efficiency': 'arithmetic mean per episode-run; population std retained',
            'tokens': 'sum and arithmetic mean per episode-run, across both agents/all calls',
            'groups': 'task type x episode outcome; subgoal type is independent',
            'communication_total': 'end-to-end send agent-sum workload; not joint elapsed time',
            'extra_time': 'execution - physical, including minimum control-step overhead',
        },
        'groups': groups,
        'episode_record_count': len(records),
        'valid_episode_count': len(valid_records),
        'invalid_metric_record_count': len(records) - len(valid_records),
        'task_type_counts': dict(sorted(task_type_counts.items())),
        'physical_time_source_counts': dict(sorted(physical_time_source_counts.items())),
        'success_rates': success_rates,
        'averages': averages,
        'totals': totals,
        'numeric_summaries': numeric_summaries,
        'termination_reason_counts': dict(sorted(termination_reason_counts.items())),
        'event_counts': {
            'invalid_reason_counts': dict(sorted(invalid_reason_counts.items())),
            'action_status_counts': dict(sorted(action_status_counts.items())),
            'coop_failure_event_counts': dict(sorted(coop_failure_event_counts.items())),
            'coop_event_counts': dict(sorted(coop_event_counts.items())),
        },
    }


def _format_rate(entry):
    success_count = int(entry.get('success_count') or 0)
    total_count = int(entry.get('total_count') or 0)
    if total_count <= 0:
        return 'N/A (0 / 0)'
    percentage = float(entry.get('rate') or 0.0) * 100.0
    return '{} / {} ({:.1f}%)'.format(success_count, total_count, percentage)


def _format_summary(summary, digits=3):
    if not summary or summary.get('mean') is None:
        return 'N/A'
    return ('mean={mean:.{digits}f}, pop.std={std:.{digits}f} '
            '[min={minimum:.{digits}f}, max={maximum:.{digits}f}, n={count}]').format(
        mean=float(summary['mean']),
        std=float(summary.get('population_std') or 0.0),
        minimum=float(summary['min']),
        maximum=float(summary['max']),
        count=int(summary['count']),
        digits=digits,
    )


def _format_count_map(values):
    if not values:
        return 'none'
    return ', '.join('{}={}'.format(key, values[key]) for key in sorted(values))


def _format_value(value, digits=3):
    if value is None:
        return 'N/A'
    return ('{:.%df}' % digits).format(float(value))


def print_episode_metrics(
    metrics, episode_id=None, run_id=None, task_name=None,
    prefix='C-WAH Episode Metrics',
):
    """Print episode totals; auxiliary diagnostics follow the core metrics."""
    if metrics.get('error'):
        print('=== {} | episode={} | run={} | ERROR: {} ==='.format(
            prefix, episode_id, run_id, metrics['error']))
        return
    metrics = finalize_episode_metrics(metrics)
    rates = metrics.get('success_rates') or {}
    execution = metrics.get('execution_time') or {}
    failure = metrics.get('failure_summary') or {}
    print('=== {} | episode={} | run={} | task={} ==='.format(
        prefix, episode_id, run_id, task_name or 'unknown'))
    print('  Outcome   : {} | type={} | termination={}'.format(
        'SUCCESS' if metrics.get('task_success_within_time') else 'FAILURE',
        metrics.get('task_type') or 'unknown', metrics.get('termination_reason') or 'unknown'))
    print('  Task SR   : all {} | ordinary {} | coop {}'.format(*[
        _format_rate(rates.get(key) or {}) for key in SUCCESS_RATE_KEYS[:3]]))
    print('  Subgoal SR: all {} | ordinary {} | coop {}'.format(*[
        _format_rate(rates.get(key) or {}) for key in SUCCESS_RATE_KEYS[3:]]))
    print('  Time(s)   : execution={} / budget={} | physical={}'.format(
        _format_value(execution.get('embodied_execution_time')),
        _format_value(metrics.get('max_embodied_execution_time')),
        _format_value(execution.get('physical_execution_time'))))
    print('  Comm(s)   : end-to-end(agent-sum)={} | step-sum={} | extra(exec-physical-idle)={}'.format(
        _format_value(execution.get('communication_total_time_agent_sum')),
        _format_value(execution.get('communication_time_step_sum')),
        _format_value(execution.get('communication_extra_time'))))
    print('  Idle      : joint-steps={} | time(s)={} | nonidle-execution(s)={}'.format(
        _format_value(execution.get('idle_step_count'), 0),
        _format_value(execution.get('idle_execution_time')),
        _format_value(execution.get('nonidle_execution_time'))))
    print('  Efficiency: TGC-AUC={} | steps={}'.format(
        _format_value(metrics.get('tgc_auc'), 4), metrics.get('num_steps', 'N/A')))
    print('  LLM tokens: input={} | output={} | total={}'.format(*[
        _format_value(execution.get(key), 0) for key in
        ('llm_prompt_tokens', 'llm_completion_tokens', 'llm_total_tokens')]))
    print('  Auxiliary : failed-exec={} | invalid-actions={} | messages={} | message-chars={} | LLM-calls={}'.format(
        metrics.get('failed_exec_count', 'N/A'), failure.get('invalid_action_count', 'N/A'),
        execution.get('message_count', 'N/A'), execution.get('communication_chars', 'N/A'),
        execution.get('llm_call_count', 'N/A')))
    print('  Calibration: source={} | equivalent-frames={} | comm-critical-overhead={} s'.format(
        execution.get('physical_time_source') or 'unknown',
        execution.get('physical_action_frames', 'N/A'),
        _format_value(execution.get('communication_critical_path_overhead'))))
    for goal_key, entry in sorted((metrics.get('per_subgoal') or {}).items()):
        print('    - {} [{}] {}/{} achieved; {} remaining'.format(
            goal_key, 'coop' if entry.get('is_coop_subgoal') else 'ordinary',
            entry.get('achieved', 0), entry.get('required', 0), entry.get('unsatisfied', 0)))
    if failure.get('invalid_reason_counts'):
        print('  Invalid reasons: ' + _format_count_map(failure['invalid_reason_counts']))


def _print_table(title, headers, rows):
    print('  ' + title)
    widths = [max(len(str(row[index])) for row in [headers] + rows)
              for index in range(len(headers))]
    for index, row in enumerate([headers] + rows):
        print('    ' + ' | '.join(str(cell).ljust(width) for cell, width in zip(row, widths)))
        if index == 0:
            print('    ' + '-+-'.join('-' * width for width in widths))


def print_experiment_metrics(metrics, prefix='C-WAH Experiment Metrics'):
    """Print nine strata for success, efficiency and token cost."""
    print('=== {} ==='.format(prefix))
    print('  Comm model: {}'.format(metrics.get('communication_model', {})))
    print('  Coverage: valid {} / records {} | invalid metric records {}'.format(
        metrics.get('valid_episode_count', 0), metrics.get('episode_record_count', 0),
        metrics.get('invalid_metric_record_count', 0)))
    print('  Unit: episode-run; SR=micro; efficiency=per-episode mean; tokens=sum and per-episode mean')
    print('  Groups: task type x task outcome (NOT subgoal outcome); success/failure Task SR is conditional')
    print('  Comm=agent-sum workload; step-sum=max send time per step; extra=execution-physical-idle')
    summaries = metrics.get('numeric_summaries') or {}
    budget = summaries.get('max_embodied_execution_time') or {}
    print('  Budget(s): ' + _format_summary(budget))
    if budget.get('min') != budget.get('max'):
        print('  WARNING: mixed time budgets; normalized TGC-AUC is not a common-budget comparison')
    groups = metrics.get('groups')
    if not groups:
        print('  Nine-group tables unavailable: re-evaluate episode records with the current evaluator')
        return
    ordered = [(task + '/' + outcome, groups[task][outcome])
               for task in TASK_GROUPS for outcome in OUTCOME_GROUPS]
    success_rows, efficiency_rows, communication_rows, token_rows = [], [], [], []
    for label, group in ordered:
        n = group['episode_count']
        rates = group['success_rates']
        stats = group['numeric_summaries']
        success_rows.append([label, n] + [
            _format_rate(rates[key]) for key in
            ('task_success_rate', 'subgoal_success_rate',
             'ordinary_subgoal_success_rate', 'coop_subgoal_success_rate')])
        efficiency_rows.append([label, n] + [
            _format_value(stats[key]['mean'], 4 if key == 'tgc_auc' else 3)
            for key in ('embodied_execution_time', 'physical_execution_time', 'tgc_auc', 'num_steps')])
        communication_rows.append([label, n] + [
            _format_value(stats[key]['mean']) for key in
            ('communication_total_time_agent_sum', 'communication_time_step_sum', 'communication_extra_time', 'message_count')])
        token_rows.append([label, n] + [
            _format_value(stats[key]['sum'], 0) if stats[key]['count'] else 'N/A'
            for key in ('llm_prompt_tokens', 'llm_completion_tokens', 'llm_total_tokens')] + [
            _format_value(stats[key]['mean'], 2) for key in
            ('llm_prompt_tokens', 'llm_completion_tokens', 'llm_total_tokens')])
    _print_table('Success rates (numerator / denominator):',
                 ['Group', 'N', 'Task SR', 'Subgoal SR', 'Ordinary subgoal SR', 'Coop subgoal SR'], success_rows)
    _print_table('Execution efficiency (means; time in seconds):',
                 ['Group', 'N', 'Execution', 'Physical', 'TGC-AUC', 'Steps'], efficiency_rows)
    _print_table('Communication (means; time in seconds):',
                 ['Group', 'N', 'End-to-end(agent-sum)', 'Step-sum', 'Extra(exec-physical-idle)', 'Messages'], communication_rows)
    _print_table('LLM tokens (both agents, all calls):',
                 ['Group', 'N', 'Input sum', 'Output sum', 'Total sum',
                  'Input mean', 'Output mean', 'Total mean'], token_rows)
    missing = []
    core = ('embodied_execution_time', 'physical_execution_time', 'tgc_auc', 'num_steps',
            'message_send_time_agent_sum',
            'communication_total_time_agent_sum', 'communication_time_step_sum', 'communication_extra_time', 'message_count',
            'llm_prompt_tokens', 'llm_completion_tokens', 'llm_total_tokens')
    for label, group in ordered:
        for key in core:
            count = group['numeric_summaries'][key]['count']
            if count < group['episode_count']:
                missing.append('{}:{} n={}/{}'.format(label, key, count, group['episode_count']))
    if missing:
        print('  WARNING: incomplete metric samples (means use available records): ' + '; '.join(missing))
    print('  Auxiliary:')
    print('    Task types: ' + _format_count_map(metrics.get('task_type_counts')))
    print('    Time source: {}'.format(_format_count_map(metrics.get('physical_time_source_counts'))))
    for key in ('physical_action_frames', 'message_count', 'idle_step_count', 'idle_execution_time', 'nonidle_execution_time', 'nonphysical_extra_time', 'communication_chars', 'message_truncated_count',
                'communication_critical_path_overhead', 'failed_exec_count',
                'invalid_action_count', 'coop_event_total', 'coop_failure_event_total',
                'llm_call_count',
                'llm_wall_time', 'llm_inference_time_estimated'):
        print('    {}: {}'.format(key, _format_summary(summaries.get(key))))
    print('    Termination: ' + _format_count_map(metrics.get('termination_reason_counts')))
    for key, values in sorted((metrics.get('event_counts') or {}).items()):
        print('    {}: {}'.format(key, _format_count_map(values)))
# ==================== End Metrics: Experiment-Level Aggregation ====================
