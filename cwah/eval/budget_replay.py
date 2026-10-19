"""Re-score recorded C-WAH trajectories under a shorter embodied-time budget."""

import math

from envs.episode_limits import resolve_episode_termination

if __package__:
    from .metrics_forced_coop import _completion_fraction
else:
    from metrics_forced_coop import _completion_fraction


# ==================== Metrics: Budget Prefix Replay ====================
# Purpose: Retain whole steps and their costs, never synthesize unseen behavior.
STEP_FIELDS = (
    'progress', 'coop', 'physical_action_frames', 'execution_time_steps',
    'failed_exec', 'failed_exec_reasons', 'failed_exec_details',
    'action_feedback', 'action_status', 'invalid_reasons',
    'coop_action_feedback', 'communication', 'goals_finished',
)


def validate_evaluation_budget(budget):
    if isinstance(budget, bool):
        raise ValueError('evaluation time budget must be a finite positive number')
    try:
        budget = float(budget)
    except (TypeError, ValueError) as exc:
        raise ValueError('evaluation time budget must be a finite positive number') from exc
    if not math.isfinite(budget) or budget <= 0:
        raise ValueError('evaluation time budget must be a finite positive number')
    return budget


def replay_budget_prefix(raw, budget):
    budget = validate_evaluation_budget(budget)
    if 'max_episode_length' not in raw:
        raise ValueError('missing collection step limit evidence')
    collection_budget = validate_evaluation_budget(raw.get('max_embodied_execution_time'))
    if budget > collection_budget:
        raise ValueError('evaluation budget {} exceeds collection budget {}; unseen behavior cannot be inferred'.format(
            budget, collection_budget))
    steps = raw.get('execution_time_steps')
    if not isinstance(steps, list) or not steps:
        raise ValueError('budget replay requires nonempty per-step execution time')
    count = len(steps)
    if not isinstance(raw.get('initial_progress'), dict) or 'unsatisfied' not in raw['initial_progress']:
        raise ValueError('budget replay requires initial goal progress for exact AUC')
    for field in ('progress', 'coop', 'physical_action_frames'):
        if not isinstance(raw.get(field), list) or len(raw[field]) != count:
            raise ValueError('budget replay requires step-aligned ' + field)
    for field in ('action', 'LLM'):
        records = raw.get(field)
        if not isinstance(records, dict) or len(records) != 2:
            raise ValueError('budget replay requires both agents in ' + field)
        if any(not isinstance(values, list) or len(values) != count for values in records.values()):
            raise ValueError('budget replay requires step-aligned ' + field)
    if set(raw['action']) != set(raw['LLM']):
        raise ValueError('budget replay requires matching action/LLM agent identities')
    if any(not isinstance(record, dict) or 'llm_call_count' not in record
           for records in raw['LLM'].values() for record in records):
        raise ValueError('budget replay requires explicit per-step LLM call evidence')
    if any(not isinstance(frame, dict) for frame in raw['physical_action_frames']):
        raise ValueError('budget replay requires valid physical time step records')
    for field in STEP_FIELDS:
        values = raw.get(field)
        if values is not None and (not isinstance(values, list) or (values and len(values) != count)):
            raise ValueError('budget replay cannot align auxiliary field ' + field)
    if not raw.get('gt_goals'):
        raise ValueError('budget replay requires nonempty goal evidence')
    elapsed = 0.0
    retained = 0
    success = False
    reason = None
    for index, step in enumerate(steps):
        if not isinstance(step, dict):
            raise ValueError('invalid execution time step at index {}'.format(index))
        duration = step.get('step_embodied_execution_time')
        if duration is None:
            raise ValueError('missing execution duration at index {}'.format(index))
        duration = float(duration)
        if not math.isfinite(duration) or duration < 0:
            raise ValueError('invalid execution duration at index {}'.format(index))
        if not isinstance(raw['progress'][index], dict) or 'unsatisfied' not in raw['progress'][index]:
            raise ValueError('missing goal progress at index {}'.format(index))
        elapsed += duration
        retained = index + 1
        goal_done = _completion_fraction(raw, raw['progress'][index]) == 1.0
        # A shorter-budget replay has no synthetic step limit. A source step
        # cap before this boundary is incomplete evidence, not a time failure.
        done, success, reason = resolve_episode_termination(goal_done, elapsed, budget, retained)
        if done:
            break
    if reason is None:
        raise ValueError('trajectory stops at {:.6f} s before evaluation budget {} without success '
                         '(source termination={}); the requested prefix is censored'.format(
                             elapsed, budget, raw.get('termination_reason')))
    replayed = dict(raw)
    replayed.pop('metrics', None)
    for field in STEP_FIELDS:
        if field in raw:
            replayed[field] = raw[field][:retained]
    for field in ('action', 'LLM'):
        replayed[field] = {agent: values[:retained] for agent, values in raw[field].items()}
    # Copy step metadata before changing the budget; source objects remain intact.
    replayed['execution_time_steps'] = [dict(step, max_embodied_execution_time=budget)
                                       for step in steps[:retained]]
    replayed['max_embodied_execution_time'] = budget
    replayed['finished'] = success
    replayed['task_success_within_time'] = success
    replayed['termination_reason'] = reason
    replayed['max_episode_length'] = None
    metadata = {
        'mode': 'trajectory_prefix', 'collection_time_budget': collection_budget,
        'evaluation_time_budget': budget,
        'collection_termination_reason': raw.get('termination_reason'),
        'collection_step_limit': raw.get('max_episode_length'),
        'collection_num_steps': count, 'retained_num_steps': retained,
        'retained_execution_time': elapsed,
        'boundary_policy': 'retain complete crossing step; AUC alone clips to budget',
        'policy_assumption': 'method decisions do not depend on the maximum/remaining experiment budget',
    }
    return replayed, metadata
# ==================== End Metrics: Budget Prefix Replay ====================
