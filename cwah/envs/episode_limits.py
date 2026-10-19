"""Lightweight termination semantics shared by C-WAH execution and replay."""

# ==================== Benchmark: Optional Episode Limits ====================
# Purpose: Preserve success-first, then time, then explicitly enabled step limits.
def below_optional_limit(count, limit):
    return limit is None or count < limit


def resolve_episode_termination(goal_done, elapsed_time, time_budget, steps, step_limit=None):
    success = bool(goal_done) and (time_budget is None or elapsed_time <= time_budget)
    if success:
        return True, True, 'task_success'
    if time_budget is not None and elapsed_time >= time_budget:
        return True, False, 'embodied_time_limit'
    if not below_optional_limit(steps, step_limit):
        return True, False, 'step_limit'
    # A raw goal completed beyond the budget also reaches the time-limit branch.
    return bool(goal_done), False, None
# ==================== End Benchmark: Optional Episode Limits ====================
