"""Dependency-free TDW episode limits shared by execution and budget replay."""

import argparse
import math


def positive_finite(value):
    if isinstance(value, bool):
        raise ValueError("time budget must be a finite positive number")
    value = float(value)
    if not math.isfinite(value) or value <= 0:
        raise ValueError("time budget must be a finite positive number")
    return value


def optional_positive_int(value):
    if value is None:
        return None
    if isinstance(value, bool) or int(value) != float(value) or int(value) <= 0:
        raise ValueError("episode limit must be a positive integer or None")
    return int(value)


def add_budget_arguments(parser):
    def budget_type(value):
        try:
            return positive_finite(value)
        except (ValueError, TypeError) as error:
            raise argparse.ArgumentTypeError(str(error)) from error

    parser.add_argument("--max_frames", "--max-frames", default=None, type=int)
    parser.add_argument("--max_steps", "--max-steps", default=None, type=int)
    parser.add_argument(
        "--max_embodied_execution_time", "--max-embodied-execution-time",
        default=20.0, type=budget_type,
    )


def resolve_episode_termination(goal_done, elapsed_time, time_budget, steps,
                                step_limit=None, frames=0, frame_limit=None):
    success = bool(goal_done) and (time_budget is None or elapsed_time <= time_budget)
    if success:
        return True, True, "task_success"
    if time_budget is not None and elapsed_time >= time_budget:
        return True, False, "embodied_time_limit"
    if frame_limit is not None and frames >= frame_limit:
        return True, False, "frame_limit"
    if step_limit is not None and steps >= step_limit:
        return True, False, "step_limit"
    return bool(goal_done), False, None
