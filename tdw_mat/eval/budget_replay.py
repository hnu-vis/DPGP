"""Re-evaluate whole recorded TDW steps under a shorter time budget."""

import sys
from pathlib import Path

TDW_GYM_DIR = Path(__file__).resolve().parents[1] / "tdw-gym"
if str(TDW_GYM_DIR) not in sys.path:
    sys.path.append(str(TDW_GYM_DIR))

if __package__:
    from .evaluation_records import validate_record
else:
    from evaluation_records import validate_record

from episode_limits import positive_finite, resolve_episode_termination


def replay_budget_prefix(record, budget):
    validate_record(record)
    budget = positive_finite(budget)
    collection = record["metadata"]
    collection_budget = positive_finite(collection["max_embodied_execution_time"])
    if budget > collection_budget:
        raise ValueError("evaluation budget exceeds collection budget; unseen behavior cannot be inferred")
    elapsed, reason, success = 0.0, None, False
    retained = []
    for source in record["trace"]:
        row = dict(source)
        row["execution_time_step"] = dict(source["execution_time_step"], max_embodied_execution_time=budget)
        elapsed += row["execution_time_step"]["step_embodied_execution_time"]
        retained.append(row)
        done, success, reason = resolve_episode_termination(
            row["finish"][0] == row["finish"][1], elapsed, budget, len(retained),
        )
        row["task_success_within_time"] = success
        row["termination_reason"] = reason
        if done:
            break
    if reason is None:
        raise ValueError("trajectory stops before evaluation budget without success; requested prefix is censored")
    replayed = {
        "metadata": dict(collection, max_embodied_execution_time=budget, max_steps=None, max_frames=None),
        "trace": retained,
        "final": {
            "finish": retained[-1]["finish"], "completed_ids": retained[-1]["satisfied_target_ids"],
            "frames": retained[-1]["frame"], "num_steps": len(retained),
            "task_success_within_time": success, "termination_reason": reason,
        },
    }
    metadata = {
        "mode": "trajectory_prefix", "collection_time_budget": collection_budget,
        "evaluation_time_budget": budget,
        "collection_step_limit": collection["max_steps"], "collection_frame_limit": collection["max_frames"],
        "collection_termination_reason": record["final"]["termination_reason"],
        "collection_num_steps": len(record["trace"]), "retained_num_steps": len(retained),
        "retained_execution_time": elapsed,
        "boundary_policy": "retain complete crossing step and all costs; only AUC clips to budget",
        "policy_assumption": "method decisions do not depend on maximum/remaining episode budget",
    }
    return replayed, metadata
