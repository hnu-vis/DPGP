"""Formal experiment entry for the C-WAH symbolic benchmark.

This runner follows the original C-WAH experiment flow. ArenaMP owns reset and
environment stepping; this file only builds the two method agents, selects
episodes, controls seeds, and saves episode/experiment results.
"""

import json
import pickle
import random
import sys
from pathlib import Path

import numpy as np


CURRENT_DIR = Path(__file__).resolve().parent
CWAH_DIR = CURRENT_DIR.parent
PROJECT_ROOT = CWAH_DIR.parent
if str(CWAH_DIR) not in sys.path:
    sys.path.append(str(CWAH_DIR))

from algos.arena_mp2 import ArenaMP  # noqa: E402
from arguments import (  # noqa: E402
    AGENT_GOALS,
    DEBUG,
    NO_GRAPHICS,
    NUMBER_OF_AGENTS,
    OBSERVATION_TYPE,
    USE_EDITOR,
    get_args,
)
from envs.unity_environment import UnityEnvironment  # noqa: E402
from eval.metrics_forced_coop import (  # noqa: E402
    aggregate_experiment_metrics,
    build_episode_metrics,
    print_episode_metrics,
    print_experiment_metrics,
)


def build_agents(args):
    """Build two independent DPGP agents."""
    if args.method_name == "dpgp":
        from agents import DPGP

        agent_class = DPGP
    else:
        raise ValueError("Unsupported formal method: {}".format(args.method_name))

    agent_arguments = [
        {"agent_id": 1, "char_index": 0, "args": args},
        {"agent_id": 2, "char_index": 1, "args": args},
    ]
    return [
        lambda _arena_id, _env, kwargs=kwargs: agent_class(**kwargs)
        for kwargs in agent_arguments
    ]


def select_episodes(requested_episodes, dataset_size):
    """Use all dataset episodes unless --test_task explicitly selects some."""
    if requested_episodes is None:
        return list(range(dataset_size))

    episode_ids = list(dict.fromkeys(requested_episodes))
    invalid_ids = [
        episode_id
        for episode_id in episode_ids
        if episode_id < 0 or episode_id >= dataset_size
    ]
    if invalid_ids:
        raise ValueError(
            "Invalid --test_task episode ids for dataset size {}: {}".format(
                dataset_size, invalid_ids
            )
        )
    return episode_ids


def json_safe(value):
    """Convert result summaries to values accepted by json.dump."""
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def validate_existing_record(saved_info, args, log_path, expected_seed=None):
    """Reject incomplete evidence or a different experiment before resuming."""
    metrics = build_episode_metrics(saved_info)
    execution = metrics['execution_time']
    expected = {
        'method_name': args.method_name,
        'llm_model': args.lm_id,
        'prediction_mode': args.dpgp_prediction_mode,
        'method_native_config': args.method_native_config,
        'max_embodied_execution_time': args.max_embodied_execution_time,
        'max_episode_length': args.max_episode_length,
        'communication_count_unit': 'unicode_codepoint',
        'communication_send_chars_per_second': args.communication_send_chars_per_second,
        'communication_mode': args.communication_mode,
        'communication_max_message_chars': args.communication_max_message_chars,
        'minimum_step_frames': args.minimum_step_frames,
    }
    if expected_seed is not None:
        expected['seed'] = expected_seed
    if any(key not in saved_info or saved_info[key] != value for key, value in expected.items()):
        raise ValueError('Existing episode has different or missing experiment settings: {}. '
                         'Use --existing-result-policy overwrite or a new --mode.'.format(log_path))
    if (execution['physical_time_source'] != args.physical_time_source
            or execution['physical_time_parameters'] != args.physical_time_parameters):
        raise ValueError('Existing episode has different physical time settings: {}'.format(log_path))

# ==================== Benchmark: Resume Limit Preflight ====================
# Purpose: Reject incompatible cached episodes before rewriting configuration or starting Unity.
def preflight_existing_records(args, env_task_set, test_episodes, record_dir):
    if args.existing_result_policy != 'skip':
        return
    for run_id in range(args.num_runs):
        for episode_id in test_episodes:
            task = env_task_set[episode_id]
            log_path = record_dir / 'logs_agent_{}_{}_{}.pik'.format(
                task['task_id'], task['task_name'], run_id)
            if log_path.is_file():
                with log_path.open('rb') as stream:
                    saved_info = pickle.load(stream)
                validate_existing_record(saved_info, args, log_path,
                                         args.seed + run_id * len(env_task_set) + episode_id)
# ==================== End Benchmark: Resume Limit Preflight ====================


def save_episode_result(
    path, episode_id, run_id, episode_seed, steps, saved_info
):
    result = {
        "method": saved_info.get("method_name"),
        "llm_model": saved_info["llm_model"],
        "prediction_mode": saved_info["prediction_mode"],
        "method_native_config": saved_info["method_native_config"],
        "episode_id": episode_id,
        "run_id": run_id,
        "seed": episode_seed,
        "task_id": saved_info.get("task_id"),
        "task_name": saved_info.get("task_name"),
        "finished": bool(saved_info.get("finished")),
        "steps": steps,
        "metrics": saved_info.get("metrics", {}),
    }
    with path.open("w", encoding="utf-8") as stream:
        json.dump(json_safe(result), stream, ensure_ascii=False, indent=2)


def print_episode_result(episode_id, run_id, saved_info):
    metrics = saved_info.get("metrics")
    if metrics:
        print_episode_metrics(
            metrics,
            episode_id=episode_id,
            run_id=run_id,
            task_name=saved_info.get("task_name"),
        )


def main():
    args = get_args()
    args.communication_count_unit = "unicode_codepoint"

    with open(args.dataset_path, "rb") as stream:
        env_task_set = pickle.load(stream)
    if not isinstance(env_task_set, list) or not env_task_set:
        raise ValueError("--dataset_path must contain a non-empty task list")

    test_episodes = select_episodes(args.test_task, len(env_task_set))
    record_dir = PROJECT_ROOT / "test_results" / args.mode
    record_dir.mkdir(parents=True, exist_ok=True)
    args.record_dir = str(record_dir)
    preflight_existing_records(args, env_task_set, test_episodes, record_dir)
    with (record_dir / "experiment_config.json").open(
        "w", encoding="utf-8"
    ) as stream:
        json.dump(json_safe(vars(args)), stream, ensure_ascii=False, indent=2)

    executable_args = {
        "file_name": args.executable_file,
        "no_graphics": NO_GRAPHICS,
    }

    def env_fn(env_id):
        return UnityEnvironment(
            num_agents=NUMBER_OF_AGENTS,
            max_episode_length=args.max_episode_length,
            port_id=env_id,
            env_task_set=env_task_set,
            agent_goals=list(AGENT_GOALS),
            observation_types=[OBSERVATION_TYPE] * NUMBER_OF_AGENTS,
            use_editor=USE_EDITOR,
            executable_args=executable_args,
            base_port=args.base_port,
            communication_mode=args.communication_mode,
            coop_action_interface=args.coop_action_interface,
            communication_max_message_chars=args.communication_max_message_chars,
            communication_send_chars_per_second=args.communication_send_chars_per_second,
            minimum_step_frames=args.minimum_step_frames,
            physical_action_render_mode=args.physical_action_render_mode,
            physical_action_frame_rate=args.physical_action_frame_rate,
            max_embodied_execution_time=args.max_embodied_execution_time,
            seed=args.seed,
        )

    arena = ArenaMP(
        0,
        env_fn,
        build_agents(args),
        args.record_dir,
        DEBUG,
        reset_max_attempts=args.reset_max_attempts,
    )

    successes = {episode_id: [] for episode_id in test_episodes}
    lengths = {episode_id: [] for episode_id in test_episodes}
    test_results = {}
    all_episode_metrics = []
    run_metrics = {}
    run_outcomes = {}
    overall_metrics = aggregate_experiment_metrics([])

    try:
        for run_id in range(args.num_runs):
            failed_tasks = []
            finished_steps = []
            run_episode_metrics = []

            for episode_id in test_episodes:
                task = env_task_set[episode_id]
                file_stem = "agent_{}_{}_{}".format(
                    task["task_id"], task["task_name"], run_id
                )
                log_path = record_dir / ("logs_{}.pik".format(file_stem))
                result_path = record_dir / ("result_{}.json".format(file_stem))
                episode_seed = (
                    args.seed + run_id * len(env_task_set) + episode_id
                )

                if (
                    args.existing_result_policy == "skip"
                    and log_path.is_file()
                ):
                    with log_path.open("rb") as stream:
                        saved_info = pickle.load(stream)
                    validate_existing_record(saved_info, args, log_path,
                                         args.seed + run_id * len(env_task_set) + episode_id)
                    success = bool(saved_info.get("finished"))
                    steps = max(
                        len(saved_info.get("action", {}).get(0, [])),
                        len(saved_info.get("action", {}).get(1, [])),
                    )
                    print("skip existing episode:", episode_id, "run:", run_id)
                else:
                    random.seed(episode_seed)
                    np.random.seed(episode_seed)
                    if hasattr(arena.env, "rnd"):
                        arena.env.rnd.seed(episode_seed)
                    for agent_index, agent in enumerate(arena.agents):
                        agent.seed = episode_seed + agent_index

                    print("episode:", episode_id, "run:", run_id)
                    arena.reset(episode_id)
                    success, steps, saved_info = arena.run()
                    saved_info["method_name"] = args.method_name
                    saved_info["llm_model"] = args.lm_id
                    saved_info["prediction_mode"] = args.dpgp_prediction_mode
                    saved_info["method_native_config"] = dict(args.method_native_config)
                    saved_info["seed"] = episode_seed

                    with log_path.open("wb") as stream:
                        pickle.dump(saved_info, stream)

                # Recompute metrics from raw telemetry, including skipped records.
                try:
                    episode_metrics = build_episode_metrics(saved_info)
                except Exception as exc:
                    episode_metrics = {"error": repr(exc)}
                saved_info["metrics"] = episode_metrics
                print_episode_result(episode_id, run_id, saved_info)
                run_episode_metrics.append(episode_metrics)
                all_episode_metrics.append(episode_metrics)

                # Refresh the JSON summary without mutating existing pickle logs.
                save_episode_result(
                    result_path,
                    episode_id,
                    run_id,
                    episode_seed,
                    steps,
                    saved_info,
                )

                successes[episode_id].append(1 if success else 0)
                lengths[episode_id].append(steps)
                test_results[episode_id] = {
                    "S": successes[episode_id],
                    "L": lengths[episode_id],
                }
                if success:
                    finished_steps.append(steps)
                else:
                    failed_tasks.append(episode_id)

            average_steps = (
                float(np.mean(finished_steps)) if finished_steps else None
            )
            run_summary = aggregate_experiment_metrics(run_episode_metrics)
            run_metrics[str(run_id)] = run_summary
            run_outcomes[str(run_id)] = {
                "successful_tasks": [
                    episode_id
                    for episode_id in test_episodes
                    if episode_id not in failed_tasks
                ],
                "failed_tasks": list(failed_tasks),
                "average_steps_successful_only": average_steps,
            }
            overall_metrics = aggregate_experiment_metrics(all_episode_metrics)

            with (record_dir / "results.pik").open("wb") as stream:
                pickle.dump(test_results, stream)
            with (record_dir / "eval_result.json").open(
                "w", encoding="utf-8"
            ) as stream:
                json.dump(
                    json_safe({
                        "method": args.method_name,
                        "llm_model": args.lm_id,
                        "completed_run_count": run_id + 1,
                        "episode_results": test_results,
                        "run_outcomes": run_outcomes,
                        "run_metrics": run_metrics,
                        "metrics": overall_metrics,
                    }),
                    stream,
                    ensure_ascii=False,
                    indent=2,
                )
        print_experiment_metrics(overall_metrics)
    finally:
        arena.close()


if __name__ == "__main__":
    main()
