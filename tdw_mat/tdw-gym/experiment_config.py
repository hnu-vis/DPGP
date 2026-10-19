"""Record public experiment configuration and preflight safe result reuse."""

import hashlib
import json
from dataclasses import asdict
from pathlib import Path


from agents.dpgp.config import MethodConfig
from benchmark_config import (
    COMMUNICATION_MODE, COOP_ACTION_INTERFACE, COMMUNICATION_MAX_MESSAGE_CHARS,
    COMMUNICATION_SEND_CHARS_PER_SECOND, MINIMUM_STEP_FRAMES,
    SCREEN_SIZE, GT_MASK, NUMBER_OF_AGENTS, PHYSICAL_ACTION_FRAME_RATE, COMMUNICATION_COUNT_UNIT,
    PHYSICAL_TIME_SOURCE,
)


def _build_method_config(args):
    if args.method_name != "dpgp":
        raise ValueError("unsupported method_name: {}".format(args.method_name))
    # Normalize tuple-valued fields before both persistence and reuse checks.
    config = json.loads(json.dumps(asdict(MethodConfig(
        message_limit=COMMUNICATION_MAX_MESSAGE_CHARS,
    ))))
    config.update(
        name="dpgp", source="openai", lm_id=args.lm_id,
        t=args.t, top_p=args.top_p, max_tokens=args.max_tokens, n=args.n,
        prediction_mode=args.dpgp_prediction_mode,
    )
    return config


def build_experiment_config(args, data):
    benchmark = {
        key: getattr(args, key)
        for key in ("max_embodied_execution_time", "max_steps", "max_frames")
    }
    benchmark.update(
        communication_mode=COMMUNICATION_MODE,
        coop_action_interface=COOP_ACTION_INTERFACE,
        communication_max_message_chars=COMMUNICATION_MAX_MESSAGE_CHARS,
        communication_send_chars_per_second=COMMUNICATION_SEND_CHARS_PER_SECOND,
        minimum_step_frames=MINIMUM_STEP_FRAMES,
    )
    benchmark.update({
        "dataset_sha256": hashlib.sha256(json.dumps(data, sort_keys=True, separators=(",", ":")).encode()).hexdigest(),
        "physical_time_source": PHYSICAL_TIME_SOURCE,
        "communication_count_unit": COMMUNICATION_COUNT_UNIT,
        "physical_action_frame_rate": PHYSICAL_ACTION_FRAME_RATE,
        "screen_size": SCREEN_SIZE, "gt_mask": GT_MASK,
        "number_of_agents": NUMBER_OF_AGENTS,
    })
    episodes = list(range(len(data))) if args.eval_episodes[0] == -1 else list(args.eval_episodes)
    if len(episodes) != len(set(episodes)) or any(value < 0 or value >= len(data) for value in episodes):
        raise ValueError("eval_episodes must contain unique valid dataset indices")
    return {
        "method": args.method_name, "run_id": args.run_id,
        "eval_episodes": episodes, "dataset_task_count": len(data),
        "data_prefix": str(Path(args.data_prefix).resolve()), "data_path": args.data_path,
        "execution_config": {
            "num_shards": int(getattr(args, "num_shards", 1)),
            "experiment_seed": int(getattr(args, "experiment_seed", 1024)),
            "infrastructure_error_policy": getattr(
                args, "infrastructure_error_policy", "stop_shard"
            ),
        },
        "effective_config": {
            "benchmark": benchmark,
            "method": _build_method_config(args),
            "artifacts": {
                "save_agent_images": getattr(args, "save_images", False),
                "save_top_down_images": getattr(
                    args, "save_top_down_images", False
                ),
            },
        },
    }


def load_existing_episode_result(directory, config):
    """Return a validated reusable result, or None for an incomplete/error attempt.

    Configuration mismatches are never treated as resumable crash debris. They
    remain a hard error so that a formal run can't silently mix experiments.
    """
    directory = Path(directory)
    attempt_path = directory / "attempt_status.json"
    if attempt_path.exists():
        try:
            attempt = json.loads(attempt_path.read_text())
        except (OSError, ValueError, TypeError):
            return None
        if attempt.get("status") != "complete":
            return None
    result_path = directory / "result_episode.json"
    if not result_path.exists():
        return None
    try:
        result = json.loads(result_path.read_text())
    except (OSError, ValueError, TypeError):
        return None
    if (result.get("metrics") or {}).get("error"):
        return None
    effective = config["effective_config"]
    result_effective = result.get("effective_config")
    if result_effective is not None and result_effective != effective:
        raise ValueError(
            "episode {} configuration differs; use a new run_id or overwrite".format(
                directory.name
            )
        )
    if result_effective is None:
        return None
    metadata_path = directory / "episode_metadata.json"
    trace_path = directory / "episode_trace.json"
    if not metadata_path.exists() or not trace_path.exists():
        return None
    try:
        metadata = json.loads(metadata_path.read_text())
    except (OSError, ValueError, TypeError):
        return None
    metadata_effective = metadata.get("effective_config")
    if metadata_effective is not None and metadata_effective != effective:
        raise ValueError(
            "episode {} configuration differs; use a new run_id or overwrite".format(
                directory.name
            )
        )
    if metadata_effective is None:
        return None
    benchmark = effective["benchmark"]
    for key in ("max_embodied_execution_time", "max_steps", "max_frames"):
        if key not in metadata:
            return None
        if metadata[key] != benchmark[key]:
            raise ValueError(
                "episode {} has a different {} setting".format(
                    directory.name, key
                )
            )
    try:
        from eval.evaluation_records import load_episode_record

        load_episode_record(directory)
    except (OSError, ValueError, TypeError, KeyError, AttributeError):
        return None
    return result


def preflight_existing_results(output_dir, config, policy):
    """Run before writing configuration or starting TDW/LLM dependencies."""
    if policy != "skip":
        return
    root = Path(output_dir)
    config_path = root / "experiment_config.json"
    if config_path.exists():
        previous = json.loads(config_path.read_text())
        if previous != config:
            raise ValueError(
                "existing experiment configuration differs; use a new run_id "
                "or overwrite"
            )
    for episode in config["eval_episodes"]:
        directory = root / str(episode)
        load_existing_episode_result(directory, config)
