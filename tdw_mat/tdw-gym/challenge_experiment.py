"""TDW forced-cooperation experiment entry point."""

from __future__ import annotations

import argparse
from contextlib import nullcontext
import json
import logging
import os
import sys


THIS_DIR = os.path.dirname(os.path.abspath(__file__))
TDW_MAT_DIR = os.path.abspath(os.path.join(THIS_DIR, ".."))
TDW_GYM_DIR = os.path.join(TDW_MAT_DIR, "tdw-gym")
# Resolve this benchmark's modules before unrelated same-name installed packages.
for path in [TDW_MAT_DIR, THIS_DIR]:
    if path in sys.path:
        sys.path.remove(path)
    sys.path.insert(0, path)

import gym  # noqa: E402

from episode_limits import add_budget_arguments, optional_positive_int
from experiment_config import (
    build_experiment_config,
    load_existing_episode_result,
    preflight_existing_results,
)
from execution_control import (
    EpisodeClaim,
    atomic_write_json,
    create_or_validate_json,
    seed_episode_randomness,
    select_shard_episodes,
    validate_shard_arguments,
)
from benchmark_config import (
    COMMUNICATION_MODE, COOP_ACTION_INTERFACE, COMMUNICATION_MAX_MESSAGE_CHARS,
    COMMUNICATION_SEND_CHARS_PER_SECOND, MINIMUM_STEP_FRAMES,
    SCREEN_SIZE, GT_MASK, NUMBER_OF_AGENTS,
)
from eval.evaluation_records import load_episode_record
from eval.llm_usage import delta_llm_stats, step_llm_stats

from eval.metrics_forced_coop import (  # noqa: E402
    aggregate_experiment_metrics,
    build_record_metrics,
    make_episode_metadata,
    json_safe,
    print_episode_metrics,
    print_experiment_metrics,
)


LAUNCH_BUILD = True
gym.envs.registration.register(
    id="transport_challenge_MA",
    entry_point="tdw_gym:TDW",
)


class EpisodeInfrastructureError(RuntimeError):
    def __init__(self, kind, error, steps=0, frames=0):
        self.kind = str(kind)
        self.original_type = type(error).__name__
        self.steps = int(steps)
        self.frames = int(frames)
        super().__init__(str(error))


class ShardInfrastructureAbort(RuntimeError):
    """Stop one worker after preserving an invalid infrastructure attempt."""


def build_agents(args, logger):
    """Build the two DPGP instances using the fixed Benchmark settings."""
    if args.method_name != "dpgp":
        raise ValueError("Unsupported method: {}".format(args.method_name))
    from agents.dpgp import DPGP

    return [
        DPGP(
            agent_id,
            logger,
            args.max_frames,
            args.output_dir,
            llm_model=args.lm_id,
            llm_temperature=args.t,
            llm_top_p=args.top_p,
            llm_max_tokens=args.max_tokens,
            llm_n=args.n,
            communication_mode=COMMUNICATION_MODE,
            communication_max_message_chars=COMMUNICATION_MAX_MESSAGE_CHARS,
            prediction_mode=args.dpgp_prediction_mode,
        )
        for agent_id in range(NUMBER_OF_AGENTS)
    ]


class Challenge:
    def __init__(self, logger, args, experiment_config):
        self.logger = logger
        self.args = args
        self.experiment_config = experiment_config
        self.effective_config = experiment_config["effective_config"]
        self.output_dir = args.output_dir
        self.summary_dir = getattr(args, "worker_output_dir", self.output_dir)
        with open(os.path.join(args.data_prefix, args.data_path), "r") as stream:
            self.data = json.load(stream)
        self.env = self._make_env()
        self.logger.info("Forced-cooperation environment created")

    def _make_env(self):
        args = self.args
        return gym.make(
            "transport_challenge_MA",
            port=args.port,
            number_of_agents=NUMBER_OF_AGENTS,
            save_dir=self.summary_dir,
            max_frames=args.max_frames,
            max_steps=args.max_steps,
            launch_build=LAUNCH_BUILD,
            screen_size=SCREEN_SIZE,
            data_prefix=args.data_prefix,
            gt_mask=GT_MASK,
            communication_mode=COMMUNICATION_MODE,
            coop_action_interface=COOP_ACTION_INTERFACE,
            communication_max_message_chars=COMMUNICATION_MAX_MESSAGE_CHARS,
            communication_send_chars_per_second=COMMUNICATION_SEND_CHARS_PER_SECOND,
            minimum_step_frames=MINIMUM_STEP_FRAMES,
            max_embodied_execution_time=args.max_embodied_execution_time,
            save_top_down_images=getattr(args, "save_top_down_images", False),
            check_version=False,
        )

    def _restart_env(self):
        old_env = self.env
        self.env = None
        try:
            if old_env is not None:
                old_env.close()
        except Exception:
            self.logger.exception("Failed to close environment during restart")
        self.env = self._make_env()
        self.logger.info("Forced-cooperation environment recreated")

    @staticmethod
    def _method_dependency_errors(agents):
        errors = {}
        for agent_id, agent in enumerate(agents):
            getter = getattr(agent, "get_method_dependency_error", None)
            error = getter() if callable(getter) else None
            if error:
                errors[str(agent_id)] = str(error)
        return errors

    def _write_summary(self, results, episode_metrics, total_finish):
        experiment_metrics = aggregate_experiment_metrics(episode_metrics)
        valid_result_count = sum(
            1
            for result in results.values()
            if not (result.get("metrics") or {}).get("error")
        )
        summary = {
            "episode_results": results,
            "avg_finish": total_finish / max(valid_result_count, 1),
            "metrics": experiment_metrics,
            "method": self.args.method_name,
            "experiment_config": self.experiment_config,
        }
        atomic_write_json(
            os.path.join(self.summary_dir, "eval_result.json"),
            json_safe(summary),
        )
        return summary

    @staticmethod
    def _error_result(episode, error):
        kind = getattr(error, "kind", "episode_runtime_error")
        error_info = {
            "kind": kind,
            "type": getattr(error, "original_type", type(error).__name__),
            "message": str(error),
        }
        metrics = {
            "episode": int(episode),
            "task_success": False,
            "termination_reason": kind,
            "error": error_info,
        }
        return {
            "finish": 0,
            "total": 0,
            "done": False,
            "termination_reason": kind,
            "steps": int(getattr(error, "steps", 0)),
            "frames": int(getattr(error, "frames", 0)),
            "metrics": metrics,
        }

    def _run_episode(self, agents, episode, episode_dir):
        episode_log = []
        trace_path = os.path.join(episode_dir, "episode_trace.json")
        episode_seed = seed_episode_randomness(
            getattr(self.args, "experiment_seed", 1024),
            self.args.run_id,
            episode,
        )
        try:
            try:
                if self.env is None:
                    self.env = self._make_env()
                self.logger.info("Resetting with data: %s", self.data[episode])
                state, info, env_api = self.env.reset(
                    seed=self.data[episode]["seed"],
                    options=self.data[episode],
                    output_dir=episode_dir,
                )
            except Exception as error:
                raise EpisodeInfrastructureError(
                    "environment_reset_error", error
                ) from error
            metadata = make_episode_metadata(self.env, episode, info, self.effective_config)
            metadata.update({
                "run_id": self.args.run_id,
                "method": self.args.method_name,
                "execution_provenance": {
                    "num_shards": int(getattr(self.args, "num_shards", 1)),
                    "shard_id": int(getattr(self.args, "shard_id", 0)),
                    "episode_seed": episode_seed,
                },
            })
            atomic_write_json(
                os.path.join(episode_dir, "episode_metadata.json"), metadata
            )
            before_reset_llm = [agent.get_llm_stats() for agent in agents]
            for agent_id, agent in enumerate(agents):
                curr_api = env_api[agent_id] if isinstance(env_api, list) else env_api
                agent.reset(
                    obs=state[str(agent_id)],
                    goal_objects=info.get("goal_description"),
                    output_dir=episode_dir,
                    env_api=curr_api,
                    rooms_name=info.get("rooms_name"),
                    agent_color=info["agent_colors"][agent_id],
                    agent_id=agent_id,
                    gt_mask=GT_MASK,
                    save_img=self.args.save_images,
                )
            initial_llm_stats = {
                str(agent_id): delta_llm_stats(before_reset_llm[agent_id], agent.get_llm_stats())
                for agent_id, agent in enumerate(agents)
            }

            done = False
            step_num = 0
            local_finish = metadata["initial_finish"]
            dependency_errors = {}

            while not done:
                step_num += 1
                if self.args.save_images:
                    self.env.save_images(os.path.join(episode_dir, "Images"))
                actions = {}
                previous_call_counts = [agent.get_llm_stats().get("llm_call_count") for agent in agents]
                for agent_id, agent in enumerate(agents):
                    actions[str(agent_id)] = agent.act(state[str(agent_id)])
                llm_step = {
                    str(agent_id): step_llm_stats(agent.get_llm_stats(), previous_call_counts[agent_id])
                    for agent_id, agent in enumerate(agents)
                }
                method_diagnostics = {
                    str(agent_id): (
                        agent.get_method_diagnostics()
                        if callable(getattr(agent, "get_method_diagnostics", None))
                        else None
                    )
                    for agent_id, agent in enumerate(agents)
                }
                dependency_errors = self._method_dependency_errors(agents)
                state, reward, done, info = self.env.step(actions)
                if dependency_errors:
                    done = True
                    info = dict(info)
                    info["termination_reason"] = "method_dependency_error"
                    info["method_dependency_errors"] = dependency_errors
                    info["task_success_within_time"] = False
                local_finish = info["goal_progress"]
                record = json_safe({
                    "runner_step": step_num,
                    "env_step": info.get("env_step"),
                    "env_step_index": info.get("env_step_index"),
                    "frame": self.env.num_frames,
                    "actions": actions,
                    "llm_stats": llm_step,
                    "initial_llm_stats": initial_llm_stats if step_num == 1 else None,
                    "method_diagnostics": method_diagnostics,
                    "reward": reward,
                    "finish": local_finish,
                    "satisfied_target_ids": info["satisfied_target_ids"],
                    "task_success_within_time": info.get("task_success_within_time"),
                    "raw_goal_success": info.get("raw_goal_success"),
                    "action_status": info.get("action_status"),
                    "invalid_action_events": info.get("invalid_action_events"),
                    "physical_execution_failure_events": info.get(
                        "physical_execution_failure_events"
                    ),
                    "action_failure_events": info.get("action_failure_events"),
                    "action_feedback": info.get("action_feedback"),
                    "message_feedback": info.get("message_feedback"),
                    "coop_action_events": info.get("coop_action_events"),
                    "coop_feedback_events": info.get("coop_feedback_events"),
                    "obs_coop_action_feedback": {
                        agent_id: state[agent_id].get("coop_action_feedback")
                        for agent_id in state
                    },
                    "obs_grasp_property_feedback": {
                        agent_id: state[agent_id].get("grasp_property_feedback")
                        for agent_id in state
                    },
                    "obs_object_coop_properties": {
                        agent_id: state[agent_id].get("object_coop_properties")
                        for agent_id in state
                    },
                    "obs_class_coop_properties": {
                        agent_id: state[agent_id].get("class_coop_properties")
                        for agent_id in state
                    },
                    "obs_coop_info": {
                        agent_id: state[agent_id].get("coop_info")
                        for agent_id in state
                    },
                    "active_coop_carrying": info.get("active_coop_carrying"),
                    "communication": info.get("communication"),
                    "execution_time_step": info.get("execution_time_step"),
                    "physical_action_frame_info": info.get("physical_action_frame_info"),
                    "elapsed_embodied_execution_time": info.get(
                        "elapsed_embodied_execution_time"
                    ),
                    "termination_reason": info.get("termination_reason"),
                })
                episode_log.append(record)
                self.logger.info(
                    "step=%s frame=%s actions=%s finish=%s failure_reasons=%s coop=%s",
                    step_num,
                    self.env.num_frames,
                    actions,
                    local_finish,
                    {agent_id: feedback.get("reason_code")
                     for agent_id, feedback in (info.get("action_feedback") or {}).items()},
                    info.get("coop_action_events"),
                )
                if done:
                    break

            metrics = build_record_metrics({
                "metadata": metadata, "trace": episode_log,
                "final": {
                    "finish": local_finish, "num_steps": step_num, "frames": self.env.num_frames,
                    "completed_ids": info["satisfied_target_ids"],
                    "task_success_within_time": info.get("task_success_within_time"),
                    "termination_reason": info.get("termination_reason"),
                },
            })
            if dependency_errors:
                metrics["error"] = {
                    "kind": "method_dependency_error",
                    "details": dependency_errors,
                }
            print_episode_metrics(metrics, logger=self.logger)
            return {
                "finish": local_finish[0],
                "total": local_finish[1],
                "done": bool(metrics.get("task_success")),
                "termination_reason": metrics.get("termination_reason"),
                "steps": step_num,
                "frames": self.env.num_frames,
                "metrics": metrics,
                "effective_config": self.effective_config,
            }
        except EpisodeInfrastructureError:
            raise
        except Exception as error:
            raise EpisodeInfrastructureError(
                "episode_runtime_error",
                error,
                steps=len(episode_log),
                frames=(
                    getattr(self.env, "num_frames", 0)
                    if self.env is not None
                    else 0
                ),
            ) from error
        finally:
            atomic_write_json(trace_path, episode_log)

    def submit(self, agents, eval_episodes):
        if eval_episodes[0] == -1:
            eval_episodes = list(range(len(self.data)))
        else:
            eval_episodes = list(eval_episodes)
        results = {}
        total_finish = 0.0
        episode_metrics = []

        for index, episode in enumerate(eval_episodes):
            episode_dir = os.path.join(self.output_dir, str(episode))
            os.makedirs(episode_dir, exist_ok=True)
            self.logger.info("Episode %s (%s/%s)", episode, index + 1, len(eval_episodes))
            claim = (
                EpisodeClaim(
                    episode_dir,
                    episode,
                    getattr(self.args, "shard_id", 0),
                )
                if int(getattr(self.args, "num_shards", 1)) > 1
                else nullcontext()
            )
            with claim:
                result_path = os.path.join(episode_dir, "result_episode.json")
                if self.args.existing_result_policy == "skip":
                    result = load_existing_episode_result(
                        episode_dir, self.experiment_config
                    )
                    if result is not None:
                        result["metrics"] = build_record_metrics(
                            load_episode_record(episode_dir)
                        )
                        atomic_write_json(result_path, json_safe(result))
                        episode_metrics.append(result["metrics"])
                        total_finish += result["finish"] / max(result["total"], 1)
                        results[str(episode)] = result
                        print_episode_metrics(result["metrics"], logger=self.logger)
                        self.logger.info(
                            "Skipping episode %s because a valid existing result was found",
                            episode,
                        )
                        self._write_summary(results, episode_metrics, total_finish)
                        continue
                    if os.path.exists(result_path):
                        self.logger.info(
                            "Re-running episode %s because its previous attempt is "
                            "an error or incomplete record",
                            episode,
                        )

                infrastructure_error = None
                attempt_status_path = os.path.join(
                    episode_dir, "attempt_status.json"
                )
                attempt_status = {
                    "episode": int(episode),
                    "run_id": self.args.run_id,
                    "num_shards": int(getattr(self.args, "num_shards", 1)),
                    "shard_id": int(getattr(self.args, "shard_id", 0)),
                    "status": "running",
                }
                atomic_write_json(attempt_status_path, attempt_status)
                try:
                    result = self._run_episode(agents, episode, episode_dir)
                except Exception as error:
                    infrastructure_error = error
                    self.logger.exception(
                        "Episode %s failed with an infrastructure error", episode
                    )
                    result = self._error_result(episode, error)
                    if (
                        getattr(
                            self.args,
                            "infrastructure_error_policy",
                            "continue",
                        ) == "continue"
                        and index + 1 < len(eval_episodes)
                    ):
                        try:
                            self._restart_env()
                        except Exception:
                            self.logger.exception(
                                "Environment recreation failed; the next episode will retry"
                            )

                metrics = result["metrics"]
                result["effective_config"] = self.effective_config
                episode_metrics.append(metrics)
                if not metrics.get("error"):
                    total_finish += result["finish"] / max(result["total"], 1)
                atomic_write_json(result_path, json_safe(result))
                atomic_write_json(
                    attempt_status_path,
                    dict(
                        attempt_status,
                        status=("failed" if metrics.get("error") else "complete"),
                    ),
                )
                results[str(episode)] = result
                self._write_summary(results, episode_metrics, total_finish)
                if (
                    getattr(
                        self.args,
                        "infrastructure_error_policy",
                        "continue",
                    ) == "stop_shard"
                    and (infrastructure_error is not None or metrics.get("error"))
                ):
                    raise ShardInfrastructureAbort(
                        "stopping shard after infrastructure error in episode {}; "
                        "rerun the same command with existing_result_policy=skip"
                        .format(episode)
                    ) from infrastructure_error

        summary = self._write_summary(results, episode_metrics, total_finish)
        print_experiment_metrics(summary["metrics"], logger=self.logger)
        self.logger.info("Evaluation done: %s", summary)
        return summary

    def close(self):
        if self.env is not None:
            self.env.close()


def init_logs(output_dir, name="forced_coop_experiment"):
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    fh = logging.FileHandler(os.path.join(output_dir, "output.log"))
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(formatter)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)
    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, default=os.path.join(TDW_MAT_DIR, "results"))
    parser.add_argument("--experiment_name", type=str, default="forced_coop_experiment")
    parser.add_argument("--run_id", type=str, default="run_0")
    parser.add_argument(
        "--method_name",
        choices=["dpgp"],
        default="dpgp",
    )
    parser.add_argument("--data_path", type=str, default="test_env.json")
    parser.add_argument("--data_prefix", type=str, default=os.path.join(TDW_MAT_DIR, "dataset", "dataset_test"))
    parser.add_argument("--port", default=10072, type=int)
    parser.add_argument("--eval_episodes", nargs="+", default=[0], type=int)
    parser.add_argument(
        "--num_shards",
        "--num-shards",
        choices=[1, 2, 3],
        default=1,
        type=int,
        help="Number of execution workers in one logical run (1, 2, or 3).",
    )
    parser.add_argument(
        "--shard_id",
        "--shard-id",
        default=0,
        type=int,
        help="Zero-based worker index for matched-pair sharding.",
    )
    parser.add_argument(
        "--experiment_seed",
        "--experiment-seed",
        default=1024,
        type=int,
        help="Base seed for episode-local Python/NumPy randomness.",
    )
    parser.add_argument(
        "--infrastructure_error_policy",
        "--infrastructure-error-policy",
        choices=["continue", "stop_shard"],
        default="stop_shard",
        help=(
            "Stop this worker after preserving an infrastructure error, or "
            "continue to later episodes."
        ),
    )
    add_budget_arguments(parser)
    parser.add_argument(
        "--dpgp_prediction_mode",
        choices=["deterministic_llm", "discovery_probability"],
        default="discovery_probability",
    )
    parser.add_argument("--lm_id", type=str, default="gpt-4o-mini")
    parser.add_argument("--t", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--max_tokens", type=int, default=4096)
    parser.add_argument("--n", type=int, default=1)
    parser.add_argument("--save_images", action="store_true", default=False,
                        help="Save agent RGB/depth images in each episode directory.")
    parser.add_argument(
        "--save_top_down_images",
        action="store_true",
        default=False,
        help=(
            "Enable the dataset diagnostic top-down camera and save one image "
            "per TDW frame (disabled by default)."
        ),
    )
    parser.add_argument(
        "--existing_result_policy",
        choices=["skip", "overwrite"],
        default="skip",
    )
    args = parser.parse_args()
    try:
        validate_shard_arguments(args.num_shards, args.shard_id)
    except ValueError as error:
        parser.error(str(error))
    if args.experiment_seed < 0:
        parser.error("--experiment_seed must be a non-negative integer")
    if args.port <= 0:
        parser.error("--port must be positive")
    for name in ("max_steps", "max_frames"):
        try:
            optional_positive_int(getattr(args, name))
        except ValueError as error:
            parser.error("--{}: {}".format(name, error))

    args.data_prefix = os.path.abspath(os.path.expanduser(args.data_prefix))
    args.data_path = os.path.expanduser(args.data_path)
    args.output_dir = os.path.abspath(os.path.expanduser(args.output_dir))
    args.output_dir = os.path.join(
        args.output_dir, args.experiment_name, args.run_id
    )
    with open(os.path.join(args.data_prefix, args.data_path)) as stream:
        data = json.load(stream)
    experiment_config = build_experiment_config(args, data)
    full_episodes = list(experiment_config["eval_episodes"])
    assigned_episodes = select_shard_episodes(
        data,
        full_episodes,
        args.num_shards,
        args.shard_id,
    )
    if not assigned_episodes:
        parser.error("this shard has no assigned episodes")
    os.makedirs(args.output_dir, exist_ok=True)
    config_path = os.path.join(args.output_dir, "experiment_config.json")
    if args.num_shards > 1:
        create_or_validate_json(config_path, experiment_config)
        preflight_existing_results(
            args.output_dir,
            experiment_config,
            args.existing_result_policy,
        )
        args.worker_output_dir = os.path.join(
            args.output_dir, "shards", "shard_{}".format(args.shard_id)
        )
    else:
        preflight_existing_results(
            args.output_dir,
            experiment_config,
            args.existing_result_policy,
        )
        atomic_write_json(config_path, experiment_config)
        args.worker_output_dir = args.output_dir
    os.makedirs(args.worker_output_dir, exist_ok=True)
    worker_config = {
        "run_id": args.run_id,
        "num_shards": args.num_shards,
        "shard_id": args.shard_id,
        "port": args.port,
        "full_eval_episodes": full_episodes,
        "assigned_episodes": assigned_episodes,
        "experiment_seed": args.experiment_seed,
        "infrastructure_error_policy": args.infrastructure_error_policy,
    }
    atomic_write_json(
        os.path.join(args.worker_output_dir, "worker_config.json"),
        worker_config,
    )
    status_path = os.path.join(args.worker_output_dir, "status.json")
    atomic_write_json(status_path, dict(worker_config, status="running"))
    logger = init_logs(
        args.worker_output_dir,
        name="forced_coop_experiment.shard_{}".format(args.shard_id),
    )
    challenge = None
    try:
        agents = build_agents(args, logger)
        challenge = Challenge(logger, args, experiment_config)
        challenge.submit(agents, assigned_episodes)
    except BaseException as error:
        if isinstance(error, KeyboardInterrupt):
            worker_status = "interrupted"
        elif isinstance(error, ShardInfrastructureAbort):
            worker_status = "stopped_on_infrastructure_error"
        else:
            worker_status = "failed"
        atomic_write_json(
            status_path,
            dict(
                worker_config,
                status=worker_status,
                error_type=type(error).__name__,
                error_message=str(error),
            ),
        )
        raise
    else:
        atomic_write_json(status_path, dict(worker_config, status="complete"))
    finally:
        if challenge is not None:
            challenge.close()


if __name__ == "__main__":
    main()
