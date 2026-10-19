"""Arguments for the C-WAH benchmark and DPGP experiments."""

import argparse
import math
from pathlib import Path

from envs.physical_time import PHYSICAL_TIME_SETTINGS


CWAH_DIR = Path(__file__).resolve().parent

# ==================== Formal Experiment: Fixed Benchmark Settings ====================
# Purpose: Prevent command-line options from changing the official environment setup.
NUMBER_OF_AGENTS = 2
OBSERVATION_TYPE = "partial"
AGENT_GOALS = ("full", "full")
NO_GRAPHICS = True
USE_EDITOR = False
DEBUG = False
BASE_SEED = 1
COMMUNICATION_ENABLED = True
LLM_SOURCE = "openai"
LLM_TEMPERATURE = 0.0
LLM_TOP_P = 1.0
LLM_MAX_TOKENS = 4096
LLM_NUM_SAMPLES = 1
SAVE_IMAGE = False
COMMUNICATION_MODE = "exclusive_action"
COOP_ACTION_INTERFACE = "implicit_pairing"
COMMUNICATION_MAX_MESSAGE_CHARS = 12000
DEFAULT_PLANNING_ACTION_HORIZON = 8
COMMUNICATION_SEND_CHARS_PER_SECOND = 2500.0
MINIMUM_STEP_FRAMES = 1
DEFAULT_MAX_EMBODIED_EXECUTION_TIME = 600.0
PHYSICAL_ACTION_RENDER_MODE = "joint"
DEFAULT_PHYSICAL_TIME_SOURCE = "calibrated_action_model"
DEFAULT_PHYSICAL_ACTION_FRAME_RATE = PHYSICAL_TIME_SETTINGS.frame_rate
DEFAULT_PHYSICAL_MOVEMENT_SPEED = PHYSICAL_TIME_SETTINGS.movement_speed
DEFAULT_PHYSICAL_TURN_TIME = PHYSICAL_TIME_SETTINGS.turn_time
DEFAULT_PHYSICAL_GRAB_TIME = PHYSICAL_TIME_SETTINGS.grab_time
DEFAULT_PHYSICAL_OPEN_TIME = PHYSICAL_TIME_SETTINGS.open_time
DEFAULT_PHYSICAL_PUT_TIME = PHYSICAL_TIME_SETTINGS.put_time
# ==================== End Formal Experiment: Fixed Benchmark Settings ====================

DEFAULT_METHOD_NAME = "dpgp"
SUPPORTED_METHOD_NAMES = (DEFAULT_METHOD_NAME,)
DEFAULT_RESULT_MODE = "cwah_symbolic_experiment"
DEFAULT_DATASET_PATH = str(
    CWAH_DIR / "dataset" / "Test" / "aaa_test_env_set_help_100.pik"
)
DEFAULT_EXECUTABLE_FILE = str(
    CWAH_DIR.parent / "executable" / "linux_exec.v2.3.0.x86_64"
)


def get_args():
    parser = argparse.ArgumentParser(
        description="C-WAH formal symbolic experiment"
    )

    # ==================== Formal Experiment: Runtime Arguments ====================
    # Purpose: Expose only experiment identity, workload, and deployment controls.
    parser.add_argument(
        "--method-name",
        "--method_name",
        dest="method_name",
        choices=SUPPORTED_METHOD_NAMES,
        default=DEFAULT_METHOD_NAME,
        help="formal method adapter to run",
    )
    parser.add_argument(
        "--dataset-path",
        "--dataset_path",
        dest="dataset_path",
        type=str,
        default=DEFAULT_DATASET_PATH,
        help="C-WAH benchmark pickle path",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default=DEFAULT_RESULT_MODE,
        help="result directory name under the repository's test_results directory",
    )
    parser.add_argument(
        "--num-runs",
        "--num_runs",
        dest="num_runs",
        type=int,
        default=1,
        help="number of independent runs per episode",
    )
    parser.add_argument(
        "--existing-result-policy",
        "--existing_result_policy",
        dest="existing_result_policy",
        choices=("skip", "overwrite"),
        default="skip",
        help="skip existing episode logs or rerun and overwrite them",
    )
    parser.add_argument(
        "--test-task",
        "--test_task",
        dest="test_task",
        type=int,
        nargs="+",
        default=None,
        help="dataset episode indices to run; omit to run the full dataset",
    )
    parser.add_argument(
        "--reset-max-attempts",
        "--reset_max_attempts",
        dest="reset_max_attempts",
        type=int,
        default=3,
        help="maximum environment reset attempts before failing",
    )
    parser.add_argument(
        "--max-episode-length",
        "--max_episode_length",
        dest="max_episode_length",
        type=int,
        help="optional episode step limit; omitted means time-only termination",
    )
    parser.add_argument(
        "--planning-action-horizon",
        dest="planning_action_horizon",
        type=int,
        default=DEFAULT_PLANNING_ACTION_HORIZON,
        help="maximum non-wait actions sent to the peer per global plan",
    )
    # Method-local discovery controls; public benchmark timing is unchanged.
    parser.add_argument(
        "--dpgp-prediction-mode", choices=("deterministic_llm", "discovery_probability"),
        default="discovery_probability", help="single-point LLM prediction or finite-count discovery event planning",
    )
    parser.add_argument("--discovery-candidates", type=int, default=6)
    parser.add_argument("--discovery-samples", type=int, default=8)
    parser.add_argument("--discovery-time-horizon", type=float, default=120.)
    parser.add_argument("--discovery-astar-expansions", type=int, default=2000)
    parser.add_argument(
        "--max-embodied-execution-time",
        dest="max_embodied_execution_time",
        type=float,
        default=DEFAULT_MAX_EMBODIED_EXECUTION_TIME,
        help="simulated execution-time budget in seconds (default: 600); must be finite and positive",
    )
    parser.add_argument(
        "--executable-file",
        "--executable_file",
        dest="executable_file",
        type=str,
        default=DEFAULT_EXECUTABLE_FILE,
        help="VirtualHome Unity executable path",
    )
    parser.add_argument(
        "--base-port",
        dest="base_port",
        type=int,
        default=6217,
        help="base Unity communication port",
    )
    parser.add_argument(
        "--llm-model",
        "--lm-id",
        "--lm_id",
        dest="lm_id",
        type=str,
        default="gpt-4o-mini",
        help="LLM model used by the formal method",
    )
    # ==================== End Formal Experiment: Runtime Arguments ====================

    args = parser.parse_args()

    # Benchmark rules are fixed; record their actual values for runners and reports.
    args.communication_mode = COMMUNICATION_MODE
    args.coop_action_interface = COOP_ACTION_INTERFACE
    args.communication_max_message_chars = COMMUNICATION_MAX_MESSAGE_CHARS
    args.communication_send_chars_per_second = COMMUNICATION_SEND_CHARS_PER_SECOND
    args.minimum_step_frames = MINIMUM_STEP_FRAMES
    args.physical_action_render_mode = PHYSICAL_ACTION_RENDER_MODE

    if args.num_runs <= 0:
        parser.error("--num-runs must be positive")
    if args.reset_max_attempts <= 0:
        parser.error("--reset-max-attempts must be positive")
    if args.max_episode_length is not None and args.max_episode_length <= 0:
        parser.error("--max-episode-length must be positive")
    if args.planning_action_horizon <= 0:
        parser.error("--planning-action-horizon must be positive")
    if not math.isfinite(args.max_embodied_execution_time) or args.max_embodied_execution_time <= 0:
        parser.error("--max-embodied-execution-time must be finite and positive")
    if not 1 <= args.base_port <= 65535:
        parser.error("--base-port must be between 1 and 65535")
    if Path(args.mode).name != args.mode or args.mode in {"", ".", ".."}:
        parser.error("--mode must be a single non-empty directory name")
    if not args.lm_id.strip():
        parser.error("--llm-model must be non-empty")

    # Fixed method values are present on the namespace but cannot be overridden.
    args.seed = BASE_SEED
    args.communication = COMMUNICATION_ENABLED
    args.source = LLM_SOURCE
    args.t = LLM_TEMPERATURE
    args.top_p = LLM_TOP_P
    args.max_tokens = LLM_MAX_TOKENS
    args.n = LLM_NUM_SAMPLES

    from agents.dpgp.config import MethodConfig

    args.method_native_config = {
        "method": "DPGP",
        "prediction_mode": args.dpgp_prediction_mode,
        **MethodConfig(
            action_horizon=args.planning_action_horizon,
            message_limit=args.communication_max_message_chars,
            discovery_candidates=args.discovery_candidates,
            discovery_samples=args.discovery_samples,
            discovery_time_horizon=args.discovery_time_horizon,
            discovery_astar_expansions=args.discovery_astar_expansions,
        ).to_dict(args.dpgp_prediction_mode),
        "temperature": LLM_TEMPERATURE,
        "top_p": LLM_TOP_P,
        "max_tokens": LLM_MAX_TOKENS,
        "n": LLM_NUM_SAMPLES,
    }

    args.save_image = SAVE_IMAGE
    args.debug = DEBUG
    # Record the fixed physical-time settings used by the benchmark.
    args.physical_time_parameters = PHYSICAL_TIME_SETTINGS.to_dict()
    args.physical_time_source = DEFAULT_PHYSICAL_TIME_SOURCE
    args.physical_action_frame_rate = DEFAULT_PHYSICAL_ACTION_FRAME_RATE
    args.physical_movement_speed = DEFAULT_PHYSICAL_MOVEMENT_SPEED
    args.physical_turn_time = DEFAULT_PHYSICAL_TURN_TIME
    args.physical_grab_time = DEFAULT_PHYSICAL_GRAB_TIME
    args.physical_open_time = DEFAULT_PHYSICAL_OPEN_TIME
    args.physical_put_time = DEFAULT_PHYSICAL_PUT_TIME
    return args
