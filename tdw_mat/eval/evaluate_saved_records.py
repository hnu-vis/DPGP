"""Independent TDW evaluation CLI: standard library only, no simulator or LLM."""

import argparse
import csv
import io
import json
from collections import defaultdict
from contextlib import redirect_stdout
from pathlib import Path
import sys

TDW_MAT_DIR = Path(__file__).resolve().parents[1]
TDW_GYM_DIR = TDW_MAT_DIR / "tdw-gym"
for path in (TDW_MAT_DIR, TDW_GYM_DIR):
    if str(path) not in sys.path:
        sys.path.append(str(path))

if __package__:
    from .budget_replay import replay_budget_prefix
    from .evaluation_records import load_episode_record
    from .metrics_forced_coop import (
        NUMERIC_METRIC_PATHS, SUCCESS_RATE_KEYS, aggregate_experiment_metrics,
        build_record_metrics, json_safe, print_episode_metrics,
        print_experiment_metrics,
    )
else:
    from budget_replay import replay_budget_prefix
    from evaluation_records import load_episode_record
    from metrics_forced_coop import (
        NUMERIC_METRIC_PATHS, SUCCESS_RATE_KEYS, aggregate_experiment_metrics,
        build_record_metrics, json_safe, print_episode_metrics,
        print_experiment_metrics,
    )

from episode_limits import positive_finite


def evaluate_directory(record_dir, source="raw", budget=None, expected_task_count=None,
                       expected_run_count=None):
    root = Path(record_dir).resolve()
    if not root.is_dir():
        raise ValueError("record directory does not exist")
    if budget is not None:
        budget = positive_finite(budget)
    if source != "raw":
        raise ValueError("evaluation requires raw metadata and step trace")
    records, errors, warnings = [], [], []
    seen, configurations = set(), set()
    per_run, coverage = defaultdict(list), {}
    configs = {}
    for path in root.rglob("experiment_config.json"):
        try:
            config = json.loads(path.read_text())
            if not isinstance(config, dict):
                raise ValueError("configuration must be an object")
            configs[path.parent] = config
        except (OSError, ValueError) as error:
            errors.append("cannot read configuration {}: {}".format(path, error))
    run_names = {}
    for directory, config in configs.items():
        run_id = str(config.get("run_id", directory.name))
        if run_id in run_names and run_names[run_id] != directory:
            errors.append("duplicate run ID configuration: " + run_id)
        run_names[run_id] = directory
        expected = list(range(expected_task_count)) if expected_task_count is not None else config.get("eval_episodes")
        coverage[run_id] = {"expected": expected, "observed": []}
        if config.get("effective_config"):
            configurations.add(json.dumps(config["effective_config"], sort_keys=True))
    for path in sorted(root.rglob("result_episode.json")):
        directory = path.parent
        config = configs.get(directory.parent, {})
        run_id = str(config.get("run_id", directory.parent.name))
        try:
            episode = int(directory.name)
        except ValueError:
            errors.append("invalid episode directory: " + str(directory))
            continue
        key = (run_id, episode)
        if key in seen:
            errors.append("duplicate episode-run: {} / {}".format(*key))
            continue
        seen.add(key)
        item = coverage.setdefault(run_id, {
            "expected": list(range(expected_task_count)) if expected_task_count is not None else None,
            "observed": [],
        })
        item["observed"].append(episode)
        provenance = {"run_id": run_id, "episode": episode, "record_path": str(path)}
        try:
            result = json.loads(path.read_text())
            if (result.get("metrics") or {}).get("error"):
                raise ValueError("source infrastructure error: " + str(result["metrics"]["error"]))
            raw = load_episode_record(directory)
            if raw["metadata"]["episode"] != episode or raw["metadata"].get("run_id", run_id) != run_id:
                raise ValueError("raw episode/run identity differs from directory/configuration")
            effective = raw["metadata"].get("effective_config")
            if effective:
                configurations.add(json.dumps(effective, sort_keys=True))
            if config.get("effective_config") and effective != config["effective_config"]:
                raise ValueError("run configuration differs from raw episode configuration")
            if budget is not None:
                raw, replay = replay_budget_prefix(raw, budget)
                provenance["budget_replay"] = replay
            metrics = build_record_metrics(raw)
            provenance["source"] = "raw_json"
            provenance["collection_config"] = effective
            if metrics.get("task_type") not in ("ordinary", "coop"):
                warnings.append("{} / {}: task type unknown; not inferred as ordinary".format(*key))
            if metrics.get("task_success_within_time", metrics.get("task_success")) is None:
                warnings.append("{} / {}: outcome unknown; not inferred as failure".format(*key))
        except (OSError, ValueError, TypeError, KeyError, AttributeError) as error:
            errors.append("{} / {}: {}".format(run_id, episode, error))
            metrics = {"episode": episode, "error": {"kind": "evaluation_record_error", "message": str(error)}}
        provenance["metrics"] = metrics
        records.append(provenance)
        per_run[run_id].append(metrics)
    for run_id, item in coverage.items():
        observed, expected = set(item["observed"]), item["expected"]
        item["observed"] = sorted(observed)
        item["missing"] = sorted(set(expected) - observed) if expected is not None else None
        item["unexpected"] = sorted(observed - set(expected)) if expected is not None else None
        if expected is None:
            warnings.append(run_id + ": expected task coverage unknown")
        elif item["missing"] or item["unexpected"]:
            errors.append(run_id + ": missing/unexpected task coverage")
    if not records:
        errors.append("no episode records found")
    if expected_run_count is not None and len(coverage) != expected_run_count:
        errors.append("observed run count differs from expected run count")
    elif expected_run_count is None:
        warnings.append("expected run count not specified; wholly absent runs cannot be detected")
    if len(configurations) > 1:
        errors.append("mixed identifiable experiment configurations; pooled report is not a fair main comparison")
    if not configurations:
        warnings.append("experiment configuration unknown; comparability cannot be verified")
    pooled = aggregate_experiment_metrics([item["metrics"] for item in records])
    for key, summary in pooled["numeric_summaries"].items():
        if summary["count"] < pooled["valid_episode_count"]:
            warnings.append("{} available for {}/{} valid episode-runs".format(key, summary["count"], pooled["valid_episode_count"]))
    return json_safe({
        "evaluation_time_budget": budget, "source_mode": source,
        "coverage": coverage, "coverage_complete": bool(coverage) and all(
            item["expected"] is not None and not item["missing"] and not item["unexpected"] for item in coverage.values()
        ) and pooled["invalid_metric_record_count"] == 0 and not errors,
        "errors": errors, "warnings": warnings, "episodes": records,
        "runs": {run: aggregate_experiment_metrics(per_run[run]) for run in coverage},
        "pooled": pooled,
    })


def write_report(report, output_dir, overwrite=False, print_episodes=False):
    output_dir = Path(output_dir)
    paths = [output_dir / name for name in ("evaluation.json", "groups.csv", "report.txt")]
    if not overwrite and any(path.exists() for path in paths):
        raise ValueError("report already exists; use a new output directory or --overwrite")
    output = io.StringIO()
    with redirect_stdout(output):
        print("TDW offline evaluation | budget=" + str(report["evaluation_time_budget"]))
        print("Coverage complete: " + str(report["coverage_complete"]))
        for message in report["errors"]:
            print("ERROR: " + message)
        for message in report["warnings"]:
            print("WARNING: " + message)
        if print_episodes:
            for item in report["episodes"]:
                if not item["metrics"].get("error"):
                    print("Run: " + item["run_id"])
                    print_episode_metrics(item["metrics"])
        for run, metrics in report["runs"].items():
            print_experiment_metrics(metrics, prefix="Run " + run)
        print_experiment_metrics(report["pooled"], prefix="Pooled TDW metrics")
    output_dir.mkdir(parents=True, exist_ok=True)
    paths[0].write_text(json.dumps(report, indent=2, allow_nan=False))
    paths[2].write_text(output.getvalue())
    columns = ["scope", "run_id", "task_type", "outcome", "episode_count"]
    columns += [key + "_" + suffix for key in SUCCESS_RATE_KEYS for suffix in ("success_count", "total_count", "rate")]
    columns += [key + "_" + suffix for key in report["pooled"]["numeric_summaries"]
                for suffix in ("count", "sum", "mean", "population_std", "min", "max")]
    with paths[1].open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        for scope, run_id, metrics in [("pooled", "", report["pooled"])] + [
            ("run", run, metrics) for run, metrics in report["runs"].items()
        ]:
            for task_type, groups in metrics["groups"].items():
                for outcome, group in groups.items():
                    row = dict(scope=scope, run_id=run_id, task_type=task_type,
                               outcome=outcome, episode_count=group["episode_count"])
                    for key, entry in group["success_rates"].items():
                        row.update({key + "_" + suffix: value for suffix, value in entry.items()})
                    for key, entry in group["numeric_summaries"].items():
                        row.update({key + "_" + suffix: value for suffix, value in entry.items()})
                    writer.writerow(row)
    return output.getvalue()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--record-dir", required=True)
    parser.add_argument("--source", choices=("raw",), default="raw")
    parser.add_argument("--expected-task-count", type=int)
    parser.add_argument("--expected-run-count", type=int)
    parser.add_argument("--max-embodied-execution-time", "--time-budget", dest="budget", type=float)
    parser.add_argument("--output-dir")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--print-episodes", action="store_true")
    args = parser.parse_args(argv)
    for name in ("expected_task_count", "expected_run_count"):
        if getattr(args, name) is not None and getattr(args, name) <= 0:
            parser.error(name + " must be positive")
    try:
        report = evaluate_directory(args.record_dir, args.source, args.budget,
                                    args.expected_task_count, args.expected_run_count)
        suffix = "offline_evaluation" if args.budget is None else "offline_evaluation_budget_{:g}".format(positive_finite(args.budget))
        output_dir = args.output_dir or str(Path(args.record_dir) / suffix)
        print(write_report(report, output_dir, args.overwrite, args.print_episodes), end="")
    except (OSError, ValueError, TypeError) as error:
        parser.exit(2, "evaluation failed: {}\n".format(error))
    return 2 if report["errors"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
