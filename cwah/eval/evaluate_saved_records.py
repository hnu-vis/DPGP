"""Offline C-WAH evaluation. Only load pickle files from trusted experiments."""

import argparse
import contextlib
import csv
import io
import json
import pickle
import re
import sys
from pathlib import Path

if __package__:
    from .budget_replay import replay_budget_prefix, validate_evaluation_budget
    from .metrics_forced_coop import (
        OUTCOME_GROUPS,
        SUCCESS_RATE_KEYS, TASK_GROUPS, aggregate_experiment_metrics,
        build_episode_metrics, print_episode_metrics, print_experiment_metrics,
        finalize_episode_metrics,
    )
else:
    # Direct-file entry still imports only lightweight benchmark helpers.
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from budget_replay import replay_budget_prefix, validate_evaluation_budget
    from metrics_forced_coop import (
        OUTCOME_GROUPS,
        SUCCESS_RATE_KEYS, TASK_GROUPS, aggregate_experiment_metrics,
        build_episode_metrics, print_episode_metrics, print_experiment_metrics,
        finalize_episode_metrics,
    )


# ==================== Metrics: Offline Record Loading ====================
# Purpose: Recompute trusted raw logs without importing agents or environments.
RECORD_PATTERN = re.compile(r'^(logs|result)_agent_(\d+)_(.+)_(\d+)\.(pik|json)$')
DETAIL_FIELDS = {
    'communication_details', 'execution_time_step_details',
    'physical_action_details', 'llm_call_details',
}


def compact_metrics(metrics):
    result = dict(metrics)
    result['execution_time'] = {
        key: value for key, value in (metrics.get('execution_time') or {}).items()
        if key not in DETAIL_FIELDS
    }
    return result


def discover_records(record_dir, source='auto'):
    candidates = {}
    for path in sorted(record_dir.iterdir()):
        match = RECORD_PATTERN.match(path.name)
        if not path.is_file() or not match:
            continue
        kind, episode, task, run, suffix = match.groups()
        if (kind, suffix) not in (('logs', 'pik'), ('result', 'json')):
            continue
        key = (int(episode), int(run))
        candidates.setdefault(key, {}).setdefault(suffix, []).append((path, task))
    selected, duplicates, ignored = [], [], []
    for key, formats in sorted(candidates.items()):
        if source == 'pickle':
            chosen = 'pik'
        elif source == 'json':
            chosen = 'json'
        else:
            chosen = 'pik' if 'pik' in formats else 'json'
        paths = formats.get(chosen, [])
        if len(paths) > 1:
            duplicates.append({'episode_id': key[0], 'run_id': key[1],
                               'paths': [str(path) for path, _ in paths]})
            continue
        if paths:
            path, task = paths[0]
            selected.append((key, path, task))
            ignored.extend(str(path) for suffix, entries in formats.items()
                           if suffix != chosen for path, _ in entries)
    return selected, duplicates, ignored


def load_record(path, episode_id, run_id, task_name, config, evaluation_time_budget=None):
    warnings = []
    replay_metadata = None
    if path.suffix == '.pik':
        with path.open('rb') as stream:
            raw = pickle.load(stream)
        if not isinstance(raw, dict):
            raise ValueError('raw episode must be a dictionary')
        required = ('gt_goals', 'progress', 'action', 'coop', 'finished')
        missing = [key for key in required if key not in raw]
        if missing:
            raise ValueError('missing success/type evidence: ' + ', '.join(missing))
        if not any(isinstance(step, dict) and 'coop_constraints' in step for step in raw['coop']):
            raise ValueError('raw log lacks explicit task/subgoal type evidence (coop_constraints)')
        if raw.get('task_id') is not None and int(raw['task_id']) != episode_id:
            raise ValueError('task_id conflicts with filename')
        if evaluation_time_budget is not None:
            raw, replay_metadata = replay_budget_prefix(raw, evaluation_time_budget)
        metrics = build_episode_metrics(raw)  # Never reuse raw['metrics'].
        if replay_metadata is not None:
            metrics['budget_evaluation'] = replay_metadata
        method = raw['method_name']
        model = raw['llm_model']
        time_steps = raw.get('execution_time_steps') or []
        communication_modes = sorted({str(step.get('communication_mode')) for step in time_steps
                                      if isinstance(step, dict) and step.get('communication_mode')})
        if raw.get('termination_reason') == 'step_limit':
            warnings.append('step-censored trajectory; cannot infer a longer-budget outcome')
        seed = raw.get('seed')
    else:
        if evaluation_time_budget is not None:
            raise ValueError('budget replay requires original pickle; JSON lacks complete progress/step-aligned usage')
        with path.open(encoding='utf-8') as stream:
            raw = json.load(stream)
        for key, expected in (('episode_id', episode_id), ('run_id', run_id)):
            if key in raw and int(raw[key]) != expected:
                raise ValueError(key + ' conflicts with filename')
        stored = raw.get('metrics')
        if not isinstance(stored, dict) or stored.get('error'):
            raise ValueError('JSON has no valid episode metrics')
        if stored.get('task_type') not in ('ordinary', 'coop'):
            raise ValueError('JSON lacks task type')
        if not all(key in (stored.get('success_rates') or {}) for key in SUCCESS_RATE_KEYS):
            raise ValueError('JSON lacks success numerator/denominator evidence')
        metrics = finalize_episode_metrics(stored)
        method = raw['method']
        model = raw['llm_model']
        communication_modes = []
        seed = raw.get('seed')
        warnings.append('JSON summary: metrics are validated; complete per-step evidence requires the pickle record')
    return {
        'episode_id': episode_id, 'run_id': run_id, 'task_name': task_name,
        'method': method, 'llm_model': model, 'seed': seed,
        'prediction_mode': raw['prediction_mode'], 'method_native_config': raw['method_native_config'],
        'record_path': str(path), 'source': path.suffix[1:],
        'communication_modes': communication_modes,
        'budget_evaluation': replay_metadata,
        'warnings': warnings, 'metrics': compact_metrics(metrics),
    }
# ==================== End Metrics: Offline Record Loading ====================


# ==================== Metrics: Offline Reports and Coverage ====================
# Purpose: Preserve sample coverage and per-run statistics beside pooled reports.
def evaluate_directory(record_dir, source='auto', expected_task_count=None, evaluation_time_budget=None):
    if evaluation_time_budget is not None:
        evaluation_time_budget = validate_evaluation_budget(evaluation_time_budget)
        if source == 'json':
            raise ValueError('budget replay requires --source pickle or auto, not json')
    record_dir = Path(record_dir).resolve()
    config_path = record_dir / 'experiment_config.json'
    config = json.loads(config_path.read_text(encoding='utf-8')) if config_path.is_file() else {}
    selected, duplicates, ignored = discover_records(record_dir, source)
    episodes, errors = [], []
    for (episode, run), path, task in selected:
        try:
            episodes.append(load_record(path, episode, run, task, config, evaluation_time_budget))
        except Exception as exc:
            errors.append({'episode_id': episode, 'run_id': run,
                           'record_path': str(path), 'error': str(exc)})
    metrics = aggregate_experiment_metrics(
        [episode['metrics'] for episode in episodes] + [{'error': error['error']} for error in errors]
        + [{'error': 'duplicate episode-run'} for _ in duplicates])
    expected_ids = config.get('test_task')
    if expected_ids is not None:
        expected_ids = set(int(value) for value in expected_ids)
    elif expected_task_count is not None:
        expected_ids = set(range(expected_task_count))
    observed_runs = {key[1] for key, _, _ in selected} | {item['run_id'] for item in duplicates}
    expected_runs = set(range(int(config['num_runs']))) if 'num_runs' in config else observed_runs
    successful_keys = {(item['episode_id'], item['run_id']) for item in episodes}
    expected_keys = ({(episode, run) for episode in expected_ids for run in expected_runs}
                     if expected_ids is not None else None)
    missing = sorted(expected_keys - successful_keys) if expected_keys is not None else []
    unexpected = sorted(successful_keys - expected_keys) if expected_keys is not None else []
    fingerprints = sorted({json.dumps({
        'method': item['method'], 'model': item['llm_model'],
        'budget': item['metrics'].get('max_embodied_execution_time'),
        'collection_budget': ((item.get('budget_evaluation') or {}).get('collection_time_budget')
                              or item['metrics'].get('max_embodied_execution_time')),
        'time_source': item['metrics']['execution_time'].get('physical_time_source'),
        'physical_time_parameters': item['metrics']['execution_time']['physical_time_parameters'],
        'prediction_mode': item['prediction_mode'],
        'method_native_config': item['method_native_config'],
        'communication_modes': item['communication_modes'] or [config.get('communication_mode')],
        'communication_count_unit': item['metrics']['execution_time'].get('communication_count_unit'),
        'communication_send_chars_per_second': item['metrics']['execution_time'].get('communication_send_chars_per_second'),
        'communication_max_message_chars': item['metrics']['execution_time'].get('communication_max_message_chars'),
        'minimum_step_frames': item['metrics']['execution_time'].get('minimum_step_frames'),

    }, sort_keys=True) for item in episodes})
    warnings = []
    if expected_keys is None:
        warnings.append('expected task coverage unknown; supply --expected-task-count for a full benchmark')
    if len(fingerprints) > 1:
        warnings.append('mixed method/model/prediction mode/budget/physical or communication settings: pooled metrics are descriptive only')
    if errors or duplicates or missing or unexpected:
        warnings.append('coverage is incomplete or inconsistent; valid-record SR is not a complete benchmark SR')
    run_metrics = {
        str(run): aggregate_experiment_metrics(
            [item['metrics'] for item in episodes if item['run_id'] == run]
            + [{'error': item['error']} for item in errors if item['run_id'] == run]
            + [{'error': 'duplicate episode-run'} for item in duplicates if item['run_id'] == run]
        ) for run in sorted(expected_runs | observed_runs)
    }
    return {
        'record_dir': str(record_dir),
        'source_policy': source, 'experiment_config': config,
        'evaluation_mode': 'trajectory_prefix' if evaluation_time_budget is not None else 'recorded_episode',
        'evaluation_time_budget': evaluation_time_budget,
        'coverage': {'expected_episode_run_count': len(expected_keys) if expected_keys is not None else None,
                     'valid_episode_run_count': len(episodes), 'missing_or_invalid': missing,
                     'unexpected': unexpected, 'duplicate_records': duplicates,
                     'record_errors': errors, 'ignored_alternate_formats': ignored},
        'warnings': warnings, 'configuration_fingerprints': [json.loads(value) for value in fingerprints],
        'run_metrics': run_metrics, 'metrics': metrics, 'episodes': episodes,
    }


def group_csv_rows(report):
    rows = []
    for run, metrics in [('pooled', report['metrics'])] + list(report['run_metrics'].items()):
        for task in TASK_GROUPS:
            for outcome in OUTCOME_GROUPS:
                group = metrics['groups'][task][outcome]
                row = {'run': run, 'task_type': task, 'outcome': outcome, 'N': group['episode_count'],
                       'evaluation_mode': report['evaluation_mode'],
                       'evaluation_time_budget': report['evaluation_time_budget']}
                for key, entry in group['success_rates'].items():
                    for field in ('success_count', 'total_count', 'rate'):
                        row[key + '_' + field] = entry[field]
                for key, summary in group['numeric_summaries'].items():
                    for field in ('count', 'sum', 'mean', 'population_std', 'min', 'max'):
                        row[key + '_' + field] = summary[field]
                rows.append(row)
    return rows


def render_report(report, include_episodes=False):
    stream = io.StringIO()
    with contextlib.redirect_stdout(stream):
        print('Offline evaluation | directory={}'.format(report['record_dir']))
        print('Evaluation mode: {} | requested budget: {} s'.format(
            report['evaluation_mode'], report['evaluation_time_budget']))
        if report['evaluation_time_budget'] is not None:
            print('Collection budgets(s): {} | evaluation budget(s): {}'.format(
                sorted({item['budget_evaluation']['collection_time_budget'] for item in report['episodes']}),
                report['evaluation_time_budget']))
            print('Replay: complete crossing step and all its costs retained; no new actions/LLM calls generated')
        for warning in report['warnings']:
            print('WARNING: ' + warning)
        coverage = report['coverage']
        print('Expected episode-runs: {} | missing/invalid: {} | unexpected: {} | duplicates: {}'.format(
            coverage['expected_episode_run_count'], len(coverage['missing_or_invalid']),
            len(coverage['unexpected']), len(coverage['duplicate_records'])))
        for error in coverage['record_errors']:
            print('RECORD ERROR: ' + json.dumps(error, ensure_ascii=False))
        for item in report['episodes']:
            for warning in item['warnings']:
                print('RECORD WARNING: episode={} run={}: {}'.format(
                    item['episode_id'], item['run_id'], warning))
            if include_episodes:
                print_episode_metrics(item['metrics'], item['episode_id'], item['run_id'], item['task_name'])
        print_experiment_metrics(report['metrics'], prefix='C-WAH Offline Pooled Metrics')
        for run, metrics in report['run_metrics'].items():
            print_experiment_metrics(metrics, prefix='C-WAH Offline Run {} Metrics'.format(run))
    return stream.getvalue()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--record-dir', required=True, type=Path)
    parser.add_argument('--output-dir', type=Path,
                        help='default: RECORD_DIR/offline_evaluation[_budget_VALUE]')
    parser.add_argument('--source', choices=('auto', 'pickle', 'json'), default='auto')
    parser.add_argument('--expected-task-count', type=int, help='full dataset size; targeted config takes precedence')
    parser.add_argument('--max-embodied-execution-time', '--time-budget', dest='evaluation_time_budget',
                        type=float, help='optional shorter evaluation budget; requires complete raw pickle traces')
    parser.add_argument('--print-episodes', action='store_true')
    parser.add_argument('--overwrite', action='store_true', help='replace only the three generated report files')
    args = parser.parse_args(argv)
    if not args.record_dir.is_dir():
        parser.error('--record-dir must be an existing directory')
    if args.expected_task_count is not None and args.expected_task_count <= 0:
        parser.error('--expected-task-count must be positive')
    if args.evaluation_time_budget is not None:
        try:
            args.evaluation_time_budget = validate_evaluation_budget(args.evaluation_time_budget)
        except ValueError as exc:
            parser.error(str(exc))
        if args.source == 'json':
            parser.error('budget replay requires --source pickle or auto, not json')
    output_name = ('offline_evaluation' if args.evaluation_time_budget is None else
                   'offline_evaluation_budget_{}'.format(format(args.evaluation_time_budget, '.15g')))
    output_dir = (args.output_dir or args.record_dir / output_name).resolve()
    outputs = [output_dir / name for name in ('evaluation.json', 'groups.csv', 'report.txt')]
    if not args.overwrite and any(path.exists() for path in outputs):
        parser.error('report files already exist; choose a new --output-dir or use --overwrite')
    report = evaluate_directory(args.record_dir, args.source, args.expected_task_count, args.evaluation_time_budget)
    rendered = render_report(report, args.print_episodes)
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs[0].write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    rows = group_csv_rows(report)
    with outputs[1].open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    outputs[2].write_text(rendered, encoding='utf-8')
    print(rendered, end='')
    print('Reports: ' + str(output_dir))
    coverage = report['coverage']
    return 2 if (not report['episodes'] or coverage['record_errors'] or coverage['duplicate_records']
                 or coverage['missing_or_invalid'] or coverage['unexpected']
                 or len(report['configuration_fingerprints']) > 1) else 0


if __name__ == '__main__':
    sys.exit(main())
# ==================== End Metrics: Offline Reports and Coverage ====================
