"""Author and execute real-fit notebooks only after a coordinator compute grant.

This harness is UNEXECUTED. All generated cases remain UNTESTED until their
notebooks, raw oracle files, and visual review have been inspected.
Uses local generated data adapted from repository synthetic notebooks and
test_prediction_output_columns.py, never PR172 downloads or credentials.
"""

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import importlib.metadata
import json
import math
from pathlib import Path
import shutil
import sys
import traceback
from uuid import uuid4


TASKS = {
    'binary': ('binary_classification', 'PREDICT ents.FLAG = 1 FOR EACH ents.ID'),
    'multiclass': ('multiclass_classification',
                   'PREDICT ents.CATEGORY FOR EACH ents.ID'),
    'multilabel': ('multilabel_classification',
                   'PREDICT ents.TAGS CLASSIFY TOP 6 FOR EACH ents.ID'),
    'multilabel_ranking': ('multilabel_ranking',
                          'PREDICT ents.TAGS RANK TOP 5 FOR EACH ents.ID'),
    'regression': ('regression',
                   'PREDICT COUNT(events.*, 0, 7, days) FOR EACH ents.ID'),
    'forecasting': ('forecasting', 'PREDICT COUNT(events.*, 0, 7, days) '
                     'FORECAST 3 TIMEFRAMES FOR EACH ents.ID'),
    'static_link': ('static_link_prediction',
                    'PREDICT LIST_DISTINCT(events.ITEM_ID) '
                    'RANK TOP 5 FOR EACH ents.ID'),
    'temporal_link': ('temporal_link_prediction',
                      'PREDICT LIST_DISTINCT(events.ITEM_ID, 0, 7, days) '
                      'RANK TOP 5 FOR EACH ents.ID'),
}
SCENARIOS = ('baseline', 'multi_trial', 'many_trials', 'early_stop',
             'single_epoch', 'refit_trainval', 'refit_full', 'diagnostics',
             'weighted_link', 'local_mlflow')


_AT_K = [f'{metric}@{k}' for metric in ('map', 'precision', 'recall', 'f1',
                                         'ndcg') for k in (1, 5, 10)]
METRICS = {
    'binary': ['auroc', 'auprc', 'ap', 'acc', 'f1', 'precision', 'recall'],
    'multiclass': ['acc', 'f1', 'precision', 'recall'],
    'multilabel': ['ap_macro', 'ap_micro', 'auroc_macro', 'auroc_micro',
                   'auprc_macro', 'auprc_micro'],
    'multilabel_ranking': _AT_K,
    'regression': ['mae', 'mse', 'rmse', 'smape'],
    'forecasting': ['mae', 'mse', 'rmse', 'smape'],
    'static_link': _AT_K,
    'temporal_link': _AT_K,
}


def write_json(path, value):
    """Preserve non-finite values explicitly instead of turning them into zero."""
    def clean(item):
        if isinstance(item, float) and not math.isfinite(item):
            return {'nonfinite': repr(item)}
        if isinstance(item, dict):
            return {str(k): clean(v) for k, v in item.items()}
        if isinstance(item, (list, tuple)):
            return [clean(v) for v in item]
        if type(item).__module__.startswith('numpy'):
            if hasattr(item, 'tolist'):
                return clean(item.tolist())
            if hasattr(item, 'item'):
                return clean(item.item())
        if isinstance(item, (str, int, float, bool)) or item is None:
            return item
        return repr(item)
    Path(path).write_text(json.dumps(clean(value), indent=2, allow_nan=False)
                          + '\n', encoding='utf-8')


def sha256(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def inputs_fingerprint(result):
    paths = {str(p): p for p in result.bundle_dir.rglob('*') if p.is_file()}
    for value in result.artifact_paths.values():
        path = Path(value)
        if path.is_file():
            paths[str(path)] = path
    return {name: sha256(path) for name, path in sorted(paths.items())}


def mlflow_visibility(hide=False):
    """Record installed packages; optionally hide imports in this fresh kernel."""
    versions = {}
    for name in ('mlflow', 'mlflow-skinny', 'mlflow-tracing'):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    names = ('mlflow', 'mlflow.tracking', 'mlflow.entities', 'sagemaker_mlflow')
    if hide:
        loaded = [name for name in sys.modules
                  if (name == 'mlflow' or name.startswith('mlflow.') or
                      name == 'sagemaker_mlflow') and sys.modules[name] is not None]
        if loaded:
            raise RuntimeError(f'Require fresh kernel before hiding imports: {loaded}')
        for name in names:
            sys.modules[name] = None
        assert importlib.util.find_spec('mlflow') is None
    return {'mode': 'import-hidden' if hide else 'normal-imports',
            'installed_versions': versions, 'uninstalled': False,
            'hidden_module_names': list(names) if hide else []}


def assert_mlflow_hidden():
    names = ('mlflow', 'mlflow.tracking', 'mlflow.entities', 'sagemaker_mlflow')
    assert all(name in sys.modules and sys.modules[name] is None for name in names)
    assert importlib.util.find_spec('mlflow') is None


def prepare(repo, output, task, scenario, grant, hide_mlflow=False):
    """Generate and inspect local tables on the approved dedicated GPU only."""
    if not grant.strip():
        raise RuntimeError('Coordinator sole-compute grant is required')
    if hide_mlflow and scenario == 'local_mlflow':
        raise ValueError('local_mlflow is incompatible with --hide-mlflow')
    if sys.platform != 'linux':
        raise RuntimeError('No local execution: approved Linux Brev only')
    visibility = mlflow_visibility(hide_mlflow)
    import numpy as np
    import pandas as pd
    import torch
    from kumoapi.graph import Edge
    from kumoapi.model_plan import TrainingTableGenerationPlan
    from kumoapi.typing import Stype
    import kumo_core
    import kumo.engine as kumo

    repo = Path(repo).resolve()
    approved_repo = Path('/home/ubuntu/kumo-engine-fork').resolve()
    approved_python = approved_repo / '.venv/bin/python'
    if repo != approved_repo:
        raise RuntimeError(f'Require approved dedicated checkout: {approved_repo}')
    # Preserve the venv entry-point path; resolving it can follow a base-Python
    # symlink outside the environment even when the correct venv is active.
    if Path(sys.executable).absolute() != approved_python:
        raise RuntimeError(f'Require approved interpreter: {approved_python}')
    imports = {}
    for module in (kumo, kumo_core):
        locations = ([Path(module.__file__).resolve()]
                     if module.__file__ else
                     [Path(path).resolve() for path in module.__path__])
        if not locations or any(not path.is_relative_to(repo)
                                for path in locations):
            raise RuntimeError(
                f'{module.__name__} imports outside approved repo: {locations}')
        imports[module.__name__] = [str(path) for path in locations]
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError('Require approved single-GPU environment')
    output = Path(output).resolve()
    requested_output = output
    if output.exists():
        suffix = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
        output = output.with_name(f'{output.name}-rerun-{suffix}-{uuid4().hex[:10]}')
        print(f'Preserving prior output; fresh fit directory: {output}', flush=True)
    output.mkdir(parents=True, exist_ok=False)
    data = output / 'data'
    data.mkdir()
    rng = np.random.default_rng(0)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.set_num_threads(4)
    # Product-label and event-table shapes follow the repository generators.
    # Extra columns and reduced sizes are explicit validation adaptations.
    count, events_count = 240, 24000
    tags = [f'tag{i}' for i in range(6)]
    feature = rng.normal(size=count)
    # Noisy labels share a learnable feature, never another label or ID parity.
    # Keep only this task's target below, before writing any source Parquet.
    binary_target = (feature + rng.normal(scale=0.9, size=count) > 0).astype(int)
    category_target = np.digitize(
        feature + rng.normal(scale=0.9, size=count), [-0.8, 0.0, 0.8])
    tag_probability = 1 / (1 + np.exp(-(
        0.7 * feature[:, None] + np.linspace(-0.8, 0.8, len(tags)))))
    tag_mask = rng.random((count, len(tags))) < tag_probability
    for row in np.flatnonzero(~tag_mask.any(axis=1)):
        tag_mask[row, rng.integers(len(tags))] = True
    target_column = {
        'binary': 'FLAG', 'multiclass': 'CATEGORY',
        'multilabel': 'TAGS', 'multilabel_ranking': 'TAGS',
    }.get(task)
    frames = {
        'ents': pd.DataFrame({
            'ID': np.arange(count), 'FLAG': binary_target,
            'CATEGORY': [f'class{i}' for i in category_target],
            'TAGS': [[tag for tag, present in zip(tags, row) if present]
                     for row in tag_mask],
            'VALUE': feature,
        }),
        'items': pd.DataFrame({'IID': np.arange(24)}),
        'events': pd.DataFrame({
            'EID': np.arange(events_count),
            'ENT_ID': rng.integers(0, count, events_count),
            'ITEM_ID': rng.integers(0, 24, events_count),
            'TS': pd.Timestamp('2024-01-01') + pd.to_timedelta(
                rng.integers(0, 240, events_count), unit='D'),
            'V': rng.normal(size=events_count),
        }),
    }
    removed_labels = [name for name in ('FLAG', 'CATEGORY', 'TAGS')
                      if name != target_column]
    frames['ents'] = frames['ents'].drop(columns=removed_labels)
    expected_columns = {'ID', 'VALUE'} | ({target_column} if target_column else set())
    assert set(frames['ents'].columns) == expected_columns
    inspection = {'generator': {
        'version': 'v2-isolated-noisy-targets', 'seed': 0,
        'target_column': target_column, 'removed_label_columns': removed_labels,
        'entity_columns': list(frames['ents'].columns),
        'signal': 'VALUE is seeded normal; labels have independent conditional noise',
        'no_label_or_id_parity_proxy': True,
    }}
    for name, frame in frames.items():
        frame.to_parquet(data / f'{name}.parquet', index=False)
        inspection[name] = {
            'rows': len(frame), 'schema': frame.dtypes.astype(str).to_dict(),
            'sample': frame.head().to_dict('records'),
            'missing': frame.isna().sum().to_dict(),
            'sha256': sha256(data / f'{name}.parquet'),
        }
    assert frames['ents'].ID.is_unique and frames['items'].IID.is_unique
    assert frames['events'].ENT_ID.isin(frames['ents'].ID).all()
    assert frames['events'].ITEM_ID.isin(frames['items'].IID).all()
    inspection['target_counts'] = {}
    if target_column:
        labels = frames['ents'][target_column]
        if target_column == 'TAGS':
            assert labels.map(len).between(1, len(tags)).all()
            labels = labels.explode()
        counts = labels.value_counts().to_dict()
        assert len(counts) == {'FLAG': 2, 'CATEGORY': 4, 'TAGS': 6}[target_column]
        inspection['target_counts'][target_column] = counts
    inspection['time_range'] = [str(frames['events'].TS.min()),
                                str(frames['events'].TS.max())]
    write_json(output / 'data-inspection.json', inspection)
    print((output / 'data-inspection.json').read_text(encoding='utf-8'))
    kumo.configure(scratch_dir=output / 'scratch')
    connector = kumo.FileConnector(root_dir=str(data))
    ents = kumo.Table.from_source_table(connector['ents.parquet'],
                                        primary_key='ID').infer_metadata()
    items = kumo.Table.from_source_table(connector['items.parquet'],
                                         primary_key='IID').infer_metadata()
    options = {} if task == 'static_link' else {'time_column': 'TS'}
    # Static link data has no time column at all, including inferred metadata.
    if task == 'static_link':
        frames['events'].drop(columns='TS').to_parquet(
            data / 'events.parquet', index=False)
        inspection['events']['sha256'] = sha256(data / 'events.parquet')
        inspection['events']['schema'].pop('TS')
        inspection['events']['sample'] = frames['events'].drop(
            columns='TS').head().to_dict('records')
        inspection['events']['missing'].pop('TS')
        write_json(output / 'data-inspection.json', inspection)
    events = kumo.Table.from_source_table(
        connector['events.parquet'], primary_key='EID', **options
    ).infer_metadata().with_column('ENT_ID', stype=Stype.ID).with_column(
        'ITEM_ID', stype=Stype.ID)
    graph = kumo.Graph({'ents': ents, 'items': items, 'events': events}, [
        Edge('events', 'ENT_ID', 'ents'), Edge('events', 'ITEM_ID', 'items')])
    graph.validate()
    snapshot = graph.snapshot()
    expected, query_text = TASKS[task]
    query = kumo.PredictiveQuery(graph, query_text)
    query.validate()
    assert query.get_task_type().value == expected
    if task in ('binary', 'multiclass', 'multilabel', 'multilabel_ranking'):
        generation = TrainingTableGenerationPlan(
            split='RandomSplit([0.6, 0.2, 0.2], seed=0)')
    elif task == 'static_link':
        generation = TrainingTableGenerationPlan(
            split='RandomLinkSplit([0.6, 0.2, 0.2])')
    else:
        ranges = [('2024-04-01', '2024-05-01'),
                  ('2024-05-01', '2024-06-01'),
                  ('2024-06-01', '2024-07-01')]
        days = 21 if task == 'forecasting' else 7
        generation = TrainingTableGenerationPlan(
            split=f'TimeRangeSplit({ranges!r}, timeframe={days}, '
                  'flexible_boundaries=False)', timeframe_step=7,
            **({'lag_timesteps': 3, 'year_over_year': False}
               if task == 'forecasting' else {}))
    table = query.generate_training_table(snapshot, plan=generation)
    table.export(kumo.TrainingTableOutputConfig(connector, 'training-rows'))
    rows = pd.read_parquet(data / 'training-rows')
    assert set(rows.SPLIT) == {0, 1, 2}, 'Require nonempty three-way split'
    write_json(output / 'training-rows.json', {
        'schema': rows.dtypes.astype(str).to_dict(),
        'split_counts': rows.SPLIT.value_counts().to_dict(),
        'sample': rows.head().to_dict('records')})
    if scenario == 'weighted_link':
        if task != 'temporal_link':
            raise ValueError('weighted_link currently requires temporal_link')
        from kumoapi.train import TrainingTableSpec
        weighted = rows.copy()
        # Link weights align to each row's target-list coordinates, following
        # test_custom_link_table_ingests_multiple_weighted_parts.
        weighted['WEIGHT'] = weighted.TARGET1.map(
            lambda values: [float(i + 1) for i in range(len(values))])
        assert all(len(w) == len(t) for w, t in zip(
            weighted.WEIGHT, weighted.TARGET1))
        weighted.to_parquet(data / 'weighted.parquet', index=False)
        table = kumo.TrainingTable.from_source_table(
            connector['weighted.parquet'],
            TrainingTableSpec(weight_col='WEIGHT', split_col='SPLIT'),
            pq=query, snapshot=snapshot, plan=generation)
        write_json(output / 'weight-oracle.json', {
            'file_sha256': sha256(data / 'weighted.parquet'),
            'split_counts': weighted.SPLIT.value_counts().to_dict(),
            'sample': weighted[['ENTITY1', 'TARGET1', 'WEIGHT', 'SPLIT']]
                      .head().to_dict('records'),
            'alignment': 'one positive weight per target coordinate',
            'weight_mode': 'weighted_loss'})
    spec = importlib.util.spec_from_file_location(
        'summary_smoke_reference', Path(repo) /
        'packages/kumo-engine/test/wheel_gpu_smoke.py')
    smoke = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(smoke)
    plan = smoke.make_plan(task)
    # Record the task's usual comparison metrics, not only the tune metric.
    plan.training_job.metrics = list(dict.fromkeys(
        (plan.training_job.tune_metric, *METRICS[task])))
    plan.optimization.max_epochs = 4
    plan.optimization.min_steps_per_epoch = 4
    plan.optimization.max_steps_per_epoch = 4
    if scenario == 'multi_trial':
        plan.training_job.num_experiments = 2
        plan.optimization.base_lr = [0.001, 0.01]
    elif scenario == 'many_trials':
        # Enough experiments to exercise the experiment dropdown.
        plan.training_job.num_experiments = 10
        plan.optimization.base_lr = [0.0002 * 2**i for i in range(10)]
    elif scenario == 'single_epoch':
        plan.optimization.max_epochs = 1
    elif scenario == 'early_stop':
        plan.optimization.max_epochs = 8
        plan.optimization.early_stopping = [{'min_delta': 1e9, 'patience': 1}]
    elif scenario == 'refit_trainval':
        plan.training_job.refit_trainval = True
    elif scenario == 'refit_full':
        plan.training_job.refit_full = True
    elif scenario == 'diagnostics':
        plan.training_job.disable_explain = False
    elif scenario == 'weighted_link':
        from kumoapi.model_plan import WeightMode
        plan.optimization.weight_mode = [WeightMode.WEIGHTED_LOSS]
    trainer = kumo.Trainer(plan, accelerator='gpu')
    write_json(output / 'environment.json', {
        'compute_grant': grant, 'python': sys.version, 'executable': sys.executable,
        'mlflow_visibility': visibility,
        'requested_output': str(requested_output), 'actual_output': str(output),
        'engine_path': kumo.__file__, 'verified_import_paths': imports,
        'torch': torch.__version__,
        'cuda': torch.version.cuda, 'gpu': torch.cuda.get_device_name(0),
        'task': task, 'scenario': scenario, 'query': query_text,
        'synthetic_generator': inspection['generator'],
        'expected_task': expected, 'plan_repr': repr(plan),
        'status': 'PREPARED_NOT_FITTED'})
    return {'trainer': trainer, 'snapshot': snapshot, 'table': table,
            'output': output, 'task': task, 'scenario': scenario}


def retain(ctx, result):
    """Retain raw inputs without invoking summary's selection/format helpers."""
    from kumo_core.io.filesystem import local_path
    from kumo_core.train.checkpoint import load_kumo_checkpoint
    root = ctx['output']
    shutil.copytree(result.bundle_dir, root / 'retained-bundle')
    oracle = root / 'oracle'
    oracle.mkdir()
    # QA-only retained dataclass data, never a serialized executable model.
    from dataclasses import asdict
    write_json(oracle / 'retained-summaries.json', [
        {'type': type(summary).__name__, 'data': asdict(summary)}
        for summary in result._summaries])
    raw = {}
    for path in result.bundle_dir.rglob('*.json'):
        relative = path.relative_to(result.bundle_dir)
        destination = oracle / 'bundle' / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, destination)
    for role, value in result.artifact_paths.items():
        path = Path(value)
        raw[role] = {'path': value, 'exists': path.exists()}
        if path.is_file():
            raw[role]['sha256'] = sha256(path)
            if path.suffix == '.json':
                destination = oracle / 'roles' / f'{role}.json'
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(path, destination)
    checkpoint = local_path(str(result.artifact_paths['selected_checkpoint']))
    saved = load_kumo_checkpoint(str(checkpoint), map_location='cpu')
    steps = saved.get('global_step')
    if not isinstance(steps, int) or steps < 1:
        raise AssertionError('Selected real checkpoint has no optimizer work')
    write_json(oracle / 'result.json', {
        'metrics': result.metrics, 'artifacts': raw,
        'manifest': result.manifest.to_dict() if hasattr(
            result.manifest, 'to_dict') else repr(result.manifest),
        'metric_history': result.metric_history,
        'checkpoint': {'epoch': saved.get('epoch'), 'global_step': steps,
                       'sha256': sha256(checkpoint)},
        'selection_review': 'PENDING_INDEPENDENT_COMPARISON',
        'visual_review': 'PENDING', 'coverage': 'REAL_FIT_RENDER_NOT_YET_CHECKED'})
    ctx['original_metrics'] = dict(result.metrics)
    ctx['fingerprint'] = inputs_fingerprint(result)


def check_render_unchanged(ctx, result):
    assert result.metrics == ctx['original_metrics']
    assert inputs_fingerprint(result) == ctx['fingerprint']
    write_json(ctx['output'] / 'checks.json', {
        'artifact_and_metric_stability': 'PASS',
        'selection_semantics': 'UNTESTED', 'visual_review': 'UNTESTED',
        'render_does_not_compute_metrics': 'UNTESTED',
        'note': 'Successful rendering alone is not semantic acceptance.'})


def check_report(result, output, label, expected=()):
    """Save an actual rendered report; require its fixed structure."""
    report = result.summary()
    rendered = report._repr_html_()
    text = repr(report)
    for item in ('data-panel="main:evaluation"', 'data-panel="main:training"',
                 *expected):
        assert item in rendered, f'{label}: missing {item!r}'
    assert 'Traceback (most recent call last)' not in rendered + text
    destination = Path(output)
    destination.mkdir(parents=True, exist_ok=True)
    (destination / f'{label}.html').write_text(rendered, encoding='utf-8')
    write_json(destination / f'{label}.json', {
        'expected': list(expected), 'text': text, 'structure': 'PASS',
        'semantic_review': 'UNTESTED'})
    return report


def variant(result, name, output):
    """Explicit fault injections on copies; never count as additional fits."""
    from dataclasses import replace
    from kumo.engine.explain.results import TrainingExplanations
    paths = dict(result.artifact_paths)
    if name == 'missing_history_role':
        paths.pop('metric_history', None)
        return replace(result, artifact_paths=paths)
    if name == 'missing_history_file':
        paths['metric_history'] = str(Path(output) / 'absent-history.json')
        return replace(result, artifact_paths=paths)
    if name == 'missing_diagnostics':
        return replace(result, explanations=TrainingExplanations.unavailable())
    if name == 'missing_holdout':
        return replace(result, holdout_path=None)
    if name == 'undefined_metrics_injected':
        return replace(result, metrics={key: float('nan')
                                        for key in result.metrics})
    if name == 'history_without_selection_derived':
        history = result.metric_history
        if history is None:
            raise RuntimeError('No real history available for derived fixture')
        # This is a derived compatibility fixture, not an old-release fit.
        removed = {}
        for key in ('selected_trial', 'selected_epoch', 'best_trial',
                    'best_epoch', 'selection'):
            if key in history:
                removed[key] = history.pop(key)
        destination = Path(output) / 'history-without-selection.json'
        write_json(destination, history)
        write_json(Path(output) / 'derived-history-differences.json', removed)
        paths['metric_history'] = str(destination)
        return replace(result, artifact_paths=paths)
    raise ValueError(name)


def notebook(repo, output, task, scenario, grant, kernel='result-summary',
             hide_mlflow=False):
    """Return notebook JSON without importing Engine or running any case."""
    harness = Path(__file__).resolve()
    cells = []
    def code(source):
        cells.append({'cell_type': 'code', 'metadata': {}, 'id': uuid4().hex[:12],
                      'execution_count': None, 'outputs': [],
                      'source': source.splitlines(keepends=True)})
    cells.append({'cell_type': 'markdown', 'metadata': {},
                  'id': uuid4().hex[:12], 'source': [
        f'# Real-fit summary validation: {task} / {scenario}\n',
        'UNTESTED until executed and independently reviewed.\n',
        'Source-adapted generated data; no external fixture downloads.\n']})
    code('import importlib.util\n'
         f'spec = importlib.util.spec_from_file_location("validation_harness", {str(harness)!r})\n'
         'h = importlib.util.module_from_spec(spec)\nspec.loader.exec_module(h)\n'
         f'ctx = h.prepare({str(repo)!r}, {str(output)!r}, {task!r}, {scenario!r}, {grant!r}, hide_mlflow={hide_mlflow!r})\n')
    if scenario == 'local_mlflow':
        probe = Path(__file__).with_name('local_mlflow_probe.py').resolve()
        code(f'probe_spec = importlib.util.spec_from_file_location("local_mlflow_probe", {str(probe)!r})\n'
             'probe = importlib.util.module_from_spec(probe_spec)\n'
             'probe_spec.loader.exec_module(probe)\n'
             'result = probe.run(ctx)\n')
    else:
        code('result = ctx["trainer"].fit(ctx["table"], snapshot=ctx["snapshot"])\n'
             'h.retain(ctx, result)\n')
    code('result.summary()\n')
    code('result.summary()\n')
    code('result.summary()\n')
    code('h.check_render_unchanged(ctx, result)\n'
         'import kumo.engine as kumo\n'
         'result.export(kumo.ModelOutputConfig(\n'
         '    kumo.FileConnector(root_dir=str(ctx["output"])), path="exported-model"))\n'
         'reloaded = kumo.Trainer.load(ctx["output"] / "exported-model", accelerator="gpu").result\n'
         'h.write_json(ctx["output"] / "oracle" / "reloaded-metrics.json", reloaded.metrics)\n')
    code('reloaded.summary()\n')
    code('h.check_report(reloaded, ctx["output"] / "reload-checks", "reload")\n')
    for name in ('missing_history_role', 'missing_history_file',
                 'missing_diagnostics', 'missing_holdout',
                 'undefined_metrics_injected',
                 'history_without_selection_derived'):
        code(f'# Fault injection on real-fit copy: {name}\n'
             f'h.variant(result, {name!r}, ctx["output"]).summary()\n')
    if hide_mlflow:
        code('h.assert_mlflow_hidden()\n'
             'h.write_json(ctx["output"] / "mlflow-import-hidden-check.json",\n'
             '    {"mode": "import-hidden", "uninstalled": False, "status": "PASS"})\n')
    return {'nbformat': 4, 'nbformat_minor': 5, 'metadata': {
        'kernelspec': {'display_name': kernel, 'language': 'python',
                       'name': kernel},
        'language_info': {'name': 'python'},
    }, 'cells': cells}


def reload_notebook(repo, bundle, grant, kernel, output, hide_mlflow=False):
    """Fresh-kernel rendering of an existing real-fit bundle; no new fit."""
    setup = (
        'import sys\nfrom pathlib import Path\n'
        f'assert {grant!r}.strip(), "Coordinator compute grant required"\n'
        'assert sys.platform == "linux"\n'
        f'repo = Path({str(repo)!r}).resolve()\n'
        'assert repo == Path("/home/ubuntu/kumo-engine-fork").resolve()\n'
        'assert Path(sys.executable).absolute() == repo / ".venv/bin/python"\n'
        'import importlib.util\n'
        f'spec = importlib.util.spec_from_file_location("validation_harness", {str(Path(__file__).resolve())!r})\n'
        'h = importlib.util.module_from_spec(spec)\nspec.loader.exec_module(h)\n'
        f'visibility = h.mlflow_visibility({hide_mlflow!r})\n'
        f'h.write_json(Path({str(Path(output).with_name(Path(output).name + "-environment.json"))!r}), visibility)\n'
        'import kumo.engine as kumo\nimport kumo_core\n'
        'for module in (kumo, kumo_core):\n'
        '    paths = ([Path(module.__file__).resolve()] if module.__file__ else\n'
        '             [Path(p).resolve() for p in module.__path__])\n'
        '    assert paths and all(p.is_relative_to(repo) for p in paths)\n'
        f'result = kumo.Trainer.load({str(bundle)!r}, accelerator="gpu").result\n'
        'assert result is not None\n'
        'print("Reloaded metrics:", result.metrics)\n'
    )
    return {'nbformat': 4, 'nbformat_minor': 5, 'metadata': {
        'kernelspec': {'display_name': kernel, 'language': 'python',
                       'name': kernel}, 'language_info': {'name': 'python'},
    }, 'cells': [
        {'cell_type': 'code', 'metadata': {}, 'id': uuid4().hex[:12],
         'execution_count': None,
         'outputs': [], 'source': source.splitlines(keepends=True)}
        for source in (setup, 'result.summary()\n', 'result.summary()\n',
                       'h.check_report(result, '
                       f'{str(output)!r}, "fresh-reload")\n',
                       'h.assert_mlflow_hidden()\n' if hide_mlflow else
                       'print("normal-imports; no package absence claim")\n')
    ]}


def fresh_paths(output, notebook_path):
    """Allocate matching rerun suffixes; never overwrite earlier evidence."""
    output, notebook_path = Path(output).resolve(), Path(notebook_path).resolve()
    prior = [output, notebook_path, notebook_path.with_suffix('.html'),
             notebook_path.with_suffix('.errors.json'),
             notebook_path.with_suffix('.trust.json'),
             notebook_path.with_suffix('.request.json')]
    if any(path.exists() for path in prior):
        suffix = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
        suffix += '-' + uuid4().hex[:10]
        output = output.with_name(f'{output.name}-rerun-{suffix}')
        notebook_path = notebook_path.with_name(
            f'{notebook_path.stem}-rerun-{suffix}{notebook_path.suffix}')
    return output, notebook_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--task', choices=TASKS, required=True)
    parser.add_argument('--scenario', choices=SCENARIOS, default='baseline')
    parser.add_argument('--notebook', type=Path, required=True)
    parser.add_argument('--compute-grant', default='')
    parser.add_argument('--kernel', default='result-summary')
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--hide-mlflow', action='store_true',
                        help='Hide MLflow imports in fresh kernels; no uninstall')
    parser.add_argument('--sign-notebook', action='store_true',
                        help='Optional trust evidence; unsigned is required for final visual QA')
    parser.add_argument('--reload-bundle', type=Path,
                        help='Render an existing bundle in a fresh kernel; no fit')
    args = parser.parse_args()
    if args.hide_mlflow and args.scenario == 'local_mlflow':
        parser.error('local_mlflow is incompatible with --hide-mlflow')
    requested_output, requested_notebook = args.output, args.notebook
    args.output, args.notebook = fresh_paths(args.output, args.notebook)
    artifact_root = args.repo.resolve() / 'artifacts/result-summary'
    if not all(path.is_relative_to(artifact_root)
               for path in (args.output, args.notebook)):
        parser.error('Output and notebook must be under artifacts/result-summary')
    args.notebook.parent.mkdir(parents=True, exist_ok=True)
    allocation = {
        'requested_output': str(requested_output),
        'requested_notebook': str(requested_notebook),
        'output': str(args.output), 'notebook': str(args.notebook),
        'command': sys.argv, 'status': 'ALLOCATED_NOT_EXECUTED',
        'mlflow_mode': 'import-hidden' if args.hide_mlflow else 'normal-imports',
    }
    write_json(args.notebook.with_suffix('.request.json'), allocation)
    print(json.dumps(allocation), flush=True)
    document = (reload_notebook(args.repo.resolve(), args.reload_bundle.resolve(),
                                args.compute_grant, args.kernel, args.output,
                                args.hide_mlflow)
                if args.reload_bundle is not None else
                notebook(args.repo.resolve(), args.output.resolve(), args.task,
                         args.scenario, args.compute_grant, args.kernel,
                         args.hide_mlflow))
    write_json(args.notebook, document)
    if not args.execute:
        print(f'UNTESTED notebook authored: {args.notebook}')
        return
    if not args.compute_grant.strip():
        parser.error('--execute requires explicit coordinator --compute-grant')
    import nbformat
    from nbclient import NotebookClient
    from jupyter_client.kernelspec import KernelSpecManager
    kernel_dir = args.repo.resolve() / '.venv/share/jupyter/kernels'
    manager = KernelSpecManager(kernel_dirs=[str(kernel_dir)])
    selected_kernel = manager.get_kernel_spec(args.kernel)
    if Path(selected_kernel.resource_dir).resolve() != kernel_dir / args.kernel:
        raise RuntimeError('Selected kernelspec is outside approved venv')
    if Path(selected_kernel.argv[0]).absolute() != (
            args.repo.resolve() / '.venv/bin/python'):
        raise RuntimeError('Selected kernelspec uses an unapproved interpreter')
    from jupyter_client import KernelManager
    kernel_manager = KernelManager(kernel_name=args.kernel,
                                    kernel_spec_manager=manager)
    nb = nbformat.read(args.notebook, as_version=4)
    execution_completed = False
    try:
        # Preserve independent fault-injection outcomes after an earlier error.
        NotebookClient(nb, timeout=3600, kernel_name=args.kernel,
                       km=kernel_manager, allow_errors=True).execute()
        execution_completed = True
    except Exception:
        traceback.print_exc()
        raise
    finally:
        # Passing our validated manager makes nbclient treat it as externally
        # owned, so this harness must explicitly release its kernel/GPU process.
        if kernel_manager.has_kernel:
            kernel_manager.shutdown_kernel(now=True)
        nbformat.write(nb, args.notebook)
        html_outputs = []
        for cell in nb.cells:
            for item in cell.get('outputs', []):
                value = item.get('data', {}).get('text/html')
                if value:
                    html_outputs.append(value)
        args.notebook.with_suffix('.html').write_text(
            '<!doctype html><meta charset="utf-8">\n' +
            '\n<hr>\n'.join(html_outputs), encoding='utf-8')
        errors = [
            {'cell': index, 'ename': item.get('ename'),
             'evalue': item.get('evalue'), 'traceback': item.get('traceback')}
            for index, cell in enumerate(nb.cells)
            for item in cell.get('outputs', [])
            if item.get('output_type') == 'error'
        ]
        write_json(args.notebook.with_suffix('.errors.json'), errors)
    if errors:
        raise RuntimeError(f'{len(errors)} notebook cell(s) failed; see errors JSON')
    if not execution_completed or any(
            cell.get('execution_count') is None for cell in nb.cells
            if cell.cell_type == 'code' and ''.join(cell.source).strip()):
        raise RuntimeError('Incomplete notebook must not be signed')
    # Normal Jupyter trust workflow: sign the exact successfully saved notebook
    # in this user's signature database. No trust is granted to failed outputs.
    from nbformat.sign import NotebookNotary
    saved_notebook = nbformat.read(args.notebook, as_version=4)
    notary = NotebookNotary()
    if args.sign_notebook:
        notary.sign(saved_notebook)
    signature_verified = notary.check_signature(saved_notebook)
    if args.sign_notebook and not signature_verified:
        raise RuntimeError('Saved notebook signature verification failed')
    if not args.sign_notebook and signature_verified:
        raise RuntimeError('Unsigned QA requires a fresh notebook without a prior signature')
    write_json(args.notebook.with_suffix('.trust.json'), {
        'status': ('SIGNED_SUCCESSFUL_EXECUTION' if args.sign_notebook else
                   'UNSIGNED_SUCCESSFUL_EXECUTION'),
        'notebook': str(args.notebook), 'sha256': sha256(args.notebook),
        'signature_verified': signature_verified,
        'mlflow_mode': allocation['mlflow_mode'],
        'visual_qa': 'UNTESTED'})


if __name__ == '__main__':
    main()
