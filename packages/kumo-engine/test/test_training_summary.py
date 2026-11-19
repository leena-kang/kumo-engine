# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import base64
import json
import re
from dataclasses import replace
from html import unescape
from pathlib import Path

import pytest
from kumo_core.summaries import Scalar, Table

from kumo.engine.trainer._summary import _chart, _ticks
from kumo.engine.trainer.trainer import TrainingResult


def _write(result, role, payload):
    Path(result.artifact_paths[role]).write_text(json.dumps(payload))


def _config(result, **changes):
    path = Path(result.artifact_paths['trainer_config'])
    config = json.loads(path.read_text())
    config.update(changes)
    path.write_text(json.dumps(config))


def _history(trials):
    return {
        'trial_progress': {
            str(trial): {
                'metrics': {
                    str(epoch): {
                        'train_metrics': {
                            f'train_{name}': value
                            for name, value in train.items()
                        },
                        'validation_metrics': {
                            f'val_{name}': value
                            for name, value in validation.items()
                        }
                    }
                    for epoch, (train, validation) in enumerate(epochs)
                }
            }
            for trial, epochs in trials.items()
        }
    }


@pytest.fixture
def result(tmp_path):
    artifacts = {
        'trainer_config': {
            'version': 2,
            'tune_metric': 'auprc',
            'refit_cfg': {
                'best_val_epoch': 1,
                'refitted_trainval': False,
                'refitted_full': False
            }
        },
        'prediction_schema': {
            'task_type': 'binary_classification'
        },
        'metric_history':
        _history({
            0: [({
                'loss': 99,
                'auprc': 99
            }, {})],
            2: [({
                'loss': .4,
                'auprc': .8
            }, {
                'loss': .5,
                'auprc': .7
            }), ({
                'loss': .5,
                'auprc': .75
            }, {
                'loss': .6,
                'auprc': .6
            })],
        }),
    }
    paths = {}
    for role, payload in artifacts.items():
        path = tmp_path / f'{role}.json'
        path.write_text(json.dumps(payload))
        paths[role] = str(path)
    return TrainingResult(
        metrics={
            'train_auprc': .75,
            'val_auprc': .6,
            'test_auprc': .65,
            'train_loss': .5,
            'val_loss': .6,
            'test_loss': .55,
            'test_auroc': .9,
        }, artifact_paths=paths, bundle_dir=tmp_path)


def _rows(html, table='ks-metrics'):
    body = re.search(rf'<table class="{table}">.*?</table>', html, re.S)
    assert body is not None
    return [[
        re.sub(r'<(?!/?b>)[^>]+>', '', cell)
        for cell in re.findall(r'<t[dh][^>]*>(.*?)</t[dh]>', row, re.S)
    ] for row in re.findall(r'<tr[^>]*>(.*?)</tr>', body.group(0), re.S)]


def _find(pattern, html):
    match = re.search(pattern, html)
    assert match is not None, pattern
    return match.group(1)


def _explanations(html):
    """Notes other than the list of metrics without a per-epoch history."""
    return [
        note for note in re.findall(r'<p class="ks-note">([^<]*)', html)
        if not note.startswith('No per-epoch history: ')
    ]


def _charts(html):
    return [(alt, json.loads(unescape(series)),
             base64.b64decode(source).decode())
            for alt, series, source in re.findall(
                r'<img class="ks-chart"[^>]* alt="([^"]*)" '
                r'data-series="([^"]*)" '
                r'src="data:image/svg\+xml;base64,([^"]*)">', html)]


def test_report_reads_recorded_results_without_touching_the_model(result):
    class NoModel:
        def __getattribute__(self, name):
            raise AssertionError(f'Model accessed: {name}')

    result._trainer = NoModel()
    before = {
        name: Path(path).read_bytes()
        for name, path in result.artifact_paths.items()
    }
    metrics = dict(result.metrics)
    first, second = result.summary(), result.summary()
    assert first is not second
    assert first._repr_html_() == second._repr_html_()
    assert result.metrics == metrics
    assert before == {
        name: Path(path).read_bytes()
        for name, path in result.artifact_paths.items()
    }


def test_report_has_two_tabs_and_no_headline_query_or_details(result):
    (result.bundle_dir / 'training-spec.json').write_text(
        json.dumps({'predictive_query':
                    'PREDICT users.FLAG FOR EACH users.ID'}))
    report = result.summary()
    html, text = report._repr_html_(), repr(report)
    assert re.findall(
        r'data-panel="main:(\w+)" data-label="(\w+)"( data-selected)?',
        html) == [('evaluation', 'Evaluation', ' data-selected'),
                  ('training', 'Training', '')]
    assert '<h2' not in html and 'higher is better' not in html + text
    assert 'PREDICT' not in html + text
    assert 'Availability' not in html + text and 'Model:' not in html + text
    # Controls exist only once the script builds them: an untrusted
    # notebook, which strips scripts, styles and ``hidden``, shows no dead
    # tab or slider, only both panels stacked under their titles.
    static = html.split('<script>')[0].split('</style>')[1]
    assert '<button' not in static and '<input' not in static
    assert '<select' not in static and ' hidden' not in static
    assert '<h3 class="ks-fallback-title">Training</h3>' in static


def test_metrics_table_compares_splits_at_the_best_epoch(result):
    report = result.summary()
    rows = _rows(report._repr_html_())
    assert rows[0] == ['Metric', 'Training', 'Validation', 'Holdout']
    # The tune metric leads; training and validation come from epoch 1 of
    # experiment 2, not from the fit-end values or from another experiment.
    assert rows[1] == ['AUPRC *', '<b>0.800</b>', '0.700', '0.650']
    assert rows[2] == ['AUROC', '-', '-', '0.900']
    assert len(rows) == 3 and 'Loss' not in repr(report).split('Learning')[0]
    assert 'AUPRC * | 0.800 | 0.700 | 0.650' in repr(report)
    assert '<tr class="ks-tune">' in report._repr_html_()


def test_best_split_is_bold_by_metric_direction(result):
    result = replace(result, metrics={
        'train_mae': 1.0, 'val_mae': 2.0, 'test_mae': 3.0,
        'train_auroc': .5, 'val_auroc': .7, 'test_auroc': .7,
        'train_acc': .5, 'val_acc': .6, 'test_acc': .4,
        'train_ap': .5, 'val_ap': .50004, 'test_ap': .5,
        'test_unknown': 1.0, 'val_unknown': 2.0,
    })  # yapf: disable
    _config(result, tune_metric='mae', refit_cfg={})
    rows = {row[0]: row[1:] for row in _rows(result.summary()._repr_html_())}
    assert rows['MAE *'] == ['<b>1.000</b>', '2.000', '3.000']
    # Splits that tie for best, or display the same, have no best one.
    assert rows['AUROC'] == ['0.500', '0.700', '0.700']
    assert rows['Accuracy'] == ['0.500', '<b>0.600</b>', '0.400']
    assert rows['AP'] == ['0.500', '0.500', '0.500']
    assert rows['UNKNOWN'] == ['-', '2.000', '1.000']


def test_best_epoch_values_take_precedence_over_fit_end_values(result):
    # The best epoch recorded no validation AUPRC: the fit-end value shows.
    _write(result, 'metric_history', _history({2: [({'auprc': .8}, {})]}))
    report = result.summary()
    assert _rows(report._repr_html_())[1] == [
        'AUPRC *', '<b>0.800</b>', '0.600', '0.650'
    ]
    assert '†' not in repr(report) + report._repr_html_()


@pytest.mark.parametrize('tune,order', [('recall@10', ['Recall', 'MAP']),
                                        ('map@10', ['MAP', 'Recall'])])
def test_ranking_metrics_group_by_cutoff_with_the_tune_group_first(
        result, tune, order):
    result = replace(result, metrics={
        f'{split}_{name}@{k}': value
        for split in ('val', 'test')
        for name, value in (('map', .1), ('recall', .2))
        for k in (1, 10, 100)
    })  # yapf: disable
    _config(result, tune_metric=tune, refit_cfg={})
    html = result.summary()._repr_html_()
    rows = _rows(html)
    assert rows[0] == ['Metric', 'K', 'Training', 'Validation', 'Holdout']
    assert [row[:2] for row in rows[1:]] == [
        [order[0], '1'], ['', '10 *'], ['', '100'],
        [order[1], '1'], ['', '10'], ['', '100'],
    ]  # yapf: disable
    assert html.count('<tr class="ks-tune ks-group">') == 1


@pytest.mark.parametrize(
    'mode,refit_metrics,header',
    [
        ('trainval', True, 'Holdout'),
        ('trainval', False, 'Holdout (before refit)'),
        ('full', False, 'Holdout (before refit)'),
        # A full refit after a train+validation refit: the latter's holdout is
        # the latest evidence, and still precedes the published model.
        ('full', True, 'Holdout (before refit)'),
    ])
def test_refit_never_presents_search_holdout_as_the_refitted_model(
        result, mode, refit_metrics, header):
    metrics = dict(result.metrics)
    if refit_metrics:
        metrics['test_refit_auprc'] = .9
    result = replace(result, metrics=metrics)
    _config(
        result, refit_cfg={
            'best_val_epoch': 1,
            'refitted_trainval': mode == 'trainval',
            'refitted_full': mode == 'full'
        })
    report = result.summary()
    rows = _rows(report._repr_html_())
    assert rows[0][-1] == header and not _explanations(report._repr_html_())
    expected = '0.900' if refit_metrics else '0.650'
    assert rows[1][-1].strip('<b>/') == expected
    assert rows[1][2] == '0.700'  # Validation stays the search fit's.


def test_missing_selection_does_not_guess_an_experiment(result):
    _config(result, version=None)
    html = result.summary()._repr_html_()
    assert '★' not in html and 'Best epoch' not in ''.join(
        svg for _, _, svg in _charts(html))
    # Without a best epoch, training and validation are the fit-end values.
    assert _rows(html)[1] == ['AUPRC *', '<b>0.750</b>', '0.600', '0.650']


@pytest.mark.parametrize('failure', ['missing', 'malformed', 'empty'])
def test_absent_history_and_metadata_are_safe(result, tmp_path, failure):
    path = Path(result.artifact_paths['metric_history'])
    if failure == 'missing':
        path.unlink()
    else:
        path.write_text('{' if failure == 'malformed' else '{}')
    report = result.summary()
    assert 'No training history retained.' in report._repr_html_()
    assert not _charts(report._repr_html_())
    empty = TrainingResult({}, {}, tmp_path).summary()
    assert 'No evaluation metrics recorded.' in repr(empty)
    assert 'No training history retained.' in repr(empty)


def test_every_experiment_has_curves_and_the_best_is_shown_first(result):
    report = result.summary()
    html = report._repr_html_()
    # The script turns these into a dropdown with the best selected.
    assert re.findall(
        r'<(\w+) class="ks-panel" data-panel="experiment:(\d+)" '
        r'data-label="([^"]+)"( data-selected)?>', html) == [
            # Without scripting the other experiments are collapsed.
            ('details', '0', 'Experiment 1', ''),
            ('div', '2', 'Experiment 3 ★', ' data-selected'),
        ]
    assert 'Learning curves: Experiment 3 ★' in repr(report)
    charts = _charts(html)
    assert [alt for alt, _, _ in charts] == [
        'Loss by epoch', 'AUPRC by epoch', 'Loss by epoch', 'AUPRC by epoch'
    ]
    # Epochs are one-based and values are the recorded ones.
    assert charts[2][1] == {
        'train': [[1, .4], [2, .5]],
        'val': [[1, .5], [2, .6]]
    }
    assert charts[0][1] == {'train': [[1, 99]], 'val': [[1, None]]}
    # Only the best experiment marks the best epoch.
    assert ['Best epoch' in svg
            for _, _, svg in charts] == [False, False, True, True]
    assert 'Validation' not in charts[0][2] and 'Validation' in charts[2][2]
    assert '<svg' not in html and '<script' not in ''.join(
        svg for _, _, svg in charts)
    # AUROC is holdout-only, so it has no curve and is named as such.
    assert 'No per-epoch history: AUROC.' in html


def test_curve_points_carry_their_hover_labels(result):
    html = result.summary()._repr_html_()
    points = json.loads(
        unescape(_find(r'data-metric="auprc" data-points="([^"]*)"', html)))
    assert [label for _, _, label in points] == ['AUPRC, epoch 1: Train 99.000']
    best = re.findall(r'data-metric="auprc" data-points="([^"]*)"', html)[1]
    assert [label for _, _, label in json.loads(unescape(best))] == [
        'AUPRC, epoch 1: Validation 0.700', 'AUPRC, epoch 2: Validation 0.600',
        'AUPRC, epoch 1: Train 0.800', 'AUPRC, epoch 2: Train 0.750'
    ]


def test_every_metric_with_history_has_curves_tune_metric_first(result):
    _write(
        result, 'metric_history',
        _history({
            2: [({
                'loss': .5,
                'recall@1': .1,
                'recall@5': .2,
                'acc': .3
            }, {
                'recall@5': .4
            })]
        }))
    result = replace(result, metrics={
        'val_recall@1': .1, 'val_recall@5': .2, 'val_acc': .3,
        'test_recall@5': .4, 'test_mrr': .5,
    }, _summaries=(Scalar('test/logloss', .6, 'model'),))  # yapf: disable
    _config(result, tune_metric='recall@5')
    html = result.summary()._repr_html_()
    assert re.findall(r'data-metric="([^"]*)"',
                      html) == ['loss', 'recall@5', 'recall@1', 'accuracy']
    # Without scripting, metrics other than the tune metric are closed.
    lead, more = html.split('<details class="ks-more"><summary>More metrics')
    assert lead.count('<img class="ks-chart"') == 2
    assert more.count('<img class="ks-chart"') == 2
    assert 'alt="Recall@5 by epoch"' in lead
    assert 'data-tune="recall@5"' in html
    assert 'No per-epoch history: Log loss, MRR.' in html


def test_one_experiment_needs_no_selector(result):
    _write(result, 'metric_history', _history({2: [({'loss': .4}, {})]}))
    html = result.summary()._repr_html_()
    assert 'data-panel="experiment' not in html and len(_charts(html)) == 1


def _panels(html):
    """Each experiment panel's chart metrics and missing-history note."""
    return {
        key:
        (re.findall(r'data-metric="([^"]*)"',
                    body), _find(r'No per-epoch history: ([^<]*)\.</p>', body))
        for key, body in re.findall(
            r'data-panel="experiment:(\d+)"(.*?)(?=data-panel="experiment:|'
            r'<script>)', html, re.S)
    }


def test_curve_availability_follows_each_experiments_finite_values(result):
    # The best experiment records AP only as nulls and no MRR; another
    # experiment records MRR. Each experiment offers what it recorded.
    _write(result, 'metric_history',
           _history({0: [({'loss': .5, 'mrr': .3}, {})],
                     2: [({'loss': .4, 'auprc': .8, 'ap': None},
                          {'ap': None})]}))  # yapf: disable
    result = replace(result, metrics=dict(result.metrics, val_ap=.2,
                                          test_mrr=.1))
    assert _panels(result.summary()._repr_html_()) == {
        '0': (['loss', 'mrr'], 'AUPRC, AP, AUROC'),
        '2': (['loss', 'auprc'], 'AP, AUROC, MRR'),
    }


def test_overlapping_points_share_one_hover_label(result):
    _write(
        result, 'metric_history',
        _history({
            2: [({
                'auprc': .5
            }, {
                'auprc': .5
            }), ({
                'auprc': .6
            }, {
                'auprc': .7
            })]
        }))
    html = result.summary()._repr_html_()
    points = json.loads(
        unescape(_find(r'data-metric="auprc" data-points="([^"]*)"', html)))
    assert [label for _, _, label in points] == [
        'AUPRC, epoch 1: Validation 0.500, Train 0.500',
        'AUPRC, epoch 2: Validation 0.700', 'AUPRC, epoch 2: Train 0.600'
    ]


def test_loss_tune_metric_draws_one_loss_chart(result):
    _config(result, tune_metric='loss')
    report = result.summary()
    assert [alt for alt, _, _ in _charts(report._repr_html_())] == [
        'Loss by epoch', 'AUPRC by epoch', 'Loss by epoch', 'AUPRC by epoch'
    ]
    assert not [row for row in _rows(report._repr_html_()) if 'Loss' in row[0]]


def test_chart_breaks_lines_at_gaps_and_marks_the_best_epoch():
    html = _chart(
        'AUROC', {
            'train': [(1, .1), (2, None), (3, .3), (4, .4), (6, .6)],
            'val': [(1, None), (2, None)]
        }, 3, 'auroc')
    (_, series, svg), = _charts(html)
    assert series['train'][1] == [2, None]
    path = re.search(r'<path d="(M[^"]*)" fill="none" stroke="#115e59"', svg)
    assert path is not None
    # Epochs 1 | 3-4 | 6: neither the missing value nor the missing epoch
    # is bridged.
    assert re.sub(r'[\d. ]', '', path.group(1)) == 'MMLM'
    assert svg.count('<circle') == 4 + 1  # Four points and one legend dot.
    assert 'Best epoch' in svg and 'Validation' not in svg
    assert 'Best epoch' not in _charts(
        _chart('AUROC', {
            'train': [(1, .1)],
            'val': []
        }, 5, 'auroc'))[0][2]


@pytest.mark.parametrize('low,high', [(0.0, 1.0), (.52, .61), (-3.0, 250.0),
                                      (1e-5, 3e-5), (2.0, 2.0)])
def test_axis_ticks_are_round_and_inside_the_range(low, high):
    ticks = _ticks(low, high)
    assert 1 <= len(ticks) <= 7
    assert all(low - 1e-12 <= tick <= high + 1e-12 for tick in ticks)
    assert ticks == sorted(set(ticks))


def _thresholds():
    rows = []
    for i in range(101):
        tp, fp = max(0, 15 - i // 5), max(0, 26 - i // 3)
        # Recorded rates are deliberately not the counts' own: the report
        # must show what Core recorded, not recompute it.
        rows.append((i / 100, .9, tp, fp, 15 - tp, 26 - fp,
                     None if i == 100 else .11, .22, .33))
    return Table('holdout/by_threshold',
                 ('threshold', 'share at or above', 'tp', 'fp', 'fn', 'tn',
                  'precision', 'recall', 'fpr'), tuple(rows))


def test_binary_confusion_matrix_has_a_threshold_slider(result):
    result = replace(
        result, _summaries=(Scalar('test/logloss', .5, 'model'), _thresholds(),
                            Table('holdout/calibration', ('from', 'rows'),
                                  ((0.0, 1), ))))
    report = result.summary()
    html = report._repr_html_()
    assert 'Calibration' not in html and 'calibration' not in html
    assert ['Log loss', '-', '-', '0.500'] in _rows(html)
    states = json.loads(unescape(_find(r'data-states="([^"]*)"', html)))
    assert len(states) == 101
    # threshold 0.50: tp=5, fp=10, fn=10, tn=16 of 41 rows.
    state = states[50]
    assert [cell[0] for cell in state['cells']
            ] == ['16 (39.0%)', '10 (24.4%)', '10 (24.4%)', '5 (12.2%)']
    assert state['cells'][0][1] == 'rgb(201,222,225)'
    assert state['scores'] == ['0.50', '0.11', '0.22', '0.33', '0.90']
    # The static page shows the same state the slider starts on.
    matrix = _rows(html, 'ks-matrix')
    assert matrix[1] == ['Actual', 'False', 'True']
    assert matrix[2] == ['False', '16 (39.0%)', '10 (24.4%)']
    assert matrix[3] == ['True', '10 (24.4%)', '5 (12.2%)']
    assert _rows(html, 'ks-scores') == [['Threshold', '0.50'],
                                        ['Precision', '0.11'],
                                        ['Recall', '0.22'],
                                        ['False Positive Rate', '0.33'],
                                        ['Predicted Positive Rate', '0.90']]
    assert states[100]['scores'][1] == 'N/A'  # Recorded as undefined.
    assert 'data-start="50"' in html and not _explanations(html)


def test_slider_starts_at_the_fitted_threshold(result):
    result = replace(
        result, _summaries=(_thresholds(),
                            Table('holdout/at_threshold_0.30', ('actual', ), (),
                                  brief=True)))
    html = result.summary()._repr_html_()
    assert 'data-start="30"' in html
    assert _rows(html, 'ks-scores')[0] == ['Threshold', '0.30']


def test_threshold_slider_is_only_for_binary_classification(result):
    _write(result, 'prediction_schema',
           {'task_type': 'multilabel_classification'})
    result = replace(
        result, _summaries=(_thresholds(),
                            Table('holdout/by_label', ('label', 'f1', 'r2'),
                                  (('<a>', .5, 1), ))))
    html = result.summary()._repr_html_()
    assert 'class="ks-threshold"' not in html
    assert _rows(html, 'ks-plain') == [['Label', 'F1', 'R²'],
                                       ['&lt;a&gt;', '0.500', '1']]
    assert 'Per-Label Metrics' in html


def test_multiclass_shows_per_class_metrics_without_a_matrix(result):
    from kumo_core.summaries import Confusion
    _write(result, 'prediction_schema',
           {'task_type': 'multiclass_classification'})
    result = replace(
        result, _summaries=(Confusion('holdout/confusion', ('a', 'b'),
                                      ((6, 2), (0, 2))),
                            Table('holdout/by_class', ('class', 'support'),
                                  (('a', 8), ))))
    html = result.summary()._repr_html_()
    assert 'ks-matrix">' not in html and 'Confusion' not in html
    assert _rows(html, 'ks-plain') == [['Class', 'Support'], ['a', '8']]


def test_labels_are_escaped(result):
    _config(result, tune_metric='<script>')
    html = result.summary()._repr_html_()
    assert html.count('<script>') == 1  # Only the report's own script.


def test_report_saves_as_a_standalone_page(result, tmp_path):
    summary = result.summary()
    page = summary.save(tmp_path / 'report.html', 'a<b training summary')

    text = page.read_text(encoding='utf-8')
    assert text.startswith('<!doctype html><html lang="en"><head>')
    assert '<title>a&lt;b training summary</title>' in text
    assert text.endswith(f'<body>{summary._repr_html_()}</body></html>')


def test_a_tracked_fit_logs_its_report_as_a_page(result):
    from kumo.engine.trainer._run_spec import SUMMARY_FILE, summary_artifacts

    assert summary_artifacts(result, 'churn') == {
        SUMMARY_FILE: result.summary().to_html('churn training summary')
    }


def test_a_report_that_cannot_render_is_left_out(caplog):
    from kumo.engine.trainer._run_spec import summary_artifacts

    class Broken:
        def summary(self):
            raise OSError('scratch directory is gone')

    assert summary_artifacts(Broken()) == {}
    assert 'Could not render the training summary: scratch directory is gone' \
        in caplog.text
