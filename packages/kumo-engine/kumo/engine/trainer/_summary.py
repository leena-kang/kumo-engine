# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Read-only notebook report of a fitted model: evaluation and training."""

import base64
import json
import math
from dataclasses import dataclass
from html import escape
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from kumo_core.summaries import Scalar, Table

from kumo.engine._tracking import logged_names, metric_name
from kumo.engine.trainer._spec import SPEC_FILE

Points = List[Tuple[int, Optional[float]]]

_SPLITS = (('train', 'Training'), ('val', 'Validation'), ('test', 'Holdout'))
_TRAIN, _VALIDATION, _BEST = '#115e59', '#76b900', '#64748b'
_LABELS = {
    'loss': 'Loss',
    'accuracy': 'Accuracy',
    'precision': 'Precision',
    'recall': 'Recall',
    'logloss': 'Log loss',
    'brier': 'Brier score',
    'bias': 'Bias',
    'r2': 'R²',
}
_HOLDOUT = ' class="ks-holdout"'
# Core's ``holdout/by_threshold`` columns shown beside the confusion matrix.
_THRESHOLD_COLUMNS = {
    'threshold': 'Threshold',
    'precision': 'Precision',
    'recall': 'Recall',
    'fpr': 'False Positive Rate',
    'share at or above': 'Predicted Positive Rate',
}
_TASK_TABLES = {
    'multiclass_classification': ('holdout/by_class', 'Per-Class Metrics'),
    'multilabel_classification': ('holdout/by_label', 'Per-Label Metrics'),
    'forecasting': ('holdout/by_horizon', 'By Forecast Horizon'),
}


@dataclass(frozen=True)
class TrainingSummary:
    """An independent HTML/text snapshot; displaying it performs no reads."""

    _html: str
    _text: str

    def _repr_html_(self) -> str:
        return self._html

    def __repr__(self) -> str:
        return self._text


def _number(value: Any) -> Optional[float]:
    return (float(value) if isinstance(value, (int, float))
            and not isinstance(value, bool) and math.isfinite(value) else None)


def _cell(value: Any) -> str:
    if isinstance(value, float):
        if not math.isfinite(value):
            return '-'
        return f'{value:.2e}' if 0 < abs(value) < .0005 else f'{value:.3f}'
    return '-' if value is None else str(value)


def _metric_title(name: str) -> str:
    base, _, averaging = name.partition('_')
    title = _LABELS.get(base, base.upper())
    return f'{title} ({averaging.replace("_", " ")})' if averaging else title


def _column_title(name: str) -> str:
    """A diagnostic table's header: metric columns read as in the table above."""
    return (_metric_title(name) if name in ('mae', 'rmse', 'r2',
                                            'f1') else name.capitalize())


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, dict) else {}


def _read(path: Any) -> Mapping[str, Any]:
    """Retained JSON, or nothing when it is absent or unreadable."""
    try:
        return _mapping(json.loads(Path(path).read_text(encoding='utf-8')))
    except (OSError, ValueError, TypeError):
        return {}


def _lower_is_better(name: str, cache: Dict[str,
                                            Optional[bool]]) -> Optional[bool]:
    if name not in cache:
        from kumo_core.evaluation.suite import MetricStore

        try:
            # Core names accuracy ``acc``; ``metric_name`` renamed it.
            cache[name] = MetricStore.tune_mode('acc' if name ==
                                                'accuracy' else name) == 'min'
        except Exception:  # A direction is optional: unknown bolds nothing.
            cache[name] = None
    return cache[name]


def training_summary(result: Any) -> TrainingSummary:
    paths = result.artifact_paths
    config = _read(paths.get('trainer_config'))
    plan = _read(paths.get('resolved_model_plan'))
    task = _read(paths.get('prediction_schema')).get('task_type')
    try:
        mode = getattr(result.manifest, 'fit_mode', None)
    except (OSError, ValueError, KeyError, TypeError):
        mode = None
    refit = _mapping(config.get('refit_cfg'))
    full = mode == 'refit_full' or refit.get('refitted_full') is True
    trainval = mode == 'refit_trainval' or refit.get(
        'refitted_trainval') is True
    job = _mapping(plan.get('training_job')) or _mapping(
        _mapping(_read(result.bundle_dir /
                       SPEC_FILE).get('model_plan')).get('training_job'))
    tune = config.get('tune_metric') or job.get('tune_metric')
    tune = tune if isinstance(tune, str) and tune != '???' else None
    try:
        history = _mapping(result.metric_history)
    except (OSError, ValueError, TypeError):
        history = {}
    trials = {
        key: _mapping(_mapping(trial).get('metrics'))
        for key, trial in _mapping(history.get('trial_progress')).items()
    }
    trials = {key: epochs for key, epochs in trials.items() if epochs}
    best = None if config.get('version') is None else str(config['version'])
    epoch = refit.get('best_val_epoch')
    epoch = (epoch if isinstance(epoch, int) and not isinstance(epoch, bool)
             and epoch > 0 else None)

    # The published model is the best experiment's best epoch: what that
    # epoch recorded takes precedence over the fit-end value.
    at_best = _mapping(trials.get(best or '', {}).get(str(epoch -
                                                          1))) if epoch else {}
    metrics = logged_names(result.metrics, refit_full=full)
    for field in ('train_metrics', 'validation_metrics'):
        metrics.update((name, value) for name, value in logged_names(
            _mapping(at_best.get(field))).items() if _number(value) is not None)
    # Never present a pre-refit holdout as the refitted model's.
    holdout = 'before_refit' if full else 'test'
    before_refit = full or (trainval and not any(
        name.startswith('test_refit_') for name in result.metrics))
    values = {
        **{
            s.name: s.value
            for s in result._summaries if isinstance(s, Scalar)
        },
        **metrics
    }
    primary = metric_name(tune) if tune else None
    names = sorted({
        name.split('/', 1)[1]
        for name, value in values.items() if name.startswith(
            ('train/', 'val/', f'{holdout}/')) and _number(value) is not None
    } - {'loss'}, key=lambda name: _order(name, primary))

    text: List[str] = []
    evaluation = (
        _metrics_table(names, values, holdout, primary, before_refit, text) +
        _diagnostics(result._summaries, str(task)))
    training = _curves(trials, best, epoch, names, primary, text)
    return TrainingSummary(
        '<div class="kumo-training-summary">' + _STYLE +
        _panel('main', 'evaluation', 'Evaluation', evaluation) +
        _panel('main', 'training', 'Training', training, False) + '</div>' +
        _SCRIPT, '\n'.join(text))


def _order(name: str, primary: Optional[str]) -> Tuple[bool, str, int]:
    """The tune metric's group first; each ``metric@k`` group by k."""
    metric, _, cutoff = name.partition('@')
    return (metric != (primary or '').partition('@')[0], metric,
            int(cutoff) if cutoff.isdigit() else 0)


def _panel(group: str, key: str, label: str, body: str,
           selected: bool = True) -> str:
    """One panel. Scripting switches ``main`` panels with tabs, others with a
    dropdown.

    Without scripting (an untrusted notebook strips it, with styles and
    ``hidden``), panels are stacked under their titles, and unselected
    dropdown panels are native disclosures.
    """
    attributes = (f'class="ks-panel" data-panel="{group}:'
                  f'{escape(key, quote=True)}" data-label="'
                  f'{escape(label, quote=True)}"' +
                  (' data-selected' if selected else ''))
    if group != 'main' and not selected:
        return (f'<details {attributes}><summary class="ks-fallback-title">'
                f'{escape(label)}</summary>{body}</details>')
    return (f'<div {attributes}><h3 class="ks-fallback-title">'
            f'{escape(label)}</h3>{body}</div>')


def _metrics_table(names: Sequence[str], values: Mapping[str, Any],
                   holdout: str, primary: Optional[str], before_refit: bool,
                   text: List[str]) -> str:
    splits = [
        (key if key != 'test' else holdout,
         label + (' (before refit)' if key == 'test' and before_refit else ''))
        for key, label in _SPLITS
    ]
    at_k = any('@' in name for name in names)
    header = ['Metric', *(['K'] if at_k else []), *(s[1] for s in splits)]
    text.append('Evaluation metrics')
    text.append(' | '.join(header))
    if not names:
        text.append('No evaluation metrics recorded.')
        return ('<h4>Evaluation Metrics</h4>'
                '<p class="ks-note">No evaluation metrics recorded.</p>')
    rows: List[str] = []
    previous = None
    directions: Dict[str, Optional[bool]] = {}
    for index, name in enumerate(names):
        metric, _, cutoff = name.partition('@')
        numbers = [
            _number(values.get(f'{split}/{name}')) for split, _ in splits
        ]
        # Compare what is displayed: bold the one best split, and none when
        # several display the same best value.
        shown = [_cell(number) for number in numbers]
        present = [float(cell) for cell in shown if cell != '-']
        lower = _lower_is_better(name, directions) if present else None
        top = None
        if lower is not None:
            best = (min if lower else max)(present)
            top = best if len(present) > 1 and present.count(
                best) == 1 else None
        tuned = name == primary
        star = ' *' if tuned else ''
        title = _metric_title(metric) + ('' if cutoff else star)
        cells = [
            f'<b>{cell}</b>' if cell != '-' and float(cell) == top else cell
            for cell in shown
        ]
        nxt = names[index +
                    1].partition('@')[0] if index + 1 < len(names) else None
        classes = ' '.join(
            filter(None, ('ks-tune' if tuned else '',
                          'ks-group' if at_k and nxt == metric else '')))
        rows.append(f'<tr class="{classes}"><td>' +
                    ('' if cutoff and metric == previous else escape(title)) +
                    '</td>' +
                    (f'<td>{escape(cutoff)}{star if cutoff else ""}</td>'
                     if at_k else '') + ''.join(f'<td><span>{cell}</span></td>'
                                                for cell in cells) + '</tr>')
        text.append(' | '.join(
            (title, *([cutoff + (star if cutoff else '')] if at_k else []),
             *shown)))
        previous = metric
    return ('<h4>Evaluation Metrics</h4><table class="ks-metrics"><thead><tr>' +
            ''.join(f'<th{_HOLDOUT if label.startswith("Holdout") else ""}>'
                    f'{escape(label)}</th>' for label in header) +
            '</tr></thead><tbody>' + ''.join(rows) + '</tbody></table>')


def _diagnostics(summaries: Sequence[Any], task: str) -> str:
    html = ''
    for summary in summaries:
        if (task == 'binary_classification' and isinstance(summary, Table)
                and summary.name == 'holdout/by_threshold'):
            default = next((_number(_float(s.name.rsplit('_', 1)[-1]))
                            for s in summaries if isinstance(s, Table)
                            and s.name.startswith('holdout/at_threshold_')),
                           None)
            html += _thresholds(summary, .5 if default is None else default)
        elif isinstance(summary, Table) and _TASK_TABLES.get(
                task, ('', ))[0] == summary.name:
            html += (
                f'<h4>{escape(_TASK_TABLES[task][1])}</h4>'
                '<div class="ks-scroll"><table class="ks-plain"><thead><tr>' +
                ''.join(f'<th>{escape(_column_title(str(column)))}</th>'
                        for column in summary.columns) +
                '</tr></thead><tbody>' +
                ''.join('<tr>' + ''.join(f'<td>{escape(_cell(value))}</td>'
                                         for value in row) + '</tr>'
                        for row in summary.rows) + '</tbody></table></div>')
    return html


def _float(value: str) -> Optional[float]:
    try:
        return float(value)
    except ValueError:
        return None


def _matrix_cell(count: int, total: float) -> Tuple[str, str]:
    """The UI's cell: count, its share of all rows, and a white-teal blend."""
    share = count / total if total else 0.0
    color = ','.join(
        str(round(255 + (channel - 255) * share))
        for channel in (116, 171, 179))
    return f'{count:,} ({share * 100:.1f}%)', f'rgb({color})'


def _confusion(
    labels: Sequence[str], cells: Sequence[Sequence[Tuple[str, str]]],
    marks: Sequence[Sequence[str]] = ()
) -> str:
    head = ''.join(f'<th>{escape(str(label))}</th>' for label in labels)
    body = ''.join(f'<tr><th>{escape(str(label))}</th>' + ''.join(
        f'<td{f" data-cell={chr(34)}{marks[i][j]}{chr(34)}" if marks else ""} '
        f'style="background:{color}">{escape(shown)}</td>'
        for j, (shown, color) in enumerate(row)) + '</tr>'
                   for i, (label, row) in enumerate(zip(labels, cells)))
    return ('<div class="ks-scroll"><table class="ks-matrix"><thead><tr>'
            f'<th></th><th class="ks-axis" colspan="{len(labels)}">Predicted'
            f'</th></tr><tr><th class="ks-axis">Actual</th>{head}</tr></thead>'
            f'<tbody>{body}</tbody></table></div>')


def _thresholds(table: Table, default: float) -> str:
    """Holdout confusion at every retained threshold; a slider picks one.

    Shows Core's recorded columns as they are: its counts are of calibrated
    probabilities, unlike the fit's own threshold metrics in the table above.
    """
    states, thresholds = [], []
    for row in table.rows:
        at = dict(zip(table.columns, row))
        thresholds.append(float(at['threshold']))
        counts = [int(at[name]) for name in ('tn', 'fp', 'fn', 'tp')]
        states.append({
            'cells': [_matrix_cell(count, sum(counts)) for count in counts],
            'scores': [
                'N/A' if _number(at[name]) is None else f'{at[name]:.2f}'
                for name in _THRESHOLD_COLUMNS
            ],
        })
    if not states:
        return ''
    start = min(range(len(states)), key=lambda i:
                (abs(thresholds[i] - default), i))
    cells = states[start]['cells']
    return (
        '<h4>Classification Threshold</h4>'
        f'<div class="ks-threshold" data-states="'
        f'{escape(json.dumps(states, separators=(",", ":")), quote=True)}" '
        f'data-start="{start}"><div class="ks-tradeoff">' + _confusion(
            ('False', 'True'), (cells[:2], cells[2:]),
            (('tn', 'fp'), ('fn', 'tp'))) + '<table class="ks-scores"><tbody>' +
        ''.join(f'<tr><td>{label}</td><td data-score="{i}">{score}</td></tr>'
                for i, (label, score) in enumerate(
                    zip(_THRESHOLD_COLUMNS.values(), states[start]['scores'])))
        + '</tbody></table></div></div>')


def _series(epochs: Mapping[str, Any], metric: str) -> Dict[str, Points]:
    series: Dict[str, Points] = {}
    for split, field in (('train', 'train_metrics'), ('val',
                                                      'validation_metrics')):
        series[split] = sorted((
            int(epoch) + 1,
            _number(
                _mapping(_mapping(values).get(field)).get(f'{split}_{metric}')))
                               for epoch, values in epochs.items()
                               if str(epoch).isdigit())
    return series


def _curve_title(name: str) -> str:
    metric, _, cutoff = name.partition('@')
    return _metric_title(metric) + (f'@{cutoff}' if cutoff else '')


def _recorded(epochs: Mapping[str, Any]) -> Dict[str, str]:
    """Metrics with a finite per-epoch value in one experiment's history,
    by Evaluation name, mapped to their history name."""
    recorded: Dict[str, str] = {}
    for values in epochs.values():
        for split, field in (('train', 'train_metrics'),
                             ('val', 'validation_metrics')):
            for key, value in _mapping(_mapping(values).get(field)).items():
                if (str(key).startswith(f'{split}_')
                        and _number(value) is not None):
                    raw = str(key)[len(split) + 1:]
                    recorded.setdefault(metric_name(raw), raw)
    return recorded


def _curves(trials: Mapping[str, Mapping[str, Any]], best: Optional[str],
            epoch: Optional[int], names: Sequence[str], primary: Optional[str],
            text: List[str]) -> str:
    """Each experiment's loss and metric curves from its per-epoch history.

    Each experiment shows the Evaluation metrics it recorded per epoch, tune
    metric first, and names those it did not. Scripting adds an experiment
    and a metric dropdown; without it, each experiment shows the loss and
    its first metric, with the others in a closed disclosure.
    """
    if not trials:
        text.append('No training history retained.')
        return ('<h4>Learning Curves</h4>'
                '<p class="ks-note">No training history retained.</p>')
    keys = sorted(
        trials, key=lambda key: (not key.isdigit(), int(key)
                                 if key.isdigit() else 0, key))
    shown = best if best is not None and best in trials else keys[0]

    def label(key: str) -> str:
        return (f'Experiment {int(key) + 1 if key.isdigit() else key}' +
                (' ★' if key == best else ''))

    def body(key: str) -> Tuple[str, List[str], List[str]]:
        recorded = _recorded(trials[key])
        metrics = [
            'loss', *sorted((name for name in names if name in recorded),
                            key=lambda name: name != primary)
        ]
        missing = [name for name in names if name not in recorded]
        drawn = [
            (name,
             _chart(_curve_title(name), series, epoch if key == best else None,
                    name)) for name in metrics
            for series in (_series(trials[key], recorded.get(name, name)), )
            if any(value is not None for points in series.values()
                   for _, value in points)
        ]
        lead = ''.join(chart for name, chart in drawn if name in metrics[:2])
        rest = ''.join(chart for name, chart in drawn
                       if name not in metrics[:2])
        html = (
            (f'<div class="ks-charts">{lead}</div>'
             if lead else '<p class="ks-note">No curves recorded.</p>') +
            ('<details class="ks-more"><summary>More metrics</summary>'
             f'<div class="ks-charts">{rest}</div></details>' if rest else '') +
            ('<p class="ks-note">No per-epoch history: ' + escape(', '.join(
                _curve_title(name)
                for name in missing)) + '.</p>' if missing else ''))
        return html, metrics, missing

    bodies = {key: body(key) for key in keys}
    _, metrics, missing = bodies[shown]
    recorded = _recorded(trials[shown])
    text.append(f'Learning curves: {label(shown)}')
    for name in metrics[:2]:
        for split, points in _series(trials[shown], recorded.get(name,
                                                                 name)).items():
            values = [value for _, value in points if value is not None]
            if values:
                text.append(f'  {split} {name}: ' +
                            ', '.join(_cell(value) for value in values))
    if missing:
        text.append('No per-epoch history: ' +
                    ', '.join(_curve_title(name) for name in missing))
    return (
        '<h4>Learning Curves</h4><div class="ks-toolbar" data-tune="' +
        escape(primary or '', quote=True) + '"></div>' +
        (bodies[shown][0] if len(keys) == 1 else ''.join(
            _panel('experiment', key, label(key), bodies[key][0], key == shown)
            for key in keys)))


def _ticks(low: float, high: float, count: int = 4) -> List[float]:
    span = (high - low) or abs(high) or 1.0
    raw = span / count
    scale = 10**math.floor(math.log10(raw))
    step = next(m for m in (1, 2, 2.5, 5, 10) if m * scale >= raw) * scale
    first = math.ceil(low / step - 1e-9)
    return [(first + i) * step for i in range(count + 2)
            if (first + i) * step <= high + step * 1e-9]


def _chart(title: str, series: Mapping[str, Points], best_epoch: Optional[int],
           metric: str) -> str:
    """One metric by epoch as a scripting-free SVG image.

    ``data-points`` holds each point's position and label for the hover
    tooltip that scripting adds.
    """
    points = [(epoch, value) for split in series.values()
              for epoch, value in split if value is not None]
    first, last = min(e for e, _ in points), max(e for e, _ in points)
    low, high = min(v for _, v in points), max(v for _, v in points)
    pad = (high - low) * .08 or abs(high) * .05 or .5
    low, high = low - pad, high + pad
    width, height, left, right, top, bottom = 460, 280, 58, 16, 44, 44

    def x(epoch: float) -> float:
        # Inset by 10 so the first and last points clear the axis.
        return left + 10 + (width - left - right - 20) * (
            (epoch - first) / (last - first) if last > first else .5)

    def y(value: float) -> float:
        return top + (height - top - bottom) * (1 - (value - low) /
                                                (high - low))

    ticks = _ticks(low, high)
    digits = 3
    if len(ticks) > 1:
        step = ticks[1] - ticks[0]
        power = math.floor(math.log10(step))
        digits = max(0, -power + (round(step / 10**power, 1) == 2.5))
    every = max(1, math.ceil((last - first + 1) / 10))
    axis = '#1e293b'
    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} '
        f'{height}" font-family="system-ui,sans-serif" font-size="12">'
        f'<rect width="{width}" height="{height}" fill="#fff"/>'
        f'<text x="{left}" y="20" font-size="14" font-weight="700">'
        f'{escape(title)}</text>'
        f'<path d="M{left} {top}V{height - bottom}H{width - right}" '
        f'fill="none" stroke="{axis}"/>'
        f'<text x="{(left + width - right) / 2}" y="{height - 8}" '
        f'text-anchor="middle" font-weight="500" fill="{axis}">Epoch</text>'
    ]
    for tick in ticks:
        svg.append(
            f'<path d="M{left - 4} {y(tick):.1f}H{left}" stroke="{axis}"/>'
            f'<text x="{left - 8}" y="{y(tick) + 4:.1f}" text-anchor="end" '
            f'fill="{axis}">{tick:.{digits}f}</text>')
    for epoch in range(first, last + 1, every):
        svg.append(
            f'<path d="M{x(epoch):.1f} {height - bottom}v4" stroke="{axis}"/>'
            f'<text x="{x(epoch):.1f}" y="{height - bottom + 18}" '
            f'text-anchor="middle" fill="{axis}">{epoch}</text>')
    legend, tips = width - right, []
    if best_epoch is not None and first <= best_epoch <= last:
        svg.append(f'<path d="M{x(best_epoch):.1f} {top}V{height - bottom}" '
                   f'stroke="{_BEST}" stroke-dasharray="4 3"/>')
        legend -= 78
        svg.append(f'<path d="M{legend} 12v10" stroke="{_BEST}" '
                   f'stroke-dasharray="4 3"/><text x="{legend + 7}" y="21" '
                   f'fill="{axis}">Best epoch</text>')
    for split, name, color, size in (('val', 'Validation', _VALIDATION, 84),
                                     ('train', 'Train', _TRAIN, 52)):
        recorded = [(epoch, value) for epoch, value in series[split]]
        if not any(value is not None for _, value in recorded):
            continue
        legend -= size
        svg.append(f'<circle cx="{legend}" cy="17" r="4" fill="{color}"/>'
                   f'<text x="{legend + 8}" y="21" fill="{axis}">{name}</text>')
        path, dots, previous = '', '', None
        for epoch, value in recorded:
            if value is None:
                continue
            # A missing epoch or value breaks the line instead of bridging it.
            path += (f'{"L" if previous == epoch - 1 else "M"}'
                     f'{x(epoch):.1f} {y(value):.1f}')
            previous = epoch
            dots += (f'<circle cx="{x(epoch):.1f}" cy="{y(value):.1f}" '
                     f'r="3" fill="{color}"/>')
            tips.append((round(x(epoch),
                               1), round(y(value),
                                         1), epoch, f'{name} {_cell(value)}'))
        svg.append(f'<path d="{path}" fill="none" stroke="{color}" '
                   f'stroke-width="2" stroke-linejoin="round"/>{dots}')
    # Each tooltip names the metric; points drawn on top of each other
    # share one.
    merged: Dict[Tuple[float, float, int], List[str]] = {}
    for cx, cy, at, value in tips:
        merged.setdefault((cx, cy, at), []).append(value)
    hover = [(cx, cy, f'{title}, epoch {at}: ' + ', '.join(values))
             for (cx, cy, at), values in merged.items()]
    encoded = base64.b64encode(
        (''.join(svg) + '</svg>').encode('utf-8')).decode('ascii')
    values = json.dumps(dict(series), separators=(',', ':'))
    return (f'<img class="ks-chart" width="{width}" height="{height}" '
            f'data-metric="{escape(metric, quote=True)}" data-points="'
            f'{escape(json.dumps(hover, separators=(",", ":")), quote=True)}" '
            f'alt="{escape(title, quote=True)} by epoch" '
            f'data-series="{escape(values, quote=True)}" '
            f'src="data:image/svg+xml;base64,{encoded}">')


# Type and spacing follow the Kumo UI's Model Performance card; colors are
# the UI's NVIDIA tokens in packages/kumo-ui-catalog/src/kumo-ui-catalog.css:
# --brand-green #76b900, --brand-green-active #3f8500, --kumo-grey-50 #f7f7f7.
_STYLE = '''<style>
.kumo-training-summary{font:14px/1.3 "NVIDIA Sans",ui-sans-serif,system-ui,
 sans-serif;max-width:980px;background:#fff;color:#000;padding:4px 0;
 position:relative;box-sizing:border-box}
.kumo-training-summary *{box-sizing:border-box}
.kumo-training-summary [hidden]{display:none!important}
.kumo-training-summary h4{font-size:16px;font-weight:700;line-height:1;
 margin:24px 0 12px;color:#000}
.kumo-training-summary .ks-panel>h4:first-of-type{margin-top:4px}
.kumo-training-summary[data-ready] .ks-fallback-title{display:none}
.kumo-training-summary .ks-fallback-title{font-size:18px;font-weight:700;
 margin:16px 0 8px}
.kumo-training-summary .ks-panel .ks-panel .ks-fallback-title{font-size:14px}
.kumo-training-summary summary.ks-fallback-title{cursor:pointer}
.kumo-training-summary .ks-note{color:#636363;font-size:13px;margin:8px 0;
 max-width:760px}
.kumo-training-summary .ks-tabs{display:flex;flex-wrap:wrap;gap:0 32px;
 box-shadow:inset 0 -2px 0 0 #0003;margin:0 0 20px}
.kumo-training-summary .ks-tabs button{all:unset;cursor:pointer;height:40px;
 line-height:40px;font-size:16px;font-weight:500;color:#636363;
 border-bottom:4px solid transparent;box-sizing:border-box}
.kumo-training-summary .ks-tabs button:hover{color:#000;
 box-shadow:inset 0 -2px 0 0 #000000e5}
.kumo-training-summary .ks-tabs button:focus-visible{outline:2px solid #76b900;
 outline-offset:2px}
.kumo-training-summary .ks-tabs button[aria-selected=true]{color:#000;
 border-bottom-color:#76b900;box-shadow:none}
.kumo-training-summary .ks-toolbar{display:flex;flex-wrap:wrap;gap:0 12px}
.kumo-training-summary .ks-more>summary{cursor:pointer;font-weight:600;
 margin:8px 0}
.kumo-training-summary .ks-tip{position:absolute;z-index:2;pointer-events:none;
 background:#fff;color:#000;border:1px solid #ccc;border-radius:4px;
 padding:4px 8px;font-size:12px;white-space:nowrap;box-shadow:0 2px 6px #0002}
.kumo-training-summary .ks-select{font:500 14px/1.3 "NVIDIA Sans",
 ui-sans-serif,system-ui,sans-serif;height:32px;padding:0 8px;margin:0 0 12px;
 border:1px solid #ccc;border-radius:4px;background:#fff;color:#000;
 cursor:pointer}
.kumo-training-summary .ks-select:focus-visible{outline:2px solid #76b900;
 outline-offset:2px}
.kumo-training-summary table{border-collapse:collapse;border:0;margin:0;
 font-size:14px;color:#000}
/* !important: notebook front ends style output tables more specifically. */
.kumo-training-summary tr{background:#fff!important}
.kumo-training-summary thead{border:0!important}
.kumo-training-summary th,.kumo-training-summary td{
 text-align:left!important;padding:6px 12px;border:0;vertical-align:middle;
 white-space:nowrap}
.kumo-training-summary .ks-metrics{width:100%;max-width:760px}
.kumo-training-summary .ks-metrics th,.kumo-training-summary .ks-plain th{
 background:#f7f7f7;color:#636363;font-weight:500;height:36px}
.kumo-training-summary .ks-metrics th:first-child,
.kumo-training-summary .ks-plain th:first-child{border-radius:4px 0 0 0}
.kumo-training-summary .ks-metrics th:last-child,
.kumo-training-summary .ks-plain th:last-child{border-radius:0 4px 0 0}
.kumo-training-summary .ks-metrics th.ks-holdout{color:#3f8500}
.kumo-training-summary .ks-metrics td,.kumo-training-summary .ks-plain td{
 height:38px;border-bottom:1px solid #ccc}
.kumo-training-summary .ks-metrics tr.ks-group td{border-bottom:0}
.kumo-training-summary .ks-metrics b{font-weight:700}
.kumo-training-summary .ks-metrics td:nth-last-child(-n+3){padding:6px 0}
.kumo-training-summary .ks-metrics td:nth-last-child(-n+3) span{display:block;
 padding:0 12px;height:26px;line-height:26px}
.kumo-training-summary .ks-metrics th:nth-last-child(-n+3){width:22%}
.kumo-training-summary .ks-tune td:nth-last-child(-n+3) span{color:#3f8500;
 background:#f7f7f7}
.kumo-training-summary .ks-tune td:nth-last-child(3) span{
 border-radius:4px 0 0 4px}
.kumo-training-summary .ks-tune td:last-child span{border-radius:0 4px 4px 0}
.kumo-training-summary .ks-scroll{overflow:auto;max-height:420px;
 max-width:100%}
.kumo-training-summary .ks-matrix,.kumo-training-summary .ks-scores{
 font-family:"Source Code Pro",ui-monospace,SFMono-Regular,Menlo,monospace}
.kumo-training-summary .ks-matrix th{font:500 14px/1.3 "NVIDIA Sans",
 ui-sans-serif,system-ui,sans-serif;color:#64748b;
 text-align:center!important}
.kumo-training-summary .ks-matrix tbody th{text-align:right!important}
.kumo-training-summary .ks-matrix .ks-axis{color:#6b7280;font-weight:400}
.kumo-training-summary .ks-matrix td{height:56px;min-width:120px;
 text-align:center!important;font-weight:600;color:#1f2937;border:2px solid #ccc}
.kumo-training-summary .ks-tradeoff{display:flex;flex-wrap:wrap;gap:16px 48px;
 align-items:flex-end}
.kumo-training-summary .ks-tradeoff .ks-matrix td{min-width:176px}
.kumo-training-summary .ks-scores td{border:1px solid #94a3b8;padding:8px 12px;
 font:400 14px/1.3 "NVIDIA Sans",ui-sans-serif,system-ui,sans-serif;
 color:#64748b}
.kumo-training-summary .ks-scores td+td{font:700 14px/1.3 "Source Code Pro",
 ui-monospace,SFMono-Regular,Menlo,monospace;color:#1f2937;min-width:72px}
.kumo-training-summary .ks-slider{position:relative;max-width:560px;
 padding:34px 0 0;margin:0 0 12px}
.kumo-training-summary .ks-slider input{-webkit-appearance:none;
 appearance:none;display:block;width:100%;height:24px;margin:0;padding:0;
 background:linear-gradient(#e5e7eb,#e5e7eb) center/100% 2px no-repeat;
 border:0;cursor:pointer;outline:none}
.kumo-training-summary .ks-slider input:focus-visible::-webkit-slider-thumb{
 background:#6b7280}
.kumo-training-summary .ks-slider input:focus-visible::-moz-range-thumb{
 background:#6b7280}
.kumo-training-summary .ks-slider input::-webkit-slider-thumb{
 -webkit-appearance:none;width:16px;height:24px;border-radius:2px;
 background:#9ca3af;border:0}
.kumo-training-summary .ks-slider input::-moz-range-thumb{width:16px;
 height:24px;border-radius:2px;background:#9ca3af;border:0}
.kumo-training-summary .ks-slider input::-moz-range-track{background:none}
.kumo-training-summary .ks-slider output{position:absolute;top:4px;width:56px;
 height:24px;line-height:22px;text-align:center;border:1px solid #9ca3af;
 border-radius:4px;background:#f9fafb;font-weight:600;color:#1f2937}
.kumo-training-summary .ks-ends{display:flex;justify-content:space-between;
 font-weight:600;color:#1f2937;margin-top:2px}
.kumo-training-summary .ks-charts{display:flex;flex-wrap:wrap;gap:12px 24px}
.kumo-training-summary .ks-chart{display:block;flex:1 1 360px;min-width:0;
 max-width:460px;height:auto;margin:0}
</style>'''

# Progressive enhancement: builds the tabs, the experiment and metric
# dropdowns, the curve tooltips and the threshold slider, so a page without
# scripting never shows a dead control. Every value they show was rendered
# above, so the script holds no metric logic.
_SCRIPT = '''<script>
(function () {
  var roots = document.querySelectorAll(
    '.kumo-training-summary:not([data-ready])');
  var list = function (nodes) { return Array.prototype.slice.call(nodes); };
  var make = function (tag, className, parent) {
    var node = document.createElement(tag);
    node.className = className || '';
    if (parent) parent.appendChild(node);
    return node;
  };
  list(roots).forEach(function (root) {
    root.setAttribute('data-ready', '');
    var toolbar = root.querySelector('.ks-toolbar');
    var switched = [];  // Run after the experiment dropdown changes.
    var groups = {};
    list(root.querySelectorAll('[data-panel]')).forEach(function (panel) {
      var group = panel.getAttribute('data-panel').split(':')[0];
      (groups[group] = groups[group] || []).push(panel);
    });
    Object.keys(groups).forEach(function (group) {
      var panels = groups[group];
      var tabs = group === 'main';
      var bar = make(tabs ? 'div' : 'select', tabs ? 'ks-tabs' : 'ks-select');
      bar.setAttribute(tabs ? 'role' : 'aria-label', tabs ? 'tablist' : group);
      if (tabs || !toolbar) panels[0].parentNode.insertBefore(bar, panels[0]);
      else toolbar.appendChild(bar);
      var show = function (index) {
        panels.forEach(function (panel, i) {
          panel.hidden = i !== index;
          panel.open = true;
          if (tabs) bar.children[i].setAttribute('aria-selected', i === index);
        });
        bar.selectedIndex = index;
      };
      var selected = 0;
      panels.forEach(function (panel, i) {
        var item = make(tabs ? 'button' : 'option', '', bar);
        item.textContent = panel.getAttribute('data-label');
        if (tabs) {
          item.type = 'button';
          item.setAttribute('role', 'tab');
          item.addEventListener('click', function () { show(i); });
        }
        if (panel.hasAttribute('data-selected')) selected = i;
      });
      if (!tabs) bar.addEventListener('change', function () {
        show(bar.selectedIndex);
        switched.forEach(function (run) { run(); });
      });
      show(selected);
    });
    // One metric at a time: the other metrics' charts join the visible ones.
    list(root.querySelectorAll('details.ks-more')).forEach(function (more) {
      list(more.querySelectorAll('img.ks-chart')).forEach(function (chart) {
        more.previousElementSibling.appendChild(chart);
      });
      more.parentNode.removeChild(more);
    });
    // The metric dropdown lists the metrics the shown experiment recorded,
    // and keeps the chosen one when the next experiment has it too.
    var charts = list(root.querySelectorAll('img.ks-chart[data-metric]'));
    if (toolbar && charts.length) {
      var picker = make('select', 'ks-select', toolbar);
      picker.setAttribute('aria-label', 'metric');
      var metrics = [], chosen = null;
      var pick = function () {
        charts.forEach(function (chart) {
          var metric = chart.getAttribute('data-metric');
          chart.hidden = metric !== 'loss' && metric !== chosen;
        });
      };
      var refresh = function () {
        metrics = [];
        picker.textContent = '';
        charts.forEach(function (chart) {
          var panel = chart.closest('[data-panel^="experiment:"]');
          var metric = chart.getAttribute('data-metric');
          if ((panel && panel.hidden) || metric === 'loss' ||
              metrics.indexOf(metric) >= 0) return;
          metrics.push(metric);
          make('option', '', picker).textContent =
            chart.alt.replace(/ by epoch$/, '') +
            (metric === toolbar.getAttribute('data-tune') ? ' *' : '');
        });
        picker.hidden = !metrics.length;
        chosen = metrics.indexOf(chosen) >= 0 ? chosen : metrics[0];
        picker.selectedIndex = metrics.indexOf(chosen);
        pick();
      };
      picker.addEventListener('change', function () {
        chosen = metrics[picker.selectedIndex];
        pick();
      });
      switched.push(refresh);
      refresh();
    }
    var tip = make('div', 'ks-tip', root);
    tip.hidden = true;
    charts.forEach(function (chart) {
      var points = JSON.parse(chart.getAttribute('data-points'));
      chart.addEventListener('mousemove', function (event) {
        var box = chart.getBoundingClientRect();
        var scale = box.width / +chart.getAttribute('width');
        var x = (event.clientX - box.left) / scale;
        var y = (event.clientY - box.top) / scale;
        var near = null, distance = 16;
        points.forEach(function (point) {
          var d = Math.sqrt(Math.pow(point[0] - x, 2) + Math.pow(point[1] - y, 2));
          if (d < distance) { distance = d; near = point; }
        });
        tip.hidden = !near;
        if (!near) return;
        var base = root.getBoundingClientRect();
        tip.textContent = near[2];
        var left = box.left - base.left + near[0] * scale;
        tip.style.left = (near[0] > +chart.getAttribute('width') / 2 ?
          left - tip.offsetWidth - 10 : left + 10) + 'px';
        tip.style.top = (box.top - base.top + near[1] * scale - 34) + 'px';
      });
      chart.addEventListener('mouseleave', function () { tip.hidden = true; });
    });
    list(root.querySelectorAll('.ks-threshold')).forEach(function (box) {
      var states = JSON.parse(box.getAttribute('data-states'));
      var slider = make('div', 'ks-slider');
      var bubble = make('output', '', slider);
      var input = make('input', '', slider);
      var ends = make('div', 'ks-ends', slider);
      make('span', '', ends).textContent = states[0].scores[0];
      make('span', '', ends).textContent = states[states.length - 1].scores[0];
      input.type = 'range';
      input.min = 0;
      input.max = states.length - 1;
      input.value = box.getAttribute('data-start');
      input.setAttribute('aria-label', 'Classification threshold');
      box.insertBefore(slider, box.firstChild);
      var update = function () {
        var state = states[+input.value];
        var share = states.length > 1 ? input.value / (states.length - 1) : 0;
        bubble.textContent = state.scores[0];
        bubble.style.left =
          'calc(' + share * 100 + '% - ' + (share * 16 + 20) + 'px)';
        ['tn', 'fp', 'fn', 'tp'].forEach(function (name, i) {
          var cell = box.querySelector('[data-cell=' + name + ']');
          cell.textContent = state.cells[i][0];
          cell.style.background = state.cells[i][1];
        });
        state.scores.forEach(function (score, i) {
          box.querySelector('[data-score="' + i + '"]').textContent = score;
        });
      };
      input.addEventListener('input', update);
      update();
    });
  });
})();
</script>'''
