"""Raw wire-sine exports and reception-time response fits, without smoothing."""

import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from .experiment_plotting import FEEDBACK_COLOR, PLAN_COLOR, _figure, _style, save_figure
from .response_signals import WireSineSignal


def _signal(snapshot):
    source = snapshot['signal']
    return WireSineSignal(frequency_hz=source['frequency_hz'], cycles=source['cycles'],
                          baseline_s=source['baseline_s'], recovery_s=source['recovery_s'])


def actual_inputs(snapshot, times):
    """Zero-order hold of successful publish calls, including nonzero origin."""
    origin = np.asarray(snapshot['origin_handles'][:3], float)
    commands = [c for c in snapshot['commands'] if c['publish_success']]
    ticks = np.asarray([-np.inf] + [c['publish_elapsed_s'] for c in commands])
    values = np.asarray([origin] + [c['handles'][:3] for c in commands], float) - origin
    return values[np.searchsorted(ticks, times, side='right') - 1]


def harmonic_fit(times, values, frequency, phase_origin):
    times, values = np.asarray(times, float), np.asarray(values, float)
    if len(times) < 4 or np.ptp(times) <= 0:
        return None
    phase = 2 * np.pi * frequency * (times - phase_origin)
    design = np.column_stack((np.sin(phase), np.cos(phase), np.ones(len(times))))
    dt = np.diff(times)
    weights = np.r_[dt[0] / 2, (dt[:-1] + dt[1:]) / 2, dt[-1] / 2]
    coefficients = np.linalg.lstsq(design * np.sqrt(weights[:, None]), values * np.sqrt(weights), rcond=None)[0]
    residual = values - design @ coefficients
    return dict(amplitude=float(np.hypot(*coefficients[:2])),
                phase_deg=float(np.degrees(np.arctan2(coefficients[1], coefficients[0]))),
                offset=float(coefficients[2]), coefficients=coefficients.tolist(),
                residual_rmse=float(np.sqrt(np.average(residual ** 2, weights=weights))))


def wire_metrics(snapshot):
    rows = snapshot['feedback']
    signal = _signal(snapshot)
    result = dict(input_definition='cumulative handle sine; actual successful publishes held between calls',
                  time_basis='host publish call to host feedback receipt; device sample time unavailable',
                  fitted_cycles=0, frequency_hz=signal.frequency_hz, native={}, base={},
                  coupling_note='same-frequency coexcitation; effective responses, not independent transfer functions')
    if not rows:
        return result
    t = np.asarray([r['elapsed_s'] for r in rows])
    xyz = np.asarray([r['actual_xyz_um'] for r in rows])
    q = np.asarray([r['joint_actual'] for r in rows])
    result['feedback_interval_s'] = dict(median=float(np.median(np.diff(t))) if len(t) > 1 else None,
                                         maximum=float(np.max(np.diff(t))) if len(t) > 1 else None)
    result['base_peak_to_peak_um'] = np.ptp(xyz, axis=0).tolist()
    result['final_base_residual_um'] = xyz[-1].tolist()
    result['final_base_residual_norm_um'] = float(np.linalg.norm(xyz[-1]))
    baseline = xyz[t < signal.baseline_s]
    result['baseline_std_um'] = np.std(baseline, axis=0).tolist() if len(baseline) else None
    result['native_peak_to_peak'] = np.ptp(q, axis=0).tolist()
    start = signal.baseline_s + 1 / signal.frequency_hz
    end = signal.baseline_s + signal.active_s
    window = (t >= start) & (t <= end)
    # All three analysis cycles require feedback on both boundaries and no
    # watchdog gap. Never fit across an interrupted or missing cycle.
    coverage = t[(t >= start - snapshot['control']['state_timeout_s']) &
                 (t <= end + snapshot['control']['state_timeout_s'])]
    if (signal.cycles < 4 or len(coverage) < 4 or coverage[0] > start or coverage[-1] < end or
            np.max(np.diff(coverage)) > snapshot['control']['state_timeout_s'] or
            np.count_nonzero(window) < 10):
        result['fit_unavailable_reason'] = 'fewer than three complete continuous steady cycles'
        return result
    result['fitted_cycles'] = signal.cycles - 1
    result['fit_window_s'] = [start, end]
    inputs = actual_inputs(snapshot, t[window])
    input_fits = [harmonic_fit(t[window], inputs[:, i], signal.frequency_hz, signal.baseline_s) for i in range(3)]
    def response(output, excitation, noise, floor):
        fit = harmonic_fit(t[window], output, signal.frequency_hz, signal.baseline_s)
        fit['gain_per_handle_count'] = None
        fit['lag_deg'] = None
        fit['equivalent_lag_s'] = None
        fit['output_detected'] = fit['amplitude'] > max(3 * noise, floor)
        if excitation and excitation['amplitude'] > .1:
            fit['gain_per_handle_count'] = fit['amplitude'] / excitation['amplitude']
        if excitation and excitation['amplitude'] > .1 and fit['output_detected']:
            lag = (excitation['phase_deg'] - fit['phase_deg'] + 180) % 360 - 180
            fit.update(gain_per_handle_count=fit['amplitude'] / excitation['amplitude'],
                       lag_deg=lag, equivalent_lag_s=lag / 360 / signal.frequency_hz)
        return fit
    for i, axis in enumerate('ABC'):
        noise = float(np.std(q[t < signal.baseline_s, i])) if np.any(t < signal.baseline_s) else 0.
        result['native'][axis] = dict(input=input_fits[i], output=response(q[window, i], input_fits[i], noise, (.0002, .0002, .2)[i]),
                                      output_unit=('deg', 'deg', 'um')[i],
                                      interpretation='effective response under recorded simultaneous excitation')
    excited = [i for i, a in enumerate(snapshot['amplitudes_counts']) if a]
    reference_axis = max(excited, key=lambda i: input_fits[i]['amplitude'])
    result['base_reference_axis'] = 'ABC'[reference_axis]
    for i, axis in enumerate('XYZ'):
        noise = float(np.std(baseline[:, i])) if len(baseline) else 0.
        fit = response(xyz[window, i], input_fits[reference_axis], noise, .0002)
        fit['gain_unit'] = 'um / handle count of ' + result['base_reference_axis'] + ' under coexcitation'
        result['base'][axis] = fit
    return result


def _write_records(path, records):
    if not records:
        path.write_text('')
        return
    def flatten(row):
        flat = {}
        for key, value in row.items():
            if isinstance(value, (list, tuple)):
                axes = 'ABCE' if key in ('robot', 'handle', 'handles', 'planned_handles', 'delta') else (
                    'XYZ' if key in ('junction_mm', 'tip_mm', 'actual_xyz_um') else 'ABC')
                for axis, item in zip(axes, value):
                    flat[key + '_' + axis] = item
            else:
                flat[key] = value
        return flat
    rows = [flatten(row) for row in records]
    columns = list(dict.fromkeys(key for row in rows for key in row))
    with path.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def export_wire_sine(snapshot, directory):
    root = Path(directory)
    root.mkdir(parents=True, exist_ok=False)
    (root / 'snapshot.json').write_text(json.dumps(snapshot, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    for name in ('feedback', 'decisions', 'commands'):
        _write_records(root / (name + '.csv'), snapshot[name])
    signal = _signal(snapshot)
    reference = []
    for t in np.r_[np.arange(0, signal.duration_s, .05), signal.duration_s]:
        value, phase = signal.at(t)
        ideal = tuple(a * value / signal.amplitude_um for a in snapshot['amplitudes_counts'])
        reference.append(dict(elapsed_s=float(t), phase=phase, ideal_handle_offset=ideal,
                              planned_handles=tuple(o + round(v) for o, v in zip(snapshot['origin_handles'], ideal + (0,)))))
    _write_records(root / 'reference.csv', reference)
    metrics = wire_metrics(snapshot)
    metadata = {k: v for k, v in snapshot.items() if k not in ('feedback', 'decisions', 'commands')}
    metadata.update(metrics=metrics, analysis_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                    reference_basis='handle counts; internal signal amplitude_um is normalization only, not a physical target',
                    accuracy_note='robot feedback, not externally measured accuracy')
    (root / 'metadata.json').write_text(json.dumps(metadata, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    return root


def build_wire_figures(snapshot):
    rows = snapshot['feedback']
    if not rows:
        return {}
    metrics = wire_metrics(snapshot)
    signal = _signal(snapshot)
    t = np.asarray([r['elapsed_s'] for r in rows])
    timeout = snapshot['control']['state_timeout_s']
    figures = {}
    def new(name, title, nrows, ncols=1, height=8):
        figure = _figure((12, height))
        axes = figure.subplots(nrows, ncols, squeeze=False, sharex=True)
        figure.subplots_adjust(left=.10, right=.96, top=.80, bottom=.13, hspace=.64, wspace=.25)
        figure.suptitle(title, x=.10, y=.96, ha='left', fontsize=16, fontweight='bold', color='#17212B')
        figure.text(.10, .91, '%s | %s | %.3f Hz | %s: %s' % (
            snapshot['metadata'].get('data_source', 'ROS feedback').upper(), snapshot['mode'],
            signal.frequency_hz, snapshot['status'], snapshot['reason']), fontsize=9, color='#5B6875')
        figure.text(.10, .035, 'Raw feedback; no smoothing or phase alignment. Robot coordinates, not external accuracy.\n'
                    'First cycle excluded from fitting. Same-frequency coexcitation does not identify independent coupling.',
                    fontsize=8, color='#5B6875')
        for axis in axes.flat:
            _style(axis, '')
            axis.axvspan(0, signal.baseline_s, color='#EAF2F8', alpha=.6)
            axis.axvspan(signal.baseline_s, signal.baseline_s + 1 / signal.frequency_hz, color='#F3ECF8', alpha=.6)
            axis.axvspan(signal.baseline_s + signal.active_s, max(t[-1], signal.baseline_s + signal.active_s), color='#E8F5EE', alpha=.6)
            axis.set_xlim(0, max(1, t[-1]))
            axis.axhline(0, color='#64748B', lw=.6)
        for axis in axes[-1]:
            axis.set_xlabel('Elapsed time (s)')
        figures[name] = figure
        return axes
    def raw(axis, values, **kw):
        values = np.asarray(values, float).copy()
        # Insert NaNs instead of bridging watchdog gaps.
        gaps = np.flatnonzero(np.diff(t) > timeout) + 1
        tt = np.insert(t, gaps, np.nan)
        vv = np.insert(values, gaps, np.nan)
        axis.plot(tt, vv, color=FEEDBACK_COLOR, lw=.9, marker='.', ms=1.7, **kw)
    grid = np.arange(0, max(t[-1], .05) + .025, .05)
    ideal = np.array([[a * signal.at(x)[0] / signal.amplitude_um for a in snapshot['amplitudes_counts']] for x in grid])
    sent = [c for c in snapshot['commands'] if c['publish_success'] and c['action'] == 'send']
    axes = new('01_control_input', 'Cumulative control signal and actual packet increments', 3, 2, 9)
    sent_times = np.array([0.] + [c['publish_elapsed_s'] for c in snapshot['commands'] if c['publish_success']] + [t[-1]])
    actual = actual_inputs(snapshot, sent_times)
    for i, axis_name in enumerate('ABC'):
        left, right = axes[i]
        left.set_title(axis_name + ' | cumulative offset', loc='left', fontsize=10)
        left.plot(grid, ideal[:, i], color=PLAN_COLOR, lw=1., ls='--', label='Ideal sine')
        left.step(sent_times, actual[:, i], where='post', color=FEEDBACK_COLOR, lw=.9, label='Actual published (held)')
        left.set_ylabel('Handle counts')
        right.set_title(axis_name + ' | packet difference', loc='left', fontsize=10)
        right.vlines([c['publish_elapsed_s'] for c in sent], 0, [c['delta'][i] for c in sent], color=FEEDBACK_COLOR, lw=.8)
        right.set_ylabel('Counts / packet')
    figures['01_control_input'].legend(*axes[0, 0].get_legend_handles_labels(),
                                       frameon=False, fontsize=9, ncol=2,
                                       loc='upper left', bbox_to_anchor=(.095, .885))
    axes = new('02_encoder_response', 'Encoder response to cumulative sine input', 3)
    q = np.asarray([r['joint_actual'] for r in rows])
    for i, axis in enumerate(axes[:, 0]):
        axis.set_title('ABC'[i] + ' | relative to experiment start', loc='left', fontsize=10)
        raw(axis, q[:, i])
        axis.set_ylabel(('Angle (deg)', 'Linkage angle (deg)', 'Axial travel (µm)')[i])
        axis.text(.99, .88, 'Peak-to-peak %.6g' % np.ptp(q[:, i]), transform=axis.transAxes, ha='right', fontsize=8)
    axes = new('03_base_response', 'Base XYZ response | needle junction', 3)
    xyz = np.asarray([r['actual_xyz_um'] for r in rows])
    for i, axis in enumerate(axes[:, 0]):
        axis.set_title('XYZ'[i] + ' | base displacement', loc='left', fontsize=10)
        raw(axis, xyz[:, i], label='Robot feedback')
        axis.set_ylabel('Displacement (µm)')
        if metrics['fitted_cycles'] >= 3:
            fit = metrics['base']['XYZ'[i]]
            start, end = metrics['fit_window_s']
            mask = (t >= start) & (t <= end)
            phase = 2 * np.pi * signal.frequency_hz * (t[mask] - signal.baseline_s)
            a, b, c = fit['coefficients']
            axis.plot(t[mask], a * np.sin(phase) + b * np.cos(phase) + c,
                      color=PLAN_COLOR, lw=1., ls='--', label='Steady fundamental fit')
        axis.text(.99, .88, 'Peak-to-peak %.5g µm | final %+.5g µm' % (np.ptp(xyz[:, i]), xyz[-1, i]),
                  transform=axis.transAxes, ha='right', fontsize=8)
    figures['03_base_response'].legend(*axes[0, 0].get_legend_handles_labels(),
                                      frameon=False, fontsize=9, ncol=2,
                                      loc='upper left', bbox_to_anchor=(.095, .885))
    figure = _figure((11, 5.4))
    figures['04_frequency_response'] = figure
    figure.suptitle('Base frequency response under simultaneous excitation', x=.10, y=.96,
                   ha='left', fontsize=14, fontweight='bold', color='#17212B')
    axes = figure.subplots(1, 2)
    figure.subplots_adjust(left=.10, right=.96, top=.78, bottom=.22, wspace=.32)
    figure.text(.10, .86, snapshot['metadata'].get('data_source', 'ROS feedback').upper() +
                ' | ' + snapshot['status'] + ': ' + snapshot['reason'], fontsize=9, color='#5B6875')
    for axis, title in zip(axes, ('Effective fundamental gain', 'Phase lag at %.3f Hz' % signal.frequency_hz)):
        _style(axis, title)
    if metrics['fitted_cycles'] >= 3:
        gains = [metrics['base'][a]['gain_per_handle_count'] for a in 'XYZ']
        lags = [metrics['base'][a]['lag_deg'] for a in 'XYZ']
        for axis, values, color in zip(axes, (gains, lags), (FEEDBACK_COLOR, PLAN_COLOR)):
            axis.bar(list('XYZ'), [v if v is not None else 0 for v in values], color=color, width=.5)
            for i, v in enumerate(values):
                if v is None:
                    axis.text(i, .5, 'N/A', ha='center', transform=axis.get_xaxis_transform(), fontsize=9)
        axes[0].set_ylabel('µm / handle count of ' + metrics['base_reference_axis'])
        axes[1].set_ylabel('Lag (deg; principal value)')
    else:
        for axis in axes:
            axis.text(.5, .5, 'Unavailable: fewer than 3 complete\ncontinuous steady cycles',
                      transform=axis.transAxes, ha='center', va='center', fontsize=10)
    figure.text(.10, .07, 'Effective gains are conditional on the recorded coexcitation; not independent transfer functions.\n'
                'No target-position tracking error is inferred from direct handle input.', fontsize=8, color='#5B6875')
    # These English figures also include Chinese stop reasons from ROS errors.
    from matplotlib import font_manager
    installed = {font.name for font in font_manager.fontManager.ttflist}
    cjk = next((name for name in ('Noto Sans CJK SC', 'WenQuanYi Micro Hei', 'Noto Sans CJK JP') if name in installed), None)
    if cjk:
        from matplotlib.text import Text
        for figure in figures.values():
            for label in figure.findobj(Text):
                if any(ord(char) > 127 for char in label.get_text()):
                    label.set_fontfamily(cjk)
    return figures


def export_wire_figures(snapshot, directory):
    figures = build_wire_figures(snapshot)
    try:
        for name, figure in figures.items():
            for extension in ('png', 'pdf'):
                save_figure(figure, Path(directory) / (name + '.' + extension))
    finally:
        for figure in figures.values():
            figure.clear()
