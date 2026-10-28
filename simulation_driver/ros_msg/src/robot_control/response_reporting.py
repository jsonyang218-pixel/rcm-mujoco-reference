"""Reproducible time-domain metrics, raw exports and publication figures.

No ROS, no smoothing, no lag correction. References are evaluated at feedback
receipt time. Gaps exceeding the watchdog are excluded from time integration.
"""

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from .experiment_plotting import FEEDBACK_COLOR, PLAN_COLOR, _figure, _style, save_figure
from .response_signals import ResponseSignal
from .joint_response import joint_reference


TIP_COLOR = "#009E73"


def _weights(t):
    if len(t) < 2:
        return np.ones(len(t))
    dt = np.diff(t)
    return np.r_[dt[0] / 2, (dt[:-1] + dt[1:]) / 2, dt[-1] / 2]


def _slope(t, y):
    if len(t) < 3 or np.ptp(t) <= 0:
        return None
    design = np.column_stack((t - t[0], np.ones(len(t))))
    weights = np.sqrt(_weights(t))
    return float(np.linalg.lstsq(design * weights[:, None], y * weights, rcond=None)[0][0])


def _crossing(t, y, level):
    for i in range(1, len(t)):
        if y[i - 1] < level <= y[i]:
            return float(t[i - 1] + (t[i] - t[i - 1]) * (level - y[i - 1]) / (y[i] - y[i - 1]))
    return float(t[0]) if len(t) and y[0] >= level else None


def _edge_metrics(t, y, start, end, initial, target, noise_sigma, tolerance, settle_s, settle_frames, gap_limit):
    mask = (t >= start) & (t < end)
    ts, ys = t[mask], y[mask]
    if len(ts) < 3:
        return dict(available=False, reason="insufficient_samples")
    if np.any(np.diff(ts) > gap_limit):
        return dict(available=False, reason="feedback_gap_in_edge_window")
    sign = 1 if target > initial else -1
    size = abs(target - initial)
    progress = (ys - initial) * sign
    low, high = ((_crossing(ts, progress, size * .1), _crossing(ts, progress, size * .9))
                 if size > .2 else (None, None))
    threshold = max(3 * noise_sigma, .2)
    onset = None
    onset_start = None
    for i, value in enumerate(progress):
        if value <= threshold:
            onset_start = None
            continue
        if onset_start is None:
            onset_start = i
        if i - onset_start + 1 >= 3 and ts[i] - ts[onset_start] >= .1 - 1e-9:
            onset = float(ts[onset_start] - start)
            break
    outside = np.flatnonzero(abs(ys - target) > tolerance)
    index = int(outside[-1] + 1) if len(outside) else 0
    settling = None
    if index < len(ts) and len(ts) - index >= settle_frames and ts[-1] - ts[index] >= settle_s:
        settling = float(ts[index] - start)
    tail = ys[ts >= max(start, end - 3)]
    return dict(available=True, onset_threshold_um=threshold, onset_delay_s=onset,
                rise_10_90_s=high - low if high is not None and low is not None else None,
                t10_s=low - start if low is not None else None,
                t90_s=high - start if high is not None else None,
                overshoot_percent=float(max(0, np.max(progress) - size) / size * 100) if size else None,
                settling_s=settling, settled_in_observation=settling is not None,
                final_mean_error_um=float(target - np.mean(tail)) if len(tail) else None,
                observed_until_s=float(ts[-1] - start))


def _scalar_metrics(snapshot, amplitude_scale=1.0):
    rows = snapshot["feedback"]
    if not rows:
        return dict(available=False, reason="no_feedback")
    t = np.array([r["elapsed_s"] for r in rows])
    actual = np.array([r["actual_um"] for r in rows])
    error = np.array([r["error_um"] for r in rows])
    if not np.isfinite(np.r_[t, actual, error]).all() or (np.diff(t) <= 0).any():
        raise ValueError("Feedback must have finite values and strictly increasing timestamps")
    signal = ResponseSignal(**snapshot["signal"])
    control = snapshot["control"]
    dt = np.diff(t)
    valid = dt <= control["state_timeout_s"]
    duration = float(np.sum(dt[valid]))
    integral = np.sum((error[:-1] ** 2 + error[1:] ** 2)[valid] * dt[valid] / 2)
    norm_error = np.array([r["error_norm_um"] for r in rows])
    norm_integral = np.sum((norm_error[:-1] ** 2 + norm_error[1:] ** 2)[valid] * dt[valid] / 2)
    baseline = actual[t < (30 if signal.kind == "noise" else signal.baseline_s)]
    sigma = float(np.std(baseline, ddof=1)) if len(baseline) > 1 else 0.0
    initial = float(np.median(baseline)) if len(baseline) else 0.0
    result = dict(available=True, count=len(t),
                  position_3d_rmse_um=math.sqrt(norm_integral / duration) if duration else None,
                  position_3d_max_um=float(np.max(norm_error)), time_weighted_rmse_um=math.sqrt(integral / duration) if duration else None,
                  sample_rmse_um=float(np.sqrt(np.mean(error ** 2))), max_abs_error_um=float(np.max(abs(error))),
                  integrated_duration_s=duration, excluded_gap_s=float(np.sum(dt[~valid])),
                  baseline_sigma_um=sigma, baseline_peak_to_peak_um=float(np.ptp(baseline)) if len(baseline) else None,
                  baseline_drift_um_s=_slope(t[:len(baseline)], baseline),
                  feedback_interval_median_s=float(np.median(dt)) if len(dt) else None,
                  feedback_interval_p95_s=float(np.percentile(dt, 95)) if len(dt) else None,
                  feedback_interval_max_s=float(np.max(dt)) if len(dt) else None)
    amplitude = signal.amplitude_um * amplitude_scale
    start, end = signal.baseline_s, signal.baseline_s + signal.active_s
    complete = t[-1] >= signal.duration_s - control["state_timeout_s"]
    result["observation_complete"] = bool(complete)
    decisions = snapshot.get("decisions", ())
    # Loaded decision rows are CSV strings; keep the offline replot path exact.
    decision_times = np.array([float(d["elapsed_s"]) for d in decisions])
    decision_intervals = np.diff(decision_times)
    result.update(decision_interval_median_s=float(np.median(decision_intervals)) if len(decision_intervals) else None,
                  decision_interval_p95_s=float(np.percentile(decision_intervals, 95)) if len(decision_intervals) else None,
                  decision_interval_max_s=float(np.max(decision_intervals)) if len(decision_intervals) else None,
                  max_feedback_age_at_decision_s=max((float(d["feedback_age_s"]) for d in decisions), default=None),
                  publish_attempts=len(snapshot.get("commands", ())),
                  failed_publish_attempts=sum(not c.get("publish_success", True) for c in snapshot.get("commands", ())))
    if signal.kind == "step":
        result["step_outward"] = _edge_metrics(t, actual, start, end, initial, amplitude, sigma,
                                               control["tolerance_um"], control["settle_s"], control["settle_samples"], control["state_timeout_s"])
        before_return = actual[(t >= max(start, end - 3)) & (t < end)]
        return_initial = float(np.median(before_return)) if len(before_return) else amplitude
        result["step_return"] = _edge_metrics(t, actual, end, signal.duration_s, return_initial, 0, sigma,
                                              control["tolerance_um"], control["settle_s"], control["settle_samples"], control["state_timeout_s"])
    if signal.kind == "sine":
        mask = (t >= start + 1 / signal.frequency_hz) & (t < end)
        ts, ys = t[mask], actual[mask]
        steady_start = start + 1 / signal.frequency_hz
        left = max(0, int(np.searchsorted(t, steady_start)) - 1)
        right = min(len(t), int(np.searchsorted(t, end)) + 1)
        covered = t[0] <= steady_start and t[-1] >= end
        gap_free = not np.any(np.diff(t[left:right]) > control["state_timeout_s"])
        if len(ts) >= 8 and covered and gap_free:
            angle = 2 * math.pi * signal.frequency_hz * (ts - start)
            design = np.column_stack((np.sin(angle), np.cos(angle), np.ones(len(ts))))
            weights = np.sqrt(_weights(ts))
            coefficients = np.linalg.lstsq(design * weights[:, None], ys * weights, rcond=None)[0]
            a, b, bias = coefficients
            lag = float(-math.degrees(math.atan2(b, a)))
            result["sine"] = dict(available=True, amplitude_ratio=float(math.hypot(a, b) / amplitude),
                                  phase_lag_deg=lag if math.hypot(a, b) > max(3 * sigma, .1) else None,
                                  equivalent_lag_s=lag / (360 * signal.frequency_hz) if math.hypot(a, b) > max(3 * sigma, .1) else None,
                                  phase_available=bool(math.hypot(a, b) > max(3 * sigma, .1)),
                                  bias_um=float(bias), fit_residual_rmse_um=float(np.sqrt(np.average(
                                      (ys - design @ coefficients) ** 2, weights=weights ** 2))),
                                  window_start_s=float(ts[0]), window_end_s=float(ts[-1]),
                                  phase_convention="positive means lag; principal phase, not unique transport delay")
        else:
            result["sine"] = dict(available=False, reason="need_three_complete_steady_cycles_without_gaps")
    if signal.kind == "ramp":
        travel = signal.amplitude_um / signal.speed_um_s
        result["ramp"] = {}
        for label, edge in (("outward", start), ("return", start + travel + signal.hold_s)):
            first, last = edge + .1 * travel, edge + .9 * travel
            mask = (t >= first) & (t <= last)
            left = max(0, int(np.searchsorted(t, first)) - 1)
            right = min(len(t), int(np.searchsorted(t, last)) + 1)
            covered = t[0] <= first and t[-1] >= last
            gap_free = not np.any(np.diff(t[left:right]) > control["state_timeout_s"])
            result["ramp"][label + "_speed_um_s"] = _slope(t[mask], actual[mask]) if covered and gap_free else None
        result["ramp"]["fit_window"] = "central 10%-90% of each scheduled ramp"
    if signal.kind == "pulse":
        mask = (t >= start) & (t < end)
        recovery = (t >= max(end, signal.duration_s - 3))
        peak = float(np.max(actual[mask]) - initial) if mask.any() else None
        before_return = actual[(t >= max(start, end - .25)) & (t < end)]
        return_initial = float(np.median(before_return)) if len(before_return) else amplitude
        result["pulse"] = dict(peak_displacement_um=peak,
                               peak_including_recovery_um=float(np.max(actual[t >= start]) - initial) if (t >= start).any() else None,
                               amplitude_ratio=peak / amplitude if peak is not None else None,
                               final_residual_um=float(np.mean(actual[recovery])) if recovery.any() else None,
                               recovery=_edge_metrics(t, actual, end, signal.duration_s, return_initial, 0,
                                                     sigma, control["tolerance_um"], control["settle_s"], control["settle_samples"], control["state_timeout_s"]))
    return result


def _xyz(snapshot):
    """Same-frame Cartesian arrays; legacy exports retain projected data."""
    rows = snapshot["feedback"]
    direction = np.asarray(snapshot["direction"])
    drive = np.asarray(snapshot.get("drive_vector", direction))
    reference = np.array([r.get("reference_xyz_um", np.asarray(r["reference_um"]) * direction) for r in rows])
    point = "junction" if snapshot["mode"].startswith("junction") else "tip"
    origin = np.asarray(snapshot["origin_" + point + "_mm"])
    actual = np.array([r.get("actual_xyz_um", (np.asarray(r[point + "_mm"]) - origin) * 1000) for r in rows])
    if rows and "actual_xyz_um" not in rows[0]:
        # Preserve legacy projected values, including independently recorded spikes.
        actual += (np.array([r["actual_um"] for r in rows]) - actual @ direction)[:, None] * direction
    error = reference - actual
    return reference, actual, error, drive


def reference_xyz_at(snapshot, elapsed):
    signal = ResponseSignal(**snapshot["signal"])
    if snapshot["mode"] == "junction_abc":
        return joint_reference(signal, snapshot["origin_robot"], snapshot["joint_amplitudes"], elapsed)[2]
    drive = snapshot.get("drive_vector", snapshot["direction"])
    return tuple(signal.at(elapsed)[0] * v for v in drive)


def _joint_metrics(snapshot):
    result = {}
    for i, (name, unit) in enumerate(zip("ABC", ("deg", "deg", "um"))):
        scale = 1000 if unit == "deg" else 1
        rows = [dict(row, actual_um=row["joint_actual"][i] * scale,
                     error_um=row["joint_error"][i] * scale, error_norm_um=abs(row["joint_error"][i]) * scale)
                for row in snapshot["feedback"]]
        sub = dict(snapshot, feedback=rows, control=dict(snapshot["control"],
                   tolerance_um=snapshot["joint_tolerances"][i] * scale))
        sign = -1 if snapshot["joint_amplitudes"][i] < 0 else 1
        for row in rows:
            row["actual_um"] *= sign
            row["error_um"] *= sign
        stats = _scalar_metrics(sub, max(abs(snapshot["joint_amplitudes"][i]), 1e-12) * scale / snapshot["signal"]["amplitude_um"])
        for key in tuple(stats):
            if key.startswith("position_3d"):
                stats.pop(key)
        if not snapshot["joint_amplitudes"][i] or snapshot["signal"]["kind"] == "noise":
            for key in ("sine", "ramp", "step_outward", "step_return", "pulse"):
                stats.pop(key, None)
        def convert(data):
            if not isinstance(data, dict):
                return data
            output = {}
            for key, value in data.items():
                physical = key.endswith("_um") or key.endswith("_um_s")
                output[key.replace("_um", "_" + unit) if physical else key] = (
                    value / scale if physical and value is not None else convert(value))
            return output
        result[name] = dict(convert(stats), unit=unit,
                            reference_amplitude=abs(snapshot["joint_amplitudes"][i]))
    return result


def response_metrics(snapshot):
    result = _scalar_metrics(snapshot, snapshot.get("reference_scale", 1.0))
    if not result["available"]:
        return result
    reference, actual, error, drive = _xyz(snapshot)
    result["axes"] = {}
    for index, name in enumerate("xyz"):
        excited = abs(drive[index]) > 1e-10 and snapshot["signal"]["kind"] != "noise"
        sign = -1 if drive[index] < 0 else 1
        sub = dict(snapshot)
        sub["feedback"] = [dict(row, actual_um=float(actual[i, index] * sign),
                                error_um=float(error[i, index] * sign),
                                error_norm_um=float(abs(error[i, index])))
                           for i, row in enumerate(snapshot["feedback"])]
        axis = _scalar_metrics(sub, abs(drive[index]) if excited else 1.0)
        if not excited:
            for key in ("sine", "ramp", "step_outward", "step_return", "pulse"):
                axis.pop(key, None)
        axis.update(excited=bool(excited), signed_mean_error_um=float(np.mean(error[:, index])),
                    reference_amplitude_um=float(abs(drive[index]) * snapshot["signal"]["amplitude_um"]) if excited else 0.0)
        result["axes"][name] = axis
    if snapshot.get("mode") == "junction_xyz" and snapshot["signal"]["kind"] == "sine":
        fits = [result["axes"][a].get("sine", {}) for a in "xyz"]
        if all(f.get("available") and f.get("phase_lag_deg") is not None for f in fits):
            phases = np.radians([f["phase_lag_deg"] for f in fits])
            result["pairwise_phase_difference_deg"] = {
                a + b: float(np.degrees(np.angle(np.exp(1j * (phases[i] - phases[j])))))
                for i, a in enumerate("xyz") for j, b in enumerate("xyz") if i < j}
    if snapshot["mode"] == "junction_abc":
        for stats in [result] + list(result["axes"].values()):
            for key in ("sine", "ramp", "step_outward", "step_return", "pulse"):
                stats.pop(key, None)
        result["joints"] = _joint_metrics(snapshot)
        result["reference_kind"] = "XYZ forward-model reference, not independently excited Cartesian signals"
        for i, name in enumerate("xyz"):
            result["axes"][name]["excited"] = bool(np.ptp(reference[:, i]) > .01)
            result["axes"][name].pop("reference_amplitude_um", None)
    return result


def export_response(snapshot, directory):
    """One experiment per new directory; JSON installed last as manifest."""
    root = Path(directory)
    root.mkdir(parents=True, exist_ok=False)
    feedback_columns = ("elapsed_s", "unix_time", "mono_time", "sequence", "phase", "reference_um", "actual_um",
                        "error_um", "error_norm_um", "transverse_um", "posture_deg", "forward", "backward")
    reference_xyz, actual_xyz, error_xyz, drive = _xyz(snapshot)
    origin = np.asarray(snapshot["origin_junction_mm"] if snapshot["mode"].startswith("junction") else snapshot["origin_tip_mm"])
    snapshot = dict(snapshot, feedback=[dict(row, reference_xyz_um=tuple(reference_xyz[i]), actual_xyz_um=tuple(actual_xyz[i]),
                         error_xyz_um=tuple(error_xyz[i]), target_mm=tuple(origin + reference_xyz[i] / 1000))
                         for i, row in enumerate(snapshot["feedback"])])
    vector_columns = {"reference_xyz_um": tuple("reference_" + a + "_um" for a in "xyz"),
                      "actual_xyz_um": tuple("actual_" + a + "_um" for a in "xyz"),
                      "error_xyz_um": tuple("error_" + a + "_um" for a in "xyz"),
                      "target_mm": tuple("target_" + a + "_mm" for a in "xyz"),
                      "junction_mm": tuple("junction_" + a + "_mm" for a in "xyz"),
                      "tip_mm": tuple("tip_" + a + "_mm" for a in "xyz"),
                      "robot": tuple("robot_" + a for a in "ABCE"),
                      "handle": tuple("handle_" + a for a in "ABCE")}
    if snapshot["mode"] == "junction_abc":
        vector_columns.update({key: tuple(key + "_" + a for a in "ABC")
                               for key in ("joint_reference", "joint_actual", "joint_error")})
        vector_columns["joint_target_robot"] = tuple("joint_target_robot_" + a for a in "ABCE")
    def write(name, columns, rows):
        with (root / name).open("w", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream)
            writer.writerow(columns)
            writer.writerows(rows)
    write("feedback.csv", feedback_columns + sum(vector_columns.values(), ()),
          ([row[k] for k in feedback_columns] + [v for k in vector_columns for v in row[k]]
           for row in snapshot["feedback"]))
    common = ("elapsed_s", "mono_time", "reference_um", "phase", "feedback_sequence", "feedback_age_s",
              "error_norm_um", "action")
    for name, rows, timing in (("decisions.csv", snapshot["decisions"], ()),
                               ("commands.csv", snapshot["commands"], ("publish_elapsed_s", "publish_mono", "publish_unix", "publish_success"))):
        columns = common + timing + tuple("reference_" + a + "_um" for a in "xyz") + tuple("target_" + a + "_mm" for a in "xyz") + tuple("handle_" + a for a in "ABCE") + tuple("delta_" + a for a in "ABCE")
        native = snapshot["mode"] == "junction_abc"
        if native:
            columns += tuple("joint_reference_" + a for a in "ABC") + tuple("joint_target_robot_" + a for a in "ABCE")
        write(name, columns, ([row[k] for k in common + timing] + list(row.get("reference_xyz_um", np.asarray(snapshot["direction"]) * row["reference_um"])) + list(row.get("target_mm", origin + np.asarray(snapshot["direction"]) * row["reference_um"] / 1000)) + list(row["handles"]) + list(row["delta"]) + (list(row["joint_reference"]) + list(row["joint_target_robot"]) if native else []) for row in rows))
    signal = ResponseSignal(**snapshot["signal"])
    # Full scheduled reference, including discontinuities and every 50 ms grid
    # point. It remains distinct from the reference actually used by decisions.
    times = set(np.arange(0, signal.duration_s, snapshot["control"]["interval_s"]).tolist())
    times.update((0.0, signal.baseline_s, signal.baseline_s + signal.active_s, signal.duration_s))
    for boundary in (signal.baseline_s, signal.baseline_s + signal.active_s):
        if boundary > 0:
            times.add(np.nextafter(boundary, 0.0))
    ref_columns = (("elapsed_s", "reference_um", "phase") + tuple("reference_" + a + "_um" for a in "xyz") +
                   tuple("target_" + a + "_mm" for a in "xyz"))
    native = snapshot["mode"] == "junction_abc"
    if native:
        ref_columns += tuple("joint_reference_" + a for a in "ABC") + tuple("joint_target_robot_" + a for a in "ABCE")
    def reference_rows():
        for elapsed in sorted(times):
            xyz = np.array(reference_xyz_at(snapshot, elapsed))
            row = [elapsed, float(xyz @ snapshot["direction"]), signal.at(elapsed)[1], *xyz, *(origin + xyz / 1000)]
            if native:
                offsets, robot, _, _ = joint_reference(signal, snapshot["origin_robot"], snapshot["joint_amplitudes"], elapsed)
                row.extend((*offsets, *robot))
            yield row
    write("reference.csv", ref_columns, reference_rows())
    metadata = {k: v for k, v in snapshot.items() if k not in ("feedback", "commands", "decisions")}
    metadata.update(analysis_reporting_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                    amplitude_basis="independent A/B degrees and C micrometres" if native else "per Cartesian axis" if snapshot["mode"] == "junction_xyz" else "along drive vector",
                    reference_um_basis="signed displacement projected onto unit drive direction",
                    xyz_timing="all Cartesian components from one feedback frame and receipt timestamp",
                    planned_duration_s=signal.duration_s,
                    recorded_duration_s=snapshot["feedback"][-1]["elapsed_s"] if snapshot["feedback"] else 0,
                    elapsed_to_stop_s=(snapshot["stopped_mono"] - snapshot["started_mono"]
                                       if snapshot.get("stopped_mono") is not None else None),
                    metrics=response_metrics(snapshot), time_basis="host monotonic elapsed time; Unix timestamps for correlation only",
                    publish_time_basis="publish API invocation; not hardware receipt or topic callback",
                    accuracy_note="Robot feedback and reconstructed needle tip; external accuracy is not inferred",
                    metric_method="unsmoothed reception-time error; trapezoidal time weighting; no interpolation across watchdog gaps",
                    files=("reference.csv", "feedback.csv", "decisions.csv", "commands.csv"))
    (root / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    return root


def load_response(directory):
    root = Path(directory)
    snapshot = json.loads((root / "metadata.json").read_text())
    if snapshot.get("schema_version") != 1:
        raise ValueError("Unknown response schema")
    def read(name):
        with (root / name).open(newline="", encoding="utf-8") as stream:
            return list(csv.DictReader(stream))
    feedback = []
    for r in read("feedback.csv"):
        row = {key: (r[key] if key == "phase" else float(r[key])) for key in (
            "elapsed_s", "unix_time", "mono_time", "sequence", "phase", "reference_um", "actual_um", "error_um", "error_norm_um", "transverse_um", "posture_deg")}
        row.update(junction_mm=tuple(float(r["junction_" + a + "_mm"]) for a in "xyz"),
                   tip_mm=tuple(float(r["tip_" + a + "_mm"]) for a in "xyz"),
                   robot=tuple(int(r["robot_" + a]) for a in "ABCE"),
                   handle=tuple(int(r["handle_" + a]) for a in "ABCE"),
                   forward=r["forward"] == "True", backward=r["backward"] == "True")
        for key, prefix, unit in (("reference_xyz_um", "reference", "um"), ("actual_xyz_um", "actual", "um"),
                                  ("error_xyz_um", "error", "um"), ("target_mm", "target", "mm")):
            if prefix + "_x_" + unit in r:
                row[key] = tuple(float(r[prefix + "_" + a + "_" + unit]) for a in "xyz")
        if "joint_reference_A" in r:
            for key in ("joint_reference", "joint_actual", "joint_error", "joint_target_robot"):
                row[key] = tuple(float(r[key + "_" + a]) for a in ("ABCE" if key == "joint_target_robot" else "ABC"))
        feedback.append(row)
    snapshot["feedback"] = feedback
    def control_rows(name):
        result = []
        for raw in read(name):
            row = {k: (v if k in ("phase", "action") else v == "True" if k == "publish_success" else float(v))
                   for k, v in raw.items() if not k.startswith(("handle_", "delta_", "reference_x", "reference_y", "reference_z", "target_", "joint_"))}
            row.update(delta=tuple(int(raw["delta_" + a]) for a in "ABCE"),
                       handles=tuple(int(raw["handle_" + a]) for a in "ABCE"))
            if "joint_reference_A" in raw:
                row["joint_reference"] = tuple(float(raw["joint_reference_" + a]) for a in "ABC")
                row["joint_target_robot"] = tuple(float(raw["joint_target_robot_" + a]) for a in "ABCE")
            if "reference_x_um" in raw:
                row["reference_xyz_um"] = tuple(float(raw["reference_" + a + "_um"]) for a in "xyz")
                row["target_mm"] = tuple(float(raw["target_" + a + "_mm"]) for a in "xyz")
            result.append(row)
        return result
    snapshot["commands"] = control_rows("commands.csv")
    snapshot["decisions"] = control_rows("decisions.csv")
    return snapshot


def _gapped(t, y, timeout):
    """Insert NaNs so lost feedback is never drawn as a measured line."""
    t, y = np.asarray(t, dtype=float), np.asarray(y, dtype=float)
    if len(t) < 2:
        return t, y
    indexes = np.flatnonzero(np.diff(t) > timeout) + 1
    return np.insert(t, indexes, np.nan), np.insert(y, indexes, np.nan)


def build_response_figures(snapshot):
    signal = ResponseSignal(**snapshot["signal"])
    rows = snapshot["feedback"]
    if not rows:
        raise ValueError("No feedback to plot")
    metrics = response_metrics(snapshot)
    t = np.array([r["elapsed_s"] for r in rows])
    timeout = snapshot["control"]["state_timeout_s"]
    figures = {}
    def canvas(title, labels):
        figure = _figure((12, 3 + 2.2 * len(labels)))
        axes = figure.subplots(len(labels), 1, sharex=True)
        figure.subplots_adjust(left=.09, right=.96, bottom=.20, top=.84, hspace=.42)
        figure.suptitle(title, x=.09, y=.965, ha="left", fontsize=16, fontweight="bold", color="#17212B")
        figure.text(.09, .913, "%s  |  %s  |  repeat %s  |  %s: %s" % (
            signal.kind.upper(), snapshot["mode"], snapshot.get("metadata", {}).get("repeat", "—"),
            snapshot["status"], snapshot["reason"]), fontsize=9, color="#5B6875")
        for axis, label in zip(axes, labels):
            _style(axis, label)
            axis.margins(x=.01, y=.12)
            axis.ticklabel_format(axis="y", style="plain", useOffset=False)
            axis.set_xlim(0, max(float(t[-1]), .1))
        axes[-1].set_xlabel("Elapsed time (s)")
        for axis in axes:
            phase_start = 0
            previous = rows[0]["phase"]
            colors = {"baseline": "#EAF2F8", "plateau": "#FDF0DF", "pulse": "#FDF0DF",
                      "recovery": "#E8F5EE", "steady": "#E8F5EE", "transient": "#F3ECF8"}
            for i in range(1, len(rows) + 1):
                if i == len(rows) or rows[i]["phase"] != previous:
                    axis.axvspan(t[phase_start], t[i - 1], color=colors.get(previous, "#F4F6F8"), alpha=.6, zorder=0)
                    if i < len(rows):
                        phase_start, previous = i, rows[i]["phase"]
        rmse = metrics["position_3d_rmse_um"]
        figure.text(.09, .115, "3D time-weighted RMSE: %s µm    |    Max 3D error: %.3f µm    |    Samples: %d" % (
            "unavailable" if rmse is None else "%.3f" % rmse, metrics["position_3d_max_um"], len(rows)), fontsize=10)
        def number(value, digits=3):
            return "unavailable" if value is None else ("%.*f" % (digits, value))
        detail = "Baseline σ: %s µm  |  Feedback interval median / P95: %s / %s ms" % (
            number(metrics["baseline_sigma_um"]),
            number(1000 * metrics["feedback_interval_median_s"]) if metrics["feedback_interval_median_s"] is not None else "unavailable",
            number(1000 * metrics["feedback_interval_p95_s"]) if metrics["feedback_interval_p95_s"] is not None else "unavailable")
        if snapshot["mode"] == "junction_abc":
            detail = "A/B/C amplitudes: %s° / %s° / %s µm  |  XYZ reference from forward kinematics" % tuple(snapshot["joint_amplitudes"])
        elif signal.kind == "sine":
            fit = metrics["sine"]
            detail = ("Drive-direction gain: %s  |  Phase lag: %s° (%s s equivalent)  |  First cycle excluded" % (
                number(fit["amplitude_ratio"]), number(fit["phase_lag_deg"]), number(fit["equivalent_lag_s"]))
                if fit["available"] else "Steady-state fit unavailable: three complete cycles without gaps required")
        elif signal.kind == "ramp":
            fit = metrics["ramp"]
            detail = "Drive-direction velocity: outward %s / return %s µm/s  |  Reference speed: %s µm/s" % (
                number(fit["outward_speed_um_s"]), number(fit["return_speed_um_s"]), number(signal.speed_um_s * snapshot.get("reference_scale", 1.0)))
        elif signal.kind == "step" and metrics["step_outward"].get("available"):
            edge = metrics["step_outward"]
            detail = "Drive-direction rise (10–90%%): %s s  |  Overshoot: %s%%  |  Settling: %s s" % (
                number(edge["rise_10_90_s"]), number(edge["overshoot_percent"]), number(edge["settling_s"]))
        elif signal.kind == "pulse":
            pulse = metrics["pulse"]
            detail = "Scheduled pulse: %s s  |  Peak during pulse: %s µm  |  Final residual: %s µm" % (
                number(signal.pulse_width_s), number(pulse["peak_displacement_um"]), number(pulse["final_residual_um"]))
        figure.text(.09, .08, detail, fontsize=9, color="#334155")
        source = snapshot.get("metadata", {}).get("data_source", "Robot feedback; reconstructed tip")
        figure.text(.09, .045, source + ". Host reference → feedback receipt. No smoothing or lag alignment.",
                    fontsize=8, color="#5B6875")
        return figure, axes
    def measured(axis, values, label, color, **kwargs):
        axis.plot(*_gapped(t, values, timeout), color=color, label=label, linewidth=1.0,
                  marker=".", markersize=1.5, **kwargs)
    def legend(axis):
        axis.legend(loc="upper left", bbox_to_anchor=(0, 1.02), frameon=False, fontsize=8, ncol=3,
                    borderaxespad=0)
    reference_xyz, actual_xyz, error_xyz, drive = _xyz(snapshot)
    tip_xyz = (np.array([r["tip_mm"] for r in rows]) - snapshot["origin_tip_mm"]) * 1000
    figure, axes = canvas("Base XYZ position response" if snapshot.get("controller") == "base_position_residual" else "XYZ position response", tuple("(%s) %s — same-frame position" % (chr(97+i), a) for i, a in enumerate("XYZ")))
    rt = np.arange(0, min(signal.duration_s, t[-1]) + .025, .05)
    boundaries = [signal.baseline_s, signal.baseline_s + signal.active_s]
    rt = np.unique(np.r_[rt, [b for b in boundaries if b <= t[-1]],
                         [np.nextafter(b, 0) for b in boundaries if 0 < b <= t[-1]]])
    controlled_label = "Base (controlled)" if snapshot.get("controller") == "base_position_residual" else "Junction (observed)" if snapshot["mode"] == "junction_abc" else "Junction (controlled)" if snapshot["mode"].startswith("junction") else "Needle tip (controlled)"
    scheduled_xyz = np.array([reference_xyz_at(snapshot, v) for v in rt])
    for i, axis in enumerate(axes):
        axis.plot(rt, scheduled_xyz[:, i], color=PLAN_COLOR, linestyle="--", linewidth=1.35, label="FK reference" if snapshot["mode"] == "junction_abc" else "Reference")
        measured(axis, actual_xyz[:, i], controlled_label, FEEDBACK_COLOR)
        if (snapshot["mode"].startswith("junction") and snapshot["mode"] != "junction_abc"
                and snapshot.get("controller") != "base_position_residual"):
            measured(axis, tip_xyz[:, i], "Reconstructed tip (auxiliary)", TIP_COLOR, linestyle=":", alpha=.8)
        axis.set_ylabel("%s displacement (µm)" % "XYZ"[i])
        legend(axis)
    figures["01_position"] = figure
    figure, axes = canvas("Synchronous XYZ tracking error", tuple("(%s) %s: reference − actual" % (chr(97+i), a) for i, a in enumerate("XYZ")) + ("(d) 3D error norm — same timestamp",))
    tolerance = snapshot["control"]["tolerance_um"]
    for i, axis in enumerate(axes[:3]):
        measured(axis, error_xyz[:, i], "Signed %s error" % "XYZ"[i], FEEDBACK_COLOR)
        axis.axhspan(-tolerance, tolerance, color=TIP_COLOR, alpha=.12, label="±1 µm component guide")
        axis.axhline(0, color="#64748B", linewidth=.65)
        axis.set_ylabel("Error (µm)")
        stat = metrics["axes"]["xyz"[i]]
        rmse = stat["time_weighted_rmse_um"]
        axis.text(.99, .93, "RMSE %s µm  |  max %.3f µm" % (
            "n/a" if rmse is None else "%.3f" % rmse, stat["max_abs_error_um"]),
            transform=axis.transAxes, ha="right", va="top", fontsize=8, color="#334155",
            bbox=dict(facecolor="white", edgecolor="none", alpha=.85, pad=3))
        legend(axis)
    measured(axes[3], np.linalg.norm(error_xyz, axis=1), "3D error norm", FEEDBACK_COLOR)
    axes[3].axhline(tolerance, color=TIP_COLOR, linestyle="--", linewidth=.8, label="1 µm final 3D criterion")
    axes[3].set_ylabel("3D error (µm)")
    legend(axes[3])
    figures["02_error"] = figure
    figure, axes = canvas("Command and feedback displacement", ("(a) Increment per publish API call", "(b) Actual displacement between adjacent feedback frames"))
    commands = [c for c in snapshot["commands"] if c.get("publish_success", True) and c["action"] != "hold"]
    if commands:
        ct = [c["publish_elapsed_s"] for c in commands]
        for i, color in enumerate(("#CC79A7", "#E69F00", FEEDBACK_COLOR)):
            values = [c["delta"][i] for c in commands]
            if any(values):
                axes[0].vlines(ct, 0, values, color=color, linewidth=.7, label=AXES_LABELS[i] + " handle increment")
                axes[0].scatter(ct, values, color=color, s=5, alpha=.5)
    else:
        axes[0].text(.5, .5, "No motion commands published", ha="center", transform=axes[0].transAxes, color="#5B6875")
    dt = np.diff(t)
    valid = dt <= timeout
    for i, color in enumerate((FEEDBACK_COLOR, PLAN_COLOR, TIP_COLOR)):
        axes[1].plot(t[1:], np.where(valid, np.diff(actual_xyz[:, i]), np.nan), color=color, linewidth=.9, label="%s displacement / feedback frame" % "XYZ"[i])
    axes[0].set_ylabel("Handle increment (counts / packet)")
    axes[1].set_ylabel("Displacement (µm / frame)")
    for axis in axes:
        axis.axhline(0, color="#64748B", linewidth=.65)
        if axis.get_legend_handles_labels()[0]:
            legend(axis)
    figures["03_command_displacement"] = figure
    if snapshot["mode"] == "junction_abc":
        figure, axes = canvas("A/B/C time response", ("(a) A rotation φ", "(b) B linkage angle α", "(c) C axial displacement"))
        for i, axis in enumerate(axes):
            unit = snapshot["joint_units"][i]
            planned = [joint_reference(signal, snapshot["origin_robot"], snapshot["joint_amplitudes"], v)[0][i] for v in rt]
            axis.plot(rt, planned, color=PLAN_COLOR, linestyle="--", linewidth=1.35, label="Axis reference")
            measured(axis, [r["joint_actual"][i] for r in rows], "Encoder feedback", FEEDBACK_COLOR)
            axis.set_ylabel("Relative %s (%s)" % ("ABC"[i], unit))
            stat = metrics["joints"]["ABC"[i]]
            value = stat["time_weighted_rmse_" + unit]
            axis.text(.99, .9, "RMSE %s %s" % ("n/a" if value is None else "%.5f" % value, unit),
                      transform=axis.transAxes, ha="right", fontsize=8)
            legend(axis)
        figures["04_joint_response"] = figure
    if snapshot.get("controller") == "base_position_residual":
        figure, axes = canvas("Reconstructed needle tip (auxiliary observation)",
                               tuple("(%s) Tip %s displacement" % (chr(97+i), a) for i, a in enumerate("XYZ")))
        for i, axis in enumerate(axes):
            measured(axis, tip_xyz[:, i], "Reconstructed tip — auxiliary", TIP_COLOR)
            axis.set_ylabel("Tip %s (µm)" % "XYZ"[i])
            legend(axis)
        figures["04_auxiliary_tip"] = figure
        figure, axes = canvas("Successful cumulative input and raw encoder changes",
                               ("(a) Cumulative handle changes", "(b) Native A/B encoder changes", "(c) Native C encoder change"))
        sent = [c for c in snapshot["commands"] if c.get("publish_success", True)]
        origin_handles = snapshot.get("origin_handles") or (sent[0]["handles"] if sent else (0,0,0,0))
        for i, color in enumerate(("#CC79A7", "#E69F00", FEEDBACK_COLOR)):
            if sent:
                axes[0].step([0]+[c["publish_elapsed_s"] for c in sent],
                             [0]+[c["handles"][i]-origin_handles[i] for c in sent],
                             where="post", color=color, lw=1, label="Handle " + AXES_LABELS[i])
            raw = [r["robot"][i]-snapshot["origin_robot"][i] for r in rows]
            measured(axes[1 if i < 2 else 2], raw, "Encoder " + AXES_LABELS[i], color)
        for axis in axes:
            axis.set_ylabel("Count change")
            legend(axis)
        figures["05_input_encoders"] = figure
    # CLI exports do not inherit the Tk application's CJK font settings.
    # Set fonts on this report only, leaving other figures and rcParams alone.
    from matplotlib import font_manager
    from matplotlib.text import Text
    installed = {font.name for font in font_manager.fontManager.ttflist}
    cjk_font = next((name for name in (
        "Noto Sans CJK SC", "Noto Sans CJK JP", "WenQuanYi Zen Hei",
        "Droid Sans Fallback",
    ) if name in installed), None)
    if cjk_font:
        for figure in figures.values():
            for label in figure.findobj(Text):
                if any("\u3400" <= char <= "\u9fff" for char in label.get_text()):
                    label.set_fontfamily(cjk_font)
    return figures


AXES_LABELS = ("A", "B", "C", "E")


def export_response_figures(snapshot, directory):
    root = Path(directory)
    root.mkdir(parents=True, exist_ok=True)
    figures = build_response_figures(snapshot)
    try:
        for name, figure in figures.items():
            save_figure(figure, root / (name + ".png"))
            save_figure(figure, root / (name + ".pdf"))
    finally:
        for figure in figures.values():
            figure.clear()
    return response_metrics(snapshot)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory")
    parser.add_argument("--output")
    args = parser.parse_args()
    snapshot = load_response(args.directory)
    export_response_figures(snapshot, args.output or args.directory)


if __name__ == "__main__":
    main()
