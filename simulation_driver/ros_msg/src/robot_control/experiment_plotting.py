"""ROS-free, unsmoothed experiment figures and reproducible batch reports.

Run with ``python3 -m robot_control.experiment_plotting INPUT OUTPUT``.
Geometry uses all recorded feedback; error statistics use recorded tracking.
"""

import argparse
import csv
from dataclasses import dataclass, field
import hashlib
import json
import math
from pathlib import Path

import numpy as np


FEEDBACK_COLOR = "#0072B2"
PLAN_COLOR = "#D55E00"
AXIS_COLORS = ("#0072B2", "#009E73", "#CC79A7")
PHASE_COLORS = ("#EAF2F8", "#FDF0DF", "#E8F5EE", "#F3ECF8")


def error_metrics(values):
    """Sample-weighted statistics; nearest-rank P95 matches the recorder."""
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return None
    if values.ndim != 1 or not np.isfinite(values).all() or (values < 0).any():
        raise ValueError("Error samples must be finite, nonnegative scalars")
    ordered = np.sort(values)
    return dict(count=int(values.size), mean_um=float(values.mean()),
                rms_um=float(np.sqrt(np.mean(values ** 2))),
                p95_um=float(ordered[math.ceil(0.95 * len(ordered)) - 1]),
                max_um=float(ordered[-1]))


def nearest_polyline_points(points, plan, chunk_size=128):
    """Return nearest XYZ and segment index, including clamped endpoints.

    Degenerate segments represent their start point. Ties select the first
    segment; the projection describes geometry, never time correspondence.
    """
    points = np.asarray(points, dtype=float).reshape((-1, 3))
    plan = np.asarray(plan, dtype=float).reshape((-1, 3))
    if len(plan) < 2 or not np.isfinite(plan).all() or not np.isfinite(points).all():
        raise ValueError("Projection requires finite XYZ and at least two plan points")
    if not isinstance(chunk_size, int) or chunk_size <= 0:
        raise ValueError("Projection chunk_size must be a positive integer")
    starts, steps = plan[:-1], np.diff(plan, axis=0)
    lengths = np.einsum("si,si->s", steps, steps)
    nearest = np.empty_like(points)
    segments = np.empty(len(points), dtype=int)
    for offset in range(0, len(points), chunk_size):
        block = points[offset:offset + chunk_size]
        relative = block[:, None, :] - starts
        numerator = np.einsum("bsi,si->bs", relative, steps)
        fraction = np.divide(numerator, lengths, out=np.zeros_like(numerator),
                             where=lengths > 0)
        np.clip(fraction, 0, 1, out=fraction)
        candidates = starts + fraction[:, :, None] * steps
        delta = block[:, None, :] - candidates
        best = np.argmin(np.einsum("bsi,bsi->bs", delta, delta), axis=1)
        nearest[offset:offset + len(block)] = candidates[np.arange(len(block)), best]
        segments[offset:offset + len(block)] = best
    return nearest, segments


def _array(rows, columns, label):
    result = np.asarray(rows, dtype=float).reshape((-1, columns))
    if not np.isfinite(result).all():
        raise ValueError(label + " contains non-finite values")
    if len(result) and (np.diff(result[:, 0]) < 0).any():
        raise ValueError(label + " timestamps must be chronological")
    return result


@dataclass
class ExperimentPlotData:
    run_id: str
    run: dict
    feedback: object  # Nx4: Unix time, RCM-frame XYZ in mm
    tracking: object  # Nx10: Unix, tip XYZ, target XYZ, errors in um, waypoint
    plan: object      # Nx3: RCM-frame XYZ in mm
    phases: tuple = ()
    notes: list = field(default_factory=list)
    source_files: tuple = ()
    _deviation_cache: object = field(default=None, init=False, repr=False)

    def __post_init__(self):
        self.feedback = _array(self.feedback, 4, "Feedback")
        self.tracking = _array(self.tracking, 10, "Tracking")
        self.plan = np.asarray(self.plan, dtype=float).reshape((-1, 3))
        if not np.isfinite(self.plan).all():
            raise ValueError("Plan contains non-finite values")
        if len(self.phases) != len(self.tracking):
            raise ValueError("Phase count must match tracking count")
        if len(self.tracking):
            if (self.tracking[:, 7:9] < 0).any():
                raise ValueError("Recorded errors must be nonnegative")
            indexes = self.tracking[:, 9]
            if ((indexes < 0).any() or (indexes != np.floor(indexes)).any()
                    or (len(self.plan) and (indexes >= len(self.plan)).any())):
                raise ValueError("Tracking waypoint index outside the recorded plan")
        if not len(self.feedback):
            if not len(self.tracking):
                raise ValueError("No needle-tip feedback for selected run")
            self.notes.append("Geometry falls back to tracking; full feedback unavailable")
        if not len(self.plan):
            self.notes.append("No recorded plan; geometry only")

    @property
    def geometry(self):
        return self.feedback if len(self.feedback) else self.tracking[:, :4]

    @property
    def origin(self):
        return self.plan[0] if len(self.plan) else self.geometry[0, 1:4]

    @property
    def t0(self):
        times = [self.geometry[0, 0]]
        if len(self.tracking):
            times.append(self.tracking[0, 0])
        return min(times)

    @property
    def name(self):
        return str(self.run.get("validation_experiment") or
                   self.run.get("experiment_id") or self.run.get("kind") or "trajectory")

    @property
    def plane_axis(self):
        plane = str(self.run.get("plane", "")).lower()
        return {"xy": 2, "xz": 1, "yz": 0}.get(plane)

    @property
    def deviations(self):
        if self._deviation_cache is None:
            nearest, segments = nearest_polyline_points(self.tracking[:, 1:4], self.plan)
            self._deviation_cache = (nearest, (self.tracking[:, 1:4] - nearest) * 1000, segments)
        return self._deviation_cache

    def phase_metrics(self):
        result = {}
        for phase in dict.fromkeys(self.phases):
            mask = np.array([value == phase for value in self.phases])
            result[phase or "unlabelled"] = error_metrics(self.tracking[mask, 7])
        return result

    def summary(self):
        return dict(experiment=self.name, run_id=self.run_id,
                    feedback_count=len(self.feedback), tracking_count=len(self.tracking),
                    geometry_count=len(self.geometry),
                    duration_s=float(max(self.geometry[-1, 0],
                        self.tracking[-1, 0] if len(self.tracking) else self.t0) - self.t0),
                    origin_rcm_mm=self.origin.tolist(),
                    path_error=error_metrics(self.tracking[:, 7]),
                    target_error=error_metrics(self.tracking[:, 8]),
                    phases=self.phase_metrics(), notes=self.notes,
                    status=self.run.get("status", "unknown"))


def data_from_snapshot(snapshot, runs, run_id=None):
    """Use each sample's registered RCM, selecting exactly one recorded run."""
    if runs:
        run_id = run_id if run_id is not None else next(reversed(runs))
        run = runs[run_id]
    else:
        run_id, run = "feedback_only", {}
    feedback = []
    for record in snapshot["records"]:
        context = record.get("context", {})
        if runs and context.get("run_id") != run_id:
            continue
        tip, rcm = record.get("tip"), context.get("rcm_mm")
        if tip is not None and rcm is not None:
            feedback.append((record["unix_time"],
                             *(tip[i] - rcm[i] for i in range(3))))
    tracking, phases = [], []
    labels = run.get("waypoint_phases", ())
    for row in run.get("tracking", ()):
        index = row[5]
        tracking.append((row[0], *row[1], *row[2], row[3] * 1000,
                         row[4] * 1000, index))
        phases.append(labels[int(index)] if labels and 0 <= int(index) < len(labels) else "")
    notes = []
    if snapshot.get("dropped_feedback_records"):
        notes.append("Feedback ring buffer dropped %d samples (session total)" %
                     snapshot["dropped_feedback_records"])
    if run.get("tracking_dropped"):
        notes.append("Tracking dropped %d samples" % run["tracking_dropped"])
    return ExperimentPlotData(str(run_id), run, feedback, tracking,
                              run.get("planned_path", ()), tuple(phases), notes)


def _read_csv(path):
    with path.open(newline="", encoding="utf-8-sig") as stream:
        return list(csv.DictReader(stream))


def load_experiment(directory, run_id=None):
    """Read an export without mutating it; require selection for multi-run data."""
    directory = Path(directory)
    metadata_path = directory / "experiment_metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("schema_version") != 2 or metadata.get("coordinate_units") != "mm":
        raise ValueError("Expected schema_version=2 with mm coordinates")
    runs = metadata["runs"]
    if run_id is None:
        if len(runs) != 1:
            raise ValueError("Select --run-id for an export containing multiple runs")
        run_id = next(iter(runs))
    if run_id not in runs:
        raise ValueError("Unknown run_id: " + run_id)
    run = runs[run_id]
    paths = {}
    for key in ("feedback", "plans", "tracking"):
        name = metadata["files"][key]
        if Path(name).name != name:
            raise ValueError("Export file references must be local basenames")
        paths[key] = directory / name
    feedback = []
    for row in _read_csv(paths["feedback"]):
        if row["run_id"] != run_id:
            continue
        xyz = [row["tip_rcm_" + axis + "_mm"] for axis in "xyz"]
        if not all(xyz):
            raise ValueError("Selected feedback lacks RCM-frame needle-tip coordinates")
        feedback.append((float(row["unix_time"]), *map(float, xyz)))
    plans = [row for row in _read_csv(paths["plans"]) if row["run_id"] == run_id]
    if [int(row["waypoint_index"]) for row in plans] != list(range(len(plans))):
        raise ValueError("Plan waypoint indices must be contiguous from zero")
    plan = [tuple(float(row[axis + "_mm"]) for axis in "xyz") for row in plans]
    tracking, phases = [], []
    for row in _read_csv(paths["tracking"]):
        if row["run_id"] != run_id:
            continue
        tracking.append(tuple(float(row[key]) for key in (
            "unix_time", "tip_x_mm", "tip_y_mm", "tip_z_mm", "target_x_mm",
            "target_y_mm", "target_z_mm", "path_error_um", "target_error_um", "waypoint_index")))
        phases.append(row.get("phase", ""))
    notes = []
    for key in ("dropped_feedback_records", "dropped_command_records"):
        if metadata.get(key):
            notes.append("%s: %d (session total)" % (key, metadata[key]))
    if run.get("tracking_dropped"):
        notes.append("Tracking dropped %d samples" % run["tracking_dropped"])
    data = ExperimentPlotData(run_id, run, feedback, tracking, plan, tuple(phases), notes,
                              (metadata_path, *paths.values()))
    recorded = run.get("metrics")
    calculated = data.summary()["path_error"]
    if recorded and calculated and any(
            not math.isclose(float(recorded[key]), calculated[key], rel_tol=1e-9, abs_tol=1e-9)
            for key in calculated):
        raise ValueError("Tracking statistics disagree with recorded metadata")
    return data


def deviation_focus(data):
    """Select recorded contour/outward samples, otherwise preserve the whole run."""
    for phase in ("contour", "outward"):
        if phase in data.phases:
            return np.array([value == phase for value in data.phases]), phase
    return np.ones(len(data.tracking), dtype=bool), "all phases"


def deviation_window(data):
    """Choose the largest in-plane residual within the explicitly named focus."""
    mask, phase = deviation_focus(data)
    indices = {"xy": (0, 1), "xz": (0, 2), "yz": (1, 2)}.get(
        str(data.run.get("plane", "")).lower(), (0, 2))
    nearest, residual, _ = data.deviations
    candidates = np.flatnonzero(mask)
    magnitudes = np.linalg.norm(residual[candidates][:, indices], axis=1)
    sample = int(candidates[np.argmax(magnitudes)])
    centre = (nearest[sample] - data.origin) * 1000
    radius = max(4.0, float(np.linalg.norm(residual[sample, list(indices)])) * 3)
    return mask, phase, indices, sample, centre, radius


def _figure(figsize):
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure
    figure = Figure(figsize=figsize, facecolor="white")
    FigureCanvasAgg(figure)
    return figure


def _style(axis, title):
    axis.set_title(title, loc="left", fontsize=10, fontweight="bold", pad=16)
    axis.tick_params(labelsize=8, colors="#4B5563")
    axis.grid(True, color="#CBD5E1", alpha=0.45, linewidth=0.5)
    axis.set_axisbelow(True)
    for spine in axis.spines.values():
        spine.set_color("#CBD5E1")
    axis.xaxis.label.set_size(9)
    axis.yaxis.label.set_size(9)


def _geometry_limits(axis, points, plan, indices):
    values = np.concatenate([points, plan]) if len(plan) else points
    lower, upper = values[:, indices].min(axis=0), values[:, indices].max(axis=0)
    centre = (lower + upper) / 2
    radius = max(float(max(upper - lower)) * 0.56, 1.0)
    axis.set_xlim(centre[0] - radius, centre[0] + radius)
    axis.set_ylim(centre[1] - radius, centre[1] + radius)
    axis.set_aspect("equal", adjustable="box")


def _projection(axis, points, plan, indices, title):
    _style(axis, title)
    axis.plot(points[:, indices[0]], points[:, indices[1]], color=FEEDBACK_COLOR,
              lw=0.85, alpha=0.85, label="All feedback", rasterized=True)
    if len(plan):
        axis.plot(plan[:, indices[0]], plan[:, indices[1]], color=PLAN_COLOR,
                  lw=1, ls=(0, (5, 4)), label="Executed plan")
    axis.scatter(*points[0, indices], s=40, marker="o", facecolors="none",
                 edgecolors="#009E73", linewidths=1.3, zorder=5, label="Start")
    axis.scatter(*points[-1, indices], s=28, marker="x", color="#111827",
                 linewidths=1.1, zorder=6, label="End")
    for setter, index in zip((axis.set_xlabel, axis.set_ylabel), indices):
        setter("Δ%s (µm)" % "XYZ"[index])
    _geometry_limits(axis, points, plan, indices)


def _phase_spans(axis, data):
    if not len(data.tracking):
        return
    times = data.tracking[:, 0] - data.t0
    phases = data.phases
    unique = list(dict.fromkeys(phases))
    starts = [0] + [i for i in range(1, len(phases)) if phases[i] != phases[i-1]]
    for start, stop in zip(starts, starts[1:] + [len(phases)]):
        left = times[start] if start == 0 else (times[start-1] + times[start]) / 2
        right = times[-1] if stop == len(phases) else (times[stop-1] + times[stop]) / 2
        color = PHASE_COLORS[unique.index(phases[start]) % len(PHASE_COLORS)]
        axis.axvspan(left, right, color=color, zorder=0)
        if len(starts) <= 3:
            axis.text((left+right)/2, 1.015, phases[start] or "unlabelled",
                      ha="center", va="bottom", fontsize=7, color="#64748B",
                      transform=axis.get_xaxis_transform())


def build_figure(data):
    """Nine panels preserve physical aspect and expose small deviations."""
    figure = _figure((15, 11.5))
    grid = figure.add_gridspec(3, 3, left=0.06, right=0.98, bottom=0.14,
                              top=0.88, wspace=0.31, hspace=0.48)
    points = (data.geometry[:, 1:4] - data.origin) * 1000
    plan = (data.plan - data.origin) * 1000
    time_values = data.geometry[:, 0] - data.t0
    axes = [figure.add_subplot(grid[0, 0], projection="3d")]
    axes.extend(figure.add_subplot(grid[row, col])
                for row, col in ((0, 1), (0, 2), (1, 0), (1, 1), (1, 2),
                                 (2, 0), (2, 1), (2, 2)))
    axis = axes[0]
    _style(axis, "(a) 3D trajectory · equal scale")
    axis.plot(*points.T, color=FEEDBACK_COLOR, lw=0.85, rasterized=True)
    if len(plan):
        axis.plot(*plan.T, color=PLAN_COLOR, lw=1, ls=(0, (5, 4)))
    axis.scatter(*points[0], s=40, facecolors="none", edgecolors="#009E73", depthshade=False)
    axis.scatter(*points[-1], s=28, marker="x", color="#111827", depthshade=False)
    values = np.concatenate([points, plan]) if len(plan) else points
    lower, upper = values.min(axis=0), values.max(axis=0)
    centre, radius = (lower+upper)/2, max(float(max(upper-lower))*0.56, 1)
    for setter, label, c in zip((axis.set_xlim, axis.set_ylim, axis.set_zlim), "XYZ", centre):
        setter(c-radius, c+radius)
    axis.set_xlabel("ΔX (µm)", labelpad=2, fontsize=8)
    axis.set_ylabel("ΔY (µm)", labelpad=2, fontsize=8)
    axis.set_zlabel("ΔZ (µm)", labelpad=2, fontsize=8)
    axis.set_box_aspect((1, 1, 1))
    axis.view_init(elev=26, azim=-55)
    _projection(axes[1], points, plan, (0, 1), "(b) XY projection · equal scale")
    _projection(axes[2], points, plan, (0, 2), "(c) XZ projection · equal scale")
    _projection(axes[3], points, plan, (1, 2), "(d) YZ projection · equal scale")
    figure.legend(*axes[1].get_legend_handles_labels(), loc="upper center",
                  bbox_to_anchor=(0.5, 0.944), ncol=4, frameon=False, fontsize=9)
    axis = axes[4]
    _style(axis, "(e) Signed deviation from nearest plan point")
    if len(data.plan) >= 2 and len(data.tracking):
        for i in range(3):
            axis.plot(data.tracking[:, 0] - data.t0, data.deviations[1][:, i],
                      color=AXIS_COLORS[i], lw=0.7, label="e" + "XYZ"[i], rasterized=True)
        axis.axhline(0, color=PLAN_COLOR, ls="--", lw=0.8)
        axis.legend(loc="best", fontsize=7, ncol=3, frameon=False)
    else:
        _unavailable(axis)
    axis.set_xlabel("Time (s)")
    axis.set_ylabel("Geometric deviation (µm)")
    _phase_spans(axis, data)
    axis = axes[5]
    plane_axis = data.plane_axis
    if plane_axis is not None and len(plan) and np.ptp(plan[:, plane_axis]) < 1e-6:
        _style(axis, "(f) Out-of-plane Δ%s · expanded scale" % "XYZ"[plane_axis])
        axis.plot(time_values, points[:, plane_axis], color=AXIS_COLORS[plane_axis],
                  lw=0.7, rasterized=True)
        axis.axhline(0, color=PLAN_COLOR, ls="--", lw=1)
        axis.set_xlabel("Time (s)")
        axis.set_ylabel("Signed plane deviation (µm)")
        _phase_spans(axis, data)
    else:
        _style(axis, "(f) Path RMS by recorded phase")
        metrics = data.phase_metrics()
        if metrics:
            axis.bar(list(metrics), [x["rms_um"] for x in metrics.values()],
                     color=FEEDBACK_COLOR, width=0.65)
            axis.tick_params(axis="x", labelrotation=30, labelsize=7)
            axis.set_ylabel("Path RMS (µm)")
        else:
            _unavailable(axis)
    stats = data.summary()
    for axis, column, title, key in ((axes[6], 7, "(g) Distance to entire planned path", "path_error"),
                                   (axes[7], 8, "(h) Residual to current target waypoint", "target_error")):
        _style(axis, title)
        axis.set_xlabel("Time (s)")
        axis.set_ylabel("Distance (µm)")
        if len(data.tracking):
            times = data.tracking[:, 0] - data.t0
            axis.plot(times, data.tracking[:, column], color=FEEDBACK_COLOR,
                      lw=0.6, rasterized=True)
            metric = stats[key]
            axis.axhline(metric["rms_um"], color=PLAN_COLOR, ls="--", lw=1,
                         label="RMS %.2f" % metric["rms_um"])
            axis.axhline(metric["p95_um"], color="#64748B", ls=":", lw=1,
                         label="P95 %.2f" % metric["p95_um"])
            axis.set_ylim(bottom=0, top=max(float(data.tracking[:, column].max())*1.18, 1))
            axis.legend(loc="upper right", fontsize=7, frameon=False)
            _phase_spans(axis, data)
        else:
            _unavailable(axis)
    axis = axes[8]
    _style(axis, "(i) Path error distribution by phase")
    axis.set_xlabel("Distance to planned path (µm)")
    axis.set_ylabel("Samples at or below error (%)")
    if len(data.tracking):
        for phase in dict.fromkeys(data.phases):
            mask = np.array([p == phase for p in data.phases])
            values = np.sort(data.tracking[mask, 7])
            axis.step(values, np.arange(1, len(values)+1)/len(values)*100,
                      where="post", label=phase or "unlabelled", lw=1)
        axis.axhline(95, color="#64748B", ls=":", lw=0.8)
        axis.set_ylim(0, 102)
        axis.set_xlim(left=0)
        axis.legend(loc="lower right", fontsize=6.5, ncol=2 if len(set(data.phases)) > 4 else 1,
                    frameon=False)
    else:
        _unavailable(axis)
    figure.suptitle(data.name.replace("_", " ").upper() + " | trajectory and tracking diagnostics",
                   y=0.98, fontsize=16, fontweight="bold")
    origin_text = ", ".join("%.6f" % x for x in data.origin)
    path = stats["path_error"]
    metric_text = ("Path: RMS %.3f | mean %.3f | P95 %.3f | max %.3f µm" %
                   tuple(path[k] for k in ("rms_um", "mean_um", "p95_um", "max_um"))
                   if path else "No recorded tracking errors")
    caption = ("Run %s   |   Geometry %s: %d samples   |   Tracking: %d samples   |   Duration %.3f s\n"
               "All coordinates: ΔXYZ = RCM-frame XYZ − planned start (%s) mm. No smoothing.\n"
               "%s\nReconstructed needle-tip feedback; external positioning accuracy is not inferred. "
               "Path distance uses the entire polyline; waypoint residual is a separate measure." %
               (data.run_id, "full feedback" if len(data.feedback) else "tracking fallback",
                len(data.geometry), len(data.tracking), stats["duration_s"], origin_text, metric_text))
    if data.notes:
        caption += "\n" + "; ".join(data.notes)
    figure.text(0.06, 0.035, caption, fontsize=8, color="#475569", linespacing=1.8)
    return figure


def build_deviation_figure(data):
    """Local physical zoom and separately scaled XYZ deviations without clipping."""
    from matplotlib.patches import Rectangle

    if len(data.plan) < 2 or not len(data.tracking):
        raise ValueError("Deviation detail requires a recorded plan and tracking")
    mask, phase, indices, sample, centre, radius = deviation_window(data)
    nearest, residual, _ = data.deviations
    figure = _figure((13, 11))
    grid = figure.add_gridspec(3, 2, left=0.075, right=0.98, bottom=0.17,
                              top=0.87, wspace=0.28, hspace=0.48)
    axes = [figure.add_subplot(grid[row, col]) for row in range(3) for col in range(2)]
    points, plan = (data.geometry[:, 1:4] - data.origin) * 1000, (data.plan - data.origin) * 1000
    plane = "".join("XYZ"[i] for i in indices)
    _projection(axes[0], points, plan, indices, "(a) %s overview · zoom location marked" % plane)
    x, y = centre[list(indices)]
    axes[0].add_patch(Rectangle((x-radius, y-radius), 2*radius, 2*radius,
                               fill=False, edgecolor="#CC79A7", lw=1.5, zorder=7))
    axes[0].annotate("Zoom below", (x, y), xytext=(0.04, 0.9), textcoords="axes fraction",
                     fontsize=8, color="#9D3979", arrowprops=dict(arrowstyle="->", color="#9D3979"))
    figure.legend(*axes[0].get_legend_handles_labels(), loc="upper center",
                  bbox_to_anchor=(0.5, 0.944), ncol=4, frameon=False, fontsize=9)
    axis = axes[2]
    _style(axis, "(c) Local %s zoom · true coordinates, equal scale" % plane)
    selected = data.tracking[mask]
    geometry_mask = ((data.geometry[:, 0] >= selected[0, 0]) &
                     (data.geometry[:, 0] <= selected[-1, 0]))
    local = points[geometry_mask]
    if not len(local):
        local = (selected[:, 1:4] - data.origin) * 1000
    axis.plot(plan[:, indices[0]], plan[:, indices[1]], color=PLAN_COLOR, lw=1.2, ls="--")
    axis.plot(local[:, indices[0]], local[:, indices[1]], color=FEEDBACK_COLOR,
              lw=0.8, rasterized=True)
    tip = (data.tracking[sample, 1:4] - data.origin) * 1000
    axis.scatter(x, y, s=24, color=PLAN_COLOR, zorder=5)
    axis.scatter(*tip[list(indices)], s=24, color=FEEDBACK_COLOR, zorder=5)
    axis.plot([x, tip[indices[0]]], [y, tip[indices[1]]], color="#9D3979", lw=1.2)
    axis.set_xlim(x-radius, x+radius)
    axis.set_ylim(y-radius, y+radius)
    axis.set_aspect("equal", adjustable="box")
    axis.ticklabel_format(style="plain", useOffset=False)
    axis.set_xlabel("Δ%s (µm)" % "XYZ"[indices[0]])
    axis.set_ylabel("Δ%s (µm)" % "XYZ"[indices[1]])
    overview_width = np.diff(axes[0].get_xlim())[0]
    axis.text(0.02, 0.02, "Window %.1f × %.1f µm\n%.0f× axis magnification" %
              (2*radius, 2*radius, overview_width/(2*radius)), transform=axis.transAxes,
              fontsize=7, bbox=dict(facecolor="white", edgecolor="none", alpha=0.85))
    times = data.tracking[mask, 0] - data.t0
    for i, axis in enumerate((axes[1], axes[3], axes[5])):
        _style(axis, "(%s) Signed e%s · %s · independent scale" %
               ("bdf"[i], "XYZ"[i], phase))
        values = residual[mask, i]
        axis.plot(times, values, color=AXIS_COLORS[i], lw=0.8, rasterized=True)
        axis.axhline(0, color=PLAN_COLOR, ls="--", lw=0.9)
        lower, upper = min(float(values.min()), 0), max(float(values.max()), 0)
        pad = max((upper-lower)*0.12, 0.05)
        axis.set_ylim(lower-pad, upper+pad)
        axis.set_xlabel("Time (s)")
        axis.set_ylabel("e%s (µm)" % "XYZ"[i])
        axis.text(0.02, 0.97, "range [%.3f, %.3f] µm" % (values.min(), values.max()),
                  va="top", fontsize=8, transform=axis.transAxes)
    axis = axes[4]
    _style(axis, "(e) Path distance · %s only" % phase)
    axis.plot(times, selected[:, 7], color=FEEDBACK_COLOR, lw=0.7, rasterized=True)
    metric = error_metrics(selected[:, 7])
    axis.axhline(metric["rms_um"], color=PLAN_COLOR, ls="--", lw=1,
                 label="RMS %.3f µm" % metric["rms_um"])
    axis.set_ylim(0, max(float(selected[:, 7].max())*1.15, 0.1))
    axis.set_xlabel("Time (s)")
    axis.set_ylabel("Recorded path distance (µm)")
    axis.legend(loc="upper right", fontsize=8, frameon=False)
    figure.suptitle(data.name.replace("_", " ").upper() + " | micrometre deviation detail",
                   y=0.98, fontsize=16, fontweight="bold")
    caption = ("Focus: %s (%d tracking samples). Full run and all peaks remain in trajectory.png.\n"
               "eXYZ = tip − nearest point on the entire executed polyline; each component has its own axis scale.\n"
               "Zoom selected at maximum %s in-plane deviation within focus: t = %.3f s; no displacement amplification or smoothing.\n"
               "Local geometry uses feedback within the selected time interval; zoom width %.3f µm. "
               "All coordinates use µm relative to planned start.\n"
               "This is geometric deviation, not time-aligned following error or external positioning accuracy." %
               (phase, int(mask.sum()), plane, data.tracking[sample, 0]-data.t0, 2*radius))
    figure.text(0.075, 0.03, caption, fontsize=8, color="#475569", linespacing=1.8)
    return figure


def _unavailable(axis):
    axis.text(0.5, 0.5, "No recorded tracking samples", ha="center", va="center",
              transform=axis.transAxes, fontsize=9)


def save_figure(figure, filename):
    """Explicit Agg canvas works without a display or Tk/ROS event loop."""
    filename = Path(filename)
    figure.savefig(str(filename), dpi=220, facecolor="white", bbox_inches="tight")


def save_panel_png(figure, axis, filename, pad=0.08):
    """Save one labelled subplot as a standalone PNG crop.

    The renderer is drawn before measuring the tight bounding box so titles,
    tick labels and axis labels are included.  This keeps the source figure
    layout unchanged while making every ``(a)``, ``(b)`` ... panel directly
    reusable in a report or slide.
    """
    filename = Path(filename)
    figure.canvas.draw()
    renderer = figure.canvas.get_renderer()
    bbox = axis.get_tightbbox(renderer)
    if bbox is None:
        raise ValueError("Cannot determine subplot bounds")
    bbox = bbox.transformed(figure.dpi_scale_trans.inverted()).expanded(1.08, 1.12)
    figure.savefig(str(filename), dpi=220, facecolor="white", bbox_inches=bbox,
                   pad_inches=pad)


TRAJECTORY_PANEL_NAMES = (
    "a_3d_trajectory", "b_xy_projection", "c_xz_projection", "d_yz_projection",
    "e_signed_deviation", "f_plane_or_phase", "g_path_distance",
    "h_target_residual", "i_error_distribution",
)
DEVIATION_PANEL_NAMES = (
    "a_overview", "b_ex", "c_local_zoom", "d_ey", "e_path_distance", "f_ez",
)


def save_panel_set(figure, directory, names):
    """Save all axes from a composite figure as numbered PNG panels."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    if len(figure.axes) < len(names):
        raise ValueError("Panel name count does not match figure axes")
    paths = []
    for index, name in enumerate(names, start=1):
        path = directory / ("%02d_%s.png" % (index, name))
        save_panel_png(figure, figure.axes[index - 1], path)
        paths.append(path)
    return paths


def export_figure(data, filename):
    figure = build_figure(data)
    target = Path(filename)
    try:
        save_figure(figure, target)
        save_panel_set(figure, target.with_name(target.stem + "_parts"),
                       TRAJECTORY_PANEL_NAMES)
    finally:
        figure.clear()
    if len(data.plan) >= 2 and len(data.tracking):
        detail = build_deviation_figure(data)
        try:
            detail_target = target.with_name(target.stem + "_deviation" + target.suffix)
            save_figure(detail, detail_target)
            save_panel_set(detail, target.with_name(target.stem + "_deviation_parts"),
                           DEVIATION_PANEL_NAMES)
        finally:
            detail.clear()


def build_overview(datasets):
    columns = 3
    rows = math.ceil(len(datasets)/columns)
    figure = _figure((15, rows*3.7 + 1.4))
    grid = figure.add_gridspec(rows, columns, left=0.06, right=0.98, bottom=0.12,
                              top=0.90, wspace=0.38, hspace=0.65)
    for i, data in enumerate(datasets):
        axis = figure.add_subplot(grid[i//columns, i%columns])
        plane = str(data.run.get("plane", "")).lower()
        indices = {"xy": (0, 1), "xz": (0, 2), "yz": (1, 2)}.get(plane, (0, 2))
        points, plan = (data.geometry[:, 1:4]-data.origin)*1000, (data.plan-data.origin)*1000
        _projection(axis, points, plan, indices, data.name.replace("_", " "))
        metric = data.summary()["path_error"]
        if metric:
            axis.text(0, -0.30, "RMS %.2f | P95 %.2f | max %.2f µm" %
                      tuple(metric[k] for k in ("rms_um", "p95_um", "max_um")),
                      fontsize=8, transform=axis.transAxes)
    figure.suptitle("Validation batch | geometric projections and recorded path error",
                   fontsize=16, fontweight="bold", y=0.985)
    figure.legend(*figure.axes[0].get_legend_handles_labels(), loc="upper center",
                  bbox_to_anchor=(0.5, 0.958), ncol=4, fontsize=9, frameon=False)
    figure.text(0.06, 0.018, "Each panel uses an equal physical scale; limits vary by experiment. "
                "Space experiments show XZ projection; see individual figures for 3D and all projections.\n"
                "Geometry: all feedback. Metrics: recorded tracking, all phases. "
                "Coordinates relative to each planned start. Reconstructed tip, no external accuracy claim.",
                fontsize=9, color="#475569")
    return figure


def build_metrics_comparison(datasets):
    figure = _figure((13, max(4, len(datasets)*0.35 + 1.8)))
    grid = figure.add_gridspec(1, 3, left=0.16, right=0.98, bottom=0.14,
                              top=0.84, wspace=0.25)
    metrics = [data.summary()["path_error"] for data in datasets]
    positions = np.arange(len(datasets))
    for col, key, title in ((0, "rms_um", "RMS"), (1, "p95_um", "P95"), (2, "max_um", "Maximum")):
        axis = figure.add_subplot(grid[0, col])
        _style(axis, title)
        values = [metric[key] if metric else float("nan") for metric in metrics]
        axis.barh(positions, values, color=FEEDBACK_COLOR, height=0.65)
        axis.set_yticks(positions)
        axis.set_yticklabels([data.name.replace("_", " ") for data in datasets] if col == 0 else
                            [""]*len(datasets))
        axis.invert_yaxis()
        axis.set_xlabel("Distance to planned path (µm)")
        axis.set_xlim(0, max([v for v in values if math.isfinite(v)] or [1])*1.22)
        for y, value in zip(positions, values):
            if math.isfinite(value):
                axis.text(value, y, "  %.2f" % value, va="center", fontsize=8)
    figure.suptitle("Validation batch | recorded path error comparison", y=0.97,
                   fontsize=15, fontweight="bold")
    figure.text(0.16, 0.035, "All phases included; sample-weighted tracking statistics. "
                "P95 uses nearest rank. Different motion geometries and durations.\n"
                "Reconstructed needle-tip feedback; external positioning accuracy is not inferred.",
                fontsize=9, color="#475569")
    return figure


def build_total_table(datasets):
    """Render the final batch table after all individual figures are written.

    The overview and metric plots are useful for visual comparison, while this
    table is the stable one-page index for reporting: one row per run, the
    sample counts, path-error statistics and target residual are visible
    together.  It is deliberately generated last by ``replot_batch`` so a
    partially exported batch cannot be mistaken for a complete summary.
    """
    from matplotlib.colors import LinearSegmentedColormap, Normalize

    figure = _figure((16, 8.8))
    axis = figure.add_axes([0.03, 0.12, 0.94, 0.76])
    axis.axis("off")
    rows = []
    metrics = []
    for data in datasets:
        summary = data.summary()
        path, target = summary["path_error"], summary["target_error"]
        metrics.append(path["rms_um"] if path else float("nan"))
        rows.append([
            summary["experiment"].replace("_", " "),
            str(data.run.get("plane", "space") or "space").upper(),
            "%d" % summary["feedback_count"],
            "%d" % summary["tracking_count"],
            "%.3f" % path["rms_um"] if path else "—",
            "%.3f" % path["p95_um"] if path else "—",
            "%.3f" % path["max_um"] if path else "—",
            "%.3f" % target["rms_um"] if target else "—",
            "%.1f s" % summary["duration_s"],
        ])
    columns = ["Experiment", "Plane", "Feedback", "Tracking", "Path RMS (µm)",
               "P95 (µm)", "Max (µm)", "Target RMS (µm)", "Duration"]
    table = axis.table(cellText=rows, colLabels=columns, loc="center",
                       cellLoc="center", colLoc="center",
                       colWidths=[0.20, 0.08, 0.10, 0.10, 0.11, 0.10, 0.10, 0.12, 0.09])
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.55)
    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor("#CBD5E1")
        cell.set_linewidth(0.6)
        if row == 0:
            cell.set_facecolor("#173F5F")
            cell.get_text().set_color("white")
            cell.get_text().set_weight("bold")
        else:
            cell.set_facecolor("#F8FAFC" if row % 2 else "#EEF4F8")
            cell.get_text().set_color("#1F2937")
    finite = [value for value in metrics if math.isfinite(value)]
    if finite:
        norm = Normalize(vmin=min(finite), vmax=max(finite) or 1)
        cmap = LinearSegmentedColormap.from_list("rms", ["#D9F0E5", "#FFF2CC", "#F6C7C3"])
        for row, value in enumerate(metrics, start=1):
            if math.isfinite(value):
                table[(row, 4)].set_facecolor(cmap(norm(value)))
                table[(row, 4)].get_text().set_weight("bold")
    figure.suptitle("Validation batch | final export table", y=0.965,
                   fontsize=16, fontweight="bold")
    figure.text(0.03, 0.055,
                "Rows follow the individual export order. Path error uses recorded tracking and the nearest point on the full planned polyline; "
                "P95 is nearest-rank and all values are sample-weighted. Reconstructed tip feedback is not an external accuracy measurement.",
                fontsize=9, color="#475569")
    return figure


def replot_batch(input_directory, output_directory, run_id=None):
    root, output = Path(input_directory).resolve(), Path(output_directory).resolve()
    if output == root:
        raise ValueError("Use a separate output directory to preserve source figures")
    sources = ([root] if (root / "experiment_metadata.json").is_file() else
               sorted(p.parent for p in root.glob("*/experiment_metadata.json")))
    if not sources:
        raise ValueError("No experiment_metadata.json exports found")
    datasets = [load_experiment(source, run_id) for source in sources]
    ordered = sorted(zip(sources, datasets),
                     key=lambda item: (item[1].run.get("started_unix", item[1].t0), item[0].name))
    sources, datasets = zip(*ordered)
    # Reject source overwrite and refuse reuse of an existing report directory.
    if any(source == output or output in source.parents for source in sources):
        raise ValueError("Output must not contain or equal an input experiment")
    output.mkdir(parents=True, exist_ok=False)
    summaries, manifest = [], []
    export_order = []
    for export_index, (source, data) in enumerate(zip(sources, datasets), start=1):
        destination = output / source.name
        destination.mkdir()
        parts = destination / "parts"
        parts.mkdir()
        exported_files = []

        # Export each run's visual components first.  The numbered panel PNGs
        # make every (a), (b) ... subplot directly reusable; the composite
        # names at the run root keep existing links and UI exports compatible.
        figure = build_figure(data)
        try:
            root_file = destination / "trajectory.png"
            save_figure(figure, root_file)
            panel_paths = save_panel_set(figure, parts, TRAJECTORY_PANEL_NAMES)
            exported_files.append(str(root_file.relative_to(output)))
            exported_files.extend(str(path.relative_to(output)) for path in panel_paths)
        finally:
            figure.clear()
        if len(data.plan) >= 2 and len(data.tracking):
            detail = build_deviation_figure(data)
            try:
                root_file = destination / "trajectory_deviation.png"
                save_figure(detail, root_file)
                panel_paths = save_panel_set(detail, parts, DEVIATION_PANEL_NAMES)
                exported_files.append(str(root_file.relative_to(output)))
                exported_files.extend(str(path.relative_to(output)) for path in panel_paths)
            finally:
                detail.clear()
            nearest, residual, segments = data.deviations
            with (destination / "geometric_deviation.csv").open("w", newline="", encoding="utf-8") as stream:
                writer = csv.writer(stream)
                writer.writerow(("unix_time", "time_s", "phase", "segment_index", "nearest_x_mm",
                                 "nearest_y_mm", "nearest_z_mm", "e_x_um", "e_y_um", "e_z_um", "norm_um"))
                for i, row in enumerate(data.tracking):
                    writer.writerow((row[0], row[0]-data.t0, data.phases[i], int(segments[i]),
                                     *nearest[i], *residual[i], np.linalg.norm(residual[i])))
        summary = data.summary()
        summaries.append(summary)
        (destination / "export_manifest.json").write_text(
            json.dumps(dict(order=export_index, experiment=summary["experiment"],
                            files=exported_files), ensure_ascii=False, indent=2),
            encoding="utf-8")
        export_order.append(dict(order=export_index, experiment=summary["experiment"],
                                 run_id=summary["run_id"], directory=source.name,
                                 files=exported_files))
        for path in data.source_files:
            manifest.append(dict(file=str(path.relative_to(root)),
                                 sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    for name, builder in (("overview", build_overview), ("metrics", build_metrics_comparison)):
        overview = builder(datasets)
        try:
            save_figure(overview, output / (name + ".png"))
        finally:
            overview.clear()
    # The final artifact is a single table generated only after all individual
    # experiment components and comparison plots have been saved.
    total_table = build_total_table(datasets)
    try:
        save_figure(total_table, output / "total_table.png")
    finally:
        total_table.clear()
    export_order.append(dict(order="final", artifact="total_table",
                             files=["total_table.png"]))
    (output / "export_order.json").write_text(
        json.dumps(export_order, ensure_ascii=False, indent=2), encoding="utf-8")
    (output / "summary.json").write_text(json.dumps(summaries, ensure_ascii=False, indent=2,
                                                   allow_nan=False), encoding="utf-8")
    with (output / "summary.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(("experiment", "run_id", "feedback_count", "tracking_count", "duration_s",
                         "path_rms_um", "path_mean_um", "path_p95_um", "path_max_um", "target_rms_um"))
        for row in summaries:
            path, target = row["path_error"] or {}, row["target_error"] or {}
            writer.writerow((row["experiment"], row["run_id"], row["feedback_count"], row["tracking_count"],
                             row["duration_s"], *(path.get(k) for k in
                             ("rms_um", "mean_um", "p95_um", "max_um")), target.get("rms_um")))
    with (output / "phase_metrics.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(("experiment", "run_id", "phase", "count", "mean_um", "rms_um", "p95_um", "max_um"))
        for row in summaries:
            for phase, metric in row["phases"].items():
                writer.writerow((row["experiment"], row["run_id"], phase,
                                 *(metric[k] for k in ("count", "mean_um", "rms_um", "p95_um", "max_um"))))
    (output / "provenance.json").write_text(json.dumps(dict(
        schema_version=1, inputs=manifest,
        plotting_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        numpy_version=np.__version__, matplotlib_version=__import__("matplotlib").__version__,
        statistics="Recorded tracking; nearest-rank P95; no smoothing; sample-weighted",
        coordinates="RCM-frame mm minus planned start, displayed in um"), indent=2), encoding="utf-8")
    report = ["# 实验重绘报告", "", "导出顺序：先完成 12 项实验的各面板 PNG，再导出整组对比和指标图，最后生成 `total_table.png` 总表。", "",
              "![整组轨迹对比](overview.png)", "",
              "![误差指标对比](metrics.png)", "",
              "![最终总表](total_table.png)", "",
              "原始 CSV/JSON 与旧图未修改。每项实验提供 trajectory.png 全程图、trajectory_deviation.png 偏差细节图，以及 `parts/` 下对应 (a)(b)… 面板 PNG。",
              "**微小偏差请查看细节图**：左列有放大窗口定位和真实等比例局部图，右列 XYZ 偏差各用独立纵轴。",
              "细节图优先选择正式 contour，其次 outward，其余为全程；全部峰值仍保留在全程图中。",
              "几何使用全量反馈；误差沿用 tracking，按样本加权，P95 为最近秩。",
              "坐标平移至规划首点，单位 µm。针尖由反馈重建，不能据此推断外部定位精度。", "",
              "| 实验 | 反馈 / tracking | RMS (µm) | P95 (µm) | 最大 (µm) | 全程图 | 偏差细节 |",
              "|---|---:|---:|---:|---:|---|---|"]
    for source, row, data in zip(sources, summaries, datasets):
        metric = row["path_error"]
        values = ["%.3f" % metric[k] if metric else "—" for k in ("rms_um", "p95_um", "max_um")]
        detail_link = ("[PNG](%s/trajectory_deviation.png) · [面板](%s/parts/)" %
                       (source.name, source.name) if len(data.plan) >= 2 and len(data.tracking) else "无记录")
        report.append("| %s | %d / %d | %s | %s | %s | [PNG](%s/trajectory.png) · [面板](%s/parts/) | %s |" %
                      (row["experiment"], row["feedback_count"], row["tracking_count"], *values,
                       source.name, source.name, detail_link))
    report.extend(["", "阶段明细见 phase_metrics.csv；完整指标见 summary.json；导出次序见 export_order.json；输入与绘图源码哈希见 provenance.json。所有本轮生成图均为 PNG。",
                   "路径误差为到整条规划折线的最短距离，不能判断往返方向或与时间有关的滞后；",
                   "当前目标路点残差在目标切换时自然增大，不等于轮廓误差。进场/返回计入总指标，",
                   "正式轮廓指标应查看 contour 行。时间为本机回调接收时间，没有设备时间戳。",
                   "各实验 geometric_deviation.csv 保存最近规划点和 XYZ 偏差，单位见表头；",
                   "最近点投影到整条实际执行折线，端点夹紧，重复段按点处理，等距时选首段。",
                   "局部图在标明阶段内选择平面内偏差最大处，横纵轴同时等比例放大；各 XYZ 纵轴独立适配且保留所选阶段全部极值。"])
    (output / "README.md").write_text("\n".join(report)+"\n", encoding="utf-8")
    return output


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", help="One exported experiment directory or a batch root")
    parser.add_argument("output", help="New report directory; original data and figures remain intact")
    parser.add_argument("--run-id", help="Required for a multi-run export")
    args = parser.parse_args(argv)
    try:
        output = replot_batch(args.input, args.output, args.run_id)
    except (OSError, ValueError, TypeError, KeyError) as error:
        parser.exit(1, "Replot failed: %s\n" % error)
    print("Replotted experiments: " + str(output))


if __name__ == "__main__":
    main()
