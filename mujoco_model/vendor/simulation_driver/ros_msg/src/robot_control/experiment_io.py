"""Export reproducible feedback, commands, plans and run metadata without ROS."""

import csv
import hashlib
import json
import os
import tempfile
from pathlib import Path


def software_fingerprints():
    """Capture source identities once at startup without exposing local paths."""
    folder = Path(__file__).parent
    return {name: hashlib.sha256((folder / name).read_bytes()).hexdigest()
            for name in ("kinematics.py", "trajectory.py", "validation_experiments.py",
                         "data_model.py", "experiment_io.py",
                         "experiment_plotting.py", "response_signals.py",
                         "response_experiments.py", "response_reporting.py",
                         "joint_response.py", "wire_sine.py",
                         "wire_sine_reporting.py", "base_position.py")}


def _csv_file(filename, columns, rows):
    with open(filename, "w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(columns)
        writer.writerows(rows)


def write_plan(filename, route):
    """Save RCM-frame XYZ columns that can also be imported by the planner."""
    _csv_file(filename, ("waypoint_index", "x_mm", "y_mm", "z_mm"),
              ((index, *point) for index, point in enumerate(route)))


def export_experiment(filename, snapshot, runs, run_ids=None):
    """Write a feedback CSV and associated files from one frozen snapshot.

    Raw XYZ and raw reconstructed tip use RobotState's frame. Relative tip
    coordinates only exist when that sample has a registered RCM. Geometry
    belongs to the sample, never to the UI state at export time.
    """
    target = Path(filename)
    stem = target.stem
    paths = {
        "feedback": target,
        "commands": target.with_name(stem + "_commands.csv"),
        "plans": target.with_name(stem + "_plans.csv"),
        "tracking": target.with_name(stem + "_tracking.csv"),
        "metadata": target.with_name(stem + "_metadata.json"),
    }
    selected = None if run_ids is None else set(run_ids)
    records = [record for record in snapshot["records"]
               if selected is None or record.get("context", {}).get("run_id") in selected]
    export_runs = {key: value for key, value in runs.items()
                   if selected is None or key in selected}
    t0 = records[0]["unix_time"] if records else None

    def feedback_rows():
        for record in records:
            context = record.get("context", {})
            rcm = context.get("rcm_mm")
            tip = record["tip"]
            relative_tip = (tuple(tip[i] - rcm[i] for i in range(3))
                            if tip is not None and rcm is not None else (None,) * 3)
            run_id = context.get("run_id", "")
            run = export_runs.get(run_id, {})
            started = run.get("started_unix")
            yield (
                record["unix_time"], record["unix_time"] - t0,
                record["sequence"], run_id,
                record["unix_time"] - started if started is not None else None,
                *record["position"], *(tip or (None,) * 3), *relative_tip,
                *record["robot"], *record["handle"],
                int(record["forward"]), int(record["backward"]),
                *(rcm or (None,) * 3), context.get("tip_offset_mm"),
                context.get("needle_offset_sign"), context.get("needle_profile"),
            )

    # Stage every output first; malformed input cannot replace existing files.
    # Metadata is installed last and describes the entire successful export.
    with tempfile.TemporaryDirectory(prefix=".trajectory-export-", dir=target.parent) as directory:
        staged = {key: Path(directory) / path.name for key, path in paths.items()}
        _csv_file(staged["feedback"], (
            "unix_time", "elapsed_s", "feedback_sequence", "run_id", "run_elapsed_s",
            "junction_x_mm", "junction_y_mm", "junction_z_mm",
            "tip_raw_x_mm", "tip_raw_y_mm", "tip_raw_z_mm",
            "tip_rcm_x_mm", "tip_rcm_y_mm", "tip_rcm_z_mm",
            "robot_A", "robot_B", "robot_C", "robot_E",
            "handle_A", "handle_B", "handle_C", "handle_E",
            "step_forward", "step_backward", "rcm_x_mm", "rcm_y_mm", "rcm_z_mm",
            "tip_offset_mm", "needle_offset_sign", "needle_profile",
        ), feedback_rows())
        _csv_file(staged["commands"], (
            "unix_time", "handle_A", "handle_B", "handle_C", "handle_E",
            "step_forward", "step_backward",
        ), ((record["unix_time"], *record["handle"], int(record["forward"]),
             int(record["backward"])) for record in snapshot["commands"]))
        _csv_file(staged["plans"], (
            "run_id", "waypoint_index", "x_mm", "y_mm", "z_mm",
        ), ((run_id, index, *point) for run_id, run in export_runs.items()
            for index, point in enumerate(run["planned_path"])))
        _csv_file(staged["tracking"], (
            "run_id", "unix_time", "tip_x_mm", "tip_y_mm", "tip_z_mm",
            "target_x_mm", "target_y_mm", "target_z_mm", "path_error_um",
            "target_error_um", "waypoint_index", "phase",
        ), ((run_id, row[0], *row[1], *row[2], row[3] * 1000.0,
             row[4] * 1000.0, row[5],
             (run.get("waypoint_phases", ())[int(row[5])] if
              run.get("waypoint_phases") and 0 <= int(row[5]) < len(run["waypoint_phases"])
              else "")) for run_id, run in export_runs.items()
            for row in run.get("tracking", ())))
        metadata = {
            "schema_version": 2,
            "coordinate_units": "mm",
            "time_basis": "local ROS callback receipt time, Unix seconds; not device timestamps",
            "feedback_count": len(records),
            "dropped_feedback_records": snapshot["dropped_feedback_records"],
            "dropped_command_records": snapshot.get("dropped_command_records", 0),
            "files": {key: path.name for key, path in paths.items()},
            "runs": {key: {name: value for name, value in run.items()
                           if name not in ("tracking", "planned_path")}
                     for key, run in export_runs.items()},
            "accuracy_note": "Needle tip is reconstructed feedback; external accuracy is not inferred.",
        }
        with open(staged["metadata"], "w", encoding="utf-8") as stream:
            json.dump(metadata, stream, ensure_ascii=False, indent=2, allow_nan=False)
        for key, path in paths.items():
            os.replace(staged[key], path)
    return {key: str(path) for key, path in paths.items()}
