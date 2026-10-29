#!/usr/bin/env python3
"""Read-only by default; prepare a workpoint or run a base XYZ time response."""

import argparse
from collections import deque
import json
import math
from pathlib import Path
import signal
import threading
import time

import rospy
from ros_msg.msg import Encoder, controlSignal
from robot_control.base_position import (
    BasePositionExperiment, stationary_base_protocol_origin,
)
from robot_control.data_model import RobotData
from robot_control.experiment_io import software_fingerprints
from robot_control.kinematics import (
    ENCODER_COUNTS_PER_MM,
    C_HANDLE_COUNTS_PER_MM,
    make_control_message,
    model_tip_from_robot_encoders,
    needle_direction_from_robot_encoders,
)
from robot_control.joint_response import joint_coordinates
from robot_control.response_experiments import (
    ResponseControl, ResponseExperiment,
    response_state_key,
)
from robot_control.response_signals import ResponseSignal
from robot_control.response_reporting import export_response, export_response_figures


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--feedback-topic", default="/robot/RobotState")
    parser.add_argument(
        "--control-topic", default="/robot/controlSignal",
        help="command topic; use /simulation/controlSignal to isolate the real robot",
    )
    parser.add_argument(
        "--simulation-only", action="store_true",
        help="refuse the real robot control topic; requires a /simulation/ topic",
    )
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--confirm-safe", action="store_true")
    parser.add_argument("--amplitudes", nargs=3, type=float, default=(10.0, 6.0, 4.0))
    parser.add_argument(
        "--kind", choices=("noise", "step", "ramp", "sine", "pulse"), default="sine",
        help="time-response signal; noise is read-only, other signals use base XYZ residual closure",
    )
    parser.add_argument("--amplitude", type=float, choices=(10.0, 20.0), default=10.0)
    parser.add_argument(
        "--frequency", type=float, choices=(0.01, 0.02, 0.05), default=0.01
    )
    parser.add_argument("--cycles", type=int, choices=(1, 4), default=1)
    parser.add_argument("--baseline-s", type=float, default=30.0)
    parser.add_argument("--hold-s", type=float, default=15.0)
    parser.add_argument("--speed", type=float, choices=(1.0, 2.0, 5.0), default=1.0)
    parser.add_argument("--pulse-width", type=float, choices=(2.0, 5.0, 10.0), default=10.0)
    parser.add_argument(
        "--prepare-axial-um",
        type=float,
        default=0.0,
        help="relative axial workpoint preparation; signed, absolute value at most 500 um",
    )
    parser.add_argument("--repeat", type=int, choices=(1, 2, 3), default=1)
    parser.add_argument(
        "--data-source", choices=("robot", "simulation"), default="robot"
    )
    args = parser.parse_args(rospy.myargv()[1:])
    if args.execute and not args.confirm_safe:
        parser.error("--execute requires --confirm-safe")
    if (not 2 <= args.baseline_s <= 60 or not -500 <= args.prepare_axial_um <= 500
            or not 0.25 <= args.hold_s <= 60):
        parser.error("baseline 2..60 s; hold 0.25..60 s; preparation -500..500 um")
    if args.kind == "noise" and args.execute:
        parser.error("noise is read-only")
    if args.simulation_only and not args.control_topic.startswith("/simulation/"):
        parser.error("--simulation-only requires --control-topic under /simulation/")
    return args


def run(args):
    root = Path(args.output).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=False)
    stream = (root / "events.jsonl").open("w", buffering=1)
    lock = threading.RLock()
    data, history, engine = RobotData(20000), deque(maxlen=4096), [None]
    interrupted = threading.Event()
    signal.signal(signal.SIGINT, lambda *_: interrupted.set())
    signal.signal(signal.SIGTERM, lambda *_: interrupted.set())
    publisher, hold_sent = [None], [True]
    stages, initial, preparation = [], [None], abs(args.prepare_axial_um) > 1e-12

    def log(kind, **values):
        stream.write(json.dumps(dict(event=kind, **values), allow_nan=False) + "\n")

    def observer(state, unix, mono, sequence):
        with lock:
            history.append((mono, response_state_key(state)))
            log(
                "feedback",
                state=state,
                unix_time=unix,
                mono_time=mono,
                sequence=sequence,
            )
            if engine[0] is not None:
                engine[0].observe(state, unix, mono, sequence)
                if (
                    preparation
                    and initial[0] is not None
                    and engine[0].status == "running"
                ):
                    q, q0 = joint_coordinates(state["robot"]), joint_coordinates(
                        initial[0]["robot"]
                    )
                    if math.dist(
                        state["position"], initial[0]["position"]
                    ) * 1000 > abs(args.prepare_axial_um) + 20 or any(
                        abs(q[i] - q0[i]) > 0.02 for i in (0, 1)
                    ):
                        engine[0].fault = (
                            engine[0].fault or "preparation_global_envelope"
                        )

    data.state_observer = observer
    rospy.init_node("base_position_response", anonymous=True, disable_signals=True)
    subscriber = rospy.Subscriber(
        args.feedback_topic, Encoder, data.state_callback, queue_size=100
    )
    control = ResponseControl(state_timeout_s=0.5)
    metadata = dict(
        repeat=args.repeat,
        needle_profile="new",
        software=software_fingerprints(),
        data_source=args.data_source,
        safety_basis="user authorized safe-state hardware diagnosis and base XYZ time-response plan",
        protocol_origin_source="stationary session echo frozen once",
        startup_quiet_policy="1 s; base/tip span <=0.25 um; half-mean drift <=0.05 um; exact echo/E/flags",
        feedback_topic=args.feedback_topic,
        control_topic=args.control_topic,
        simulation_only=args.simulation_only,
    )
    result = dict(execute=args.execute, prepare_axial_um=args.prepare_axial_um)

    def fresh():
        snap = data.snapshot_without_points()
        if (
            snap["last_state_monotonic"] is None
            or time.monotonic() - snap["last_state_monotonic"] > 0.5
        ):
            raise RuntimeError("feedback_timeout")
        return snap

    def exclusive():
        code, message, system = rospy.get_master().getSystemState()
        if code != 1:
            raise RuntimeError(message)
        others = set(dict(system[0]).get(args.control_topic, ())) - {
            rospy.get_name()
        }
        if others:
            raise RuntimeError("other_control_publishers:" + repr(sorted(others)))

    def hold():
        r = engine[0]
        if publisher[0] is not None and r is not None and not hold_sent[0]:
            with r.lock:
                now = time.monotonic()
                ref = r.reference_at(now - r.started_mono)
                d = dict(
                    elapsed_s=now - r.started_mono,
                    mono_time=now,
                    **ref,
                    feedback_sequence=r.last_sequence or 0,
                    feedback_age_s=now - r.latest["mono_time"] if r.latest else 0,
                    error_norm_um=r.latest["error_norm_um"] if r.latest else 0,
                    action="hold",
                    delta=(0, 0, 0, 0),
                    handles=r.handles
                )
                publisher[0].publish(make_control_message(r.handles, "hold"))
                r.published(d, now, time.time())
                hold_sent[0] = True
                log("hold", **r.commands[-1])

    def quiet():
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            if interrupted.is_set():
                raise RuntimeError("operator_stop")
            snap = fresh()
            exclusive()
            with lock:
                try:
                    origin = stationary_base_protocol_origin(
                        list(history), snap["state"], time.monotonic(), 0.5
                    )
                except ValueError:
                    origin = None
            if origin is not None:
                return snap["state"], origin
            time.sleep(0.03)
        raise RuntimeError("session_not_stationary")

    def create(wave, state, origin, amplitudes, preparation=False,
               drive_vector_override=None):
        common = dict(
            signal=wave,
            state=state,
            handles=origin,
            started_mono=time.monotonic(),
            started_unix=time.time(),
            tip_offset_mm=4.0,
            offset_sign=-1.0,
            control=control,
            metadata=dict(metadata, protocol_origin=origin),
        )
        if preparation:
            # The RCM is a Cartesian inverse-kinematics singularity.  Leave
            # it with the calibrated axial path before constructing a base XYZ
            # experiment; the normal execute() safety checks still apply.
            return ResponseExperiment(
                mode="c_axis",
                drive_vector_override=drive_vector_override,
                **common
            )
        return BasePositionExperiment(xyz_amplitudes_um=amplitudes, **common)

    def execute(r, move=False):
        # The engine has already preflighted before a publisher is created.
        if publisher[0] is None:
            publisher[0] = rospy.Publisher(
                args.control_topic, controlSignal, queue_size=1
            )
            deadline = time.monotonic() + 3
            while not publisher[0].get_num_connections():
                fresh()
                if interrupted.is_set() or time.monotonic() > deadline:
                    raise RuntimeError("no_subscriber_or_interrupted")
                time.sleep(0.01)
        quiet()
        with lock:
            current = fresh()["state"]
            origin = stationary_base_protocol_origin(
                list(history), current, time.monotonic(), 0.5
            )
            if origin != r.handles:
                raise RuntimeError("startup_protocol_changed")
            # Freeze state and reference origin atomically after connection;
            # then repeat the full preflight before any motion packet.
            r = create(
                r.signal,
                current,
                origin,
                tuple(v * r.signal.amplitude_um for v in r.drive_vector),
                preparation=move,
                drive_vector_override=(r.drive_vector if move else None),
            )
            r.started_mono, r.started_unix = time.monotonic(), time.time()
            r.last_progress = r.started_mono
            r.progress_time = [r.started_mono] * 3
            engine[0] = r
            stages.append(r)
            hold_sent[0] = False
        settle_start, settle_count = None, 0
        next_exclusive, next_print = 0.0, 0.0
        while r.status == "running":
            now = time.monotonic()
            if interrupted.is_set() or rospy.is_shutdown():
                r.stop("operator_stop", now)
                break
            if now >= next_exclusive:
                exclusive()
                next_exclusive = time.monotonic() + 0.25
            if r.latest is None and now - r.started_mono < 0.5:
                time.sleep(0.003)
                continue
            d = r.tick(now)
            if d and d["action"] == "send":
                with r.lock:
                    if not r.validate_dispatch(d, time.monotonic()):
                        break
                    if move and any(d["delta"][i] for i in (0, 1, 3)):
                        r.stop("preparation_requires_c_only", now)
                        break
                    mono, unix = time.monotonic(), time.time()
                    try:
                        publisher[0].publish(make_control_message(d["handles"], "hold"))
                    except Exception:
                        r.published(d, mono, unix, False)
                        raise
                    r.published(d, mono, unix)
                    log("publish", stage=len(stages), **r.commands[-1])
            row = r.latest
            if (
                move
                and row
                and row["sequence"] != getattr(r, "preparation_sequence", None)
            ):
                r.preparation_sequence = row["sequence"]
                if row["phase"] == "plateau" and row["error_norm_um"] <= 1:
                    settle_start = (
                        row["mono_time"] if settle_start is None else settle_start
                    )
                    settle_count += 1
                    if settle_count >= 3 and row["mono_time"] - settle_start >= 0.25:
                        r.stop("preparation_target_reached", now, "completed")
                else:
                    settle_start, settle_count = None, 0
                if now - r.started_mono >= 15:
                    r.stop("preparation_segment_timeout", now)
            if now >= next_print:
                print(
                    "stage %d | %.1f s | %s | error %s um"
                    % (
                        len(stages),
                        now - r.started_mono,
                        r.status,
                        row["error_norm_um"] if row else None,
                    ),
                    flush=True,
                )
                next_print = now + 10
            time.sleep(0.003)
        hold()
        if r.status != "completed":
            raise RuntimeError(r.reason)

    try:
        deadline = time.monotonic() + 8
        while data.snapshot_without_points()["state"] is None:
            if interrupted.is_set() or time.monotonic() > deadline:
                raise RuntimeError("no_feedback_or_interrupted")
            time.sleep(0.01)
        exclusive()
        start = time.monotonic()
        last_check = start
        while time.monotonic() - start < args.baseline_s:
            fresh()
            if interrupted.is_set():
                raise RuntimeError("operator_stop")
            if time.monotonic() - last_check >= 0.25:
                exclusive()
                last_check = time.monotonic()
            time.sleep(0.01)
        state, origin = quiet()
        initial[0] = state
        result.update(initial_state=state, protocol_origin=origin)
        wave = ResponseSignal(
            kind=args.kind,
            amplitude_um=args.amplitude,
            frequency_hz=args.frequency,
            speed_um_s=args.speed,
            pulse_width_s=args.pulse_width,
            hold_s=args.hold_s,
            cycles=args.cycles if args.kind == "sine" else 4,
            pretrial=args.kind == "sine" and args.cycles == 1,
        )
        if preparation:
            direction = needle_direction_from_robot_encoders(state["robot"])
            goal = tuple(
                state["position"][i] + direction[i] * args.prepare_axial_um / 1000
                for i in range(3)
            )
            candidate = dict(state)
            candidate["robot"] = tuple(
                state["robot"][i]
                + (
                    round(args.prepare_axial_um / 1000 * ENCODER_COUNTS_PER_MM)
                    if i == 2
                    else 0
                )
                for i in range(4)
            )
            candidate["position"] = goal
            candidate_handles = tuple(
                origin[i] + (round(args.prepare_axial_um / 1000.0 * C_HANDLE_COUNTS_PER_MM)
                             if i == 2 else 0)
                for i in range(4)
            )
            candidate["handle"] = candidate_handles
            planned = create(
                wave,
                candidate,
                candidate_handles,
                args.amplitudes,
                preparation=True,
                drive_vector_override=tuple(
                    (goal[i] - state["position"][i]) * 1000 / wave.amplitude_um
                    for i in range(3)
                ),
            )
            result.update(
                preflight_candidate_robot=candidate["robot"],
                preflight_base_envelope_um=planned.junction_envelope_um,
                preflight_tip_envelope_um=planned.tip_envelope_um,
                goal_position_mm=goal,
            )
            log("workpoint_preflight", **result)
            if args.execute:
                for index in range(80):
                    state, origin = quiet()
                    remaining = tuple(
                        (goal[i] - state["position"][i]) * 1000 for i in range(3)
                    )
                    distance = math.sqrt(sum(v * v for v in remaining))
                    if distance <= 1:
                        break
                    if distance > abs(args.prepare_axial_um) + 20:
                        raise RuntimeError("preparation_reverse_or_divergence")
                    amplitudes = tuple(v * min(1, 10 / distance) for v in remaining)
                    move_wave = ResponseSignal(
                        kind="step", amplitude_um=10, baseline_s=0.2, hold_s=15, recovery_s=15
                    )
                    execute(
                        create(
                            move_wave,
                            state,
                            origin,
                            amplitudes,
                            preparation=True,
                            drive_vector_override=tuple(v / move_wave.amplitude_um
                                                         for v in amplitudes),
                        ),
                        move=True,
                    )
                else:
                    raise RuntimeError("preparation_segment_budget")
                result.update(
                    status="completed",
                    reason="workpoint_prepared",
                    segments=len(stages),
                )
            else:
                result.update(
                    status="read_only_finished", reason="workpoint_preflight_only"
                )
        elif args.execute:
            execute(create(wave, state, origin, args.amplitudes))
            result.update(status=engine[0].status, reason=engine[0].reason)
        else:
            result.update(
                status="read_only_finished", reason="no_control_publisher_created"
            )
    except (Exception, KeyboardInterrupt) as error:
        result.update(status="stopped", reason=str(error) or type(error).__name__)
        if engine[0] is not None:
            engine[0].stop(result["reason"], time.monotonic())
    finally:
        try:
            hold()
        except Exception as error:
            result.update(status="stopped", reason="hold_publish_failed:" + str(error))
        if publisher[0] is not None:
            end = time.monotonic() + 10
            while time.monotonic() < end and not rospy.is_shutdown():
                time.sleep(0.01)
            publisher[0].unregister()
        subscriber.unregister()
        for index, r in enumerate(stages):
            snap = r.snapshot()
            folder = root / ("segment_%03d" % (index + 1) if preparation else "result")
            try:
                export_response(snap, folder)
                (folder / "snapshot.json").write_text(
                    json.dumps(snap, indent=2, ensure_ascii=False, allow_nan=False)
                    + "\n"
                )
                if not preparation:
                    export_response_figures(snap, folder)
            except Exception as error:
                result["export_error"] = str(error)
        result.update(
            final_state=data.snapshot_without_points()["state"],
            feedback_records=data.snapshot_without_points()["state_sequence"],
        )
        (root / "result.json").write_text(
            json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
        )
        with lock:
            stream.close()
        rospy.signal_shutdown("experiment ended")
    print(json.dumps(result, ensure_ascii=False), flush=True)
    return 0 if result.get("status") in ("completed", "read_only_finished") else 2


if __name__ == "__main__":
    raise SystemExit(run(arguments()))
