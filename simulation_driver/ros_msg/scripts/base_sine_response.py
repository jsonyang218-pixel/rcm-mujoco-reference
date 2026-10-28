#!/usr/bin/env python3
"""Record base feedback, or execute the same guarded sine engine as the UI.

Read-only by default. --execute --confirm-safe enables bounded excitation.
No automatic reset or return is performed after a fault.
"""

import argparse
from collections import deque
import json
from pathlib import Path
import signal
import threading
import time

import rospy
from ros_msg.msg import Encoder, controlSignal
from robot_control.data_model import RobotData
from robot_control.experiment_io import software_fingerprints
from robot_control.kinematics import make_control_message
from robot_control.profiles import LEGACY_NEEDLE_PROFILE, NEW_NEEDLE_PROFILE
from robot_control.response_experiments import ResponseControl, response_state_key, stationary_protocol_origin
from robot_control.response_signals import WireSineSignal
from robot_control.wire_sine import WireSineExperiment
from robot_control.wire_sine_reporting import export_wire_sine, export_wire_figures


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    parser.add_argument('--feedback-topic', default='/robot/RobotState')
    parser.add_argument('--control-topic', default='/robot/controlSignal',
                        help='command topic; use /simulation/controlSignal to isolate the real robot')
    parser.add_argument('--simulation-only', action='store_true',
                        help='refuse the real robot control topic; requires a /simulation/ topic')
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--confirm-safe', action='store_true')
    parser.add_argument('--amplitudes', nargs=3, type=int, default=(3, 3, 8), metavar=('A', 'B', 'C'))
    parser.add_argument('--frequency', type=float, choices=(.01, .02, .05), default=.01)
    parser.add_argument('--cycles', type=int, choices=(1, 4), default=4)
    parser.add_argument('--baseline-s', type=float, default=30.)
    parser.add_argument('--needle', choices=('legacy', 'new'), default='new')
    parser.add_argument('--repeat', type=int, choices=(1, 2, 3), default=1)
    parser.add_argument('--data-source', choices=('robot', 'simulation'), default='robot',
                        help='report label only; does not alter control or protections')
    args = parser.parse_args(rospy.myargv()[1:])
    if args.execute and not args.confirm_safe:
        parser.error('--execute requires --confirm-safe')
    if not 2 <= args.baseline_s <= 60:
        parser.error('baseline-s must be 2..60')
    if args.simulation_only and not args.control_topic.startswith('/simulation/'):
        parser.error('--simulation-only requires --control-topic under /simulation/')
    return args


def run(args):
    root = Path(args.output).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=False)
    stream = (root / 'events.jsonl').open('w', buffering=1)
    lock = threading.RLock()
    data, history, engine = RobotData(10000), deque(maxlen=4096), [None]
    interrupted = threading.Event()
    signal.signal(signal.SIGINT, lambda *_: interrupted.set())
    signal.signal(signal.SIGTERM, lambda *_: interrupted.set())
    def log(kind, **values):
        stream.write(json.dumps(dict(event=kind, **values), allow_nan=False) + '\n')
    def observer(state, unix, mono, sequence):
        with lock:
            history.append((mono, response_state_key(state)))
            log('feedback', state=state, unix_time=unix, mono_time=mono, sequence=sequence)
            if engine[0] is not None:
                engine[0].observe(state, unix, mono, sequence)
    data.state_observer = observer
    rospy.init_node('base_sine_response', anonymous=True, disable_signals=True)
    subscriber = rospy.Subscriber(args.feedback_topic, Encoder, data.state_callback, queue_size=100)
    publisher, held, result = None, False, dict(execute=args.execute)
    def exclusive():
        code, message, system = rospy.get_master().getSystemState()
        if code != 1:
            raise RuntimeError(message)
        others = set(dict(system[0]).get(args.control_topic, [])) - {rospy.get_name()}
        if others:
            raise RuntimeError('other_control_publishers: ' + repr(sorted(others)))
    def fresh():
        snapshot = data.snapshot_without_points()
        received = snapshot['last_state_monotonic']
        if received is None or time.monotonic() - received > .5:
            raise RuntimeError('feedback_timeout')
        return snapshot
    def hold():
        nonlocal held
        run = engine[0]
        if publisher is not None and run is not None and run.commands and not held:
            held = True
            now = time.monotonic()
            decision = dict(elapsed_s=now - run.started_mono, mono_time=now,
                            phase='stop', action='hold', handles=run.handles, delta=(0, 0, 0, 0))
            try:
                publisher.publish(make_control_message(run.handles, 'hold'))
                run.published(decision, now, time.time())
                log('hold', **run.commands[-1])
            except Exception as error:
                run.published(decision, now, time.time(), success=False)
                run.status, run.reason = 'stopped', 'hold_publish_failed: ' + str(error)
    try:
        deadline = time.monotonic() + 8
        while data.snapshot_without_points()['state'] is None:
            if interrupted.is_set() or time.monotonic() >= deadline:
                raise RuntimeError('no_feedback_or_interrupted')
            time.sleep(.01)
        exclusive()
        baseline_start = time.monotonic()
        last_print = baseline_start
        while time.monotonic() - baseline_start < args.baseline_s:
            fresh()
            if interrupted.is_set():
                raise RuntimeError('operator_stop')
            if time.monotonic() - last_print > 5:
                exclusive()
                print('Read-only baseline %.1f / %.1f s' % (time.monotonic() - baseline_start, args.baseline_s), flush=True)
                last_print = time.monotonic()
            time.sleep(.01)
        snapshot = fresh()
        exclusive()
        with lock:
            origin = stationary_protocol_origin(list(history), snapshot['state'], time.monotonic(), .5)
        result.update(initial_state=snapshot['state'], protocol_origin=origin)
        if args.execute:
            profile = NEW_NEEDLE_PROFILE if args.needle == 'new' else LEGACY_NEEDLE_PROFILE
            waveform = WireSineSignal(frequency_hz=args.frequency, cycles=args.cycles)
            # Construct/preflight before registering any control publisher.
            run = WireSineExperiment(waveform, snapshot['state'], origin, time.monotonic(), time.time(),
                                     amplitudes=tuple(args.amplitudes), tip_offset_mm=profile.tip_offset_mm,
                                     offset_sign=profile.offset_sign, control=ResponseControl(state_timeout_s=.5),
                                     metadata=dict(repeat=args.repeat, needle_profile=profile.name,
                                                   software=software_fingerprints(), data_source=args.data_source,
                                                   safety_basis='user authorized live diagnosis and stated safe state',
                                                   protocol_origin_source='stationary session echo frozen once',
                                                   feedback_topic=args.feedback_topic,
                                                   control_topic=args.control_topic,
                                                   simulation_only=args.simulation_only))
            publisher = rospy.Publisher(args.control_topic, controlSignal, queue_size=1)
            deadline = time.monotonic() + 3
            while not publisher.get_num_connections():
                fresh()
                if interrupted.is_set() or time.monotonic() >= deadline:
                    raise RuntimeError('no_control_subscriber_or_interrupted')
                time.sleep(.01)
            exclusive()
            # Restart the time origin only before attaching the observer;
            # connection setup must not consume the baseline or hide a gap.
            with lock:
                current = fresh()
                stationary_protocol_origin(list(history), current['state'], time.monotonic(), .5)
                if current['state'] != snapshot['state']:
                    raise RuntimeError('startup_state_changed')
                run.started_mono, run.started_unix = time.monotonic(), time.time()
                run.progress_time = [run.started_mono] * 3
                engine[0] = run
            next_exclusive, next_print = 0., 0.
            while run.status == 'running':
                now = time.monotonic()
                if interrupted.is_set() or rospy.is_shutdown():
                    run.stop('operator_stop', now)
                    break
                if now >= next_exclusive:
                    exclusive()
                    next_exclusive = time.monotonic() + .25
                now = time.monotonic()
                if run.latest is None and now - run.started_mono < .5:
                    time.sleep(.005)
                    continue
                decision = run.tick(now)
                if decision and decision['action'] == 'send':
                    with run.lock:
                        if not run.validate_dispatch(decision, time.monotonic()):
                            break
                        mono, unix = time.monotonic(), time.time()
                        try:
                            publisher.publish(make_control_message(decision['handles'], 'hold'))
                        except Exception:
                            run.published(decision, mono, unix, success=False)
                            raise
                        run.published(decision, mono, unix)
                        log('publish', **run.commands[-1])
                if now >= next_print:
                    print('Sine %.1f / %.1f s | handles %s | %d feedback | %s' % (
                        now - run.started_mono, waveform.duration_s, run.handles, len(run.feedback), run.status), flush=True)
                    next_print = now + 10
                time.sleep(.003)
            result.update(status=run.status, reason=run.reason)
            hold()
        else:
            result.update(status='read_only_finished', reason='no_control_publisher_created')
    except (Exception, KeyboardInterrupt) as error:
        result.update(status='stopped', reason=str(error) or type(error).__name__)
        if engine[0] is not None:
            engine[0].stop(result['reason'], time.monotonic())
    finally:
        hold()
        # Keep raw feedback after stop to expose late motion; never auto-return.
        if publisher is not None:
            end = time.monotonic() + 2
            while time.monotonic() < end and not rospy.is_shutdown():
                time.sleep(.01)
            publisher.unregister()
        subscriber.unregister()
        if engine[0] is not None:
            result.update(status=engine[0].status, reason=engine[0].reason)
            exported = engine[0].snapshot()
            try:
                export_wire_sine(exported, root / 'result')
                export_wire_figures(exported, root / 'result')
            except Exception as error:
                result['export_error'] = str(error)
        result.update(final_state=data.snapshot_without_points()['state'],
                      feedback_records=data.snapshot_without_points()['state_sequence'])
        (root / 'result.json').write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
        with lock:
            stream.close()
        rospy.signal_shutdown('experiment ended')
    print(json.dumps(result, ensure_ascii=False), flush=True)
    return 0 if result.get('status') in ('completed', 'read_only_finished') else 2


if __name__ == '__main__':
    raise SystemExit(run(arguments()))
