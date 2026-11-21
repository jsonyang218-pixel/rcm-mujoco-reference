"""Bounded cumulative-handle sine excitation; feedback never rewrites input."""

from dataclasses import asdict
import math
import threading

from .kinematics import (
    AXES, model_tip_from_robot_encoders, needle_pose_from_robot_encoders,
    stepped_values, validate_cartesian_delta,
    validate_manual_jog,
)
from .joint_response import joint_coordinates
from .response_experiments import ResponseControl


class WireSineExperiment:
    mode = "base_wire_sine"

    def __init__(self, signal, state, handles, started_mono, started_unix,
                 amplitudes=(3, 3, 8), tip_offset_mm=4., offset_sign=1.,
                 control=None, metadata=None):
        if signal.kind != "sine":
            raise ValueError("基点控制信号实验仅接受正弦")
        self.amplitudes = tuple(amplitudes)
        if (len(self.amplitudes) != 3 or
                any(not isinstance(v, int) or isinstance(v, bool) for v in self.amplitudes) or
                any(abs(v) > limit for v, limit in zip(self.amplitudes, (6, 6, 17))) or
                not any(self.amplitudes)):
            raise ValueError("手柄幅值需为整数，A/B ≤6、C ≤17 计数，至少一轴非零")
        if not (math.isfinite(tip_offset_mm) and tip_offset_mm >= 0 and offset_sign in (-1, 1)):
            raise ValueError("针型几何无效")
        self.signal, self.control = signal, control or ResponseControl()
        self.lock = threading.RLock()
        self.started_mono, self.started_unix = started_mono, started_unix
        self.tip_offset_mm, self.offset_sign = tip_offset_mm, offset_sign
        self.origin_handles = self.handles = tuple(stepped_values(handles, AXES, (0, 0, 0, 0)))
        self.start_robot = tuple(state['robot'])
        self.start_position = self.start_junction = tuple(state['position'])
        if len(self.start_junction) != 3 or not all(map(math.isfinite, self.start_junction)):
            raise ValueError("基点坐标无效")
        if state['forward'] or state['backward']:
            raise ValueError("启动时方向标志必须停止")
        self.start_tip = self.tip(self.start_robot, self.start_junction)
        self.q0 = joint_coordinates(self.start_robot)
        self.metadata = dict(metadata or {})
        self.feedback, self.decisions, self.commands = [], [], []
        self.status, self.reason, self.fault = 'running', '', None
        self.latest, self.last_sequence, self.last_tick = None, None, None
        self.stopped_mono = None
        self.final_stable, self.settle_start, self.settle_count = False, None, 0
        self.progress_q = list(self.q0)
        self.progress_time = [started_mono] * 3
        self.input_travel = [0] * 3
        self.preflight()

    def tip(self, robot, position):
        return tuple(needle_pose_from_robot_encoders(
            robot, position, self.tip_offset_mm, self.offset_sign)['display_tip'])

    def reference_at(self, elapsed):
        scalar, phase = self.signal.at(elapsed)
        ideal = tuple(a * scalar / self.signal.amplitude_um for a in self.amplitudes)
        offsets = tuple(round(v) for v in ideal)
        target = tuple(stepped_values(self.origin_handles, AXES, offsets + (0,)))
        return dict(phase=phase, ideal_handle_offset=ideal, planned_handles=target,
                    reference_handle_offset=offsets)

    def preflight(self):
        validate_manual_jog(self.start_robot, (0, 0, 0, 0))
        self.tip_envelope_um = self.junction_envelope_um = 0.
        self.native_bounds = [[0., 0.] for _ in range(3)]
        # Include every integer transition on the common scalar and both sides
        # of ties, so coarse quantisation cannot evade the full-range preflight.
        fractions = {-1., 0., 1.}
        for amplitude in self.amplitudes:
            for k in range(-abs(amplitude), abs(amplitude)):
                boundary = (k + .5) / abs(amplitude)
                fractions.update((boundary, boundary - 1e-12, boundary + 1e-12))
        for fraction in sorted(fractions):
            delta = tuple(round(a * fraction) for a in self.amplitudes) + (0,)
            stepped_values(self.origin_handles, AXES, delta)
            robot = validate_cartesian_delta(
                self.start_robot, delta, self.tip_offset_mm, self.offset_sign,
                max_step_mm=self.control.correction_limit_um / 1000)
            q = joint_coordinates(robot)
            for i in range(3):
                self.native_bounds[i][0] = min(self.native_bounds[i][0], q[i] - self.q0[i])
                self.native_bounds[i][1] = max(self.native_bounds[i][1], q[i] - self.q0[i])
            for offset, name in ((0, 'junction_envelope_um'), (self.tip_offset_mm, 'tip_envelope_um')):
                distance = math.dist(model_tip_from_robot_encoders(robot, offset, self.offset_sign),
                                     model_tip_from_robot_encoders(self.start_robot, offset, self.offset_sign)) * 1000
                setattr(self, name, max(getattr(self, name), distance))

    def observe(self, state, unix_time, mono_time, sequence):
        with self.lock:
            if self.status != 'running' or mono_time < self.started_mono:
                return
            if self.latest and sequence <= self.latest['sequence']:
                return
            try:
                robot, position = tuple(state['robot']), tuple(state['position'])
                validate_manual_jog(robot, (0, 0, 0, 0))
                if len(position) != 3 or not all(map(math.isfinite, position)):
                    raise ValueError('invalid_base_position')
                stepped_values(state['handle'], AXES, (0, 0, 0, 0))
                tip = self.tip(robot, position)
                q = joint_coordinates(robot)
                elapsed = mono_time - self.started_mono
                reference = self.reference_at(elapsed)
                row = dict(elapsed_s=elapsed, mono_time=mono_time, unix_time=unix_time,
                           sequence=sequence, **reference, robot=robot, handle=tuple(state['handle']),
                           junction_mm=position, tip_mm=tip,
                           actual_xyz_um=tuple((v - o) * 1000 for v, o in zip(position, self.start_junction)),
                           joint_actual=tuple(v - o for v, o in zip(q, self.q0)),
                           forward=bool(state['forward']), backward=bool(state['backward']))
                if len(self.feedback) >= self.control.max_records:
                    raise ValueError('record_capacity_exceeded')
                if self.latest and mono_time - self.latest['mono_time'] > self.control.state_timeout_s:
                    self.fault = 'feedback_gap_exceeded'
                if state['forward'] or state['backward']:
                    self.fault = self.fault or 'unexpected_direction_flag'
                if robot[3] != self.start_robot[3]:
                    self.fault = self.fault or 'unexpected_e_motion'
                for i, margin in enumerate((.05, .05, self.control.envelope_margin_um)):
                    if not self.native_bounds[i][0] - margin <= q[i] - self.q0[i] <= self.native_bounds[i][1] + margin:
                        self.fault = self.fault or 'joint_' + AXES[i] + '_envelope'
                    if self.latest and abs(row['joint_actual'][i] - self.latest['joint_actual'][i]) > (.03, .03, 5.)[i]:
                        self.fault = self.fault or 'joint_' + AXES[i] + '_jump'
                    threshold = (.0002, .0002, self.control.movement_um)[i]
                    if abs(q[i] - self.progress_q[i]) >= threshold:
                        self.progress_q[i], self.progress_time[i], self.input_travel[i] = q[i], mono_time, 0
                for actual, origin, envelope, name in (
                        (position, self.start_junction, self.junction_envelope_um, 'base_envelope'),
                        (tip, self.start_tip, self.tip_envelope_um, 'needle_tip_envelope')):
                    if math.dist(actual, origin) * 1000 > envelope + self.control.envelope_margin_um:
                        self.fault = self.fault or name
                stable = (reference['phase'] == 'recovery' and
                          math.dist(position, self.start_junction) * 1000 <= 1 and
                          all(abs(v) <= tol for v, tol in zip(row['joint_actual'], (.01, .005, 1.))))
                if stable:
                    if self.settle_start is None:
                        self.settle_start = mono_time
                    self.settle_count += 1
                    self.final_stable = self.settle_count >= 3 and mono_time - self.settle_start >= .25
                else:
                    self.settle_start, self.settle_count, self.final_stable = None, 0, False
                self.feedback.append(row)
                self.latest = row
            except (ValueError, TypeError, OverflowError) as error:
                self.fault = 'invalid_feedback: ' + str(error)

    def stop(self, reason, now, status='stopped'):
        with self.lock:
            if self.status == 'running':
                self.status, self.reason, self.stopped_mono = status, reason, now

    def tick(self, now):
        with self.lock:
            if self.status != 'running':
                return None
            if self.fault:
                self.stop(self.fault, now)
                return None
            if self.latest is None or now - self.latest['mono_time'] > self.control.state_timeout_s:
                self.stop('feedback_timeout', now)
                return None
            elapsed = now - self.started_mono
            if elapsed >= self.signal.duration_s:
                self.stop('schedule_finished' if self.final_stable else 'schedule_finished_unsettled',
                          now, 'completed' if self.final_stable else 'unsettled')
                return None
            if self.last_tick is not None and now - self.last_tick < self.control.interval_s - 1e-9:
                return None
            if self.last_sequence == self.latest['sequence']:
                return None
            self.last_tick, self.last_sequence = now, self.latest['sequence']
            reference = self.reference_at(elapsed)
            handles = reference['planned_handles']
            delta = tuple(a - b for a, b in zip(handles, self.handles))
            # Count meaningful cumulative input travel since real feedback
            # progress, including across quantisation plateaus and reversals.
            # One complete out-and-back input excursion without progress stops.
            for i, amplitude in enumerate(self.amplitudes):
                if (amplitude and self.input_travel[i] >= max(4, 2 * abs(amplitude)) and
                        now - self.progress_time[i] >= self.control.no_progress_s):
                    self.stop('no_physical_progress_' + AXES[i], now)
                    return None
            try:
                if any(delta):
                    validate_cartesian_delta(self.latest['robot'], delta, self.tip_offset_mm,
                                             self.offset_sign, max_step_mm=self.control.correction_limit_um / 1000)
                decision = dict(elapsed_s=elapsed, mono_time=now, **reference,
                                feedback_sequence=self.latest['sequence'],
                                feedback_age_s=now - self.latest['mono_time'],
                                handles=handles, delta=delta, action='send' if any(delta) else 'quantized_hold')
                if len(self.decisions) >= self.control.max_records:
                    raise ValueError('record_capacity_exceeded')
                self.decisions.append(decision)
                return decision
            except (ValueError, TypeError, OverflowError) as error:
                self.stop('invalid_command: ' + str(error), now)
                return None

    def published(self, decision, mono_time, unix_time, success=True):
        with self.lock:
            if success:
                for i in range(3):
                    self.input_travel[i] += abs(decision['delta'][i])
                self.handles = tuple(decision['handles'])
            else:
                self.stop('publish_failed', mono_time)
            if len(self.commands) >= self.control.max_records:
                self.stop('record_capacity_exceeded', mono_time)
                return
            self.commands.append(dict(decision, publish_mono=mono_time,
                                      publish_unix=unix_time, publish_elapsed_s=mono_time - self.started_mono,
                                      publish_success=success))

    def validate_dispatch(self, decision, now):
        """Call under lock immediately before publishing; callback faults win."""
        with self.lock:
            if self.status != 'running':
                return False
            if self.fault:
                self.stop(self.fault, now)
                return False
            if self.latest is None or now - self.latest['mono_time'] > self.control.state_timeout_s:
                self.stop('feedback_timeout', now)
                return False
            if tuple(a - b for a, b in zip(decision['handles'], self.handles)) != tuple(decision['delta']):
                self.stop('stale_command_origin', now)
                return False
            try:
                validate_cartesian_delta(self.latest['robot'], decision['delta'], self.tip_offset_mm,
                                         self.offset_sign, max_step_mm=self.control.correction_limit_um / 1000)
            except (ValueError, TypeError, OverflowError) as error:
                self.stop('invalid_command: ' + str(error), now)
                return False
            return True

    def snapshot(self):
        with self.lock:
            return dict(schema_version=2, mode=self.mode, control_object='base',
                        signal=asdict(self.signal), control=asdict(self.control),
                        amplitudes_counts=self.amplitudes, origin_handles=self.origin_handles,
                        origin_robot=self.start_robot, origin_junction_mm=self.start_junction,
                        origin_tip_mm=self.start_tip, tip_offset_mm=self.tip_offset_mm,
                        offset_sign=self.offset_sign, started_mono=self.started_mono,
                        started_unix=self.started_unix, stopped_mono=self.stopped_mono,
                        status=self.status, reason=self.reason, metadata=self.metadata,
                        final_stable=self.final_stable,
                        junction_envelope_um=self.junction_envelope_um, tip_envelope_um=self.tip_envelope_um,
                        native_bounds=self.native_bounds,
                        feedback=list(self.feedback), decisions=list(self.decisions), commands=list(self.commands))
