"""Offline protocol plant for the uploaded controller; no sockets or ROS calls.

Handle fields are cumulative absolute targets, not motor encoders or per-packet
increments. Targets are anchored to the accepted session origin so retransmits
and packet splitting do not accumulate encoder-rounding errors. Dynamics are a
chosen first-order simulation model, not measured motor identification.
"""
import math
import numbers
import struct
from driver_loader import load_driver, get_message_types


class MotorBackend:
    def __init__(self, initial_robot=(0, 1466018, 893518, 0),
                 initial_handles=(0, 0, 0, 0), needle_profile='new',
                 tau_s=.04, timeout_s=.5, speed_limits=(10., 10., 2.)):
        self.kin = load_driver()
        self.initial_robot = self._integers(initial_robot)
        self.initial_handles = self._integers(initial_handles)
        if needle_profile not in ('new', 'legacy'):
            raise ValueError('needle_profile must be new or legacy')
        self.needle_profile = needle_profile
        self.offset_sign = -1. if needle_profile == 'new' else 1.
        self.tip_offset_mm = float(self.kin.DEFAULT_TIP_OFFSET_MM)
        self.tau_s = self._positive(tau_s, 'tau_s', zero=True)
        self.timeout_s = self._positive(timeout_s, 'timeout_s')
        if len(speed_limits) != 3:
            raise ValueError('speed_limits are A deg/s, B deg/s, C mm/s')
        self.speed_limits = tuple(self._positive(v, 'speed limit') for v in speed_limits)
        self._check_robot(self.initial_robot)
        self.robot = self.initial_robot
        self.handles = self.initial_handles
        self.target = self.robot
        self._native = self._to_native(self.robot)
        self._target_native = self._native
        self._anchor_robot = self.robot
        self._anchor_handles = self.handles
        self.last_command_s = None
        self.now_s = None
        self.fault = None
        self.status = 'idle'
        self.accepted_commands = 0
        self.history = []

    @staticmethod
    def _positive(value, name, zero=False):
        value = float(value)
        if not math.isfinite(value) or value < 0 or (not zero and value == 0):
            raise ValueError(name + ' must be finite and positive')
        return value

    @staticmethod
    def _integers(values):
        values = tuple(values)
        if len(values) != 4:
            raise ValueError('Expected exactly A/B/C/E')
        if any(isinstance(v, bool) or not isinstance(v, numbers.Integral) or
               not -(2**31) <= int(v) < 2**31 for v in values):
            raise ValueError('Encoder/handle fields must be int32 integers')
        return tuple(int(v) for v in values)

    def _check_robot(self, robot):
        # Original bounds and original nonlinear linkage branch, without edits.
        self.kin.validate_manual_jog(robot, (0, 0, 0, 0))
        alpha = self.kin.b_alpha_from_robot_encoder(robot[1])
        reencoded = round(self.kin.ENCODER_COUNTS_PER_MM * self.kin.b_distance_from_alpha(alpha))
        if abs(reencoded - robot[1]) > 1:
            raise ValueError('B encoder has no consistent uploaded-driver linkage branch')

    def _to_native(self, robot):
        return (robot[0] * 360. / 200000.,
                self.kin.b_alpha_from_robot_encoder(robot[1]),
                robot[2] / self.kin.ENCODER_COUNTS_PER_MM)

    def _from_native(self, values):
        phi, alpha, c_mm = values
        return (round(phi * 200000. / 360.),
                round(self.kin.b_distance_from_alpha(alpha) * self.kin.ENCODER_COUNTS_PER_MM),
                round(c_mm * self.kin.ENCODER_COUNTS_PER_MM), self.initial_robot[3])

    def command_for_robot_target(self, robot):
        """Convert absolute robot counts to cumulative handles in this session.

        A/C have protocol gear ratios; B handles encode angular increments,
        not differences in the nonlinear B encoder coordinate.
        """
        robot = self._integers(robot)
        self._check_robot(robot)
        if robot[3] != self.initial_robot[3]:
            raise ValueError('E-axis targets are unsupported')
        delta = self.kin.virtual_handles_from_robot_encoders(robot, self._anchor_robot)
        handles = tuple(a + b for a, b in zip(self._anchor_handles, delta))
        self._integers(handles)
        self.kin.validate_manual_jog(self._anchor_robot, delta)
        return self.kin.make_control_message(handles, 'hold')

    def hold(self):
        """Freeze this offline simulator at its feedback state; not a physical brake."""
        self.target = self.robot
        self._native = self._to_native(self.robot)
        self._target_native = self._native
        self._anchor_robot = self.robot
        self._anchor_handles = self.handles
        self.status = 'simulation_hold'
        self.history.append({'event': 'simulation_hold', 'time_s': self.now_s,
                             'robot': self.robot, 'physical_braking_validated': False})

    def reset_clock(self, time_s=0.):
        """Begin a new clock epoch with a held, reanchored session."""
        time_s = float(time_s)
        if not math.isfinite(time_s) or time_s < 0:
            raise ValueError('Clock origin must be finite and nonnegative')
        self.hold()
        self.now_s = time_s
        self.last_command_s = None
        self.fault = None

    def consume(self, command, now_s):
        now_s = float(now_s)
        if not math.isfinite(now_s) or (self.now_s is not None and now_s < self.now_s - 1e-12):
            raise ValueError('Command timestamp must be finite and monotonic')
        values = self._integers(getattr(command, 'handleEncoder' + a) for a in 'ABCE')
        flags = (command.stepForward, command.stepBackward)
        if any(type(flag) is not bool for flag in flags):
            raise ValueError('Direction fields must be bool')
        if any(flags):
            raise ValueError('Direction-only insertion rate is unspecified; use cumulative C handles, hold flags false')
        if values[3] != self.initial_handles[3]:
            raise ValueError('E-axis conversion is not calibrated; E changes are unsupported')
        delta = tuple(a-b for a,b in zip(values, self.handles))
        total = tuple(a-b for a,b in zip(values, self._anchor_handles))
        # All checks complete before accepting a packet or changing motion.
        candidate = self.kin.validate_manual_jog(self._anchor_robot, total)
        self._check_robot(candidate)
        target_native = self._to_native(candidate)
        duplicate = not any(delta)
        self.handles = values
        self.last_command_s = now_s
        self.now_s = now_s
        self.accepted_commands += 1
        if not duplicate:
            self.target = candidate
            self._target_native = target_native
            self.fault = None
            self.status = 'moving'
        self.history.append({'event':'command','time_s':now_s,'handles':values,
                             'handle_delta':delta,'target_robot':self.target,'duplicate':duplicate})
        return {'accepted':True,'duplicate':duplicate,'delta':delta,'target':self.target}

    def step(self, dt, now_s=None):
        dt = self._positive(dt, 'dt')
        now_s = (0. if self.now_s is None else self.now_s) + dt if now_s is None else float(now_s)
        if not math.isfinite(now_s) or (self.now_s is not None and now_s < self.now_s - 1e-12):
            raise ValueError('Step timestamp must be finite and monotonic')
        self.now_s = now_s
        if self.last_command_s is not None and now_s - self.last_command_s > self.timeout_s:
            if self.fault != 'feedback_command_timeout':
                self.target = self.robot
                self._native = self._to_native(self.robot)
                self._target_native = self._native
                self._anchor_robot = self.robot
                self._anchor_handles = self.handles
                self.fault = 'feedback_command_timeout'
                self.status = 'timeout_hold'
                self.history.append({'event':'watchdog_hold','time_s':now_s,'robot':self.robot})
            return self.feedback()
        weight = 1. if self.tau_s == 0 else -math.expm1(-dt / self.tau_s)
        native = []
        for actual, target, speed in zip(self._native, self._target_native, self.speed_limits):
            shift = (target-actual) * weight
            shift = max(-speed*dt,min(speed*dt,shift))
            native.append(actual+shift)
        robot = self._from_native(native)
        self._check_robot(robot)
        self._native = tuple(native)
        self.robot = robot
        if self.robot == self.target and self.fault is None:
            self.status = 'holding'
        return self.feedback()

    def feedback(self):
        _, Encoder = get_message_types()
        message = Encoder()
        for axis, handle, robot in zip('ABCE', self.handles, self.robot):
            setattr(message,'handleEncoder'+axis,int(handle))
            setattr(message,'robotEncoder'+axis,int(robot))
        junction = self.kin.model_tip_from_robot_encoders(self.robot, 0.)
        for axis, value in zip('XYZ', junction):
            # Preserve Encoder.msg float32 feedback semantics in offline mode.
            setattr(message,'pos'+axis,struct.unpack('f',struct.pack('f',float(value)))[0])
        message.stepForward = False
        message.stepBackward = False
        return message

    def tip_mm(self):
        return self.kin.model_tip_from_robot_encoders(self.robot,self.tip_offset_mm,self.offset_sign)

