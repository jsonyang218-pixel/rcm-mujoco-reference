"""Thread-safe ROS feedback storage shared by closed-loop nodes."""

import csv
import threading
import time
from collections import deque

from .kinematics import AXES, MAX_PLOT_POINTS


class RobotData:
    """Keep recent RobotState/command data without any Tk or ROS side effects."""

    tip_pose_fn = None

    def __init__(self, max_points):
        self._lock = threading.Lock()
        self._max_points = max(2, int(max_points))
        self._points = deque(maxlen=self._max_points)
        self._tip_points = deque(maxlen=self._max_points)
        self._records = deque(maxlen=self._max_points)
        self._commands = deque(maxlen=self._max_points)
        self.export_context_fn = None
        self._dropped_records = 0
        self._dropped_commands = 0
        self._tip_pose_fn = getattr(type(self), "tip_pose_fn", None)
        self._state = None
        self._command = None
        self._last_state_time = None
        self._last_state_monotonic = None
        self.state_observer = None
        self._last_command_time = None
        self._state_sequence = 0

    def state_callback(self, message):
        now = time.time()
        mono = time.monotonic()
        state = {
            "position": (float(message.posX), float(message.posY), float(message.posZ)),
            "robot": tuple(int(getattr(message, "robotEncoder" + axis)) for axis in AXES),
            "handle": tuple(int(getattr(message, "handleEncoder" + axis)) for axis in AXES),
            "forward": bool(message.stepForward),
            "backward": bool(message.stepBackward),
        }
        with self._lock:
            self._state = state
            self._state_sequence += 1
            self._last_state_time = now
            self._last_state_monotonic = mono
            self._points.append((now,) + state["position"])
            tip = None
            if self._tip_pose_fn is not None:
                try:
                    tip = tuple(self._tip_pose_fn(
                        state["robot"], state["position"],
                    )["display_tip"])
                    self._tip_points.append((now,) + tip)
                except (TypeError, ValueError):
                    pass
            context = dict(self.export_context_fn()) if self.export_context_fn else {}
            if len(self._records) == self._max_points:
                self._dropped_records += 1
            self._records.append({
                "unix_time": now,
                "sequence": self._state_sequence,
                "position": state["position"],
                "robot": state["robot"],
                "handle": state["handle"],
                "forward": state["forward"],
                "backward": state["backward"],
                "tip": tip,
                "context": context,
            })

            sequence = self._state_sequence
        observer = self.state_observer
        if observer is not None:
            observer(state, now, mono, sequence)

    def command_callback(self, message):
        now = time.time()
        command = {
            "handle": tuple(int(getattr(message, "handleEncoder" + axis)) for axis in AXES),
            "forward": bool(message.stepForward),
            "backward": bool(message.stepBackward),
        }
        with self._lock:
            self._command = command
            self._last_command_time = now
            if len(self._commands) == self._max_points:
                self._dropped_commands += 1
            self._commands.append({"unix_time": now, **command})

    def snapshot(self):
        return self._snapshot(include_points=True)

    def _snapshot(self, include_points):
        with self._lock:
            return {
                "state": None if self._state is None else dict(self._state),
                "command": None if self._command is None else dict(self._command),
                "last_state_time": self._last_state_time,
                "last_state_monotonic": self._last_state_monotonic,
                "last_command_time": self._last_command_time,
                "state_sequence": self._state_sequence,
                "point_count": len(self._points),
                "points": list(self._points) if include_points else None,
            }

    def snapshot_without_points(self):
        return self._snapshot(include_points=False)

    @staticmethod
    def _sample(points, maximum):
        maximum = max(2, int(maximum))
        count = len(points)
        if count <= maximum:
            return list(points)
        stride = float(count - 1) / float(maximum - 1)
        sampled = []
        next_index = 0.0
        for index, point in enumerate(points):
            if index >= next_index or index == count - 1:
                sampled.append(point)
                next_index += stride
        return sampled

    def plot_points(self, maximum=MAX_PLOT_POINTS):
        with self._lock:
            return self._sample(self._points, maximum)

    def tip_plot_points(self, maximum=MAX_PLOT_POINTS):
        with self._lock:
            return self._sample(self._tip_points, maximum)

    def clear_points(self, clear_records=True):
        with self._lock:
            self._points.clear()
            self._tip_points.clear()
            if clear_records:
                self._records.clear()
                self._commands.clear()
                self._dropped_records = 0
                self._dropped_commands = 0

    def feedback_records(self):
        """Return complete feedback samples, including encoders and tip pose."""
        with self._lock:
            return [dict(record) for record in self._records]

    def export_snapshot(self):
        """Take feedback and commands together without holding a lock for I/O."""
        with self._lock:
            return {"records": list(self._records), "commands": list(self._commands),
                    "dropped_feedback_records": self._dropped_records,
                    "dropped_command_records": self._dropped_commands}

    def export_points(self, filename):
        snapshot = self.snapshot()
        with open(filename, "w", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream)
            writer.writerow(("unix_time", "posX_mm", "posY_mm", "posZ_mm"))
            writer.writerows(snapshot["points"])
        return len(snapshot["points"])
