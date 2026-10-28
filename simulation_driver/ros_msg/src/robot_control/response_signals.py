"""ROS-free, immutable elapsed-time response signals."""

from dataclasses import asdict, dataclass
import math


@dataclass(frozen=True)
class ResponseSignal:
    kind: str = "step"
    amplitude_um: float = 10.0
    frequency_hz: float = 0.01
    speed_um_s: float = 1.0
    pulse_width_s: float = 10.0
    cycles: int = 4  # One transient cycle followed by three analysis cycles.
    baseline_s: float = 5.0
    hold_s: float = 15.0
    recovery_s: float = 15.0
    pretrial: bool = False

    def __post_init__(self):
        if self.kind not in ("noise", "step", "sine", "ramp", "pulse"):
            raise ValueError("未知响应信号")
        values = tuple(value for name, value in asdict(self).items() if name not in ("kind", "pretrial"))
        if not all(math.isfinite(v) and v > 0 for v in values):
            raise ValueError("信号参数必须为有限正数")
        if self.amplitude_um not in (10, 20):
            raise ValueError("首轮幅值仅允许 10 或 20 µm")
        if self.frequency_hz not in (0.01, 0.02, 0.05):
            raise ValueError("频率仅允许 0.01、0.02、0.05 Hz")
        if self.speed_um_s not in (1, 2, 5):
            raise ValueError("速度仅允许 1、2、5 µm/s")
        if self.pulse_width_s not in (2, 5, 10):
            raise ValueError("脉宽仅允许 2、5、10 s")
        if (not isinstance(self.pretrial, bool) or not isinstance(self.cycles, int) or
                (self.cycles != 1 if self.pretrial else self.cycles < 4)):
            raise ValueError("正弦预试为1周期；正式至少四周期，首周期过渡")

    @property
    def active_s(self):
        return {"noise": 30.0, "step": self.hold_s,
                "sine": self.cycles / self.frequency_hz,
                "ramp": 2 * self.amplitude_um / self.speed_um_s + self.hold_s,
                "pulse": self.pulse_width_s}[self.kind]

    @property
    def duration_s(self):
        return 30.0 if self.kind == "noise" else self.baseline_s + self.active_s + self.recovery_s

    @property
    def bounds_um(self):
        return (-self.amplitude_um if self.kind == "sine" else 0.0,
                0.0 if self.kind == "noise" else self.amplitude_um)

    def at(self, elapsed_s):
        """Return (position in µm, phase); no arrival-dependent timing."""
        t = float(elapsed_s)
        if not math.isfinite(t):
            raise ValueError("时间必须有限")
        t = max(0.0, t)
        if self.kind == "noise":
            return 0.0, "noise"
        if t < self.baseline_s:
            return 0.0, "baseline"
        u = t - self.baseline_s
        if u >= self.active_s:
            return 0.0, "recovery"
        if self.kind == "step":
            return self.amplitude_um, "plateau"
        if self.kind == "pulse":
            return self.amplitude_um, "pulse"
        if self.kind == "sine":
            return (self.amplitude_um * math.sin(2 * math.pi * self.frequency_hz * u),
                    "transient" if u < 1 / self.frequency_hz else "steady")
        travel = self.amplitude_um / self.speed_um_s
        if u < travel:
            return self.speed_um_s * u, "outward"
        if u < travel + self.hold_s:
            return self.amplitude_um, "plateau"
        return self.amplitude_um - self.speed_um_s * (u - travel - self.hold_s), "return"


@dataclass(frozen=True)
class WireSineSignal:
    """Dimensionless normalized sine, including a one-cycle safety pretrial.

    amplitude_um is a compatibility normalization only; no physical position
    target is implied. Protocol amplitudes live in WireSineExperiment.
    """
    kind: str = "sine"
    amplitude_um: float = 1.0
    frequency_hz: float = .01
    cycles: int = 4
    baseline_s: float = 5.
    recovery_s: float = 15.

    def __post_init__(self):
        if (self.kind != "sine" or self.amplitude_um != 1 or
                self.frequency_hz not in (.01, .02, .05) or
                not isinstance(self.cycles, int) or self.cycles not in (1, 4) or
                not all(math.isfinite(v) and v > 0 for v in (self.baseline_s, self.recovery_s))):
            raise ValueError("信号正弦需 0.01/0.02/0.05 Hz，预试1周期或正式4周期")

    @property
    def active_s(self):
        return self.cycles / self.frequency_hz

    @property
    def duration_s(self):
        return self.baseline_s + self.active_s + self.recovery_s

    def at(self, elapsed_s):
        if not math.isfinite(elapsed_s):
            raise ValueError("时间必须有限")
        u = elapsed_s - self.baseline_s
        if u < 0:
            return 0., 'baseline'
        if u >= self.active_s:
            return 0., 'recovery'
        return math.sin(2 * math.pi * self.frequency_hz * u), 'transient' if u < 1 / self.frequency_hz else 'steady'
