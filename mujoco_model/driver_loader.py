"""Load the unchanged uploaded controller helpers, with explicit offline schemas.

The compatibility dataclasses below are NOT ROS messages and cannot serialize
onto ROS. They are installed only when generated ros_msg messages are absent.
The optional ROS bridge refuses to run with these schemas.
"""
from dataclasses import dataclass
import importlib
from pathlib import Path
import sys
import types

HERE = Path(__file__).resolve().parent
VENDOR_SRC = HERE / 'vendor/simulation_driver/ros_msg/src'


@dataclass
class OfflineControlSignal:
    handleEncoderA: int = 0
    handleEncoderB: int = 0
    handleEncoderC: int = 0
    handleEncoderE: int = 0
    stepForward: bool = False
    stepBackward: bool = False
    _offline_schema = True


@dataclass
class OfflineEncoder(OfflineControlSignal):
    robotEncoderA: int = 0
    robotEncoderB: int = 0
    robotEncoderC: int = 0
    robotEncoderE: int = 0
    posX: float = 0.0
    posY: float = 0.0
    posZ: float = 0.0
    _offline_schema = True


def get_message_types(require_ros=False):
    try:
        messages = importlib.import_module('ros_msg.msg')
    except ModuleNotFoundError as exc:
        if exc.name not in ('ros_msg', 'ros_msg.msg'):
            raise
        if require_ros:
            raise RuntimeError('Generated ROS1 ros_msg messages are required; source catkin devel/setup.bash') from exc
        # Supply only the exact fields needed by the unchanged pure helpers.
        package = types.ModuleType('ros_msg')
        package.__path__ = []
        messages = types.ModuleType('ros_msg.msg')
        messages.controlSignal = OfflineControlSignal
        messages.Encoder = OfflineEncoder
        package.msg = messages
        sys.modules['ros_msg'] = package
        sys.modules['ros_msg.msg'] = messages
    control, encoder = messages.controlSignal, messages.Encoder
    if require_ros and (getattr(control, '_offline_schema', False) or
                        getattr(encoder, '_offline_schema', False)):
        raise RuntimeError('Offline schemas are not ROS messages; launch a fresh sourced ROS1 process')
    return control, encoder


def load_driver():
    if not VENDOR_SRC.is_dir():
        raise FileNotFoundError(f'Pinned unchanged controller sources absent: {VENDOR_SRC}')
    get_message_types()
    path = str(VENDOR_SRC)
    if path not in sys.path:
        sys.path.insert(0, path)
    return importlib.import_module('robot_control.kinematics')

