"""Load a pinned controller version, defaulting to G15, with offline schemas.

The compatibility dataclasses below are NOT ROS messages and cannot serialize
onto ROS. They are installed only when generated ros_msg messages are absent.
The optional ROS bridge refuses to run with these schemas.
"""
from dataclasses import dataclass
import hashlib
import importlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import types

HERE = Path(__file__).resolve().parent
VENDOR_SRC = HERE / 'vendor/simulation_driver/ros_msg/src'
G15_SRC = HERE / 'vendor/simulation_driver_g15/ros_msg/src'
DEFAULT_DRIVER_VARIANT = 'g15'
DRIVER_SOURCES = {'g10': VENDOR_SRC, 'g15': G15_SRC}


def selected_driver_variant():
    variant = os.environ.get('RCM_DRIVER_VARIANT', DEFAULT_DRIVER_VARIANT)
    if variant not in DRIVER_SOURCES:
        raise ValueError('RCM_DRIVER_VARIANT must be g10 or g15')
    return variant


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
    variant = selected_driver_variant()
    source = DRIVER_SOURCES[variant]
    package_dir = source / 'robot_control'
    if not package_dir.is_dir():
        raise FileNotFoundError(f'Pinned {variant} controller sources absent: {source}')
    # A process uses one package version. Never silently reuse another driver's
    # cached modules, including a package preloaded by an external ROS checkout.
    for name, module in tuple(sys.modules.items()):
        if name == 'robot_control' or name.startswith('robot_control.'):
            filename = getattr(module, '__file__', None)
            if filename is None or Path(filename).resolve().parent != package_dir.resolve():
                raise RuntimeError('A different robot_control is already loaded; '
                                   'select RCM_DRIVER_VARIANT in a fresh process')
    manifest_path = HERE / ('source_manifest_g15.json' if variant == 'g15'
                            else 'source_manifest.json')
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    kin_path = package_dir / 'kinematics.py'
    expected = manifest['files_sha256']['ros_msg/src/robot_control/kinematics.py']
    if hashlib.sha256(kin_path.read_bytes()).hexdigest() != expected:
        raise RuntimeError(f'Pinned {variant} kinematics SHA-256 mismatch')
    get_message_types()
    if 'robot_control' not in sys.modules:
        spec = importlib.util.spec_from_file_location(
            'robot_control', package_dir / '__init__.py',
            submodule_search_locations=[str(package_dir)])
        package = importlib.util.module_from_spec(spec)
        sys.modules['robot_control'] = package
        try:
            spec.loader.exec_module(package)
        except Exception:
            for name in tuple(sys.modules):
                if name == 'robot_control' or name.startswith('robot_control.'):
                    del sys.modules[name]
            raise
    kin = importlib.import_module('robot_control.kinematics')
    if kin.CONTROL_GEAR != int(variant[1:]):
        raise RuntimeError(f'Pinned {variant} CONTROL_GEAR mismatch')
    return kin


def driver_identity():
    kin = load_driver()
    return {'variant': selected_driver_variant(), 'control_gear': kin.CONTROL_GEAR,
            'kinematics_path': str(Path(kin.__file__).resolve()),
            'kinematics_sha256': hashlib.sha256(Path(kin.__file__).read_bytes()).hexdigest()}

