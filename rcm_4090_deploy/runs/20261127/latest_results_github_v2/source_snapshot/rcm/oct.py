"""Registered OCT interface, currently backed by ideal MuJoCo axial depth.

This module does not simulate OCT image formation, produce a B-scan, or connect
to hardware. All lengths are metres. Depth is camera optical-axis z, not range.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import json
import math
from pathlib import Path
import time
from typing import Any, Literal, Protocol, runtime_checkable
from uuid import uuid4

import numpy as np


SIMULATION_SOURCE = "mujoco_ideal_registered_oct_simulation"
DEPTH_CONVENTION = "positive_camera_optical_axis_z_not_ray_range"


def _finite_scalar(value, name):
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be a finite number")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite number") from exc
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _json_copy(value):
    """Validate portable JSON without arrays, NaNs, or caller-owned objects."""
    if isinstance(value, np.ndarray):
        raise ValueError("metadata must be JSON scalars/containers, not depth arrays")
    try:
        return json.loads(json.dumps(value, allow_nan=False))
    except (TypeError, ValueError) as exc:
        raise ValueError("metadata must contain finite JSON values") from exc


@dataclass(frozen=True)
class RegisteredOCTFrame:
    """Depth maps registered to one camera, with explicit acquisition provenance.

    The current provider is a simulation. Future adapters must preserve truthful
    source/clock/calibration fields and supply already segmented, registered
    metric depth. Valid masks mean finite positive depth inside configured
    bounds; they do not establish feature identity or physical OCT observability.
    """

    scene_surface_depth_m: np.ndarray = field(repr=False)
    registered_feature_depth_m: np.ndarray = field(repr=False)
    scene_valid_mask: np.ndarray = field(repr=False)
    feature_valid_mask: np.ndarray = field(repr=False)
    frame_id: str
    sequence_id: int
    timestamp_s: float
    acquired_monotonic_s: float
    nominal_period_s: float
    camera_name: str
    stream_id: str = ""
    clock_domain: str = "simulation"
    source: str = SIMULATION_SOURCE
    is_simulation: bool = True
    raw_oct_available: bool = False
    realtime_validated: bool = False
    depth_unit: str = "m"
    depth_convention: str = DEPTH_CONVENTION
    min_valid_depth_m: float = 0.0
    max_valid_depth_m: float | None = None
    timestamp_repeated: bool = False
    registration_description: str = (
        "Ideal renderer depth registered pixel-for-pixel to the named MuJoCo "
        "camera; scene includes fundus, feature pass excludes fundus backdrop. "
        "No physical OCT calibration or hardware registration is validated."
    )
    extra_metadata: dict[str, Any] = field(default_factory=dict, repr=False)

    def metadata(self):
        """Return small JSON metadata; full maps stay in the frame/NPZ file."""
        return {
            "schema_version": 1, "frame_id": self.frame_id,
            "stream_id": self.stream_id,
            "sequence_id": self.sequence_id, "timestamp_s": self.timestamp_s,
            "clock_domain": self.clock_domain,
            "acquired_monotonic_s": self.acquired_monotonic_s,
            "nominal_period_s": self.nominal_period_s,
            "nominal_frequency_hz": 1.0 / self.nominal_period_s,
            "nominal_rate_is_wall_clock_guarantee": False,
            "source": self.source, "is_simulation": self.is_simulation,
            "raw_oct_available": self.raw_oct_available,
            "realtime_validated": self.realtime_validated,
            "camera_name": self.camera_name,
            "registration_frame": self.camera_name,
            "registration_description": self.registration_description,
            "depth_unit": self.depth_unit, "depth_convention": self.depth_convention,
            "shape_hw": list(self.scene_surface_depth_m.shape),
            "depth_dtype": str(self.scene_surface_depth_m.dtype),
            "validity_rule": "finite and strictly greater than min; strictly less than max when set",
            "min_valid_depth_m": self.min_valid_depth_m,
            "max_valid_depth_m": self.max_valid_depth_m,
            "scene_valid_pixels": int(np.count_nonzero(self.scene_valid_mask)),
            "feature_valid_pixels": int(np.count_nonzero(self.feature_valid_mask)),
            "timestamp_repeated": self.timestamp_repeated,
            "fresh_simulation_timestamp": not self.timestamp_repeated,
            "extra": _json_copy(self.extra_metadata),
        }

    def sample(self, pixel, *, map: Literal["feature", "scene"] = "feature"):
        """Sample a valid integer (u, v) pixel, returning axial z in metres.

        No interpolation or clamping is applied; subpixel coordinates must be
        resolved explicitly by the caller rather than silently rounded here.
        """
        if map not in ("feature", "scene"):
            raise ValueError("map must be 'feature' or 'scene'")
        coordinates = np.asarray(pixel)
        if coordinates.shape != (2,) or coordinates.dtype.kind not in "iuf":
            raise ValueError("pixel must contain two finite integer coordinates (u, v)")
        if not np.isfinite(coordinates).all() or not np.equal(coordinates, np.rint(coordinates)).all():
            raise ValueError("pixel must contain two finite integer coordinates (u, v)")
        u, v = (int(value) for value in coordinates)
        values, mask = ((self.registered_feature_depth_m, self.feature_valid_mask)
                        if map == "feature" else
                        (self.scene_surface_depth_m, self.scene_valid_mask))
        height, width = values.shape
        if not (0 <= u < width and 0 <= v < height):
            raise IndexError(f"pixel {(u, v)} outside image {(width, height)}")
        if not mask[v, u] or not math.isfinite(float(values[v, u])) or values[v, u] <= 0:
            raise ValueError(f"pixel {(u, v)} has no valid {map} depth")
        return float(values[v, u])

    def save(self, path):
        """Write numeric NPZ and adjacent JSON metadata, without pickle."""
        path = Path(path)
        if path.suffix != ".npz":
            path = path.with_suffix(".npz")
        path.parent.mkdir(parents=True, exist_ok=True)
        metadata = self.metadata()
        encoded = json.dumps(metadata, ensure_ascii=False, allow_nan=False, indent=2)
        np.savez_compressed(
            path, scene_surface_depth_m=self.scene_surface_depth_m,
            registered_feature_depth_m=self.registered_feature_depth_m,
            scene_valid_mask=self.scene_valid_mask,
            feature_valid_mask=self.feature_valid_mask,
            metadata_json=np.asarray(encoded),
        )
        path.with_suffix(".json").write_text(encoded + "\n", encoding="utf-8")
        return path

    @classmethod
    def load(cls, path):
        """Read a saved frame with allow_pickle=False and validate its maps."""
        with np.load(path, allow_pickle=False) as saved:
            metadata = json.loads(str(saved["metadata_json"].item()))
            scene = np.array(saved["scene_surface_depth_m"], copy=True)
            feature = np.array(saved["registered_feature_depth_m"], copy=True)
            scene_mask = np.array(saved["scene_valid_mask"], copy=True)
            feature_mask = np.array(saved["feature_valid_mask"], copy=True)
        if metadata.get("schema_version") != 1:
            raise ValueError("unsupported OCT frame schema")
        provider = SyntheticOCTProvider(
            nominal_period_s=metadata["nominal_period_s"],
            min_valid_depth_m=metadata["min_valid_depth_m"],
            max_valid_depth_m=metadata["max_valid_depth_m"],
        )
        validated = provider.capture(
            scene, feature, timestamp_s=metadata["timestamp_s"],
            camera_name=metadata["camera_name"],
            acquisition_wall_s=metadata["acquired_monotonic_s"],
            metadata=metadata["extra"],
        )
        if not np.array_equal(scene_mask, validated.scene_valid_mask) or not np.array_equal(feature_mask, validated.feature_valid_mask):
            raise ValueError("saved valid masks do not match depth/bounds")
        fixed = {key: metadata[key] for key in (
            "source", "is_simulation", "raw_oct_available", "realtime_validated",
            "clock_domain", "depth_unit", "depth_convention")}
        if fixed != {key: getattr(validated, key) for key in fixed}:
            raise ValueError("saved provenance is inconsistent with simulation frame")
        sequence = metadata["sequence_id"]
        if isinstance(sequence, bool) or not isinstance(sequence, int) or sequence < 0:
            raise ValueError("saved sequence_id must be a nonnegative integer")
        return cls(
            scene_surface_depth_m=validated.scene_surface_depth_m,
            registered_feature_depth_m=validated.registered_feature_depth_m,
            scene_valid_mask=validated.scene_valid_mask,
            feature_valid_mask=validated.feature_valid_mask,
            frame_id=str(metadata["frame_id"]), sequence_id=sequence,
            timestamp_s=validated.timestamp_s,
            acquired_monotonic_s=validated.acquired_monotonic_s,
            nominal_period_s=validated.nominal_period_s,
            camera_name=validated.camera_name,
            stream_id=str(metadata["stream_id"]),
            min_valid_depth_m=validated.min_valid_depth_m,
            max_valid_depth_m=validated.max_valid_depth_m,
            timestamp_repeated=bool(metadata["timestamp_repeated"]),
            registration_description=str(metadata["registration_description"]),
            extra_metadata=validated.extra_metadata,
        )


OCTFrame = RegisteredOCTFrame


@runtime_checkable
class OCTSource(Protocol):
    """A future registered source implements acquisition returning this frame.

    Hardware implementations normally own acquisition arguments. A real adapter
    must set its actual device provenance, clock, latency and registration;
    raw B-scan data alone is not a registered depth map.
    """

    def capture(self, *args, **kwargs) -> RegisteredOCTFrame: ...


class SyntheticOCTProvider:
    """Adapt rendered metric depth without reading MuJoCo state or truth APIs."""

    def __init__(self, nominal_period_s=.05, *, min_valid_depth_m=0., max_valid_depth_m=None):
        self.nominal_period_s = _finite_scalar(nominal_period_s, "nominal_period_s")
        self.min_valid_depth_m = _finite_scalar(min_valid_depth_m, "min_valid_depth_m")
        self.max_valid_depth_m = (None if max_valid_depth_m is None else
                                 _finite_scalar(max_valid_depth_m, "max_valid_depth_m"))
        if self.nominal_period_s <= 0 or self.min_valid_depth_m < 0:
            raise ValueError("period must be positive and minimum depth nonnegative")
        if self.max_valid_depth_m is not None and self.max_valid_depth_m <= self.min_valid_depth_m:
            raise ValueError("maximum depth must exceed minimum depth")
        self._next_sequence_id = 0
        self._last_timestamp_s = None
        self._last_acquired_monotonic_s = None
        self.stream_id = f"sim-oct-{uuid4().hex}"

    def _depth_map(self, value, name):
        array = np.asarray(value)
        if array.ndim != 2 or not all(array.shape) or array.dtype.kind not in "iuf":
            raise ValueError(f"{name} must be a nonempty 2D numeric depth map in metres")
        with np.errstate(over="ignore", invalid="ignore"):
            depth = array.astype(np.float32, copy=True)
        valid = np.isfinite(depth) & (depth > self.min_valid_depth_m)
        if self.max_valid_depth_m is not None:
            valid &= depth < self.max_valid_depth_m
        return depth, valid

    def capture(self, scene_depth_m, feature_depth_m, *, timestamp_s,
                camera_name="eye_internal", acquisition_wall_s=None,
                allow_repeated_timestamp=False, metadata=None):
        """Copy depth maps as metres; never rescale, mutate or infer raw OCT.

        timestamp_s belongs to the simulation clock. acquisition_wall_s is a
        monotonic wall-clock reading, defaulting to time.monotonic(). Equal
        simulation timestamps require explicit opt-in and are marked nonfresh.
        Optional bounds exclude far-clip/no-return pixels through valid masks;
        values themselves are retained unchanged apart from float32 conversion.
        """
        timestamp = _finite_scalar(timestamp_s, "timestamp_s")
        if timestamp < 0:
            raise ValueError("simulation timestamp_s must be nonnegative")
        repeated = self._last_timestamp_s is not None and timestamp == self._last_timestamp_s
        if self._last_timestamp_s is not None and timestamp < self._last_timestamp_s:
            raise ValueError("simulation timestamp_s must not decrease")
        if repeated and not allow_repeated_timestamp:
            raise ValueError("repeated simulation timestamp requires allow_repeated_timestamp=True")
        if not isinstance(camera_name, str) or not camera_name.strip():
            raise ValueError("camera_name must identify the registration camera")
        acquired = (time.monotonic() if acquisition_wall_s is None else
                    _finite_scalar(acquisition_wall_s, "acquisition_wall_s"))
        if acquired < 0:
            raise ValueError("acquisition_wall_s must be a nonnegative monotonic reading")
        if self._last_acquired_monotonic_s is not None and acquired < self._last_acquired_monotonic_s:
            raise ValueError("acquisition_wall_s must not decrease within a source stream")
        extra = {} if metadata is None else _json_copy(metadata)
        if not isinstance(extra, dict):
            raise ValueError("metadata must be a JSON object")
        scene, scene_valid = self._depth_map(scene_depth_m, "scene_depth_m")
        feature, feature_valid = self._depth_map(feature_depth_m, "feature_depth_m")
        if scene.shape != feature.shape:
            raise ValueError("scene and feature depth maps must have the same shape")
        sequence = self._next_sequence_id
        frame = RegisteredOCTFrame(
            scene_surface_depth_m=scene, registered_feature_depth_m=feature,
            scene_valid_mask=scene_valid, feature_valid_mask=feature_valid,
            frame_id=f"{self.stream_id}/{sequence:08d}", sequence_id=sequence,
            timestamp_s=timestamp, acquired_monotonic_s=acquired,
            nominal_period_s=self.nominal_period_s, camera_name=camera_name,
            stream_id=self.stream_id,
            min_valid_depth_m=self.min_valid_depth_m,
            max_valid_depth_m=self.max_valid_depth_m,
            timestamp_repeated=repeated, extra_metadata=extra,
        )
        self._next_sequence_id += 1
        self._last_timestamp_s = timestamp
        self._last_acquired_monotonic_s = acquired
        return frame


def relative_depth(tip_xyz_camera, target_xyz_camera, *, source_flags=None,
                   needle_axis_camera=None):
    """Describe target-minus-tip differences in one metric camera frame.

    Axial depth difference is delta-z. Euclidean separation and projection onto
    a supplied positive needle axis are distinct quantities, never substitutes.
    This function does not infer feature locations from OCT or consult truth.
    """
    tip, target = (np.asarray(value, dtype=float) for value in (tip_xyz_camera, target_xyz_camera))
    if tip.shape != (3,) or target.shape != (3,) or not np.isfinite(tip).all() or not np.isfinite(target).all():
        raise ValueError("tip and target must be finite camera XYZ vectors in metres")
    delta = target - tip
    result = {
        "definition": "target_minus_tip", "unit": "m",
        "coordinate_frame": "camera_optical",
        "delta_xyz_m": delta.tolist(),
        "axial_depth_difference_m": float(delta[2]),
        "euclidean_distance_m": float(np.linalg.norm(delta)),
        "along_needle_difference_m": None,
        "source_flags": _json_copy({} if source_flags is None else source_flags),
    }
    if needle_axis_camera is not None:
        axis = np.array(needle_axis_camera, dtype=float, copy=True)
        if axis.shape != (3,) or not np.isfinite(axis).all() or np.linalg.norm(axis) <= 0:
            raise ValueError("needle axis must be a finite nonzero camera XYZ vector")
        axis /= np.linalg.norm(axis)
        result["positive_needle_axis_camera"] = axis.tolist()
        result["along_needle_difference_m"] = float(delta @ axis)
    return result
