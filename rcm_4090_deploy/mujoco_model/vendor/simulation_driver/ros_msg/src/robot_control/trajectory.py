"""Pure trajectory geometry and file loading helpers."""

import csv
import math


MIN_CIRCLE_RADIUS_UM = 10.0
MAX_CIRCLE_RADIUS_UM = 10_000.0
MIN_CIRCLE_POINTS = 8
MAX_CIRCLE_POINTS = 360
DEFAULT_CIRCLE_RADIUS_UM = 200.0
MIN_RECTANGLE_SIZE_UM = 10.0
MAX_RECTANGLE_SIZE_UM = 10_000.0
MIN_SINE_SIZE_UM = 10.0
MAX_SINE_SIZE_UM = 10_000.0
MAX_SINE_CYCLES = 20
MIN_SINE_SEGMENTS_PER_CYCLE = 16
MAX_SINE_SEGMENTS = 1440
MIN_TRAJECTORY_STEP_UM = 0.0
MAX_TRAJECTORY_STEP_UM = 600.0
DEFAULT_TRAJECTORY_STEP_UM = 300.0
PLANE_AXES = {
    "xy": (0, 1),
    "xz": (0, 2),
    "yz": (1, 2),
}


MAX_SPATIAL_POINTS = 10000
MAX_EXECUTION_POINTS = 5000


def load_spatial_targets(filename, units="mm"):
    """Read finite XYZ samples, with an optional named header, in mm/um.

    Named XYZ columns may appear in any order (including an exported plan's
    waypoint_index column). Unit suffixes must match the selected units.
    Numeric files without a header must have exactly three columns.
    """
    if units not in ("mm", "um"):
        raise ValueError("Spatial path units must be mm or um")
    scale = 0.001 if units == "um" else 1.0
    points = []
    columns = None
    first = True
    row_count = 0
    with open(filename, "r", newline="", encoding="utf-8-sig") as stream:
        for line_number, raw in enumerate(stream, start=1):
            if not raw.strip() or raw.lstrip().startswith("#"):
                continue
            row_count += 1
            if row_count > MAX_SPATIAL_POINTS + 1:
                raise ValueError("Spatial CSV exceeds %d data rows" % MAX_SPATIAL_POINTS)
            fields = next(csv.reader([raw]))
            if first:
                first = False
                names = [field.strip().lower() for field in fields]
                if any(name in ("x", "x_mm", "x_um") for name in names):
                    suffix = "_" + units
                    if all(names.count(axis + suffix) == 1 for axis in "xyz"):
                        columns = [names.index(axis + suffix) for axis in "xyz"]
                    elif all(names.count(axis) == 1 for axis in "xyz"):
                        columns = [names.index(axis) for axis in "xyz"]
                    else:
                        raise ValueError("XYZ header units must match selected " + units)
                    continue
            indices = columns if columns is not None else (0, 1, 2)
            if (columns is None and len(fields) != 3) or len(fields) <= max(indices):
                raise ValueError("Spatial path line %d must contain X,Y,Z" % line_number)
            try:
                point = tuple(float(fields[index]) * scale for index in indices)
            except ValueError:
                raise ValueError("Spatial path line %d has non-numeric XYZ" % line_number)
            if not all(math.isfinite(value) for value in point):
                raise ValueError("Spatial path line %d has non-finite XYZ" % line_number)
            # Consecutive duplicates add no geometry or useful arrival test.
            if not points or point != points[-1]:
                points.append(point)
            if len(points) > MAX_SPATIAL_POINTS:
                raise ValueError("Spatial path exceeds %d samples" % MAX_SPATIAL_POINTS)
    if len(points) < 2:
        raise ValueError("Spatial path requires at least two distinct points (non-zero distance)")
    return points


def build_spatial_targets(start_tip, samples, frame="relative", retrace=False):
    """Position a sampled 3D curve and optionally retrace it to start.

    relative: XYZ offsets from the current tip; rcm: absolute RCM-frame XYZ.
    A connecting line from the current tip to the first sample is explicit.
    Retrace includes that line and returns to the exact starting tip.
    """
    start = _centre(start_tip)
    if frame not in ("relative", "rcm"):
        raise ValueError("Spatial frame must be relative or rcm")
    if not 2 <= len(samples) <= MAX_SPATIAL_POINTS:
        raise ValueError("Spatial path requires 2–%d points" % MAX_SPATIAL_POINTS)
    path = [start]
    for sample in samples:
        point = _centre(sample)
        if frame == "relative":
            point = tuple(start[i] + point[i] for i in range(3))
        if point != path[-1]:
            path.append(point)
    if len(path) < 2:
        raise ValueError("Spatial path has no displacement")
    return path + list(reversed(path[:-1])) if retrace else path


def _plane_axes(plane):
    try:
        return PLANE_AXES[plane]
    except KeyError:
        raise ValueError("Trajectory plane must be one of: xy, xz, yz")


def _centre(centre_tip):
    centre = tuple(float(value) for value in centre_tip)
    if len(centre) != 3 or not all(math.isfinite(value) for value in centre):
        raise ValueError("Trajectory centre must contain three finite coordinates")
    return centre


def _bounded_size(value, name, minimum, maximum):
    size = float(value)
    if not math.isfinite(size) or not minimum <= size <= maximum:
        raise ValueError(
            "%s must be between %.0f and %.0f um" %
            (name, minimum, maximum)
        )
    return size


def _bounded_integer(value, name, minimum, maximum):
    number = float(value)
    if (not math.isfinite(number) or not number.is_integer() or
            not minimum <= number <= maximum):
        raise ValueError("%s must be an integer between %d and %d" %
                         (name, minimum, maximum))
    return int(number)


def trajectory_step_mm_from_um(value):
    """Convert the UI trajectory step to mm; zero selects the default step."""
    step = float(value)
    if (not math.isfinite(step) or
            not MIN_TRAJECTORY_STEP_UM <= step <= MAX_TRAJECTORY_STEP_UM):
        raise ValueError(
            "单步步长必须在 %.0f–%.0f µm 范围内" %
            (MIN_TRAJECTORY_STEP_UM, MAX_TRAJECTORY_STEP_UM)
        )
    if step == MIN_TRAJECTORY_STEP_UM:
        step = DEFAULT_TRAJECTORY_STEP_UM
    return step / 1000.0


def recommended_circle_points(radius_um, sample_arc_step_um=300.0):
    """Return samples from circumference length, clamped to 8..360 points."""
    radius = _bounded_size(
        radius_um, "Circle radius", MIN_CIRCLE_RADIUS_UM, MAX_CIRCLE_RADIUS_UM,
    )
    step = float(sample_arc_step_um)
    if not math.isfinite(step) or step <= 0.0:
        raise ValueError("Circle sample arc step must be positive")
    estimated = int(math.ceil(2.0 * math.pi * radius / step))
    return max(MIN_CIRCLE_POINTS, min(MAX_CIRCLE_POINTS, estimated))


def build_circle_targets(centre_tip, plane="xy",
                         radius_um=DEFAULT_CIRCLE_RADIUS_UM,
                         points=None, clockwise=False,
                         sample_arc_step_um=DEFAULT_TRAJECTORY_STEP_UM):
    centre = _centre(centre_tip)
    first_axis, second_axis = _plane_axes(plane)
    radius = _bounded_size(
        radius_um, "Circle radius", MIN_CIRCLE_RADIUS_UM, MAX_CIRCLE_RADIUS_UM,
    )
    if points is None:
        point_count = recommended_circle_points(radius, sample_arc_step_um)
    else:
        point_count = int(points)
        if (point_count != points or
                not MIN_CIRCLE_POINTS <= point_count <= MAX_CIRCLE_POINTS):
            raise ValueError(
                "Circle point count must be an integer between %d and %d" %
                (MIN_CIRCLE_POINTS, MAX_CIRCLE_POINTS)
            )

    radius_mm = radius / 1000.0
    direction = -1.0 if clockwise else 1.0
    targets = []
    for index in range(point_count + 1):
        angle = direction * 2.0 * math.pi * index / point_count
        target = list(centre)
        target[first_axis] += radius_mm * math.cos(angle)
        target[second_axis] += radius_mm * math.sin(angle)
        targets.append(tuple(target))
    targets[-1] = targets[0]
    targets.append(centre)
    return targets


def build_rectangle_targets(centre_tip, plane="xy", width_um=200.0,
                            height_um=200.0, clockwise=False,
                            points_per_edge=1):
    centre = _centre(centre_tip)
    first_axis, second_axis = _plane_axes(plane)
    width = _bounded_size(
        width_um, "Rectangle width", MIN_RECTANGLE_SIZE_UM,
        MAX_RECTANGLE_SIZE_UM,
    ) / 1000.0
    height = _bounded_size(
        height_um, "Rectangle height", MIN_RECTANGLE_SIZE_UM,
        MAX_RECTANGLE_SIZE_UM,
    ) / 1000.0
    edge_segments = int(points_per_edge)
    if (edge_segments != points_per_edge or edge_segments < 1 or
            edge_segments > 120):
        raise ValueError("Rectangle points per edge must be an integer between 1 and 120")
    offsets = (
        (-width / 2.0, -height / 2.0),
        (width / 2.0, -height / 2.0),
        (width / 2.0, height / 2.0),
        (-width / 2.0, height / 2.0),
    )
    if clockwise:
        offsets = (offsets[0],) + tuple(reversed(offsets[1:]))
    targets = []
    for corner_index in range(4):
        start = offsets[corner_index]
        end = offsets[(corner_index + 1) % 4]
        for segment_index in range(edge_segments):
            fraction = segment_index / edge_segments
            first_offset = start[0] + fraction * (end[0] - start[0])
            second_offset = start[1] + fraction * (end[1] - start[1])
            target = list(centre)
            target[first_axis] += first_offset
            target[second_axis] += second_offset
            targets.append(tuple(target))
    targets.append(targets[0])
    targets.append(centre)
    return targets


def build_sine_targets(start_tip, plane="xy", length_um=200.0,
                       amplitude_um=200.0, cycles=1, segments=None,
                       inverted=False,
                       sample_step_um=DEFAULT_TRAJECTORY_STEP_UM):
    """Sample u=L*t, v=A*sin(2*pi*N*t), then retrace to the exact start.

    Coordinates are mm; length and single-sided amplitude inputs are um.
    The plane's first axis advances positively, its second axis oscillates,
    and its third coordinate stays fixed. Returned targets omit the initial
    start point, include the end once, and finish at start_tip.
    """
    start = _centre(start_tip)
    first_axis, second_axis = _plane_axes(plane)
    length = _bounded_size(
        length_um, "Sine length", MIN_SINE_SIZE_UM, MAX_SINE_SIZE_UM,
    )
    amplitude = _bounded_size(
        amplitude_um, "Sine amplitude", MIN_SINE_SIZE_UM, MAX_SINE_SIZE_UM,
    )
    period_count = _bounded_integer(cycles, "Sine cycles", 1, MAX_SINE_CYCLES)
    sample_step = float(sample_step_um)
    if not math.isfinite(sample_step) or sample_step <= 0.0:
        raise ValueError("Sine sample step must be positive")
    minimum_segments = MIN_SINE_SEGMENTS_PER_CYCLE * period_count
    if segments is None:
        # L+4*A*N bounds the full curve length from above. Always retain at
        # least 16 samples per cycle, so zero crossings cannot alias a wave
        # into a straight line. The common planner densifies longer chords.
        segment_count = min(MAX_SINE_SEGMENTS, max(
            minimum_segments,
            int(math.ceil((length + 4.0 * amplitude * period_count) / sample_step)),
        ))
    else:
        segment_count = _bounded_integer(
            segments, "Sine segments", minimum_segments, MAX_SINE_SEGMENTS,
        )

    sign = -1.0 if inverted else 1.0
    outward = [start]
    for index in range(1, segment_count + 1):
        fraction = index / segment_count
        target = list(start)
        target[first_axis] += length / 1000.0 * fraction
        if index != segment_count:
            target[second_axis] += (sign * amplitude / 1000.0 *
                                    math.sin(2.0 * math.pi * period_count * fraction))
        outward.append(tuple(target))
    return outward[1:] + list(reversed(outward[:-1]))


def densify_targets(targets, max_step_mm=0.300, max_points=None):
    """Split each leg into full steps plus a remainder, retaining every vertex.

    Short legs stay intact: samples, corners and endpoints constrain spacing.
    In particular, a 0.4 mm leg with a 0.3 mm step becomes 0.3 + 0.1 mm,
    rather than being evenly divided into two 0.2 mm legs.
    """
    points = [tuple(float(value) for value in target) for target in targets]
    if not points or any(len(point) != 3 or
                         not all(math.isfinite(value) for value in point)
                         for point in points):
        raise ValueError("Trajectory targets must contain finite XYZ points")
    step = float(max_step_mm)
    if not math.isfinite(step) or step <= 0.0:
        raise ValueError("Waypoint spacing must be positive")
    result = [points[0]]
    for start, end in zip(points, points[1:]):
        distance = math.dist(start, end)
        count = max(1, int(math.ceil(distance / step)))
        if max_points is not None and len(result) + count > max_points:
            raise ValueError("Execution path exceeds %d waypoints" % max_points)
        for index in range(1, count + 1):
            travel = min(index * step, distance)
            if math.isclose(travel, distance, rel_tol=1e-12, abs_tol=1e-12):
                result.append(end)
                break
            fraction = travel / distance
            result.append(tuple(
                start[axis] + fraction * (end[axis] - start[axis])
                for axis in range(3)
            ))
    return result


def point_to_polyline_distance_mm(point, polyline):
    """Return the shortest Euclidean distance from a point to a 3D path.

    The distance is measured to each line segment, rather than only to the
    sampled waypoints.  This makes the trajectory error a geometric
    cross-track error against the planned path and avoids treating a moving
    point between two waypoints as an endpoint error.
    """
    candidate = tuple(float(value) for value in point)
    path = [tuple(float(value) for value in item) for item in polyline]
    if (len(candidate) != 3 or not all(math.isfinite(value) for value in candidate) or
            len(path) < 2 or any(len(item) != 3 or
                                  not all(math.isfinite(value) for value in item)
                                  for item in path)):
        raise ValueError("Point and planned path must contain finite XYZ coordinates")
    best_squared = float("inf")
    for start, end in zip(path, path[1:]):
        vector = tuple(end[index] - start[index] for index in range(3))
        offset = tuple(candidate[index] - start[index] for index in range(3))
        length_squared = sum(value * value for value in vector)
        if length_squared <= 1e-18:
            fraction = 0.0
        else:
            fraction = max(0.0, min(1.0, sum(
                offset[index] * vector[index] for index in range(3)
            ) / length_squared))
        squared = sum(
            (candidate[index] - start[index] - fraction * vector[index]) ** 2
            for index in range(3)
        )
        best_squared = min(best_squared, squared)
    return math.sqrt(best_squared)
