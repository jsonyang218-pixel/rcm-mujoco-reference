"""Seeded RRT/PRM in normalized RCM task space and three-goal order selection.

Validity covers linkage geometry and the tip in the reference eye ellipsoid.
It is not a finite-radius shaft or tissue collision checker.
"""
from __future__ import annotations

import heapq
import itertools
import math

import numpy as np

from rcm_navigation import inside_eye, task_si_units
from rcm_fundus import clearance_lower_bound_mm, standoff_task


def move_duration(q_start, q_end, settings):
    """Quintic rest-to-rest duration; inputs use degrees, degrees, millimeters."""
    delta = np.abs(np.asarray(q_end)-q_start)
    if np.max(delta) < 1e-12:
        return 0.0
    vmax = np.asarray(settings["max_velocity_A_deg_B_deg_C_mm_per_s"])
    amax = np.asarray(settings["max_acceleration_A_deg_B_deg_C_mm_per_s2"])
    return max(float(settings["minimum_move_time_s"]),
               float(np.max(1.875*delta/vmax)),
               float(np.max(np.sqrt((10*math.sqrt(3)/3)*delta/amax))))


class TrapezoidMotion:
    """Synchronized, continuous-velocity straight motion with bounded acceleration.

    Jerk is not constrained. Inputs/outputs use degree, degree, millimeter.
    """
    def __init__(self, start, end, settings):
        self.start, self.end = np.asarray(start), np.asarray(end)
        self.delta = self.end-self.start
        active = np.abs(self.delta) > 1e-12
        if not active.any():
            self.acceleration = self.speed = self.accel_time_s = self.cruise_time_s = self.duration_s = 0.
            return
        vmax = np.asarray(settings["max_velocity_A_deg_B_deg_C_mm_per_s"])
        amax = np.asarray(settings["max_acceleration_A_deg_B_deg_C_mm_per_s2"])
        self.acceleration = float(np.min(amax[active]/np.abs(self.delta[active])))
        self.speed = min(float(np.min(vmax[active]/np.abs(self.delta[active]))), math.sqrt(self.acceleration))
        self.accel_time_s = self.speed/self.acceleration
        self.cruise_time_s = max(0., (1-self.speed*self.speed/self.acceleration)/self.speed)
        self.duration_s = 2*self.accel_time_s+self.cruise_time_s

    def sample(self, time_s):
        t = max(0., min(float(time_s), self.duration_s))
        if t >= self.duration_s:
            return self.end.copy(), np.zeros(3), np.zeros(3)
        if t < self.accel_time_s:
            s, speed, acceleration = .5*self.acceleration*t*t, self.acceleration*t, self.acceleration
        elif t < self.accel_time_s+self.cruise_time_s:
            s = .5*self.speed*self.accel_time_s+self.speed*(t-self.accel_time_s)
            speed, acceleration = self.speed, 0.
        else:
            remaining = self.duration_s-t
            s, speed, acceleration = 1-.5*self.acceleration*remaining**2, self.acceleration*remaining, -self.acceleration
        return self.start+s*self.delta, speed*self.delta, acceleration*self.delta


def shortcut_path(space, path):
    """Remove avoidable corners only when the entire replacement edge is valid."""
    result, index = [np.asarray(path[0])], 0
    while index < len(path)-1:
        for candidate in range(len(path)-1, index, -1):
            if space.edge_valid(space.normalize(path[index]), space.normalize(path[candidate])):
                index = candidate
                if not np.allclose(result[-1], path[index], atol=1e-12, rtol=0):
                    result.append(np.asarray(path[index]))
                break
        else:
            raise RuntimeError("PATH_SHORTCUT_HAS_NO_VALID_EDGE")
    return result


def select_visit_order(start, solutions, settings):
    """Select an order by a shared time surrogate, without a return leg.

    Small target sets use the exact permutation search used by the original
    three-point experiment.  Larger random episodes use a deterministic
    nearest-next heuristic so configurable N does not explode factorially.
    """
    policy = settings.get("fundus_transition")

    def leg_duration(previous, solution):
        goal = np.asarray(solution["requested_task_A_deg_B_deg_C_mm"])
        if policy:
            depart = standoff_task(previous, policy["retract_mm"])
            arrive = standoff_task(goal, policy["retract_mm"])
            pairs = [(previous, depart), (depart, arrive), (arrive, goal)]
        else:
            pairs = [(previous, goal)]
        return sum(TrapezoidMotion(a, b, settings).duration_s
                   if settings.get("waypoint_trajectory") == "trapezoid" else move_duration(a, b, settings)
                   for a, b in pairs)

    if len(solutions) > 8:
        remaining = list(solutions)
        previous = np.asarray(start)
        selected_sequence = []
        durations = []
        while remaining:
            next_solution = min(remaining, key=lambda item: (
                leg_duration(previous, item), item["id"]))
            duration = leg_duration(previous, next_solution)
            selected_sequence.append(next_solution)
            durations.append(duration)
            previous = np.asarray(next_solution["requested_task_A_deg_B_deg_C_mm"])
            remaining.remove(next_solution)
        selected_order = [s["id"] for s in selected_sequence]
        return selected_sequence, {
            "objective": "deterministic nearest-next surrogate",
            "scope": "greedy heuristic for more than eight targets; exact permutation search for eight or fewer",
            "return_to_start": False, "input_order": [s["id"] for s in solutions],
            "selected_order": selected_order,
            "selected_estimated_motion_time_s": float(sum(durations)),
            "selected_estimated_leg_times_s": durations,
            "candidates": [{"order": selected_order,
                            "estimated_motion_time_s": float(sum(durations)),
                            "estimated_leg_times_s": durations}],
        }
    candidates = []
    for sequence in itertools.permutations(solutions):
        previous = np.asarray(start)
        durations = []
        for solution in sequence:
            goal = np.asarray(solution["requested_task_A_deg_B_deg_C_mm"])
            if policy:
                depart = standoff_task(previous, policy["retract_mm"]) if durations else previous
                arrive = standoff_task(goal, policy["retract_mm"])
                pairs = [(previous, depart), (depart, arrive), (arrive, goal)]
            else:
                pairs = [(previous, goal)]
            durations.append(sum(TrapezoidMotion(a, b, settings).duration_s
                                 if settings.get("waypoint_trajectory") == "trapezoid" else move_duration(a, b, settings)
                                 for a, b in pairs))
            previous = goal
        candidates.append({"order": [s["id"] for s in sequence],
                           "estimated_motion_time_s": sum(durations),
                           "estimated_leg_times_s": durations})
    candidates.sort(key=lambda item: (item["estimated_motion_time_s"], item["order"]))
    selected = candidates[0]
    by_id = {s["id"]: s for s in solutions}
    return [by_id[key] for key in selected["order"]], {
        "objective": ("sum of axial retract, interior transfer and axial approach durations" if settings.get("fundus_transition") else
                      "sum of synchronized trapezoidal rest-to-rest move durations"
                      if settings.get("waypoint_trajectory") == "trapezoid" else "sum of straight task-space quintic rest-to-rest move durations"),
        "scope": "exact minimum of this surrogate over permutations; not optimal RRT/PRM or visual trajectory time",
        "return_to_start": False,
        "input_order": [s["id"] for s in solutions],
        "selected_order": selected["order"],
        "selected_estimated_motion_time_s": selected["estimated_motion_time_s"],
        "candidates": candidates,
    }


class TaskSpace:
    def __init__(self, reference, tasks):
        self.reference = reference
        tasks = np.asarray(tasks, dtype=float)
        margin = np.array([8., 8., 2.])
        self.low = tasks.min(axis=0)-margin
        self.high = tasks.max(axis=0)+margin
        self.low[2] = max(self.low[2], -reference.tip_reference_mm+0.1)
        self.span = self.high-self.low
        self.edge_resolution = np.array([0.25, 0.25, 0.05])
        self.validity_checks = 0
        self.minimum_clearance_mm = 0.

    def normalize(self, q):
        return (np.asarray(q)-self.low)/self.span

    def denormalize(self, point):
        return self.low+np.asarray(point)*self.span

    def valid(self, point):
        self.validity_checks += 1
        if not np.isfinite(point).all() or np.any(point < -1e-10) or np.any(point > 1+1e-10):
            return False
        try:
            q = self.reference.validate_task(task_si_units(self.denormalize(point)))
            robot = self.reference.nominal.forward(self.reference.state_from_task(q), enforce_limits=False)
            return bool(inside_eye(self.reference, self.reference.robot_to_world(robot))[0]
                        and (self.minimum_clearance_mm == 0. or
                             clearance_lower_bound_mm(self.reference, self.denormalize(point)) >= self.minimum_clearance_mm-1e-8))
        except ValueError:
            return False

    def edge_valid(self, start, end):
        count = max(1, math.ceil(float(np.max(np.abs(end-start)*self.span/self.edge_resolution))))
        return all(self.valid(start+(end-start)*(i/count)) for i in range(count+1))


def rrt(space, start, goal, rng, max_iterations=5000, step_size=0.12, goal_bias=0.15):
    start, goal = space.normalize(start), space.normalize(goal)
    if not space.valid(start) or not space.valid(goal):
        raise ValueError("RRT_INVALID_ENDPOINT")
    nodes, parents = [start], [-1]
    for iteration in range(max_iterations):
        sample = goal if rng.random() < goal_bias else rng.random(3)
        distances = np.linalg.norm(np.asarray(nodes)-sample, axis=1)
        nearest = int(np.argmin(distances))
        delta = sample-nodes[nearest]
        length = float(np.linalg.norm(delta))
        if length < 1e-12:
            continue
        candidate = nodes[nearest]+delta*min(1., step_size/length)
        if not space.edge_valid(nodes[nearest], candidate):
            continue
        nodes.append(candidate)
        parents.append(nearest)
        if np.linalg.norm(candidate-goal) <= step_size and space.edge_valid(candidate, goal):
            path = [goal]
            index = len(nodes)-1
            while index >= 0:
                path.append(nodes[index])
                index = parents[index]
            path.reverse()
            return [space.denormalize(p) for p in path], {
                "iterations": iteration+1, "tree_nodes": len(nodes),
                "step_size_normalized": step_size, "goal_bias": goal_bias,
            }
    raise RuntimeError("RRT_SEARCH_EXHAUSTED")


class PRM:
    def __init__(self, space, queries, rng, samples=160, neighbors=12):
        self.space = space
        self.nodes = [space.normalize(q) for q in queries]
        if not all(space.valid(node) for node in self.nodes):
            raise ValueError("PRM_INVALID_ENDPOINT")
        for _ in range(samples*100):
            if len(self.nodes) >= samples+len(queries):
                break
            candidate = rng.random(3)
            if space.valid(candidate):
                self.nodes.append(candidate)
        if len(self.nodes) < samples+len(queries):
            raise RuntimeError("PRM_SAMPLING_EXHAUSTED")
        self.nodes = np.asarray(self.nodes)
        distances = np.linalg.norm(self.nodes[:, None, :]-self.nodes[None, :, :], axis=2)
        pairs = set()
        for index in range(len(self.nodes)):
            for other in np.argsort(distances[index])[1:neighbors+1]:
                pairs.add(tuple(sorted((index, int(other)))))
        self.graph = [[] for _ in self.nodes]
        edges = 0
        for i, j in sorted(pairs):
            if space.edge_valid(self.nodes[i], self.nodes[j]):
                distance = float(distances[i, j])
                self.graph[i].append((j, distance))
                self.graph[j].append((i, distance))
                edges += 1
        self.info = {"roadmap_nodes": len(self.nodes), "roadmap_edges": edges,
                     "random_samples": samples, "neighbors": neighbors,
                     "edge_cost": "Euclidean distance in normalized task space",
                     "reuse": "one roadmap shared by all three legs"}

    def query(self, start_index, goal_index):
        costs = {start_index: 0.}
        parents = {start_index: None}
        queue = [(0., start_index)]
        while queue:
            cost, current = heapq.heappop(queue)
            if cost > costs[current]:
                continue
            if current == goal_index:
                path = []
                while current is not None:
                    path.append(self.space.denormalize(self.nodes[current]))
                    current = parents[current]
                return path[::-1]
            for other, distance in self.graph[current]:
                new_cost = cost+distance
                if new_cost < costs.get(other, math.inf):
                    costs[other], parents[other] = new_cost, current
                    heapq.heappush(queue, (new_cost, other))
        raise RuntimeError("PRM_NO_CONNECTED_PATH")
