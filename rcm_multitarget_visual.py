"""Strict measurement-driven fundus visit phases and aligned episode recording.

Scene coordinates are consumed only by ``set_target_world`` and independent
post-arrival evaluation. No scene target inverse or target joint path is built.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
import time

import numpy as np

from rcm_fundus import no_surface_slide_edge
from rcm_navigation import navigation_settings, task_user_units, write_json
from rcm_visual_servo import VisualServoController, VisualServoSensor, visual_servo_settings


def alignment_goal(target, tip, rcm):
    """Measured target direction at the currently measured radial tip distance."""
    target, tip, rcm = map(lambda value: np.asarray(value, dtype=float), (target, tip, rcm))
    direction = target-rcm
    distance = float(np.linalg.norm(direction))
    radius = float(np.linalg.norm(tip-rcm))
    if target.shape != (3,) or tip.shape != (3,) or rcm.shape != (3,) or not np.isfinite([target, tip, rcm]).all():
        raise RuntimeError('INVALID_VISUAL_ALIGNMENT_MEASUREMENT')
    if distance < 1e-6 or radius < 1e-6:
        raise RuntimeError('UNOBSERVABLE_VISUAL_ALIGNMENT_DIRECTION')
    return rcm+radius*direction/distance


def measured_retraction_goal(geometry, retract_mm):
    """A virtual destination from one stopped visual tip/shaft observation."""
    tip = np.asarray(geometry.get('tip_world_mm'), dtype=float)
    axis = np.asarray(geometry.get('tip_quality', {}).get('axis_world'), dtype=float)
    if tip.shape != (3,) or axis.shape != (3,) or not np.isfinite([tip, axis]).all():
        raise RuntimeError('RETRACTION_REQUIRES_OBSERVED_TIP_AND_SHAFT')
    norm = float(np.linalg.norm(axis))
    if norm < 1e-6:
        raise RuntimeError('RETRACTION_REQUIRES_OBSERVED_SHAFT_DIRECTION')
    return tip-float(retract_mm)*axis/norm


def _perception(reference, observation, phase):
    geometry, tracking = observation.get('geometry', {}), observation.get('tracking', {})
    tip = geometry.get('tip_world_mm')
    target = tracking.get('used_target_world_mm', geometry.get('target_world_mm'))
    estimate = tracking.get('estimated_position_error_um')
    if estimate is None and tip is not None and target is not None:
        estimate = float(np.linalg.norm(np.asarray(target)-tip)*1000.)
    observed = observation.get('target_pixel') is not None and geometry.get('target_world_mm') is not None
    intervention = observation.get('occlusion_intervention', {})
    return {'target_source': tracking.get('target_source', 'observed' if observed else 'unavailable'),
            'tip_source': geometry.get('tip_quality', {}).get('source', 'unavailable'),
            'controller_state': tracking.get('controller_state', phase),
            'cached_target_age_s': tracking.get('cached_target_age_s'),
            'estimated_position_error_um': estimate,
            'true_position_error_um': float(reference.metrics()['tip_target_error_um']),
            'target_observed': bool(tracking.get('target_observed', observed)),
            'control_frame_index': observation.get('experiment_frame_index'),
            'occlusion_event_id': str(intervention.get('event_id') or 'unknown')}


def run_strict_visual(reference, scene_solutions, directory, writer, fixture=None, *,
                      retract_mm=None, minimum_transfer_clearance_mm=None,
                      controller_factory=None, sensor_factory=None):
    """Visit input-ordered markers with measured retract/align/approach phases.

    ``scene_solutions`` needs only ``id`` and ``requested_world_mm``. The input
    position is used to place each marker, never as a movement destination.
    Retraction keeps the previous marker and dataset waypoint ID so replayed
    images retain the same scene. Arrivals are evaluated before any retraction.

    Returns the same ``(frames, target_results, info)`` shape as run_visual.
    ``writer.finish`` remains the caller's responsibility.
    """
    if not scene_solutions:
        raise ValueError('Strict visual navigation requires at least one scene marker')
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    settings = navigation_settings(reference)
    protocol = settings.get('strict_visual_navigation', {})
    retract_mm = float(protocol.get('retract_mm', 2.) if retract_mm is None else retract_mm)
    clearance = float(protocol.get('minimum_transfer_clearance_mm', 1.)
                      if minimum_transfer_clearance_mm is None else minimum_transfer_clearance_mm)
    if not np.isfinite([retract_mm, clearance]).all() or retract_mm <= 0. or clearance < 0.:
        raise ValueError('Retraction must be positive and transfer clearance nonnegative')
    factory = controller_factory or VisualServoController
    make_sensor = sensor_factory or VisualServoSensor
    base = visual_servo_settings(reference)
    base.update(position_feedback='geometry', use_target_pose_calibration=False,
                require_settled=True, settle_frames=max(3, int(base.get('settle_frames', 3))))
    dt = float(base['sample_period_s'])
    frames, results, details, transitions = [], [], [], []
    elapsed = computation = 0.
    velocity, max_velocity, max_acceleration = np.zeros(3), np.zeros(3), np.zeros(3)
    pending = None
    frame_counts = {}
    last_geometry = None
    last_marker_id = None
    failure = None
    rcm = np.asarray(reference.rcm_mm, dtype=float).copy()

    def append_frame(timestamp, q, marker_id, phase):
        item = {'time_s': float(timestamp), 'q': np.asarray(q).copy(),
                'waypoint_id': marker_id, 'phase': phase}
        if frames and abs(item['time_s']-frames[-1]['time_s']) < 1e-10:
            if not np.allclose(item['q'], frames[-1]['q'], atol=1e-10, rtol=0):
                raise RuntimeError('DISCONTINUOUS_VISUAL_PHASE_BOUNDARY')
            # Preserve the current phase label while retaining one physical instant.
            frames[-1] = item
        else:
            frames.append(item)

    def emit(sample, action, terminal=False):
        writer.add(sample['timestamp'], sample['q'], action, sample['marker_id'], sample['rgb'],
                   terminal=terminal, phase=sample['phase'], visibility=sample['visibility'],
                   perception=sample['perception'])

    def hook(marker_id, phase, clock=None):
        def update(iteration, state):
            frame = frame_counts.get(marker_id, 0)
            frame_counts[marker_id] = frame+1
            intervention = None
            if fixture is not None:
                intervention = fixture.update(marker_id, phase, frame)
            if clock is not None:
                clock['experiment_frame_index'] = frame
                clock['occlusion_intervention'] = intervention or {}
                clock['phase_time_s'] = float(state.get('time_s', iteration*dt))
        return update

    def guard(a, b):
        return no_surface_slide_edge(reference, task_user_units(a), task_user_units(b), clearance)

    def run_phase(marker_id, phase, axes, *, goal=None, transform=None, transition_to=None):
        nonlocal pending, elapsed, computation, velocity, max_velocity, max_acceleration, last_geometry
        phase_start = elapsed
        phase_settings = {**base, 'position_tolerance_um': 1. if phase == 'visual_align' else 8.}
        controller = factory(reference, settings=phase_settings,
                             output_dir=directory/marker_id/phase,
                             initial_velocity=velocity, command_validator=guard,
                             control_axes=axes, completion_axes=axes,
                             measurement_goal=goal, target_transform=transform,
                             independent_endpoint_acceptance=phase == 'visual_approach')
        clock = {}
        controller.sensor.observation_time_origin_s = phase_start
        controller.observation_hook = hook(marker_id, phase, clock)
        last_captured_observation = None

        def capture(iteration, observation, *, sample_time=None, visibility_sensor=None):
            nonlocal pending, last_geometry, last_captured_observation
            observation.update(clock)
            q = np.asarray(task_user_units(reference.current_task()))
            if pending is not None:
                emit(pending, q)
                pending = None
            row_time = (float(controller.rows[-1]['time_s']) if controller.rows else iteration*dt)
            if sample_time is not None:
                row_time = float(sample_time)
            timestamp = phase_start+row_time
            if writer.records:
                timestamp = max(timestamp, float(writer.records[-1]['timestamp'])+1e-9)
            visibility = (visibility_sensor or controller.sensor).visibility_metrics(observation.get('segmentation'))
            pending = {'timestamp': timestamp, 'q': q.copy(), 'marker_id': marker_id,
                       'phase': phase, 'rgb': observation['rgb'].copy(), 'visibility': visibility,
                       'perception': _perception(reference, observation, phase)}
            last_geometry = observation.get('geometry', {})
            last_captured_observation = observation
            append_frame(phase_start+row_time, q, marker_id, phase)
            if iteration % 100 == 0:
                print(json.dumps({'waypoint': marker_id, 'phase': phase, 'iteration': iteration,
                                  'estimated_position_error_um': pending['perception']['estimated_position_error_um']},
                                 ensure_ascii=False), flush=True)
            return True

        controller.iteration_callback = capture
        started = time.perf_counter()
        try:
            report = controller.run()
        except (RuntimeError, ValueError, np.linalg.LinAlgError) as exc:
            controller.sensor.close()
            report = {'result': 'FAIL', 'reason': str(exc), 'controller_result': 'FAIL',
                      'terminal_stop_validated': False, 'iterations': len(controller.rows)}
        wall = time.perf_counter()-started
        computation += wall
        terminal_observation_recorded = False
        final_observation = getattr(controller, 'final_observation', None)
        if (report['result'] != 'PASS' and final_observation is not None
                and final_observation is not last_captured_observation):
            # Loss can occur before the controller's normal callback. Retain
            # that actual image with the same intervention clock; do not
            # advance or hide the fixture merely to obtain a clean terminal.
            audit_sensor = make_sensor(reference, phase_settings)
            try:
                capture(int(clock.get('experiment_frame_index', len(controller.rows))), final_observation,
                        sample_time=clock.get('phase_time_s', elapsed-phase_start),
                        visibility_sensor=audit_sensor)
                terminal_observation_recorded = True
            finally:
                audit_sensor.close()
        controller.write(report, directory/marker_id/phase)
        duration = float(controller.rows[-1]['time_s']) if controller.rows else 0.
        if terminal_observation_recorded:
            duration = max(duration, float(clock.get('phase_time_s', duration)))
        final_q = np.asarray(task_user_units(reference.current_task()))
        last_row_q = None if not controller.rows else np.asarray(
            [controller.rows[-1][key] for key in ('A_deg', 'B_deg', 'C_mm')])
        if (not terminal_observation_recorded and last_row_q is not None
                and not np.allclose(final_q, last_row_q, atol=1e-10, rtol=0)):
            # A final issued command may not have a following visual observation.
            if pending is not None:
                emit(pending, final_q)
                pending = None
            duration += dt
            append_frame(phase_start+duration, final_q, marker_id, phase)
        elapsed = phase_start+duration
        velocity = np.asarray(controller.final_velocity).copy()
        max_velocity = np.maximum(max_velocity, controller.max_velocity)
        max_acceleration = np.maximum(max_acceleration, controller.max_acceleration)
        last_geometry = report.get('final_geometry_measurement', last_geometry)
        detail = {**report, 'waypoint_id': marker_id, 'phase': phase,
                  'start_s': phase_start, 'duration_s': duration, 'algorithm_wall_s': wall,
                  'virtual_goal_source': ('observed_tip_and_shaft' if goal is not None else
                                          'observed_marker_direction_and_current_tip_radius' if transform else
                                          'current_or_bounded_history_target_observation'),
                  'transition_to': transition_to}
        detail['terminal_failure_observation_recorded'] = terminal_observation_recorded
        details.append(detail)
        if phase == 'visual_retract':
            transitions.append(detail)
        return report

    try:
        for index, scene in enumerate(scene_solutions):
            marker_id = str(scene['id'])
            waypoint_start, wall_start = elapsed, computation
            if index:
                try:
                    goal = measured_retraction_goal(last_geometry or {}, retract_mm)
                    report = run_phase(last_marker_id, 'visual_retract', (False, False, True),
                                       goal=goal, transition_to=marker_id)
                except RuntimeError as exc:
                    report = {'result': 'FAIL', 'reason': str(exc), 'terminal_stop_validated': False}
                results[-1]['transition_to_next_result'] = report['result']
                if report['result'] != 'PASS':
                    failure = {'waypoint_id': last_marker_id, 'phase': 'visual_retract', 'reason': report['reason']}
                    break
            # Scene intervention only: never pass this coordinate to a controller.
            reference.set_target_world(scene['requested_world_mm'])
            last_marker_id = marker_id
            if fixture is not None:
                fixture.hide('waypoint_marker_changed')
            print(json.dumps({'waypoint': marker_id, 'event': 'strict_visual_begin',
                              'dataset_frames': len(writer.records)}, ensure_ascii=False), flush=True)
            report = run_phase(marker_id, 'visual_align', (True, True, False),
                               transform=lambda target, tip, observation: alignment_goal(target, tip, rcm))
            if report['result'] == 'PASS':
                report = run_phase(marker_id, 'visual_approach', (False, False, True))
            # Independent truth belongs only to this arrival evaluation, before retracting.
            error = float(reference.metrics()['tip_target_error_um'])
            stopped = bool(report.get('terminal_stop_validated', False))
            passed = report['result'] == 'PASS' and error < 20. and stopped
            result = {'waypoint_id': marker_id, 'result': 'PASS' if passed else 'FAIL',
                      'reason': report['reason'] if report['result'] != 'PASS' else
                                'VISUAL_ARRIVAL_AND_STOP' if passed else 'INDEPENDENT_ARRIVAL_REJECTED',
                      'tip_error_um': error, 'motion_time_s': elapsed-waypoint_start,
                      'algorithm_wall_s': computation-wall_start,
                      'terminal_stop_validated': stopped, 'success_with_stop': passed,
                      'arrival_task_A_deg_B_deg_C_mm': task_user_units(reference.current_task()),
                      'arrival_geometry_measurement': last_geometry}
            results.append(result)
            print(json.dumps({'waypoint': marker_id, 'event': 'strict_visual_end', 'result': result['result'],
                              'tip_error_um': error}, ensure_ascii=False), flush=True)
            if not passed:
                failure = {'waypoint_id': marker_id, 'phase': details[-1]['phase'], 'reason': result['reason']}
                break
        completed = {row['waypoint_id'] for row in results}
        for scene in scene_solutions:
            if str(scene['id']) not in completed:
                results.append({'waypoint_id': str(scene['id']), 'result': 'NOT_RUN',
                                'reason': 'PREVIOUS_VISUAL_PHASE_FAILED', 'tip_error_um': None,
                                'motion_time_s': 0., 'algorithm_wall_s': 0.,
                                'terminal_stop_validated': False, 'success_with_stop': False})
        if pending is not None:
            q = np.asarray(task_user_units(reference.current_task()))
            if not np.allclose(q, pending['q'], atol=1e-10, rtol=0):
                raise RuntimeError('UNRECORDED_FINAL_VISUAL_COMMAND')
            # The last state is an actual held observation, not a fabricated frame.
            emit(pending, q, terminal=True)
            pending = None
        elif last_marker_id is not None:
            sensor = make_sensor(reference, base)
            try:
                clock = {}
                source_phase = details[-1]['phase'] if details else 'visual_failure_hold'
                hook(last_marker_id, source_phase, clock)(0, {})
                observation = sensor.observe()
                observation.update(clock)
                q = np.asarray(task_user_units(reference.current_task()))
                timestamp = max(elapsed, float(writer.records[-1]['timestamp'])+dt if writer.records else elapsed)
                sample = {'timestamp': timestamp, 'q': q, 'marker_id': last_marker_id,
                          'phase': source_phase, 'rgb': observation['rgb'],
                          'visibility': sensor.visibility_metrics(observation.get('segmentation')),
                          'perception': _perception(reference, observation, 'visual_failure_hold')}
                emit(sample, q, terminal=True)
                append_frame(timestamp, q, last_marker_id, source_phase)
            finally:
                sensor.close()
    finally:
        if fixture is not None:
            fixture.hide('strict_visual_complete')
    scale = np.array([180/math.pi, 180/math.pi, 1000.])
    info = {'algorithm_wall_s': computation, 'execution_mode': 'strict_visual_multitarget',
            'known_target_pose_used_for_feature_calibration': False,
            'ground_truth_used_for_fundus_transfer': False,
            'visual_feedback_scope': 'Measured shaft/tip retraction, measured target-direction alignment, and measured axial approach throughout.',
            'fundus_command_guard_enabled': True, 'eye_containment_guard_enabled': True,
            'max_velocity': (max_velocity*scale).tolist(),
            'max_acceleration': (max_acceleration*scale).tolist(),
            'terminal_velocity': (velocity*scale).tolist(),
            'terminal_stop_validated': bool(results and all(r['success_with_stop'] for r in results)),
            'terminal_stop_scope': 'Every successful visual phase requires consecutive zero-command-velocity observations; abort holds are not physical brakes.',
            'control_details': details, 'transition_details': transitions,
            'failure': failure, 'retract_mm': retract_mm,
            'minimum_transfer_clearance_mm': clearance,
            'input_visit_order_preserved': True,
            'fixture_visual_frame_counts': frame_counts,
            'controlled_occlusion_fixture': None if fixture is None else fixture.audit,
            'occlusion_handling': base.get('occlusion_handling', 'stop')}
    write_json(directory/'strict_visual_protocol.json', info)
    return frames, results, info
