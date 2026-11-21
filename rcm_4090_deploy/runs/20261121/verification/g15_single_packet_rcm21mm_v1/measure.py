"""Repeat the frozen single-packet diagnostic at a 21 mm tip-to-RCM radius."""
from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import sys
import traceback

ROOT = Path(__file__).resolve().parents[4]
BASELINE_DIR = ROOT / 'runs/20261121/verification/g15_single_packet_v1'


def save(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')


def record(message):
    with (ROOT / 'TEST_LOG.md').open('ab') as stream:
        stream.write(('\n- ' + datetime.now().astimezone().isoformat() +
                      '：G15 single-packet RCM21mm ' + message + '\n').encode())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    assert not (out/'preflight.json').exists(), 'Output already used'
    baseline = json.loads((BASELINE_DIR/'summary.json').read_text())
    baseline_preflight = json.loads((BASELINE_DIR/'preflight.json').read_text())
    helper = BASELINE_DIR/'measure.py'
    for name, digest in baseline_preflight['source_sha256'].items():
        assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest() == digest, name
    spec = importlib.util.spec_from_file_location('single_packet_probe', helper)
    probe = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(probe)
    identity = probe.driver_identity()
    assert identity == baseline['driver'], 'Driver differs from baseline'
    kin = probe.MotorBackend(needle_profile='legacy').kin
    source_home = probe.HOME
    requested_radius_mm = 21.
    # Legacy tip radius = rC/K -14 +4; set the actual integer encoder state.
    initial = (source_home[0], source_home[1],
               round((requested_radius_mm + 14. - 4.) * kin.ENCODER_COUNTS_PER_MM), 0)
    probe.HOME = initial
    # Reuse only the frozen integration function; the wrapper owns per-case
    # recording so the logged command correctly reproduces this 21 mm setup.
    probe.append_log = lambda message: None
    actual_radius_mm = math.sqrt(sum(v*v for v in kin.model_tip_from_robot_encoders(initial, 4., 1.)))
    assert abs(actual_radius_mm - requested_radius_mm) <= .5/kin.ENCODER_COUNTS_PER_MM
    specs = [('hold', 0), ('A', 1), ('A', -1), ('B', 1), ('B', -1), ('C', 1), ('C', -1)]
    command = '.venv/bin/python -E -B ' + str(Path(__file__).relative_to(ROOT)) + ' --output-dir ' + str(out.relative_to(ROOT))
    preflight = {'timestamp': datetime.now().astimezone().isoformat(), 'driver': identity,
                 'hypothesis': 'A/B displacement scales with tip-to-RCM radius at fixed posture; C step remains axial',
                 'main_change': 'initialize tip-to-RCM radius21mm instead of4mm; no control source changes',
                 'requested_tip_rcm_radius_mm': requested_radius_mm,
                 'actual_tip_rcm_radius_mm': actual_radius_mm, 'initial_robot_ABCE': initial,
                 'equivalent_C_insertion_from_default_home_mm': (initial[2]-source_home[2])/kin.ENCODER_COUNTS_PER_MM,
                 'initialization_scope': 'explicit simulator reset at workpoint; no17mm approach trajectory tested',
                 'baseline': str((BASELINE_DIR/'summary.json').relative_to(ROOT)),
                 'baseline_sha256': hashlib.sha256((BASELINE_DIR/'summary.json').read_bytes()).hexdigest(),
                 'helper_path': str(helper.relative_to(ROOT)), 'helper_sha256': hashlib.sha256(helper.read_bytes()).hexdigest(),
                 'wrapper_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                 'source_sha256': baseline_preflight['source_sha256'],
                 'cases': specs, 'needle_profile': 'legacy', 'tau_s': .04,
                 'speed_limits_A_deg_B_deg_C_mm_s': [10., 10., 2.], 'timeout_s': .5,
                 'integration_step_s': .002, 'observation_period_s': .05,
                 'max_steps_per_case': 250, 'max_simulation_time_s_per_case': .5,
                 'accepted_packets_per_case': 1, 'worker_count': 1, 'wall_budget_s': 30,
                 'dataset': None, 'checkpoint': None, 'training_rounds': 0, 'seed': None,
                 'tier': 'deterministic three-axis diagnostic, not navigation screen10/dev20',
                 'device': 'CPU small scalar/MuJoCo FK; no rendering or heavy tensors',
                 'gates': baseline_preflight['gates'][:-1],
                 'failure_policy': 'record partial scope and stop; no retries or parameter changes',
                 'command': command, 'output_dir': str(out)}
    save(out/'preflight.json', preflight)
    results = []
    try:
        model = probe.DriverModel(needle_profile='legacy')
        for axis, count in specs:
            result = probe.measure_case(model, axis, count, out)
            results.append(result)
            record(result['case'] + ' **PASS**；one packet at r=' + str(actual_radius_mm) +
                   'mm, initial=' + str(initial) + '；tip travel=' + str(result['tip_displacement_mujoco_um']) +
                   'um；encoder delta=' + str(result['robot_count_delta_ABCE']) + '；command `' + command +
                   '`；dataset/checkpoint N/A; CPU small scalar/MuJoCo FK；evidence `' +
                   str((out/(result['case']+'.json')).relative_to(ROOT)) + '`；software simulation only.')
        positive = {row['axis']: row for row in results if row['handle_count'] == 1}
        comparison = {}
        for axis in 'ABC':
            old = baseline['positive_one_handle'][axis]
            new = positive[axis]
            ratio = new['tip_displacement_mujoco_um'] / old['tip_displacement_mujoco_um']
            expected_ratio = new['tip_rcm_radius_before_mm']/old['tip_rcm_radius_before_mm'] if axis in 'AB' else 1.
            assert abs(ratio-expected_ratio) < 1e-6, (axis, ratio, expected_ratio)
            comparison[axis] = {'radius4mm_tip_step_um': old['tip_displacement_mujoco_um'],
                                'radius21mm_tip_step_um': new['tip_displacement_mujoco_um'],
                                'ratio': ratio, 'expected_ratio': expected_ratio}
        assert results[0]['tip_displacement_mujoco_um'] == 0.
        summary = {'status': 'PASS', 'completed_cases': len(results), 'driver': identity,
                   'requested_tip_rcm_radius_mm': requested_radius_mm,
                   'actual_tip_rcm_radius_mm': actual_radius_mm, 'initial_robot_ABCE': initial,
                   'positive_one_handle': positive, 'all_cases': results, 'comparison_to_radius4mm': comparison,
                   'minimum_tested_single_axis_displacement_um': min(r['tip_displacement_mujoco_um'] for r in results if r['axis']!='hold'),
                   'strict_0p35um_navigation_success': 'not evaluated',
                   'scope': 'fixed zero-pose A/B; tip radius21mm means C insertion~17mm, not C insertion21mm; software step size, not hardware accuracy',
                   'command': command, 'output_dir': str(out)}
        save(out/'summary.json', summary)
        record('**COMPLETED / PASS 7/7**；radius21mm vs prior4mm, fixed A/B posture, matched G15 source/dynamics/one-packet protocol; A/B scale check passed, C axial step unchanged；command `' + command + '`；dataset/checkpoint N/A; CPU small scalar/MuJoCo FK; no render/training/hardware or strict-navigation claim；summary `' + str((out/'summary.json').relative_to(ROOT)) + '`。')
        print(json.dumps({'status': 'PASS', 'comparison': comparison, 'result': str(out/'summary.json')}, ensure_ascii=False, indent=2))
    except BaseException as exc:
        save(out/'failure.json', {'status': 'FAIL', 'completed_cases': len(results),
                                 'error': str(exc), 'traceback': traceback.format_exc(), 'results': results})
        record('**FAIL/PARTIAL**；completed=' + str(len(results)) + '；reason=' + str(exc) +
               '；command `' + command + '`；dataset/checkpoint N/A；evidence `' +
               str((out/'failure.json').relative_to(ROOT)) + '`；no retry.')
        raise


if __name__ == '__main__':
    main()
