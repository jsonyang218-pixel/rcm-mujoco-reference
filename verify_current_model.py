"""Check the repository's default model and CLI from outside the checkout."""
from pathlib import Path
import hashlib
import json
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET

import mujoco

HERE = Path(__file__).resolve().parent


def run(cwd, *arguments):
    result = subprocess.run([sys.executable, str(HERE / 'run_reference.py'), *arguments],
                            cwd=cwd, text=True, capture_output=True, timeout=90)
    if result.returncode:
        raise RuntimeError(f'{arguments}: {result.stderr[-2000:]}')
    return json.loads(result.stdout)


def check_scene(path):
    model = mujoco.MjModel.from_xml_path(str(path))
    assert (model.nq, model.nu, model.neq) == (7, 3, 4)
    for name in ('A', 'B', 'C', 'silver_B_linear'):
        assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name) >= 0, name
    for name in ('scan_fixed_housing', 'scan_parallel_cover'):
        assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name) >= 0, name
    assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, 'A_azimuth') == -1


def main():
    before = hashlib.sha256((HERE / 'scene.xml').read_bytes()).hexdigest()
    check_scene(HERE / 'scene.xml')
    with tempfile.TemporaryDirectory(prefix='current-model-entry-') as work:
        home = run(work, '--no-viewer')
        assert home['profile'] == 'new' and home['installation_pitch_y_deg'] == -40
        assert home['max_fk_error_m'] < 1e-9
        for arguments, expected in (
            (('--pose', '4', '3', '1'), (4, 3, 1)),
            (('--encoders', '0', '1466018', '893518'), (0, 0, 0)),
            (('--demo', 'abc', '--sample-time', '2'), (8, 5, 2)),
        ):
            result = run(work, '--no-viewer', *arguments)
            actual = (result['A_deg'], result['B_relative_deg'], result['C_axial_mm'])
            assert all(abs(a-b) < .003 for a, b in zip(actual, expected)), (actual, expected)
            assert result['max_fk_error_m'] < 1e-9
        export = Path(work) / 'nested' / 'current.xml'
        run(work, '--no-viewer', '--export', str(export))
        check_scene(export)
        legacy_export = Path(work) / 'legacy_needle.xml'
        run(work, '--no-viewer', '--profile', 'legacy', '--export', str(legacy_export))
        check_scene(legacy_export)
        tip = ET.parse(legacy_export).find(".//site[@name='needle_tip']")
        assert float(tip.get('pos').split()[2]) == .004
        rejected = subprocess.run([sys.executable, str(HERE / 'run_reference.py'), '--auto-reach'],
                                  cwd=work, text=True, capture_output=True, timeout=30)
        assert rejected.returncode == 2 and 'run_legacy_reference.py' in rejected.stderr
    assert hashlib.sha256((HERE / 'scene.xml').read_bytes()).hexdigest() == before
    print(json.dumps({'status': 'PASS', 'default_model': 'v9 scanned arm',
        'root_scene_loads': True, 'headless_entry_from_other_cwd': True,
        'pose_encoders_and_demo_units': True, 'exported_mesh_paths_load': True,
        'both_needle_profiles_exported': True, 'legacy_flags_redirected': True,
        'default_scene_not_overwritten_by_other_exports': True,
        'live_desktop_GUI_tested': False}, indent=2))


if __name__ == '__main__':
    main()
