"""Authorized three-process CUDA continuation of the fixed P2 diagnostics."""
from __future__ import annotations

import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import shutil
import signal
import sys
import time

from lerobot_training.run_model_pipeline import digest, start_logged, stop_logged, write_json
from scripts.tools.experiment_workflow import ROOT, gate_result


def now():
    return datetime.now().astimezone().isoformat(timespec='seconds')


def record(output, status, detail):
    line = (f'\n- {now()}: P2 GPU parallel **{status}**; command '
            f'`{" ".join(sys.argv)}`; {detail}; output `{output}`. '
            'Fixed fast5 seeds42/44/48/50/52, checkpoint step_004000.pth; '
            'software/MuJoCo only.\n')
    with (ROOT / 'TEST_LOG.md').open('ab') as stream:
        stream.write(line.encode())


def read(path):
    return json.loads(path.read_text())


def validate_source(source):
    """Require complete matching teachers and all original pinned assets."""
    from scripts.research.control_g15_diagnostics import identities, FAST5
    source = source.resolve()
    state = read(source / 'workflow_status.json')
    plan = read(source / 'preflight.json')
    stages = {item['id']: item for item in plan['stages']}
    reused = [name for name in stages if name.startswith('s1_') or name.startswith('s2_')]
    assets = {}
    for stage in plan['stages']:
        assets.update(stage['assets'])
    for name, expected in assets.items():
        if digest(Path(name)) != expected:
            raise ValueError(f'Pinned asset changed: {name}')
    identity = identities()
    for name in reused:
        if state['stages'][name]['status'] != 'completed':
            raise ValueError(f'Cannot reuse unfinished stage: {name}')
        summary = read(source / name / 'summary.json')
        if summary['status'] != 'completed' or summary['identity'] != identity:
            raise ValueError(f'Incomplete or incompatible evidence: {name}')
        if name.startswith('s2_') and name != 's2_compare':
            rows = summary['episodes']
            if ([r['episode_index'] for r in rows] != list(FAST5)
                    or [r['seed'] for r in rows] != [42 + i for i in FAST5]
                    or any(r['status'] != 'completed' or r['steps'] != 1600 for r in rows)):
                raise ValueError(f'Teacher target/seed/budget differs: {name}')
    comparison = read(source / 's2_compare/summary.json')
    if not comparison['step3_allowed'] or not comparison['gate2b_passed']:
        raise ValueError('Teacher comparison did not pass the registered gate')
    if any(state['stages'][name]['status'] == 'completed'
           for name in ('s3_v0', 's3_v3', 's3_v2')):
        raise ValueError('A policy stage already completed; avoid duplicate evaluation')
    return {'source': str(source), 'source_id': plan['id'], 'identity': identity,
            'assets': assets, 'reuse_stages': reused,
            'policy_stages': [stages[name] for name in ('s3_v0', 's3_v3', 's3_v2')],
            'compare_stage': stages['s3_compare'], 'supervisor_pid': state['pid']}


def stop_sequential(source, expected_pid, output):
    """Stop only the identified source supervisor and its existing child."""
    state = read(source / 'workflow_status.json')
    pid = state['pid']
    if pid != expected_pid:
        raise ValueError('Supervisor changed since preflight')
    if state['status'] != 'running':
        raise ValueError('Source is no longer running; continuation needs review')
    cmdline = (Path('/proc') / str(pid) / 'cmdline').read_bytes()
    if b'scripts.tools.experiment_workflow' not in cmdline or str(source).encode() not in cmdline:
        raise ValueError('Supervisor process identity mismatch')
    print(f'Gracefully interrupting sequential supervisor PID {pid}; preserving {source}', flush=True)
    os.kill(pid, signal.SIGTERM)
    deadline = time.monotonic() + 25
    while time.monotonic() < deadline:
        state = read(source / 'workflow_status.json')
        if state['status'] != 'running':
            break
        time.sleep(.2)
    else:
        raise TimeoutError('Sequential supervisor did not stop; no parallel jobs launched')
    # The supervisor closes child processes before writing its final state.
    if state['status'] != 'interrupted':
        raise ValueError(f'Unexpected source stop status: {state["status"]}')
    write_json(output / 'source_interruption.json', state)
    record(output, 'source_interrupted',
           f'User authorized GPU parallel migration; source {source}; original files preserved')


def relocate(stage, output):
    stage = dict(stage)
    stage['argv'] = list(stage['argv'])
    stage['output_dir'] = str(output / stage['id'])
    index = stage['argv'].index('--output-dir') + 1
    stage['argv'][index] = stage['output_dir']
    if '--workflow-dir' in stage['argv']:
        index = stage['argv'].index('--workflow-dir') + 1
        stage['argv'][index] = str(output)
    return stage


def run(output, source, expected_pid):
    checked = validate_source(source)
    if checked['supervisor_pid'] != expected_pid:
        raise ValueError('Source supervisor mismatch')
    output.mkdir(parents=True, exist_ok=False)
    state = {'status': 'preparing', 'workers': 3, 'device': 'CUDA physical GPU0 RTX4090 / EGL',
             'source': str(source), 'stages': {}, 'started_at': now()}
    jobs = []

    def save():
        state['updated_at'] = now()
        write_json(output / 'status.json', state)

    def interrupt(signum, _frame):
        raise KeyboardInterrupt(f'received {signal.Signals(signum).name}')

    for signum in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
        signal.signal(signum, interrupt)
    save()
    try:
        stop_sequential(source, expected_pid, output)
        receipts = []
        for name in checked['reuse_stages']:
            original, destination = source / name, output / name
            shutil.copytree(original, destination)
            hashes = {}
            for path in sorted(original.rglob('*')):
                if path.is_file():
                    relative = path.relative_to(original)
                    expected = digest(path)
                    if digest(destination / relative) != expected:
                        raise ValueError(f'Reuse copy differs: {name}/{relative}')
                    hashes[str(relative)] = expected
            receipts.append({'stage': name, 'source': str(original), 'destination': str(destination),
                             'files_sha256': hashes, 'identity': checked['identity']})
        write_json(output / 'reuse_receipt.json', receipts)
        record(output, 'reuse_verified', f'Eight complete stages copied byte-for-byte from {source}; matching identity and teacher target coverage')
        state['status'] = 'running'
        for original in checked['policy_stages']:
            stage = relocate(original, output)
            name = stage['id']
            print(f'[{name}] CUDA GPU0 parallel; output {stage["output_dir"]}', flush=True)
            job = start_logged(stage['argv'], output / f'{name}.log', cwd=ROOT,
                               env=os.environ.copy(), label=name)
            job.update(stage=stage, started=time.monotonic())
            jobs.append(job)
            state['stages'][name] = {'status': 'running', 'pid': job['process'].pid,
                                     'started_at': now(), 'output_dir': stage['output_dir'],
                                     'timeout_s': stage['timeout_s']}
            save()
        pending = list(jobs)
        next_heartbeat = time.monotonic() + 30
        while pending:
            for job in list(pending):
                stage = job['stage']
                item = state['stages'][stage['id']]
                code = job['process'].poll()
                elapsed = time.monotonic() - job['started']
                if code is None and elapsed < stage['timeout_s']:
                    continue
                if code is None:
                    stop_logged([job], grace_seconds=10)
                    item.update(status='interrupted', reason=f'timed out after {stage["timeout_s"]} seconds')
                elif code:
                    item.update(status='failed', reason=f'child exit {code}')
                else:
                    job['reader'].join(timeout=1)
                    try:
                        summary = read(Path(stage['output_dir']) / 'summary.json')
                        if summary['status'] != 'completed' or summary['completed_targets'] != 5:
                            raise ValueError('Incomplete policy stage')
                        for gate in stage.get('gates', []):
                            passed, detail = gate_result(Path(stage['output_dir']), gate)
                            if not passed:
                                raise ValueError(detail)
                        item.update(status='completed', final_mean_um=summary['final_mean_um'],
                                    strict_success_count=summary['strict_success_count'])
                    except Exception as error:
                        item.update(status='failed', reason=str(error))
                item.update(finished_at=now(), elapsed_s=elapsed)
                pending.remove(job)
                record(output, item['status'], f'{stage["id"]}: {json.dumps(item)}')
                save()
            if time.monotonic() >= next_heartbeat:
                print('Parallel workers: ' + ', '.join(f'{name}={item["status"]}'
                      for name, item in state['stages'].items()), flush=True)
                next_heartbeat = time.monotonic() + 30
            if pending:
                time.sleep(.5)
        if any(item['status'] != 'completed' for item in state['stages'].values()):
            raise RuntimeError('Parallel policy stage incomplete; dependent comparison skipped, no retry')
        compare = relocate(checked['compare_stage'], output)
        job = start_logged(compare['argv'], output / 's3_compare.log', cwd=ROOT,
                           env=os.environ.copy(), label='s3_compare')
        jobs.append(job)
        try:
            code = job['process'].wait(timeout=compare['timeout_s'])
        finally:
            stop_logged([job], grace_seconds=10)
        if code:
            raise RuntimeError(f'Comparison exit {code}')
        result = read(output / 's3_compare/summary.json')
        if result['status'] != 'completed':
            raise ValueError('Comparison not completed')
        from scripts.research.control_g15_diagnostics import identities
        if identities() != checked['identity']:
            raise ValueError('Evaluation identity changed during parallel continuation')
        state.update(status='completed', comparison=str(output / 's3_compare/summary.json'),
                     best_variant=result['best_variant'], strict_success_not_regressed=result['strict_success_not_regressed'])
        write_json(output / 'summary.json', {**state, 'comparison': result,
                   'scope': 'Diagnostic fast5 only; parallel scheduling differs from prior sequential run; not hardware accuracy'})
    except BaseException as error:
        state.update(status='interrupted' if isinstance(error, KeyboardInterrupt) else 'failed',
                     reason=repr(error))
        raise
    finally:
        stop_logged(jobs, grace_seconds=10)
        for item in state['stages'].values():
            if item['status'] == 'running':
                item.update(status='interrupted', reason='parallel supervisor stopped')
        state['finished_at'] = now()
        save()
        record(output, state['status'], json.dumps(state))
        print(f'Parallel {state["status"]}: {output / "status.json"}', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-workflow', type=Path, required=True)
    parser.add_argument('--supervisor-pid', type=int, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    run(args.output_dir.resolve(), args.source_workflow.resolve(), args.supervisor_pid)


if __name__ == '__main__':
    main()
