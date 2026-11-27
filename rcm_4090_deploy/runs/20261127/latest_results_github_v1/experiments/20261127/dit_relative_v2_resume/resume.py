"""Resume the frozen v2 experiment, preserving complete real branches and gates."""
from __future__ import annotations

import argparse
import concurrent.futures
import fcntl
import json
import math
import os
from pathlib import Path
import selectors
import signal
import subprocess
import sys
import threading
import time
import traceback

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[2]
OLD = ROOT / 'runs/20261126/dit_relative_terminal_v2'
sys.path.insert(0, str(OUT))
import campaign as c

PYTHON = str(ROOT / '.venv/bin/python')
STOP = False
ACTIVE = []
ATTEMPT = None
DEADLINE = None
START = None


def stop(signum, frame):
    global STOP
    STOP = True


def rows(path):
    with Path(path).open() as stream:
        return [json.loads(line) for line in stream if line.strip()]


def branch_check(folder, spec, root):
    """Require complete provenance, all frames, action alignment, and RGB hashes."""
    summary = c.read(folder / 'summary.json')
    assert summary['status'] == 'completed' and summary['branch'] == spec
    n = spec['teacher_steps']
    assert summary['teacher_actions'] == n and summary['observations'] == n + 3
    assert summary['real_sensor_calls'] == n + 3
    assert summary['final_post_RGB_OCT_saved'] and summary['no_midbranch_resets']
    assert summary['neural_action_calls'] == 0
    obs = rows(folder / 'observations.jsonl')
    actions = rows(folder / 'actions.jsonl')
    trajectory = c.read(folder / 'teacher_trajectory.json')
    assert trajectory['early_stop_reason'] is None
    assert len(obs) == n + 3 and len(actions) == n + 2
    for index, row in enumerate(obs):
        assert row['index'] == index and row['actual_sensor_observe_call'] == index + 1
        assert row['target_id'] == spec['target'] and row['role'] == spec['role']
        assert math.isclose(row['time_s'], index * .05, abs_tol=1e-10)
        image = root / row['rgb_path']
        assert image.resolve().is_relative_to(folder.resolve()), str(image)
        assert c.sha(image) == row['rgb_sha256'], str(image)
        assert all(math.isfinite(float(x)) for x in row['signed_OCT_mm'])
    for index, action in enumerate(actions):
        assert action['pre_observation'] == index and action['post_observation'] == index + 1
        assert action['global_action'] == index
        assert action['phase'] == ('prefix' if index < 2 else 'teacher')
        if index >= 2:
            assert action['teacher_step'] == index - 2
        assert obs[index + 1]['previous_accepted_deg_deg_mm'] == action['accepted_deg_deg_mm']
    return dict(id=spec['id'], source=str(folder.resolve()), observations=len(obs),
                teacher_actions=n, summary_sha256=c.sha(folder / 'summary.json'),
                observations_sha256=c.sha(folder / 'observations.jsonl'),
                actions_sha256=c.sha(folder / 'actions.jsonl'), all_rgb_hashes_verified=True)


def preflight():
    assert not (OUT / 'actual_start_receipt.json').exists(), 'Already launched; use launch --resume for another batch'
    changed = [p for p, digest in c.read(OLD / 'input_fingerprints.json').items() if c.sha(p) != digest]
    assert not changed, f'Frozen inputs changed: {changed}'
    assert c.sha(OUT / 'model.py') == c.sha(OLD / 'model.py')
    assert c.sha(OUT / 'collection_helper.py') == c.sha(OLD / 'collection_helper.py')
    plans = c.expansion_plans()
    (OUT / 'data').mkdir(exist_ok=True)
    reused, pending, rejected = [], [], []
    for spec in plans:
        source = OLD / 'data' / spec['id']
        dest = OUT / 'data' / spec['id']
        if (source / 'summary.json').exists():
            try:
                proof = branch_check(source, spec, OLD)
            except Exception as err:
                rejected.append(dict(id=spec['id'], error=repr(err)))
                pending.append(spec['id'])
                continue
            if not dest.exists():
                dest.symlink_to(source, target_is_directory=True)
            assert dest.is_symlink() and dest.resolve() == source
            reused.append(proof)
        else:
            assert not dest.exists(), str(dest)
            pending.append(spec['id'])
    manifest = dict(time=c.now(), planned=len(plans), reused_count=len(reused), pending_count=len(pending),
                    reused=reused, pending=pending, rejected_complete=rejected,
                    old_partial_directories_untouched=True, old_diagnostics_not_rerun=True,
                    excluded_smoke_summaries=sorted(p.parent.name for p in (OLD / 'data').glob('preflight_*/summary.json')))
    c.save('reuse_manifest.json', manifest)
    # This is the exact worker entrypoint and deployment interpreter, with no simulation or fitting.
    cmd = [PYTHON, '-E', '-B', '-u', str(OUT / 'campaign.py'), 'probe']
    probe = subprocess.run(cmd, cwd=ROOT, env={**os.environ, **c.RUNTIME}, capture_output=True, text=True, timeout=35)
    (OUT / 'preflight_worker_probe.log').write_text(probe.stdout + probe.stderr)
    assert probe.returncode == 0, probe.stderr[-2000:]
    loaded = json.loads(probe.stdout.strip().splitlines()[-1])
    assert loaded['output_dir'] == str(OUT) and loaded['data_output'] == str(OUT)
    assert Path(loaded['model']).resolve().parent == OUT and '4090' in loaded['cuda_device']
    frozen = dict(c.read(OLD / 'input_fingerprints.json'))
    for name in ('campaign.py', 'resume.py', 'data_pipeline.py', 'model.py', 'collection_helper.py', 'experiment_card.json', 'launch.sh'):
        frozen[str(OUT / name)] = c.sha(OUT / name)
    c.save('input_fingerprints.json', frozen)
    result = dict(status='passed', time=c.now(), reused_branches=len(reused), pending_branches=pending,
                  worker_probe=loaded, scientific_card=str(OLD / 'experiment_card.json'),
                  batch_budget_s=1919, concurrency_max=12, standard_concurrency=5,
                  complete_qualification='summary+exact plan+counts+timestamps+actions+RGB hashes',
                  skipped_old_diagnostics=True, rejected_complete=rejected)
    c.save('preflight.json', result)
    c.record('resume_preflight_passed', {k: v for k, v in result.items() if k != 'pending_branches'} | {'pending_count':len(pending)})
    print(json.dumps(dict(status='passed', reused=len(reused), pending=len(pending), worker=loaded), ensure_ascii=False))


def archive_partial(path, label):
    if path.exists():
        assert not path.is_symlink(), 'Never archive or overwrite an old reused branch'
        dest = ATTEMPT / 'partials' / label
        dest.parent.mkdir(parents=True, exist_ok=True)
        path.rename(dest)
        c.record('interrupted_partial_preserved', dict(original=str(path), preserved=str(dest)))


def guard():
    if STOP or time.monotonic() >= DEADLINE - 12:
        raise TimeoutError('Batch interrupted or 1919-second actual work budget exhausted; persistent progress retained')


def group(stage, commands, limit, concurrency):
    assert 1 <= concurrency <= 12
    if not commands:
        return
    guard()
    started = time.monotonic()
    end = min(DEADLINE - 8, started + limit)
    queue = list(commands)
    active = []
    completed = []
    c.status(stage, attempt=str(ATTEMPT), queued=len(queue), concurrency=concurrency)
    try:
        while queue or active:
            if STOP or time.monotonic() >= end:
                raise TimeoutError(f'{stage}: actual-work batch or total budget reached')
            while queue and len(active) < concurrency:
                name, args = queue.pop(0)
                log = ATTEMPT / 'logs' / f'{name}.log'
                log.parent.mkdir(exist_ok=True)
                stream = log.open('x')
                cmd = [PYTHON, '-E', '-B', '-u', str(OUT / 'campaign.py'), *args]
                remaining = end - time.monotonic()
                env = {**os.environ, **c.RUNTIME, 'DIT_STEP_SECONDS': str(remaining)}
                process = subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=subprocess.PIPE,
                                           stderr=subprocess.STDOUT, start_new_session=True, text=True, bufsize=1)
                def tee(p=process, f=stream):
                    for line in p.stdout:
                        f.write(line); f.flush()
                        sys.stdout.write(line); sys.stdout.flush()
                thread = threading.Thread(target=tee, daemon=True)
                thread.start()
                item = (name, process, stream, thread)
                active.append(item); ACTIVE.append(process)
                c.append(OUT / 'commands.jsonl', dict(time=c.now(), attempt=str(ATTEMPT), stage=stage,
                    command=cmd, pid=process.pid, log=str(log), output_dir=str(OUT), timeout_s=remaining))
            for item in active[:]:
                name, process, stream, thread = item
                rc = process.poll()
                if rc is None:
                    continue
                thread.join(timeout=3); stream.close(); active.remove(item); ACTIVE.remove(process)
                proof_path = OUT / 'process_receipts' / f'{process.pid}.json'
                proof = c.read(proof_path) if proof_path.exists() else {}
                assert proof.get('output_dir') == str(OUT) and proof.get('data_output') == str(OUT), f'Wrong child output: {name}'
                c.append(OUT / 'worker_exit_receipts.jsonl', dict(time=c.now(), name=name, exit_code=rc, proof=str(proof_path)))
                if rc:
                    if args_for_name(commands, name)[0] == 'train' and (OUT / 'candidate1/resume_progress.json').exists():
                        progress = c.read(OUT / 'candidate1/resume_progress.json')
                        if progress['actual_updates'] < 3000 and 'Training paused below 3000' in (ATTEMPT/'logs'/f'{name}.log').read_text():
                            raise TimeoutError('Training preserved below 3000; dependent evaluations blocked')
                    c.save('failure_block.json', dict(time=c.now(), name=name, exit_code=rc, stage=stage, log=str(ATTEMPT/'logs'/f'{name}.log')))
                    raise RuntimeError(f'{name} failed with exit {rc}; no automatic retry or scientific-negative claim')
                completed.append(name)
                c.append(OUT / 'completed_units.jsonl', dict(time=c.now(), stage=stage, name=name, attempt=str(ATTEMPT)))
            if active:
                time.sleep(.15)
    finally:
        for _, process, _, _ in active:
            if process.poll() is None:
                try:
                    os.killpg(process.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
        # Training handles TERM at an update boundary and atomically saves optimizer/RNG.
        for _, process, stream, thread in active:
            try:
                process.wait(timeout=25)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL); process.wait()
            thread.join(timeout=2); stream.close()
            if process in ACTIVE:
                ACTIVE.remove(process)
        c.append(OUT/'budget_ledger.jsonl', dict(time=c.now(), attempt=str(ATTEMPT), stage=stage,
                 elapsed_s=time.monotonic()-started, batch_elapsed_s=time.monotonic()-START,
                 remaining_s=max(0,DEADLINE-time.monotonic()), completed=completed))


def args_for_name(commands, name):
    return next(args for command_name, args in commands if command_name == name)


def eval_complete(path, expected, checkpoint_sha):
    if not (path / 'summary.json').exists():
        return False
    result = c.read(path / 'summary.json')
    evidence = rows(path / 'cycles.jsonl')
    assert result['status'] == 'completed' and result['expected_cycles'] == expected
    assert result['model_checkpoint_sha256'] == checkpoint_sha
    assert len(evidence) == expected and result['metrics']['completed_cycles'] == expected
    actual = c.score([r['post']['true_error_um'] for r in evidence], expected)
    assert actual == result['metrics']
    return True


def execute():
    global ATTEMPT, START, DEADLINE
    with (OUT / '.run.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        assert not (OUT/'failure_block.json').exists(), 'A prior engineering failure needs explicit repair; see failure_block.json'
        changed = [p for p, digest in c.read(OUT/'input_fingerprints.json').items() if c.sha(p) != digest]
        assert not changed, f'Inputs changed after preflight: {changed}'
        for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
            signal.signal(sig, stop)
        attempt_id = os.environ['DIT_ATTEMPT_ID']
        ATTEMPT = OUT/'attempts'/attempt_id
        ATTEMPT.mkdir(exist_ok=True)
        START = time.monotonic(); c.START = START; DEADLINE = START + 1919
        receipt = dict(time=c.now(),pid=os.getpid(),ppid=os.getppid(),actual_start=True,
                       attempt=attempt_id,output_dir=str(OUT),source=str(OLD),budget_s=1919,
                       interpreter=sys.executable,concurrency_max=12,old_diagnostics_skipped=True)
        c.save(ATTEMPT/'actual_start_receipt.json',receipt)
        c.save('actual_start_receipt.json',receipt)
        c.status('resume_actual_started', **{'attempt':attempt_id,'budget_s':1919})
        c.record('resume_actual_start',receipt)
        fd = os.open(ATTEMPT/'startup_ack.fifo',os.O_WRONLY|os.O_NONBLOCK)
        os.write(fd,(json.dumps(receipt)+'\n').encode());os.close(fd)
        summary = dict(workflow_status='running',goal_achieved=False,attempt=attempt_id,simulation_only=True,
                       source=str(OLD),output_dir=str(OUT),budget_s=1919,actual_start=receipt)
        try:
            plans = c.expansion_plans()
            reused = c.read(OUT/'reuse_manifest.json')
            pending = []
            # Hash the reused logs against the prelaunch manifest; RGB was verified in preflight.
            frozen_by_id = {p['id']:p for p in reused['reused']}
            for spec in plans:
                path = OUT/'data'/spec['id']
                if spec['id'] in frozen_by_id:
                    proof = frozen_by_id[spec['id']]
                    assert path.is_symlink() and str(path.resolve()) == proof['source']
                    for name,key in [('summary.json','summary_sha256'),('observations.jsonl','observations_sha256'),('actions.jsonl','actions_sha256')]:
                        assert c.sha(path/name) == proof[key]
                    continue
                if (path/'summary.json').exists():
                    branch_check(path,spec,OUT)
                    continue
                if (path/'failure.json').exists():
                    raise RuntimeError(f'Previously failed collection requires repair: {path}')
                archive_partial(path,'collection_'+spec['id'])
                pending.append(spec)
            # Each actual <=12-branch work batch gets its own collection allowance.
            for offset in range(0,len(pending),12):
                batch = pending[offset:offset+12]
                group(f'collection_batch_{offset//12+1}',[(s['id'],['collect','--branch',s['id']]) for s in batch],750,12)
                for spec in batch:
                    branch_check(OUT/'data'/spec['id'],spec,OUT)
            c.record('collection_146_complete',dict(planned=146,reused=reused['reused_count'],new_complete=len(reused['pending'])))
            build_stamp = OUT/'build_complete.json'
            if not build_stamp.exists():
                for name in ('relative_windows.pt','relative_window_manifest.json','coverage.json','sampling_strata.json'):
                    archive_partial(OUT/name,'build_'+name)
                group('build_relative_windows',[('build',['build-expanded'])],200,1)
                assert c.read(OUT/'coverage.json')['status']=='passed'
                c.save(build_stamp,dict(time=c.now(),windows_sha256=c.sha(OUT/'relative_windows.pt'),manifest_sha256=c.sha(OUT/'relative_window_manifest.json')))
            else:
                stamp=c.read(build_stamp)
                assert c.sha(OUT/'relative_windows.pt')==stamp['windows_sha256']
                assert c.sha(OUT/'relative_window_manifest.json')==stamp['manifest_sha256']
            training_summary = OUT/'candidate1/summary.json'
            trained = (training_summary.exists() and c.read(training_summary).get('actual_updates')==3000
                       and c.read(training_summary).get('minimum_updates_reached') is True
                       and (OUT/'candidate1/input_ablation_after.json').exists())
            if not trained:
                group('DiT_3000_updates',[('train',['train','--candidate','candidate1'])],780,1)
            assert c.read(training_summary)['actual_updates']==3000
            assert c.read(OUT/'candidate1/parameter_update_evidence.json')['actual_updates']==3000
            checkpoint_sha=c.sha(OUT/'candidate1/checkpoint.pth')
            commands=[]
            for kind in c.KINDS:
                path=OUT/'candidate1/near'/kind
                if not eval_complete(path,240,checkpoint_sha):
                    if (path/'failure.json').exists():
                        raise RuntimeError(f'Previously failed E41 evaluation requires repair: {path}')
                    archive_partial(path,'near_'+kind)
                    commands.append((kind,['evaluate','--candidate','candidate1','--kind',kind]))
            group('E41_four_real_50um',commands,400,4)
            near=c.near_results('candidate1');summary['near_E41']=near
            summary['C_bias']=c.imp('relative_bias_resume',OUT/'data_pipeline.py').bias_report()
            if not near['all_passed']:
                summary.update(workflow_status='complete_near_negative',standard_five='not_run_E41_gate_failed')
            else:
                assert c.read(OUT/'coverage.json')['status']=='passed'
                commands=[]
                for index in (0,2,6,8,10):
                    path=OUT/'candidate1/standard'/f'E{index:02d}'
                    if not eval_complete(path,1600,checkpoint_sha):
                        if (path/'failure.json').exists():
                            raise RuntimeError(f'Previously failed standard evaluation requires repair: {path}')
                        archive_partial(path,f'standard_E{index:02d}')
                        commands.append((f'E{index:02d}',['evaluate','--candidate','candidate1','--index',str(index)]))
                group('standard_five_parallel',commands,890,5)
                standards=[c.read(OUT/'candidate1/standard'/f'E{i:02d}'/'summary.json') for i in (0,2,6,8,10)]
                passed=sum(r['metrics']['strict20um_pass'] for r in standards)
                summary['standard_five']=dict(completed=5,denominator=5,cycles_each=1600,strict20um_pass=passed)
                summary['goal_achieved']=passed==5
                summary['workflow_status']='complete_strict_pass' if passed==5 else 'complete_standard_negative'
        except BaseException as err:
            partial=isinstance(err,TimeoutError)
            summary.update(workflow_status='budget_partial' if partial else 'technical_failure',
                           error=repr(err),traceback=traceback.format_exc(),scientific_negative=False,
                           next_command=f'{PYTHON} -E -B {OUT}/resume.py launch --resume')
            if not partial and not (OUT/'failure_block.json').exists():
                c.save('failure_block.json',dict(time=c.now(),error=repr(err),traceback=summary['traceback']))
        finally:
            summary.update(time=c.now(),elapsed_s=time.monotonic()-START)
            c.save(ATTEMPT/'summary.json',summary);c.save('summary.json',summary)
            c.status(summary['workflow_status'],summary_path=str(OUT/'summary.json'),attempt=attempt_id)
            c.record('resume_batch_finished',summary)
            (OUT/'report.md').write_text('# DiT relative v2 resume\n\n'+json.dumps(summary,indent=2,ensure_ascii=False)+'\n')
            print(f'RESULTS {OUT}/summary.json {OUT}/report.md',flush=True)


def launch(resuming):
    assert c.read(OUT/'preflight.json')['status']=='passed'
    assert not (OUT/'failure_block.json').exists(), 'Repair recorded engineering failure first'
    if not resuming:
        assert not (OUT/'actual_start_receipt.json').exists()
        assert not list((OUT/'attempts').glob('*/gnome_launch_request.json')) if (OUT/'attempts').exists() else True
    else:
        assert c.read(OUT/'summary.json')['workflow_status']=='budget_partial', 'Only resume a budget-interrupted attempt'
    with (OUT/'.launch.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        attempt_id=time.strftime('%Y%m%d_%H%M%S')
        attempt=OUT/'attempts'/attempt_id;attempt.mkdir(parents=True,exist_ok=False)
        fifo=attempt/'startup_ack.fifo';os.mkfifo(fifo,0o600)
        fd=os.open(fifo,os.O_RDWR|os.O_NONBLOCK)
        cmd=['gnome-terminal','--window','--title=DiT relative v2 resume 1919s',
             '--working-directory='+str(ROOT),'--','bash',str(OUT/'launch.sh'),attempt_id]
        c.save(attempt/'gnome_launch_request.json',dict(time=c.now(),command=cmd,single_launch=True))
        try:
            result=subprocess.run(cmd,env={**os.environ,**c.DESKTOP},capture_output=True,text=True,timeout=15)
            c.save(attempt/'gnome_launch_receipt.json',dict(time=c.now(),returncode=result.returncode,stdout=result.stdout,stderr=result.stderr,command=cmd))
            assert result.returncode==0,result.stderr
            selector=selectors.DefaultSelector();selector.register(fd,selectors.EVENT_READ)
            ready=selector.select(25)
            message=os.read(fd,65536).decode() if ready else ''
            selector.close()
            assert message,'GNOME accepted launch but no actual process acknowledgement; no background fallback or retry'
            receipt=json.loads(message.splitlines()[0])
            assert receipt['output_dir']==str(OUT) and receipt['actual_start'] and receipt['pid']>0
            ack=dict(time=c.now(),actual_start=receipt,one_startup_ack_only=True,no_postlaunch_polling=True)
            c.save('launch_startup_acknowledgement.json',ack)
            c.save(attempt/'launch_startup_acknowledgement.json',ack)
            c.record('GNOME_resume_actual_start_acknowledged',ack)
            # Read back only the launch and TEST_LOG receipts, never poll experiment progress.
            assert c.read(OUT/'launch_startup_acknowledgement.json')==ack
            checks=rows(OUT/'test_log_receipts.jsonl')
            latest=next(r for r in reversed(checks) if r['stage']=='GNOME_resume_actual_start_acknowledged')
            assert all(r['exact_readback'] for r in latest['receipts'])
            c.save('launch_verified.json',dict(time=c.now(),pid=receipt['pid'],attempt=attempt_id,
                   startup_receipt_readback=True,root_and_local_TEST_LOG_readback=True,no_postlaunch_polling=True))
            print('ACTUAL_START_RECEIPT '+json.dumps(ack,ensure_ascii=False),flush=True)
        except BaseException as err:
            c.save(attempt/'launch_failure.json',dict(time=c.now(),error=repr(err),no_background_fallback=True))
            raise
        finally:
            os.close(fd)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode',choices=('preflight','launch','run'))
    parser.add_argument('--resume',action='store_true')
    args=parser.parse_args()
    if args.mode=='preflight':
        preflight()
    elif args.mode=='launch':
        launch(args.resume)
    else:
        execute()
