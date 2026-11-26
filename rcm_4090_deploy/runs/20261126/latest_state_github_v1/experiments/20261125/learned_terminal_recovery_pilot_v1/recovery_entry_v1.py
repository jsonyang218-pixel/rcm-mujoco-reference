"""Authorized infrastructure-only recovery; retain every original failed artifact."""
import argparse
from datetime import datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
import selectors
import shutil
import subprocess
import sys
import traceback

BASE = Path(__file__).resolve().parent
ROOT = BASE.parents[2]
RECOVERY = BASE / 'recovery_v1'
RUNTIME = RECOVERY / 'run'
PARENT_LOG = ROOT.parent / 'TEST_LOG.md'
TITLE = 'Learned terminal recovery pilot v1 - authorized recovery'
DESKTOP = dict(DISPLAY=':1', XDG_RUNTIME_DIR='/run/user/1000',
               DBUS_SESSION_BUS_ADDRESS='unix:path=/run/user/1000/bus')
ORIGINALS = ('experiment_card.json', 'input_fingerprints.json', 'launch.sh',
             'launch_failure_report.json', 'launch_once.py', 'launch_receipt.json',
             'pilot.py', 'report.md', 'reuse_audit.json', 'student_policy.py',
             'workflow_status.json')
EXECUTION_ARTIFACTS = ('model.json', 'training_result.json', 'training_arrays.npz',
                       'training_samples.json', 'host_preflight.json', 'failure.json',
                       'trajectories', 'local_metrics.json', 'summary.json', 'terminal.log')


def now():
    return datetime.now().astimezone().isoformat()


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write_new(path, value):
    with Path(path).open('x', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write('\n')


def log(stage, detail):
    entry = (f'\n- {now()}: learned_terminal_recovery_pilot_v1 recovery_v1 {stage}; '
             f'original_config={BASE}/experiment_card.json; output={RUNTIME}; '
             f'result={detail}; infrastructure-only recovery, frozen recipe unchanged; '
             'local MuJoCo test, no dev20/final test or publication.\n')
    for path in (ROOT/'TEST_LOG.md', PARENT_LOG):
        with path.open('ab') as stream:
            stream.write(entry.encode())
    print(entry.strip(), flush=True)


def process_snapshot():
    relevant, active = [], []
    names = {str(BASE/'pilot.py'), str(BASE/'launch.sh'), str(BASE/'launch_recovery_v1.sh'),
             str(BASE/'recovery_entry_v1.py')}
    for folder in Path('/proc').iterdir():
        if not folder.name.isdigit():
            continue
        try:
            argv = [v.decode(errors='replace') for v in (folder/'cmdline').read_bytes().split(b'\0') if v]
            if not argv:
                continue
            executable = Path(argv[0]).name
            if 'python' not in executable and executable not in ('bash', 'timeout'):
                continue
            if any(v in names or v.endswith('/learned_terminal_recovery_pilot_v1/pilot.py')
                   or v.endswith('/learned_terminal_recovery_pilot_v1/launch.sh') for v in argv):
                row = dict(pid=int(folder.name), argv=argv)
                relevant.append(row)
                if int(folder.name) != os.getpid() and not (
                    str(BASE/'recovery_entry_v1.py') in argv and 'launch' in argv):
                    active.append(row)
        except (OSError, ProcessLookupError):
            pass
    return relevant, active


def verify_originals():
    manifest = read(RECOVERY/'recovery_manifest.json')
    mismatches = [name for name, sha in manifest['original_files_sha256'].items()
                  if digest(BASE/name) != sha]
    if mismatches:
        raise RuntimeError(f'Original failure artifacts changed: {mismatches}')
    return dict(original_file_count=len(manifest['original_files_sha256']), unchanged=True)


def desktop_environment():
    environment = os.environ.copy()
    appdir = environment.get('APPDIR')
    if appdir:
        for key in ('PYTHONHOME','PYTHONPATH','LD_LIBRARY_PATH','GTK_PATH', 'GTK_EXE_PREFIX',
                    'GIO_MODULE_DIR','GI_TYPELIB_PATH','PATH'):
            kept = [v for v in environment.get(key,'').split(os.pathsep)
                    if v and not Path(v).is_relative_to(appdir)]
            if kept:
                environment[key] = os.pathsep.join(kept)
            else:
                environment.pop(key, None)
    environment.update(DESKTOP)
    return environment


def launch():
    with (BASE/'.recovery_v1_launch.lock').open('a') as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        if RECOVERY.exists():
            raise FileExistsError('Recovery already prepared/attempted; never launch a second copy')
        original_status = read(BASE/'workflow_status.json')
        original_receipt = read(BASE/'launch_receipt.json')
        assert original_status['status'] == 'launch_failed' and original_status['experiment_started'] is False
        assert original_receipt['status'] == 'gnome_launch_failed' and original_receipt['returncode'] == 1
        assert 'Cannot open display' in original_receipt['stderr']
        relevant, active = process_snapshot()
        assert not active, f'Existing execution processes: {active}'
        artifacts = [name for name in EXECUTION_ARTIFACTS if (BASE/name).exists()]
        assert not artifacts, f'Existing execution artifacts: {artifacts}'
        sys.path.insert(0, str(BASE))
        import pilot
        pilot.check_identity()
        environment = desktop_environment()
        x11 = subprocess.run(['xdpyinfo','-display',':1'],env=environment, capture_output=True, text=True, timeout=8)
        assert x11.returncode == 0, x11.stderr
        dbus = subprocess.run(['gdbus','call','--session','--dest','org.freedesktop.DBus',
            '--object-path','/org/freedesktop/DBus','--method','org.freedesktop.DBus.NameHasOwner',
            'org.gnome.Terminal'],env=environment,capture_output=True,text=True,timeout=8)
        assert dbus.returncode == 0 and dbus.stdout.strip() == '(true,)', dbus.stderr
        gpu = subprocess.run(['nvidia-smi','--query-gpu=name,memory.used,utilization.gpu',
            '--format=csv,noheader'],capture_output=True,text=True,timeout=8)
        assert gpu.returncode == 0 and '4090' in gpu.stdout, gpu.stderr
        RECOVERY.mkdir()
        RUNTIME.mkdir()
        snapshot = {name:digest(BASE/name) for name in ORIGINALS}
        manifest = dict(schema_version=1,created=now(), authorization='User explicitly authorized recovery of verified zero-execution GNOME-only failure; same experiment, no duplicate process or recipe change.',
            original_files_sha256=snapshot, original_status=original_status,
            original_launch_receipt_sha256=digest(BASE/'launch_receipt.json'),
            no_active_experiment_processes=True, process_snapshot=relevant,
            no_execution_results=True, checked_absent=list(EXECUTION_ARTIFACTS),
            frozen_input_hashes_verified=len(read(BASE/'input_fingerprints.json')['files']),
            desktop_environment=DESKTOP, x11_probe='\n'.join(x11.stdout.splitlines()[:10]),
            gnome_terminal_server_owner=dbus.stdout.strip(),gpu_probe=gpu.stdout.strip(),
            preserved_original_directory=str(BASE), resumed_runtime_directory=str(RUNTIME),
            prepared_state_restoration='New recovery runtime state only; original failed workflow_status.json is immutable.',
            recipe_changes=[], infrastructure_changes=['Explicit host desktop environment',
                'Separate runtime output path beneath the same experiment directory',
                'Startup acknowledgment and duplicate-execution guard',
                'Append stage records to parent and deployment TEST_LOG.md'],
            frozen_pipeline_code_sha256=digest(BASE/'pilot.py'),
            recovery_entry_sha256=digest(__file__), recovery_shell_sha256=digest(BASE/'launch_recovery_v1.sh'))
        write_new(RECOVERY/'recovery_manifest.json',manifest)
        for name in ('experiment_card.json','input_fingerprints.json','student_policy.py','reuse_audit.json'):
            shutil.copy2(BASE/name,RUNTIME/name)
            assert digest(BASE/name)==digest(RUNTIME/name)
        write_new(RUNTIME/'workflow_status.json',dict(status='prepared',stage='authorized_recovery_before_first_actual_start',
            completed_segments=0,completed_local_rollouts=0,independent_standard_acceptance=False,
            original_failure_preserved=str(BASE/'workflow_status.json')))
        (RECOVERY/'README.md').write_text(
            '# Authorized launch recovery v1\n\n'
            'Original files in the parent directory remain byte-for-byte unchanged. '
            'The original failed status/receipt/report remain historical failure evidence.\n\n'
            'Recovery/runtime state and experiment results: `run/workflow_status.json`, '
            '`run/report.md`, `run/summary.json`, `run/local_metrics.json`, `run/model.json`, '
            '`run/trajectories/`. Startup receipts: `first_actual_start.json`, '
            '`actual_start_receipt.json`, `gnome_launch_receipt.json`. '
            'Software errors: `software_failure.json`, `run/failure.json`, `terminal.log`.\n\n'
            'No recipe, model math, target, sampling, budget, or gate is changed. '
            'The frozen pilot module runs with only its OUT directory redirected here. '
            'The parent project TEST_LOG.md and deployment TEST_LOG.md receive appended records.\n')
        fifo = RECOVERY/'startup_ack.fifo'
        os.mkfifo(fifo,0o600)
        fd = os.open(fifo,os.O_RDWR | os.O_NONBLOCK)
        selection = selectors.DefaultSelector()
        selection.register(fd,selectors.EVENT_READ)
        command = ['/usr/bin/gnome-terminal','--window',f'--working-directory={ROOT}',
            f'--title={TITLE}','--','/usr/bin/env',
            *(f'{k}={v}' for k,v in DESKTOP.items()),'/bin/bash',str(BASE/'launch_recovery_v1.sh')]
        write_new(RECOVERY/'gnome_launch_request.json',dict(time=now(),command=command,one_attempt_only=True))
        log('prepared state safely restored',dict(originals=verify_originals(),runtime=str(RUNTIME),
            execution_processes_before=[],results_before=[],parent_log=str(PARENT_LOG)))
        try:
            result = subprocess.run(command,cwd=ROOT,env=environment,capture_output=True,text=True,timeout=15)
            receipt=dict(time=now(),command=command,returncode=result.returncode,
                stdout=result.stdout,stderr=result.stderr,desktop_environment=DESKTOP,
                original_files_preserved=verify_originals(), runtime_output_dir=str(RUNTIME),
                completion_verified=False)
            if result.returncode:
                receipt['status']='gnome_recovery_launch_failed'
                write_new(RECOVERY/'gnome_launch_receipt.json',receipt)
                raise RuntimeError(result.stderr)
            # One startup-only FIFO acknowledgment, never a training/result polling loop.
            ready = selection.select(timeout=25)
            ack = json.loads(os.read(fd,65536).decode()) if ready else None
            windows = subprocess.run(['xwininfo','-root','-tree'],env=environment,
                capture_output=True,text=True,timeout=8)
            visible_window_lines=[line.strip() for line in windows.stdout.splitlines() if TITLE in line]
            receipt.update(status='actual_start_confirmed' if ack and ack.get('status')=='host_preflight_passed'
                else 'software_startup_failed' if ack else 'gnome_accepted_startup_ack_timeout',
                startup_acknowledgment=ack, x11_window_matches=visible_window_lines,
                startup_confirmation_only=True, training_or_completion_polled=False)
            write_new(RECOVERY/'gnome_launch_receipt.json',receipt)
            log('GNOME recovery actual receipt',dict(status=receipt['status'],ack=ack,
                x11_window_matches=visible_window_lines,receipt=str(RECOVERY/'gnome_launch_receipt.json')))
            print(json.dumps(receipt,ensure_ascii=False,indent=2),flush=True)
        except BaseException as exc:
            failure=dict(time=now(),error=repr(exc),traceback=traceback.format_exc(),no_automatic_retry=True)
            write_new(RECOVERY/'recovery_launch_failure.json',failure)
            log('recovery launch failed',failure)
            raise
        finally:
            selection.close()
            os.close(fd)


def acknowledge(payload):
    try:
        fd=os.open(RECOVERY/'startup_ack.fifo',os.O_WRONLY | os.O_NONBLOCK)
        try:
            os.write(fd,(json.dumps(payload)+'\n').encode())
        finally:
            os.close(fd)
    except OSError:
        pass  # A disconnected client must not interrupt the authorized experiment.


def ancestry():
    rows=[]
    pid=os.getpid()
    for _ in range(10):
        try:
            status=Path(f'/proc/{pid}/status').read_text().splitlines()
            ppid=int(next(v.split(':',1)[1] for v in status if v.startswith('PPid:')))
            args=Path(f'/proc/{pid}/cmdline').read_bytes().replace(b'\0',b' ').decode(errors='replace').strip()
            rows.append(dict(pid=pid,ppid=ppid,args=args))
            if ppid<=1:
                break
            pid=ppid
        except OSError:
            break
    return rows


def execute():
    with (RECOVERY/'.execution.lock').open('a') as lock:
        fcntl.flock(lock.fileno(),fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (RECOVERY/'first_actual_start.json').exists():
            raise RuntimeError('An actual execution already started; no duplicate or automatic retry')
        for key,value in DESKTOP.items():
            assert os.environ.get(key)==value,(key,os.environ.get(key))
        preservation=verify_originals()
        chain=ancestry()
        assert any('gnome-terminal-server' in row['args'] for row in chain),'Must execute as actual GNOME child'
        first=dict(time=now(),status='process_started',pid=os.getpid(),ppid=os.getppid(),
            tty=os.ttyname(0),desktop_environment=DESKTOP,ancestry=chain,
            original_files_preserved=preservation,runtime_output_dir=str(RUNTIME),
            frozen_recipe_sha256=digest(BASE/'experiment_card.json'),
            frozen_code_sha256=digest(BASE/'pilot.py'),model_loaded_or_trained_before_start=False)
        write_new(RECOVERY/'first_actual_start.json',first)
        ack_sent=False
        try:
            sys.path.insert(0,str(BASE))
            import pilot
            pilot.OUT=RUNTIME  # Only artifact routing changes; original module and recipe stay frozen.
            original_record=pilot.record
            def record_both(stage,result):
                nonlocal ack_sent
                original_record(stage,result)
                with PARENT_LOG.open('ab') as stream:
                    stream.write((f'\n- {now()}: learned_terminal_recovery_pilot_v1 recovery_v1 {stage}; '
                        f'config={BASE}/experiment_card.json; result={result}; output={RUNTIME}; '
                        'local training-split MuJoCo pilot only.\n').encode())
                if stage=='host preflight passed' and not ack_sent:
                    ack=dict(status='host_preflight_passed',time=now(),pid=os.getpid(),ppid=os.getppid(),
                        tty=first['tty'],gpu=result['gpu']['device'],
                        actual_start_path=str(RECOVERY/'first_actual_start.json'),
                        original_files_preserved=verify_originals(),runtime_output_dir=str(RUNTIME),
                        completion_verified=False)
                    write_new(RECOVERY/'actual_start_receipt.json',ack)
                    acknowledge(ack)
                    ack_sent=True
            pilot.record=record_both
            log('first actual GNOME process started',first)
            pilot.run()
            write_new(RECOVERY/'execution_finish_receipt.json',dict(time=now(),status='finished',
                runtime_status=read(RUNTIME/'workflow_status.json'),original_files_preserved=verify_originals()))
        except BaseException as exc:
            failure=dict(time=now(),pid=os.getpid(),error=repr(exc),traceback=traceback.format_exc(),
                runtime_failure=str(RUNTIME/'failure.json'),terminal_log=str(RECOVERY/'terminal.log'),
                original_files_preserved=verify_originals(),automatic_retry=False)
            write_new(RECOVERY/'software_failure.json',failure)
            log('software execution failed',failure)
            if not ack_sent:
                acknowledge(dict(status='software_startup_failed',time=now(),pid=os.getpid(),
                    error=repr(exc),diagnostics=str(RECOVERY/'software_failure.json')))
            raise


def terminal_exit(code):
    write_new(RECOVERY/'terminal_exit.json',dict(time=now(),exit_code=code,
        workflow_status_path=str(RUNTIME/'workflow_status.json'),original_files_preserved=verify_originals()))
    log('terminal command exited',dict(exit_code=code,status_path=str(RUNTIME/'workflow_status.json'),
        diagnostics=str(RECOVERY/'software_failure.json'),terminal_log=str(RECOVERY/'terminal.log')))


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('command',choices=['launch','_run','_terminal_exit'])
    parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--exit-code',type=int)
    args=parser.parse_args()
    assert args.output_dir.resolve()==BASE
    if args.command=='launch':
        launch()
    elif args.command=='_run':
        execute()
    else:
        terminal_exit(args.exit_code)
