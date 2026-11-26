"""Launch this prepared experiment once through the project GNOME SOP."""
import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

OUT=Path(__file__).resolve().parent
sys.path.insert(0,str(OUT))
from pilot import ROOT, digest, read, record, save

parser=argparse.ArgumentParser()
parser.add_argument('--output-dir',type=Path,required=True)
args=parser.parse_args()
assert args.output_dir.resolve()==OUT
assert read(OUT/'workflow_status.json')['status']=='prepared'
assert not (OUT/'launch_receipt.json').exists(), 'No repeated launch allowed'
environment=os.environ.copy()
appdir=environment.get('APPDIR')
if appdir:
    for key in ('PYTHONHOME','PYTHONPATH','LD_LIBRARY_PATH','GTK_PATH','GTK_EXE_PREFIX',
                'GIO_MODULE_DIR','GI_TYPELIB_PATH','PATH'):
        kept=[v for v in environment.get(key,'').split(os.pathsep)
              if v and not Path(v).is_relative_to(appdir)]
        if kept:
            environment[key]=os.pathsep.join(kept)
        else:
            environment.pop(key,None)
terminal=shutil.which('gnome-terminal',path=environment.get('PATH'))
command=[terminal or 'gnome-terminal','--window',f'--working-directory={ROOT}',
         '--title=Learned terminal recovery pilot v1 (local only)',
         '--','/bin/bash',str(OUT/'launch.sh')]
receipt=dict(time=datetime.now().astimezone().isoformat(),command=command,
    output_dir=str(OUT),launch_script_sha256=digest(OUT/'launch.sh'),
    poll_after_launch=False,background_fallback=False)
try:
    if terminal is None:
        raise FileNotFoundError('GNOME terminal unavailable; no background fallback')
    launched=subprocess.run(command,cwd=ROOT,env=environment,capture_output=True,text=True,timeout=15)
    receipt.update(returncode=launched.returncode,stdout=launched.stdout,stderr=launched.stderr)
    if launched.returncode:
        raise RuntimeError(f'GNOME launch returned {launched.returncode}: {launched.stderr}')
    receipt['status']='gnome_launch_accepted'
    receipt['completion_verified']=False
    save('launch_receipt.json',receipt)
    record('GNOME launch accepted',dict(output_dir=str(OUT),one_launch=True,
        completion_verified=False,host_gpu_and_process_preflight='inside launched terminal'))
    print(json.dumps(receipt,ensure_ascii=False,indent=2))
except BaseException as exc:
    receipt.update(status='gnome_launch_failed',error=repr(exc),completion_verified=False)
    save('launch_receipt.json',receipt)
    save('workflow_status.json',dict(status='launch_failed',reason=repr(exc),experiment_started=False,
        no_background_fallback=True))
    record('GNOME launch failed',receipt)
    raise
