"""Create an additive verified GitHub snapshot on the label-fix branch."""
from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import os
from pathlib import Path
import subprocess

from lerobot_training.navigation_overlay import file_sha256
from scripts.research.homog_label_fix_common import (
    BASE, BRANCH, FIX_DATA, REPORTS, ROOT, SOURCE_REPO, TASK, WEIGHTS,
    read, record, write_json,
)
from scripts.research.homog_replay_publish import preserve_log_history

SSH_KEY = Path('/home/him1/.ssh/rcm_github_deploy_ed25519')


def publication_files(output):
    files = {}
    sources = list((ROOT/'scripts/research').glob('homog_label_fix_*.py'))
    sources += [ROOT/'scripts/verification/verify_homog_label_fix.py',
                ROOT/'scripts/tools/prepare_homog_label_fix.py',
                ROOT/'TEST_LOG.md']
    for path in sources:
        if not path.is_file():
            raise FileNotFoundError(path)
        files['rcm_4090_deploy/'+path.relative_to(ROOT).as_posix()] = path
    files['规划6.txt'] = TASK
    for directory in (output, *REPORTS):
        if not directory.exists():
            continue
        for path in directory.rglob('*'):
            if (path.is_file() and path.suffix in
                    ('.json','.jsonl','.npz','.csv','.png','.md','.log','.txt')):
                files['rcm_4090_deploy/'+path.relative_to(ROOT).as_posix()] = path
    if FIX_DATA.exists():
        for path in FIX_DATA.rglob('*'):
            if (path.is_file() and 'shards' not in path.relative_to(FIX_DATA).parts
                    and path.suffix in ('.json','.npz','.txt','.md')):
                files['rcm_4090_deploy/'+path.relative_to(ROOT).as_posix()] = path
    if WEIGHTS.exists():
        for path in WEIGHTS.rglob('*'):
            if path.is_file() and path.suffix in ('.json','.jsonl','.npz'):
                files['rcm_4090_deploy/'+path.relative_to(ROOT).as_posix()] = path
    return files


def publish(output, *, push=False):
    output = Path(output).resolve()
    env = os.environ.copy()
    env['GIT_SSH_COMMAND'] = f'ssh -i {SSH_KEY} -o IdentitiesOnly=yes -o BatchMode=yes'

    def git(*args, data=None):
        return subprocess.run(['git','-C',str(SOURCE_REPO),*args], input=data,
            capture_output=True, check=True, env=env).stdout

    parent = git('rev-parse', BRANCH).decode().strip()
    if git('merge-base', parent, BASE).decode().strip() != BASE:
        raise ValueError('Publication branch no longer descends from authorized base')
    local_log = (ROOT/'TEST_LOG.md').read_bytes()
    local_snapshot = output/f'local_TEST_LOG_{hashlib.sha256(local_log).hexdigest()[:16]}.md'
    if local_snapshot.exists() and local_snapshot.read_bytes() != local_log:
        raise ValueError('TEST_LOG snapshot hash collision')
    if not local_snapshot.exists():
        local_snapshot.write_bytes(local_log)
    base_log = git('show', BASE+':rcm_4090_deploy/TEST_LOG.md')
    published_log, log_merge = preserve_log_history(base_log, local_log)
    parent_log = git('show', parent+':rcm_4090_deploy/TEST_LOG.md')
    if not published_log.startswith(parent_log):
        raise ValueError('Would lose prior branch TEST_LOG history')
    files = publication_files(output)
    inventory = []
    if WEIGHTS.exists():
        for path in sorted(WEIGHTS.rglob('*')):
            if path.is_file() and path.suffix in ('.pth','.png'):
                inventory.append({'path': str(path), 'bytes': path.stat().st_size,
                                  'sha256': file_sha256(path), 'uploaded': False})
    write_json(output/'retained_large_assets.json', {
        'assets': inventory,
        'policy': 'Weights and raw correction RGB remain local; publication retains paths, sizes and SHA256.'})
    files['rcm_4090_deploy/'+(output/'retained_large_assets.json').relative_to(ROOT).as_posix()] = (
        output/'retained_large_assets.json')
    frozen = {name: path.read_bytes() for name, path in files.items()}
    frozen['rcm_4090_deploy/TEST_LOG.md'] = published_log
    manifest = {
        'base': BASE, 'parent': parent, 'branch': BRANCH,
        'TEST_LOG_preservation': {
            **log_merge, 'local_complete_snapshot': str(local_snapshot),
            'base_sha256': hashlib.sha256(base_log).hexdigest(),
            'local_sha256': hashlib.sha256(local_log).hexdigest(),
            'published_sha256': hashlib.sha256(published_log).hexdigest()},
        'files': [{'path': name, 'bytes': len(data),
                   'sha256': hashlib.sha256(data).hexdigest()}
                  for name, data in sorted(frozen.items())],
    }
    write_json(output/'publication_manifest.json', manifest)
    frozen['rcm_4090_deploy/'+(output/'publication_manifest.json').relative_to(ROOT).as_posix()] = (
        output/'publication_manifest.json').read_bytes()
    git('read-tree', parent)
    modes = {}
    for entry in git('ls-tree','-r','-z',parent).split(b'\0'):
        if entry:
            metadata, path = entry.split(b'\t',1)
            modes[path.decode()] = metadata.split()[0].decode()
    entries = []
    for name, data in sorted(frozen.items()):
        if len(data) >= 95_000_000:
            raise ValueError(f'Artifact exceeds safe GitHub blob size: {name}')
        if any(marker in data for marker in (
                b'-----BEGIN '+b'OPENSSH PRIVATE KEY-----',
                b'-----BEGIN '+b'RSA PRIVATE KEY-----')):
            raise ValueError('Private-key material rejected')
        blob = git('hash-object','-w','--stdin',data=data).strip().decode()
        entries.append(f'{modes.get(name, "100644")} {blob}\t{name}\0'.encode())
    git('update-index','-z','--index-info',data=b''.join(entries))
    tree = git('write-tree').decode().strip()
    deleted = git('diff-tree','--no-commit-id','-r','--diff-filter=D',
                  '--name-only','-z',parent,tree)
    if deleted:
        raise ValueError('Additive publication unexpectedly deletes files')
    for protected in ('simulation_driver',
                      'rcm_4090_deploy/mujoco_model/vendor/simulation_driver',
                      'rcm_4090_deploy/mujoco_model/vendor/simulation_driver_g15',
                      'rcm_4090_deploy/data/plan2_plus_original_train_224_20261119'):
        if git('rev-parse', BASE+':'+protected) != git('rev-parse', tree+':'+protected):
            raise ValueError('Protected base tree changed: '+protected)
    if not frozen['rcm_4090_deploy/TEST_LOG.md'].startswith(base_log):
        raise ValueError('TEST_LOG base prefix was not preserved')
    commit = git('commit-tree', tree, '-p', parent, '-m',
        'Repair homogeneous replay labels and evaluate retrained checkpoints').decode().strip()
    git('update-ref','refs/heads/'+BRANCH,commit,parent)
    receipt = {
        'status': 'committed_verified', 'branch': BRANCH, 'base': BASE,
        'parent': parent, 'commit': commit, 'tree': tree, 'files': len(frozen),
        'timestamp': datetime.now().astimezone().isoformat(),
        'protected_assets_unchanged': True, 'unintended_deletions': 0,
        'weights_and_RGB_retained_locally': True,
    }
    write_json(output/'publication_status.json', receipt)
    if push:
        git('push','origin',f'refs/heads/{BRANCH}:refs/heads/{BRANCH}')
        remote = git('ls-remote','origin',f'refs/heads/{BRANCH}').decode().split()[0]
        if remote != commit:
            raise ValueError('Remote branch did not match committed snapshot')
        receipt.update(status='pushed_verified', remote_commit=remote)
        write_json(output/'publication_status.json', receipt)
        record('publication','pushed_verified',output,receipt)
    print('Publication:', receipt, flush=True)
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--push', action='store_true')
    args = parser.parse_args()
    publish(args.output_dir.resolve(), push=args.push)


if __name__ == '__main__':
    main()
