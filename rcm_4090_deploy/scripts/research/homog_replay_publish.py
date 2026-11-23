"""Additive, verified snapshot commits for the independent repair branch."""
from __future__ import annotations

import argparse
from datetime import datetime
import os
from pathlib import Path
import subprocess

from lerobot_training.navigation_overlay import file_sha256
from scripts.research.homog_replay_common import (
    ROOT, DATA, WEIGHTS, REPORTS, PUB, BASE, read, record, write_json,
)

SSH_KEY = Path('/home/him1/.ssh/rcm_github_deploy_ed25519')


def preserve_log_history(base, local):
    """Keep the complete published prefix; retain the full local variant separately."""
    if local.startswith(base):
        return local, {'merged': False, 'local_tail_offset': len(base)}
    marker = local.find(b'homog_replay_20261123')
    if marker < 0:
        raise ValueError('divergent local TEST_LOG has no identifiable current-task boundary')
    boundary = local.rfind(b'\n-', 0, marker)
    if boundary < 0:
        raise ValueError('current-task TEST_LOG record boundary unavailable')
    offset = boundary+1
    note = (b'\n\n## Homogeneous replay 2026-11-23 publication addendum\n\n'
            b'The pre-existing local log diverges from the published historical prefix. '
            b'The complete local byte snapshot is retained in the workflow report directory; '
            b'only current-task append records follow here. Neither historical version is rewritten.\n\n')
    return base+note+local[offset:], {'merged': True, 'local_tail_offset': offset}


def publication_files(output):
    files = {}
    sources = list((ROOT/'scripts/research').glob('homog_replay_*.py'))
    sources += [ROOT/'scripts/verification/verify_homog_replay.py', ROOT/'TEST_LOG.md',
                ROOT/'scripts/research/gpu_episode_pool.py',
                ROOT/'scripts/verification/verify_homog_teacher_intent.py',
                ROOT/'scripts/verification/diagnose_homog_plateau.py',
                ROOT/'scripts/tools/prepare_teacher_intent_repair.py',
                ROOT/'docs/HOMOG_REPLAY_PLAN_20261123.md']
    for path in sources:
        if not path.is_file():
            raise FileNotFoundError(path)
        files['rcm_4090_deploy/'+path.relative_to(ROOT).as_posix()] = path
    files['规划5.txt'] = ROOT.parent/'规划5.txt'
    if (PUB/'SUMMARY.md').exists():
        files['SUMMARY.md'] = PUB/'SUMMARY.md'
    for directory in (output, *REPORTS):
        if not directory.exists():
            continue
        for path in directory.rglob('*'):
            if (path.is_file() and 'images' not in path.relative_to(directory).parts
                    and path.suffix in ('.json', '.jsonl', '.npz', '.csv', '.png', '.md', '.log')
                    and path.name != 'publication_manifest.json'):
                files['rcm_4090_deploy/'+path.relative_to(ROOT).as_posix()] = path
    for directory in (WEIGHTS, DATA):
        for path in directory.rglob('*'):
            if path.is_file() and path.suffix in ('.json', '.jsonl', '.npz'):
                files['rcm_4090_deploy/'+path.relative_to(ROOT).as_posix()] = path
    for verification in (ROOT/'runs/20261123/verification').glob('homog_replay*'):
        for path in verification.rglob('*'):
            if (path.is_file() and 'images' not in path.relative_to(verification).parts
                    and path.suffix in ('.json', '.log', '.png', '.csv', '.md')):
                files['rcm_4090_deploy/'+path.relative_to(ROOT).as_posix()] = path
    return files


def publish(output, *, push=False):
    output = Path(output)
    info = read(PUB/'branch.json')
    repo, branch = Path(info['repository']), info['branch']
    env = os.environ.copy()
    env['GIT_SSH_COMMAND'] = f'ssh -i {SSH_KEY} -o IdentitiesOnly=yes -o BatchMode=yes'

    def git(*args, data=None):
        return subprocess.run(['git', '-C', str(repo), *args], input=data,
                              capture_output=True, check=True, env=env).stdout

    parent = git('rev-parse', branch).decode().strip()
    if git('merge-base', parent, BASE).decode().strip() != BASE:
        raise ValueError('publication branch no longer descends from task base')
    local_log = (ROOT/'TEST_LOG.md').read_bytes()
    import hashlib
    local_snapshot = output/f'local_TEST_LOG_{hashlib.sha256(local_log).hexdigest()[:16]}.md'
    if local_snapshot.exists() and local_snapshot.read_bytes() != local_log:
        raise ValueError('local log snapshot name collision')
    if not local_snapshot.exists():
        local_snapshot.write_bytes(local_log)
    old_log = git('show', BASE+':rcm_4090_deploy/TEST_LOG.md')
    published_log, log_merge = preserve_log_history(old_log, local_log)
    parent_log = git('show', parent+':rcm_4090_deploy/TEST_LOG.md')
    if not published_log.startswith(parent_log):
        raise ValueError('current branch TEST_LOG prefix would be lost')
    files = publication_files(output)
    inventory = []
    for directory in (WEIGHTS, DATA):
        for path in sorted(directory.rglob('*')):
            if path.is_file() and path.suffix in ('.pth', '.png'):
                inventory.append({'path': str(path), 'bytes': path.stat().st_size,
                                  'sha256': file_sha256(path), 'uploaded': False})
    write_json(output/'retained_large_assets.json', {'assets': inventory,
        'policy': 'Weights/resume state and raw correction RGB remain local; all paths/sizes/SHA256 retained. Original datasets and models untouched.'})
    files['rcm_4090_deploy/'+(output/'retained_large_assets.json').relative_to(ROOT).as_posix()] = output/'retained_large_assets.json'
    # Freeze byte content once. No live log is read while a commit is assembled.
    frozen = {name: path.read_bytes() for name, path in files.items()}
    frozen['rcm_4090_deploy/TEST_LOG.md'] = published_log
    manifest = {'base': BASE, 'parent': parent, 'branch': branch,
                'TEST_LOG_preservation': {**log_merge, 'local_complete_snapshot': str(local_snapshot),
                    'base_sha256': hashlib.sha256(old_log).hexdigest(),
                    'local_sha256': hashlib.sha256(local_log).hexdigest(),
                    'published_sha256': hashlib.sha256(published_log).hexdigest()},
                'files': [{'path': name, 'bytes': len(data), 'sha256': hashlib.sha256(data).hexdigest()}
                          for name, data in sorted(frozen.items())]}
    write_json(output/'publication_manifest.json', manifest)
    frozen['rcm_4090_deploy/'+(output/'publication_manifest.json').relative_to(ROOT).as_posix()] = (output/'publication_manifest.json').read_bytes()
    git('read-tree', parent)
    modes = {}
    for entry in git('ls-tree', '-r', '-z', parent).split(b'\0'):
        if entry:
            meta, path = entry.split(b'\t', 1)
            modes[path.decode()] = meta.split()[0].decode()
    entries = []
    for name, data in sorted(frozen.items()):
        if len(data) >= 95_000_000:
            raise ValueError(f'Artifact exceeds safe GitHub blob size; not silently dropped: {name}')
        if any(marker in data for marker in (b'-----BEGIN '+b'OPENSSH PRIVATE KEY-----', b'-----BEGIN '+b'RSA PRIVATE KEY-----')):
            raise ValueError('private-key material rejected')
        blob = git('hash-object', '-w', '--stdin', data=data).strip().decode()
        entries.append(f'{modes.get(name, "100644")} {blob}\t{name}\0'.encode())
    git('update-index', '-z', '--index-info', data=b''.join(entries))
    tree = git('write-tree').strip().decode()
    deleted = git('diff-tree', '--no-commit-id', '-r', '--diff-filter=D', '--name-only', '-z', parent, tree)
    if deleted:
        raise ValueError('additive publication unexpectedly deletes files')
    for protected in ('simulation_driver', 'rcm_4090_deploy/mujoco_model/vendor/simulation_driver',
                      'rcm_4090_deploy/mujoco_model/vendor/simulation_driver_g15',
                      'rcm_4090_deploy/data/plan2_plus_original_train_224_20261119'):
        if git('rev-parse', BASE+':'+protected) != git('rev-parse', tree+':'+protected):
            raise ValueError('protected base tree changed: '+protected)
    if not frozen['rcm_4090_deploy/TEST_LOG.md'].startswith(old_log):
        raise ValueError('TEST_LOG history prefix no longer preserved')
    commit = git('commit-tree', tree, '-p', parent, '-m',
                 'Add homogeneous RGB-D replay collection and gated full-horizon evaluation').strip().decode()
    git('update-ref', 'refs/heads/'+branch, commit, parent)
    receipt = {'status': 'committed_verified', 'branch': branch, 'base': BASE, 'parent': parent,
               'commit': commit, 'tree': tree, 'files': len(frozen),
               'timestamp': datetime.now().astimezone().isoformat(),
               'protected_assets_unchanged': True, 'unintended_deletions': 0,
               'weights_and_RGB_retained_locally': True}
    write_json(output/'publication_status.json', receipt)
    if push:
        git('push', 'origin', 'refs/heads/'+branch+':refs/heads/'+branch)
        remote = git('ls-remote', 'origin', 'refs/heads/'+branch).decode().split()[0]
        if remote != commit:
            raise ValueError('remote branch did not match published commit')
        receipt.update(status='pushed_verified', remote_commit=remote)
        write_json(output/'publication_status.json', receipt)
        record('publication', 'pushed_verified', output, receipt)
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
