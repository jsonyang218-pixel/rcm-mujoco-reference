"""Publish v4 scripts and lightweight evidence with an additive Git tree."""
from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import os
from pathlib import Path
import subprocess
import tempfile

from lerobot_training.navigation_overlay import file_sha256
from scripts.research.homog_replay_publish import preserve_log_history
from scripts.research.overnight_label_fix_v4_6h import (
    BASE,
    BRANCH,
    DATASETS,
    OUTPUTS,
    ROOT,
    SOURCE_DATA,
    SOURCE_REPO,
    read,
    record,
    write_json,
)

SSH_KEY = Path("/home/him1/.ssh/rcm_github_deploy_ed25519")
ALLOWED_SUFFIXES = {".json", ".jsonl", ".csv", ".png", ".md", ".log", ".txt"}
PRIVATE_KEY_MARKERS = (
    b"-----BEGIN " + b"OPENSSH PRIVATE KEY-----",
    b"-----BEGIN " + b"RSA PRIVATE KEY-----",
)
SOURCE_FILES = (
    "scripts/research/overnight_label_fix_v4_6h.py",
    "scripts/research/overnight_label_fix_v4_publish.py",
    "scripts/research/homog_label_fix_repair.py",
    "scripts/research/homog_replay_evaluate.py",
    "scripts/research/homog_replay_common.py",
    "lerobot_training/dit_experience_replay.py",
)


def _lightweight_files(root, *, allowed_suffixes=ALLOWED_SUFFIXES):
    root = Path(root)
    if not root.exists():
        return []
    result = []
    for path in root.rglob("*"):
        relative = path.relative_to(root)
        if not path.is_file() or path.suffix not in allowed_suffixes:
            continue
        if "shards" in relative.parts or "raw" in relative.parts:
            continue
        result.append(path)
    return sorted(result)


def publication_files(output):
    files = {}
    for relative in SOURCE_FILES:
        path = ROOT / relative
        if not path.is_file():
            raise FileNotFoundError(path)
        files["rcm_4090_deploy/" + relative] = path
    files["rcm_4090_deploy/TEST_LOG.md"] = ROOT / "TEST_LOG.md"
    output = Path(output).resolve()
    for root in (output, OUTPUTS, *DATASETS.values()):
        for path in _lightweight_files(root):
            relative = path.relative_to(ROOT)
            files["rcm_4090_deploy/" + relative.as_posix()] = path
    return files


def _git_env(index_path):
    env = os.environ.copy()
    env["GIT_SSH_COMMAND"] = (
        f"ssh -i {SSH_KEY} -o IdentitiesOnly=yes -o BatchMode=yes"
    )
    env["GIT_INDEX_FILE"] = str(index_path)
    return env


def publish(output, *, push=False):
    output = Path(output).resolve()
    if not SSH_KEY.is_file():
        raise FileNotFoundError(SSH_KEY)
    with tempfile.NamedTemporaryFile(
        prefix="homog_label_fix_v4_index_", dir=output, delete=False
    ) as temporary:
        index_path = Path(temporary.name)
    index_path.unlink()
    env = _git_env(index_path)

    def git(*args, data=None, check=True):
        return subprocess.run(
            ["git", "-C", str(SOURCE_REPO), *args],
            input=data,
            capture_output=True,
            check=check,
            env=env,
        ).stdout

    try:
        parent = git("rev-parse", BRANCH).decode().strip()
        if git("merge-base", parent, BASE).decode().strip() != BASE:
            raise ValueError("v4 publication branch does not descend from 703003bc")
        local_log = (ROOT / "TEST_LOG.md").read_bytes()
        base_log = git("show", BASE + ":rcm_4090_deploy/TEST_LOG.md")
        parent_log = git("show", parent + ":rcm_4090_deploy/TEST_LOG.md")
        published_log, log_merge = preserve_log_history(base_log, local_log)
        if not published_log.startswith(parent_log):
            raise ValueError("publication would lose prior TEST_LOG history")

        frozen = {
            path: source.read_bytes()
            for path, source in publication_files(output).items()
        }
        if any(path.endswith((".pth", ".npz")) for path in frozen):
            raise ValueError("weight or raw dataset entered the publication allowlist")
        frozen["rcm_4090_deploy/TEST_LOG.md"] = published_log
        manifest = {
            "status": "prepared",
            "timestamp": datetime.now().astimezone().isoformat(timespec="seconds"),
            "base": BASE,
            "branch": BRANCH,
            "parent": parent,
            "policy": (
                "Explicit lightweight allowlist: scripts, audits, summaries, "
                "charts and TEST_LOG. Weights, raw NPZ datasets and shard assets excluded."
            ),
            "TEST_LOG_preservation": {
                **log_merge,
                "base_sha256": hashlib.sha256(base_log).hexdigest(),
                "parent_sha256": hashlib.sha256(parent_log).hexdigest(),
                "local_sha256": hashlib.sha256(local_log).hexdigest(),
                "published_sha256": hashlib.sha256(published_log).hexdigest(),
            },
            "files": [
                {
                    "path": path,
                    "bytes": len(value),
                    "sha256": hashlib.sha256(value).hexdigest(),
                }
                for path, value in sorted(frozen.items())
            ],
            "retained_large_assets": [
                {
                    "path": str(path),
                    "bytes": path.stat().st_size,
                    "sha256": file_sha256(path),
                }
                for root in (SOURCE_DATA, OUTPUTS, *DATASETS.values())
                if Path(root).exists()
                for path in sorted(Path(root).rglob("*"))
                if path.is_file() and path.suffix in (".pth", ".npz")
            ],
        }
        manifest_path = (
            output / "homog_label_fix_20261124_v4_publication_manifest.json"
        )
        write_json(manifest_path, manifest)
        frozen[
            "rcm_4090_deploy/"
            + manifest_path.relative_to(ROOT).as_posix()
        ] = manifest_path.read_bytes()

        git("read-tree", parent)
        modes = {}
        for entry in git("ls-tree", "-r", "-z", parent).split(b"\0"):
            if not entry:
                continue
            metadata, path = entry.split(b"\t", 1)
            modes[path.decode()] = metadata.split()[0].decode()
        index_entries = []
        for name, data in sorted(frozen.items()):
            if len(data) >= 95_000_000:
                raise ValueError(f"artifact exceeds GitHub blob safety limit: {name}")
            if any(marker in data for marker in PRIVATE_KEY_MARKERS):
                raise ValueError("private key material rejected")
            blob = git("hash-object", "-w", "--stdin", data=data).strip().decode()
            mode = modes.get(name, "100644")
            index_entries.append(f"{mode} {blob}\t{name}\0".encode())
        git("update-index", "-z", "--index-info", data=b"".join(index_entries))
        tree = git("write-tree").decode().strip()
        deleted = git(
            "diff-tree", "--no-commit-id", "-r", "--diff-filter=D",
            "--name-only", "-z", parent, tree,
        )
        if deleted:
            raise ValueError("additive publication unexpectedly deletes files")
        protected = (
            "rcm_4090_deploy/outputs/homog_label_fix_20261123_v3",
            "rcm_4090_deploy/outputs/overnight_c_fix_symlog_081234",
            "rcm_4090_deploy/data/homog_clean_replay_20261123_mixed_1600_600_12_hold_v2",
        )

        def tree_entry(ref, path):
            result = subprocess.run(
                ["git", "-C", str(SOURCE_REPO), "rev-parse", f"{ref}:{path}"],
                capture_output=True,
                check=False,
                env=env,
            )
            return result.stdout.decode().strip() if result.returncode == 0 else None

        for path in protected:
            before = tree_entry(BASE, path)
            after = tree_entry(tree, path)
            if before != after:
                raise ValueError(f"protected frozen asset tree changed: {path}")

        commit = git(
            "commit-tree", tree, "-p", parent, "-m",
            "Add six-hour homog label-fix v4 gated experiment",
        ).decode().strip()
        git("update-ref", f"refs/heads/{BRANCH}", commit, parent)
        receipt = {
            "status": "committed_verified",
            "branch": BRANCH,
            "base": BASE,
            "parent": parent,
            "commit": commit,
            "tree": tree,
            "files": len(frozen),
            "timestamp": datetime.now().astimezone().isoformat(timespec="seconds"),
            "unintended_deletions": 0,
            "weights_raw_data_and_shards_excluded": True,
            "protected_assets_unchanged": True,
        }
        receipt_path = (
            output / "homog_label_fix_20261124_v4_publication_status.json"
        )
        write_json(receipt_path, receipt)
        if push:
            git("push", "origin", f"refs/heads/{BRANCH}:refs/heads/{BRANCH}")
            remote = git(
                "ls-remote", "origin", f"refs/heads/{BRANCH}"
            ).decode().split()[0]
            if remote != commit:
                raise ValueError("remote branch does not match publication commit")
            receipt.update(status="pushed_verified", remote_commit=remote)
            write_json(receipt_path, receipt)
            record("publication", "pushed_verified", output, receipt)
        print("Publication:", receipt, flush=True)
        return receipt
    finally:
        try:
            index_path.unlink()
        except FileNotFoundError:
            pass


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--push", action="store_true")
    args = parser.parse_args()
    publish(args.output_dir.resolve(), push=args.push)


if __name__ == "__main__":
    main()
