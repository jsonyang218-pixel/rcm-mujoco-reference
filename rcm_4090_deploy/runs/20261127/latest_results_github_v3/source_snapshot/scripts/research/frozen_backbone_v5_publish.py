"""Publish the v5 experiment with an explicit lightweight Git allowlist."""
from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import os
from pathlib import Path
import subprocess
import tempfile

from lerobot_training.navigation_overlay import file_sha256
from scripts.research.frozen_backbone_v5_3h import (
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
from scripts.research.homog_replay_publish import preserve_log_history

SSH_KEY = Path("/home/him1/.ssh/rcm_github_deploy_ed25519")
ALLOWED_SUFFIXES = {".json", ".jsonl", ".csv", ".png", ".md", ".log", ".txt"}
EXCLUDED_SUFFIXES = {".pth", ".npz", ".npy"}
SECRET_MARKERS = (
    b"-----BEGIN " + b"OPENSSH PRIVATE KEY-----",
    b"-----BEGIN " + b"RSA PRIVATE KEY-----",
    b"-----BEGIN " + b"EC PRIVATE KEY-----",
    b"-----BEGIN " + b"DSA PRIVATE KEY-----",
    b"github" + b"_pat_",
    b"gh" + b"p_",
    b"AK" + b"IA",
    b"sk-" + b"proj-",
)
SOURCE_FILES = (
    "scripts/research/frozen_backbone_v5_3h.py",
    "scripts/research/frozen_backbone_v5_publish.py",
    "scripts/research/homog_label_fix_repair.py",
    "scripts/research/homog_replay_evaluate.py",
    "scripts/research/homog_replay_common.py",
    "scripts/research/homog_replay_publish.py",
    "lerobot_training/dit_experience_replay.py",
    "lerobot_training/dataset_contract.py",
    "tests/test_frozen_backbone_v5.py",
    ".gitignore",
)
DATASET_EVIDENCE = (
    "dataset_audit.json",
    "dataset_receipt.json",
    "metadata.json",
)


def _excluded(relative):
    parts = set(Path(relative).parts)
    return bool(parts & {"shards", "raw"})


def _lightweight_files(root):
    root = Path(root)
    if not root.exists():
        return []
    result = []
    for path in root.rglob("*"):
        relative = path.relative_to(root)
        if (
            not path.is_file()
            or path.suffix in EXCLUDED_SUFFIXES
            or path.suffix not in ALLOWED_SUFFIXES
            or _excluded(relative)
        ):
            continue
        result.append(path)
    return sorted(result)


def _smoke_evidence(path, relative):
    parts = Path(relative).parts
    if not parts or parts[0] != "smoke":
        return True
    name = path.name
    if name.endswith(("_smoke_receipt.json", "_smoke.log")):
        return True
    return name in {
        "training_config.json",
        "training_history.json",
        "training_manifest.json",
        "training_metrics.jsonl",
        "training_receipt.json",
    }


def publication_files(output):
    files = {}
    for relative in SOURCE_FILES:
        path = ROOT / relative
        if not path.is_file():
            raise FileNotFoundError(path)
        files["rcm_4090_deploy/" + relative] = path
    files["rcm_4090_deploy/TEST_LOG.md"] = ROOT / "TEST_LOG.md"
    output = Path(output).resolve()
    for path in _lightweight_files(output):
        relative = path.relative_to(output)
        if _smoke_evidence(path, relative):
            files["rcm_4090_deploy/" + path.relative_to(ROOT).as_posix()] = path
    for path in _lightweight_files(OUTPUTS):
        files["rcm_4090_deploy/" + path.relative_to(ROOT).as_posix()] = path
    for dataset in DATASETS.values():
        for name in DATASET_EVIDENCE:
            path = dataset / name
            if path.is_file():
                files[
                    "rcm_4090_deploy/" + path.relative_to(ROOT).as_posix()
                ] = path
    return files


def scan_secret_material(files):
    """Reject secret-like bytes without ever printing their contents."""
    rejected = []
    for name, path in files.items():
        data = Path(path).read_bytes()
        if any(marker in data for marker in SECRET_MARKERS):
            rejected.append(name)
    if rejected:
        raise ValueError(f"secret-like material rejected: {rejected}")


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
        prefix="homog_label_fix_v5_index_", dir=output, delete=False
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
            raise ValueError("v5 branch does not descend from f49a3a27")
        local_log = (ROOT / "TEST_LOG.md").read_bytes()
        base_log = git("show", BASE + ":rcm_4090_deploy/TEST_LOG.md")
        parent_log = git("show", parent + ":rcm_4090_deploy/TEST_LOG.md")
        published_log, log_merge = preserve_log_history(base_log, local_log)
        if not published_log.startswith(parent_log):
            raise ValueError("publication would lose prior TEST_LOG history")
        files = publication_files(output)
        if any(
            Path(name).suffix in EXCLUDED_SUFFIXES
            or _excluded(name)
            for name in files
        ):
            raise ValueError("weight/raw/shard asset entered publication allowlist")
        scan_secret_material(files)
        frozen = {name: path.read_bytes() for name, path in files.items()}
        frozen["rcm_4090_deploy/TEST_LOG.md"] = published_log
        retained = [
            {
                "path": str(path),
                "bytes": path.stat().st_size,
                "sha256": file_sha256(path),
            }
            for root in (SOURCE_DATA, OUTPUTS, *DATASETS.values())
            if Path(root).exists()
            for path in sorted(Path(root).rglob("*"))
            if path.is_file()
            and path.suffix in EXCLUDED_SUFFIXES
            and not _excluded(path.relative_to(root))
        ]
        manifest = {
            "status": "prepared",
            "timestamp": datetime.now().astimezone().isoformat(timespec="seconds"),
            "base": BASE,
            "branch": BRANCH,
            "parent": parent,
            "policy": (
                "Explicit lightweight allowlist: code, tests, audits, "
                "summaries, charts, CSV, logs and TEST_LOG. Weights, NPZ/NPY "
                "data and raw trajectory/shard directories are excluded."
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
                    "path": name,
                    "bytes": len(data),
                    "sha256": hashlib.sha256(data).hexdigest(),
                }
                for name, data in sorted(frozen.items())
            ],
            "retained_large_assets": retained,
        }
        manifest_path = (
            output / "homog_label_fix_20261124_v5_publication_manifest.json"
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
                raise ValueError(
                    f"artifact exceeds GitHub blob safety limit: {name}"
                )
            if any(marker in data for marker in SECRET_MARKERS):
                raise ValueError(f"secret-like material rejected: {name}")
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
            "rcm_4090_deploy/outputs/homog_label_fix_20261124_v4",
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
            if tree_entry(BASE, path) != tree_entry(tree, path):
                raise ValueError(f"protected frozen asset tree changed: {path}")
        commit = git(
            "commit-tree", tree, "-p", parent, "-m",
            "Add frozen-backbone v5 sprint and validation",
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
            "secret_scan_passed": True,
            "protected_assets_unchanged": True,
        }
        receipt_path = (
            output / "homog_label_fix_20261124_v5_publication_status.json"
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
