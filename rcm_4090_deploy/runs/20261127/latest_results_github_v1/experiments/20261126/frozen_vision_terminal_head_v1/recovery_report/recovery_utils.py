from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import os

ROOT = Path('/home/him1/mujoco_bc/rcm_4090_deploy')
TASK = ROOT / 'runs/20261126/frozen_vision_terminal_head_v1'
OUT = TASK / 'recovery_report'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def save(name, value):
    path = OUT / name
    data = json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n'
    with path.open('x') as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())
    assert path.read_text() == data


def record(check, result):
    row = dict(time_utc=datetime.now(timezone.utc).isoformat(),
               task=TASK.name, recovery_check=check, result=result,
               evidence=str(OUT), config=str(TASK / 'experiment_card.json'),
               additional_training_updates=0, additional_model_forwards=0,
               additional_simulation_cycles=0)
    line = '\n- ' + json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n'
    receipts = []
    for path in (TASK / 'TEST_LOG.md', ROOT / 'TEST_LOG.md'):
        with path.open('a+b') as f:
            f.seek(0, 2)
            offset = f.tell()
            f.write(line.encode())
            f.flush()
            os.fsync(f.fileno())
            f.seek(offset)
            assert f.read(len(line.encode())) == line.encode()
        receipts.append(dict(path=str(path), offset=offset,
                             bytes=len(line.encode()), exact_readback=True))
    with (OUT / 'test_log_readback_receipts.jsonl').open('a') as f:
        f.write(json.dumps(dict(check=check, receipts=receipts)) + '\n')
    print(check, json.dumps(result, ensure_ascii=False))
