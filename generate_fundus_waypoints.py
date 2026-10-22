"""Generate three surface targets from the configured, oriented MuJoCo eye."""
import argparse
import json
from pathlib import Path

from rcm_fundus import make_fundus_targets
from rcm_navigation import write_json
from rcm_reference import HERE, Reference


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=HERE/"config.json")
    parser.add_argument("--output", type=Path, default=HERE/"examples"/"three_waypoints.json")
    args = parser.parse_args()
    reference = Reference(json.loads(args.config.read_text(encoding="utf-8")))
    payload = make_fundus_targets(reference)
    write_json(args.output, payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
