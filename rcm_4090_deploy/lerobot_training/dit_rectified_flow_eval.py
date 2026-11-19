#!/usr/bin/env python3
"""Evaluate an original DiT rectified-Flow checkpoint."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import eval_policy


def main(argv=None):
    arguments = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("eval",))
    parser.parse_args(arguments[:1])
    sys.argv = [eval_policy.__file__, *arguments[1:]]
    return eval_policy.main()


if __name__ == "__main__":
    raise SystemExit(main())
