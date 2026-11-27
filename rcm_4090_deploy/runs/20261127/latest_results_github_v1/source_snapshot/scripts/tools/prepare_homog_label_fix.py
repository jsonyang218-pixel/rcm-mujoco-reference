"""Launch the authorized planning-file-6 label repair workflow."""
from __future__ import annotations

import sys

from scripts.research import homog_label_fix_pipeline as pipeline


def main():
    # Preserve the established module interface while the implementation lives
    # with the other homogeneous-replay orchestration code.
    pipeline.main()


if __name__ == '__main__':
    main()
