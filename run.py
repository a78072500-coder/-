"""PyInstaller 진입점 (python -m jp_subtitle 과 같음)."""

import sys

from jp_subtitle.__main__ import run

if __name__ == "__main__":
    sys.exit(run())
