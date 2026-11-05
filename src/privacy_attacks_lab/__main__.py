"""Entrypoint: ``python -m privacy_attacks_lab``."""

from __future__ import annotations

import sys

from privacy_attacks_lab.cli import main

if __name__ == "__main__":
    sys.exit(main())
