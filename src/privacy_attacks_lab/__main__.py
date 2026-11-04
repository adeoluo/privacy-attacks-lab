"""Entrypoint: `python -m privacy_attacks_lab`."""

from __future__ import annotations

from privacy_attacks_lab import __version__
from privacy_attacks_lab.configuration import settings


def main() -> None:
    header = f"Privacy Attacks Lab v{__version__}"
    status = f"environment={settings.environment}, seed={settings.seed}"
    print(f"{header} - {status}")


if __name__ == "__main__":
    main()
