"""Fail when research_service grows new flat implementation modules."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESEARCH_SERVICE = ROOT / "src" / "quant_platform" / "services" / "research_service"
ALLOWED_TOP_LEVEL_FILES = {"__init__.py"}


def main() -> int:
    offenders = sorted(
        path.relative_to(ROOT).as_posix()
        for path in RESEARCH_SERVICE.glob("*.py")
        if path.name not in ALLOWED_TOP_LEVEL_FILES
    )
    if offenders:
        print("research_service top-level implementation modules are not allowed:")
        for offender in offenders:
            print(f"  - {offender}")
        print("Move new research code into a domain package under research_service/.")
        return 1
    print("research_service layout check passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
