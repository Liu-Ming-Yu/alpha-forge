"""Fail new oversized production modules unless explicitly excepted.

This is a P2 architecture ratchet, not a one-time cleanup script.  Production
modules should stay small enough to review, test, and split along ownership
boundaries before they grow into orchestration junk drawers.  Temporary
exceptions must carry a reason and are treated as stale once they disappear or
fall back under the threshold.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MAX_LINES = 300

ARCHITECTURE_EXCEPTIONS: dict[str, str] = {
    "src/quant_platform/application/research/requests.py": (
        "Flat catalog of frozen research request DTOs (one per subcommand); "
        "field declarations only, no orchestration logic."
    ),
    "src/quant_platform/cli/commands/research/request_factories/__init__.py": (
        "Flat 1:1 argparse-namespace -> research DTO factories; "
        "mechanical field mapping, no orchestration logic."
    ),
}


@dataclass(frozen=True)
class ModuleSizeViolation:
    """One module over the threshold without an architecture exception."""

    path: str
    lines: int
    max_lines: int


def _posix_relative(path: Path, root: Path) -> str:
    return path.relative_to(root).as_posix()


def _python_files(src_root: Path) -> list[Path]:
    return [
        path
        for path in sorted(src_root.rglob("*.py"))
        if "__pycache__" not in path.parts and "alembic" not in path.parts
    ]


def _line_count(path: Path) -> int:
    return len(path.read_text(encoding="utf-8").splitlines())


def collect_module_size_violations(
    *,
    root: Path = ROOT,
    src_root: Path | None = None,
    max_lines: int = DEFAULT_MAX_LINES,
    exceptions: dict[str, str] | None = None,
) -> tuple[list[ModuleSizeViolation], list[str]]:
    """Return oversized modules and stale exceptions."""
    root = root.resolve()
    src_root = (src_root or root / "src" / "quant_platform").resolve()
    if exceptions is None:
        exceptions = ARCHITECTURE_EXCEPTIONS

    seen: set[str] = set()
    stale_below_threshold: list[str] = []
    violations: list[ModuleSizeViolation] = []
    for path in _python_files(src_root):
        rel = _posix_relative(path, root)
        lines = _line_count(path)
        if rel in exceptions:
            seen.add(rel)
            if lines <= max_lines:
                stale_below_threshold.append(rel)
            continue
        if lines > max_lines:
            violations.append(ModuleSizeViolation(rel, lines, max_lines))

    stale_exceptions = sorted((set(exceptions) - seen) | set(stale_below_threshold))
    return violations, stale_exceptions


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--src-root", type=Path, default=None)
    parser.add_argument("--max-lines", type=int, default=DEFAULT_MAX_LINES)
    args = parser.parse_args(argv)

    violations, stale = collect_module_size_violations(
        root=args.root,
        src_root=args.src_root,
        max_lines=args.max_lines,
    )
    if violations or stale:
        if violations:
            print(f"Production modules over {args.max_lines} lines without exception:")
            for violation in violations:
                print(f"  - {violation.path}: {violation.lines} lines")
        if stale:
            print("Stale module-size exceptions:")
            for path in stale:
                print(f"  - {path}")
        return 1

    print(f"Module-size check passed at {args.max_lines} lines.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
