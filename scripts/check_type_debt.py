"""Track the production type-debt ratchet.

This script intentionally measures coarse, hard-to-game signals before mypy
fully owns the whole source tree: global disabled error codes, source
``# type: ignore`` comments, bare ignores, and newly added ignores in the
current git diff.
"""

from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
import tomllib
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PYPROJECT = ROOT / "pyproject.toml"
SOURCE_ROOT = ROOT / "src" / "quant_platform"

MAX_GLOBAL_DISABLED_ERROR_CODES = 2
MAX_SOURCE_TYPE_IGNORES = 0
MAX_BARE_SOURCE_TYPE_IGNORES = 0

ALLOWED_GLOBAL_DISABLED_ERROR_CODES = {
    "import-not-found",
    "import-untyped",
}

TYPE_IGNORE_RE = re.compile(r"#\s*type:\s*ignore(?:\[([^\]]+)\])?")
ADDED_IGNORE_RE = re.compile(r"^\+(?!\+\+).*#\s*type:\s*ignore")


@dataclass(frozen=True)
class TypeDebtReport:
    global_disabled_error_codes: tuple[str, ...]
    source_type_ignores: int
    bare_source_type_ignores: int
    added_type_ignores: tuple[str, ...]


def load_global_disabled_error_codes(pyproject: Path = PYPROJECT) -> tuple[str, ...]:
    data = tomllib.loads(pyproject.read_text(encoding="utf-8"))
    mypy = data.get("tool", {}).get("mypy", {})
    raw_codes = mypy.get("disable_error_code", [])
    if not isinstance(raw_codes, list):
        raise TypeError("tool.mypy.disable_error_code must be a list")
    codes: list[str] = []
    for code in raw_codes:
        if not isinstance(code, str):
            raise TypeError("tool.mypy.disable_error_code entries must be strings")
        codes.append(code)
    return tuple(codes)


def collect_type_ignore_counts(source_root: Path = SOURCE_ROOT) -> tuple[int, int]:
    total = 0
    bare = 0
    for path in sorted(source_root.rglob("*.py")):
        if "__pycache__" in path.parts:
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            match = TYPE_IGNORE_RE.search(line)
            if match is None:
                continue
            total += 1
            if match.group(1) is None:
                bare += 1
    return total, bare


def _git_executable() -> str:
    git = shutil.which("git")
    if git is None:
        raise RuntimeError("git executable not found")
    return git


def collect_added_type_ignores(root: Path = ROOT) -> tuple[str, ...]:
    result = subprocess.run(
        [_git_executable(), "diff", "--unified=0", "--", "src/quant_platform"],
        cwd=root,
        check=False,
        capture_output=True,
        encoding="utf-8",
        errors="replace",
    )
    if result.returncode not in {0, 1}:
        raise RuntimeError(result.stderr.strip() or "git diff failed")
    added: list[str] = []
    current_file = ""
    for line in result.stdout.splitlines():
        if line.startswith("+++ b/"):
            current_file = line.removeprefix("+++ b/")
            continue
        if ADDED_IGNORE_RE.search(line):
            added.append(f"{current_file}: {line[1:].strip()}")
    return tuple(added)


def build_report(root: Path = ROOT) -> TypeDebtReport:
    source_root = root / "src" / "quant_platform"
    total, bare = collect_type_ignore_counts(source_root)
    return TypeDebtReport(
        global_disabled_error_codes=load_global_disabled_error_codes(root / "pyproject.toml"),
        source_type_ignores=total,
        bare_source_type_ignores=bare,
        added_type_ignores=collect_added_type_ignores(root),
    )


def validate_report(report: TypeDebtReport, *, allow_new_ignores: bool) -> list[str]:
    errors: list[str] = []
    disabled = set(report.global_disabled_error_codes)
    unexpected = sorted(disabled - ALLOWED_GLOBAL_DISABLED_ERROR_CODES)
    if unexpected:
        errors.append(f"unexpected global mypy disabled error codes: {', '.join(unexpected)}")
    if len(report.global_disabled_error_codes) > MAX_GLOBAL_DISABLED_ERROR_CODES:
        errors.append(
            "global mypy disabled error-code count "
            f"{len(report.global_disabled_error_codes)} > {MAX_GLOBAL_DISABLED_ERROR_CODES}"
        )
    if report.source_type_ignores > MAX_SOURCE_TYPE_IGNORES:
        errors.append(
            f"source # type: ignore count {report.source_type_ignores} > {MAX_SOURCE_TYPE_IGNORES}"
        )
    if report.bare_source_type_ignores > MAX_BARE_SOURCE_TYPE_IGNORES:
        errors.append(
            f"bare source # type: ignore count {report.bare_source_type_ignores} > "
            f"{MAX_BARE_SOURCE_TYPE_IGNORES}"
        )
    if report.added_type_ignores and not allow_new_ignores:
        errors.append("new source # type: ignore lines are blocked outside cleanup windows")
        errors.extend(f"  - {line}" for line in report.added_type_ignores)
    return errors


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--allow-new-ignores",
        action="store_true",
        help="permit newly added source type ignores in the current git diff",
    )
    args = parser.parse_args(argv)

    report = build_report()
    errors = validate_report(report, allow_new_ignores=args.allow_new_ignores)
    if errors:
        print("Type-debt ratchet failed:", file=sys.stderr)
        for error in errors:
            print(f"  - {error}", file=sys.stderr)
        return 1

    print(
        "Type-debt ratchet passed: "
        f"{len(report.global_disabled_error_codes)} global disabled codes, "
        f"{report.source_type_ignores} source ignores, "
        f"{report.bare_source_type_ignores} bare source ignores."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
