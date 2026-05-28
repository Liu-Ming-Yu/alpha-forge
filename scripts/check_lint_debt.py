"""Track the production Ruff lint-debt ratchet.

The project still carries broad global Ruff ignores while architecture and
typing hardening are underway.  This script makes that debt visible and hard to
grow: it counts global ignores, per-file ignores, source ``# noqa`` comments,
newly added ``# noqa`` lines, and the currently ignored rule violations.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
import tokenize
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
PYPROJECT = ROOT / "pyproject.toml"
PRODUCTION_ROOTS = (ROOT / "src" / "quant_platform", ROOT / "scripts")
RUFF_TARGETS = ("scripts", "src", "tests")

MAX_GLOBAL_IGNORED_RULES = 0
# 45 (was 44, was 43): pre-existing ANN401 ignores relocated from the
# bootstrap/** umbrella to explicit entries as modules moved layers --
# application/runtime/state.py (state container) and research/** (the research
# composition package, F5). Net lint posture unchanged -- relocation, not growth.
MAX_PER_FILE_IGNORED_RULES = 45
MAX_SOURCE_NOQA = 25
MAX_BARE_SOURCE_NOQA = 0

ALLOWED_GLOBAL_IGNORED_RULES: set[str] = set()

TRACKED_RUFF_RULES = tuple(
    sorted(
        {
            *ALLOWED_GLOBAL_IGNORED_RULES,
            "B007",
            "B905",
            "E501",
            "N802",
            "N803",
            "N806",
            "N815",
            "S101",
            "S104",
            "S105",
            "S106",
            "SIM102",
            "SIM103",
            "SIM105",
            "SIM108",
            "TC001",
            "TC002",
            "TC003",
            "UP017",
            "UP042",
        }
    )
)

MAX_SELECTED_RULE_COUNTS = {
    "ANN001": 0,
    "ANN003": 0,
    "ANN201": 0,
    "ANN202": 0,
    "ANN401": 0,
    "B007": 3,
    "B904": 0,
    "B905": 7,
    "E402": 5,
    "E501": 11,
    "N802": 0,
    "N803": 0,
    "N806": 0,
    "N815": 0,
    "S101": 0,
    "S104": 2,
    "S105": 3,
    "S106": 2,
    "S110": 6,
    "S112": 0,
    "S311": 10,
    "S603": 5,
    "S607": 1,
    "S608": 0,
    "SIM102": 6,
    "SIM103": 2,
    "SIM105": 6,
    "SIM108": 0,
    "TC001": 0,
    "TC002": 0,
    "TC003": 0,
    "UP017": 0,
    "UP042": 0,
}

NOQA_RE = re.compile(r"#\s*noqa(?::\s*([A-Z0-9,\s]+))?")
ADDED_NOQA_RE = re.compile(r"^\+(?!\+\+)[^\"']*#\s*noqa")


@dataclass(frozen=True)
class RuffViolation:
    code: str
    filename: str
    row: int


@dataclass(frozen=True)
class LintDebtReport:
    global_ignored_rules: tuple[str, ...]
    per_file_ignored_rules: int
    source_noqa: int
    bare_source_noqa: int
    added_noqa: tuple[str, ...]
    selected_rule_counts: dict[str, int]
    package_rule_counts: dict[str, dict[str, int]]


def _load_pyproject(pyproject: Path) -> dict[str, Any]:
    data = tomllib.loads(pyproject.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise TypeError("pyproject.toml must parse to a table")
    return data


def _ruff_lint_config(pyproject: Path) -> dict[str, Any]:
    data = _load_pyproject(pyproject)
    lint = data.get("tool", {}).get("ruff", {}).get("lint", {})
    if not isinstance(lint, dict):
        raise TypeError("tool.ruff.lint must be a table")
    return lint


def load_global_ignored_rules(pyproject: Path = PYPROJECT) -> tuple[str, ...]:
    raw_rules = _ruff_lint_config(pyproject).get("ignore", [])
    if not isinstance(raw_rules, list):
        raise TypeError("tool.ruff.lint.ignore must be a list")
    rules: list[str] = []
    for rule in raw_rules:
        if not isinstance(rule, str):
            raise TypeError("tool.ruff.lint.ignore entries must be strings")
        rules.append(rule)
    return tuple(rules)


def load_per_file_ignore_count(pyproject: Path = PYPROJECT) -> int:
    raw_ignores = _ruff_lint_config(pyproject).get("per-file-ignores", {})
    if not isinstance(raw_ignores, dict):
        raise TypeError("tool.ruff.lint.per-file-ignores must be a table")
    total = 0
    for rules in raw_ignores.values():
        if not isinstance(rules, list):
            raise TypeError("per-file-ignore entries must be lists")
        total += len(rules)
    return total


def collect_noqa_counts(roots: tuple[Path, ...] = PRODUCTION_ROOTS) -> tuple[int, int]:
    total = 0
    bare = 0
    for root in roots:
        if not root.exists():
            continue
        for path in sorted(root.rglob("*.py")):
            if "__pycache__" in path.parts:
                continue
            with tokenize.open(path) as handle:
                tokens = tokenize.generate_tokens(handle.readline)
                comments = [token.string for token in tokens if token.type == tokenize.COMMENT]
            for comment in comments:
                match = NOQA_RE.search(comment)
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


def collect_added_noqa(root: Path = ROOT) -> tuple[str, ...]:
    result = subprocess.run(
        [_git_executable(), "diff", "--unified=0", "--", "src/quant_platform", "scripts"],
        cwd=root,
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    if result.returncode not in {0, 1}:
        raise RuntimeError(result.stderr.strip() or "git diff failed")
    added: list[str] = []
    current_file = ""
    for line in result.stdout.splitlines():
        if line.startswith("+++ b/"):
            current_file = line.removeprefix("+++ b/")
            continue
        if ADDED_NOQA_RE.search(line):
            added.append(f"{current_file}: {line[1:].strip()}")
    return tuple(added)


def _package_for_filename(filename: str) -> str:
    path = Path(filename)
    parts = path.parts
    if "scripts" in parts:
        return "scripts"
    if "src" in parts and "quant_platform" in parts:
        idx = parts.index("quant_platform")
        if idx + 1 < len(parts):
            return f"src/quant_platform/{parts[idx + 1]}"
        return "src/quant_platform"
    if "tests" in parts:
        return "tests"
    return "other"


def _parse_ruff_violations(payload: str) -> tuple[RuffViolation, ...]:
    raw_items = json.loads(payload or "[]")
    if not isinstance(raw_items, list):
        raise TypeError("ruff JSON output must be a list")
    violations: list[RuffViolation] = []
    for item in raw_items:
        if not isinstance(item, dict):
            raise TypeError("ruff violation entries must be objects")
        code = item.get("code")
        filename = item.get("filename")
        location = item.get("location", {})
        if not isinstance(code, str) or not isinstance(filename, str):
            raise TypeError("ruff violation entries require string code and filename")
        row = 0
        if isinstance(location, dict):
            raw_row = location.get("row", 0)
            row = raw_row if isinstance(raw_row, int) else 0
        violations.append(RuffViolation(code=code, filename=filename, row=row))
    return tuple(violations)


def collect_selected_rule_counts(
    *,
    root: Path = ROOT,
    rules: tuple[str, ...] = TRACKED_RUFF_RULES,
    targets: tuple[str, ...] = RUFF_TARGETS,
) -> tuple[dict[str, int], dict[str, dict[str, int]]]:
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "ruff",
            "check",
            *targets,
            "--select",
            ",".join(rules),
            "--config",
            "lint.ignore=[]",
            "--output-format",
            "json",
        ],
        cwd=root,
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    if result.returncode not in {0, 1}:
        raise RuntimeError(result.stderr.strip() or "ruff selected-rule probe failed")

    counts = {rule: 0 for rule in rules}
    package_counts: dict[str, dict[str, int]] = {}
    for violation in _parse_ruff_violations(result.stdout):
        counts[violation.code] = counts.get(violation.code, 0) + 1
        package = _package_for_filename(violation.filename)
        package_counts.setdefault(package, {})
        package_counts[package][violation.code] = package_counts[package].get(violation.code, 0) + 1
    return counts, package_counts


def build_report(root: Path = ROOT, *, include_ruff_probe: bool = True) -> LintDebtReport:
    total_noqa, bare_noqa = collect_noqa_counts((root / "src" / "quant_platform", root / "scripts"))
    selected_counts: dict[str, int] = {}
    package_counts: dict[str, dict[str, int]] = {}
    if include_ruff_probe:
        selected_counts, package_counts = collect_selected_rule_counts(root=root)
    return LintDebtReport(
        global_ignored_rules=load_global_ignored_rules(root / "pyproject.toml"),
        per_file_ignored_rules=load_per_file_ignore_count(root / "pyproject.toml"),
        source_noqa=total_noqa,
        bare_source_noqa=bare_noqa,
        added_noqa=collect_added_noqa(root),
        selected_rule_counts=selected_counts,
        package_rule_counts=package_counts,
    )


def validate_report(
    report: LintDebtReport,
    *,
    allow_new_noqa: bool,
    validate_ruff_probe: bool = True,
) -> list[str]:
    errors: list[str] = []
    ignored = set(report.global_ignored_rules)
    unexpected = sorted(ignored - ALLOWED_GLOBAL_IGNORED_RULES)
    if unexpected:
        errors.append(f"unexpected global Ruff ignored rules: {', '.join(unexpected)}")
    if len(report.global_ignored_rules) > MAX_GLOBAL_IGNORED_RULES:
        errors.append(
            f"global Ruff ignore count {len(report.global_ignored_rules)} > "
            f"{MAX_GLOBAL_IGNORED_RULES}"
        )
    if report.per_file_ignored_rules > MAX_PER_FILE_IGNORED_RULES:
        errors.append(
            f"Ruff per-file ignore count {report.per_file_ignored_rules} > "
            f"{MAX_PER_FILE_IGNORED_RULES}"
        )
    if report.source_noqa > MAX_SOURCE_NOQA:
        errors.append(f"source # noqa count {report.source_noqa} > {MAX_SOURCE_NOQA}")
    if report.bare_source_noqa > MAX_BARE_SOURCE_NOQA:
        errors.append(
            f"bare source # noqa count {report.bare_source_noqa} > {MAX_BARE_SOURCE_NOQA}"
        )
    if report.added_noqa and not allow_new_noqa:
        errors.append("new source # noqa lines are blocked outside cleanup windows")
        errors.extend(f"  - {line}" for line in report.added_noqa)
    if validate_ruff_probe:
        for rule, count in sorted(report.selected_rule_counts.items()):
            max_count = MAX_SELECTED_RULE_COUNTS.get(rule)
            if max_count is not None and count > max_count:
                errors.append(f"selected Ruff rule {rule} count {count} > {max_count}")
    return errors


def _format_top_package_counts(package_counts: dict[str, dict[str, int]]) -> str:
    if not package_counts:
        return "none"
    parts: list[str] = []
    for package, counts in sorted(package_counts.items()):
        total = sum(counts.values())
        top_rules = sorted(counts.items(), key=lambda item: -item[1])[:3]
        worst = ", ".join(f"{rule}={count}" for rule, count in top_rules)
        parts.append(f"{package}: {total} ({worst})")
    return "; ".join(parts)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--allow-new-noqa",
        action="store_true",
        help="permit newly added source noqa lines in the current git diff",
    )
    parser.add_argument(
        "--skip-ruff-probe",
        action="store_true",
        help="skip selected-rule counting for fast local diagnostics",
    )
    args = parser.parse_args(argv)

    report = build_report(include_ruff_probe=not args.skip_ruff_probe)
    errors = validate_report(
        report,
        allow_new_noqa=args.allow_new_noqa,
        validate_ruff_probe=not args.skip_ruff_probe,
    )
    if errors:
        print("Lint-debt ratchet failed:", file=sys.stderr)
        for error in errors:
            print(f"  - {error}", file=sys.stderr)
        return 1

    print(
        "Lint-debt ratchet passed: "
        f"{len(report.global_ignored_rules)} global ignored rules, "
        f"{report.per_file_ignored_rules} per-file ignored rules, "
        f"{report.source_noqa} source noqa comments, "
        f"{report.bare_source_noqa} bare source noqa comments."
    )
    if report.package_rule_counts:
        package_summary = _format_top_package_counts(report.package_rule_counts)
        print(f"Selected-rule debt by package: {package_summary}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
