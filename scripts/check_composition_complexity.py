"""Ratchet operator composition so use cases stay out of bootstrap adapters."""

from __future__ import annotations

import argparse
import ast
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
COMPOSITION_ROOT_RELATIVE = Path("src") / "quant_platform" / "application" / "operator_use_cases"
ADAPTER_ROOT_RELATIVE = Path("src") / "quant_platform" / "bootstrap" / "operator_adapters"
# The research adapter lives in the research composition package rather than the
# bootstrap operator_adapters root; ratchet it under the same adapter rules.
EXTRA_ADAPTER_FILES_RELATIVE = (
    Path("src") / "quant_platform" / "research" / "adapters" / "__init__.py",
)
DEFAULT_COMPOSITION_ROOT = ROOT / COMPOSITION_ROOT_RELATIVE
DEFAULT_ADAPTER_ROOT = ROOT / ADAPTER_ROOT_RELATIVE
DEFAULT_MAX_LINES = 220
DEFAULT_MAX_REGISTRATIONS = 12


@dataclass(frozen=True)
class CompositionComplexityViolation:
    """One composition module over an explicit complexity threshold."""

    path: str
    reason: str
    value: int
    limit: int


def _posix_relative(path: Path, root: Path) -> str:
    return path.relative_to(root).as_posix()


def _line_count(path: Path) -> int:
    return len(path.read_text(encoding="utf-8").splitlines())


def _registration_count(path: Path) -> int:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    count = 0
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if isinstance(func, ast.Attribute) and func.attr == "register":
            count += 1
    return count


def collect_composition_complexity_violations(
    *,
    root: Path = ROOT,
    composition_root: Path | None = None,
    adapter_root: Path | None = None,
    max_lines: int = DEFAULT_MAX_LINES,
    max_registrations: int = DEFAULT_MAX_REGISTRATIONS,
) -> list[CompositionComplexityViolation]:
    """Return composition modules that are too large or register too much."""
    root = root.resolve()
    composition_root = (composition_root or root / COMPOSITION_ROOT_RELATIVE).resolve()
    adapter_root = (adapter_root or root / ADAPTER_ROOT_RELATIVE).resolve()
    violations: list[CompositionComplexityViolation] = []
    for path in sorted(composition_root.glob("*.py")):
        if path.name == "__init__.py":
            continue
        rel = _posix_relative(path, root)
        lines = _line_count(path)
        if lines > max_lines:
            violations.append(CompositionComplexityViolation(rel, "lines", lines, max_lines))
        registrations = _registration_count(path)
        if registrations > max_registrations:
            violations.append(
                CompositionComplexityViolation(
                    rel,
                    "use_case_registrations",
                    registrations,
                    max_registrations,
                )
            )
    adapter_paths = [p for p in sorted(adapter_root.glob("*.py")) if p.name != "__init__.py"]
    adapter_paths.extend(
        path for path in (root / rel for rel in EXTRA_ADAPTER_FILES_RELATIVE) if path.is_file()
    )
    for path in adapter_paths:
        rel = _posix_relative(path, root)
        lines = _line_count(path)
        if lines > max_lines:
            violations.append(CompositionComplexityViolation(rel, "lines", lines, max_lines))
        registrations = _registration_count(path)
        if registrations:
            violations.append(
                CompositionComplexityViolation(
                    rel,
                    "bootstrap_use_case_registrations",
                    registrations,
                    0,
                )
            )
    return violations


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--composition-root", type=Path, default=None)
    parser.add_argument("--adapter-root", type=Path, default=None)
    parser.add_argument("--max-lines", type=int, default=DEFAULT_MAX_LINES)
    parser.add_argument("--max-registrations", type=int, default=DEFAULT_MAX_REGISTRATIONS)
    args = parser.parse_args(argv)

    violations = collect_composition_complexity_violations(
        root=args.root,
        composition_root=args.composition_root,
        adapter_root=args.adapter_root,
        max_lines=args.max_lines,
        max_registrations=args.max_registrations,
    )
    if violations:
        print("Composition complexity ratchet failed:")
        for violation in violations:
            print(f"  - {violation.path}: {violation.reason}={violation.value} > {violation.limit}")
        return 1
    print("Composition complexity ratchet passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
