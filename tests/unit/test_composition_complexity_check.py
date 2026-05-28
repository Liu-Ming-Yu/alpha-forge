from __future__ import annotations

from typing import TYPE_CHECKING

from scripts.check_composition_complexity import (
    CompositionComplexityViolation,
    collect_composition_complexity_violations,
)

if TYPE_CHECKING:
    from pathlib import Path


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def test_composition_complexity_check_blocks_large_modules(tmp_path: Path) -> None:
    root = tmp_path
    composition_root = root / "src" / "quant_platform" / "application" / "operator_use_cases"
    _write(composition_root / "large.py", "\n".join(f"x{idx}=1" for idx in range(6)))

    violations = collect_composition_complexity_violations(
        root=root,
        composition_root=composition_root,
        max_lines=5,
    )

    assert violations == [
        CompositionComplexityViolation(
            path="src/quant_platform/application/operator_use_cases/large.py",
            reason="lines",
            value=6,
            limit=5,
        )
    ]


def test_composition_complexity_check_blocks_over_registered_modules(tmp_path: Path) -> None:
    root = tmp_path
    composition_root = root / "src" / "quant_platform" / "application" / "operator_use_cases"
    _write(
        composition_root / "busy.py",
        "def register(registry):\n"
        "    registry.register('a', object())\n"
        "    registry.register('b', object())\n",
    )

    violations = collect_composition_complexity_violations(
        root=root,
        composition_root=composition_root,
        max_registrations=1,
    )

    assert violations == [
        CompositionComplexityViolation(
            path="src/quant_platform/application/operator_use_cases/busy.py",
            reason="use_case_registrations",
            value=2,
            limit=1,
        )
    ]


def test_composition_complexity_check_blocks_bootstrap_registrations(tmp_path: Path) -> None:
    root = tmp_path
    composition_root = root / "src" / "quant_platform" / "application" / "operator_use_cases"
    adapter_root = root / "src" / "quant_platform" / "bootstrap" / "operator_adapters"
    _write(composition_root / "runtime.py", "def register(registry):\n    pass\n")
    _write(
        adapter_root / "runtime.py",
        "def register(registry):\n    registry.register('a', object())\n",
    )

    violations = collect_composition_complexity_violations(
        root=root,
        composition_root=composition_root,
        adapter_root=adapter_root,
    )

    assert violations == [
        CompositionComplexityViolation(
            path="src/quant_platform/bootstrap/operator_adapters/runtime.py",
            reason="bootstrap_use_case_registrations",
            value=1,
            limit=0,
        )
    ]
