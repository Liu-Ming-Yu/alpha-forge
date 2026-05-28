from __future__ import annotations

from typing import TYPE_CHECKING

from scripts.check_module_size import ARCHITECTURE_EXCEPTIONS, collect_module_size_violations

if TYPE_CHECKING:
    from pathlib import Path


def _write_lines(path: Path, count: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(f"line_{idx} = {idx}" for idx in range(count)), encoding="utf-8")


def test_module_size_check_blocks_new_oversized_modules(tmp_path: Path) -> None:
    src_root = tmp_path / "src" / "quant_platform"
    _write_lines(src_root / "small.py", 3)
    _write_lines(src_root / "too_big.py", 6)

    violations, stale = collect_module_size_violations(
        root=tmp_path,
        src_root=src_root,
        max_lines=5,
        exceptions={},
    )

    assert stale == []
    assert [(violation.path, violation.lines) for violation in violations] == [
        ("src/quant_platform/too_big.py", 6)
    ]


def test_module_size_check_accepts_documented_exception(tmp_path: Path) -> None:
    src_root = tmp_path / "src" / "quant_platform"
    oversized = src_root / "compat.py"
    _write_lines(oversized, 6)

    violations, stale = collect_module_size_violations(
        root=tmp_path,
        src_root=src_root,
        max_lines=5,
        exceptions={"src/quant_platform/compat.py": "public facade"},
    )

    assert violations == []
    assert stale == []


def test_module_size_check_reports_stale_exceptions(tmp_path: Path) -> None:
    src_root = tmp_path / "src" / "quant_platform"
    _write_lines(src_root / "small.py", 3)

    violations, stale = collect_module_size_violations(
        root=tmp_path,
        src_root=src_root,
        max_lines=5,
        exceptions={"src/quant_platform/missing.py": "removed facade"},
    )

    assert violations == []
    assert stale == ["src/quant_platform/missing.py"]


def test_module_size_check_reports_exception_under_threshold_as_stale(tmp_path: Path) -> None:
    src_root = tmp_path / "src" / "quant_platform"
    _write_lines(src_root / "compat.py", 3)

    violations, stale = collect_module_size_violations(
        root=tmp_path,
        src_root=src_root,
        max_lines=5,
        exceptions={"src/quant_platform/compat.py": "previously oversized facade"},
    )

    assert violations == []
    assert stale == ["src/quant_platform/compat.py"]


def test_default_module_size_exception_list_is_documented() -> None:
    assert all(reason.strip() for reason in ARCHITECTURE_EXCEPTIONS.values())
