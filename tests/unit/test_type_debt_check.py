from __future__ import annotations

from typing import TYPE_CHECKING

from scripts.check_type_debt import (
    TypeDebtReport,
    collect_type_ignore_counts,
    load_global_disabled_error_codes,
    validate_report,
)

if TYPE_CHECKING:
    from pathlib import Path


def test_load_global_disabled_error_codes(tmp_path: Path) -> None:
    pyproject = tmp_path / "pyproject.toml"
    pyproject.write_text(
        """
[tool.mypy]
disable_error_code = ["import-not-found", "import-untyped"]
""".strip(),
        encoding="utf-8",
    )

    assert load_global_disabled_error_codes(pyproject) == ("import-not-found", "import-untyped")


def test_collect_type_ignore_counts_requires_explicit_codes(tmp_path: Path) -> None:
    source_root = tmp_path / "src" / "quant_platform"
    source_root.mkdir(parents=True)
    (source_root / "typed.py").write_text(
        "\n".join(
            [
                "x = value  # type: ignore[arg-type]",
                "y = other  # type: ignore",
            ]
        ),
        encoding="utf-8",
    )

    assert collect_type_ignore_counts(source_root) == (2, 1)


def test_validate_report_blocks_phase_one_regression_codes() -> None:
    report = TypeDebtReport(
        global_disabled_error_codes=("import-not-found", "dict-item"),
        source_type_ignores=0,
        bare_source_type_ignores=0,
        added_type_ignores=(),
    )

    assert validate_report(report, allow_new_ignores=False) == [
        "unexpected global mypy disabled error codes: dict-item"
    ]


def test_validate_report_blocks_new_source_ignores() -> None:
    report = TypeDebtReport(
        global_disabled_error_codes=("import-untyped",),
        source_type_ignores=0,
        bare_source_type_ignores=0,
        added_type_ignores=("src/quant_platform/example.py: value  # type: ignore[arg-type]",),
    )

    errors = validate_report(report, allow_new_ignores=False)

    assert errors[0] == "new source # type: ignore lines are blocked outside cleanup windows"


def test_validate_report_accepts_current_phase_one_thresholds() -> None:
    report = TypeDebtReport(
        global_disabled_error_codes=("import-not-found", "import-untyped"),
        source_type_ignores=0,
        bare_source_type_ignores=0,
        added_type_ignores=(),
    )

    assert validate_report(report, allow_new_ignores=False) == []
