from __future__ import annotations

import json
from typing import TYPE_CHECKING

from scripts.check_lint_debt import (
    LintDebtReport,
    _parse_ruff_violations,
    collect_noqa_counts,
    load_global_ignored_rules,
    load_per_file_ignore_count,
    validate_report,
)

if TYPE_CHECKING:
    from pathlib import Path


def test_load_global_ignored_rules(tmp_path: Path) -> None:
    pyproject = tmp_path / "pyproject.toml"
    pyproject.write_text(
        """
[tool.ruff.lint]
ignore = ["ANN001", "B904"]
""".strip(),
        encoding="utf-8",
    )

    assert load_global_ignored_rules(pyproject) == ("ANN001", "B904")


def test_load_per_file_ignore_count(tmp_path: Path) -> None:
    pyproject = tmp_path / "pyproject.toml"
    pyproject.write_text(
        """
[tool.ruff.lint.per-file-ignores]
"tests/**/*.py" = ["S101", "F841"]
"src/example.py" = ["N802"]
""".strip(),
        encoding="utf-8",
    )

    assert load_per_file_ignore_count(pyproject) == 3


def test_collect_noqa_counts_requires_explicit_codes(tmp_path: Path) -> None:
    source_root = tmp_path / "src" / "quant_platform"
    source_root.mkdir(parents=True)
    (source_root / "linted.py").write_text(
        "\n".join(
            [
                "import optional  # noqa: F401",
                "value = ignored_call()  # noqa",
            ]
        ),
        encoding="utf-8",
    )

    assert collect_noqa_counts((source_root,)) == (2, 1)


def test_parse_ruff_violations_groups_core_fields() -> None:
    payload = json.dumps(
        [
            {
                "code": "SIM102",
                "filename": "/repo/src/quant_platform/core/example.py",
                "location": {"row": 12, "column": 5},
            }
        ]
    )

    violations = _parse_ruff_violations(payload)

    assert len(violations) == 1
    assert violations[0].code == "SIM102"
    assert violations[0].row == 12


def test_validate_report_blocks_unexpected_global_ignores() -> None:
    report = LintDebtReport(
        global_ignored_rules=("PLR0913",),
        per_file_ignored_rules=0,
        source_noqa=0,
        bare_source_noqa=0,
        added_noqa=(),
        selected_rule_counts={},
        package_rule_counts={},
    )

    errors = validate_report(report, allow_new_noqa=False)

    assert "unexpected global Ruff ignored rules: PLR0913" in errors
    assert "global Ruff ignore count 1 > 0" in errors


def test_validate_report_blocks_new_source_noqa() -> None:
    report = LintDebtReport(
        global_ignored_rules=(),
        per_file_ignored_rules=0,
        source_noqa=1,
        bare_source_noqa=0,
        added_noqa=("src/quant_platform/example.py: value  # noqa: ANN401",),
        selected_rule_counts={},
        package_rule_counts={},
    )

    errors = validate_report(report, allow_new_noqa=False)

    assert errors[0] == "new source # noqa lines are blocked outside cleanup windows"


def test_validate_report_blocks_selected_rule_regressions() -> None:
    report = LintDebtReport(
        global_ignored_rules=(),
        per_file_ignored_rules=0,
        source_noqa=0,
        bare_source_noqa=0,
        added_noqa=(),
        selected_rule_counts={"ANN001": 999},
        package_rule_counts={},
    )

    assert validate_report(report, allow_new_noqa=False) == [
        "selected Ruff rule ANN001 count 999 > 0"
    ]
