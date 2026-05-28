from __future__ import annotations

from typing import TYPE_CHECKING

from scripts.check_service_coupling import (
    ServiceImportEdge,
    build_report,
    collect_cross_service_imports,
)

if TYPE_CHECKING:
    from pathlib import Path


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def test_collect_cross_service_imports_reports_direct_service_edges(tmp_path: Path) -> None:
    source_root = tmp_path / "src" / "quant_platform"
    _write(
        source_root / "services" / "research_service" / "example.py",
        "from quant_platform.services.portfolio_service.portfolio_constructor import X\n",
    )
    _write(
        source_root / "services" / "research_service" / "local.py",
        "from quant_platform.services.research_service.features import X\n",
    )
    _write(
        source_root / "services" / "research_service" / "core.py",
        "from quant_platform.core.contracts import PortfolioConstructor\n",
    )

    edges = collect_cross_service_imports(root=tmp_path, source_root=source_root)

    assert edges == (
        ServiceImportEdge(
            path="src/quant_platform/services/research_service/example.py",
            imported="quant_platform.services.portfolio_service.portfolio_constructor",
            source_service="research_service",
            target_service="portfolio_service",
        ),
    )


def test_service_coupling_report_blocks_unapproved_edges(tmp_path: Path) -> None:
    source_root = tmp_path / "src" / "quant_platform"
    _write(
        source_root / "services" / "signal_service" / "example.py",
        "import quant_platform.services.research_service.features.factors\n",
    )

    report = build_report(root=tmp_path, source_root=source_root, approved=frozenset())

    assert [edge.key for edge in report.unapproved] == [
        (
            "src/quant_platform/services/signal_service/example.py",
            "quant_platform.services.research_service.features.factors",
        )
    ]


def test_service_coupling_report_accepts_current_approval(tmp_path: Path) -> None:
    source_root = tmp_path / "src" / "quant_platform"
    path = "src/quant_platform/services/signal_service/example.py"
    imported = "quant_platform.services.research_service.features.factors"
    _write(
        tmp_path / path,
        "import quant_platform.services.research_service.features.factors\n",
    )

    report = build_report(
        root=tmp_path,
        source_root=source_root,
        approved=frozenset({(path, imported)}),
    )

    assert report.unapproved == ()
    assert report.stale_approvals == ()


def test_service_coupling_report_flags_stale_approvals(tmp_path: Path) -> None:
    source_root = tmp_path / "src" / "quant_platform"
    source_root.mkdir(parents=True)

    report = build_report(
        root=tmp_path,
        source_root=source_root,
        approved=frozenset(
            {
                (
                    "src/quant_platform/services/signal_service/missing.py",
                    "quant_platform.services.research_service.features.factors",
                )
            }
        ),
    )

    assert report.stale_approvals == (
        (
            "src/quant_platform/services/signal_service/missing.py",
            "quant_platform.services.research_service.features.factors",
        ),
    )
