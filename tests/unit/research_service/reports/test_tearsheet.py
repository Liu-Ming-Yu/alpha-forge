"""Tests for the Markdown tearsheet renderer."""

from __future__ import annotations

import json
import uuid
from typing import TYPE_CHECKING

from quant_platform.services.research_service.reports.tearsheet import render_tearsheet

if TYPE_CHECKING:
    from pathlib import Path


def test_tearsheet_handles_missing_sidecars(tmp_path: Path) -> None:
    run_id = uuid.uuid4()
    out = render_tearsheet(run_id, tmp_path)
    assert out.is_file()
    text = out.read_text(encoding="utf-8")
    assert str(run_id) in text
    assert "Information Coefficient" in text
    assert "Attribution" in text
    assert "Execution Quality" in text
    assert "missing execution_quality.json" in text


def test_tearsheet_renders_ic_and_attribution(tmp_path: Path) -> None:
    run_id = uuid.uuid4()
    run_dir = tmp_path / str(run_id)
    run_dir.mkdir(parents=True)

    (run_dir / "ic_report.json").write_text(
        json.dumps(
            {
                "horizons": [1, 5, 10, 20],
                "series": [
                    {
                        "factor": "momentum_1m",
                        "decay": {"1": 0.04, "5": 0.03, "10": 0.02, "20": 0.01},
                    }
                ],
            }
        )
    )
    (run_dir / "attribution.json").write_text(
        json.dumps(
            {
                "total_pnl": 0.123,
                "num_cycles": 4,
                "factor_pnl": {"momentum_1m": 0.1, "vol_compression": 0.02},
                "sector_pnl": {"Tech": 0.05, "Energy": 0.07},
                "regime_pnl": {"risk_on": 0.1, "risk_off": 0.023},
            }
        )
    )
    (run_dir / "metrics.json").write_text(
        json.dumps({"equity_curve": [100.0, 101.0, 102.5, 99.0, 103.0]})
    )

    out = render_tearsheet(run_id, tmp_path)
    text = out.read_text(encoding="utf-8")
    assert "momentum_1m" in text
    assert "Tech" in text
    assert "Energy" in text
    assert "risk_on" in text


def test_tearsheet_renders_risk_table_from_run_summary(tmp_path: Path) -> None:
    """Commit 8 / R-GOV-03: the Risk table comes from run_summary.json."""
    run_id = uuid.uuid4()
    run_dir = tmp_path / str(run_id)
    run_dir.mkdir(parents=True)

    (run_dir / "run_summary.json").write_text(
        json.dumps(
            {
                "initial_capital": "100000",
                "final_capital": "110000",
                "total_return": "0.10",
                "annualised_sharpe": "1.2345",
                "max_drawdown": "-0.05",
                "gross_turnover": "0.45",
                "equity_curve": [100000, 102000, 110000],
            }
        )
    )

    out = render_tearsheet(run_id, tmp_path)
    text = out.read_text(encoding="utf-8")
    assert "## Risk" in text
    assert "Annualised Sharpe" in text
    assert "+1.2345" in text
    assert "Max drawdown" in text
    assert "-0.0500" in text
    assert "Gross turnover" in text


def test_tearsheet_renders_execution_quality(tmp_path: Path) -> None:
    run_id = uuid.uuid4()
    run_dir = tmp_path / str(run_id)
    run_dir.mkdir(parents=True)

    (run_dir / "execution_quality.json").write_text(
        json.dumps(
            {
                "aggregate": {
                    "orders_count": 1,
                    "fills_count": 1,
                    "requested_quantity": 100,
                    "filled_quantity": 50,
                    "fill_rate": 0.5,
                    "average_participation_pct": 0.05,
                    "max_participation_pct": 0.05,
                    "total_commission": "1.00",
                    "total_slippage_cost": "2.50",
                    "average_implementation_shortfall_bps": 4.2,
                },
                "orders": [
                    {
                        "side": "buy",
                        "fill_ratio": 0.5,
                        "participation_pct": 0.05,
                        "spread_bps": 8.0,
                        "implementation_shortfall_bps": 4.2,
                        "is_complete": False,
                    }
                ],
            }
        )
    )

    out = render_tearsheet(run_id, tmp_path)
    text = out.read_text(encoding="utf-8")
    assert "## Execution Quality" in text
    assert "Fill rate" in text
    assert "50.00%" in text
    assert "Average implementation shortfall bps" in text
    assert "+4.20" in text


def test_tearsheet_risk_section_reports_na_for_null_sharpe(tmp_path: Path) -> None:
    """Sub-20-sample runs record sharpe=None; the table shows N/A."""
    run_id = uuid.uuid4()
    run_dir = tmp_path / str(run_id)
    run_dir.mkdir(parents=True)

    (run_dir / "run_summary.json").write_text(
        json.dumps(
            {
                "initial_capital": "100000",
                "final_capital": "100500",
                "total_return": "0.005",
                "annualised_sharpe": None,
                "max_drawdown": "0",
                "gross_turnover": "0",
                "equity_curve": [100000, 100500],
            }
        )
    )

    out = render_tearsheet(run_id, tmp_path)
    text = out.read_text(encoding="utf-8")
    assert "Annualised Sharpe | N/A" in text


def test_tearsheet_overwrites_existing(tmp_path: Path) -> None:
    run_id = uuid.uuid4()
    out = render_tearsheet(run_id, tmp_path)
    first = out.read_text(encoding="utf-8")
    (tmp_path / str(run_id) / "attribution.json").write_text(
        json.dumps(
            {
                "total_pnl": 0.01,
                "num_cycles": 1,
                "factor_pnl": {},
                "sector_pnl": {},
                "regime_pnl": {},
            }
        )
    )
    out2 = render_tearsheet(run_id, tmp_path)
    second = out2.read_text(encoding="utf-8")
    assert first != second
