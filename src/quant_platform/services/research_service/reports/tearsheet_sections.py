"""Pure Markdown section renderers for backtest tearsheets."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import uuid
    from collections.abc import Iterable, Sequence
    from pathlib import Path

JsonObject = dict[str, Any]


def render_ic_section(ic_report: JsonObject | None) -> str:
    if ic_report is None:
        return "_No IC report available (run `compute_ic_report` + `write_ic_report`)._"
    lines: list[str] = []
    horizons = ic_report.get("horizons", [])
    lines.append("| Factor | " + " | ".join(f"IC@{h}d" for h in horizons) + " |")
    lines.append("|---|" + "|".join(["---"] * len(horizons)) + "|")
    for series in ic_report.get("series", []):
        factor = series.get("factor", "-")
        decay = series.get("decay", {})
        row = [factor]
        for horizon in horizons:
            val = decay.get(str(horizon))
            row.append("N/A" if val is None else f"{float(val):+.4f}")
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def render_attribution_section(attribution: JsonObject | None) -> str:
    if attribution is None:
        return (
            "_No attribution artifact available (run `compute_attribution` + `write_attribution`)._"
        )
    parts: list[str] = []
    parts.append(
        f"Total P&L: **{attribution.get('total_pnl', 0):+.4f}** over "
        f"{attribution.get('num_cycles', 0)} cycles.\n"
    )
    parts.append("### Per-factor\n")
    parts.append(_format_table(attribution.get("factor_pnl", {}).items(), ("Factor", "P&L")))
    parts.append("\n### Per-sector\n")
    parts.append(_format_table(attribution.get("sector_pnl", {}).items(), ("Sector", "P&L")))
    parts.append("\n### Per-regime\n")
    parts.append(_format_table(attribution.get("regime_pnl", {}).items(), ("Regime", "P&L")))
    return "\n".join(parts)


def render_equity_section(metrics: JsonObject | None, root: Path, run_id: uuid.UUID) -> str:
    if not metrics:
        return "_No equity curve available (missing metrics.json)._"
    curve = metrics.get("equity_curve")
    if not curve:
        return "_Metrics artifact present but has no equity_curve entries._"
    if not isinstance(curve, list):
        return "_Metrics artifact present but equity_curve is not a list._"
    values = [_payload_float(value) for value in curve]
    png_path = _maybe_render_png(values, root / str(run_id) / "equity.png")
    if png_path is not None:
        return "![equity curve](./equity.png)"
    return f"```\n{_ascii_sparkline(values)}\n```"


def render_risk_section(summary: JsonObject | None) -> str:
    if summary is None:
        return "_No run summary available (missing run_summary.json)._"

    sharpe = summary.get("annualised_sharpe")
    sharpe_cell = "N/A" if sharpe in (None, "None") else f"{_payload_float(sharpe):+.4f}"

    lines = [
        "| Metric | Value |",
        "|---|---|",
        f"| Annualised Sharpe | {sharpe_cell} |",
        f"| Total return | {_fmt_summary(summary, 'total_return')} |",
        f"| Max drawdown | {_fmt_summary(summary, 'max_drawdown')} |",
        f"| Gross turnover | {_fmt_summary(summary, 'gross_turnover')} |",
    ]
    return "\n".join(lines)


def render_execution_quality_section(execution_quality: JsonObject | None) -> str:
    if execution_quality is None:
        return "_No execution quality artifact available (missing execution_quality.json)._"

    aggregate = execution_quality.get("aggregate", {})
    fill_rate = _optional_float(aggregate, "fill_rate")
    avg_participation = _optional_float(aggregate, "average_participation_pct")
    max_participation = _optional_float(aggregate, "max_participation_pct")
    avg_shortfall = _optional_float(aggregate, "average_implementation_shortfall_bps")

    lines = [
        "| Metric | Value |",
        "|---|---|",
        f"| Fill rate | {'N/A' if fill_rate is None else f'{fill_rate:.2%}'} |",
        "| Average participation | "
        f"{'N/A' if avg_participation is None else f'{avg_participation:.2%}'} |",
        "| Max participation | "
        f"{'N/A' if max_participation is None else f'{max_participation:.2%}'} |",
        f"| Total commission | {aggregate.get('total_commission', 'N/A')} |",
        f"| Total slippage cost | {aggregate.get('total_slippage_cost', 'N/A')} |",
        "| Average implementation shortfall bps | "
        f"{'N/A' if avg_shortfall is None else f'{avg_shortfall:+.2f}'} |",
    ]

    orders = execution_quality.get("orders", [])
    if orders:
        lines.extend(
            [
                "\n### Largest implementation shortfall",
                "| Side | Fill ratio | Participation | Spread bps | Shortfall bps | Complete |",
                "|---|---|---|---|---|---|",
            ]
        )
        sorted_orders = sorted(
            orders,
            key=lambda row: abs(float(row.get("implementation_shortfall_bps", 0.0))),
            reverse=True,
        )
        for row in sorted_orders[:5]:
            lines.append(_execution_order_row(row))

    return "\n".join(lines)


def render_industrial_evidence_section(
    evidence: JsonObject | None,
    reconciliation: JsonObject | None,
) -> str:
    if evidence is None and reconciliation is None:
        return "_No industrial intraday evidence manifest available._"
    lines = ["| Evidence | Value |", "|---|---|"]
    if evidence is not None:
        lines.append(f"| Evidence passed | {bool(evidence.get('passed'))} |")
        lines.append(f"| Code commit | {evidence.get('code_commit', 'N/A')} |")
        lines.append(f"| Config hash | {evidence.get('config_hash', 'N/A')} |")
        datasets = evidence.get("dataset_ids", [])
        lines.append(f"| Dataset IDs | {', '.join(str(value) for value in datasets) or 'N/A'} |")
        blockers = evidence.get("blockers", [])
        lines.append(f"| Blockers | {', '.join(str(value) for value in blockers) or 'none'} |")
    if reconciliation is not None:
        lines.append(f"| Reconciliation passed | {bool(reconciliation.get('passed'))} |")
        lines.append(f"| Reconciliation status | {reconciliation.get('status', 'N/A')} |")
        lines.append(
            f"| Target max diff bps | {reconciliation.get('target_weight_max_diff_bps', 'N/A')} |"
        )
        lines.append(f"| NAV diff bps | {reconciliation.get('nav_diff_bps', 'N/A')} |")
        lines.append(
            f"| Max drawdown diff bps | {reconciliation.get('max_drawdown_diff_bps', 'N/A')} |"
        )
        breaches = reconciliation.get("breaches", [])
        lines.append(
            f"| Reconciliation breaches | {', '.join(str(value) for value in breaches) or 'none'} |"
        )
    return "\n".join(lines)


def _format_table(rows: Iterable[tuple[str, float]], header: tuple[str, str]) -> str:
    lines = [f"| {header[0]} | {header[1]} |", "|---|---|"]
    for label, value in sorted(rows, key=lambda item: -abs(item[1])):
        lines.append(f"| {label} | {value:+.4f} |")
    return "\n".join(lines)


def _fmt_summary(summary: JsonObject, key: str, precision: int = 4) -> str:
    raw = summary.get(key)
    if raw is None:
        return "N/A"
    try:
        return f"{_payload_float(raw):+.{precision}f}"
    except (TypeError, ValueError):
        return str(raw)


def _optional_float(payload: JsonObject, key: str) -> float | None:
    raw = payload.get(key)
    if raw is None:
        return None
    try:
        return _payload_float(raw)
    except (TypeError, ValueError):
        return None


def _payload_float(value: object) -> float:
    if isinstance(value, int | float | str):
        return float(value)
    raise TypeError(f"value must be numeric, got {type(value).__name__}")


def _execution_order_row(row: JsonObject) -> str:
    fill_ratio = _payload_float(row.get("fill_ratio", 0.0))
    participation = _payload_float(row.get("participation_pct", 0.0))
    spread = _payload_float(row.get("spread_bps", 0.0))
    shortfall = _payload_float(row.get("implementation_shortfall_bps", 0.0))
    complete = "yes" if row.get("is_complete") else "no"
    return (
        "| "
        f"{row.get('side', 'N/A')} | "
        f"{fill_ratio:.2%} | "
        f"{participation:.2%} | "
        f"{spread:.2f} | "
        f"{shortfall:+.2f} | "
        f"{complete} |"
    )


def _ascii_sparkline(values: Sequence[float]) -> str:
    if not values:
        return ""
    lo = min(values)
    hi = max(values)
    if hi == lo:
        return "-" * min(len(values), 60)
    chars = ".:-=+*#%@"
    scale = (len(chars) - 1) / (hi - lo)
    return "".join(chars[int((value - lo) * scale)] for value in values[:60])


def _maybe_render_png(values: Sequence[float], path: Path) -> Path | None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return None
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.plot(values)
    ax.set_title("Equity curve")
    ax.set_xlabel("Rebalance")
    ax.set_ylabel("NAV")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
    return path


__all__ = [
    "render_attribution_section",
    "render_equity_section",
    "render_execution_quality_section",
    "render_ic_section",
    "render_industrial_evidence_section",
    "render_risk_section",
]
