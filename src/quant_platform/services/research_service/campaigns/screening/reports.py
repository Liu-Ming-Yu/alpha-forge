"""Markdown reports for governed candidate screens."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import cast

from quant_platform.services.research_service.campaigns.screening.common import (
    RANKED_DIAGNOSTIC_LIMIT,
    float_value,
)


def render_candidate_screen_report(
    screen: Mapping[str, object],
    *,
    title: str,
    feature_set_key: str,
    feature_set_label: str,
    next_action: str,
) -> str:
    raw_rows = screen.get("candidates", ())
    rows = (
        [cast("Mapping[str, object]", row) for row in raw_rows if isinstance(row, Mapping)]
        if isinstance(raw_rows, Sequence) and not isinstance(raw_rows, str)
        else []
    )
    lines = [
        f"# {title}: {screen.get('candidate_family', '')}",
        "",
        "## Summary",
        f"- Passed: `{screen.get('passed')}`",
        f"- {feature_set_label}: `{screen.get(feature_set_key)}`",
        f"- Candidate set: `{screen.get('candidate_set', 'seed')}`",
        f"- Screened candidates: `{screen.get('screened_candidate_count', len(rows))}`",
        f"- Promotion artifacts written: `{screen.get('promotion_artifacts_written')}`",
        f"- Passing candidates: `{screen.get('passing_candidates')}`",
        "",
    ]
    if len(rows) <= RANKED_DIAGNOSTIC_LIMIT:
        lines.append("## Candidate Health")
        lines.extend(candidate_health_line(row) for row in rows)
    else:
        lines.append("## Ranked Diagnostics")
        for heading, key in (
            ("Top By Stability", "top_by_stability"),
            ("Top By Null Margin", "top_by_null_margin"),
            ("Near Misses", "near_misses"),
        ):
            lines.extend(["", f"### {heading}"])
            ranked_rows = screen.get(key, ())
            if (
                isinstance(ranked_rows, Sequence)
                and not isinstance(ranked_rows, str)
                and ranked_rows
            ):
                lines.extend(
                    ranked_health_line(cast("Mapping[str, object]", row))
                    for row in ranked_rows
                    if isinstance(row, Mapping)
                )
            else:
                lines.append("- None")
    blockers = screen.get("blockers", ())
    lines.extend(["", "## Blockers"])
    if isinstance(blockers, Sequence) and not isinstance(blockers, str) and blockers:
        lines.extend(f"- {blocker}" for blocker in blockers)
    else:
        lines.append("- None")
    lines.extend(["", "## Next Action", next_action, ""])
    return "\n".join(lines)


def candidate_health_line(row: Mapping[str, object]) -> str:
    metrics = cast("Mapping[str, object]", row.get("metrics", {}))
    return (
        "- `{name}` ic={ic:.4f} icir={icir:.4f} null_margin={margin:.4f} "
        "nonzero={density:.3f} neg_streak={streak} passed=`{passed}`".format(
            name=row.get("feature_name", ""),
            ic=float_value(metrics.get("ic_mean")),
            icir=float_value(metrics.get("icir")),
            margin=float_value(metrics.get("null_margin")),
            density=float_value(metrics.get("source_density")),
            streak=int(float_value(metrics.get("negative_ic_streak"))),
            passed=row.get("passed"),
        )
    )


def ranked_health_line(row: Mapping[str, object]) -> str:
    blockers = row.get("blockers", ())
    blocker_count = (
        len(blockers) if isinstance(blockers, Sequence) and not isinstance(blockers, str) else 0
    )
    return (
        "- `{name}` ic={ic:.4f} icir={icir:.4f} null_margin={margin:.4f} "
        "nonzero={density:.3f} neg_streak={streak} blockers={blockers}".format(
            name=row.get("feature_name", ""),
            ic=float_value(row.get("ic_mean")),
            icir=float_value(row.get("icir")),
            margin=float_value(row.get("null_margin")),
            density=float_value(row.get("source_density")),
            streak=int(float_value(row.get("negative_ic_streak"))),
            blockers=blocker_count,
        )
    )
