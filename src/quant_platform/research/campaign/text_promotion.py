"""Text candidate-promotion CLI handler."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any, cast

from quant_platform.application.errors import OperatorUsageError
from quant_platform.research.common import _json_default, research_json_result
from quant_platform.services.research_service.text.candidates.promotion import (
    promote_text_candidate_screens,
)

if TYPE_CHECKING:
    from pathlib import Path

    from quant_platform.application.results import UseCaseResult
    from quant_platform.config import PlatformSettings


async def _research_campaign_promote_text_candidates(
    settings: PlatformSettings,
    args: Any,
) -> UseCaseResult[dict[str, object]]:
    from quant_platform.services.research_service.sampling.factory import (
        walk_forward_object_root,
    )

    output_root: Path = args.output_root or walk_forward_object_root(
        settings.storage.object_store_root,
    )
    result = promote_text_candidate_screens(
        main_screen=_load_json_mapping(args.main_screen),
        confirmation_screen=_load_json_mapping(args.confirmation_screen),
        full_screen=_load_json_mapping(args.full_screen),
        feature_card_dir=args.feature_card_dir,
        feature_family_file=args.feature_family_file,
        min_passing_candidates=int(args.min_passing_candidates),
    )
    slug = str(
        getattr(args, "screen_name", "")
        or f"{result.get('promoted_feature_set_version', 'text')}_candidate_promotion"
    )
    diagnostics_root = output_root / "diagnostics" / slug
    diagnostics_root.mkdir(parents=True, exist_ok=True)
    result_path = diagnostics_root / "text_candidate_promotion.json"
    result_path.write_text(
        json.dumps(result, default=_json_default, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    report_path = diagnostics_root / "operator_report.md"
    report_path.write_text(_render_report(result), encoding="utf-8")
    payload = {
        "passed": bool(result["passed"]),
        "reason": result["reason"],
        "candidate_promotion": str(result_path),
        "operator_report": str(report_path),
        "promotion_artifacts_written": bool(result["promotion_artifacts_written"]),
        "shared_passing_candidates": result["shared_passing_candidates"],
        "blockers": result["blockers"],
        "feature_family_artifacts": result["feature_family_artifacts"],
    }
    passed = bool(result["passed"])
    if not passed:
        blocked_path = diagnostics_root / "blocked_candidate_promotion_summary.json"
        blocked_path.write_text(
            json.dumps(result, default=_json_default, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        payload["blocked_candidate_promotion_summary"] = str(blocked_path)
    return research_json_result(payload, passed=passed)


def _load_json_mapping(path: Path) -> dict[str, object]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise OperatorUsageError(f"failed to load JSON mapping {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise OperatorUsageError(f"JSON mapping must be an object: {path}")
    return {str(key): value for key, value in payload.items()}


def _render_report(result: dict[str, object]) -> str:
    shared = [str(name) for name in cast("list[object]", result["shared_passing_candidates"])]
    lines = [
        "# Text Candidate Promotion",
        "",
        f"- Passed: {bool(result['passed'])}",
        f"- Reason: {result['reason']}",
        f"- Promotion artifacts written: {bool(result['promotion_artifacts_written'])}",
        f"- Shared passing candidates: {', '.join(shared)}",
    ]
    blockers = [str(blocker) for blocker in cast("list[object]", result.get("blockers", []))]
    if blockers:
        lines.extend(["", "## Blockers"])
        lines.extend(f"- {blocker}" for blocker in blockers)
    return "\n".join(lines) + "\n"


__all__ = ["_research_campaign_promote_text_candidates"]
