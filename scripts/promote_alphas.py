"""Operator-facing CLI for promoting auto-discovered alphas.

The mining CLI (``scripts/mine_alphas.py``) writes one
:class:`AutoAlphaProvenance` per line to a JSONL run file. *Some* of
those candidates passed the in-run admission gate — a low bar for
"interesting". This script applies the higher
:class:`PromotionThresholds` bar and appends winners to the durable
:mod:`...formulaic.auto_library` JSONL, where they get picked up by
the formulaic family at next import.

The promotion workflow is operator-driven by design: the brief is
explicit that auto-discovered alphas must not flow straight into
production. ``--dry-run`` lets a reviewer inspect what *would* be
promoted before any file is written.

Usage
-----

::

    python scripts/promote_alphas.py \\
        --input data/parquet/research/alpha_mining/run_2026_05_25.jsonl \\
        --dry-run

    python scripts/promote_alphas.py \\
        --input data/parquet/research/alpha_mining/run_2026_05_25.jsonl \\
        --min-oos-rank-ic 0.05 --min-oos-icir 0.4

Output
------

When not ``--dry-run``, appends one
:class:`PromotedAlphaRecord` per winning candidate to the file at
``--output`` (defaults to :data:`.auto_library.DEFAULT_PROMOTED_PATH`).
A summary table is printed to stderr regardless.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import UTC, datetime
from pathlib import Path

from quant_platform.research.features.formulaic.auto_library import (
    DEFAULT_PROMOTED_PATH,
    PromotedAlphaRecord,
    append_promoted_alphas,
    build_record,
)
from quant_platform.research.features.formulaic.mining.evidence import (
    CandidateEvidence,
)
from quant_platform.research.features.formulaic.mining.provenance import (
    AutoAlphaProvenance,
)
from quant_platform.research.features.formulaic.mining.walk_forward import (
    WalkForwardEvidence,
)
from quant_platform.research.features.formulaic.promotion import (
    PromotionGate,
    PromotionThresholds,
    select_promotions,
    stable_alpha_id,
)
from quant_platform.research.features.formulaic.serialization import (
    expression_from_dict,
)

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Promote admitted mining candidates into the formulaic auto-library.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Mining run JSONL (output of scripts/mine_alphas.py).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_PROMOTED_PATH,
        help="Promoted-alphas JSONL to append to. Created if missing.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Skip writing; print would-be promotions to stderr only.",
    )
    parser.add_argument(
        "--min-oos-rank-ic",
        type=float,
        default=0.04,
        help="Minimum out-of-sample rank-IC for promotion.",
    )
    parser.add_argument(
        "--min-oos-icir",
        type=float,
        default=0.3,
        help="Minimum out-of-sample IC information ratio.",
    )
    parser.add_argument(
        "--max-fold-negative-ic-streak",
        type=int,
        default=1,
        help="Maximum allowed consecutive negative folds.",
    )
    parser.add_argument(
        "--min-n-folds-valid",
        type=int,
        default=4,
        help="Minimum number of folds with valid OOS IC.",
    )
    parser.add_argument(
        "--min-n-dates",
        type=int,
        default=100,
        help="Minimum distinct dates carrying (feature, label) pairs.",
    )
    parser.add_argument(
        "--max-turnover",
        type=float,
        default=0.4,
        help="Maximum allowed turnover (same as admission ceiling).",
    )
    parser.add_argument(
        "--allow-single-pass",
        action="store_true",
        help="Permit candidates whose evidence is single-pass (not WF). "
        "Off by default — the brief explicitly requires walk-forward "
        "validation before promotion.",
    )
    return parser


# ---------------------------------------------------------------------------
# JSONL parsing
# ---------------------------------------------------------------------------


def parse_mining_jsonl(
    input_path: Path,
) -> list[tuple[AutoAlphaProvenance, dict]]:
    """Parse a mining-run JSONL into provenance records.

    Returns a list of ``(provenance, raw_payload)`` pairs so the
    caller can re-read fields the dataclass doesn't carry (e.g.
    nested evidence fields that don't fit
    :class:`CandidateEvidence`'s shape).
    """
    if not input_path.exists():
        raise FileNotFoundError(f"mining JSONL not found: {input_path}")

    pairs: list[tuple[AutoAlphaProvenance, dict]] = []
    with input_path.open("r", encoding="utf-8") as fh:
        for line_index, raw in enumerate(fh, start=1):
            line = raw.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                print(  # noqa: T201
                    f"[promote-alphas] line {line_index}: skipping malformed JSON ({exc})",
                    file=sys.stderr,
                )
                continue
            try:
                provenance = _provenance_from_payload(payload)
            except (KeyError, ValueError, TypeError) as exc:
                print(  # noqa: T201
                    f"[promote-alphas] line {line_index}: skipping unparseable provenance ({exc})",
                    file=sys.stderr,
                )
                continue
            pairs.append((provenance, payload))
    return pairs


def _provenance_from_payload(payload: dict) -> AutoAlphaProvenance:
    """Inverse of ``scripts.mine_alphas.provenance_to_dict``."""
    expression = expression_from_dict(payload["expression"])
    evidence_payload = payload["evidence"]
    evidence: CandidateEvidence | WalkForwardEvidence
    if "fold_ics" in evidence_payload and evidence_payload["fold_ics"] is not None:
        evidence = WalkForwardEvidence(
            mean_ic=_as_float(evidence_payload.get("mean_ic")),
            rank_ic=_as_float(evidence_payload.get("rank_ic")),
            icir=_as_float(evidence_payload.get("icir")),
            fold_ics=tuple(_as_float(x) for x in evidence_payload["fold_ics"]),
            fold_rank_ics=tuple(_as_float(x) for x in evidence_payload.get("fold_rank_ics", [])),
            fold_negative_ic_streak=int(evidence_payload.get("fold_negative_ic_streak", 0)),
            n_folds_valid=int(evidence_payload.get("n_folds_valid", 0)),
            n_dates=int(evidence_payload.get("n_dates", 0)),
            turnover=_as_float(evidence_payload.get("turnover")),
            coverage=int(evidence_payload.get("coverage", 0)),
            correlation_to_baseline_max=_as_float(
                evidence_payload.get("correlation_to_baseline_max")
            ),
        )
    else:
        evidence = CandidateEvidence(
            mean_ic=_as_float(evidence_payload.get("mean_ic")),
            rank_ic=_as_float(evidence_payload.get("rank_ic")),
            icir=_as_float(evidence_payload.get("icir")),
            turnover=_as_float(evidence_payload.get("turnover")),
            coverage=int(evidence_payload.get("coverage", 0)),
            correlation_to_baseline_max=_as_float(
                evidence_payload.get("correlation_to_baseline_max")
            ),
            n_dates=int(evidence_payload.get("n_dates", 0)),
        )

    created_at = payload.get("created_at")
    if isinstance(created_at, str):
        created_dt = datetime.fromisoformat(created_at)
    else:
        created_dt = datetime.now(UTC)

    return AutoAlphaProvenance(
        name=str(payload["name"]),
        expression=expression,
        generation=int(payload.get("generation", 0)),
        seed=int(payload.get("seed", 0)),
        parent_name=payload.get("parent_name"),
        mutation_kind=payload.get("mutation_kind"),
        operator_set_version=str(payload.get("operator_set_version", "")),
        feature_set_version=str(payload.get("feature_set_version", "")),
        evidence=evidence,
        created_at=created_dt,
        admitted=bool(payload.get("admitted", False)),
        admission_reason=str(payload.get("admission_reason", "")),
    )


def _as_float(value: object | None) -> float:
    """Coerce JSON value (possibly ``None`` from a NaN scrub) to float."""
    if value is None:
        return float("nan")
    return float(value)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Promotion decision → record
# ---------------------------------------------------------------------------


def make_description(provenance: AutoAlphaProvenance, run_label: str) -> str:
    """Compose the description that ends up on the FeatureSpec.

    Includes the OOS metrics + the mining run label so any operator
    looking at the auto-library can trace an alpha back to its
    source.
    """
    evidence = provenance.evidence
    rank_ic = getattr(evidence, "rank_ic", float("nan"))
    icir = getattr(evidence, "icir", float("nan"))
    fold_streak = getattr(evidence, "fold_negative_ic_streak", None)
    n_folds = getattr(evidence, "n_folds_valid", None)
    parts = [
        f"Auto-promoted from mining run {run_label!r} "
        f"(seed={provenance.seed}, gen={provenance.generation}).",
        f"OOS rank-IC={rank_ic:.4f}, ICIR={icir:.4f}",
    ]
    if fold_streak is not None and n_folds is not None:
        parts[-1] += f", fold_streak={fold_streak}/{n_folds}"
    parts[-1] += "."
    return " ".join(parts)


def build_promoted_record(
    provenance: AutoAlphaProvenance,
    *,
    run_label: str,
) -> PromotedAlphaRecord:
    """Build the durable record for one to-be-promoted candidate."""
    name = stable_alpha_id(provenance.expression)
    description = make_description(provenance, run_label)
    evidence_payload = _evidence_to_dict(provenance.evidence)
    return build_record(
        expression=provenance.expression,
        name=name,
        description=description,
        promotion_evidence=evidence_payload,
        promoted_from_seed=provenance.seed,
        promoted_from_run=run_label,
    )


def _evidence_to_dict(evidence: CandidateEvidence | WalkForwardEvidence) -> dict:
    """Flatten an evidence dataclass into a JSON-friendly dict.

    NaN floats become ``None`` so the appended JSONL line stays
    parseable. Mirrors the same scrub from ``scripts/mine_alphas.py``.
    """
    import dataclasses

    out: dict[str, object] = {}
    for field in dataclasses.fields(evidence):
        raw = getattr(evidence, field.name)
        out[field.name] = _scrub_value(raw)
    return out


def _scrub_value(value: object) -> object:
    if isinstance(value, tuple):
        return [_scrub_value(v) for v in value]
    if isinstance(value, list):
        return [_scrub_value(v) for v in value]
    if isinstance(value, float) and math.isnan(value):
        return None
    return value


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    thresholds = PromotionThresholds(
        min_oos_rank_ic=args.min_oos_rank_ic,
        min_oos_icir=args.min_oos_icir,
        max_fold_negative_ic_streak=args.max_fold_negative_ic_streak,
        min_n_folds_valid=args.min_n_folds_valid,
        min_n_dates=args.min_n_dates,
        max_turnover=args.max_turnover,
        require_walk_forward=not args.allow_single_pass,
    )
    gate = PromotionGate(thresholds=thresholds)

    print(  # noqa: T201
        f"[promote-alphas] reading {args.input}",
        file=sys.stderr,
    )
    pairs = parse_mining_jsonl(args.input)
    print(  # noqa: T201
        f"[promote-alphas] parsed {len(pairs)} provenance rows",
        file=sys.stderr,
    )

    provenances = [prov for prov, _ in pairs]
    decisions = select_promotions(provenances=provenances, gate=gate)

    promoted_pairs = [(prov, decision) for prov, decision in decisions if decision.promoted]
    rejected_pairs = [(prov, decision) for prov, decision in decisions if not decision.promoted]

    print(  # noqa: T201
        f"[promote-alphas] would promote {len(promoted_pairs)} / "
        f"{len(provenances)} rows ({len(rejected_pairs)} rejected).",
        file=sys.stderr,
    )

    run_label = args.input.stem
    records = [build_promoted_record(prov, run_label=run_label) for prov, _ in promoted_pairs]

    if args.dry_run:
        for record in records:
            print(  # noqa: T201
                f"[promote-alphas] DRY-RUN promote → {record.name}: {record.description}",
                file=sys.stderr,
            )
        return 0

    if not records:
        print(  # noqa: T201
            "[promote-alphas] no candidates passed promotion thresholds; no file written.",
            file=sys.stderr,
        )
        return 0

    resolved, count = append_promoted_alphas(records, path=args.output)
    print(  # noqa: T201
        f"[promote-alphas] appended {count} records to {resolved}",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
