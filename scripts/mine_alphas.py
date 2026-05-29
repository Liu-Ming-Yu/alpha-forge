"""Operator-facing CLI for the formulaic alpha miner (Phase 4 of the brief).

The miner library lives at
:mod:`quant_platform.research.features.formulaic.mining`. This script
wraps it for ad-hoc operator runs:

1. Loads daily OHLCV bars for a contracts universe from the durable
   parquet store.
2. Builds a :class:`MarketPanel`, derives forward-return labels.
3. Configures :class:`AlphaGrammar` + :class:`AdmissionGate` + the
   requested search algorithm (random, evolutionary, or policy) + an
   optional :class:`MiningFoldConfig` for K-fold OOS evaluation.
4. Calls :func:`mine_alphas` with the assembled config.
5. Writes one JSONL line per provenance record — admitted or
   rejected — to ``--output``. Each line carries the full serialised
   AST (via :mod:`...formulaic.serialization`), the evaluated
   evidence, the seed, the operator-set version, and the admission
   decision plus its reason. The brief's "never store only the name"
   requirement is satisfied by construction.

Usage
-----

::

    python scripts/mine_alphas.py \\
        --contracts-file infra/config/universe_300.json \\
        --start 2023-01-01 --end 2025-01-01 \\
        --search evolutionary --population 80 --generations 6 \\
        --n-folds 5 \\
        --seed 42 \\
        --output data/parquet/research/alpha_mining/run_2026_05_25.jsonl

Output schema (one JSON object per line, ``jsonl`` extension)::

    {
        "name": "auto_alpha_000001",
        "expression": {"kind": "OpCall", "name": "rank", ...},
        "generation": 0,
        "seed": 42,
        "parent_name": null,
        "mutation_kind": null,
        "operator_set_version": "operator-set-v1",
        "feature_set_version": "formulaic-alpha-v1",
        "evidence": {
            "rank_ic": 0.043,
            "icir": 0.21,
            "fold_ics": [-0.01, 0.05, 0.03, 0.04],  # WF mode only
            "fold_negative_ic_streak": 1,           # WF mode only
            "n_folds_valid": 4,                     # WF mode only
            ...
        },
        "created_at": "2026-05-25T19:00:00+00:00",
        "admitted": false,
        "admission_reason": "icir=0.21 < min_icir=0.3"
    }

The output file is JSONL (one object per line) rather than a single
JSON array so streaming consumers and ``jq``-style ad-hoc inspection
work without loading the whole file.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pandas as pd

from quant_platform.research.features.formulaic.mining import (
    AdmissionGate,
    AdmissionThresholds,
    AlphaGrammar,
    EvolutionarySearch,
    MiningFoldConfig,
    PolicySearch,
    RandomSearch,
    SearchAlgorithm,
    make_forward_return_labels,
    mine_alphas,
)
from quant_platform.research.features.formulaic.panel import build_market_panel
from quant_platform.research.features.formulaic.serialization import (
    expression_to_dict,
)

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping

    from quant_platform.research.features.formulaic.mining.provenance import (
        AutoAlphaProvenance,
    )

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BAR_ROOT = PROJECT_ROOT / "data" / "parquet" / "bars"


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the formulaic alpha miner over a daily-bar panel.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--contracts-file",
        type=Path,
        required=True,
        help="Path to a contracts JSON file keyed by instrument_id (e.g. "
        "infra/config/universe_300.json).",
    )
    parser.add_argument(
        "--start",
        type=_parse_iso_date,
        required=True,
        help="Start of the panel window (ISO date or datetime; UTC).",
    )
    parser.add_argument(
        "--end",
        type=_parse_iso_date,
        required=True,
        help="End of the panel window (ISO date or datetime; UTC).",
    )
    parser.add_argument(
        "--bar-root",
        type=Path,
        default=DEFAULT_BAR_ROOT,
        help="Root directory of the daily-bar parquet store.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="JSONL output path. Parent directory is created if missing.",
    )
    parser.add_argument(
        "--horizon-days",
        type=int,
        default=5,
        help="Forward-return horizon in trading days.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="RNG seed. Mining is reproducible: same inputs + same seed = same history.",
    )
    parser.add_argument(
        "--search",
        choices=("random", "evolutionary", "policy"),
        default="random",
        help="Search algorithm. random = uniform AST sampling; "
        "evolutionary = small GP loop with tournament selection + mutation; "
        "policy = qlib-style policy-guided mutation of one trajectory.",
    )
    parser.add_argument(
        "--n-candidates",
        type=int,
        default=200,
        help="(random / policy search) Total candidates to evaluate.",
    )
    parser.add_argument(
        "--population",
        type=int,
        default=60,
        help="(evolutionary search) Population size per generation.",
    )
    parser.add_argument(
        "--generations",
        type=int,
        default=6,
        help="(evolutionary search) Number of generations.",
    )
    parser.add_argument(
        "--max-depth",
        type=int,
        default=4,
        help="AlphaGrammar max AST depth.",
    )
    parser.add_argument(
        "--max-total-lookback",
        type=int,
        default=252,
        help="AlphaGrammar max accumulated lookback in trading days.",
    )
    parser.add_argument(
        "--n-folds",
        type=int,
        default=0,
        help="If > 0, enable walk-forward (K-fold OOS) evaluation with K = "
        "this value. 0 falls back to single-pass IC scoring.",
    )
    parser.add_argument(
        "--embargo-days",
        type=int,
        default=None,
        help="Walk-forward embargo. Defaults to --horizon-days when WF is enabled.",
    )
    parser.add_argument(
        "--min-test-days",
        type=int,
        default=20,
        help="Walk-forward minimum dates per fold for the fold to count.",
    )
    parser.add_argument(
        "--min-rank-ic",
        type=float,
        default=0.02,
        help="Admission gate minimum rank-IC.",
    )
    parser.add_argument(
        "--min-icir",
        type=float,
        default=0.1,
        help="Admission gate minimum ICIR.",
    )
    parser.add_argument(
        "--max-turnover",
        type=float,
        default=0.4,
        help="Admission gate maximum turnover.",
    )
    parser.add_argument(
        "--max-correlation-to-admitted",
        type=float,
        default=0.7,
        help="Admission gate maximum |correlation| to any previously-"
        "admitted candidate (correlation pruning).",
    )
    parser.add_argument(
        "--operator-set-version",
        type=str,
        default="operator-set-v1",
        help="Operator catalog version pinned into every provenance row.",
    )
    parser.add_argument(
        "--feature-set-version",
        type=str,
        default="formulaic-alpha-v1",
        help="Feature-set version pinned into every provenance row.",
    )
    return parser


def _parse_iso_date(value: str) -> datetime:
    """Parse ``YYYY-MM-DD`` or full ISO datetime; assume UTC if naive."""
    if "T" in value or " " in value:
        dt = datetime.fromisoformat(value)
    else:
        dt = datetime.fromisoformat(f"{value}T00:00:00")
    return dt.replace(tzinfo=UTC) if dt.tzinfo is None else dt


# ---------------------------------------------------------------------------
# Bar loading
# ---------------------------------------------------------------------------


def load_panel_frame(
    *,
    contracts_file: Path,
    bar_root: Path,
    start: datetime,
    end: datetime,
) -> pd.DataFrame:
    """Load a multi-instrument daily OHLCV panel from the parquet store.

    Reads the contracts file (a JSON dict keyed by ``instrument_id``),
    iterates each instrument, loads its daily-bar parquet, and
    concatenates the result into a long-format DataFrame ready for
    :func:`build_market_panel`. Instruments whose bar folder is
    missing or empty are skipped silently — the miner is allowed to
    run on a partial universe.
    """
    if not contracts_file.exists():
        raise FileNotFoundError(f"contracts file not found: {contracts_file}")
    contracts = json.loads(contracts_file.read_text(encoding="utf-8"))
    if not isinstance(contracts, dict):
        raise ValueError(
            f"contracts file must be a dict keyed by instrument_id; got {type(contracts).__name__}"
        )

    frames: list[pd.DataFrame] = []
    for instrument_id in contracts:
        instrument_frame = _load_instrument_bars(
            instrument_id=str(instrument_id),
            bar_root=bar_root,
            start=start,
            end=end,
        )
        if instrument_frame is None or instrument_frame.empty:
            continue
        frames.append(instrument_frame)

    if not frames:
        raise RuntimeError(
            f"No bars loaded for any instrument in {contracts_file}. "
            "Check --bar-root and --start/--end."
        )
    return pd.concat(frames, ignore_index=True)


def _load_instrument_bars(
    *,
    instrument_id: str,
    bar_root: Path,
    start: datetime,
    end: datetime,
) -> pd.DataFrame | None:
    folder = bar_root / instrument_id
    daily_folder = folder / "daily"
    if daily_folder.exists():
        files = sorted(daily_folder.glob("*.parquet"))
    elif folder.exists():
        files = sorted(folder.glob("*.parquet"))
    else:
        return None
    if not files:
        return None

    pieces: list[pd.DataFrame] = []
    for path in files:
        piece = pd.read_parquet(
            path,
            columns=[
                "timestamp",
                "open",
                "high",
                "low",
                "close",
                "volume",
                "bar_seconds",
            ],
        )
        piece = piece[piece["bar_seconds"] == 86400]
        pieces.append(piece)
    if not pieces:
        return None

    df = pd.concat(pieces, ignore_index=True)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True).astype("datetime64[ns, UTC]")
    df = df.dropna(subset=["close"]).sort_values("timestamp").drop_duplicates("timestamp")
    df["date"] = df["timestamp"].dt.tz_convert("UTC").dt.normalize().dt.tz_localize(None)
    mask = (df["date"] >= pd.Timestamp(start.replace(tzinfo=None))) & (
        df["date"] <= pd.Timestamp(end.replace(tzinfo=None))
    )
    df = df.loc[mask].copy()
    if df.empty:
        return None
    df["instrument_id"] = instrument_id
    return df[["instrument_id", "date", "open", "high", "low", "close", "volume"]]


# ---------------------------------------------------------------------------
# Miner assembly
# ---------------------------------------------------------------------------


def build_search(args: argparse.Namespace) -> SearchAlgorithm:
    if args.search == "random":
        return RandomSearch(n_candidates=args.n_candidates)
    if args.search == "evolutionary":
        return EvolutionarySearch(
            population_size=args.population,
            n_generations=args.generations,
        )
    if args.search == "policy":
        return PolicySearch(n_candidates=args.n_candidates)
    raise ValueError(f"unknown --search: {args.search!r}")


def build_gate(args: argparse.Namespace) -> AdmissionGate:
    return AdmissionGate(
        thresholds=AdmissionThresholds(
            min_rank_ic=args.min_rank_ic,
            min_icir=args.min_icir,
            max_turnover=args.max_turnover,
            max_correlation_to_admitted=args.max_correlation_to_admitted,
        ),
    )


def build_fold_config(args: argparse.Namespace) -> MiningFoldConfig | None:
    if args.n_folds <= 0:
        return None
    embargo = args.embargo_days if args.embargo_days is not None else args.horizon_days
    return MiningFoldConfig(
        n_folds=args.n_folds,
        embargo_days=embargo,
        min_test_days=args.min_test_days,
    )


# ---------------------------------------------------------------------------
# JSONL output
# ---------------------------------------------------------------------------


def provenance_to_dict(provenance: AutoAlphaProvenance) -> dict[str, Any]:
    """Render an :class:`AutoAlphaProvenance` to a JSON-friendly dict."""
    evidence = provenance.evidence
    evidence_dict = {
        field.name: _encode_value(getattr(evidence, field.name))
        for field in dataclasses.fields(evidence)
    }
    return {
        "name": provenance.name,
        "expression": expression_to_dict(provenance.expression),
        "generation": provenance.generation,
        "seed": provenance.seed,
        "parent_name": provenance.parent_name,
        "mutation_kind": provenance.mutation_kind,
        "operator_set_version": provenance.operator_set_version,
        "feature_set_version": provenance.feature_set_version,
        "evidence": evidence_dict,
        "created_at": provenance.created_at.isoformat(),
        "admitted": bool(provenance.admitted),
        "admission_reason": provenance.admission_reason,
    }


def _encode_value(value: object) -> object:
    """Coerce a dataclass field value into a JSON-friendly Python object.

    Tuples become lists (JSON has no tuple), NaN floats become None so
    the JSON line stays valid (``json.dumps`` writes ``NaN`` by
    default, which most parsers reject).
    """
    if isinstance(value, tuple):
        return [_encode_value(v) for v in value]
    if isinstance(value, list):
        return [_encode_value(v) for v in value]
    if isinstance(value, float) and value != value:  # noqa: PLR0124 — NaN check
        return None
    return value


def write_jsonl(
    *,
    output_path: Path,
    provenances: Iterable[AutoAlphaProvenance],
) -> int:
    """Write one JSON object per line to ``output_path``. Returns the count."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with output_path.open("w", encoding="utf-8") as fh:
        for prov in provenances:
            fh.write(json.dumps(provenance_to_dict(prov), ensure_ascii=False))
            fh.write("\n")
            count += 1
    return count


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    print(  # noqa: T201 — operator-facing CLI
        f"[mine-alphas] loading bars from {args.bar_root} ({args.start.date()} → "
        f"{args.end.date()})",
        file=sys.stderr,
    )
    bars = load_panel_frame(
        contracts_file=args.contracts_file,
        bar_root=args.bar_root,
        start=args.start,
        end=args.end,
    )
    print(  # noqa: T201
        f"[mine-alphas] loaded {len(bars)} bar rows across "
        f"{bars['instrument_id'].nunique()} instruments",
        file=sys.stderr,
    )

    panel = build_market_panel(bars)
    labels = make_forward_return_labels(panel, horizon=args.horizon_days)

    grammar = AlphaGrammar(
        max_depth=args.max_depth,
        max_total_lookback=args.max_total_lookback,
    )
    search = build_search(args)
    gate = build_gate(args)
    fold_config = build_fold_config(args)
    if fold_config is not None:
        print(  # noqa: T201
            f"[mine-alphas] walk-forward mode: n_folds={fold_config.n_folds}, "
            f"embargo_days={fold_config.embargo_days}",
            file=sys.stderr,
        )

    result = mine_alphas(
        grammar=grammar,
        panel=panel,
        labels=labels,
        search=search,
        gate=gate,
        seed=args.seed,
        operator_set_version=args.operator_set_version,
        feature_set_version=args.feature_set_version,
        fold_config=fold_config,
    )

    written = write_jsonl(output_path=args.output, provenances=result.history)
    print(  # noqa: T201
        f"[mine-alphas] evaluated {result.n_evaluated}, admitted "
        f"{len(result.admitted)}, wrote {written} rows to {args.output}",
        file=sys.stderr,
    )
    return 0


def _build_initial_baseline(_args: argparse.Namespace) -> Mapping[str, pd.Series] | None:
    """Reserved for future use — loading baseline features from a previous
    JSONL run so correlation pruning carries across sessions. Currently
    unused; mining starts with an empty baseline every run."""
    return None


if __name__ == "__main__":
    raise SystemExit(main())
