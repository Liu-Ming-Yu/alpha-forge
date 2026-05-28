"""Walk-forward the starter quality+value Sharadar features and write eligibility.

Research-only driver — converts the point-in-time Sharadar feature frame into
``SupervisedAlphaSample`` rows over the available bar window, then calls the
audited campaign evaluator (``run_sample_walk_forward``). The output matches
the same eligibility-check format the production campaign emits, so the
yes/no verdict is directly comparable to prior runs under
``data/parquet/research/walk_forward/``.

Sample construction
-------------------

* **as_of**: for each trading day ``t`` and each instrument with a usable
  feature row whose ``datekey`` is at least one trading day in the past, the
  most-recent feature row is forward-filled to ``t``. This is the standard
  fundamentals discipline: features become tradable on the session **after**
  the filing date.
* **forward_return**: 21-trading-day log return from ``close[t]`` to
  ``close[t + 21]``. Matches the campaign's default horizon
  (``test_window_days=21``).
* Cross-sectional rank-normalization is performed *inside* the campaign
  evaluator on a per-day basis — we hand it raw feature values per the
  ``SupervisedAlphaSample`` contract.

Usage::

    python scripts/walkforward_sharadar_starter.py
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import time
import uuid as _uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from quant_platform.research.fundamentals import (
    FEATURE_NAMES,
    compute_starter_features,
    load_sector_map,
    load_sharadar_sf1_panel,
)
from quant_platform.services.research_service.campaigns.evaluation.walk_forward import (
    run_sample_walk_forward,
)
from quant_platform.services.research_service.campaigns.portfolio.types import (
    CampaignPortfolioConfig,
)
from quant_platform.services.research_service.modeling.walk_forward.walk_forward import (
    WalkForwardConfig,
)
from quant_platform.services.research_service.sampling.factory_models import (
    AlphaEligibilityThresholds,
)
from quant_platform.services.research_service.sampling.samples import SupervisedAlphaSample

if TYPE_CHECKING:
    from collections.abc import Mapping

PROJECT_ROOT = Path(__file__).resolve().parent.parent
BAR_ROOT = PROJECT_ROOT / "data" / "parquet" / "bars"
OUTPUT_ROOT = PROJECT_ROOT / "data" / "parquet" / "research" / "walk_forward_research"


def _load_bars_for_instrument(instrument_id: str) -> pd.DataFrame | None:
    """Load all yearly bar parquets for one instrument; return a daily close frame.

    The 2025 and 2026 parquets co-mingle daily (86400s) and 1-minute (60s) bars
    in the same file (see the side task ``fix-bar-storage-frequency-mixing``).
    We filter on ``bar_seconds == 86400`` so the resulting series is strictly
    daily, then de-duplicate timestamps defensively — the same task noted up
    to a few dozen duplicate (instrument, timestamp) rows in the mixed-year
    files, likely an upsert-not-applied bug in the intraday writer.
    """
    folder = BAR_ROOT / instrument_id
    if not folder.exists():
        return None
    files = sorted(folder.glob("*.parquet"))
    if not files:
        files = sorted((folder / "daily").glob("*.parquet"))
    if not files:
        return None
    frames = [pd.read_parquet(p, columns=["timestamp", "close", "bar_seconds"]) for p in files]
    df = pd.concat(frames, ignore_index=True)
    df = df[df["bar_seconds"] == 86400]
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True).astype("datetime64[ns, UTC]")
    df = df.dropna(subset=["close"]).sort_values("timestamp").drop_duplicates("timestamp")
    return df[["timestamp", "close"]].reset_index(drop=True)


def _sector_neutralize_labels(
    samples: list[SupervisedAlphaSample],
    sector_map: Mapping[str, str],
) -> tuple[list[SupervisedAlphaSample], dict[str, float]]:
    """Subtract per-(sector, as_of) mean from each sample's ``forward_return``.

    Approximates a sector-residual return target without needing sector-ETF
    bars: for each rebalance day, the "sector return" is the cross-sectional
    mean of forward returns *within our universe* for that sector. Subtracting
    it removes the sector beta from the label while preserving cross-name
    dispersion (the thing the score is supposed to rank against).

    Returns the rebuilt samples plus a small stats dict (mean abs adjustment,
    distinct sectors observed) for traceability.
    """
    if not samples:
        return samples, {"mean_abs_adjustment": 0.0, "sectors_observed": 0.0}

    rows = pd.DataFrame(
        {
            "instrument_id": [str(s.instrument_id) for s in samples],
            "as_of": [s.as_of for s in samples],
            "forward_return": [s.forward_return for s in samples],
        }
    )
    rows["sector"] = rows["instrument_id"].map(sector_map).fillna("__unknown__")
    sector_mean = rows.groupby(["sector", "as_of"])["forward_return"].transform("mean")
    rows["adjusted"] = rows["forward_return"] - sector_mean

    adjusted_samples = [
        dataclasses.replace(s, forward_return=float(v))
        for s, v in zip(samples, rows["adjusted"].tolist(), strict=True)
    ]
    diag = {
        "mean_abs_adjustment": float((rows["forward_return"] - rows["adjusted"]).abs().mean()),
        "sectors_observed": float(rows["sector"].nunique()),
    }
    return adjusted_samples, diag


def _build_samples(
    feature_frame: pd.DataFrame,
    *,
    horizon_days: int,
    publication_lag_days: int = 1,
    instrument_limit: int | None = None,
    progress_every: int = 25,
) -> tuple[list[SupervisedAlphaSample], dict[str, int]]:
    """Cross-product feature rows × trading days, attach forward returns.

    Stats returned:
        * ``instruments_with_bars`` — instrument count whose bar folder existed.
        * ``instruments_with_samples`` — instrument count contributing >0 samples.
        * ``samples_emitted`` — total samples returned.
        * ``samples_skipped_no_close`` — samples skipped because bar at as_of was missing.
        * ``samples_skipped_no_forward`` — samples skipped because t+horizon close was missing.
        * ``samples_skipped_no_features`` — samples skipped because the as-of
          was before the first usable filing.
    """
    samples: list[SupervisedAlphaSample] = []
    stats = {
        "instruments_with_bars": 0,
        "instruments_with_samples": 0,
        "samples_emitted": 0,
        "samples_skipped_no_close": 0,
        "samples_skipped_no_forward": 0,
        "samples_skipped_no_features": 0,
    }

    feature_cols = list(FEATURE_NAMES)
    grouped = feature_frame.groupby("instrument_id", sort=False)
    total_instruments = (
        grouped.ngroups if instrument_limit is None else min(grouped.ngroups, instrument_limit)
    )
    print(f"  building samples for {total_instruments} instruments ...", flush=True)

    for inst_idx, (instrument_id, ff_df) in enumerate(grouped, start=1):
        if instrument_limit is not None and inst_idx > instrument_limit:
            break
        bars = _load_bars_for_instrument(str(instrument_id))
        if bars is None or bars.empty:
            continue
        stats["instruments_with_bars"] += 1

        ff_df = ff_df.sort_values("datekey").reset_index(drop=True)
        ff_df["as_of"] = pd.to_datetime(ff_df["datekey"], utc=True).astype(
            "datetime64[ns, UTC]"
        ) + pd.Timedelta(days=publication_lag_days)

        # `merge_asof` — for each bar timestamp, the most recent feature row
        # whose as_of <= timestamp. We materialize a single small frame per
        # instrument and discard before moving on.
        merged = pd.merge_asof(
            bars[["timestamp", "close"]],
            ff_df[["as_of", *feature_cols]].sort_values("as_of"),
            left_on="timestamp",
            right_on="as_of",
            direction="backward",
            allow_exact_matches=True,
        )
        before = len(merged)
        merged = merged.dropna(subset=["as_of"])
        stats["samples_skipped_no_features"] += before - len(merged)
        if merged.empty:
            continue

        # Vectorized forward log return.
        close = merged["close"].to_numpy(dtype=float)
        if len(close) <= horizon_days:
            stats["samples_skipped_no_forward"] += len(close)
            continue
        future_close = np.empty_like(close)
        future_close[:-horizon_days] = close[horizon_days:]
        future_close[-horizon_days:] = np.nan
        with np.errstate(divide="ignore", invalid="ignore"):
            forward_log_return = np.log(future_close / close)

        feature_mat = merged[feature_cols].to_numpy(dtype=float)
        feature_finite = np.isfinite(feature_mat)
        feature_has_any = feature_finite.any(axis=1)
        return_finite = np.isfinite(forward_log_return)
        keep = return_finite & feature_has_any
        stats["samples_skipped_no_forward"] += int((~return_finite).sum())

        if not keep.any():
            continue

        timestamps = merged["timestamp"].to_numpy()
        inst_uuid = _uuid.UUID(str(instrument_id))
        kept_idx = np.where(keep)[0]
        for idx in kept_idx:
            features_row: dict[str, float] = {}
            row_finite = feature_finite[idx]
            for j, present in enumerate(row_finite):
                if present:
                    features_row[feature_cols[j]] = float(feature_mat[idx, j])
            # `numpy.datetime64` → Python datetime via pandas Timestamp.
            ts = pd.Timestamp(timestamps[idx]).to_pydatetime()
            samples.append(
                SupervisedAlphaSample(
                    as_of=ts,
                    instrument_id=inst_uuid,
                    features=features_row,
                    forward_return=float(forward_log_return[idx]),
                )
            )
        stats["samples_emitted"] += int(keep.sum())
        stats["instruments_with_samples"] += 1

        if inst_idx % progress_every == 0:
            print(
                f"    {inst_idx:>4}/{total_instruments}  "
                f"with_bars={stats['instruments_with_bars']:>3}  "
                f"with_samples={stats['instruments_with_samples']:>3}  "
                f"total_samples={stats['samples_emitted']:>8,}",
                flush=True,
            )
    return samples, stats


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Walk-forward the starter Sharadar quality+value features."
    )
    parser.add_argument("--train-window-days", type=int, default=252)
    parser.add_argument("--test-window-days", type=int, default=21)
    parser.add_argument("--step-days", type=int, default=21)
    parser.add_argument("--purge-days", type=int, default=5)
    parser.add_argument("--embargo-days", type=int, default=0)
    parser.add_argument("--min-folds", type=int, default=3)
    parser.add_argument(
        "--horizon-days", type=int, default=21, help="Forward-return horizon (trading days)."
    )
    parser.add_argument("--publication-lag-days", type=int, default=1)
    parser.add_argument("--slippage-bps-per-turnover", type=float, default=10.0)
    parser.add_argument("--feature-set-version", default="sharadar-starter-v5-positive-oriented")
    parser.add_argument("--model-version", default="ic-weighted-non-negative")
    parser.add_argument("--out-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument(
        "--instrument-limit",
        type=int,
        default=None,
        help="Cap instrument count for quick sanity runs.",
    )
    parser.add_argument(
        "--portfolio-mode",
        choices=["signed-rank", "long-only-vol-scaled"],
        default="signed-rank",
        help=(
            "signed-rank (default, baseline) or long-only-vol-scaled "
            "(production-style constructor)."
        ),
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=30,
        help=(
            "Top-N names held (long-only constructor only). Default 30 "
            "(~top decile of 329-name universe)."
        ),
    )
    parser.add_argument(
        "--vol-target",
        type=float,
        default=0.15,
        help="Annual vol target for the long-only constructor.",
    )
    parser.add_argument(
        "--max-gross-exposure",
        type=float,
        default=0.22,
        help=(
            "Max gross book size (long-only). Default 0.22 is a conservative "
            "standalone sleeve cap for slow fundamentals."
        ),
    )
    parser.add_argument("--max-single-name-weight", type=float, default=0.05)
    parser.add_argument(
        "--rebalance-interval-days",
        type=int,
        default=21,
        help=(
            "Rebalance cadence in trading days (long-only). Default 21 = "
            "monthly, matches fundamentals filing cadence."
        ),
    )
    parser.add_argument(
        "--no-trade-band",
        type=float,
        default=0.005,
        help="Hold current weight when desired change is below this band (long-only).",
    )
    parser.add_argument(
        "--sector-neutral",
        action="store_true",
        help=(
            "Replace each feature with its sector-relative residual "
            "(feature - sector_median per datekey). Attacks sector-clustering "
            "at the feature level."
        ),
    )
    parser.add_argument(
        "--sector-neutral-label",
        action="store_true",
        help=(
            "Subtract per-(sector, as_of) mean forward return from each "
            "sample's label. Removes sector beta from the target — directly "
            "attacks drawdown that comes from the whole factor going negative "
            "for months."
        ),
    )
    args = parser.parse_args(argv)

    print("Loading Sharadar SF1 panel + computing starter features ...")
    t0 = time.monotonic()
    panel = load_sharadar_sf1_panel()
    # Load the sector map once if any sector-aware mode is active.
    sector_map: dict[str, str] | None = None
    if args.sector_neutral or args.sector_neutral_label:
        sector_map = load_sector_map()
        print(f"  sector map loaded: {len(sector_map)} instruments")
    if args.sector_neutral:
        ff = compute_starter_features(panel, sector_neutralize=True, sector_map=sector_map)
        print("  sector-neutral FEATURES mode ON")
    else:
        ff = compute_starter_features(panel)
    feature_frame = ff.frame.dropna(subset=list(FEATURE_NAMES), how="all").reset_index(drop=True)
    print(
        f"  feature_frame rows={len(feature_frame):,}  "
        f"instruments={feature_frame['instrument_id'].nunique()}  "
        f"({time.monotonic() - t0:.1f}s)"
    )

    print(
        f"Building supervised samples (horizon={args.horizon_days}d, "
        f"pub_lag={args.publication_lag_days}d) ..."
    )
    t0 = time.monotonic()
    samples, sample_stats = _build_samples(
        feature_frame,
        horizon_days=args.horizon_days,
        publication_lag_days=args.publication_lag_days,
        instrument_limit=args.instrument_limit,
    )
    print(f"  samples={len(samples):,}  ({time.monotonic() - t0:.1f}s)")
    for key, value in sample_stats.items():
        print(f"    {key}: {value:,}")

    label_stats: dict[str, float] = {}
    if args.sector_neutral_label:
        if sector_map is None:
            raise RuntimeError("internal error: sector_map missing for sector-neutral-label")
        print(
            "Applying sector-neutral LABEL (subtracting per-(sector, as_of) "
            "mean forward return) ..."
        )
        t0 = time.monotonic()
        samples, label_stats = _sector_neutralize_labels(samples, sector_map)
        print(f"  done in {time.monotonic() - t0:.1f}s")
        for key, value in label_stats.items():
            print(f"    {key}: {value}")

    if not samples:
        print("ERROR: no samples emitted; cannot run walk-forward.")
        return 2

    wf_config = WalkForwardConfig(
        train_window_days=args.train_window_days,
        test_window_days=args.test_window_days,
        step_days=args.step_days,
        purge_days=args.purge_days,
        embargo_days=args.embargo_days,
        min_folds=args.min_folds,
    )
    thresholds = AlphaEligibilityThresholds()

    if args.portfolio_mode == "long-only-vol-scaled":
        portfolio_config = CampaignPortfolioConfig(
            mode="runtime-long-only",
            top_n=args.top_n,
            vol_target=args.vol_target,
            vol_floor=0.05,
            vol_lookback_days=63,
            max_gross_exposure=args.max_gross_exposure,
            min_cash_buffer=0.05,
            max_single_name_weight=args.max_single_name_weight,
            max_daily_turnover=0.20,
            max_position_change=0.05,
            no_trade_band=args.no_trade_band,
            rebalance_interval_days=args.rebalance_interval_days,
        )
        print(
            f"\nRunning walk-forward (long-only-vol-scaled, "
            f"top_n={args.top_n}, "
            f"rebalance={args.rebalance_interval_days}d, "
            f"no_trade_band={args.no_trade_band}) ..."
        )
    else:
        portfolio_config = None
        print("\nRunning walk-forward (ic_weighted, signed-rank baseline) ...")

    t0 = time.monotonic()
    evidence = run_sample_walk_forward(
        samples=samples,
        config=wf_config,
        model_version=args.model_version,
        feature_set_version=args.feature_set_version,
        thresholds=thresholds,
        slippage_bps_per_turnover=args.slippage_bps_per_turnover,
        feature_names=list(FEATURE_NAMES),
        weight_mode="ic_weighted",
        return_scale=1.0,
        portfolio_config=portfolio_config,
    )
    print(
        f"  folds={len(evidence.folds)}  "
        f"daily_observations={len(evidence.daily_returns)}  "
        f"({time.monotonic() - t0:.1f}s)"
    )

    # ---- Report ----
    print("\nSelected feature weights (final fold):")
    for name in FEATURE_NAMES:
        w = evidence.selected_weights.get(name, 0.0)
        marker = "*" if abs(w) > 1e-9 else " "
        print(f"  {marker} {name:>26}: {w:+.4f}")

    print("\nMetrics:")
    for k in (
        "oos_rolling_ic",
        "ic_60d",
        "fold_negative_ic_streak",
        "daily_negative_ic_streak",
        "max_drawdown",
        "slippage_adjusted_sharpe",
        "total_return",
        "turnover_avg",
        "bootstrap_ic_p05",
        "bootstrap_ic_p95",
        "top_minus_bottom_decile_ic",
    ):
        v = evidence.metrics.get(k)
        print(f"  {k:>30}: {v!r}")

    print("\nEligibility:")
    for check in evidence.eligibility["checks"]:
        mark = "PASS" if check["passed"] else "FAIL"
        print(
            f"  [{mark}] {check['name']:>26}  "
            f"actual={check['actual']:>10.4f}  "
            f"threshold={check['threshold']}"
        )
    overall = "PASS" if evidence.eligibility["passed"] else "FAIL"
    print(f"  OVERALL: {overall}")

    # ---- Persist ----
    run_dir = args.out_root / str(evidence.run_id)
    run_dir.mkdir(parents=True, exist_ok=True)

    (run_dir / "eligibility.json").write_text(
        json.dumps(evidence.eligibility, indent=2, default=float),
        encoding="utf-8",
    )
    (run_dir / "metrics.json").write_text(
        json.dumps(dict(evidence.metrics), indent=2, default=float),
        encoding="utf-8",
    )
    (run_dir / "selected_weights.json").write_text(
        json.dumps(dict(evidence.selected_weights), indent=2, default=float),
        encoding="utf-8",
    )
    summary = {
        "run_id": str(evidence.run_id),
        "generated_at": datetime.now(UTC).isoformat(),
        "model_version": args.model_version,
        "feature_set_version": args.feature_set_version,
        "feature_names": list(FEATURE_NAMES),
        "walk_forward_config": {
            "train_window_days": args.train_window_days,
            "test_window_days": args.test_window_days,
            "step_days": args.step_days,
            "purge_days": args.purge_days,
            "embargo_days": args.embargo_days,
            "min_folds": args.min_folds,
        },
        "sample_construction": {
            "horizon_days": args.horizon_days,
            "publication_lag_days": args.publication_lag_days,
            "slippage_bps_per_turnover": args.slippage_bps_per_turnover,
            **sample_stats,
            "samples_total": len(samples),
        },
        "portfolio_mode": args.portfolio_mode,
        "portfolio_config": (
            portfolio_config.to_payload() if portfolio_config is not None else None
        ),
        "sector_neutral": bool(args.sector_neutral),
        "sector_neutral_label": bool(args.sector_neutral_label),
        "label_stats": label_stats,
        "metrics": dict(evidence.metrics),
        "eligibility": dict(evidence.eligibility),
        "selected_weights": dict(evidence.selected_weights),
    }
    (run_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, default=float),
        encoding="utf-8",
    )
    print(f"\nWrote artifacts to {run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
