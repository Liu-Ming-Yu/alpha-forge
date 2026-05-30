"""Replay-validate Arm Q's full 30-name book through the REAL order path.

The parity harness (``validate_arm_q_live_parity.py``) proved Q's *scoring,
selection, and conviction sizing* match the backtest (30/30 overlap, weight-L1
0.0). It did NOT exercise order execution. This harness closes that gap: it
drives one full rebalance cycle through the **real** live order path —
``run_strategy_cycle`` over a real ``create_paper_session`` (real order planner,
pre-trade gate = risk + liquidity + the REAL market-hours check, and the
``SimulatedBrokerGateway`` fill engine).

To get full-universe coverage and an open market without a live data feed, it
pins a ``FakeClock`` to **2026-05-01 14:30 UTC** (10:30 ET, a regular Friday
session). 2026-05-01 is the last date on which *every* universe_300 name has a
bar (min per-instrument max-ts), so all 330 names price and size; 10:30 ET means
the real ``trading_hours_enforced`` gate sees an open market (it rejected the
as-of-today run as ``market_closed`` 7 min after Friday's close).

This is PLUMBING validation only — it proves the full book flows
plan -> approve -> fill on a full-data day. Partial-universe live-soak results
are NOT performance evidence (per the operator directive).

``--cash`` sets the paper NAV. The production soak runs $50k, but on $50k the
top-30 conviction book invests only ~16.5% (regime_scale x gross) = ~$275/name,
so whole-share rounding drops the low-conviction tail to 0 shares (only the top
few names clear 1 lot). To demonstrate the *full* 30-name book flowing through
fills, the default is $1,000,000 (≈$5.5k/name → every name clears >=1 share).
The thin-tail effect at $50k is a real small-account sizing constraint, not an
order-path defect — flagged separately.

Run:
    QP__RISK__MAX_GROSS_EXPOSURE=0.22 \\
    python scripts/validate_arm_q_replay_order_path.py [--cash 1000000]
"""

from __future__ import annotations

import glob
import json
import sys
import uuid
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path

# psycopg3 async requires SelectorEventLoop; Windows defaults to ProactorEventLoop.
import asyncio  # noqa: E402

if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

import pandas as pd  # noqa: E402

from quant_platform.application.operator.cli_inputs import load_instrument_contracts  # noqa: E402
from quant_platform.config import PlatformSettings  # noqa: E402
from quant_platform.core.algorithms.portfolio_construction import (  # noqa: E402
    LongOnlyPortfolioConstructor,
)
from quant_platform.core.domain.market_data.bars import MarketBar  # noqa: E402
from quant_platform.core.domain.research import RunStatus, RunType, StrategyRun  # noqa: E402
from quant_platform.infrastructure.support.clock import FakeClock  # noqa: E402
from quant_platform.services.research_service.features.pv_formulaic.family import (  # noqa: E402
    PV_FORMULAIC_FEATURE_SET_VERSION,
    build_pv_formulaic_feature_bundle,
)
from quant_platform.services.signal_service.scoring import LinearWeightSignalModel  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parent.parent
UNIVERSE = PROJECT_ROOT / "infra" / "config" / "universe_300.json"
BARS = PROJECT_ROOT / "data" / "parquet" / "bars"
G_EVIDENCE = (
    PROJECT_ROOT
    / "data/parquet/research/backtest_latest_stack_realized_v2"
    / "arm_long_only_top30_pv_formulaic_streakdial.json"
)
AS_OF = datetime(2026, 5, 1, 14, 30, tzinfo=UTC)  # 10:30 ET, regular Friday session
YEARS = (2024, 2025, 2026)
GROSS = Decimal("0.22")
SHRINKAGE = 0.25
TOP_N = 30


def _bars_by_instrument() -> dict[uuid.UUID, list[MarketBar]]:
    universe = json.loads(UNIVERSE.read_text())
    out: dict[uuid.UUID, list[MarketBar]] = {}
    for inst in universe:
        rows: list[MarketBar] = []
        for yr in YEARS:
            for fp in glob.glob(str(BARS / inst / "daily" / f"{yr}.parquet")):
                df = pd.read_parquet(fp)
                df = df[df["bar_seconds"] == 86400]
                for r in df.itertuples():
                    ts = pd.Timestamp(r.timestamp)
                    if ts.tzinfo is None:
                        ts = ts.tz_localize(UTC)
                    if ts.to_pydatetime() > AS_OF:
                        continue
                    rows.append(
                        MarketBar(
                            bar_id=uuid.uuid4(),
                            instrument_id=uuid.UUID(inst),
                            timestamp=ts.to_pydatetime(),
                            bar_seconds=86400,
                            open=Decimal(str(r.open)),
                            high=Decimal(str(r.high)),
                            low=Decimal(str(r.low)),
                            close=Decimal(str(r.close)),
                            volume=int(r.volume),
                        )
                    )
        if rows:
            out[uuid.UUID(inst)] = sorted(rows, key=lambda b: b.timestamp)
    return out


def _market_prices(bars: dict[uuid.UUID, list[MarketBar]]) -> dict[uuid.UUID, Decimal]:
    """Last close <= as_of per instrument (the price the order path sizes/fills on)."""
    return {inst: rows[-1].close for inst, rows in bars.items() if rows}


async def main() -> int:
    import argparse

    from quant_platform.bootstrap.session.public_api import create_paper_session
    from quant_platform.engines.session.public_api import run_strategy_cycle

    parser = argparse.ArgumentParser(description="Arm Q replay order-path validation")
    parser.add_argument(
        "--cash",
        type=Decimal,
        default=Decimal("1000000"),
        help="Paper NAV. Default 1e6 to clear whole-share rounding on all 30 names.",
    )
    parser.add_argument(
        "--rebalance-threshold",
        type=Decimal,
        default=Decimal("0.0005"),
        help=(
            "Min |delta_weight| to trade. The PRODUCTION default is 0.01 (1%%), which "
            "skips a 30-name conviction book's tail (avg weight ~0.7%% < 1%%) so only "
            "~3 names trade. Lowered here to 0.05%% to validate the FULL book flows."
        ),
    )
    parser.add_argument(
        "--max-cycles",
        type=int,
        default=5,
        help="Cycles to run. The order throttle (capacity 10, 2 tok/s) caps each "
        "cycle at 10 submits; the clock advances between cycles to refill it, so "
        "the 30-name book establishes over ~3 cycles.",
    )
    parser.add_argument(
        "--clock-advance",
        type=float,
        default=10.0,
        help="Seconds to advance the FakeClock between cycles (refills the token bucket).",
    )
    parsed = parser.parse_args()
    cash = parsed.cash
    rebalance_threshold = parsed.rebalance_threshold
    max_cycles = parsed.max_cycles
    clock_advance = parsed.clock_advance

    weights = json.loads(G_EVIDENCE.read_text())["selected_weights"]
    print(f"[1] Loading bars (<= {AS_OF.date()}) for universe_300 ...")
    bars = _bars_by_instrument()
    prices = _market_prices(bars)
    print(f"    {len(bars)} instruments with bars, {len(prices)} priced")

    print("[2] Computing Q features (rank-normalized) ...")
    bundle = build_pv_formulaic_feature_bundle(bars, as_of=AS_OF)
    feature_data = bundle.alpha_features
    print(f"    feature vectors: {len(feature_data)}  (set={PV_FORMULAIC_FEATURE_SET_VERSION})")

    print("[3] Building REAL paper session (FakeClock @ market-open) ...")
    settings = PlatformSettings()
    run_id = uuid.uuid4()
    clock = FakeClock(AS_OF)
    model = LinearWeightSignalModel(weights, model_version="ic-weighted-non-negative")
    constructor = LongOnlyPortfolioConstructor(top_n=TOP_N, conviction_shrinkage=SHRINKAGE)
    session = create_paper_session(
        settings=settings,
        initial_cash=cash,
        strategy_run_id=run_id,
        clock=clock,
        signal_model=model,
        portfolio_constructor=constructor,
        instrument_contracts=load_instrument_contracts(UNIVERSE),
    )
    await session.broker.connect()
    # The real planner's production default rebalance_threshold (1%) skips the
    # conviction tail; lower it so the full 30-name book establishes for this
    # plumbing test. (Reported as a finding: production needs this tuned.)
    if session.order_planner is not None:
        session.order_planner._rebalance_threshold = rebalance_threshold  # noqa: SLF001
    strategy_run = StrategyRun(
        run_id=run_id,
        strategy_name="arm_q_replay",
        strategy_version="0.1.0",
        run_type=RunType.PAPER,
        status=RunStatus.RUNNING,
        config_snapshot={},
        created_at=AS_OF,
        started_at=AS_OF,
    )

    print(f"[4] Running up to {max_cycles} real cycles through the order path ...")
    target_ids: set[uuid.UUID] = set()
    filled_ids: set[uuid.UUID] = set()
    n_target = 0
    all_reasons: set[str] = set()
    total_approved = 0
    for cycle_i in range(max_cycles):
        if cycle_i > 0:
            clock.advance(seconds=clock_advance)  # refill the submission token bucket
        result = await run_strategy_cycle(
            session,
            feature_data,
            strategy_run,
            market_prices=prices,
            as_of=clock.now(),
        )
        if result.target is not None and not target_ids:
            target_ids = set(result.target.weights.keys())
            n_target = len(target_ids)
        filled_ids |= {f.instrument_id for f in result.fills}
        total_approved += len(result.approved)
        all_reasons |= {getattr(o, "rejection_reason", None) or "?" for o in result.rejected}
        covered = len(filled_ids & target_ids) if target_ids else 0
        print(
            f"    cycle {cycle_i + 1}: approved={len(result.approved)} "
            f"submitted={len(result.submitted_ids)} fills={len(result.fills)} "
            f"cumulative_coverage={covered}/{n_target}"
        )
        if target_ids and target_ids <= filled_ids:
            break

    covered = len(filled_ids & target_ids) if target_ids else 0
    reasons = sorted(r for r in all_reasons if r != "?") or ["none"]

    print("\n============== ARM Q REPLAY - REAL ORDER PATH ==============")
    print(f"as-of (cycle 1)      : {AS_OF.isoformat()}  (full-data day, market open)")
    print(f"paper NAV            : {cash}")
    print(f"target names         : {n_target}")
    print(f"orders approved (sum): {total_approved}")
    print(f"rejection reasons    : {reasons}")
    print(f"book coverage        : {covered}/{n_target} target names filled")

    # PASS: the full top-30 book selects, every name approves through the real
    # pre-trade gate (risk + liquidity + market-hours), and the book establishes
    # to >=90% coverage across cycles with NO market_closed rejection.
    no_market_closed = "market_closed" not in all_reasons
    coverage_ok = covered >= int(0.9 * TOP_N) if cash >= Decimal("500000") else covered > 0
    ok = n_target == TOP_N and total_approved > 0 and no_market_closed and coverage_ok
    print(f"\nREPLAY ORDER-PATH: {'PASS' if ok else 'MISMATCH'}")
    if not ok:
        print(
            "  expected: target=30, approved>0, no market_closed, coverage>=90% "
            f"(NAV>=500k); got target={n_target}, approved={total_approved}, "
            f"coverage={covered}, market_closed={'yes' if not no_market_closed else 'no'}"
        )
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
