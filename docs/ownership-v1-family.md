# `ownership-v1` Feature Family

> Definitive reference for the institutional-holdings + short-interest
> feature family registered under `family="ownership"`,
> `version="ownership-v1"`. **The real 13F and short-interest data
> feeds are not yet wired** into the platform — both are paid vendor
> products. v1 ships the family scaffold against explicit input
> dataclass contracts. Operator scripts that populate the family
> from a vendor land separately; v1's tests use synthetic fixtures.

## At a glance

| Field | Value |
|---|---|
| Family name | `ownership` |
| Family version | `ownership-v1` |
| Source files | `src/quant_platform/research/features/ownership/` |
| Public entry point | `compute_ownership_features(holdings, short_interest, shares_outstanding, trading_dates, config)` |
| Required input records | `Holding13FRecord`, `ShortInterestRecord`, `SharesOutstandingRecord` |
| Feature count | **6** |
| Tests | `tests/unit/research_service/features/ownership/` (30 tests) |
| Data-feed status | **scaffold only** — populating the family requires a vendor data feed (Sharadar SF3, 13F aggregator, FINRA SI files) |

## The 6 features

### Institutional ownership (3)

| Feature | Formula |
|---|---|
| `institutional_ownership_pct` | `sum(13F shares held) / shares_outstanding`, clipped to `[0, 1]` |
| `institutional_holder_count` | count of distinct filers reporting a position |
| `institutional_ownership_change_63d` | per-instrument trailing diff of `institutional_ownership_pct` over 63 trading days (~ 1 quarter) |

### Short interest (3)

| Feature | Formula |
|---|---|
| `short_interest_ratio` | `short_interest_shares / shares_outstanding`, clipped to `[0, 1]` |
| `days_to_cover` | `short_interest_shares / avg_daily_volume_shares` |
| `short_interest_change_20d` | per-instrument trailing diff of `short_interest_ratio` over 20 trading days |

All six features ship `expected_direction="unknown"`,
`larger_is_better=False` — evidence-gated. Common a-priori intuitions
(e.g. "high short interest = bearish contrarian signal", "institutional
accumulation = bullish") are empirically inconsistent across regimes.
Promotion to a directional spec is a family-version bump.

## Input record contracts

```python
@dataclass(frozen=True)
class Holding13FRecord:
    filer_id: str           # CIK or vendor manager ID
    instrument_id: str
    period_end: datetime    # tz-aware
    shares_held: int        # >= 0
    market_value: float     # >= 0
    available_at: datetime | None = None  # default: period_end + 45 days

@dataclass(frozen=True)
class ShortInterestRecord:
    instrument_id: str
    settlement_date: datetime    # tz-aware
    short_interest_shares: int   # >= 0
    avg_daily_volume_shares: float  # > 0
    available_at: datetime | None = None  # default: settlement_date + 8 days

@dataclass(frozen=True)
class SharesOutstandingRecord:
    instrument_id: str
    period_end: datetime    # tz-aware
    shares_outstanding: int # > 0
```

Each record validates its inputs in `__post_init__`. Adding or renaming
a field is a v2 bump.

## PIT safety

The aggregator masks each record from the panel until its
`available_at` date is reached:

* **13F**: defaults to `period_end + 45 days` (SEC's 13F filing
  deadline).
* **Short interest**: defaults to `settlement_date + 8 days` (FINRA's
  typical publication lag).
* **Shares outstanding**: assumed immediately available at `period_end`
  (it lives in fundamentals data, which the platform already handles
  with its own PIT semantics).

Operators can override `available_at` per-record if a particular
vendor's lag differs.

The aggregator uses `pd.merge_asof(direction="backward", by="instrument_id")`
to forward-fill the most-recently-available record onto each row of
the daily grid. Rows before any record is available stay NaN.

## Compute pipeline

```text
holdings + short_interest + shares_outstanding + trading_dates
        │
        ▼
build_ownership_panel
        │  ├─ Group 13F by (instrument, period_end) → sum shares + nunique filers
        │  ├─ Sort short_interest by (instrument, settlement_date)
        │  └─ Materialise (instrument × trading_dates) grid
        ▼
Per-stream as-of join (merge_asof, direction="backward", by="instrument_id"):
        │  - 13F:   join on (institutional_shares_total, institutional_holder_count)
        │  - SI:    join on (short_interest_shares, avg_daily_volume_shares)
        │  - SO:    join on (shares_outstanding)
        ▼
compute_ownership_features
        │  - institutional_ownership_pct = clip(13F_total / SO, 0, 1)
        │  - institutional_ownership_change_63d = pct − pct.shift(63) per instrument
        │  - short_interest_ratio = clip(SI / SO, 0, 1)
        │  - days_to_cover = SI / avg_daily_volume
        │  - short_interest_change_20d = ratio − ratio.shift(20) per instrument
        ▼
FeatureFrame (6 cols + instrument_id + date)
```

## Configuration

```python
@dataclass(frozen=True)
class OwnershipConfig(BaseFamilyConfig):
    version: str = "ownership-v1"
    holding_13f_availability_lag_days: int = 45
    short_interest_availability_lag_days: int = 8
    holding_13f_change_window_days: int = 63    # → institutional_ownership_change_63d
    short_interest_change_window_days: int = 20 # → short_interest_change_20d
```

Constraints (enforced by `__post_init__`):

- All four integer fields ≥ 0 (lags) or ≥ 1 (windows).

The change-window values appear in the feature column names, so
changing them requires a family-version bump (same convention as
every other family in this platform).

## Operator quickstart

```python
import pandas as pd
from quant_platform.research.features.ownership import (
    DEFAULT_CONFIG,
    Holding13FRecord,
    ShortInterestRecord,
    SharesOutstandingRecord,
    compute_ownership_features,
)

# Populate records from your vendor of choice (Sharadar SF3, etc.).
holdings: list[Holding13FRecord] = ...
short_interest: list[ShortInterestRecord] = ...
shares_outstanding: list[SharesOutstandingRecord] = ...

trading_dates = pd.date_range("2024-01-01", "2025-01-01", freq="B")

feature_frame = compute_ownership_features(
    holdings=holdings,
    short_interest=short_interest,
    shares_outstanding=shares_outstanding,
    trading_dates=trading_dates,
    config=DEFAULT_CONFIG,
)

feature_frame.frame              # 8-column DataFrame (instrument_id, date, + 6 features)
feature_frame.coverage           # per-feature notna() count
```

## What's deferred

* **Real vendor wiring.** No operator script populates the input
  records from a real feed yet. Sharadar Core US has SF3 (institutional
  holdings) and SF3A (aggregates) — these would feed `Holding13FRecord`.
  FINRA publishes short-interest CSV files bi-monthly — these feed
  `ShortInterestRecord`.
* **Concentration features.** v1 ships `institutional_holder_count` but
  not the Herfindahl index of holdings. The aggregator currently
  collapses per-filer detail at the panel boundary; an HHI feature
  would need the per-filer rows to survive into the compute layer.
  Deferred to v2 once we know the per-filer detail is worth carrying
  through.
* **Insider trading features.** Form 4 filings (insider buys/sells) are
  a related signal that lives in a different SEC dataset. Out of v1
  scope.

## Where to look next

- Code: `src/quant_platform/research/features/ownership/`
- Tests: `tests/unit/research_service/features/ownership/` (30 tests)
- Comparable scaffold-first family: [`text-event-v2-family.md`](text-event-v2-family.md)
  (shipped as a scaffold ahead of real LLM provider integration)
- Phase status: [`architecture/production-roadmap.md`](architecture/production-roadmap.md)
