# `macro-v1` Feature Family

> Definitive reference for the macro/regime feature family registered
> under `family="macro"`, `version="macro-v1"`. Unlike every other
> feature family, **macro values are scalar per date** — the same
> value is broadcast across all instruments. The compute function
> takes an explicit `instruments` list and produces a standard
> (instrument_id, date)-keyed `FeatureFrame`.

## At a glance

| Field | Value |
|---|---|
| Family name | `macro` |
| Family version | `macro-v1` |
| Source files | `src/quant_platform/research/features/macro/` |
| Public entry point | `compute_macro_features(series_values, instruments, trading_dates, config)` |
| Required input | `MacroSeriesValue` records (one per series, per date) |
| Feature count | **6** |
| Tests | `tests/unit/research_service/features/macro/` (31 tests) |
| Data-feed status | **Feed-agnostic.** Family is scaffolded; an operator-only `fetch_fred_series()` helper exists for the common case (FRED is free). |

## The 6 features

| Feature | Formula | FRED series |
|---|---|---|
| `yield_curve_slope_10y_2y` | `DGS10 − DGS2` | 10y, 2y Treasury |
| `yield_curve_slope_10y_3m` | `DGS10 − DGS3MO` | NY Fed recession-model curve |
| `credit_spread_baa_aaa` | `BAA − AAA` | Moody's corporate yields |
| `vix_level` | `VIXCLS` (direct) | CBOE VIX |
| `dollar_index_change_30d` | `(DTWEXBGS[T] − DTWEXBGS[T-30]) / DTWEXBGS[T-30]` | Broad USD index |
| `real_yield_10y` | `DFII10` (direct) | 10y TIPS |

All six ship `expected_direction="unknown"`, `larger_is_better=False`
— evidence-gated. Macro indicators have well-known historical
signals (curve inversion → recession; high VIX → equity weakness)
but their predictive power on the **short-horizon** forward returns
this platform optimises for is empirically inconsistent.

## Why these 6 (and not 50)

Six distinct *dimensions* of macro state, one feature each:

1. **Two yield-curve slopes** — different predictive horizons:
   10y-2y is the trader's-favourite, 10y-3m is the NY Fed's preferred
   recession-probability input.
2. **Credit spread** — separate from rates; the stress-regime
   compass.
3. **Equity vol regime** — VIX (no cleaner indicator exists).
4. **FX momentum** — dollar strength as risk-on/off proxy.
5. **Real rate** — strips out inflation expectations. Distinct
   information from the nominal yield curve.

Higher-dimensional macro features (multi-factor regimes, mixture
models, principal components across yield-curve tenors) defer to
v2 once the v1 panel's evidence base accumulates.

## Input record contract

```python
@dataclass(frozen=True)
class MacroSeriesValue:
    series_id: str                 # FRED ID, e.g. "DGS10"
    observation_date: datetime     # tz-aware
    value: float                   # finite (NaN rejected at boundary)
```

The schema rejects NaN at construction time — the FRED helper
filters them out before record creation. Adding/renaming a field
is a v2 bump.

## PIT safety

Macro observations are forward-filled by `observation_date` using
`pd.merge_asof(direction="backward")` per series. The observation
date is treated as the public-availability date (FRED publishes
end-of-day for the named observation date — no extra lag).

Holiday handling: weekdays without an observation (US holidays,
ad-hoc late publications) get the most-recent prior value via
forward-fill.

## Compute pipeline

```text
MacroSeriesValue records + trading_dates + instruments + config
        │
        ▼
build_macro_panel
        │  ├─ Filter series_values to REQUIRED_SERIES_IDS
        │  ├─ For each required series: per-series merge_asof on the
        │  │  trading-date grid (one column per series)
        │  └─ Missing series → all-NaN column (graceful degradation)
        ▼
Per-date scalar features:
        │  - slope_10y_2y    = DGS10 − DGS2
        │  - slope_10y_3m    = DGS10 − DGS3MO
        │  - credit_spread   = BAA − AAA
        │  - vix_level       = VIXCLS (direct)
        │  - dollar_change   = safe_div(DTWEXBGS - DTWEXBGS_lag30, DTWEXBGS_lag30)
        │  - real_yield      = DFII10 (direct)
        ▼
Broadcast across instruments:
        │  - Cross-join instruments × per_date_panel
        ▼
FeatureFrame (8 cols: instrument_id + date + 6 features)
```

## Configuration

```python
@dataclass(frozen=True)
class MacroConfig(BaseFamilyConfig):
    version: str = "macro-v1"
    dollar_index_window_days: int = 30
```

The dollar-index window appears in the column name
(`dollar_index_change_30d`), so changing it requires a family-version
bump.

## Operator quickstart

### Option A — populate from FRED (free, requires API key)

```python
from quant_platform.research.features.macro import (
    DEFAULT_CONFIG,
    REQUIRED_SERIES_IDS,
    compute_macro_features,
)
from quant_platform.research.features.macro.fetcher import fetch_fred_series

# Free API key at https://fred.stlouisfed.org/docs/api/api_key.html
records = fetch_fred_series(
    series_ids=REQUIRED_SERIES_IDS,
    start_date="2020-01-01",
    end_date="2025-01-01",
    api_key="<FRED_API_KEY>",
)

ff = compute_macro_features(
    series_values=records,
    instruments=["AAPL", "MSFT", "GOOG"],
    trading_dates=pd.date_range("2020-01-01", "2025-01-01", freq="B"),
    config=DEFAULT_CONFIG,
)
```

`fetch_fred_series` lazy-imports `fredapi` (install with
`pip install fredapi`); the family itself does NOT depend on it.

### Option B — populate from any source

Construct `MacroSeriesValue` records directly. Useful when the
operator already maintains a macro-data feed (Sharadar Macro,
Quandl, Bloomberg, a CSV file):

```python
from quant_platform.research.features.macro import MacroSeriesValue

records = [
    MacroSeriesValue(
        series_id="DGS10",
        observation_date=datetime(2024, 6, 3, tzinfo=UTC),
        value=4.50,
    ),
    # ... more records from your own data source
]
```

## What's deferred

- **Per-instrument macro effects.** A more sophisticated version
  would model how the SAME macro shock (e.g. VIX spike) affects
  different instruments differently — e.g., interaction with sector
  betas. v1's broadcast is the simplest possible model.
- **Higher-dimensional regime features.** Mixture models on the
  yield-curve full tenor structure, PCA on a wider macro panel,
  Markov regime-switching states. Deferred to a learning-based v2.
- **Operator-level FRED caching.** Right now the operator calls
  `fetch_fred_series` ad-hoc; a real production setup would cache
  daily fetches to disk to avoid hitting the FRED rate limit.
  Deferred to a separate operator-tooling PR.

## Where to look next

- Code: `src/quant_platform/research/features/macro/`
- Tests: `tests/unit/research_service/features/macro/` (31 tests)
- FRED API: https://fred.stlouisfed.org/docs/api/fred/
- Comparable feed-agnostic family: [`text-event-v2-family.md`](text-event-v2-family.md)
- Phase status: [`architecture/production-roadmap.md`](architecture/production-roadmap.md)
