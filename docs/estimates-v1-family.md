# `estimates-v1` Feature Family

> Definitive reference for the analyst-estimate-revisions feature
> family registered under `family="estimates"`,
> `version="estimates-v1"`. **The real IBES / FactSet / Visible
> Alpha data feeds are not yet wired** into the platform — all paid
> vendor products. v1 ships the family scaffold against explicit
> input dataclass contracts. Tests use synthetic fixtures.

## At a glance

| Field | Value |
|---|---|
| Family name | `estimates` |
| Family version | `estimates-v1` |
| Source files | `src/quant_platform/research/features/estimates/` |
| Public entry point | `compute_estimate_features(consensus_snapshots, surprise_records, trading_dates, config)` |
| Required input records | `ConsensusSnapshot`, `EarningsSurpriseRecord` |
| Feature count | **6** |
| Tests | `tests/unit/research_service/features/estimates/` (35 tests) |
| Data-feed status | **scaffold only** — populating the family needs a vendor feed (IBES Summary file, FactSet Estimates, Visible Alpha, or Sharadar equivalents) |

## The 6 features

### Revision magnitude (2)

| Feature | Formula |
|---|---|
| `eps_estimate_revision_30d` | `(eps_mean[T] − eps_mean[T−30 cal days]) / |eps_mean[T−30]|` |
| `revenue_estimate_revision_30d` | same shape for FY1 revenue |

### Revision direction (1)

| Feature | Formula |
|---|---|
| `eps_estimate_up_vs_down_30d` | `(n_up − n_down) / (n_up + n_down)` over trailing 30 days; ∈ [-1, 1]; NaN when no revisions |

### Uncertainty + coverage (2)

| Feature | Formula |
|---|---|
| `eps_estimate_dispersion` | `std(estimates) / |mean(estimates)|`; NaN for single-analyst or zero-mean |
| `analyst_coverage_count` | count of analysts contributing to the FY1 EPS consensus |

### Surprise history (1)

| Feature | Formula |
|---|---|
| `eps_surprise_mean_4q` | mean of `(actual − consensus) / |consensus|` over the last 4 reported quarters |

All six ship `expected_direction="unknown"`, `larger_is_better=False`
— evidence-gated. The literature has decades of conflicting findings
on sign (PEAD vs overreaction, surprise drift vs reversal). Promotion
to a directional spec is a family-version bump.

## Input record contracts

```python
@dataclass(frozen=True)
class ConsensusSnapshot:
    instrument_id: str
    snapshot_date: datetime          # tz-aware; IBES Summary daily cadence
    target_period: str               # FY1 / FY2 / Q1 / Q2 / Q3 / Q4
    estimate_kind: str               # "eps" or "revenue"
    mean_estimate: float
    std_estimate: float | None       # None for single-analyst coverage
    n_estimates: int                 # > 0
    n_up_revisions_30d: int = 0
    n_down_revisions_30d: int = 0

@dataclass(frozen=True)
class EarningsSurpriseRecord:
    instrument_id: str
    fiscal_period_end: datetime      # tz-aware
    actual_eps: float
    consensus_mean_eps: float
    consensus_std_eps: float | None
    reported_at: datetime            # tz-aware; >= fiscal_period_end
```

Each record validates its inputs in `__post_init__`. Adding/renaming
a field is a v2 bump.

## PIT safety

- **Consensus snapshots** are forward-filled by their own
  `snapshot_date`. IBES publishes the Summary file daily; the snapshot
  date itself is when the consensus is public.
- **Lagged consensus** (used by the revision features) is computed by
  treating each snapshot as "available at `snapshot_date + window`"
  and doing the same `merge_asof(direction="backward")` join. This
  gives us "the consensus mean from approximately `window` days ago"
  in a single PIT-safe sweep.
- **Surprise records** are masked from the panel until
  `reported_at <= panel_date` — the actuals aren't known publicly
  before they're reported.

The aggregator uses `pd.merge_asof(direction="backward",
by="instrument_id")` for every per-stream join (4 joins total: EPS
current, EPS lagged, revenue current, revenue lagged, surprise mean).

## Compute pipeline

```text
consensus_snapshots + surprise_records + trading_dates + config
        │
        ▼
build_estimates_panel
        │  ├─ Filter consensus by (target_period, estimate_kind)
        │  ├─ For surprises, compute trailing-N pct mean per instrument
        │  └─ Materialise (instrument × trading_dates) grid
        ▼
4 as-of joins (each merge_asof, backward, by="instrument_id"):
        │  - eps_mean         (snapshot_date)
        │  - eps_mean_lag_30  (snapshot_date + 30 days)
        │  - revenue_mean     (snapshot_date)
        │  - revenue_mean_lag_30 (snapshot_date + 30 days)
        │  - eps_surprise_mean_recent (reported_at)
        ▼
compute_estimate_features
        │  - eps_estimate_revision_30d   = safe_div(eps_mean − eps_mean_lag_30, |eps_mean_lag_30|)
        │  - eps_estimate_up_vs_down_30d = safe_div(n_up − n_down, n_up + n_down)
        │  - eps_estimate_dispersion     = safe_div(eps_std, |eps_mean|)
        │  - analyst_coverage_count      = eps_n
        │  - eps_surprise_mean_4q        = eps_surprise_mean_recent
        │  - revenue_estimate_revision_30d = analogous to EPS
        ▼
FeatureFrame (8 cols: instrument_id + date + 6 features)
```

## Configuration

```python
@dataclass(frozen=True)
class EstimatesConfig(BaseFamilyConfig):
    version: str = "estimates-v1"
    eps_target_period: str = "FY1"
    revenue_target_period: str = "FY1"
    revision_window_days: int = 30
    surprise_lookback_quarters: int = 4
```

Constraints (enforced by `__post_init__`):

- `eps_target_period` / `revenue_target_period` ∈ `ALLOWED_TARGET_PERIODS = ("FY1", "FY2", "Q1", "Q2", "Q3", "Q4")`.
- `revision_window_days ≥ 1`.
- `surprise_lookback_quarters ≥ 1`.

`revision_window_days` and `surprise_lookback_quarters` both appear in
feature column names — changing either is a family-version bump.

## Operator quickstart

```python
import pandas as pd
from quant_platform.research.features.estimates import (
    DEFAULT_CONFIG,
    ConsensusSnapshot,
    EarningsSurpriseRecord,
    compute_estimate_features,
)

consensus: list[ConsensusSnapshot] = ...        # populate from IBES Summary
surprises: list[EarningsSurpriseRecord] = ...   # populate from IBES Actuals

trading_dates = pd.date_range("2024-01-01", "2025-01-01", freq="B")

ff = compute_estimate_features(
    consensus_snapshots=consensus,
    surprise_records=surprises,
    trading_dates=trading_dates,
    config=DEFAULT_CONFIG,
)

ff.frame              # 8-column DataFrame
ff.coverage           # per-feature notna() count
```

## What's deferred

- **Real vendor wiring.** No operator script populates the input
  records yet. IBES is the canonical source; FactSet, Visible Alpha,
  and Sharadar Core US Actuals are alternatives.
- **Z-score surprises.** `EarningsSurpriseRecord` carries
  `consensus_std_eps` so a future feature can compute `(actual −
  consensus) / consensus_std` — a more robust surprise metric than
  the v1 percent-surprise. Deferred until the data feed lands.
- **Pre-announcements and guidance.** Pre-announcements (vendor
  detail) and management guidance are separate signals. Out of v1
  scope.
- **Estimate-revision dispersion change.** Whether analyst dispersion
  is *widening* or *narrowing* over time is a third-derivative
  signal that may add value beyond level-dispersion. Deferred.

## Where to look next

- Code: `src/quant_platform/research/features/estimates/`
- Tests: `tests/unit/research_service/features/estimates/` (35 tests)
- Comparable scaffold-first family: [`ownership-v1-family.md`](ownership-v1-family.md)
- Phase status: [`architecture/production-roadmap.md`](architecture/production-roadmap.md)
