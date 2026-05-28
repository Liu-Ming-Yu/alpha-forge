# `options-v1` Feature Family

> Definitive reference for the options-implied feature family
> registered under `family="options"`, `version="options-v1"`.
> **The real options-chain data feed is not yet wired** into the
> platform (CBOE, OptionMetrics, Polygon options, ORATS — all paid
> vendor products). v1 ships the family scaffold against a single
> :class:`OptionsSnapshot` input contract. Tests use synthetic
> fixtures only.

## At a glance

| Field | Value |
|---|---|
| Family name | `options` |
| Family version | `options-v1` |
| Source files | `src/quant_platform/research/features/options/` |
| Public entry point | `compute_options_features(snapshots, trading_dates, config)` |
| Required input | `OptionsSnapshot` |
| Feature count | **6** |
| Tests | `tests/unit/research_service/features/options/` (31 tests) |
| Data-feed status | **scaffold only** |

## The 6 features

| Feature | Formula | Captures |
|---|---|---|
| `iv_30d_atm` | direct pass-through | At-the-money implied vol, 30-day expiry (CBOE-standard) |
| `iv_skew_25d` | `iv_25d_put − iv_25d_call` | Downside-hedging premium |
| `iv_term_slope` | `(iv_60d_atm − iv_30d_atm) / iv_30d_atm` | Contango (+) vs backwardation (−) |
| `put_call_volume_ratio` | `put_volume / call_volume` | Flow-level sentiment |
| `put_call_oi_ratio` | `put_open_interest / call_open_interest` | Position-level sentiment |
| `iv_realized_premium_30d` | `iv_30d_atm − realized_vol_21d` | Vol risk premium |

All six ship `expected_direction="unknown"`, `larger_is_better=False`
— evidence-gated. The options literature has decades of conflicting
findings (skew as crash insurance vs behavioral overpricing; VRP as
risk premium vs mean-reverting predictor).

## Input record contract

```python
@dataclass(frozen=True)
class OptionsSnapshot:
    instrument_id: str
    snapshot_date: datetime              # tz-aware
    iv_30d_atm: float | None             # ATM IV at 30-day expiry (None if vendor couldn't fit)
    iv_60d_atm: float | None             # ATM IV at 60-day expiry
    iv_25d_call: float | None            # 25-delta call IV
    iv_25d_put: float | None             # 25-delta put IV
    put_volume: int                      # >= 0
    call_volume: int                     # >= 0
    put_open_interest: int               # >= 0
    call_open_interest: int              # >= 0
    realized_vol_21d: float | None       # Trailing 21-day realized vol of the underlying
```

The contract takes already-derived metrics (ATM IV, 25Δ IV, totals)
rather than per-contract chains — that's because v1's features only
need the derived surface. Reproducing the surface from raw chains is
a separate piece of work that lives in a future PR.

## PIT safety

The aggregator forward-fills snapshots by `snapshot_date` using
`pd.merge_asof(direction="backward", by="instrument_id")`. Operators
with strict end-of-day-only feeds should ingest records with
`snapshot_date` = the date the data is FIRST available, not the
trading date the surface was fit against.

## Compute pipeline

```text
snapshots + trading_dates + config
        │
        ▼
build_options_panel
        │  ├─ Convert snapshots → long-format DataFrame
        │  └─ Materialise (instrument × trading_dates) grid + merge_asof forward-fill
        ▼
compute_options_features
        │  - iv_30d_atm                = direct
        │  - iv_skew_25d               = iv_25d_put − iv_25d_call
        │  - iv_term_slope             = safe_div(iv_60d − iv_30d, iv_30d)
        │  - put_call_volume_ratio     = safe_div(put_vol, call_vol)
        │  - put_call_oi_ratio         = safe_div(put_oi, call_oi)
        │  - iv_realized_premium_30d   = iv_30d − realized_vol_21d
        ▼
FeatureFrame (8 cols: instrument_id + date + 6 features)
```

## Configuration

```python
@dataclass(frozen=True)
class OptionsConfig(BaseFamilyConfig):
    version: str = "options-v1"
    atm_tenor_days: int = 30            # → iv_30d_atm, iv_realized_premium_30d
    term_long_tenor_days: int = 60      # iv_term_slope long end
    realized_vol_window_days: int = 21  # underlying RV window
```

Constraints (enforced by `__post_init__`):

- `atm_tenor_days ≥ 1`
- `term_long_tenor_days > atm_tenor_days` (strict)
- `realized_vol_window_days ≥ 2`

`atm_tenor_days` appears in feature column names (e.g.
`iv_realized_premium_30d`), so changing it requires a family-version
bump. Same for `term_long_tenor_days` in spec descriptions.

## What's deferred

- **Real vendor wiring.** Operator script that populates
  `OptionsSnapshot` from CBOE, OptionMetrics, Polygon options, ORATS,
  or similar.
- **Per-contract / per-strike features.** Once raw chain data lands,
  v2 can add gamma exposure, vanna, charm, dealer hedging flow
  estimators.
- **Volatility surface anchors.** Butterfly (10Δ vs 25Δ skew) and
  ATM term-structure curvature are natural v2 additions.
- **VIX-style risk-neutral variance.** Model-free implied variance
  via Carr-Madan replication of the full strike spectrum.

## Where to look next

- Code: `src/quant_platform/research/features/options/`
- Tests: `tests/unit/research_service/features/options/` (31 tests)
- Comparable scaffold-first family: [`estimates-v1-family.md`](estimates-v1-family.md)
- Phase status: [`architecture/production-roadmap.md`](architecture/production-roadmap.md)
