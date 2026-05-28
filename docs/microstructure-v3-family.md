# `microstructure-v3` Feature Family

> Definitive reference for the daily-OHLCV-derived microstructure feature
> family registered under `family="microstructure"`,
> `version="microstructure-v3"`. Designed to be **complementary to**
> `price-volume-starter-v1` — Amihud illiquidity, dollar-volume z-score,
> plain high-low range, overnight gap, and the open-to-close return
> already live there. Tick / quote-level features (Kyle's λ, VPIN, true
> effective spread, order-flow imbalance) defer to a future
> `microstructure-v4` once a minute-bar or trade-tick feed lands.

## At a glance

| Field | Value |
|---|---|
| Family name | `microstructure` |
| Family version | `microstructure-v3` |
| Source files | `src/quant_platform/research/features/microstructure/` |
| Public entry point | `compute_microstructure_features(bars, config)` |
| Required inputs | `instrument_id`, `date`, `open`, `high`, `low`, `close`, `volume` |
| Feature count | **19** (10 v1 + 6 v2 + 3 v3 additions) |
| Tests | `tests/unit/research_service/features/microstructure/` (43 tests) |

## Version history

- **v1** (10 features): range-based vols (Parkinson, Garman-Klass,
  Rogers-Satchell), OHLC spread proxies (Roll, Corwin-Schultz),
  intraday position (close-in-range), serial dependence (return +
  volume autocorrelation), volume-return coupling, range asymmetry.
- **v2** (16 features): adds Yang-Zhang volatility, bipower variation,
  realized skewness + kurtosis, Lo-MacKinlay variance ratio, range
  persistence.
- **v3** (19 features): adds Median Realized Variance (Andersen-
  Dobrev-Schaumburg 2012), tripower variation (Barndorff-Nielsen-
  Shephard 2006), realized jump intensity (Andersen-Bollerslev-
  Diebold 2007). Motivated by a v2 testing finding that bipower
  variation is only robust to **isolated** jumps — clusters
  (jump + immediate reversion) still contaminate it. The v3
  additions provide stronger isolated-jump robustness (MedRV, TPV)
  and a positive signal for the BPV-vs-RV gap (jump intensity).

The family registers through the canonical `FamilyRegistry` like every
other feature family. `bootstrap_default_families()` includes it.

## The 19 features

### Range-based realised volatility (4)

These estimators use the day's high–low (and optionally open/close)
path. Yang-Zhang adds the overnight component.

| Feature | Formula | Drift-independent? | Adds overnight? |
|---|---|---|---|
| `parkinson_vol_20d` | √( mean( ln(H/L)² ) / (4·ln 2) ) over 20 days | No | No |
| `garman_klass_vol_20d` | √( mean( 0.5·ln(H/L)² − (2·ln 2 − 1)·ln(C/O)² ) ) over 20 days | Zero-drift assumed | No |
| `rogers_satchell_vol_20d` | √( mean( ln(H/C)·ln(H/O) + ln(L/C)·ln(L/O) ) ) over 20 days | **Yes** | No |
| `yang_zhang_vol_20d` | √( σ²_overnight + k·σ²_open-to-close + (1−k)·σ²_RS ), k = 0.34 / (1.34 + (N+1)/(N−1)) | **Yes** | **Yes** |

Yang-Zhang is the gold standard daily-OHLC vol estimator: drift-independent
*and* incorporates the overnight gap, making it more efficient than the
other three.

### Bid-ask spread proxies (2)

| Feature | Source | Defined when |
|---|---|---|
| `roll_spread_60d` | Roll (1984): 2·√(−cov(rₜ, rₜ₋₁)) over 60 days | cov < 0 (bid-ask bounce signature); NaN otherwise |
| `corwin_schultz_spread_20d` | Corwin–Schultz (2012) high-low spread estimator over 20 days | Always real-valued; negative raw estimates collapse to 0 |

### Intraday-position structure (1)

| Feature | Formula |
|---|---|
| `close_in_range_20d` | Rolling mean over 20 days of `(close − low) / (high − low)` |

### Serial dependence (3)

| Feature | Computes |
|---|---|
| `return_autocorr_60d` | Per-instrument rolling Pearson correlation between `r_t` and `r_{t-1}` over 60 days |
| `volume_autocorr_60d` | Per-instrument rolling Pearson correlation between `v_t` and `v_{t-1}` over 60 days |
| `range_persistence_20d` | Per-instrument rolling autocorrelation of `(high − low) / close` over 20 days. Volatility-clustering signature |

### Volume-return coupling (1)

| Feature | Computes |
|---|---|
| `volume_return_correlation_20d` | Per-instrument rolling Pearson correlation between `\|r_t\|` and daily volume over 20 days |

### Range asymmetry (1)

| Feature | Formula |
|---|---|
| `high_low_asymmetry_20d` | `(rolling_max(high, 20) − close) / (close − rolling_min(low, 20))` |

### Jump-robust realised variance (1)

| Feature | Formula |
|---|---|
| `bipower_variation_20d` | (π/2) · mean( `\|r_t\| · \|r_{t-1}\|` ) over 20 days |

Barndorff-Nielsen & Shephard (2004). Unlike sum(r²), this is consistent
for the integrated variance under jump-diffusion price processes —
large isolated jumps drop out because each adjacent product touches at
most one jump term.

### Higher-moment realised statistics (2)

| Feature | Statistic |
|---|---|
| `realized_skew_60d` | Third standardised moment of daily log returns over 60 days (Fisher-Pearson, bias-corrected) |
| `realized_kurt_60d` | Excess kurtosis (4th standardised moment − 3) of daily log returns over 60 days |

Negative skew = downside-heavy distribution (crash risk); positive
excess kurt = fat tails.

### Random-walk test (1)

| Feature | Formula |
|---|---|
| `variance_ratio_5_1_60d` | Var(5-day log returns over 60-day window) / (5 × Var(1-day log returns over 60-day window)) |

Lo & MacKinlay (1988). Under a random walk VR(q) = 1. VR < 1 = mean
reversion (bid-ask bounce, short-term overreaction); VR > 1 = positive
serial correlation (trend / momentum at the q-day horizon).

### v3: jump-cluster-robust estimators (3)

These three features were added in v3 to address a v2 testing finding:
**bipower variation is only robust to *isolated* jumps**, not jumps
followed by an immediate mean reversion. The three v3 additions all
attack the jump problem from different angles.

| Feature | Mechanism | Robust to isolated jumps? | Robust to jump clusters? |
|---|---|---|---|
| `med_rv_60d` | Squared *median* of three adjacent `\|r\|` values, scaled by ADS 2012 constant `π/(6−4√3+π) ≈ 1.4194` | **Yes** — median ignores the single jumped outlier | Partial — clusters can put 2 of 3 values in the median window |
| `tripower_variation_20d` | `\|r_t\|^(2/3) · \|r_{t-1}\|^(2/3) · \|r_{t-2}\|^(2/3)`, scaled by BNS 2006 constant `μ_{2/3}^(-3) ≈ 1.9358` | **More robust than BPV** (sub-linear in jump magnitude) | Slightly better than BPV |
| `realized_jump_intensity_20d` | `clip((naive_RV − BPV) / naive_RV, 0, 1)` | **Saturates near 1** on isolated jumps | Partial signal — still measurably elevated above baseline |

**Honest accounting**: clusters where a jump is immediately followed
by a same-magnitude reversion **fundamentally defeat every
daily-OHLCV-only jump estimator** — the cluster looks like
continuous high-volatility on the daily scale. The right fix is
intraday data (deferred to `microstructure-v4`). v3's contribution
is to make isolated-jump detection genuinely robust (MedRV is
strictly better than BPV here, by ≥3× on the testing fixture) and
to expose the BPV-vs-RV gap as the `realized_jump_intensity`
signal — useful even when clusters partially defeat it.

The relationship between the four jump-related features:

```text
RV          = mean( r_t^2 )         — sees everything (signal + jumps + clusters)
BPV         = mean( |r_t| |r_{t-1}| ) · (π/2)
                                    — isolated-jump robust; cluster-vulnerable
TPV         = mean( |r_t|^{2/3} |r_{t-1}|^{2/3} |r_{t-2}|^{2/3} ) · scale
                                    — sub-linear; better than BPV on isolated jumps
MedRV       = mean( med(|r_{t-2}|, |r_{t-1}|, |r_t|)^2 ) · scale
                                    — fully isolated-jump robust; cluster-vulnerable
JumpIntens  = clip( (RV − BPV) / RV, 0, 1 )
                                    — uses the BPV/RV gap as a positive signal
```

## Direction conventions and evidence gating

**All 16 features ship `expected_direction="unknown"` and
`larger_is_better=False`.**

Microstructure signals are too noisy on daily bars to ship with
a-priori direction claims. Promotion to a directional spec is a
**family-version bump**, not an in-place edit. The walk-forward +
signal-gate pipeline is what earns a feature its direction.

## Compute pipeline

```text
OHLCV bars (long-format DataFrame)
        │
        ▼
_validate_inputs  (raises on missing OHLCV column)
        │
        ▼
Sort by (instrument_id, date); per-day terms:
  ln(H/L), ln(C/O), ln(H/C), ln(H/O), ln(L/C), ln(L/O)
  Parkinson / Garman-Klass / Rogers-Satchell daily variance contributions
  close-in-range daily fraction
  daily log return + lag(1)
  overnight log return ln(O/C_prev)
  bipower daily product (π/2)·|r_t|·|r_{t-1}|
  q-stride log return ln(C_t / C_{t-q})
  range-normalised daily (H-L)/C + lag(1)
        │
        ▼
Per-instrument rolling aggregates:
  v1: Range-based vol / Roll / Corwin-Schultz / close-in-range /
      return-autocorr / volume-autocorr / volume-|r| corr /
      high-low asymmetry
  v2: Yang-Zhang vol = overnight_var + k·oc_var + (1-k)·rs_var
      bipower variation = rolling mean of bipower daily product
      realized skew/kurt via pandas Rolling.skew()/kurt() (Fisher-Pearson)
      variance ratio = Var(q-stride logret, window) / (q · Var(1d logret, window))
      range persistence = corr(range_t, range_{t-1}) over short_window
        │
        ▼
Replace ±inf with NaN; assemble FeatureFrame
```

### Numerical safety

- All `log` calls guard against non-positive inputs (replaced with NaN
  before the log) so `log` never returns `-inf`.
- All `sqrt` calls clip the argument to ≥ 0 first — Rogers-Satchell
  and Yang-Zhang can produce a tiny negative variance on outlier rows
  due to OHLC integrity glitches.
- Roll's spread masks `cov ≥ 0` to NaN per the original paper
  (estimator undefined there).
- Corwin-Schultz clips negative raw spreads to 0 per the original
  paper.
- All rolling correlations / covariances / higher moments use
  **explicit per-instrument iteration** (`for inst, group in df.groupby(...)`
  + write into a pre-allocated Series) instead of `groupby.apply`.
  Newer pandas returns a transposed shape on single-instrument frames
  and that breaks alignment.
- The final pass replaces any residual `±inf` with `NaN` before
  assembling the `FeatureFrame`.

## Naming conventions

### Column naming

Most features follow `<stat>_<window>d` (e.g. `parkinson_vol_20d`,
`return_autocorr_60d`). The one exception is the variance ratio,
which is parameterised by *two* knobs (a stride `q` and a rolling
window):

```text
variance_ratio_<q>_1_<window>d
```

The `_1_` is literal — it makes the *base* horizon explicit so the
name reads as "VR of q-stride returns versus 1-stride returns over
window days." This is the family's convention for any future
two-knob feature: smallest-to-largest stride, separated by `_`,
window in days last with the `d` suffix.

### `lookback_days` convention

Every `FeatureSpec.lookback_days` in this family is **the first row
at which the feature is guaranteed non-NaN**, including warm-up cost
from `shift`/`lag` operations:

| Pattern | `lookback_days` |
|---|---|
| Pure rolling (e.g. `parkinson_vol_20d`) | `window` |
| Rolling + lag-1 (e.g. `yang_zhang_vol_20d`, `bipower_variation_20d`, `range_persistence_20d`) | `window + 1` |
| Rolling var of q-stride returns (e.g. `variance_ratio_5_1_60d`) | `window + q` |

This is the "first non-NaN row" interpretation. Walk-forward
infrastructure that pre-warms history at fold boundaries can read
`lookback_days` directly without adding its own fudge factor for the
shift cost.

### Excess kurtosis

`realized_kurt_60d` returns **Fisher's excess kurtosis** (4th
standardised moment − 3), not the raw 4th moment. The spec name
omits an `excess_` prefix per finance convention; the FeatureSpec
description spells it out. Gaussian → 0, fat-tailed → positive.

## Configuration

`MicrostructureConfig` (in `config.py`) carries three knobs. All have
sensible defaults; pass an instance to
`compute_microstructure_features(config=...)` only when you need a
non-default value:

```python
@dataclass(frozen=True)
class MicrostructureConfig(BaseFamilyConfig):
    version: str = "microstructure-v2"
    short_window: int = 20            # → 20d-named features
    long_window: int = 60             # → 60d-named features
    variance_ratio_stride: int = 5    # → variance_ratio_5_1_60d
```

Constraints (enforced by `__post_init__`):

- `short_window ≥ 5` (also protects Yang-Zhang's k-weight denominator
  `(short_window − 1)` from divide-by-zero / very small denominators
  on the noise floor — see `compute_microstructure_features` for the
  explicit YZ-side assertion).
- `long_window ≥ 10`
- `long_window > short_window` (strict)
- `variance_ratio_stride ≥ 2`
- `variance_ratio_stride < long_window`

The windows appear in every feature column name (e.g.
`parkinson_vol_20d`, `variance_ratio_5_1_60d`), so changing them
requires a feature-set version bump. **The compute path enforces this
at runtime**: if a caller drifts any window/stride from `DEFAULT_CONFIG`
while leaving `version` at the default, `compute_microstructure_features`
raises `ValueError`. To use custom windows, set
`version != "microstructure-v2"` (e.g. `"microstructure-experiment-1"`)
to take explicit ownership of the catalogue divergence.

### Yang-Zhang window policy

The YZ k-weight assumes the rolling-variance estimator saw exactly
`short_window` observations on every row. The family does not
currently expose a `min_periods_policy` knob on `MicrostructureConfig`,
but the implementation pins all YZ-relevant rolling calls to
`policy="full"`. If `min_periods_policy` is ever added to the
config, the YZ path must stay on `"full"` regardless.

### Performance note

The four explicit-iteration helpers (`_rolling_corr`, `_rolling_cov`,
`_rolling_higher_moment`, all routing through
`_per_instrument_rolling_op`) loop once per instrument and call
`pd.Series.rolling(...)` inside each group. On a ~300-instrument ×
multi-year panel this is fine. Past ~1000 names, profile before
scaling — the Python-level per-iteration overhead starts to dominate
the vectorised inner kernel.

## Operator quickstart

```python
import pandas as pd
from quant_platform.research.features.microstructure import (
    DEFAULT_CONFIG,
    compute_microstructure_features,
)

bars: pd.DataFrame = ...  # long-format (instrument_id, date, OHLCV)

feature_frame = compute_microstructure_features(bars, config=DEFAULT_CONFIG)

feature_frame.frame              # 18-column DataFrame
                                 # (instrument_id, date, + 16 features)
feature_frame.coverage           # per-feature notna() count
feature_frame.feature_specs      # name → FeatureSpec
feature_frame.training_feature_names  # alias-free training list
```

## What's deferred to `microstructure-v3`

These features need data the platform doesn't yet have on a daily-bar
feed:

- **Kyle's λ** — needs intraday signed volume to regress price impact.
- **VPIN** (volume-synchronized PIN) — needs intraday trade-arrival
  buckets.
- **Quoted spread / depth** — needs Level 1+ quote data.
- **Order-flow imbalance** — needs trade-direction signing (Lee-Ready
  or bulk volume classification).
- **Microstructure noise variance** — needs minute-bar realized
  variance.
- **Sub-daily realized vol decomposition** — needs intraday returns.

Once a trade-tick or minute-bar provider is wired, the daily-OHLCV
proxies in v2 stay valid; v3 adds the higher-frequency signals on top.

## Where to look next

- Code: `src/quant_platform/research/features/microstructure/`
- Tests: `tests/unit/research_service/features/microstructure/` (30 tests)
- Sister daily-OHLCV family: `price-volume-starter-v1`
  (`src/quant_platform/research/features/price_volume/`)
- Comparable LLM family: [`text-event-v2-family.md`](text-event-v2-family.md)
- Phase status: [`architecture/production-roadmap.md`](architecture/production-roadmap.md)
