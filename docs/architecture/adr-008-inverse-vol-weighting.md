# ADR-008 — Shrunk inverse-volatility position weighting

**Status:** Accepted (framework + Arm L). The weighting seam (`campaigns/portfolio/weighting.py`) is shipped and is the durable win — it factors per-name sizing out of the target builder behind a one-method protocol, the third application of the same pattern (`AlphaModel`, ADR-006; `TradingCostModel`, ADR-007). **Arm L** = the production lead **G** with equal-weight sizing replaced by shrunk inverse-volatility weighting. Selection (the alpha) and the gross budget are identical to G — only the per-name *size distribution* changes — so the IC series and the selected set are unchanged by construction; what moves is risk concentration and therefore Sharpe / drawdown / turnover. **Verdict on universe-300: it hurts.** Inverse-vol weighting drops G's Sharpe 1.0886 → 0.9575 (−12%) and total return 14.1% → 11.6% (−18%) while adding +19% turnover, and **L fails eligibility** (Sharpe 0.9575 < the 1.0 gate). A validated negative result: name-level inverse-vol is redundant with the existing portfolio-level vol-target exposure scale (drawdown is unchanged, −4.20% vs −4.21%) and tilts size away from the highest-alpha names. **G's equal weight remains the lead.**

**Context:** the qlib model-adoption review (item 3) flagged that equal weighting ignores risk — a 30%-vol name and a 10%-vol name get the same dollar allocation, so the high-vol name dominates portfolio variance. Inverse-volatility weighting (the risk-parity move qlib's portfolio layer supports) tilts toward lower-risk names; shrinkage toward equal weight blunts the estimation error in a single vol point estimate.

## Context

`raw_long_only_target` selected the top-N positive-score names and assigned **equal weight** to each (`investable_gross / N`, capped). Equal weight is risk-blind: it sizes by count, not by contribution to portfolio variance. Two structural consequences:

1. **The highest-vol names dominate risk.** A book of 30 equal-weighted names where a few carry 3× the volatility of the rest has its variance concentrated in those few — the opposite of diversification.
2. **No knob to express a risk preference.** The only sizing lever was the per-name cap, which is a hard limit, not a smooth tilt.

The per-name volatility needed to fix this is already computed: the price-volume family emits `low_vol_63d` (sign-flipped 63-day return std), a point-in-time feature carried on every sample. Its 63-day window matches the config's `vol_lookback_days`, so the weighting and the exposure-vol-scale see a consistent horizon.

## Decision

Factor per-name sizing into a one-method protocol in `campaigns/portfolio/weighting.py`:

* `WeightingScheme.proportions(selected) -> list[float]` — returns one non-negative proportion per selected name (summing to 1), in order. Plus `name` and `metadata()` for the audit trail.

`raw_long_only_target` computes the same investable-gross budget equal weight produced — `min(N · cap, investable)` — and splits it by the scheme's proportions. `evaluate_long_only_portfolio` and `run_sample_walk_forward` gain `weighting: WeightingScheme | None = None`; `None` uses `EqualWeight`, so arms A–K stay **bit-identical** (equal-weight proportions `1/N · gross` reproduce the prior per-name weight exactly). The per-name and gross caps are enforced downstream, identically for every scheme. `weighting` is rejected for signed-rank arms (no long-only target book to size).

Two concrete schemes ship:

* **`EqualWeight`** — `1/N` per name. The behavior-preserving default.
* **`InverseVolWeight`** — `w_i = shrinkage · (1/N) + (1 − shrinkage) · (inv_i / Σ inv)`, where `inv_i = 1 / max(|vol_i|, vol_floor)`. `shrinkage ∈ [0, 1]` interpolates equal weight (1.0) ↔ pure inverse-vol (0.0). A missing/non-finite vol falls back to the cross-sectional median, so one bad feature value never drops a name or blows up its weight.

Arm **L = G's exact construction with `InverseVolWeight(low_vol_63d, shrinkage=0.5)`** isolates the sizing change against the production lead.

## Options considered

### Inverse-vol vs inverse-variance vs full mean-variance
Chose **inverse-vol** (1/σ). Inverse-variance (1/σ²) tilts harder and is more estimation-error sensitive; full mean-variance needs a covariance matrix (off-diagonal estimation, conditioning, a much larger surface). Inverse-vol is the standard risk-parity-lite sizing and the natural first step; the seam admits the others as siblings later.

### Shrinkage toward equal weight (and why 0.5)
A single 63-day vol estimate is noisy, and pure inverse-vol concentrates the book in a few low-vol names. Shrinking halfway toward equal weight (`shrinkage=0.5`) keeps the risk tilt while bounding concentration and estimation sensitivity — the same motivation as Ledoit-Wolf covariance shrinkage, applied to the weights. 0.5 is a documented, tunable middle; `shrinkage=1.0` recovers equal weight exactly (a built-in sanity equivalence), `0.0` is pure inverse-vol.

### Vol source: reuse `low_vol_63d` vs compute fresh
Reused the existing point-in-time feature. It is already PIT-correct (`signal_timestamp="eod_after_close"`), already on every sample, and its 63-day window matches `vol_lookback_days`. Computing a fresh vol inside the constructor would duplicate logic and risk a look-ahead bug. (The feature is also one of the 36 ranker inputs; using it for sizing too is fine — sizing and signal are independent uses of the same PIT estimate.)

### Sizing-only vs selection + sizing
Chose **sizing-only**, mirroring qlib (its portfolio layer sizes; the model selects). Arm L changes only *how much* of each selected name to hold, never *which* names — so IC and the selected set are invariant and any delta is purely the weighting's. A weighting scheme that also re-ranks would conflate two effects.

## Outcome (universe-300, 63 folds, 907 OOS days)

| metric | G (equal weight) | L (inverse-vol, shrink 0.5) |
|---|---:|---:|
| slippage_adjusted_sharpe | **1.0886** | 0.9575 |
| max_drawdown | **−0.0421** | −0.0420 |
| total_return | **0.1409** | 0.1160 |
| turnover_avg | **0.0048** | 0.0057 |
| **fold_negative_ic_streak** | **4** | **4** |
| oos_rolling_ic | 0.2561 | 0.2561 |
| ic_60d | 0.0912 | 0.0912 |
| bootstrap_ic_p05 | 0.0028 | 0.0028 |
| top_minus_bottom_decile_ic | 0.0149 | 0.0149 |
| **eligibility.passed** | **True** | **False** |

A–K reproduce their prior numbers bit-identically; the weighting seam did not perturb the default path. Arm L's IC / decile / bootstrap and selected set equal G's exactly — selection is unchanged, so the only deltas are the risk-bearing return metrics.

### Findings

1. **Inverse-vol weighting hurts on this universe — L fails eligibility.** Sharpe falls 1.0886 → 0.9575, below the 1.0 gate, and total return falls 18%. This is the clean negative result the A/B was built to detect: the sizing change is the *only* difference from G, and it is net-negative.

2. **It is redundant with the portfolio-level vol target.** The construction already scales gross exposure to a 15% vol target (ADR-003), so portfolio variance is controlled at the book level. Adding name-level inverse-vol on top does not reduce drawdown (−4.20% vs −4.21% — unchanged) because the risk it would manage is already managed; it only redistributes *within* the book.

3. **The redistribution tilts away from alpha and adds churn.** With drawdown flat, the −18% return is the cost of moving weight off the highest-alpha names toward lower-vol ones whose forward returns are lower, and turnover rises +19% as relative vols drift between rebalances (each shift is a trade, charged at 10 bps). Both effects are monotonic in the tilt, so a lighter tilt (higher `shrinkage`) would *reduce* the damage but not reverse the sign — the mechanism, not the calibration, is the problem. `shrinkage=1.0` is just equal weight (= G).

4. **When inverse-vol *would* help.** The result is specific to a construction that already vol-targets at the book level and selects a small, capped top-N. Inverse-vol earns its keep when there is no portfolio-level vol control, when names span a wide vol range with no per-name cap, or for a market-neutral book where name-level risk parity is the primary risk model. None of those describe G. The seam makes those future experiments one-liners; this arm says equal weight is correct *here*.

## Consequences

* **The weighting seam is the lasting win even though the arm is negative.** Inverse-variance, ADV-capped, or risk-budgeted sizing all land behind one protocol with zero churn to selection, leakage, eligibility, cost, or dial machinery. The negative empirical result for inverse-vol does not diminish the architectural value — it is exactly the kind of question the seam exists to answer cheaply.
* **Arm L is a diagnostic, not a candidate.** It does not displace G; it documents that equal weight is the right sizing for this construction. Like Arm I (GBDT-MSE, ADR-006), it stays in the default arm set as a recorded comparison, not a promotion path.
* Arm L's evidence carries a `weighting` block in `portfolio_diagnostics` (present only for non-default weighting arms, mirroring the `cost_model` and `regime_detector` precedents), so an auditor can read the exact sizing assumption.
* Arms A–K are unchanged and bit-identical; equal weight is now `EqualWeight` but produces the same evidence.

**Related:** ADR-006 (alpha-model seam), ADR-007 (cost-model seam — the same pattern for pricing), ADR-003 (long-only construction + the vol-target exposure scale this weighting composes with), ADR-004 (eligibility thresholds), the qlib model-adoption review (item 3).
