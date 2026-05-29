# ADR-007 — Quadratic market-impact cost model (the cost-robustness arm)

**Status:** Accepted (framework + Arm K). The cost-model seam (`campaigns/portfolio/costs.py`) is shipped and is the durable win — it factors rebalance pricing out of the evaluator behind a one-method protocol, exactly as ADR-006 did for the alpha model. **Arm K** = the production lead **G** priced through a per-name quadratic impact model instead of the flat 10 bps/turnover. Because the cost change is pure *accounting* — it never touches the ranking or the weights — Arm K's IC, decile spread, and `fold_negative_ic_streak` are **identical to G by construction**; only the post-cost return and Sharpe move. Arm K is therefore a **cost-robustness test**: does G's eligibility survive a convex impact model? **Verdict on universe-300: yes.** Adding a per-name quadratic impact (10 bps at the single-name cap) on top of the flat 10 bps drops G's Sharpe 1.0886 → 1.0364 (−4.8%) and total return 14.1% → 13.4%, but **K still passes all five gates** (Sharpe 1.0364 ≥ 1.0). The margin over the gate narrows to 0.036, so the cost assumption is now an explicit, stress-testable lever rather than a hidden one.

**Context:** the qlib model-adoption review (item 2) flagged that the backtest's flat `cost = 10 bps × turnover` understates real execution cost, which grows *super*-linearly with trade size (market impact) and is worse for trades concentrated in one name. The memory's pre-paper hardening list independently names the same risk: "Recalibrate slippage — if real slippage is materially worse, G's Sharpe could drop below the 1.0 gate; check before promotion." Arm K turns that worry into a measured number.

## Context

`evaluate_long_only_portfolio` priced each rebalance with a single inlined line: `slippage_cost = turnover × (bps / 1e4)`, where `turnover = Σ_i |Δw_i|` is the portfolio-aggregate sum of per-name weight changes. That linear model is a fine first approximation for spread + commission, but it has two structural blind spots:

1. **No convexity.** Real temporary market impact (Almgren–Chriss) grows faster than linearly in trade size — pushing more size through the book moves the price against you. A flat per-unit cost cannot represent that.
2. **No concentration penalty.** `turnover` is blind to *how* the trade is distributed. Trading 20% of the book in one name and 20% spread across twenty names cost the same under the flat model, though the former is far more expensive to execute.

The per-name trade vector needed to fix both was already computed inside the evaluator (it is the difference between `today_weights` and `last_weights`); only the cost line ignored it.

## Decision

Factor rebalance pricing into a one-method protocol in `campaigns/portfolio/costs.py`:

* `TradingCostModel.cost(trades: Mapping[uuid, float]) -> float` — prices one rebalance from its per-name signed Δweight vector, returning the cost as a fraction of NAV. Plus `name` and `metadata()` for the audit trail.

`evaluate_long_only_portfolio` gains `cost_model: TradingCostModel | None = None`; `None` rebuilds the behavior-preserving `LinearTurnoverCost(slippage_bps_per_turnover)` so every pre-seam caller and arms A–J stay **bit-identical** (the `turnover × bps` arithmetic is exactly what `LinearTurnoverCost` computes). `run_sample_walk_forward` threads `cost_model` through to the long-only path and rejects it for signed-rank arms (which have no per-name trade vector to price).

Two concrete models ship:

* **`LinearTurnoverCost`** — `cost = (bps / 1e4) · Σ|Δw_i|`. The default; reproduces the prior numbers exactly.
* **`QuadraticImpactCost`** — `cost = (linear_bps / 1e4) · Σ|Δw_i| + quad_coef · Σ Δw_i²`. The linear term is the same spread/commission; the quadratic term is the per-name market impact the flat model ignored. Because it sums `Δw_i²` (not `(Σ|Δw_i|)²`), it is genuinely per-name: for fixed turnover, concentrating the trade costs strictly more than spreading it.

Arm **K = G's exact construction priced through `QuadraticImpactCost`** gives a clean A/B against the production lead, isolating the cost assumption.

## Options considered

### Per-name quadratic (`Σ Δw_i²`) vs portfolio-aggregate quadratic (`turnover²`)
Chose **per-name**. `turnover²` at G's scale (~0.5%/day) is ~`2.5e-5` — vanishingly small unless the coefficient is enormous, and it carries no concentration information. `Σ Δw_i²` is the form an Almgren temporary-impact model implies once you sum independent per-name impacts, and it rewards spreading trades — the real execution incentive.

### Accounting-only vs impact-aware optimization
Chose **accounting-only**, matching qlib (its cost model is a backtest deduction; the strategy optimizes separately). Arm K changes only what the rebalance *costs*, not the weights it chooses. This makes K a clean cost-sensitivity probe of G: IC and streak are invariant, so any delta is purely the cost model's. An impact-aware constructor (penalize `Σ Δw²` inside the optimizer) is a larger, separate change — deferred.

### Coefficient anchored to an interpretable point vs a raw number
`quad_coef` is given indirectly: trading a single name at `single_name_cap` (0.05, a full position-cap-sized trade) incurs `impact_bps_at_cap` bps of additional impact, so `quad_coef = (impact_bps_at_cap / 1e4) / cap²`. Default 10 bps at the 0.05 cap. This is interpretable and tunable; a raw coefficient would be opaque.

### Not calibrated to per-name ADV
The weight-space evaluator has no volume/ADV plumbed in, so the quadratic term is **not** venue-calibrated — it is a convex *robustness* model anchored to a documented assumption. A volume-scaled impact model (`impact_i ∝ (notional_i / ADV_i)^β`) needs ADV threaded into the evaluator and is the natural follow-up. The architectural seam supports it as a drop-in sibling.

## Outcome (universe-300, 63 folds, 907 OOS days)

| metric | G (flat 10 bps) | K (10 bps + quad impact) |
|---|---:|---:|
| slippage_adjusted_sharpe | **1.0886** | 1.0364 |
| max_drawdown | **−0.0421** | −0.0426 |
| total_return | 0.1409 | 0.1336 |
| turnover_avg | 0.0048 | 0.0048 |
| **fold_negative_ic_streak** | **4** | **4** |
| oos_rolling_ic | 0.2561 | 0.2561 |
| ic_60d | 0.0912 | 0.0912 |
| bootstrap_ic_p05 | 0.0028 | 0.0028 |
| top_minus_bottom_decile_ic | 0.0149 | 0.0149 |
| **eligibility.passed** | **True** | **True** |

G reran bit-identically against the canonical evidence; the cost seam did not perturb the default path. Every IC / ranking / turnover metric for K equals G's exactly — the cost change is pure accounting, so the only deltas are the cost-bearing return metrics (Sharpe, total_return, and a one-basis-point drawdown wiggle from the slightly different daily-return path).

### Findings

1. **G's eligibility is robust to a convex impact model.** The previously-ignored market impact costs ~73 bps of cumulative return over 907 OOS days and 0.052 of Sharpe, but G clears the gate with room (1.0364 ≥ 1.0). At this calibration the answer to the pre-paper question — "could realistic slippage sink G below the Sharpe gate?" — is *no*.

2. **The margin is now explicit and thin.** K passes by 0.036 of Sharpe. Because the quadratic cost scales linearly in `impact_bps_at_cap`, the drag is roughly proportional to it: the ~0.052 Sharpe hit at 10 bps implies the gate is breached somewhere around ~17 bps of impact-at-cap (a linear extrapolation to verify, not a measured point). The cost assumption is therefore the lever to stress before live promotion — re-run Arm K with a higher `impact_bps_at_cap` to find the true breaking point against the operator's calibrated fills.

3. **Concentration is now priced, even though G barely triggers it.** G's construction already minimizes turnover (0.48%/day) and spreads it across ≤30 capped names, so `Σ Δw_i²` is small and the convex penalty is modest. The value of the per-name form shows up the moment a future arm trades more concentrated books (a higher single-name cap, fewer names, or an event-driven rebalance): the flat model would under-charge those exactly where impact bites hardest.

4. **The streak is (again) invariant — trivially, here.** `fold_negative_ic_streak` stays at 4 because the cost model cannot change the ranking. This is not new evidence about the streak's nature (unlike the dial / regime / GBDT interventions, which changed predictions and *still* didn't move it); it is a sanity check that the seam is accounting-only.

## Consequences

* **The cost-model seam is the lasting win.** Every future cost experiment — ADV-scaled impact, borrow/financing for a future long/short arm, venue-specific fees — now lands behind one protocol with zero churn to the construction logic, return accounting, eligibility gate, or dial.
* Arm K is added to the default arm set. Its evidence carries a `cost_model` block in `portfolio_diagnostics` (present only for non-default cost arms, mirroring the `regime_detector` precedent), so an auditor can read the exact pricing assumption.
* Arms A–J are unchanged and bit-identical; the linear cost is now `LinearTurnoverCost` but produces the same evidence.
* **Pre-paper relevance:** Arm K is the standing answer to "how robust is G's Sharpe to a worse cost model?" — re-runnable with a higher `impact_bps_at_cap` to stress-test before live promotion.

**Related:** ADR-006 (pluggable alpha-model layer — the same seam pattern, for the model rather than the cost), ADR-003 (return-accounting separation + the flat slippage the linear cost preserves), ADR-004 (per-category eligibility thresholds the arm is judged against), the qlib model-adoption review (item 2).
