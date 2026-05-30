# Portfolio construction + transaction-cost framework (research input, 2026-05-29)

Source: *Quantitative Trading Strategies: A Mathematical Approach* (835pp). Extracted to attack the **IC→Sharpe "sticky" problem** (oos rank-IC ~0.16–0.19 but slippage-adjusted Sharpe stuck ~0.85). Page refs are the book's printed pages.

## The diagnosis: Fundamental Law of Active Management (p. 538, eq. 14.1)

`IR ≈ IC · √BR · TC` — IR ∝ Sharpe; IC = our rank-IC; BR = breadth (independent bets); **TC = transfer coefficient** = corr(ideal active weights, implemented weights).

Our IC is real but the Sharpe leaks on **two** axes:
1. **Low TC.** Equal-weight top-30 + hard cutoff discards conviction. Worse: our cross-sectional **rank**-normalization (the dollar-volume fix) *compresses* scores into a near-uniform [0,1], so even the backtest's score-proportional weighting (`daily_metrics`: `w = score/Σscore`) is ≈ equal-weight across the top-30. **This is the primary leak.**
2. **Low effective breadth.** 330 correlated names ⇒ effective size `m_ef = (Σ C⁻¹_ij)⁻¹` (eq. 12.48, p. 463) ≪ 330 (one dominant market eigenvalue). Fix = neutralize the common factor so residual bets are independent.

## Priority levers (agent's recommended order)

1. **Conviction tilt — the biggest expected Sharpe lift.** Factor-neutral alpha-max (§12.6.1, eq. 12.37, p. 458): `w_i ∝ α_i/d_i²` (alpha per unit idiosyncratic variance), with `Bᵀw=0` (factor-neutral), `1ᵀw=1`. A single market-factor `D` model is enough to start. Raises TC (conviction) + breadth (neutrality) at once. **Crux: our α must carry conviction spread — rank-compressed scores defeat this, so use a z-scored/Gaussian-rank composite (spread-preserving) or the raw weighted-sum as α, divided by idio risk.**
2. **Neutralize dominant factor(s)** (market, maybe sector) — raises effective breadth.
3. **Per-name square-root impact cost** (replaces the flat per-turnover charge): `c_i ≈ spread_i/2 + Y·σ_d,i·√(trade_i/ADV_i)`, Y≈0.5 (§9.10, eq. 9.38, p. 296). Flat charges under-penalize big trades in thin names, over-penalize liquid ones.
4. **No-trade / hysteresis band + partial rebalancing.** Garleanu-Pedersen (eqs. 12.69–71, p. 480): `w_{t+1} = (I−Δ)w_t + Δ·w_aim` — never fully rebalance; aim anticipates alpha decay; trade slower when costs high. No-trade band half-width `Δ* ≈ (3c/2δ)^{1/3}` (eq. 4.53, p. 133). Buy/sell **hysteresis**: buy into top-30, sell only when a name exits ~top-45 — kills boundary churn.
5. **Full QP/SOCP (eq. 12.85, p. 493)** or **SLOPE** (sorted-ℓ1, eq. 12.77, p. 484; spans equal-weight↔min-var with one λ-sequence). μ = our IC-weighted score; Σ = Ledoit-Wolf or single-factor shrinkage.
6. **Sweep rebalance frequency × band width on net Sharpe** — guaranteed interior optimum (eq. 12.86, p. 493).

## Guardrails the book flags
- **μ errors ~10× costlier than Σ errors** (Chopra-Ziemba); naive MVO on 330 point forecasts = "error maximization" — use shrinkage/constraints, never raw MVO.
- **1/N beats 14 optimizers OOS** (DeMiguel-Garlappi-Uppal) — equal-weight is a strong baseline; the win comes from conviction+neutrality+cost-awareness, not from a fancy Σ.
- **Long-only ≈ free covariance shrinkage** (Jagannathan-Ma, p. 483) — lean into the constraint.
- **Inverse-vol/risk-parity is a RISK lever, not an IC→Sharpe lever** (explains our Arm L negative) — combine with an alpha tilt, never alone.
- Equal-weighting the 36 feature weights "performs nearly as well as optimized due to estimation error" (p. 522) — don't over-fit feature weights; the lever is construction, not feature-weight tuning.

## Application plan (this codebase)
- **Experiment 1 (highest EV):** a `ConvictionWeightedConstructor` — top-K selection by composite score, then `w_i ∝ α̃_i/σ_i²` where `α̃` = cross-sectional z-score (spread-preserving) of the composite and `σ_i²` = trailing return variance (single-factor idio proxy), long-only, gross-scaled to 0.22, per-name capped. New arm vs G (equal-weight). Measures the TC lever directly.
- **Experiment 2:** per-name √-impact cost model replacing the flat slippage charge + a no-trade hysteresis band; sweep rebalance freq.
- Applies to **every** arm's construction — including the D/PCA live port (deploy D with conviction weighting, not equal-weight).
