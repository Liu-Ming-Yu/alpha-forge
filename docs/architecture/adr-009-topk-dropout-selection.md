# ADR-009 — Buffered top-k selection (the TopkDropout turnover buffer)

**Status:** Accepted (framework + Arm M). The selection seam (`campaigns/portfolio/selection.py`) is shipped and is the durable win — it factors name selection out of the target builder behind a one-method protocol, the fourth application of the pattern (`AlphaModel` ADR-006, `TradingCostModel` ADR-007, `WeightingScheme` ADR-008). **Arm M** = the production lead **G** with the fresh-top-N selection replaced by a qlib-`TopkDropoutStrategy`-inspired buffered membership: a currently-held name keeps its slot until it slips past rank `top_n + buffer`, so a name wiggling around the cutoff isn't churned every rebalance. Selection changes *which* names are held — and therefore returns and turnover — but **not** the reported IC (the driver measures IC over the full scored cross-section each day, not the held book), so the IC / decile / streak gate metrics are identical to G by construction. **Verdict on universe-300: the buffer works but the trade is mildly unfavorable here.** It cuts turnover 10% (0.48% → 0.43%/day) and improves drawdown 20 bps (−4.21% → −4.01%), but the turnover saving (~4.5 bps over 907 days) is dwarfed by the alpha cost of holding rank-31–35 names (~−75 bps total return), so Sharpe slips 1.0886 → 1.0360 (−4.8%). **M still passes all five gates** (Sharpe 1.0360 ≥ 1.0). The cause is specific to G: its existing turnover machinery (no-trade band + position cap + streak dial) had already captured most of the available turnover savings, leaving the buffer's alpha cost dominant. **G remains the production lead.**

**Context:** the qlib model-adoption review (item 5) flagged that re-selecting the fresh top-N every rebalance churns membership on noise — a held name that slips one rank below the cutoff is fully sold and the new rank-N name fully bought, a round-trip trade driven by a one-rank wiggle. qlib's `TopkDropoutStrategy` bounds per-period membership turnover so slipping names keep their slot; this arm brings that hysteresis into the platform.

## Context

`raw_long_only_target` selected the fresh top-N positive-score names every rebalance. The downstream no-trade band (0.005), per-name position-change cap (0.05), and turnover cap (0.20) smooth the *weight* transitions, but none of them prevents *membership* churn: when a held name drops from rank N to N+1 its target weight goes to zero and a new name's target goes from zero to full. Membership flips are the coarsest, most expensive trades, and at 10 bps/turnover they erode slippage-adjusted Sharpe.

The information needed to add hysteresis — the current holdings — is already threaded through the evaluator (`last_weights`); the selection step just didn't use it.

## Decision

Factor name selection into a one-method protocol in `campaigns/portfolio/selection.py`:

* `SelectionStrategy.select(ranked, *, top_n, current_holdings) -> list[(row, score)]` — chooses the held set from the positive-score candidates (already score-sorted), given the current book. Plus `name` and `metadata()`.

`raw_long_only_target` gains `selection` and `current_holdings`; `evaluate_long_only_portfolio` passes `frozenset(last_weights)` as the incumbent set each rebalance and threads `selection` through. `run_sample_walk_forward` gains `selection`; `None` uses `TopNSelection`, so arms A–L stay **bit-identical**. Rejected for signed-rank arms (no held book).

Two concrete strategies ship:

* **`TopNSelection`** — the fresh top-N, ignoring holdings. The behavior-preserving default.
* **`BufferedTopKSelection(buffer)`** — starting from the fresh top-N, incumbents that slipped into the band `[top_n, top_n + buffer)` are allowed back in, each displacing the *weakest new entrant* (a non-incumbent that just cracked the top-N). Clearly-better new names always enter; only the marginal newcomers near the cutoff are bumped; a held name gone non-positive is not a candidate. Result is always exactly `top_n`. `buffer=0` recovers `TopNSelection`.

Arm **M = G's exact construction with `BufferedTopKSelection(buffer=5)`** isolates the selection change against the production lead.

## Options considered

### Rank-tolerance band vs fixed `n_drop` swap count
qlib's `TopkDropoutStrategy` parameterizes by `n_drop` (a fixed number of names swapped per period). We implement the equivalent *rank-tolerance band* (`buffer`): an incumbent survives until it falls past `top_n + buffer`. The band is simpler to reason about and deterministic, bounds per-period membership churn the same way, and `buffer=0` gives the clean `≡ TopNSelection` equivalence. The two are duals — a band of width `buffer` caps the number of saved incumbents at `buffer` — and the seam admits a strict `n_drop` sibling later if wanted.

### Bump the weakest new entrant, never an incumbent or a strong newcomer
When an in-band incumbent reclaims a slot it displaces the *lowest-ranked non-incumbent* in the fresh top-N. This is the minimal-regret choice: it never sells an incumbent that's still in the top-N, never bumps a strong newcomer (only the marginal one nearest the cutoff), and never swaps a new entrant for another new entrant. The hysteresis is applied exactly where it's cheap.

### Why IC is invariant (and the streak with it)
The driver computes daily IC as the Spearman correlation of *every* scored candidate's score against its forward return — the full cross-section, not the held subset. Selection changes which names carry weight but not the cross-section IC is measured on, so `oos_rolling_ic`, `ic_60d`, the decile spread, the bootstrap CI, and `fold_negative_ic_streak` are identical to G. Like the cost (K) and weighting (L) seams, selection moves only the return-side metrics. (This also means M, like K and L, cannot move the binding streak gate — only a change to the *ranking* can, which selection is not.)

## Outcome (universe-300, 63 folds, 907 OOS days)

| metric | G (fresh top-N) | M (buffered top-k, buffer 5) |
|---|---:|---:|
| slippage_adjusted_sharpe | **1.0886** | 1.0360 |
| max_drawdown | −0.0421 | **−0.0401** |
| total_return | **0.1409** | 0.1334 |
| turnover_avg | 0.0048 | **0.0043** |
| **fold_negative_ic_streak** | **4** | **4** |
| oos_rolling_ic | 0.2561 | 0.2561 |
| ic_60d | 0.0912 | 0.0912 |
| bootstrap_ic_p05 | 0.0028 | 0.0028 |
| top_minus_bottom_decile_ic | 0.0149 | 0.0149 |
| **eligibility.passed** | **True** | **True** |

A–L reproduce their prior numbers bit-identically; the selection seam did not perturb the default path. Arm M's IC / decile / bootstrap / streak equal G's exactly — selection doesn't enter the IC metric — so the only deltas are turnover, drawdown, and the return-side metrics.

### Findings

1. **The buffer does what it's designed to: less churn, smoother path.** Membership turnover falls 10% (0.48% → 0.43%/day) and drawdown improves 20 bps (−4.21% → −4.01%). Holding a slipping incumbent instead of round-tripping it for a fresh name genuinely reduces trading and the path roughness it causes.

2. **But the trade is mildly unfavorable on G — Sharpe slips, though M stays eligible.** Total return falls ~75 bps and Sharpe 1.0886 → 1.0360. The arithmetic is the whole story: the turnover saving is ~4.5 bps (0.0005/day × 10 bps × 907 days), while holding rank-31–35 names instead of fresh rank-28–30 names costs far more alpha (~75 bps). The benefit is real but an order of magnitude too small to pay for the alpha given up.

3. **The cause is G's existing turnover machinery, not the buffer.** G already runs a no-trade band, a per-name position-change cap, and the streak dial, which had pushed turnover down to 0.48%/day — there was little left for the buffer to save. On a construction *without* those controls (or a higher-turnover, higher-rebalance-frequency strategy), the same buffer would save materially more and the trade could flip positive. `buffer` is the lever: a smaller band shrinks both the saving and the alpha cost; `buffer=0` is exactly G.

4. **Like K and L, M cannot move the streak — and doesn't.** Selection changes the held book but not the cross-section IC is measured on, so `fold_negative_ic_streak` stays at 4. This closes the loop on the qlib plan: of the four levers, only the ones that change the *ranking* (the no-PCA model choice, the GBDT arms) ever moved the IC/streak; the three portfolio-layer levers (cost, weighting, selection) move only the return side.

## Consequences

* **The selection seam is the lasting win.** Strict `n_drop` TopkDropout, sector-balanced selection, or liquidity-filtered membership all land behind one protocol with zero churn to leakage, eligibility, cost, weighting, or dial machinery.
* **Arm M is a diagnostic, not a candidate.** It stays eligible and improves drawdown + turnover, but its lower Sharpe means it does not displace G. Like Arm I (GBDT-MSE) and Arm L (inverse-vol), it stays in the default arm set as a recorded comparison; the finding is "G's existing turnover controls already captured this, so the buffer's alpha cost dominates here."
* Arm M's evidence carries a `selection` block in `portfolio_diagnostics` (present only for non-default selection arms, mirroring the `cost_model` / `weighting` / `regime_detector` precedents).
* Arms A–L are unchanged and bit-identical; the fresh top-N is now `TopNSelection` but produces the same evidence.
* **Completes the qlib model-adoption plan.** With the alpha-model (ADR-006), cost-model (ADR-007), weighting (ADR-008), and selection (ADR-009) seams all shipped, the four qlib-inspired levers are each a one-line, evidence-gated swap. The recurring lesson across K/L/M: none touch the *ranking*, so none move the IC/streak gate — that remains a property of the alpha and the data, addressable only by a better model, a different universe/label, or a re-examined streak threshold (ADR-004).

**Related:** ADR-006 / ADR-007 / ADR-008 (the alpha-model, cost-model, and weighting seams — same pattern), ADR-003 (long-only construction + the turnover machinery this composes with), ADR-004 (eligibility thresholds + the streak-gate discussion), the qlib model-adoption review (item 5).
