# ADR-010 — GRU sequence-model alpha ranker (Arm N)

**Status:** Accepted (model + Arm N). The pluggable alpha-model seam already shipped (ADR-006); this ADR adds the **GRU sequence ranker** behind it — the sequence learner ADR-006 named as the larger deferred follow-up, and the qlib model the seam was built to eventually carry. **Arm N = production-lead G with the linear IC ranker swapped for a PyTorch GRU** that consumes a 20-day sequence of each name's recent feature vectors instead of a single point-in-time row. Unlike the cost (K), weighting (L), and selection (M) seams — which leave the IC untouched — the GRU is a *ranking-changing* model, so its IC / decile / streak differ from G (like the GBDT arms I/J). **Verdict on universe-300: the most interesting near-miss in the whole stack.** The GRU posts the **highest slippage-adjusted Sharpe (1.1187) and total return (15.5%) of any arm**, with a bootstrap-IC lower bound 3.8× G's — but it **fails eligibility on the streak gate** (`fold_negative_ic_streak` = 5 > 4), with weaker aggregate rank-IC and a thinner decile spread. It is the **first intervention to move the streak at all** (4 → 5) — confirming that a sequence model changes the ranking where the portfolio-layer levers (K/L/M) cannot — but it moved it the *wrong* way. G remains the eligible production lead; N is a high-Sharpe diagnostic with a real follow-up hook.

**Prior vs outcome:** the prior was that the GRU likely wouldn't beat the linear ranker's IC and would be an informative negative — half right. Its *aggregate* rank-IC is indeed lower (0.1715 vs 0.2561), but its realized *Sharpe and return are the highest of any arm*, a genuine surprise. We built it for qlib-zoo completeness, the clean A/B, and the durable proof that the seam carries a heavy torch model — and got a sharper finding than expected (below).

## Context

`run_sample_walk_forward` refits an `AlphaModel` per fold and scores the test cross-section. The linear ranker and the GBDT both treat each `SupervisedAlphaSample` as an independent feature *vector*. A sequence model needs the opposite: a per-instrument time series of feature vectors. The sample is flat and carries no lookback window, so the model must *reconstruct* sequences — and do it without violating the point-in-time contract or the 21-day purge that protects it.

## Decision

Add `GRUSequenceRanker` in `campaigns/models/sequence.py`, mirroring `gbdt.py`'s shape (lazy `_require_torch()`, per-process `_cuda_probe_cache`, `device="auto"`, immutable `__slots__` fitted object, normalized-importance reporting proxy). Three design choices carry the weight:

### 1. Sequence reconstruction = "last `window` observed feature vectors per instrument"
Each sample carries `as_of_index` (its position on the global trading-day calendar). The builder groups rows by instrument, orders by `as_of_index`, and a target's sequence is the trailing `window` observed vectors (`as_of_index <= target`). Gaps/halts make `as_of_index` non-contiguous; the builder collapses them — observations are treated as adjacent. Short histories are **left-padded with zeros + a mask**, so the GRU's last-real-step read-out is unpolluted.

### 2. Train-tail cache (the PIT-safe score-time stitch)
`fit` sees only the (post-purge) train window; `score` sees only the test window — but a test row early in the test window needs trailing context from *before* it. The fitted object caches each instrument's last `window-1` **training** feature vectors plus their `as_of_index`; at score time it prepends that tail to the time-ordered test rows and rolls the window. This uses only past *features* (never labels; the train tail is post-purge), so the point-in-time contract holds. The 21-day purge hole is the gap the "observations-as-adjacent" approximation spans.

### 3. Time-gap channel
To stop the approximation from *hiding* the spacing, each window carries an auxiliary `log1p(Δas_of_index_to_prev_obs)` channel (so net `input_size = n_features + 1`). The recurrence sees that the train-tail's last step is ~21 days before the first test step, or that a halt opened a gap — turning a silent distortion into an explicit input.

### 4. IC loss by default
Arms I/J established that MSE-on-levels ranks poorly (Arm I failed `ic_60d`) while a ranking loss recovers it (Arm J passed). The GRU defaults to `objective="ic"`: a per-date **differentiable Pearson** of predictions vs forward returns, averaged across dates, maximized. `objective="mse"` is offered for the A/B.

`GRUSequenceRanker` requires the new `dl` extra (torch). Per-feature standardization uses train mean/std (PIT-safe). Seeding makes CPU runs deterministic; CUDA cuDNN is approximately deterministic (acceptable — a new arm, never bit-compared).

## Options considered

### GRU vs ALSTM/TFT
Chose **GRU** — the simplest recurrent baseline and qlib's default sequence model. ALSTM (attention read-out over the hidden sequence instead of the last hidden state) is a small sibling that drops into `_build_net`; deferred until the GRU shows the sequence framing earns its keep.

### Sequence-via-protocol vs changing the sample abstraction
Chose to reconstruct sequences *inside the model* (train-tail cache) rather than pre-computing per-sample windows in the sampler or extending `AlphaModel.fit`/`score` to pass full feature history. The model-local approach keeps the protocol and the sample DTO unchanged — every other arm and consumer is untouched — at the cost of the observations-as-adjacent approximation across the purge hole (mitigated by the time-gap channel). A sampler-level windowed dataset is the cleaner long-term home if sequence models become central.

### IC (Pearson) vs soft-rank loss
Pearson-on-scores is the standard differentiable IC surrogate and exactly recovered Arm J's lesson; a differentiable soft-rank (closer to Spearman) is `O(n²)` per date with a temperature knob — deferred as a future option.

### `feature_weights` proxy
A GRU has no linear coefficients. Normalized L1 of the GRU input weights (`weight_ih_l0`) over the named features is the cheapest deterministic proxy that keeps `selected_weights` / cross-fold `feature_stability` well-formed; it describes relative *input* influence (omitting the recurrence, head, and gap channel), not causal importance.

## Outcome (universe-300, 63 folds, 907 OOS days)

| metric | G (linear ranker) | N (GRU, IC loss) |
|---|---:|---:|
| slippage_adjusted_sharpe | 1.0886 | **1.1187** |
| total_return | 0.1409 | **0.1549** |
| max_drawdown | **−0.0421** | −0.0495 |
| turnover_avg | **0.0048** | 0.0077 |
| **fold_negative_ic_streak** | **4** | **5** |
| oos_rolling_ic | **0.2561** | 0.1715 |
| ic_60d | **0.0912** | 0.0593 |
| bootstrap_ic_p05 | 0.0028 | **0.0106** |
| bootstrap_ic_p95 | 0.0183 | **0.0269** |
| top_minus_bottom_decile_ic | **0.0149** | 0.0053 |
| **eligibility.passed** | **True** | **False** (streak 5 > 4) |

A–M reproduce their prior numbers bit-identically; adding the GRU did not perturb any other arm (it is opt-in via `model_factory`). Unlike K/L/M, Arm N's IC-side metrics differ from G — the GRU changes the ranking. GRU fit was ~810s for 63 folds on the RTX 5080 (vs ~430s for the linear arm).

### Findings

1. **The GRU is the highest-Sharpe, highest-return arm — and it fails eligibility anyway.** Sharpe 1.1187 (> G's 1.0886, the best of A–N), total return 15.5%, and a bootstrap-IC lower bound of 0.0106 (3.8× G's 0.0028 — its rolling IC is *more reliably positive*). Yet it trips the `fold_negative_ic_streak ≤ 4` gate at 5. This is the cleanest demonstration yet that the streak gate, not raw risk-adjusted return, is the binding constraint: a model can win on Sharpe, return, *and* IC-stability and still be gated out by one extra consecutive negative-IC fold.

2. **A sequence model CAN move the streak — the portfolio-layer levers can't.** K/L/M left `fold_negative_ic_streak` at 4 by construction (they don't touch the ranking). The GRU moved it — to 5. This refines the meta-finding: the streak is invariant to *portfolio* mechanics but *is* sensitive to the ranking model; the linear ranker's particular ranking happens to sit at a 4-fold streak, and a different (even better-Sharpe) model can land on a worse one. So the streak is a joint property of the *model × data*, not the data alone — but no model tried so far moves it *down*.

3. **High realized return, weaker aggregate rank-IC — a coherent profile.** The GRU's `oos_rolling_ic` (0.17) and decile spread (0.005) are below G's, yet its realized return is higher. The reconciliation: its rolling-IC *distribution* is shifted up and tighter (bootstrap p05 0.0106 vs 0.0028), i.e. reliably-positive but lower-magnitude per-fold IC, paired with a more concentrated, higher-turnover book (0.77% vs 0.48%/day) that the dial throttled more often (19 vs 14 zero-folds). It earns its return on the folds it trades, but its longest negative-IC run is one fold worse and its top-minus-bottom selectivity is weaker.

4. **Not a production lead, but the strongest follow-up hook in the stack.** N fails the current gate, but its Sharpe/return/IC-stability profile is the most attractive of any challenger. The natural next steps are exactly the ones that target the *streak* rather than the model: a streak-aware or regime-conditioned training objective, an ensemble of the GRU with G's linear ranker (blend the reliably-positive GRU IC with G's sharper decile selectivity), or — per ADR-004 — revisiting whether a ≤4 streak gate is the right bar for a candidate that otherwise leads on Sharpe. ALSTM (attention read-out) is a cheap architectural variant to try under the same seam.

## Consequences

* **The seam now carries a heavy torch model end-to-end** — lazy import, GPU auto-detect with CPU fallback, picklable factory across the ProcessPoolExecutor, evidence stamped with a hardware-independent `model_version`. ALSTM/TFT and other sequence learners are now incremental.
* Arm N requires the `dl` extra (torch). When torch is absent the worker reports the arm as errored rather than tearing down the run (same contract as the GBDT arms without `ml`). CI stays CPU-only and skips the torch-gated tests; the torch-free sequence-builder logic is covered regardless.
* The default arm set is now A–N (14 arms). Sequence-builder windowing, the train-tail stitch, and PIT-safe standardization are unit-tested without torch.
* **Post-review hardening (no metric change).** A code review found that `build_score_sequences` prepended the train tail unconditionally, so `fitted.score(train)` — called only to fit the volatility scale — built scrambled, intra-train-lookahead windows for the first `window-1` observed rows per instrument. The fix (`_tail_before`) prepends only tail rows strictly earlier than the earliest scored row, making `score(train)` windows identical to `build_train`. **Re-validation was byte-identical** to the numbers above: `fit_fold_volatility_scale` consumes only the *last* `vol_lookback_days` (63) of the train window, which monthly rebalancing has fully reconstituted from recent scores, so the early-row corruption never reached it. A latent PIT-cleanliness defect with zero numerical impact — fixed and regression-tested so a future score-consumer can't be bitten.
* **Closes the qlib model-adoption arc — with a twist.** Across the alpha-model (linear/GBDT/GRU), cost, weighting, and selection seams, only changes to the *ranking* move the IC/streak gate. The GRU finally *did* beat G on the headline numbers (Sharpe, return, IC-stability) — the first arm to do so — but landed on a worse streak (5) and so fails the gate G passes. The binding constraint remains the streak, and the lever remains a streak-targeted objective / regime conditioning / ensemble / the threshold itself (ADR-004) — not just a richer model. G stays the eligible lead; N is the most promising challenger to revisit once the streak, not the model, is the focus.

**Related:** ADR-006 (the alpha-model seam + GBDT arms I/J — the IC-vs-MSE lesson this inherits), ADR-007/008/009 (the cost/weighting/selection seams — the portfolio-layer levers that leave IC untouched), ADR-003/004 (construction + eligibility), the qlib model-adoption review (sequence-model follow-up).
