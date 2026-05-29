# qlib Adoption Map

This note records the local qlib research pass against
`C:\Users\mliu\Desktop\qlib\qlib` and where its model, strategy, and RL patterns
now live in Quant.

## qlib Patterns Reviewed

| qlib area | Local files | Pattern to adopt |
| --- | --- | --- |
| Model | `qlib/model/base.py`, `qlib/model/trainer.py`, `qlib/model/ens/*`, `qlib/model/riskmodel/*` | A fitted model exposes prediction; training is separated from inference; model families are swappable behind a stable contract. |
| Strategy | `qlib/strategy/base.py`, `qlib/contrib/strategy/signal_strategy.py`, `qlib/contrib/strategy/order_generator.py`, `qlib/contrib/strategy/cost_control.py` | Strategy turns signals into trade decisions; portfolio sizing, turnover control, and cost assumptions are explicit components. |
| RL | `qlib/rl/simulator.py`, `qlib/rl/interpreter.py`, `qlib/rl/reward.py`, `qlib/rl/utils/env_wrapper.py`, `qlib/rl/order_execution/*` | A simulator owns state transitions; interpreters bridge simulator state/actions to policy observations/actions; rewards are pluggable and composable. |

## Quant Adoption

Model adoption is already in place through `campaigns/models/`:

- `AlphaModel.fit(...) -> FittedAlphaModel` mirrors qlib's train/predict split.
- `LinearICRanker`, `GradientBoostedRanker`, and `GRUSequenceRanker` are swappable behind that seam.
- ADR-006 and ADR-010 record the GBDT rank-loss and GRU sequence-model outcomes.

Strategy adoption is already in place through the portfolio seams:

- `TradingCostModel` in `campaigns/portfolio/costs.py` adopts qlib's explicit cost-model lever.
- `WeightingScheme` in `campaigns/portfolio/weighting.py` adopts qlib-style sizing separation.
- `SelectionStrategy` in `campaigns/portfolio/selection.py` adopts qlib TopkDropout-style turnover hysteresis.
- ADR-007, ADR-008, and ADR-009 record the cost, weighting, and buffered top-k outcomes.

RL adoption is now in place for research workflows:

- `quant_platform.research.rl` defines the qlib-style contracts without importing gym or tianshou:
  `Simulator`, `StateInterpreter`, `ActionInterpreter`, `Policy`, `Reward`, `RewardCombination`,
  and `EpisodeRunner`.
- `PolicySearch` in `research/features/formulaic/mining/policy_search.py` applies the RL loop to
  formulaic alpha mining. The simulator mutates one expression trajectory, the state interpreter
  exposes compact search observations, the policy chooses mutation actions, and the action
  interpreter validates them before the simulator steps.
- `mine_alphas` continues to own evaluation, evidence, and admission, so policy-guided search gets
  the same provenance and gates as random/evolutionary search.

## Boundary Decision

Quant should not copy qlib's runtime dependencies or backtest stack wholesale. The platform already
has stricter live/paper/shadow parity, broker gates, cash-account rules, and evidence governance.
The adopted shape is therefore the stable abstraction pattern, not qlib's concrete executors.
