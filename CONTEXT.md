# Quant Platform Context

Use this glossary when naming modules, tests, docs, issues, and operator
messages. The project is a single-operator IBKR cash-account platform, so the
language is deliberately specific.

## Runtime Terms

**Operator**
: The human responsible for approving, monitoring, pausing, and recovering
trading workflows. Prefer this over "user", "admin", or "trader".

**Shadow Run**
: A non-submitting run that records what an engine would have done.

**Paper Session**
: A trading run that exercises the execution path without risking live capital.
IBKR paper and simulated paper are different broker backends under this mode.

**Live Session**
: A broker-capable run that can route approved orders to the live account.
Live mode must fail closed unless contracts, schema, readiness, cash, risk, and
broker gates pass.

**Engine**
: A strategy runtime that produces target proposals or runs a bounded cycle.
Examples include `cross_sectional_equity` and `etf_macro_allocator`.

**V2 Account Orchestrator**
: The shared-account live/paper execution path that merges multiple engine
target proposals, budgets once, writes OMS state, and delegates submission to
the existing broker gateway path.

## Research And Promotion Terms

**Production Candidate**
: Aggregated evidence that a strategy or promoted signal source can move toward
a stricter operating mode.

**Readiness Report**
: Operator-facing status of production gates at a point in time.

**Signal Gate**
: Governed evidence check that determines whether a non-classical signal source
may influence portfolio construction.

**Forecast Evidence**
: Recent prediction-quality evidence for one promoted source and horizon.

**Feature Card**
: Versioned declaration of a feature's owner, thesis, inputs, lag assumptions,
state, expected sign, risk exposures, and failure modes.

**Feature Audit**
: Evidence record that admits, quarantines, or retires a feature for a governed
state.

**Text Model Manifest**
: Immutable record of a text signal model, feature names, weights, prompt
version, and feature-card hashes.

**LLM Live Startup Assertion**
: Fresh token proving live text/LLM settings, manifest, feature audits, and
forecast evidence still match before session creation.

## Portfolio And Execution Terms

**Portfolio Target**
: Desired post-rebalance holdings produced before order planning.

**Order Intent**
: Planned buy or sell instruction produced from a portfolio target before
broker routing.

**Pre-Trade Gate**
: Cash, risk, and execution-policy checks that must pass before submission.

**Kill Switch**
: Operator or automatic halt that prevents new order submissions until reviewed
and cleared.

**Reconciliation**
: Broker-authoritative comparison between external broker state and internal
orders, fills, cash, and positions.

## Relationships

- A Shadow Run, Paper Session, and Live Session should share portfolio, cash,
  risk, regime, and pre-trade policy. The broker adapter is the main difference.
- A Text Model Manifest alone does not permit live text influence. A Production
  Candidate must pass, then the LLM Live Startup Assertion must be fresh.
- A Feature Card can be reviewed without admission. A Feature Audit decides
  whether the feature can enter a governed state.
- A Portfolio Target produces Order Intents; Order Intents are then approved,
  rejected, or submitted.
- V2 live execution is intended to be the only multi-engine live submitter.

## Ambiguities To Avoid

- Use **Live Session** for broker-capable runtime mode. Use "live feature state"
  for feature-admission state.
- Use **Text Model Manifest** for text/LLM artifacts. Use "boosting manifest"
  for XGBoost artifacts.
- Use **Operator API** for the read API. Avoid "admin API" unless discussing a
  future role model.
