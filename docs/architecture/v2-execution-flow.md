# V2 Execution Flow

## Status

V2 is the guarded shared-account execution path for multi-engine operation. When
`QP__V2__ENABLED=true` and `QP__V2__ACCOUNT_ORCHESTRATOR_ENABLED=true`, live
single-engine execution delegates to V2 or fails closed so two live submitters
cannot compete for the same account.

Per [ADR-014](adr-014-unified-engine-runtime.md), the `supervise` and
`run-multi-engine` paths now drive one shared `AccountOrchestratorLoopRunner`:
single-engine is the N=1 case of the multi-engine orchestrator, so both write the
same governance/read-model rows the operator console reads.

## Intended Flow

```text
engine feature/state checks
  -> EngineTargetProposal
  -> budget merge
  -> account-level optimizer
  -> OMS state
  -> EMS tactic routing
  -> existing broker gateway submission path
  -> lifecycle/reconciliation evidence
```

## Main Modules

| Module | Purpose |
| --- | --- |
| `engines/engine_runner` | Stable engine runner facade and bounded run modes |
| `engines/proposals` | Target proposal construction and proposal-only cycles |
| `engines/multi_engine` | Multi-engine budget/proposal merge helpers |
| `engines/account` | Account-level orchestrator, mapping, and order lifecycle |
| `infrastructure/v2` | Durable V2 state repositories |
| `bootstrap/engine/multi.py` | V2 runner composition and CLI bridge |
| `bootstrap/session` | Session wiring and V2 attachment |

## Activation

Set:

```bash
QP__V2__ENABLED=true
QP__V2__ACCOUNT_ORCHESTRATOR_ENABLED=true
```

Run paper multi-engine:

```bash
python -m quant_platform run-multi-engine ^
  --engines cross_sectional_equity,etf_macro_allocator ^
  --budgets-file ./budgets.json ^
  --mode paper ^
  --contracts-file infra/config/paper_contracts.json ^
  --cycles 1
```

Single-engine live with V2 enabled auto-delegates through the V2 account path
with a synthesized 100 percent capital-weight budget.

## Guards

- Live mode requires a contracts file.
- V2 live refuses in-memory repository bundles unless an explicit development
  escape hatch is configured.
- The single-engine live path blocks when V2 account orchestration is enabled.
- Dataset quorum, readiness, production-candidate, and paper-soak evidence are
  consumed by promotion gates.
- Existing cash, risk, execution-policy, kill-switch, broker, and reconciliation
  gates still apply after V2 proposal merging.

## Operator Evidence

Review these before promotion:

- `readiness` output.
- `production-candidate` output.
- `paper-soak` artifact.
- Dataset quorum evidence.
- Model/signal gate evidence.
- Simulator calibration artifact.
- Broker reconciliation and execution-quality evidence.
