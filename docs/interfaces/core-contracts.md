# Core Contracts

`src/quant_platform/core/contracts` contains protocol interfaces used to keep
service logic independent from concrete adapters. The domain models live under
`src/quant_platform/core/domain`.

## Invariants

- Contracts should describe behavior, not storage details.
- Services should depend on contracts and domain objects, not concrete
  infrastructure.
- Infrastructure adapters should translate external wire/storage formats into
  domain models.
- Live execution contracts must fail closed on missing data, missing identity,
  unsupported broker capability, kill switch, or cash/risk rejection.

## Data Contracts

| Contract | Purpose |
| --- | --- |
| `MarketDataProvider` | Latest or streaming market data access |
| `HistoricalDataStore` | Historical bar storage and lookup |
| `FeatureRepository` | Feature vector persistence and point-in-time reads |
| `DatasetQuorumRepository` | Vendor quorum evidence for production gates |
| `TextEventStore` | Text/catalyst events and artifact references |

## Research Contracts

| Contract | Purpose |
| --- | --- |
| `BacktestEngine` | Strategy simulation against explicit feature and price data |
| `SignalModel` | Score instruments from feature vectors |
| `ModelRegistry` | Register, promote, retire, diff, and retrieve model artifacts |
| `FeatureAuditRepository` | Store governed feature audit decisions |
| `ArtifactStore` | Write/read research evidence artifacts |

## Portfolio And Risk Contracts

| Contract | Purpose |
| --- | --- |
| `RegimeDetector` | Classify current market regime |
| `PortfolioConstructor` | Convert scores/regime into target weights |
| `RiskPolicy` | Enforce single-name, sector, gross, turnover, and drawdown limits |
| `CashConstraintEngine` | Enforce settled-cash and reservation rules |
| `ExecutionPolicy` | Enforce kill switch and submit throttles |

## Execution Contracts

| Contract | Purpose |
| --- | --- |
| `BrokerCapabilities` | Declare supported order routes and broker features |
| `BrokerSessionGateway` | Connect, disconnect, health, account and position sync |
| `BrokerOrderRoutingGateway` | Place and cancel orders |
| `BrokerGateway` | Combined broker surface for existing execution paths |
| `OrderRepository` | Durable order lifecycle state |
| `PositionRepository` | Durable position snapshots |
| `AuditSink` | Append-only audit events |
| `EventBus` | Domain event publication |
| `Clock` | Runtime/test time abstraction |

## Live Instrument Identity Rule

Live and IBKR paper paths require canonical instrument identity. Contract files
must provide broker identifiers such as IB conId where required. If a broker
position, fill, or open order cannot be mapped back to a known instrument, the
execution path must fail closed and require operator review.

## Message Schema Expectations

Events should include:

- Stable event type.
- Event timestamp.
- Strategy or run identifier when relevant.
- Instrument/order identifiers when relevant.
- Payload fields that can be serialized without leaking secrets.

Important event families:

- Market data ingest.
- Feature vector computation.
- Signal and regime publication.
- Portfolio target creation.
- Order approval/rejection/submission/fill/cancel.
- Kill-switch activation.
- Reconciliation discrepancies.
- V2 proposal and OMS lifecycle.

## Contract Review Checklist

- Does the contract expose only the behavior the caller needs?
- Can the contract be implemented by an in-memory fake and a durable adapter?
- Does the contract avoid framework, database, broker, or SDK leakage?
- Are expected failures explicit enough for fail-closed handling?
- Are domain objects stable enough for tests and docs to depend on?
