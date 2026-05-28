"""Row mappers and JSON adapters for V2 Postgres repositories."""

from __future__ import annotations

import json
import uuid
from decimal import Decimal
from typing import TYPE_CHECKING, Any

from quant_platform.core.domain.instruments import (
    AssetClass,
    Instrument,
    SecurityMasterQuality,
    SecurityMasterRecord,
    UniverseSnapshot,
)
from quant_platform.core.domain.market_data import DatasetQuorumEvidence
from quant_platform.core.domain.orders import (
    OrderStateEvent,
    OrderStateEventType,
    OrderStatus,
)
from quant_platform.core.domain.portfolio import PortfolioRiskModel
from quant_platform.core.domain.production import (
    OperatorAction,
    OperatorApiKey,
    PreflightCheck,
    ProductionProfile,
    ReadinessSnapshot,
    ReadinessState,
)
from quant_platform.core.domain.research import (
    AlphaReadinessReport,
    FeatureDataset,
    ModelArtifact,
    PromotionState,
)
from quant_platform.infrastructure.postgres.row_coercion import (
    optional_datetime,
    optional_mapping,
    optional_sequence,
    optional_string_mapping,
    require_datetime,
    require_float,
    require_mapping,
)
from quant_platform.infrastructure.v2.portfolio_json import (
    json_to_covariance as _json_to_covariance,
)
from quant_platform.infrastructure.v2.portfolio_json import (
    json_to_factor_exposures as _json_to_factor_exposures,
)
from quant_platform.infrastructure.v2.portfolio_json import (
    json_to_scenarios as _json_to_scenarios,
)

if TYPE_CHECKING:
    from collections.abc import Mapping

    from sqlalchemy.engine import RowMapping


def _json(value: object) -> object:
    if isinstance(value, str):
        return json.loads(value)
    return value


def _row_to_security_record(row: Mapping[str, Any] | RowMapping) -> SecurityMasterRecord:
    instrument = Instrument(
        instrument_id=uuid.UUID(str(row["instrument_id"])),
        symbol=str(row["symbol"]),
        exchange=str(row["exchange"]),
        asset_class=AssetClass(str(row["asset_class"])),
        currency=str(row["currency"]),
        lot_size=int(str(row["lot_size"])),
        active=bool(row["active"]),
        sector=str(row["sector"]) if row["sector"] is not None else None,
    )
    return SecurityMasterRecord(
        record_id=uuid.UUID(str(row["record_id"])),
        instrument=instrument,
        as_of=require_datetime(row, "as_of"),
        available_at=require_datetime(row, "available_at"),
        identifiers=optional_string_mapping(
            _json(row["identifiers_json"]),
            name="identifiers_json",
        ),
        primary_exchange=str(row["primary_exchange"] or ""),
        country=str(row["country"] or "US"),
        source=str(row["source"]),
        quality=SecurityMasterQuality(str(row["quality_status"])),
    )


def _row_to_universe_snapshot(row: Mapping[str, Any] | RowMapping) -> UniverseSnapshot:
    ids = tuple(
        uuid.UUID(str(item))
        for item in optional_sequence(_json(row["instrument_ids_json"]), name="instrument_ids_json")
    )
    return UniverseSnapshot(
        snapshot_id=uuid.UUID(str(row["snapshot_id"])),
        universe_name=str(row["universe_name"]),
        as_of=require_datetime(row, "as_of"),
        available_at=require_datetime(row, "available_at"),
        instrument_ids=ids,
        source=str(row["source"]),
        quality=SecurityMasterQuality(str(row["quality_status"])),
    )


def _row_to_feature_dataset(row: Mapping[str, Any] | RowMapping) -> FeatureDataset:
    source_ids = tuple(
        uuid.UUID(str(item))
        for item in optional_sequence(
            _json(row["source_dataset_ids_json"]),
            name="source_dataset_ids_json",
        )
    )
    return FeatureDataset(
        dataset_id=uuid.UUID(str(row["dataset_id"])),
        feature_set_version=str(row["feature_set_version"]),
        as_of=require_datetime(row, "as_of"),
        available_at=require_datetime(row, "available_at"),
        schema_hash=str(row["schema_hash"]),
        source_dataset_ids=source_ids,
        artifact_uri=str(row["artifact_uri"]),
        quality_status=str(row["quality_status"]),
    )


def _row_to_quorum(row: Mapping[str, Any] | RowMapping) -> DatasetQuorumEvidence:
    return DatasetQuorumEvidence(
        evidence_id=uuid.UUID(str(row["evidence_id"])),
        dataset_kind=str(row["dataset_kind"]),
        as_of=require_datetime(row, "as_of"),
        vendors=tuple(
            str(item) for item in optional_sequence(_json(row["vendors_json"]), name="vendors_json")
        ),
        passed=bool(row["passed"]),
        required_vendor_count=int(str(row["required_vendor_count"])),
        max_disagreement_bps=Decimal(str(row["max_disagreement_bps"])),
        details=dict(optional_mapping(_json(row["details_json"]), name="details_json")),
    )


def _row_to_model_artifact(row: Mapping[str, Any] | RowMapping) -> ModelArtifact:
    return ModelArtifact(
        artifact_id=uuid.UUID(str(row["artifact_id"])),
        model_name=str(row["model_name"]),
        model_version=str(row["model_version"]),
        artifact_uri=str(row["artifact_uri"]),
        artifact_hash=str(row["artifact_hash"]),
        feature_schema_hash=str(row["feature_schema_hash"]),
        training_start=require_datetime(row, "training_start"),
        training_end=require_datetime(row, "training_end"),
        created_at=require_datetime(row, "created_at"),
        promotion_state=PromotionState(str(row["promotion_state"])),
        rollback_artifact_id=uuid.UUID(str(row["rollback_artifact_id"]))
        if row["rollback_artifact_id"] is not None
        else None,
    )


def _row_to_alpha_report(row: Mapping[str, Any] | RowMapping) -> AlphaReadinessReport:
    return AlphaReadinessReport(
        report_id=uuid.UUID(str(row["report_id"])),
        alpha_source=str(row["alpha_source"]),
        as_of=require_datetime(row, "as_of"),
        promotion_state=PromotionState(str(row["promotion_state"])),
        passed=bool(row["passed"]),
        metrics={
            str(k): require_float(v, name=f"metrics_json.{k}")
            for k, v in optional_mapping(_json(row["metrics_json"]), name="metrics_json").items()
        },
        drift={
            str(k): require_float(v, name=f"drift_json.{k}")
            for k, v in optional_mapping(_json(row["drift_json"]), name="drift_json").items()
        },
        rollback_target=str(row["rollback_target"] or ""),
    )


def _row_to_risk_model(row: Mapping[str, Any] | RowMapping) -> PortfolioRiskModel:
    return PortfolioRiskModel(
        model_id=uuid.UUID(str(row["model_id"])),
        as_of=require_datetime(row, "as_of"),
        covariance=_json_to_covariance(row["covariance_json"]),
        factor_exposures=_json_to_factor_exposures(row["factor_exposures_json"]),
        scenarios=_json_to_scenarios(row["scenarios_json"]),
        dataset_id=uuid.UUID(str(row["dataset_id"])) if row["dataset_id"] is not None else None,
        schema_hash=str(row["schema_hash"] or ""),
    )


def _row_to_order_state_event(row: Mapping[str, Any] | RowMapping) -> OrderStateEvent:
    return OrderStateEvent(
        event_id=uuid.UUID(str(row["event_id"])),
        order_id=uuid.UUID(str(row["order_id"])),
        event_type=OrderStateEventType(str(row["event_type"])),
        occurred_at=require_datetime(row, "occurred_at"),
        status=OrderStatus(str(row["status"])) if row["status"] is not None else None,
        broker_order_id=str(row["broker_order_id"]) if row["broker_order_id"] is not None else None,
        idempotency_key=str(row["idempotency_key"]),
        payload=dict(optional_mapping(_json(row["payload_json"]), name="payload_json")),
    )


def _row_to_operator_action(row: Mapping[str, Any] | RowMapping) -> OperatorAction:
    return OperatorAction(
        action_id=uuid.UUID(str(row["id"])),
        occurred_at=require_datetime(row, "occurred_at"),
        action_type=str(row["action_type"]),
        actor=str(row["actor"]),
        reason=str(row["reason"] or ""),
        metadata=dict(optional_mapping(_json(row["metadata"]), name="metadata")),
    )


def _row_to_readiness_snapshot(row: Mapping[str, Any] | RowMapping) -> ReadinessSnapshot:
    checks = tuple(
        PreflightCheck(
            name=str(item["name"]),
            passed=bool(item["passed"]),
            detail=str(item["detail"]),
            severity=str(item.get("severity", "error")),
        )
        for item in (
            require_mapping(raw_item, name="readiness_check")
            for raw_item in optional_sequence(_json(row["checks_json"]), name="checks_json")
        )
    )
    return ReadinessSnapshot(
        snapshot_id=uuid.UUID(str(row["snapshot_id"])),
        profile=ProductionProfile(str(row["profile"])),
        generated_at=require_datetime(row, "generated_at"),
        state=ReadinessState(str(row["state"])),
        passed=bool(row["passed"]),
        checks=checks,
    )


def _row_to_operator_api_key(row: Mapping[str, Any] | RowMapping) -> OperatorApiKey:
    return OperatorApiKey(
        key_id=uuid.UUID(str(row["key_id"])),
        key_hash=str(row["key_hash"]),
        role=str(row["role"]),
        created_at=require_datetime(row, "created_at"),
        created_by=str(row["created_by"]),
        revoked_at=optional_datetime(row, "revoked_at"),
        label=str(row["label"] or ""),
    )
