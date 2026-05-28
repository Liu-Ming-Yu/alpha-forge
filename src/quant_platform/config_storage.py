"""Storage and infrastructure backend settings."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator


class StorageSettings(BaseModel):
    """Database and object-store connection strings.

    Intentionally optional; the in-memory adapters are used when these
    are left empty.
    """

    postgres_dsn: str = ""
    redis_url: str = ""
    object_store_root: str = "./data/parquet"
    distributed_lock_ttl_seconds: int = 120
    distributed_lock_acquire_timeout_seconds: float = 30.0
    distributed_lock_renew_interval_seconds: float = 40.0
    event_bus_backend: Literal["in_memory", "redis_streams"] = "in_memory"
    redis_stream_prefix: str = "qp:events"
    redis_stream_maxlen: int = 10000
    redis_stream_block_ms: int = 1000
    redis_stream_use_consumer_groups: bool = True
    redis_stream_group_prefix: str = "qp:cg"
    redis_stream_publish_dedupe_enabled: bool = True
    redis_stream_dedupe_ttl_seconds: int = 604800
    redis_stream_retention_ms: int = Field(
        default=7 * 24 * 3600 * 1000,  # 7 days
        description=(
            "Retention window for automated XTRIM MINID sweeps (Phase "
            "4.2).  The sweeper under DataMaintenanceSupervisor "
            "trims entries older than now - retention_ms from every "
            "configured Redis Stream.  Set to 0 to disable.  Works in "
            "addition to the per-publish ``redis_stream_maxlen`` cap."
        ),
    )
    redis_stream_sweeper_interval_seconds: float = Field(
        default=900.0,
        description=(
            "Cadence of the automated XTRIM sweeper (Phase 4.2).  "
            "Independent from feature-job cadence so a loaded "
            "maintenance loop cannot starve retention.  Set to 0 to "
            "disable."
        ),
    )
    redis_stream_dead_letter_after_retries: int = Field(
        default=5,
        description=(
            "Retries before an entry is moved to ``<stream>.dlq`` by "
            "the subscribe loop (Phase 4.3).  0 disables the DLQ and "
            "uses redelivery-only stream handling."
        ),
    )
    feature_retention_days: int = Field(
        default=0,
        description=(
            "Default --keep-days for the ``quant-platform features "
            "retention`` CLI (Phase 4.4).  0 means retention is opt-in "
            "per invocation."
        ),
    )
    postgres_pool_min: int = Field(
        default=2,
        description=(
            "Minimum number of connections in the psycopg async connection pool. "
            "Configured via QP__STORAGE__POSTGRES_POOL_MIN."
        ),
    )
    postgres_pool_max: int = Field(
        default=10,
        description=(
            "Maximum number of connections in the psycopg async connection pool. "
            "Configured via QP__STORAGE__POSTGRES_POOL_MAX."
        ),
    )
    postgres_statement_timeout_ms: int = Field(
        default=30_000,
        description=(
            "PostgreSQL statement_timeout in milliseconds applied at connection startup. "
            "0 disables the timeout. Long-running analytics queries may override per-session. "
            "Configured via QP__STORAGE__POSTGRES_STATEMENT_TIMEOUT_MS."
        ),
    )

    @field_validator("postgres_pool_max")
    @classmethod
    def pool_max_gte_min(cls, v: int, info: object) -> int:
        try:
            min_val = getattr(info, "data", {}).get("postgres_pool_min", 2)
        except Exception:
            min_val = 2
        if v < min_val:
            raise ValueError(f"postgres_pool_max ({v}) must be >= postgres_pool_min ({min_val})")
        return v
