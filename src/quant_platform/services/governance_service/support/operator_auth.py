"""Operator API key hashing and RBAC helpers."""

from __future__ import annotations

import hashlib
import secrets
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from datetime import datetime

    from quant_platform.core.contracts import OperatorAuthRepository
    from quant_platform.core.domain.production import OperatorApiKey


def hash_operator_api_key(raw_key: str) -> str:
    """Return a stable SHA-256 hash for an operator API key."""
    return hashlib.sha256(raw_key.encode("utf-8")).hexdigest()


async def authorize_operator_api_key(
    *,
    raw_key: str,
    repository: OperatorAuthRepository,
    min_role: str = "viewer",
    as_of: datetime,
) -> OperatorApiKey | None:
    """Return the active key record if the key exists and satisfies role."""
    key_hash = hash_operator_api_key(raw_key)
    record = await repository.get_api_key_by_hash(key_hash)
    if record is None:
        return None
    if record.revoked_at is not None and record.revoked_at <= as_of:
        return None
    if not _role_allows(record.role, min_role):
        return None
    if not secrets.compare_digest(record.key_hash, key_hash):
        return None
    return record


def _role_allows(actual: str, required: str) -> bool:
    rank = {"viewer": 0, "operator": 1, "admin": 2}
    return rank.get(actual, -1) >= rank.get(required, 0)
