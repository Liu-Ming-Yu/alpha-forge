"""Root pytest configuration.

Isolate the suite from the operator's local ``.env``. Since enabling the V2
account orchestrator there (ADR-014) would otherwise leak into any test that
builds default ``PlatformSettings`` (env/.env sources are read even by
``model_validate``), pin the V2 flags off for tests. Env vars take precedence
over ``.env`` in pydantic-settings, and ``setdefault`` still lets an explicit
shell export win. Tests that exercise V2 pass ``v2=V2Settings(enabled=True)``
explicitly, which overrides this.
"""

from __future__ import annotations

import os

os.environ.setdefault("QP__V2__ENABLED", "false")
os.environ.setdefault("QP__V2__ACCOUNT_ORCHESTRATOR_ENABLED", "false")
