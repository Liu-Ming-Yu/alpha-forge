"""Compatibility shim — learned-PCA artifact moved to the inner-layer kernel (ADR-012).

Canonical definitions now live in
``quant_platform.services.research_service.features.kernel.learned.artifact``.
This package's ``__init__`` (family registration) and ``trainer`` (sklearn fit)
re-export through here, and the live learned family imports the kernel directly.
"""

from __future__ import annotations

from quant_platform.services.research_service.features.kernel.learned.artifact import (
    ARTIFACT_SCHEMA_VERSION as ARTIFACT_SCHEMA_VERSION,
)
from quant_platform.services.research_service.features.kernel.learned.artifact import (
    PCAArtifact as PCAArtifact,
)
