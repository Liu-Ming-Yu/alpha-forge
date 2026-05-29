"""Compatibility shim — feature contracts moved to the inner-layer kernel (ADR-011).

The canonical definitions now live in
``quant_platform.services.research_service.features.kernel.contracts``. This
module re-exports them so the existing ``research.features.contracts`` importers
keep working unchanged. The dependency direction is legal: ``research`` is a
composition layer and may import the ``services`` inner layer (the reverse —
``services`` importing ``research`` — is the boundary this move removes).
"""

from __future__ import annotations

from quant_platform.services.research_service.features.kernel.contracts import (
    BaseFamilyConfig as BaseFamilyConfig,
)
from quant_platform.services.research_service.features.kernel.contracts import (
    FamilyManifest as FamilyManifest,
)
from quant_platform.services.research_service.features.kernel.contracts import (
    FeatureDirection as FeatureDirection,
)
from quant_platform.services.research_service.features.kernel.contracts import (
    FeatureFamily as FeatureFamily,
)
from quant_platform.services.research_service.features.kernel.contracts import (
    FeatureFrame as FeatureFrame,
)
from quant_platform.services.research_service.features.kernel.contracts import (
    FeatureSpec as FeatureSpec,
)
from quant_platform.services.research_service.features.kernel.contracts import (
    SignalTimestamp as SignalTimestamp,
)

__all__ = [
    "BaseFamilyConfig",
    "FamilyManifest",
    "FeatureDirection",
    "FeatureFamily",
    "FeatureFrame",
    "FeatureSpec",
    "SignalTimestamp",
]
