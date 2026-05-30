"""Compatibility shim — learned-PCA artifact loader moved to the kernel (ADR-012).

Canonical definitions now live in
``quant_platform.services.research_service.features.kernel.learned.loader``.
"""

from __future__ import annotations

from quant_platform.services.research_service.features.kernel.learned.loader import (
    load_pca_artifact as load_pca_artifact,
)
from quant_platform.services.research_service.features.kernel.learned.loader import (
    save_pca_artifact as save_pca_artifact,
)
