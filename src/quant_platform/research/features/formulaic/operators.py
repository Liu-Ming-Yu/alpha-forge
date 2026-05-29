"""Compatibility shim — formulaic operators moved to the inner-layer kernel (ADR-011).

Canonical definitions now live in
``quant_platform.services.research_service.features.kernel.formulaic.operators``.
"""

from __future__ import annotations

from quant_platform.services.research_service.features.kernel.formulaic.operators import (
    OPERATORS as OPERATORS,
)
from quant_platform.services.research_service.features.kernel.formulaic.operators import (
    Axis as Axis,
)
from quant_platform.services.research_service.features.kernel.formulaic.operators import (
    ComputeFn as ComputeFn,
)
from quant_platform.services.research_service.features.kernel.formulaic.operators import (
    Operator as Operator,
)
from quant_platform.services.research_service.features.kernel.formulaic.operators import (
    absolute as absolute,
)
from quant_platform.services.research_service.features.kernel.formulaic.operators import (
    decay_linear as decay_linear,
)
from quant_platform.services.research_service.features.kernel.formulaic.operators import (
    delay as delay,
)
from quant_platform.services.research_service.features.kernel.formulaic.operators import (
    delta as delta,
)
from quant_platform.services.research_service.features.kernel.formulaic.operators import (
    dispatch as dispatch,
)
from quant_platform.services.research_service.features.kernel.formulaic.operators import (
    group_rank as group_rank,
)
from quant_platform.services.research_service.features.kernel.formulaic.operators import (
    rank as rank,
)
from quant_platform.services.research_service.features.kernel.formulaic.operators import (
    sign as sign,
)
from quant_platform.services.research_service.features.kernel.formulaic.operators import (
    signed_power as signed_power,
)
from quant_platform.services.research_service.features.kernel.formulaic.operators import (
    ts_argmax as ts_argmax,
)
from quant_platform.services.research_service.features.kernel.formulaic.operators import (
    ts_corr as ts_corr,
)
from quant_platform.services.research_service.features.kernel.formulaic.operators import (
    ts_rank as ts_rank,
)
from quant_platform.services.research_service.features.kernel.formulaic.operators import (
    ts_zscore as ts_zscore,
)
from quant_platform.services.research_service.features.kernel.formulaic.operators import (
    zscore as zscore,
)

__all__ = [
    "Axis",
    "ComputeFn",
    "OPERATORS",
    "Operator",
    "absolute",
    "decay_linear",
    "delay",
    "delta",
    "dispatch",
    "group_rank",
    "rank",
    "sign",
    "signed_power",
    "ts_argmax",
    "ts_corr",
    "ts_rank",
    "ts_zscore",
    "zscore",
]
