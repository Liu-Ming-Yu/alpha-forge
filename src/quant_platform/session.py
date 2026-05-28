"""Public session facade.

Thin re-export of the composition-tier session API: the *composition* half
(``bootstrap/session/public_api.py`` -- ``create_*_session``) and the *runtime*
half (``engines/session/public_api.py`` -- ``run_strategy_cycle`` and friends).
Callers outside the composition tier get one stable, short import path.
"""

from __future__ import annotations

from quant_platform.application.runtime.state import (
    CycleResult as CycleResult,
)
from quant_platform.application.runtime.state import (
    Session as Session,
)
from quant_platform.bootstrap.session.public_api import (
    _DEV_DEFAULT_PREFIXES as _DEV_DEFAULT_PREFIXES,
)
from quant_platform.bootstrap.session.public_api import (
    _assert_live_session_defaults as _assert_live_session_defaults,
)
from quant_platform.bootstrap.session.public_api import (
    _run_sector_mapping_preflight as _run_sector_mapping_preflight,
)
from quant_platform.bootstrap.session.public_api import (
    _session_default_fields as _session_default_fields,
)
from quant_platform.bootstrap.session.public_api import (
    _SessionDrawdownGuard as _SessionDrawdownGuard,
)
from quant_platform.bootstrap.session.public_api import (
    create_ib_paper_session as create_ib_paper_session,
)
from quant_platform.bootstrap.session.public_api import (
    create_live_session as create_live_session,
)
from quant_platform.bootstrap.session.public_api import (
    create_paper_session as create_paper_session,
)
from quant_platform.engines.session.public_api import (
    _compute_market_stats_from_store as _compute_market_stats_from_store,
)
from quant_platform.engines.session.public_api import (
    hydrate_session_state as hydrate_session_state,
)
from quant_platform.engines.session.public_api import (
    model_registry_preflight as model_registry_preflight,
)
from quant_platform.engines.session.public_api import (
    run_strategy_cycle as run_strategy_cycle,
)

__all__ = [
    "CycleResult",
    "Session",
    "_DEV_DEFAULT_PREFIXES",
    "_SessionDrawdownGuard",
    "_assert_live_session_defaults",
    "_compute_market_stats_from_store",
    "_run_sector_mapping_preflight",
    "_session_default_fields",
    "create_ib_paper_session",
    "create_live_session",
    "create_paper_session",
    "hydrate_session_state",
    "model_registry_preflight",
    "run_strategy_cycle",
]
