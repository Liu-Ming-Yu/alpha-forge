"""Configuration for the ``formulaic-alpha-v1`` feature set."""

from __future__ import annotations

from dataclasses import dataclass

from quant_platform.services.research_service.features.kernel.contracts import BaseFamilyConfig

#: Feature-set version. The "v1" suffix is reserved for the
#: programmatic-AST starter library landed in this sprint; auto-
#: discovered alphas (Phase 4 of the brief) ship under a "-mined-v1"
#: bump so admitted-by-human-judgement and admitted-by-search land in
#: separate catalog rows.
FEATURE_SET_VERSION: str = "formulaic-alpha-v1"

#: Operator-set version. Bumped when an operator's compute formula
#: changes (e.g. ddof default on ts_zscore moves from 0 to 1) so old
#: and new evidence bundles aren't silently compared.
OPERATOR_SET_VERSION: str = "operator-set-v1"


@dataclass(frozen=True)
class FormulaicConfig(BaseFamilyConfig):
    """Frozen config for the formulaic alpha factory.

    Attributes
    ----------
    version:
        Feature-set version. Defaults to :data:`FEATURE_SET_VERSION`.
    operator_set_version:
        Operator catalog version. Defaults to
        :data:`OPERATOR_SET_VERSION`.
    require_full_window:
        When ``True`` (default), rolling/window operators inside an
        expression require a full lookback; warm-up rows are NaN.
        Mirrors :class:`PriceVolumeConfig.min_periods_policy="full"`.
    """

    version: str = FEATURE_SET_VERSION
    operator_set_version: str = OPERATOR_SET_VERSION
    require_full_window: bool = True

    def __post_init__(self) -> None:
        super().__post_init__()
        if not self.operator_set_version.strip():
            raise ValueError("FormulaicConfig.operator_set_version must be non-empty")


DEFAULT_CONFIG: FormulaicConfig = FormulaicConfig()


__all__ = [
    "DEFAULT_CONFIG",
    "FEATURE_SET_VERSION",
    "OPERATOR_SET_VERSION",
    "FormulaicConfig",
]
