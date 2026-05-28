"""Canonical research-arm category vocabulary.

The category strings are the governance contract that
``factory_models.THRESHOLDS_BY_ARM_CATEGORY`` keys against. They live
here (not in any one script) so:

* Multiple scripts (latest-stack, mining, paper-trade-replay, …) can
  share the same vocabulary instead of each defining its own.
* The eligibility-threshold lookup in
  :mod:`quant_platform.services.research_service.sampling.factory_models`
  types itself directly against the Literal, so an unrecognised key
  fails at type-check time rather than runtime.
* Adding a new category is a single point of edit: extend the Literal
  here, add a corresponding :class:`AlphaEligibilityThresholds`
  instance, and register it in ``THRESHOLDS_BY_ARM_CATEGORY``.

The category vocabulary distinguishes alpha categories by what
governance contract should apply to them, not by feature mix:

* ``research_ranker_baseline`` — signed-rank weights with no risk
  controls. Diagnostic tools measuring whether features rank
  forward returns. Strict eligibility gate; baselines never PASS
  unless they would also pass with risk controls.
* ``portfolio_candidate`` — long-only / hedged constructions with
  per-name caps, sector neutrality, ADV caps, or other risk
  controls that bound drawdown by construction. Eligibility gate
  trades looser streak tolerance for tighter drawdown bound.

See [ADR-004](../../../../../docs/architecture/adr-004-per-category-eligibility-thresholds.md)
for the framing and the asymmetric threshold trade-off.
"""

from __future__ import annotations

from typing import Literal

#: Canonical research-arm category vocabulary. The values are the
#: governance contract; do not introduce a new category without
#: also adding a corresponding :class:`AlphaEligibilityThresholds`
#: instance to :data:`THRESHOLDS_BY_ARM_CATEGORY` in
#: ``factory_models.py``. A category here without a registered
#: threshold set means every arm in that category will raise
#: ``KeyError`` at dispatch time.
ArmCategory = Literal["research_ranker_baseline", "portfolio_candidate"]


__all__ = ["ArmCategory"]
