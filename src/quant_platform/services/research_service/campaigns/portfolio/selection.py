"""Pluggable name-selection strategies for the long-only target builder.

`raw_long_only_target` historically re-picked the **fresh top-N** by score every
rebalance. That has no membership hysteresis: a held name that slips from rank N
to rank N+1 is dropped entirely (weight → 0) and the new rank-N name is bought
(0 → full weight) — a full round-trip trade driven purely by a one-rank wiggle.
Membership churn like this is a dominant turnover source, and turnover erodes
slippage-adjusted Sharpe at 10 bps a pop.

qlib's `TopkDropoutStrategy` addresses exactly this: hold `topk`, but only turn a
bounded number of names over per period, so a name that slipped slightly keeps
its slot. This module factors name selection into a small protocol — the fourth
application of the seam pattern (`AlphaModel` ADR-006, `TradingCostModel`
ADR-007, `WeightingScheme` ADR-008) — so a buffered-membership strategy can be
swapped in without touching sizing, leakage, eligibility, cost, or dial logic.

* :class:`TopNSelection` is the **behavior-preserving default** — it returns the
  fresh top-N, reproducing the prior selection exactly.
* :class:`BufferedTopKSelection` is the qlib-TopkDropout-inspired buffered
  variant — Arm M. See ADR-009.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    import uuid
    from collections.abc import Mapping, Sequence

    from quant_platform.services.research_service.sampling.samples import SupervisedAlphaSample


@runtime_checkable
class SelectionStrategy(Protocol):
    """Chooses which names to hold; sizing is the weighting scheme's job.

    ``select`` receives the positive-score candidates already sorted by score
    (best first) and the set of currently-held instruments, and returns the
    chosen rows (a subset, at most ``top_n``). Returning the rows — not just ids
    — keeps the (row, score) pair intact for the downstream weighting scheme.
    """

    #: Stable identifier stamped into the evidence audit trail.
    name: str

    def select(
        self,
        ranked: Sequence[tuple[SupervisedAlphaSample, float]],
        *,
        top_n: int,
        current_holdings: frozenset[uuid.UUID],
    ) -> list[tuple[SupervisedAlphaSample, float]]:
        """Return the selected (row, score) pairs, at most ``top_n`` of them."""
        ...

    def metadata(self) -> Mapping[str, object]:
        """Self-describing parameters for the evidence/manifest audit trail."""
        ...


class TopNSelection:
    """The fresh top-N by score every rebalance — the behavior-preserving default.

    Ignores ``current_holdings`` (no hysteresis), so ``raw_long_only_target``
    reproduces the prior membership exactly and arms that don't opt in are
    bit-identical.
    """

    __slots__ = ("name",)

    def __init__(self) -> None:
        self.name = "top-n"

    def select(
        self,
        ranked: Sequence[tuple[SupervisedAlphaSample, float]],
        *,
        top_n: int,
        current_holdings: frozenset[uuid.UUID],
    ) -> list[tuple[SupervisedAlphaSample, float]]:
        return list(ranked[:top_n])

    def metadata(self) -> Mapping[str, object]:
        return {"name": self.name, "type": "top_n"}


class BufferedTopKSelection:
    """Buffered top-k membership (qlib-TopkDropout-inspired).

    An incumbent (currently-held name) that slips just below the top-N cutoff is
    kept rather than churned, as long as it stays within a **buffer band** of
    ``buffer`` extra ranks (i.e. rank < ``top_n + buffer``). Concretely, starting
    from the fresh top-N:

    * incumbents that fell into the band ``[top_n, top_n + buffer)`` are allowed
      back in, each **displacing the weakest *new* entrant** (a non-incumbent
      that just cracked the top-N) — so a held name at rank 31-35 keeps its slot
      instead of being sold for a fresh name at rank 28-30;
    * clearly-better new names (well inside the top-N) always enter — only the
      *marginal* new entrants near the cutoff are bumped;
    * a held name whose score went non-positive is not a candidate at all
      (it's absent from ``ranked``), so it is dropped as before.

    The result always has exactly ``top_n`` names (when that many positive
    candidates exist). The number of incumbents that can be "saved" per rebalance
    is bounded by ``buffer``, which plays the role of qlib's ``n_drop`` in
    reverse — a turnover budget expressed as a rank-tolerance band rather than a
    fixed swap count. ``buffer=0`` recovers :class:`TopNSelection` exactly.

    Selection changes *which* names are held, and therefore the portfolio's
    returns and turnover. It does **not** change the reported IC: the driver
    measures IC over the full scored cross-section each day, not the held book,
    so the IC / decile / streak metrics are invariant to selection — only the
    return-side metrics (Sharpe, total return, turnover) move.
    """

    __slots__ = ("_buffer", "name")

    def __init__(self, *, buffer: int = 5) -> None:
        if buffer < 0:
            raise ValueError("buffer must be >= 0")
        self._buffer = int(buffer)
        self.name = f"buffered-topk-b{buffer}"

    def select(
        self,
        ranked: Sequence[tuple[SupervisedAlphaSample, float]],
        *,
        top_n: int,
        current_holdings: frozenset[uuid.UUID],
    ) -> list[tuple[SupervisedAlphaSample, float]]:
        fresh = list(ranked[:top_n])
        # Cold start, no buffer, or nothing slipping: just the fresh top-N.
        if self._buffer == 0 or not current_holdings:
            return fresh
        band = top_n + self._buffer
        # Incumbents that slipped into the buffer band [top_n, band), best first.
        buffer_incumbents = [
            (row, score)
            for rank, (row, score) in enumerate(ranked)
            if top_n <= rank < band and row.instrument_id in current_holdings
        ]
        if not buffer_incumbents:
            return fresh
        # New entrants inside the fresh top-N (non-incumbents), in rank order so
        # the *weakest* (nearest the cutoff) are at the tail and bumped first.
        new_entrants = [
            (row, score) for row, score in fresh if row.instrument_id not in current_holdings
        ]
        n_replace = min(len(buffer_incumbents), len(new_entrants))
        if n_replace == 0:
            return fresh
        bumped = {row.instrument_id for row, _ in new_entrants[len(new_entrants) - n_replace :]}
        kept = [(row, score) for row, score in fresh if row.instrument_id not in bumped]
        return kept + buffer_incumbents[:n_replace]

    def metadata(self) -> Mapping[str, object]:
        return {"name": self.name, "type": "buffered_topk", "buffer": self._buffer}


__all__ = ["BufferedTopKSelection", "SelectionStrategy", "TopNSelection"]
