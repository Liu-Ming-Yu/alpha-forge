"""Downside-robust IC-weighted ranker.

Like :class:`LinearICRanker`, but the per-feature weights are fit from the
**downside** of each feature's *daily*-IC distribution rather than its mean IC.

Motivation (ADR-011 / ADR-004 follow-up). The pv+formulaic feature set is
momentum-heavy, and medium-term momentum (``mom_3_1``/``mom_6_1``/``ret_63d``/
``ret_126d``) *inverts* during momentum-crash regimes (2023 regional-bank
crisis, 2024 summer rotation, 2025 tariff selloff): its mean IC is high but its
crash IC is sharply negative. Mean-IC weighting therefore over-indexes on
crash-fragile factors — the corrected linear lead G carries ~25% of its weight
on them and its IC flips negative through every crash episode.

A per-feature regime decomposition shows the volatility/range factors
(``high_low_range_1d``/``high_low_range_20d``), short-horizon reversal
(``ret_21d``, ``reversal_*``) and the long horizon (``ret_252d``) stay positive
*through* the crashes. Weighting by a downside-emphasised IC — the mean plus a
heavy weight on the worst-tercile days — down-weights the factors that flip in
bad regimes and up-weights the crash-robust ones, improving robustness with no
new data. In-sample this lifts the combined-score IC-IR ~0.09 → ~0.21 and flips
the crash-window IC positive; the walk-forward arm tests whether it holds OOS.

Scoring is unchanged: ``fit`` returns a :class:`_FittedLinearRanker`, so the
weights drive the same rank-normalized weighted sum as the linear ranker (the
dollar-volume scale fix and live/backtest parity carry over). Only the *fit*
differs. The daily IC is rank-based, so it is invariant to whether features are
raw or rank-normalized at fit time.

**Empirical verdict on the universe-300 pv+formulaic data (Arm O, 2026-05-29):
the OOS gain did NOT materialise.** The walk-forward (fit per training fold,
applied to unseen test folds) cut the negative-IC streak (G 7 → O 4) but gutted
the alpha (oos_ic 0.159→0.041, ic_60d 0.063→−0.029, bootstrap_p05 +0.015→−0.015).
The crash-robust factors are defensive but weak, so reweighting toward them buys
streak stability at the cost of the alpha itself — the in-sample IC-IR lift was
overfit to the specific historical episodes. The model is a correct, reusable
tool; this is a property of *this* momentum-heavy feature set (the alpha is
intrinsically momentum), not a defect. Retained as a documented research arm.
"""

from __future__ import annotations

from collections import defaultdict
from typing import TYPE_CHECKING

from quant_platform.services.research_service.campaigns.models.linear import _FittedLinearRanker
from quant_platform.services.research_service.reports.statistics import spearman_ic as _spearman

if TYPE_CHECKING:
    from collections.abc import Sequence

    from quant_platform.services.research_service.sampling.samples import SupervisedAlphaSample


class RobustICRanker:
    """IC-weighted ranker whose weights reward crash-robustness.

    ``downside_weight`` scales the penalty/credit from the worst-tercile days:
    ``robust_ic = mean(daily_ic) + downside_weight * mean(worst-q daily_ic)``.
    A factor whose IC craters on bad days gets a negative ``robust_ic`` and is
    dropped (weight 0); a factor that holds gets a positive one. ``downside_q``
    is the lower-tail fraction averaged for the downside term. ``min_dates`` is
    the minimum number of valid daily ICs a feature needs before it can earn a
    weight (guards a too-short training window).
    """

    def __init__(
        self,
        *,
        downside_weight: float = 2.0,
        downside_q: float = 0.33,
        min_dates: int = 8,
        min_cross_section: int = 30,
    ) -> None:
        if downside_weight < 0:
            raise ValueError("downside_weight must be >= 0")
        if not 0.0 < downside_q <= 1.0:
            raise ValueError("downside_q must be in (0, 1]")
        self._downside_weight = downside_weight
        self._downside_q = downside_q
        self._min_dates = min_dates
        self._min_cross_section = min_cross_section
        self.name = "robust-ic-downside"

    def fit(
        self,
        train: Sequence[SupervisedAlphaSample],
        feature_names: Sequence[str] | None = None,
    ) -> _FittedLinearRanker:
        if feature_names is not None:
            names = sorted({n for n in feature_names})
        else:
            names = sorted({n for row in train for n in row.features})

        by_day: dict[object, list[SupervisedAlphaSample]] = defaultdict(list)
        for row in train:
            by_day[row.as_of].append(row)

        daily_ic: dict[str, list[float]] = {name: [] for name in names}
        for rows in by_day.values():
            if len(rows) < self._min_cross_section:
                continue
            labels = [row.forward_return for row in rows]
            for name in names:
                values = [float(row.features.get(name, 0.0)) for row in rows]
                ic = _spearman(values, labels)
                if ic == ic:  # filter NaN (constant cross-section)
                    daily_ic[name].append(ic)

        weights: dict[str, float] = {}
        for name in names:
            ics = sorted(daily_ic[name])
            if len(ics) < self._min_dates:
                weights[name] = 0.0
                continue
            k = max(1, int(len(ics) * self._downside_q))
            downside = sum(ics[:k]) / k  # mean of the worst-tercile daily ICs
            mean_ic = sum(ics) / len(ics)
            weights[name] = max(0.0, mean_ic + self._downside_weight * downside)

        total = sum(weights.values())
        if total <= 0:
            equal = 1.0 / max(1, len(names))
            return _FittedLinearRanker({name: equal for name in names})
        return _FittedLinearRanker(
            {name: weight / total for name, weight in weights.items() if weight > 0.0}
        )
