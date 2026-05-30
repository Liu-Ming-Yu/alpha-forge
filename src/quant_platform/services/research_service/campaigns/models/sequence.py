"""GRU sequence-model alpha ranker (PyTorch), GPU-capable.

The qlib model zoo's sequence learners (GRU/ALSTM/TFT) were the larger deferred
follow-up named in ADR-006 — the one model class the ``AlphaModel`` seam was
built to eventually carry. This module delivers the first of them: a GRU ranker
that consumes a short *sequence* of each instrument's recent feature vectors
rather than a single point-in-time row, behind the same protocol as the linear
and GBDT rankers, with zero changes to the leakage / eligibility / portfolio /
dial machinery. See ADR-010.

**Sequence reconstruction (the crux).** A :class:`SupervisedAlphaSample` is a
flat PIT feature *vector*, not a sequence. We reconstruct per-instrument
sequences from the ``as_of_index`` (global trading-day calendar position) on the
samples: a target row's sequence is the **last ``window`` observed feature
vectors for that instrument** with ``as_of_index <= target``. Because
``fit``/``score`` are called per fold, the fitted object caches each
instrument's last ``window-1`` *training* vectors (the "train tail") and stitches
them onto the test rows at score time, so a test row early in the test window
still gets its trailing context — drawn only from past *features* (never
labels), so the point-in-time contract holds. The 21-day purge gap between train
and test is collapsed ("observations-as-adjacent"); a per-step ``log1p`` time-gap
channel feeds the recurrence the observation spacing so the approximation is
mitigated rather than hidden.

**Loss.** Arms I/J showed MSE-on-levels gives poor rank quality while a ranking
loss recovers it; the GRU therefore defaults to ``objective="ic"`` — a
per-date differentiable Pearson of predictions vs forward returns (maximize
cross-sectional agreement). ``objective="mse"`` is offered for the A/B.

**GPU.** ``device="auto"`` probes CUDA once per process and falls back to CPU.
torch is lazy-imported so importing this package does not require the ``dl``
extra; the sequence-construction core is torch-free and independently tested.
"""

from __future__ import annotations

import contextlib
import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import numpy as np

if TYPE_CHECKING:
    import uuid
    from collections.abc import Mapping, Sequence
    from datetime import datetime
    from types import ModuleType

    from torch import Tensor
    from torch.nn import Module as TorchModule

    from quant_platform.services.research_service.sampling.samples import SupervisedAlphaSample

Device = Literal["auto", "cuda", "cpu"]
Objective = Literal["ic", "mse"]

#: Public objective -> stable, hardware-independent model_version name.
_OBJECTIVE_NAME: dict[Objective, str] = {
    "ic": "gru-ranker-ic-v1",
    "mse": "gru-ranker-mse-v1",
}

_DEFAULT_WINDOW = 20
_DEFAULT_HIDDEN = 64
_DEFAULT_LAYERS = 1
_DEFAULT_DROPOUT = 0.0
_DEFAULT_EPOCHS = 40
_DEFAULT_LR = 1e-3
_DEFAULT_SEED = 42
_DEFAULT_GRAD_CLIP = 1.0
_EPS = 1e-8

#: Per-process CUDA capability cache. ``None`` = not probed yet. Module scope so
#: each ProcessPoolExecutor worker probes at most once; never shared across
#: processes (spawn re-imports the module).
_cuda_probe_cache: bool | None = None


def _require_torch() -> ModuleType:
    try:
        import torch  # noqa: PLC0415 — lazy so the package imports without the dl extra
    except ImportError as exc:  # pragma: no cover - exercised only without torch
        raise ImportError(
            "GRUSequenceRanker requires torch. Install the 'dl' extra: "
            "pip install 'quant-platform[dl]' (GPU needs the CUDA index, see pyproject)."
        ) from exc
    # In CI the ``dl`` extra is NOT installed, so ``import torch`` resolves to
    # ``Any`` and mypy flags ``no-any-return`` against the ModuleType annotation;
    # the ignore suppresses that. Locally (torch present) ``torch`` is a real
    # module so the ignore is unused — harmless under ``warn_unused_ignores =
    # false``. (gbdt's _require_xgboost needs none of this because CI installs the
    # ml extra; the dl extra is too heavy to add to CI just for a type check.)
    return torch  # type: ignore[no-any-return]


def _cuda_available() -> bool:
    """Return True iff torch reports a usable CUDA device (cached per process)."""
    global _cuda_probe_cache
    if _cuda_probe_cache is not None:
        return _cuda_probe_cache
    try:
        torch = _require_torch()
        _cuda_probe_cache = bool(torch.cuda.is_available())
    except Exception:  # noqa: BLE001 - any import/driver failure => fall back to CPU
        _cuda_probe_cache = False
    return _cuda_probe_cache


def _select_feature_names(
    samples: Sequence[SupervisedAlphaSample],
    feature_names: Sequence[str] | None,
) -> list[str]:
    """Mirror ``gbdt._select_feature_names`` so a GRU arm sees the exact same
    feature set (and order) the linear ranker would on the same fold."""
    available = {name for row in samples for name in row.features}
    if feature_names is None:
        return sorted(available)
    return sorted({name for name in feature_names if name in available})


def _matrix(samples: Sequence[SupervisedAlphaSample], names: Sequence[str]) -> np.ndarray:
    """Dense float32 design matrix; absent feature -> 0.0 (legacy convention)."""
    return np.array(
        [[float(row.features.get(name, 0.0)) for name in names] for row in samples],
        dtype="float32",
    )


def _as_of_index(row: SupervisedAlphaSample) -> int:
    """Global-calendar position used to order an instrument's observations.

    The latest-stack sample builder always sets ``as_of_index``; the ``-1``
    fallback only matters for ad-hoc callers that omit it, in which case ordering
    degrades to ``as_of`` alone and the time-gap channel reads as adjacent.
    """
    return int(row.as_of_index) if row.as_of_index is not None else -1


# --------------------------------------------------------------------------- #
# Torch-free sequence construction (independently unit-tested, runs in CI).
# --------------------------------------------------------------------------- #


class _Standardizer:
    """Per-feature z-score from TRAIN statistics; frozen at fit, applied at score.

    PIT-safe: test rows are standardized with train mean/std, never their own.
    A constant feature gets std=1 so it centers to 0 without a divide-by-zero.
    """

    __slots__ = ("_mean", "_std")

    def __init__(self, mean: np.ndarray, std: np.ndarray) -> None:
        self._mean = mean
        self._std = std

    @classmethod
    def fit(cls, matrix: np.ndarray) -> _Standardizer:
        mean = matrix.mean(axis=0)
        std = matrix.std(axis=0)  # population std (ddof=0): deterministic; net absorbs the scale
        std = np.where(std < _EPS, 1.0, std)
        return cls(mean.astype("float32"), std.astype("float32"))

    def transform(self, matrix: np.ndarray) -> np.ndarray:
        return np.asarray((matrix - self._mean) / self._std, dtype="float32")


@dataclass(frozen=True)
class _SequenceBatch:
    """Windowed sequences + mask + alignment metadata for one fit/score call."""

    sequences: np.ndarray  # (n_targets, window, n_channels) float32, left-padded
    mask: np.ndarray  # (n_targets, window) float32 — 1 = real step, 0 = pad
    as_of: list[datetime]  # per target (for per-date IC batching)
    labels: np.ndarray | None  # (n_targets,) float32 forward_return; None at score time
    order: list[int]  # index into the ORIGINAL samples list (scatter back)


def _augment_with_time_gap(
    z_feat: np.ndarray, idx: list[int], *, include_time_gap: bool
) -> np.ndarray:
    """Append a per-step ``log1p(Δas_of_index)`` channel to an instrument's
    time-ordered standardized feature matrix. The gap to the previous observation
    gives the recurrence the spacing signal otherwise lost to the
    observations-as-adjacent approximation (the first step's gap is 0).

    Distribution note: in the train window consecutive daily bars give a gap of 1
    (``log1p(1) ≈ 0.69``), with occasional larger values at halts. At score time
    the boundary step from the train tail to the first test row spans the ~21-day
    purge, so its gap (``log1p(21) ≈ 3.09``) is larger than most training gaps —
    the net extrapolates on that one step. It has seen halt-sized gaps in
    training, so this is a mild extrapolation, not an unseen regime; documented
    here so the train/score gap-distribution difference is explicit."""
    if not include_time_gap:
        return z_feat
    gaps = np.zeros((len(idx), 1), dtype="float32")
    for j in range(1, len(idx)):
        delta = idx[j] - idx[j - 1]
        gaps[j, 0] = math.log1p(max(0, delta))
    return np.concatenate([z_feat, gaps], axis=1)


def _left_pad(window: np.ndarray, target_len: int) -> tuple[np.ndarray, np.ndarray]:
    """Left-pad a (w, C) window to (target_len, C) with a trailing-1s mask, so the
    most recent observation is the last timestep (what the GRU read-out reflects)."""
    w, channels = window.shape
    if w >= target_len:
        trimmed = window[w - target_len :]
        return trimmed.astype("float32"), np.ones(target_len, dtype="float32")
    pad = np.zeros((target_len - w, channels), dtype="float32")
    seq = np.concatenate([pad, window], axis=0)
    mask = np.concatenate([np.zeros(target_len - w, dtype="float32"), np.ones(w, dtype="float32")])
    return seq.astype("float32"), mask


def _tail_before(tail: tuple[np.ndarray, list[int]], earliest: int) -> tuple[np.ndarray, list[int]]:
    """Restrict a train tail to rows strictly before ``earliest`` as_of_index.

    Keeps the score-time stitch from prepending vectors that are not genuinely
    in a target's past (the score(train) overlap case). ``tail_ix`` is ascending.
    """
    tail_feat, tail_ix = tail
    keep = sum(1 for t in tail_ix if t < earliest)
    if keep == len(tail_ix):
        return tail
    return tail_feat[:keep], tail_ix[:keep]


def _group_indices_by_instrument(
    samples: Sequence[SupervisedAlphaSample],
) -> dict[uuid.UUID, list[int]]:
    by_inst: dict[uuid.UUID, list[int]] = {}
    for i, row in enumerate(samples):
        by_inst.setdefault(row.instrument_id, []).append(i)
    for idxs in by_inst.values():
        idxs.sort(key=lambda i: (_as_of_index(samples[i]), samples[i].as_of))
    return by_inst


def build_train_sequences(
    train: Sequence[SupervisedAlphaSample],
    names: Sequence[str],
    standardizer: _Standardizer,
    *,
    window: int,
    include_time_gap: bool,
    raw_matrix: np.ndarray | None = None,
) -> tuple[_SequenceBatch, dict[uuid.UUID, tuple[np.ndarray, list[int]]]]:
    """Build training windows + the per-instrument train-tail cache.

    The train tail is each instrument's last ``window-1`` standardized feature
    vectors *plus their as_of_index* (so the boundary time-gap can be recomputed
    at score time when the tail is stitched onto the test rows).

    ``raw_matrix`` lets the caller pass the dense design matrix it already built
    (``GRUSequenceRanker.fit`` builds it once to fit the standardizer) so it is
    not recomputed here; ``None`` builds it, keeping the function self-contained
    for tests.
    """
    z = standardizer.transform(raw_matrix if raw_matrix is not None else _matrix(train, names))
    by_inst = _group_indices_by_instrument(train)

    sequences: list[np.ndarray] = []
    masks: list[np.ndarray] = []
    as_of: list[datetime] = []
    labels: list[float] = []
    order: list[int] = []
    train_tail: dict[uuid.UUID, tuple[np.ndarray, list[int]]] = {}

    keep = max(0, window - 1)
    for inst, idxs in by_inst.items():
        z_feat = z[idxs]  # (k, n_features) standardized, time-ordered
        ix = [_as_of_index(train[i]) for i in idxs]
        z_aug = _augment_with_time_gap(z_feat, ix, include_time_gap=include_time_gap)
        for pos, original_idx in enumerate(idxs):
            lo = max(0, pos - window + 1)
            seq, mask = _left_pad(z_aug[lo : pos + 1], window)
            sequences.append(seq)
            masks.append(mask)
            as_of.append(train[original_idx].as_of)
            labels.append(float(train[original_idx].forward_return))
            order.append(original_idx)
        train_tail[inst] = (z_feat[-keep:].copy() if keep else z_feat[:0].copy(), ix[-keep:])

    batch = _SequenceBatch(
        sequences=np.stack(sequences) if sequences else np.empty((0, window, 0), "float32"),
        mask=np.stack(masks) if masks else np.empty((0, window), "float32"),
        as_of=as_of,
        labels=np.asarray(labels, dtype="float32"),
        order=order,
    )
    return batch, train_tail


def build_score_sequences(
    score_rows: Sequence[SupervisedAlphaSample],
    names: Sequence[str],
    standardizer: _Standardizer,
    *,
    window: int,
    include_time_gap: bool,
    train_tail: Mapping[uuid.UUID, tuple[np.ndarray, list[int]]],
) -> _SequenceBatch:
    """Build score windows, stitching each instrument's train tail before its
    test rows so a test row's trailing context spans the train/test boundary."""
    z = standardizer.transform(_matrix(score_rows, names))
    by_inst = _group_indices_by_instrument(score_rows)

    sequences: list[np.ndarray] = []
    masks: list[np.ndarray] = []
    as_of: list[datetime] = []
    order: list[int] = []

    for inst, idxs in by_inst.items():
        z_feat = z[idxs]
        ix = [_as_of_index(score_rows[i]) for i in idxs]
        tail = train_tail.get(inst)
        # Prepend only tail rows STRICTLY EARLIER than the earliest passed row.
        # ``ix`` is ascending, so ``ix[0]`` is the earliest. For score(test) every
        # tail row (train, before the test window) qualifies — the correct
        # preceding context. For score(train) — which the driver calls for the
        # volatility scale — the tail rows (the *latest* window-1 train vectors)
        # are NOT earlier than the early train rows, so none are kept and the
        # windows reduce to exactly what build_train produced. Without this guard
        # the latest train vectors would be prepended in front of early train
        # rows, scrambling time order and leaking future train features.
        kept_tail = None if tail is None else _tail_before(tail, ix[0])
        if kept_tail is None or kept_tail[0].shape[0] == 0:
            feat, full_ix, offset = z_feat, ix, 0
        else:
            tail_feat, tail_ix = kept_tail
            feat = np.concatenate([tail_feat, z_feat], axis=0)
            full_ix = list(tail_ix) + ix
            offset = tail_feat.shape[0]
        z_aug = _augment_with_time_gap(feat, full_ix, include_time_gap=include_time_gap)
        for j, original_idx in enumerate(idxs):
            pos = offset + j
            lo = max(0, pos - window + 1)
            seq, mask = _left_pad(z_aug[lo : pos + 1], window)
            sequences.append(seq)
            masks.append(mask)
            as_of.append(score_rows[original_idx].as_of)
            order.append(original_idx)

    return _SequenceBatch(
        sequences=np.stack(sequences) if sequences else np.empty((0, window, 0), "float32"),
        mask=np.stack(masks) if masks else np.empty((0, window), "float32"),
        as_of=as_of,
        labels=None,
        order=order,
    )


# --------------------------------------------------------------------------- #
# Torch model (built lazily — never at import time).
# --------------------------------------------------------------------------- #


def _build_net(
    torch: ModuleType, input_size: int, hidden: int, layers: int, dropout: float
) -> TorchModule:
    nn = torch.nn

    class _GRUNet(nn.Module):  # type: ignore[misc, name-defined]
        def __init__(self) -> None:
            super().__init__()
            self.gru = nn.GRU(
                input_size=input_size,
                hidden_size=hidden,
                num_layers=layers,
                batch_first=True,
                dropout=dropout if layers > 1 else 0.0,
            )
            self.head = nn.Linear(hidden, 1)

        def forward(self, x: Tensor, mask: Tensor) -> Tensor:
            out, _ = self.gru(x)  # (B, T, H)
            # Read the last *real* timestep. Left-padding puts real steps at the
            # tail, so this is index T-1 whenever the row has >=1 observation;
            # the mask-driven form is explicit and survives a padding change.
            last = mask.sum(dim=1).long().clamp(min=1) - 1  # (B,)
            h_last = out[torch.arange(out.size(0), device=out.device), last]  # (B, H)
            return self.head(h_last).squeeze(-1)  # type: ignore[no-any-return]  # (B,)

    return _GRUNet()  # type: ignore[no-any-return]


def _seed_everything(torch: ModuleType, seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # CPU is fully deterministic; cuDNN GRU is only approximately so, hence
    # warn_only — the arm is a new arm, never bit-compared across machines.
    # suppress: older torch builds lack the flag.
    with contextlib.suppress(Exception):
        torch.use_deterministic_algorithms(True, warn_only=True)


def _date_pearson(torch: ModuleType, pred: Tensor, label: Tensor) -> Tensor:
    """Differentiable cross-sectional Pearson of predictions vs labels (soft IC)."""
    pred = pred - pred.mean()
    label = label - label.mean()
    num = (pred * label).sum()
    den = torch.sqrt((pred * pred).sum() * (label * label).sum() + _EPS)
    return num / (den + _EPS)  # type: ignore[no-any-return]


def _train_loop(
    torch: ModuleType,
    net: TorchModule,
    batch: _SequenceBatch,
    *,
    objective: Objective,
    device: str,
    epochs: int,
    lr: float,
    grad_clip: float,
) -> None:
    if batch.labels is None:  # train batch always carries labels (defensive)
        raise ValueError("_train_loop requires a labelled batch")
    x = torch.from_numpy(batch.sequences).to(device)
    mask = torch.from_numpy(batch.mask).to(device)
    y = torch.from_numpy(batch.labels).to(device)

    # Group target rows by date once (sorted for determinism), mirroring the
    # ``by_day`` pattern. Only used by the IC objective.
    date_groups: list[Tensor] = []
    if objective == "ic":
        by_date: dict[datetime, list[int]] = {}
        for i, d in enumerate(batch.as_of):
            by_date.setdefault(d, []).append(i)
        for d in sorted(by_date):
            members = by_date[d]
            if len(members) >= 2:  # Pearson needs >= 2 names in the cross-section
                date_groups.append(torch.tensor(members, dtype=torch.long, device=device))

    opt = torch.optim.Adam(net.parameters(), lr=lr)
    net.train()
    for _epoch in range(epochs):
        opt.zero_grad()
        if objective == "ic":
            if not date_groups:  # every date had < 2 names -> nothing to learn
                break
            per_date = [_date_pearson(torch, net(x[g], mask[g]), y[g]) for g in date_groups]
            loss = -torch.stack(per_date).mean()
        else:
            loss = ((net(x, mask) - y) ** 2).mean()
        if not torch.isfinite(loss):  # NaN/inf guard: skip the step rather than poison weights
            opt.zero_grad()
            continue
        loss.backward()
        if grad_clip:
            torch.nn.utils.clip_grad_norm_(net.parameters(), grad_clip)
        opt.step()


# --------------------------------------------------------------------------- #
# The AlphaModel and its fitted form.
# --------------------------------------------------------------------------- #


class _FittedGRU:
    """Immutable per-fold fit: a trained GRU + frozen standardizer + train tail."""

    __slots__ = (
        "_device",
        "_include_time_gap",
        "_names",
        "_net",
        "_standardizer",
        "_train_tail",
        "_weights",
        "_window",
    )

    def __init__(
        self,
        *,
        net: TorchModule,
        names: Sequence[str],
        standardizer: _Standardizer,
        train_tail: Mapping[uuid.UUID, tuple[np.ndarray, list[int]]],
        window: int,
        include_time_gap: bool,
        device: str,
        weights: Mapping[str, float],
    ) -> None:
        self._net = net
        self._names = list(names)
        self._standardizer = standardizer
        self._train_tail = dict(train_tail)
        self._window = window
        self._include_time_gap = include_time_gap
        self._device = device
        self._weights = dict(weights)

    def score(self, samples: Sequence[SupervisedAlphaSample]) -> list[float]:
        if not samples:
            return []
        torch = _require_torch()
        batch = build_score_sequences(
            samples,
            self._names,
            self._standardizer,
            window=self._window,
            include_time_gap=self._include_time_gap,
            train_tail=self._train_tail,
        )
        x = torch.from_numpy(batch.sequences).to(self._device)
        mask = torch.from_numpy(batch.mask).to(self._device)
        self._net.eval()
        with torch.no_grad():
            pred = self._net(x, mask).cpu().numpy()
        pred = np.nan_to_num(pred, nan=0.0, posinf=0.0, neginf=0.0)
        out = [0.0] * len(samples)
        for i, original_idx in enumerate(batch.order):
            out[original_idx] = float(pred[i])
        return out

    def feature_weights(self) -> Mapping[str, float]:
        return dict(self._weights)


class GRUSequenceRanker:
    """GRU sequence-model ranker behind the :class:`AlphaModel` protocol.

    Refit per fold: standardize features (train stats), build per-instrument
    windows, train a small GRU, cache the standardizer + train tail + net in an
    immutable :class:`_FittedGRU`. ``objective="ic"`` (default) trains a per-date
    Pearson IC loss; ``objective="mse"`` minimizes return-level MSE. ``name``
    encodes the objective but not the hardware.
    """

    def __init__(
        self,
        *,
        objective: Objective = "ic",
        device: Device = "auto",
        window: int = _DEFAULT_WINDOW,
        hidden_size: int = _DEFAULT_HIDDEN,
        num_layers: int = _DEFAULT_LAYERS,
        dropout: float = _DEFAULT_DROPOUT,
        epochs: int = _DEFAULT_EPOCHS,
        lr: float = _DEFAULT_LR,
        seed: int = _DEFAULT_SEED,
        include_time_gap: bool = True,
        grad_clip: float = _DEFAULT_GRAD_CLIP,
    ) -> None:
        if objective not in _OBJECTIVE_NAME:
            raise ValueError(f"unsupported objective: {objective!r}")
        if device not in ("auto", "cuda", "cpu"):
            raise ValueError(f"unsupported device: {device!r}")
        if window < 1:
            raise ValueError("window must be >= 1")
        if hidden_size < 1:
            raise ValueError("hidden_size must be >= 1")
        if num_layers < 1:
            raise ValueError("num_layers must be >= 1")
        if epochs < 1:
            raise ValueError("epochs must be >= 1")
        if lr <= 0.0:
            raise ValueError("lr must be > 0")
        self._objective: Objective = objective
        self._device: Device = device
        self._window = window
        self._hidden_size = hidden_size
        self._num_layers = num_layers
        self._dropout = dropout
        self._epochs = epochs
        self._lr = lr
        self._seed = seed
        self._include_time_gap = include_time_gap
        self._grad_clip = grad_clip
        self.name = _OBJECTIVE_NAME[objective]

    def _resolve_device(self) -> str:
        if self._device == "auto":
            return "cuda" if _cuda_available() else "cpu"
        return self._device

    def fit(
        self,
        train: Sequence[SupervisedAlphaSample],
        feature_names: Sequence[str] | None = None,
    ) -> _FittedGRU:
        torch = _require_torch()
        names = _select_feature_names(train, feature_names)
        if not names:
            raise ValueError("GRUSequenceRanker.fit: no features available in training samples")
        _seed_everything(torch, self._seed)
        raw = _matrix(train, names)  # built once; reused for the standardizer + windows
        standardizer = _Standardizer.fit(raw)
        batch, train_tail = build_train_sequences(
            train,
            names,
            standardizer,
            window=self._window,
            include_time_gap=self._include_time_gap,
            raw_matrix=raw,
        )
        input_size = len(names) + (1 if self._include_time_gap else 0)
        device = self._resolve_device()
        net = _build_net(torch, input_size, self._hidden_size, self._num_layers, self._dropout).to(
            device
        )
        _train_loop(
            torch,
            net,
            batch,
            objective=self._objective,
            device=device,
            epochs=self._epochs,
            lr=self._lr,
            grad_clip=self._grad_clip,
        )
        net.eval()
        return _FittedGRU(
            net=net,
            names=names,
            standardizer=standardizer,
            train_tail=train_tail,
            window=self._window,
            include_time_gap=self._include_time_gap,
            device=device,
            weights=_input_weight_importance(net, names),
        )


def _input_weight_importance(net: TorchModule, names: Sequence[str]) -> dict[str, float]:
    """Normalized L1 of the GRU input weights over the named features.

    A cheap, deterministic reporting proxy for ``selected_weights`` /
    cross-fold ``feature_stability`` — it omits the recurrence, the head, and the
    appended time-gap channel, so it describes relative *input* influence, not
    causal importance. Equal-weight fallback if degenerate.
    """
    # weight_ih_l0: (3 * hidden, input_size); sum |W| over rows -> per-input.
    w = net.gru.weight_ih_l0.detach().abs().sum(dim=0).cpu().numpy()  # type: ignore[operator, union-attr]
    per_feature = w[: len(names)]  # drop the trailing time-gap channel if present
    total = float(per_feature.sum())
    if total <= 0.0:
        equal = 1.0 / max(1, len(names))
        return {name: equal for name in names}
    return {name: float(per_feature[i]) / total for i, name in enumerate(names)}


__all__ = ["GRUSequenceRanker"]
