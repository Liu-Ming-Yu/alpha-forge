"""Live price-volume + formulaic feature family (G-live integration, ADR-011).

This package is the inner-layer (``services``) home for the live ``pv_formulaic``
feature family that lets the engine trade Arm G's construction during the paper
soak. It is built in layering-clean increments (see ADR-011 §"Sequencing"):

1. (this commit) ``bars_frame`` — a pure ``MarketBar`` → OHLCV-frame adapter, the
   live family's input conversion. Depends only on ``core`` + pandas.
2. (next) the feature-compute *kernel* port — the price_volume + formulaic math
   moved into the inner layer and consumed by both research and live, giving
   transform parity by construction.
3. the ``BundleFeatureComputer`` + family registration, and the G strategy plugin.

The high-blast-radius kernel move (step 2) is intentionally separate.
"""
