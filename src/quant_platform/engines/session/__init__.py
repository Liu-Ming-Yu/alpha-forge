"""Session runtime: strategy-cycle execution, hydration, and regime stats.

These modules *operate on* a ``Session`` -- running rebalance cycles, hydrating
durable state, computing regime stats. They live in ``engines`` (the run loop),
not ``bootstrap`` (composition), so the engine runner reaches them without a
back-edge into the composition layer. Composition of a ``Session`` stays in
``bootstrap/session/``.
"""
