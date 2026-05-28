"""Research composition package.

A composition-tier peer of ``bootstrap`` -- it assembles CLI-invoked research
workflows (campaigns, backtests, feature builds, model-registry and alpha
governance operations) on top of bootstrap's session, signal-model, data and
migration composition helpers. Like ``bootstrap`` it may import infrastructure
and other composition packages; nothing in ``bootstrap`` or ``engines`` imports
it, so the composition tier stays acyclic. The ``check_composition_layering``
ratchet treats ``research`` as a composition source layer.
"""

from __future__ import annotations
