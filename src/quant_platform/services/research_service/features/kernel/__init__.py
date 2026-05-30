"""Inner-layer feature-compute kernel (ADR-011).

Canonical home for the pure feature contracts and computation that both the
research feature factory (composition layer, via thin re-export shims) and the
live feature pipeline (``services`` layer) consume.

It lives in ``services`` so the live engine can import the feature math without
crossing the ``services -> research`` import boundary. The research feature
modules (``research.features.contracts``, ``…transforms``, ``…price_volume``,
``…formulaic``) re-export from here, so existing importers and the parity with
the research computation are preserved by construction.
"""
