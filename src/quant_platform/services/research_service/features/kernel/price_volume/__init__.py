"""Pure price-volume feature compute (inner-layer kernel, ADR-011).

Holds only the computation (config + feature math) — no ``register_family``
side-effect. The research family registration (``research.features.price_volume``)
stays in the composition layer and imports this compute; the live ``pv_formulaic``
family imports it directly.
"""
