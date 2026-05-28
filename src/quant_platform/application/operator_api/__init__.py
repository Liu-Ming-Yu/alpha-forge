"""Operator-API read models.

Projection logic and DTOs/ports for the operator API's read side. These are
application-layer concerns — pure transformations over core contracts — kept
separate from the HTTP edge in ``views/operator_api`` so both the view layer
and bootstrap composition can depend on them without a layering cycle.
"""
