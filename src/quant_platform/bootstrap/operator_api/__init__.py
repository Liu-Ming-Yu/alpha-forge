"""Operator-API bootstrap composition.

Composition for the read-only operator HTTP API: the FastAPI app builder
(``app``), runtime dependency wiring (``dependencies``), and the query/payload
adapters (``queries``, ``research_queries``) that keep view routers off
infrastructure. Intentionally import-light so importing a submodule does not
pull in the FastAPI app.
"""
