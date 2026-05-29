"""Pure formulaic-alpha engine (inner-layer kernel, ADR-011).

The expression AST, operator catalog, evaluator, market panel, curated alpha
library, and config — the compute the formulaic family and the live
``pv_formulaic`` family both need. No ``register_family`` side-effect and no
mining/serialization (those stay in ``research.features.formulaic``, importing
this core through the per-module shims).
"""
