"""Runtime session-state containers.

The ``Session`` runtime container and ``CycleResult`` live here -- in the
application layer -- because both the ``bootstrap`` composition layer (which
builds them) and the ``engines`` run loop (which operates on them) must depend
on the type without depending on each other. They sit above ``core`` (they
reference service-controller ports) but carry no wiring themselves.
"""
