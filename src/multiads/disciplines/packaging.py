"""Packaging discipline wrappers.

Phase 1 keeps this file intentionally minimal. The real solver-backed
discipline will be connected once the packaging solver exists.
"""

from multiads.disciplines import MADSDiscipline


class Packaging(MADSDiscipline):
    """Thin semantic wrapper for packaging/constraint solvers."""

