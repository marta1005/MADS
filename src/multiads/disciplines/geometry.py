"""Geometry discipline wrappers.

Phase 1 keeps this file intentionally minimal. The real solver-backed
discipline will be connected once the geometry solver exists.
"""

from multiads.disciplines import MADSDiscipline


class Geometry(MADSDiscipline):
    """Thin semantic wrapper for geometry solvers."""
    pass

