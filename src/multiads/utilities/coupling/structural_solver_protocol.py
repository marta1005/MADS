"""Structural solver protocol for CFD-CSM coupling.

This module defines the protocol interface for structural solvers
(Lagrange, MBDyn) used in CFD-CSM coupling simulations.

The StructuralSolverProtocol defines the common interface that all
structural solver adapters must implement, enabling unified coupling
management regardless of the underlying solver.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

import numpy as np

if TYPE_CHECKING:
    from pathlib import Path


@runtime_checkable
class StructuralSolverProtocol(Protocol):
    """Protocol for structural solvers (Lagrange, MBDyn).

    This protocol defines the interface that all structural solver
    adapters must implement. Adapters wrap solver-specific logic
    while presenting a unified interface for the coupling manager.

    Example:
        class LagrangeAdapter(StructuralSolverProtocol):
            ...

        class MBDynAdapter(StructuralSolverProtocol):
            ...
    """

    def apply_loads(
        self,
        dofs: list[list[int]],
        forces: np.ndarray,
        moments: np.ndarray,
    ) -> None:
        """Apply forces and moments to structural model.

        Args:
            dofs: Degrees of freedom for each load application point.
                  Each entry is a list of 6 DOFs [ux, uy, uz, rx, ry, rz].
            forces: Applied forces [N, 3] where N is number of load points.
            moments: Applied moments [N, 3] where N is number of load points.
        """
        ...

    def run(self) -> None:
        """Execute structural analysis.

        Runs the structural solver with the currently applied loads.
        After execution, displacements and rotations can be retrieved
        using get_displacements() and get_rotations().
        """
        ...

    def get_displacements(self) -> np.ndarray:
        """Get nodal displacements.

        Returns:
            Displacements array [N, 3] or [N] depending on solver format.
            For N nodes, returns displacement values.
        """
        ...

    def get_rotations(self) -> np.ndarray:
        """Get nodal rotations.

        Returns:
            Rotations array [N, 3] or [N] depending on solver format.
            For N nodes, returns rotation values.
        """
        ...


class SolverType:
    """Enum-like class for structural solver types."""

    LAGRANGE = "lagrange"
    MBDYN = "mbdyn"


class CouplingMode:
    """Enum-like class for coupling modes."""

    FILE_BASED = "file"
    PRECICE = "precice"
