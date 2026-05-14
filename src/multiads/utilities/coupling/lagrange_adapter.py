"""Lagrange adapter for CFD-CSM coupling.

This module provides the LagrangeAdapter class that implements the
StructuralSolverProtocol for coupling DUST (CFD) with Lagrange (structural).

The adapter wraps the Lagrange solver API, handling load application,
analysis execution, and displacement extraction.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from .structural_solver_protocol import StructuralSolverProtocol

if TYPE_CHECKING:
    import lagrange

logger = logging.getLogger(__name__)


class LagrangeAdapter(StructuralSolverProtocol):
    """Adapter wrapping Lagrange structural solver.

    This adapter implements the StructuralSolverProtocol to enable
    unified coupling management with DUST CFD solver.

    Attributes:
        lag_input: Path to Lagrange input file.
        lag_set: Element set number for coupling.
        _instance: Lagrange solver instance.
        _displacements: Cached displacements array.
        _rotations: Cached rotations array.

    Example:
        adapter = LagrangeAdapter(
            lag_input="lagrange.inp",
            lag_set=400,
        )
        adapter.apply_loads(dofs, forces, moments)
        adapter.run()
        displacements = adapter.get_displacements()
    """

    def __init__(
        self,
        lag_input: Path | str = "lagrange.inp",
        lag_set: int = 400,
        work_dir: Path | str | None = None,
    ) -> None:
        """Initialize Lagrange adapter.

        Args:
            lag_input: Path to Lagrange input file.
            lag_set: Element set number for coupling.
            work_dir: Working directory for Lagrange execution.
        """
        self.lag_input = Path(lag_input)
        self.lag_set = lag_set
        self.work_dir = Path(work_dir) if work_dir else Path.cwd()

        self._instance: Any = None
        self._module: Any = None
        self._displacements: np.ndarray | None = None
        self._rotations: np.ndarray | None = None
        self._n_dof: int = 0
        self._gravity: np.ndarray | None = None

        self._initialize_lagrange()

    def _initialize_lagrange(self) -> None:
        """Initialize Lagrange solver instance."""
        try:
            import lagrange as lag
        except ImportError:
            logger.warning(
                "Lagrange API not available. Structural analysis will be skipped."
            )
            self._module = None
            self._instance = None
            return

        self._module = lag

        lag_inp = self.work_dir / self.lag_input
        if not lag_inp.exists():
            logger.warning(f"Lagrange input file not found: {lag_inp}")
            return

        try:
            lag.init()
            self._instance = lag.Lagrange(str(lag_inp))
            self._instance.preproc()
            self._instance.design()
            logger.info("Lagrange initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize Lagrange: {e}")
            self._instance = None

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
        if self._instance is None:
            logger.warning("Lagrange not initialized, skipping load application")
            return

        try:
            sc_idx = self._instance.get_sc_info()
            sc_eid = sc_idx[0]
            self._n_dof = self._instance.get_sd_sc_ndof(sc_eid)

            self._instance.get_mesh_set(self.lag_set)
            pg = self._instance.get_ad_sc_pg(sc_eid)

            if self._gravity is None:
                self._gravity = self._instance.get_ad_sc_pg(sc_eid).copy()

            for i, dof in enumerate(dofs):
                for j in range(3):
                    if dof[j] - 1 < len(pg):
                        pg[dof[j] - 1] = forces[i, j] + self._gravity[dof[j] - 1]
                    if dof[j + 3] - 1 < len(pg):
                        pg[dof[j + 3] - 1] = moments[i, j]

            self._instance.set_ad_sc_user_pg(sc_eid, pg)
            logger.debug(f"Applied {len(dofs)} load points")

        except Exception as e:
            logger.error(f"Failed to apply loads: {e}")

    def run(self) -> None:
        """Execute Lagrange structural analysis."""
        if self._instance is None:
            logger.warning("Lagrange not initialized, skipping analysis")
            self._displacements = np.zeros(self._n_dof)
            self._rotations = np.zeros(self._n_dof)
            return

        try:
            self._instance.analysis()

            sc_idx = self._instance.get_sc_info()
            sc_eid = sc_idx[0]

            self._displacements = self._instance.get_ad_sc_ug(sc_eid)
            self._rotations = self._instance.get_ad_sc_ug(sc_eid)

            logger.info("Lagrange analysis completed")

        except Exception as e:
            logger.error(f"Lagrange analysis failed: {e}")
            self._displacements = np.zeros(self._n_dof)
            self._rotations = np.zeros(self._n_dof)

    def get_displacements(self) -> np.ndarray:
        """Get nodal displacements from Lagrange.

        Returns:
            Displacements array [N] or empty array if not available.
        """
        if self._displacements is None:
            return np.zeros(self._n_dof)
        return np.asarray(self._displacements)

    def get_rotations(self) -> np.ndarray:
        """Get nodal rotations from Lagrange.

        Returns:
            Rotations array [N] or empty array if not available.
        """
        if self._rotations is None:
            return np.zeros(self._n_dof)
        return np.asarray(self._rotations)

    def output(self) -> None:
        """Write Lagrange output files."""
        if self._instance is not None:
            try:
                self._instance.output()
            except Exception as e:
                logger.error(f"Failed to write output: {e}")

    def finalize(self) -> None:
        """Finalize Lagrange solver."""
        if self._module is not None:
            try:
                self._module.final()
                logger.info("Lagrange finalized")
            except Exception as e:
                logger.error(f"Failed to finalize Lagrange: {e}")

    def __del__(self) -> None:
        """Cleanup on deletion."""
        self.finalize()
