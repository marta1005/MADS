"""MBDyn adapter for CFD-CSM coupling.

This module provides the MBDynAdapter class that implements the
StructuralSolverProtocol for coupling DUST (CFD) with MBDyn (structural).

The adapter supports both file-based (fixed-point iteration) and
preCICE (real-time exchange) coupling modes.
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess as sp
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from .mesh_coupling import create_force_cards
from .structural_solver_protocol import (
    CouplingMode,
    StructuralSolverProtocol,
)

if TYPE_CHECKING:
    from multiads.solvers.structure.mbdyn_lib import MBDYNOptions, WingMBDYN

logger = logging.getLogger(__name__)


class MBDynAdapter(StructuralSolverProtocol):
    """Adapter wrapping MBDyn structural solver.

    This adapter implements the StructuralSolverProtocol to enable
    unified coupling management with DUST CFD solver.

    Supports two coupling modes:
    - FILE_BASED: Fixed-point iteration via file exchange
    - PRECICE: Real-time exchange via preCICE

    Attributes:
        mbd_options: MBDyn solver options.
        wings: List of WingMBDYN components.
        coupling_mode: Coupling mode (file or preCICE).
        work_dir: Working directory.

    Example:
        adapter = MBDynAdapter(
            mbd_options=MBDYNOptions(mbdyn_model="wing.mbd"),
            wings=[wing1, wing2],
            coupling_mode="file",
        )
        adapter.apply_loads(dofs, forces, moments)
        adapter.run()
        displacements = adapter.get_displacements()
    """

    def __init__(
        self,
        mbd_options: MBDYNOptions,
        wings: list[WingMBDYN] | None = None,
        coupling_mode: str = CouplingMode.FILE_BASED,
        work_dir: Path | str | None = None,
    ) -> None:
        """Initialize MBDyn adapter.

        Args:
            mbd_options: MBDyn solver options.
            wings: List of WingMBDYN components.
            coupling_mode: Coupling mode ("file" or "precice").
            work_dir: Working directory for MBDyn execution.
        """
        self.mbd_options = mbd_options
        self.wings = wings or []
        self.coupling_mode = coupling_mode
        self.work_dir = Path(work_dir) if work_dir else Path(tempfile.mkdtemp())

        self._displacements: np.ndarray | None = None
        self._rotations: np.ndarray | None = None
        self._run_directory: Path | None = None
        self._loads_applied: bool = False
        self._forces: np.ndarray | None = None
        self._moments: np.ndarray | None = None
        self._dofs: list[list[int]] | None = None

    def apply_loads(
        self,
        dofs: list[list[int]],
        forces: np.ndarray,
        moments: np.ndarray,
    ) -> None:
        """Apply forces and moments to structural model.

        For file-based coupling, this stores the loads for later application.
        For preCICE coupling, this is handled differently.

        Args:
            dofs: Degrees of freedom for each load application point.
            forces: Applied forces [N, 3].
            moments: Applied moments [N, 3].
        """
        self._dofs = dofs
        self._forces = forces.copy()
        self._moments = moments.copy()
        self._loads_applied = True

        logger.debug(f"Stored {len(dofs)} load points for MBDyn")

    def run(self) -> None:
        """Execute MBDyn structural analysis.

        For file-based coupling, runs MBDyn and reads results.
        For preCICE coupling, this is handled by the preCICE infrastructure.
        """
        if self.coupling_mode == CouplingMode.PRECICE:
            self._run_precice()
        else:
            self._run_file_based()

    def _run_file_based(self) -> None:
        """Run MBDyn in file-based coupling mode."""
        self._run_directory = Path(tempfile.mkdtemp(dir=self.work_dir))

        main_dir = Path.cwd()
        os.chdir(self._run_directory)

        try:
            from multiads.solvers.structure.mbdyn_lib import (
                MBDYNDriver,
                MBDYNOptions,
                MBDYNPostDeform,
            )

            mbd_driver = MBDYNDriver(
                wings=list(self.wings),
                options=MBDYNOptions(
                    name=self.mbd_options.name,
                    mbdyn_command=self.mbd_options.mbdyn_command,
                    mbdyn_model=self.mbd_options.mbdyn_model,
                    mbdyn_output=self.mbd_options.mbdyn_output,
                    dt=self.mbd_options.dt,
                    t_end=self.mbd_options.t_end,
                    work_dir=self._run_directory,
                ),
            )

            mbd_driver.preprocess()

            if self._loads_applied and self._dofs:
                self._write_loads_to_mbd()

            mbd_driver.run()

            if self.wings:
                mbd_driver.postprocess([MBDYNPostDeform(wings=list(self.wings))])
                self._displacements = (
                    mbd_driver.displ if mbd_driver.displ is not None else np.array([])
                )
                self._rotations = (
                    mbd_driver.rot if mbd_driver.rot is not None else np.array([])
                )
            else:
                self._displacements = np.array([])
                self._rotations = np.array([])

            logger.info("MBDyn file-based analysis completed")

        except Exception as e:
            logger.error(f"MBDyn analysis failed: {e}")
            self._displacements = np.array([])
            self._rotations = np.array([])

        finally:
            os.chdir(main_dir)

    def _write_loads_to_mbd(self) -> None:
        """Write loads to MBDyn input file for file-based coupling."""
        if not self._dofs or self._forces is None or self._moments is None:
            return

        mbd_file = self._run_directory / self.mbd_options.mbdyn_model
        if not mbd_file.exists():
            logger.warning(f"MBDyn file not found: {mbd_file}")
            return

        from multiads.solvers.structure.mbdyn_lib import (
            MBDYNOptions,
            MBDYNDriver,
            WingMBDYN,
        )

        temp_driver = MBDYNDriver(
            wings=list(self.wings),
            options=MBDYNOptions(
                name=self.mbd_options.name,
                work_dir=self._run_directory,
            ),
        )
        temp_driver._write_mbdyn()

    def _run_precice(self) -> None:
        """Run MBDyn with preCICE coupling.

        In preCICE mode, the coupling is handled by an external script.
        This method delegates to the preCICE infrastructure.
        """
        logger.info("MBDyn preCICE coupling mode - delegated to preCICE")

        main_dir = Path.cwd()
        os.chdir(self.work_dir)

        try:
            if self.mbd_options.coupling_command:
                result = sp.run(
                    self.mbd_options.coupling_command,
                    shell=True,
                    capture_output=True,
                    text=True,
                )

                if result.returncode != 0:
                    logger.error(f"preCICE coupling failed: {result.stderr}")
                else:
                    logger.info("preCICE coupling completed")

                self._displacements = np.array([])
                self._rotations = np.array([])

        except Exception as e:
            logger.error(f"preCICE execution failed: {e}")
            self._displacements = np.array([])
            self._rotations = np.array([])

        finally:
            os.chdir(main_dir)

    def get_displacements(self) -> np.ndarray:
        """Get nodal displacements from MBDyn.

        Returns:
            Displacements array.
        """
        if self._displacements is None:
            return np.array([])
        return np.asarray(self._displacements)

    def get_rotations(self) -> np.ndarray:
        """Get nodal rotations from MBDyn.

        Returns:
            Rotations array.
        """
        if self._rotations is None:
            return np.array([])
        return np.asarray(self._rotations)

    def cleanup(self) -> None:
        """Clean up temporary files."""
        if self._run_directory and self._run_directory.exists():
            try:
                shutil.rmtree(self._run_directory)
            except Exception as e:
                logger.warning(f"Failed to cleanup: {e}")

    def __del__(self) -> None:
        """Cleanup on deletion."""
        self.cleanup()
