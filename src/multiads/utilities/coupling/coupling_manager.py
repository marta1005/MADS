"""CFD-CSM coupling manager.

This module provides the high-level orchestrator for coupled
CFD (DUST) and CSM (Lagrange or MBDyn) simulations using an iterative
fixed-point approach.

This module is REFACTORED to use the adapter pattern for structural solvers.
The CFDCSMCouplingManager delegates to UnifiedCouplingManager internally.

For new code, consider using UnifiedCouplingManager directly with the
appropriate adapter (LagrangeAdapter or MBDynAdapter).
"""

from __future__ import annotations

import logging
import os
import shutil
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
from scipy import linalg

from .box_definition import BoxDefinition, load_boxes_from_json
from .executor import run_dust, run_dust_pre, setup_logging
from .lagrange_adapter import LagrangeAdapter
from .mbdyn_adapter import MBDynAdapter
from .mesh_coupling import ForceIntegrationResult, integrate_forces
from .mesh_deformation import DeformationResult, apply_structural_deformation
from .structural_solver_protocol import StructuralSolverProtocol

if TYPE_CHECKING:
    from pyNastran.bdf.bdf import BDF

logger = logging.getLogger(__name__)


@dataclass
class CFDCSMConfig:
    """Configuration for CFD-CSM coupling.

    Attributes:
        tolerance: Convergence tolerance.
        max_iterations: Maximum coupling iterations.
        rbf_mode: Use RBF interpolation (True) or section-based (False).
        propeller_mode: Include propeller coupling.
        box_symmetry: Enable symmetric box generation.
        box_debug: Enable box debugging plots.
        plot_convergence: Plot convergence history.
    """

    tolerance: float = 1e-3
    max_iterations: int = 100
    rbf_mode: bool = True
    propeller_mode: bool = False
    box_symmetry: bool = False
    box_debug: bool = False
    plot_convergence: bool = True


@dataclass
class CFDCSMPaths:
    """Paths for CFD-CSM coupling.

    Attributes:
        bdf: Path to NASTRAN BDF file.
        res_h5: Path to DUST results HDF5.
        geo_h5: Path to DUST geometry HDF5.
        geo_in: Path to geometry input file.
        spline_set: Path to spline set file.
        dust_work_dir: Working directory for DUST.
        lag_job_dir: Lagrange job directory.
        lag_input: Lagrange input file.
    """

    bdf: Path | str
    res_h5: Path | str
    geo_h5: Path | str
    geo_in: Path | str
    spline_set: Path | str | None = None
    dust_work_dir: Path | str = "."
    lag_job_dir: Path | str = "."
    lag_input: Path | str = "lagrange.inp"


@dataclass
class CFDCSMPropellerConfig:
    """Propeller-specific configuration.

    Attributes:
        prop_node_ids: Grid point IDs for propellers.
        n_blades_per_prop: Number of blades per propeller.
        comp_start_h5: Component start for HDF5.
        comp_start_geo: Component start for geometry.
    """

    prop_node_ids: list[int] = field(default_factory=list)
    n_blades_per_prop: list[int] = field(default_factory=list)
    comp_start_h5: int = 2
    comp_start_geo: int = 2


@dataclass
class CFDCSMConvergence:
    """Coupling convergence data.

    Attributes:
        iterations: Iteration counts.
        disp_norm: Displacement norms.
        force_norm: Force norms.
        converged: Whether coupling converged.
    """

    iterations: list[int] = field(default_factory=list)
    disp_norm: list[float] = field(default_factory=list)
    force_norm: list[float] = field(default_factory=list)
    converged: bool = False


class CFDCSMCouplingManager:
    """Manager for CFD-CSM coupling iterations.

    This class orchestrates the coupled simulation loop between
    DUST (aerodynamics) and Lagrange or MBDyn (structural dynamics).

    This class supports two modes:
    1. Legacy mode: Uses internal Lagrange handling (default, backward compatible)
    2. Adapter mode: Uses UnifiedCouplingManager with adapters for Lagrange/MBDyn

    The adapter mode is enabled by setting `use_adapter=True` or by using
    `UnifiedCouplingManager` directly with the appropriate adapter.
    """

    def __init__(
        self,
        boxes: list[BoxDefinition] | list[dict[str, Any]] | Path | str,
        paths: CFDCSMPaths,
        config: CFDCSMConfig | None = None,
        propeller_config: CFDCSMPropellerConfig | None = None,
        lag_set: int = 400,
        force_path: str = "Components/Comp001/Solution/dF",
        position_path: str = "References/Ref001/Offset",
        nodes_path: str = "Components/Comp001/Geometry/rr",
        elements_path: str = "Components/Comp001/Geometry/ee",
        n_threads: int = 10,
        use_adapter: bool = False,
        structural_solver: StructuralSolverProtocol | None = None,
        solver_type: str = "lagrange",
    ) -> None:
        """Initialize coupling manager.

        Args:
            boxes: Box definitions (list, dict, or path to JSON).
            paths: File paths configuration.
            config: Coupling configuration.
            propeller_config: Propeller configuration.
            lag_set: Lagrange element set.
            force_path: HDF5 path to forces.
            position_path: HDF5 path to position.
            nodes_path: HDF5 path to nodes.
            elements_path: HDF5 path to elements.
            n_threads: Number of threads for DUST.
            use_adapter: Use adapter-based coupling (recommended for new code).
            structural_solver: Pre-configured structural solver adapter.
            solver_type: Type of solver to use ('lagrange' or 'mbdyn').
        """
        self._load_boxes(boxes)
        self.paths = paths
        self.config = config or CFDCSMConfig()
        self.propeller_config = propeller_config
        self.lag_set = lag_set
        self.force_path = force_path
        self.position_path = position_path
        self.nodes_path = nodes_path
        self.elements_path = elements_path
        self.n_threads = n_threads
        self.solver_type = solver_type
        self.use_adapter = use_adapter

        self._unified_manager: UnifiedCouplingManager | None = None
        if use_adapter or structural_solver is not None:
            self._setup_unified_manager(structural_solver, solver_type)

        self._convergence = CFDCSMConvergence()
        self._displacements: np.ndarray | None = None
        self._forces: np.ndarray | None = None
        self._lagrange_module = None
        self._lagrange_instance = None

    def _setup_unified_manager(
        self,
        structural_solver: StructuralSolverProtocol | None,
        solver_type: str,
    ) -> None:
        """Set up the unified coupling manager with adapters.

        Args:
            structural_solver: Pre-configured adapter or None.
            solver_type: Type of solver ('lagrange' or 'mbdyn').
        """
        if structural_solver is not None:
            adapter = structural_solver
        else:
            from .unified_coupling_manager import UnifiedCouplingManager

            if solver_type.lower() == "mbdyn":
                adapter = MBDynAdapter(config=None, paths=None)
            else:
                adapter = LagrangeAdapter(config=None, paths=None)

        self._unified_manager = UnifiedCouplingManager(
            structural_solver=adapter,
            boxes=self.boxes,
            config=None,
            force_path=self.force_path,
            position_path=self.position_path,
            nodes_path=self.nodes_path,
            elements_path=self.elements_path,
            dust_work_dir=self.paths.dust_work_dir,
            geo_h5_path=self.paths.geo_h5,
            geo_in_path=self.paths.geo_in,
            bdf_path=self.paths.bdf,
        )

    def _load_boxes(
        self,
        boxes: list[BoxDefinition] | list[dict[str, Any]] | Path | str,
    ) -> None:
        """Load box definitions from various sources.

        Args:
            boxes: Box definitions or path to JSON.
        """
        if isinstance(boxes, (Path, str)):
            if Path(boxes).suffix == ".json":
                self.boxes = load_boxes_from_json(boxes)
            else:
                self.boxes = boxes
        else:
            if boxes and isinstance(boxes[0], dict):
                self.boxes = [BoxDefinition.from_dict(d) for d in boxes]
            else:
                self.boxes = boxes

    def _setup_logging(self) -> None:
        """Set up logging for coupling simulation."""
        setup_logging(log_file="cfd_csm.log", level=logging.INFO)
        logger.info("CFD-CSM coupling initialized")

    def _read_model(self) -> BDF:
        """Read NASTRAN BDF model.

        Returns:
            BDF model.
        """
        try:
            from pyNastran.bdf.bdf import BDF
        except ImportError as e:
            raise ImportError("pyNastran is required for BDF file handling.") from e

        model = BDF(debug=False, log=None)
        model.read_bdf(str(self.paths.bdf), punch=True)
        return model

    def _integrate_forces(self) -> ForceIntegrationResult:
        """Integrate aerodynamic forces.

        Returns:
            Force integration result.
        """
        boxes_data = [
            b.to_dict() if isinstance(b, BoxDefinition) else b for b in self.boxes
        ]

        return integrate_forces(
            boxes_data=boxes_data,
            res_h5_path=self.paths.res_h5,
            geo_h5_path=self.paths.geo_h5,
            force_path=self.force_path,
            position_path=self.position_path,
            nodes_path=self.nodes_path,
            elements_path=self.elements_path,
            bdf_path=self.paths.bdf,
            symmetry=self.config.box_symmetry,
            prop_node_ids=self.propeller_config.prop_node_ids
            if self.propeller_config
            else None,
            n_blades_per_prop=self.propeller_config.n_blades_per_prop
            if self.propeller_config
            else None,
            comp_start_h5=self.propeller_config.comp_start_h5
            if self.propeller_config
            else None,
            comp_start_geo=self.propeller_config.comp_start_geo
            if self.propeller_config
            else None,
        )

    def _apply_deformation(
        self,
        displacements: np.ndarray,
    ) -> DeformationResult | None:
        """Apply structural deformation to CFD mesh.

        Args:
            displacements: Lagrange displacements.

        Returns:
            Deformation result.
        """
        deformed_nodes = apply_structural_deformation(
            displacements=displacements,
            spline_set_path=self.paths.spline_set,
            bdf_path=self.paths.bdf,
            geo_h5_path=self.paths.geo_h5,
            nodes_h5_path=self.nodes_path,
            rbf_mode=self.config.rbf_mode,
            propeller_nodes=None,
        )

        if deformed_nodes is not None:
            return update_geometry_files(
                deformed_nodes=deformed_nodes,
                geo_h5_path=self.paths.geo_h5,
                geo_in_path=self.paths.geo_in,
            )

        return None

    def _check_convergence(
        self,
        ug_new: np.ndarray,
        F_new: np.ndarray,
    ) -> bool:
        """Check coupling convergence.

        Args:
            ug_new: New displacements.
            F_new: New forces.

        Returns:
            True if converged.
        """
        if self._displacements is not None and self._forces is not None:
            disp_norm = linalg.norm(ug_new - self._displacements) / linalg.norm(ug_new)
            force_norm = linalg.norm(F_new - self._forces) / linalg.norm(F_new)

            self._convergence.iterations.append(len(self._convergence.iterations) + 1)
            self._convergence.disp_norm.append(disp_norm)
            self._convergence.force_norm.append(force_norm)

            logger.info(f"Convergence: disp={disp_norm:.6e}, force={force_norm:.6e}")

            if disp_norm < self.config.tolerance and force_norm < self.config.tolerance:
                self._convergence.converged = True
                return True

        self._displacements = deepcopy(ug_new) if ug_new is not None else None
        self._forces = deepcopy(F_new) if F_new is not None else None

        return False

    def _plot_convergence(self) -> None:
        """Plot convergence history."""
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            logger.warning("matplotlib not available for convergence plots")
            return

        if not self._convergence.iterations:
            return

        fig, ax = plt.subplots(1, figsize=(15, 6), facecolor="w", edgecolor="k")
        ax.plot(self._convergence.iterations, self._convergence.disp_norm, marker="x")
        ax.plot(self._convergence.iterations, self._convergence.force_norm, marker="o")
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Convergence Criterion")
        ax.set_yscale("log")
        ax.grid(True)
        fig.savefig("convergence_cfd_csm.png", dpi=150)
        plt.close(fig)

    def _initialize_lagrange(self) -> None:
        """Initialize Lagrange structural solver."""
        try:
            import lagrange
        except ImportError:
            logger.warning("Lagrange module not available")
            self._lagrange_module = None
            return

        lagrange.init()
        self._lagrange_module = lagrange

        lag_inp = Path(self.paths.lag_job_dir) / self.paths.lag_input
        if not lag_inp.exists():
            logger.warning(f"Lagrange input file not found: {lag_inp}")
            return

        self._lagrange_instance = lagrange.Lagrange(str(lag_inp))
        self._lagrange_instance.preproc()
        self._lagrange_instance.design()

    def _run_lagrange(
        self,
        dofs: list[list[int]],
        forces: np.ndarray,
        moments: np.ndarray,
        iteration: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Run Lagrange structural analysis.

        Args:
            dofs: Degrees of freedom.
            forces: Applied forces.
            moments: Applied moments.
            iteration: Current iteration.

        Returns:
            Tuple of (displacements, applied loads).
        """
        if self._lagrange_instance is None:
            return np.zeros(sum(len(d) for d in dofs)), np.zeros(
                sum(len(d) for d in dofs)
            )

        sc_idx = self._lagrange_instance.get_sc_info()
        sc_eid = sc_idx[0]
        n_g = self._lagrange_instance.get_sd_sc_ndof(sc_eid)

        self._lagrange_instance.get_mesh_set(self.lag_set)
        pg = self._lagrange_instance.get_ad_sc_pg(sc_eid)

        if iteration == 1:
            gravity = self._lagrange_instance.get_ad_sc_pg(sc_eid)
        else:
            gravity = getattr(self, "_gravity", np.zeros(n_g))

        self._gravity = gravity

        ug = np.zeros(n_g)
        for i, dof in enumerate(dofs):
            for j in range(3):
                if dof[j] - 1 < len(pg):
                    pg[dof[j] - 1] = forces[i, j] + gravity[dof[j] - 1]
                if dof[j + 3] - 1 < len(pg):
                    pg[dof[j + 3] - 1] = moments[i, j]

        self._lagrange_instance.set_ad_sc_user_pg(sc_eid, pg)
        self._lagrange_instance.analysis()

        ug = self._lagrange_instance.get_ad_sc_ug(sc_eid)
        pg = self._lagrange_instance.get_ad_sc_pg(sc_eid)

        return ug, pg

    def _cleanup(self, result: DeformationResult | None) -> None:
        """Clean up temporary files.

        Args:
            result: Deformation result with file paths.
        """
        if result is None:
            return

        if result.copied_in_file and result.copied_in_file.exists():
            shutil.copy(result.copied_in_file, self.paths.geo_in)
            result.copied_in_file.unlink()

        if result.copied_geo_h5 and result.copied_geo_h5.exists():
            shutil.copy(result.copied_geo_h5, self.paths.geo_h5)
            result.copied_geo_h5.unlink()

    def run(self) -> CFDCSMConvergence:
        """Run the CFD-CSM coupling loop.

        Returns:
            Convergence data.
        """
        if self._unified_manager is not None:
            return self._run_with_adapter()

        return self._run_legacy()

    def _run_with_adapter(self) -> CFDCSMConvergence:
        """Run coupling using the unified manager with adapters.

        Returns:
            Convergence data.
        """
        self._setup_logging()

        if self._unified_manager is None:
            raise RuntimeError("Unified manager not initialized")

        logger.info(f"Running CFD-CSM coupling with {self.solver_type} adapter")

        self._unified_manager.run_cfd()

        iteration = 0
        while iteration < self.config.max_iterations:
            iteration += 1
            logger.info(f"Coupling iteration {iteration}")

            force_result = self._unified_manager.integrate_forces_from_cfd()
            ug, _ = self._unified_manager.apply_structural_deformation(
                force_result.forces
            )

            self._unified_manager.run_cfd()

            if self._check_convergence(ug, force_result.forces):
                logger.info("Coupling converged!")
                break

        if self.config.plot_convergence:
            self._plot_convergence()

        self._unified_manager.cleanup()

        return self._convergence

    def _run_legacy(self) -> CFDCSMConvergence:
        """Run coupling using legacy Lagrange integration.

        Returns:
            Convergence data.
        """
        self._setup_logging()
        self._initialize_lagrange()

        logger.info("Running initial CFD simulation...")
        run_dust_pre(self.n_threads, self.paths.dust_work_dir)
        run_dust("dust", self.n_threads, self.paths.dust_work_dir)

        iteration = 0
        dofs = []
        forces = np.zeros((len(self.boxes), 3))
        moments = np.zeros((len(self.boxes), 3))

        while iteration < self.config.max_iterations:
            iteration += 1
            logger.info(f"Coupling iteration {iteration}")

            force_result = self._integrate_forces()
            dofs = force_result.dofs
            forces = force_result.forces
            moments = force_result.moments

            ug, pg = self._run_lagrange(dofs, forces, moments, iteration)

            deformation_result = self._apply_deformation(ug)

            run_dust("dust", self.n_threads, self.paths.dust_work_dir)

            if self._check_convergence(ug, forces):
                logger.info("Coupling converged!")
                self._cleanup(deformation_result)
                break

        if self.config.plot_convergence:
            self._plot_convergence()

        if self._lagrange_module is not None:
            self._lagrange_instance.output()
            self._lagrange_module.final()

        return self._convergence

    @property
    def convergence(self) -> CFDCSMConvergence:
        """Get convergence data.

        Returns:
            Convergence data.
        """
        return self._convergence


def update_geometry_files(
    deformed_nodes: np.ndarray,
    geo_h5_path: Path | str,
    geo_in_path: Path | str,
    output_dir: Path | str | None = None,
) -> DeformationResult:
    """Update geometry files with deformed mesh.

    This is a convenience wrapper around mesh_deformation.update_geometry_files.

    Args:
        deformed_nodes: Deformed node positions.
        geo_h5_path: Path to geometry HDF5 file.
        geo_in_path: Path to geometry input file.
        output_dir: Output directory for copied files.

    Returns:
        DeformationResult with updated file paths.
    """
    from .mesh_deformation import update_geometry_files as _update

    return _update(
        deformed_nodes=deformed_nodes,
        geo_h5_path=geo_h5_path,
        geo_in_path=geo_in_path,
        output_dir=output_dir,
    )
