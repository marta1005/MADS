"""Unified CFD-CSM coupling manager.

This module provides the UnifiedCouplingManager class that orchestrates
coupled CFD (DUST) and structural solver (Lagrange, MBDyn) simulations
using an iterative fixed-point approach.

The manager handles:
- Force integration from DUST to structural solver
- Structural analysis execution via adapters
- Synthesis updates (wings, propellers, fuselage)
- Mesh deformation from structural to DUST
- Convergence checking
"""

from __future__ import annotations

import logging
import shutil
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
from scipy import linalg

from .box_definition import BoxDefinition, load_boxes_from_json
from .executor import run_dust, run_dust_pre, setup_logging
from .mesh_coupling import ForceIntegrationResult, integrate_forces
from .mesh_deformation import (
    DeformationResult,
    apply_structural_deformation,
    update_geometry_files,
)
from .structural_solver_protocol import StructuralSolverProtocol

if TYPE_CHECKING:
    from multiads.solvers.synthesis.synthesis_lib import SynthesisComponents

logger = logging.getLogger(__name__)


@dataclass
class CFDCSMConfig:
    """Configuration for CFD-CSM coupling.

    Attributes:
        tolerance: Convergence tolerance for displacement and force norms.
        max_iterations: Maximum number of coupling iterations.
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
        comp_start_h5: Component start number for HDF5.
        comp_start_geo: Component start number for geometry.
    """

    prop_node_ids: list[int] = field(default_factory=list)
    n_blades_per_prop: list[int] = field(default_factory=list)
    comp_start_h5: int = 2
    comp_start_geo: int = 2


@dataclass
class CFDCSMConvergence:
    """Coupling convergence data.

    Attributes:
        iterations: List of iteration counts.
        disp_norm: List of displacement norms.
        force_norm: List of force norms.
        converged: Whether coupling converged.
    """

    iterations: list[int] = field(default_factory=list)
    disp_norm: list[float] = field(default_factory=list)
    force_norm: list[float] = field(default_factory=list)
    converged: bool = False


class UnifiedCouplingManager:
    """Manager for unified CFD-CSM coupling iterations.

    This class orchestrates the coupled simulation loop between
    DUST (aerodynamics) and structural solvers (Lagrange, MBDyn)
    using an adapter pattern for solver abstraction.

    The manager handles:
    - Force integration via mesh_coupling.integrate_forces()
    - Structural analysis via adapter (LagrangeAdapter or MBDynAdapter)
    - Synthesis updates via synthesis_components
    - Mesh deformation via mesh_deformation.apply_structural_deformation()

    Attributes:
        boxes: Box definitions for force integration.
        paths: File paths configuration.
        structural_solver: Adapter implementing StructuralSolverProtocol.
        synthesis_components: Optional synthesis components.
        propeller_config: Propeller configuration.
        config: Coupling configuration.

    Example:
        from multiads.utilities.coupling import (
            UnifiedCouplingManager,
            LagrangeAdapter,
            CFDCSMConfig,
            CFDCSMPaths,
        )

        lagrange = LagrangeAdapter(lag_input="lagrange.inp")
        manager = UnifiedCouplingManager(
            boxes="boxes.json",
            paths=CFDCSMPaths(...),
            structural_solver=lagrange,
            config=CFDCSMConfig(tolerance=1e-3),
        )
        convergence = manager.run()
    """

    def __init__(
        self,
        boxes: list[BoxDefinition] | list[dict[str, Any]] | Path | str,
        paths: CFDCSMPaths,
        structural_solver: StructuralSolverProtocol,
        synthesis_components: SynthesisComponents | None = None,
        propeller_config: CFDCSMPropellerConfig | None = None,
        config: CFDCSMConfig | None = None,
        n_threads: int = 10,
        force_path: str = "Components/Comp001/Solution/dF",
        position_path: str = "References/Ref001/Offset",
        nodes_path: str = "Components/Comp001/Geometry/rr",
        elements_path: str = "Components/Comp001/Geometry/ee",
    ) -> None:
        """Initialize unified coupling manager.

        Args:
            boxes: Box definitions (list, dict, or path to JSON).
            paths: File paths configuration.
            structural_solver: Adapter implementing StructuralSolverProtocol.
            synthesis_components: Optional synthesis components for geometric updates.
            propeller_config: Propeller configuration.
            config: Coupling configuration.
            n_threads: Number of threads for DUST.
            force_path: HDF5 path to forces.
            position_path: HDF5 path to position.
            nodes_path: HDF5 path to nodes.
            elements_path: HDF5 path to elements.
        """
        self._load_boxes(boxes)
        self.paths = paths
        self.structural_solver = structural_solver
        self.synthesis_components = synthesis_components
        self.propeller_config = propeller_config
        self.config = config or CFDCSMConfig()
        self.n_threads = n_threads
        self.force_path = force_path
        self.position_path = position_path
        self.nodes_path = nodes_path
        self.elements_path = elements_path

        self._convergence = CFDCSMConvergence()
        self._displacements: np.ndarray | None = None
        self._forces: np.ndarray | None = None
        self._current_deformation_result: DeformationResult | None = None

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
        logger.info("Unified CFD-CSM coupling initialized")

    def _integrate_forces(self) -> ForceIntegrationResult:
        """Integrate aerodynamic forces from DUST.

        Calls utilities/coupling.mesh_coupling.integrate_forces().

        Returns:
            ForceIntegrationResult with DOFs, forces, and moments.
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

    def _run_structural_solver(
        self,
        dofs: list[list[int]],
        forces: np.ndarray,
        moments: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Run structural solver via adapter.

        Args:
            dofs: Degrees of freedom.
            forces: Applied forces.
            moments: Applied moments.

        Returns:
            Tuple of (displacements, rotations).
        """
        self.structural_solver.apply_loads(dofs, forces, moments)
        self.structural_solver.run()

        displacements = self.structural_solver.get_displacements()
        rotations = self.structural_solver.get_rotations()

        return displacements, rotations

    def _update_synthesis(
        self,
        displacements: np.ndarray,
        rotations: np.ndarray,
    ) -> None:
        """Update synthesis components with structural deformations.

        Args:
            displacements: Structural displacements.
            rotations: Structural rotations.
        """
        if self.synthesis_components is None:
            return

        for wing in self.synthesis_components.wings:
            if hasattr(wing, "wing_displ_z") and len(displacements) > 0:
                wing.wing_displ_z = displacements.tolist()

        if hasattr(self.synthesis_components, "update_displacements"):
            self.synthesis_components.update_displacements(displacements, rotations)

    def _apply_deformation(
        self,
        displacements: np.ndarray,
    ) -> DeformationResult | None:
        """Apply structural deformation to CFD mesh.

        Calls utilities/coupling.mesh_deformation.apply_structural_deformation().

        Args:
            displacements: Structural displacements.

        Returns:
            DeformationResult with deformed nodes.
        """
        deformed_nodes = apply_structural_deformation(
            displacements=displacements,
            spline_set_path=self.paths.spline_set,
            bdf_path=self.paths.bdf,
            geo_h5_path=self.paths.geo_h5,
            nodes_h5_path=self.nodes_path,
            rbf_mode=self.config.rbf_mode,
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
            disp_norm = linalg.norm(ug_new - self._displacements) / (
                linalg.norm(ug_new) + 1e-10
            )
            force_norm = linalg.norm(F_new - self._forces) / (
                linalg.norm(F_new) + 1e-10
            )

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
        """Run the unified CFD-CSM coupling loop.

        Returns:
            Convergence data.
        """
        self._setup_logging()
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

            ug, rot = self._run_structural_solver(dofs, forces, moments)

            self._update_synthesis(ug, rot)

            deformation_result = self._apply_deformation(ug)

            run_dust("dust", self.n_threads, self.paths.dust_work_dir)

            if self._check_convergence(ug, forces):
                logger.info("Coupling converged!")
                self._cleanup(deformation_result)
                break

        if self.config.plot_convergence:
            self._plot_convergence()

        self._finalize()

        return self._convergence

    def _finalize(self) -> None:
        """Finalize structural solver."""
        if hasattr(self.structural_solver, "output"):
            self.structural_solver.output()
        if hasattr(self.structural_solver, "finalize"):
            self.structural_solver.finalize()

    @property
    def convergence(self) -> CFDCSMConvergence:
        """Get convergence data.

        Returns:
            Convergence data.
        """
        return self._convergence
