"""Coupling utilities for CFD-CSM aeroelastic simulations.

This module provides utilities for coupling CFD (DUST) and CSM (Lagrange
or MBDyn) solvers for aeroelastic analysis. It includes:

- Force integration from CFD to structural models (mesh_coupling)
- Mesh deformation from structural to aerodynamic models (mesh_deformation)
- Box definition helpers for force extraction (box_definition)
- Subprocess execution utilities (executor)
- Structural solver adapters (lagrange_adapter, mbdyn_adapter)
- High-level coupling orchestrator (coupling_manager, unified_coupling_manager)
- Protocol for implementing new structural solver adapters

Example usage:

    # Option 1: Use adapters directly
    from multiads.utilities.coupling import LagrangeAdapter, MBDynAdapter
    from multiads.utilities.coupling import UnifiedCouplingManager, StructuralSolverProtocol

    adapter = LagrangeAdapter(config=..., paths=...)
    manager = UnifiedCouplingManager(
        structural_solver=adapter,
        boxes=boxes,
        config=...,
    )
    result = manager.run()

    # Option 2: Use CFDCSMCouplingManager (backward compatible)
    from multiads.utilities.coupling import (
        integrate_forces,
        apply_structural_deformation,
        CFDCSMCouplingManager,
        BoxDefinition,
    )

    manager = CFDCSMCouplingManager(
        boxes="boxes.json",
        paths=CFDCSMPaths(...),
        config=CFDCSMConfig(tolerance=1e-3),
    )
    convergence = manager.run()
"""

from __future__ import annotations

from .box_definition import (
    BoxDefinition,
    BoxDefinitionConfig,
    BoxDefinitionGenerator,
    load_boxes_from_json,
    load_boxes_from_list,
)
from .coupling_manager import (
    CFDCSMConfig,
    CFDCSMConvergence,
    CFDCSMCouplingManager,
    CFDCSMPaths,
    CFDCSMPropellerConfig,
)
from .executor import (
    run_command,
    run_dust,
    run_dust_pre,
    run_lagrange,
    setup_logging,
)
from .mesh_coupling import (
    ForceBox,
    ForceIntegrationResult,
    PropellerLoadAggregator,
    compute_dof_mapping,
    compute_element_centroids,
    create_force_cards,
    create_mirrored_boxes,
    integrate_forces,
)
from .mesh_deformation import (
    DeformationResult,
    RBFDeformation,
    SectionDeformation,
    apply_rigid_transform,
    apply_structural_deformation,
    compute_section_deformation,
    handle_propeller_nodes,
    update_geometry_files,
)
from .lagrange_adapter import LagrangeAdapter
from .mbdyn_adapter import MBDynAdapter
from .structural_solver_protocol import StructuralSolverProtocol
from .unified_coupling_manager import UnifiedCouplingManager

__all__ = [
    # Box definition
    "BoxDefinition",
    "BoxDefinitionConfig",
    "BoxDefinitionGenerator",
    "load_boxes_from_json",
    "load_boxes_from_list",
    # Mesh coupling
    "ForceBox",
    "ForceIntegrationResult",
    "PropellerLoadAggregator",
    "compute_dof_mapping",
    "compute_element_centroids",
    "create_force_cards",
    "create_mirrored_boxes",
    "integrate_forces",
    # Mesh deformation
    "DeformationResult",
    "RBFDeformation",
    "SectionDeformation",
    "apply_rigid_transform",
    "apply_structural_deformation",
    "compute_section_deformation",
    "handle_propeller_nodes",
    "update_geometry_files",
    # Executor
    "run_command",
    "run_dust",
    "run_dust_pre",
    "run_lagrange",
    "setup_logging",
    # Coupling manager
    "CFDCSMConfig",
    "CFDCSMConvergence",
    "CFDCSMCouplingManager",
    "CFDCSMPaths",
    "CFDCSMPropellerConfig",
    # Structural solver adapters and protocol
    "LagrangeAdapter",
    "MBDynAdapter",
    "StructuralSolverProtocol",
    "UnifiedCouplingManager",
]
