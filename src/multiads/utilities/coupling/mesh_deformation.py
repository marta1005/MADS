"""Mesh deformation utilities for CFD-CSM coupling.

This module provides utilities for transferring structural deformations
from Lagrange (structural) to DUST (CFD) mesh. It supports both section-based
rigid transformations and RBF-based mesh deformation.
"""

from __future__ import annotations

import logging
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from scipy.interpolate import RBFInterpolator

logger = logging.getLogger(__name__)


@dataclass
class SectionDeformation:
    """Section-based deformation data.

    Attributes:
        dehydral: Dihedral angle change.
        theta: Twist angle change.
        span: Span change.
    """

    dehydral: float
    theta: float
    span: float


class RBFDeformation:
    """RBF-based mesh deformation.

    Uses Radial Basis Function interpolation to deform CFD mesh
    based on structural displacements.
    """

    def __init__(
        self,
        kernel: str = "thin_plate_spline",
        smoothing: float = 0.0,
        degree: int = 1,
    ) -> None:
        """Initialize RBF deformation.

        Args:
            kernel: RBF kernel type.
            smoothing: Smoothing parameter.
            degree: Polynomial degree.
        """
        self.kernel = kernel
        self.smoothing = smoothing
        self.degree = degree
        self._interpolator: RBFInterpolator | None = None
        self._source_points: np.ndarray | None = None

    def fit(
        self,
        source_points: np.ndarray,
        displacements: np.ndarray,
    ) -> None:
        """Fit the RBF interpolator.

        Args:
            source_points: Source node positions (N, 3).
            displacements: Displacements at source points (N, 3).
        """
        from scipy.interpolate import RBFInterpolator

        self._source_points = source_points.copy()
        degree = -1 if self.degree == 1 else self.degree
        self._interpolator = RBFInterpolator(
            source_points,
            displacements,
            kernel=self.kernel,
            smoothing=self.smoothing,
            degree=degree,
        )
        logger.info(f"Fitted RBF with {len(source_points)} source points")

    def transform(self, target_points: np.ndarray) -> np.ndarray:
        """Transform target points.

        Args:
            target_points: Target node positions (M, 3).

        Returns:
            Displaced target positions (M, 3).
        """
        if self._interpolator is None:
            raise RuntimeError("RBF interpolator not fitted. Call fit() first.")

        displacements = self._interpolator(target_points)
        return target_points + displacements

    def fit_transform(
        self,
        source_points: np.ndarray,
        displacements: np.ndarray,
        target_points: np.ndarray,
    ) -> np.ndarray:
        """Fit and transform in one step.

        Args:
            source_points: Source node positions (N, 3).
            displacements: Displacements at source points (N, 3).
            target_points: Target node positions (M, 3).

        Returns:
            Displaced target positions (M, 3).
        """
        self.fit(source_points, displacements)
        return self.transform(target_points)


def compute_section_deformation(
    displacements: np.ndarray,
    span_positions: np.ndarray,
) -> SectionDeformation:
    """Compute section-based rigid deformation parameters.

    Args:
        displacements: Node displacements (N, 3).
        span_positions: Span positions along wing (N,).

    Returns:
        SectionDeformation with rigid transformation parameters.
    """
    z_disp = displacements[:, 2]
    dz_dy = np.gradient(z_disp, span_positions)
    dehydral = np.arctan(np.mean(dz_dy))

    dy_disp = displacements[:, 1]
    dy_dy = np.gradient(dy_disp, span_positions)
    theta = np.arctan(np.mean(dy_dy))

    span = np.max(span_positions) - np.min(span_positions)

    return SectionDeformation(
        dehydral=dehydral,
        theta=theta,
        span=span,
    )


def apply_rigid_transform(
    nodes: np.ndarray,
    section_data: SectionDeformation,
    axis: str = "y",
) -> np.ndarray:
    """Apply rigid transformation to nodes.

    Args:
        nodes: Node positions (N, 3).
        section_data: Section deformation parameters.
        axis: Primary axis of transformation.

    Returns:
        Transformed node positions.
    """
    transformed = nodes.copy()

    angle_idx = {"x": 0, "y": 1, "z": 2}.get(axis, 1)

    cos_t = np.cos(section_data.theta)
    sin_t = np.sin(section_data.theta)
    cos_d = np.cos(section_data.dehydral)
    sin_d = np.sin(section_data.dehydral)

    if angle_idx == 0:
        rot_matrix = np.array(
            [
                [1, 0, 0],
                [0, cos_t, -sin_t],
                [0, sin_t, cos_t],
            ]
        )
    elif angle_idx == 1:
        rot_matrix = np.array(
            [
                [cos_d, 0, sin_d],
                [0, 1, 0],
                [-sin_d, 0, cos_d],
            ]
        )
    else:
        rot_matrix = np.eye(3)

    transformed = (rot_matrix @ transformed.T).T
    transformed[:, angle_idx] += section_data.span

    return transformed


def apply_structural_deformation(
    displacements: np.ndarray,
    spline_set_path: Path | str | None = None,
    bdf_path: Path | str | None = None,
    geo_h5_path: Path | str | None = None,
    nodes_h5_path: str = "Components/Comp001/Geometry/rr",
    rbf_mode: bool = True,
    propeller_nodes: list[int] | None = None,
) -> np.ndarray | None:
    """Apply structural deformations to CFD mesh.

    This is the main function for Lagrange-to-DUST deformation transfer.

    Args:
        displacements: Structural displacements from Lagrange.
        spline_set_path: Path to spline set file.
        bdf_path: Path to NASTRAN BDF file.
        geo_h5_path: Path to geometry HDF5 file.
        nodes_h5_path: HDF5 path to node coordinates.
        rbf_mode: Use RBF interpolation (True) or section-based (False).
        propeller_nodes: Propeller node indices.

    Returns:
        Deformed node positions or None.
    """
    if bdf_path is None or geo_h5_path is None:
        logger.warning("BDF or geometry path not provided")
        return None

    import h5py

    with h5py.File(geo_h5_path, "r") as h5file:
        original_nodes = np.array(h5file[nodes_h5_path])

    spline_nodes = _read_spline_nodes(spline_set_path, bdf_path)

    if spline_nodes is None or len(spline_nodes) == 0:
        logger.warning("Could not read spline nodes")
        return original_nodes

    if len(displacements) < len(spline_nodes):
        logger.warning(
            f"Displacement vector ({len(displacements)}) shorter than "
            f"spline nodes ({len(spline_nodes)})"
        )
        disp_aligned = np.zeros((len(spline_nodes), 3))
        disp_aligned[: len(displacements)] = displacements[: len(spline_nodes)]
    else:
        disp_aligned = displacements[: len(spline_nodes)]

    if rbf_mode:
        rbf = RBFDeformation(kernel="thin_plate_spline")
        deformed_nodes = rbf.fit_transform(
            spline_nodes,
            disp_aligned,
            original_nodes,
        )
    else:
        span_positions = spline_nodes[:, 1]
        section_data = compute_section_deformation(disp_aligned, span_positions)
        deformed_nodes = apply_rigid_transform(original_nodes, section_data)

    return deformed_nodes


def _read_spline_nodes(
    spline_set_path: Path | str | None,
    bdf_path: Path | str,
) -> np.ndarray | None:
    """Read spline nodes from spline set or BDF file.

    Args:
        spline_set_path: Path to spline set file.
        bdf_path: Path to BDF file.

    Returns:
        Spline node positions.
    """
    try:
        from pyNastran.bdf.bdf import BDF
    except ImportError:
        logger.warning("pyNastran not available")
        return None

    if spline_set_path and Path(spline_set_path).exists():
        model = BDF(debug=False, log=None)
        model.read_bdf(str(spline_set_path), punch=True)
    else:
        model = BDF(debug=False, log=None)
        model.read_bdf(str(bdf_path), punch=True)

    nodes = []
    for nid, grid in model.nodes.items():
        pos = grid.get_position()
        nodes.append(pos)

    if not nodes:
        return None

    return np.array(nodes)


def handle_propeller_nodes(
    original_geo_path: Path | str,
    deformed_nodes: np.ndarray,
    prop_indices: list[int],
    hub_positions: np.ndarray,
    rotation_matrices: np.ndarray,
) -> np.ndarray:
    """Handle propeller node deformation.

    Propeller nodes need special handling as they rotate around hub.

    Args:
        original_geo_path: Path to original geometry HDF5.
        deformed_nodes: Deformed node positions.
        prop_indices: Propeller node indices.
        hub_positions: Hub positions.
        rotation_matrices: Rotation matrices for each propeller.

    Returns:
        Updated deformed nodes.
    """
    import h5py

    with h5py.File(original_geo_path, "r") as h5file:
        original_nodes = np.array(h5file["Components/Comp001/Geometry/rr"])

    result = deformed_nodes.copy()

    for i, prop_idx in enumerate(prop_indices):
        hub = hub_positions[i]
        R = rotation_matrices[i]
        R_global2prop = R.T

        local_nodes = original_nodes[prop_idx] - hub
        rotated_local = (R_global2prop @ local_nodes.T).T
        result[prop_idx] = rotated_local + hub

    return result


@dataclass
class DeformationResult:
    """Result of mesh deformation.

    Attributes:
        deformed_nodes: Deformed node positions.
        copied_geo_h5: Path to copied geometry file.
        copied_in_file: Path to copied input file.
        displacements: Applied displacements.
    """

    deformed_nodes: np.ndarray | None
    copied_geo_h5: Path | None = None
    copied_in_file: Path | None = None
    displacements: np.ndarray | None = None


def update_geometry_files(
    deformed_nodes: np.ndarray,
    geo_h5_path: Path | str,
    geo_in_path: Path | str,
    output_dir: Path | str | None = None,
) -> DeformationResult:
    """Update geometry files with deformed mesh.

    Args:
        deformed_nodes: Deformed node positions.
        geo_h5_path: Path to geometry HDF5 file.
        geo_in_path: Path to geometry input file.
        output_dir: Output directory for copied files.

    Returns:
        DeformationResult with updated file paths.
    """
    import h5py

    output_dir = Path(output_dir) if output_dir else Path(".")

    copied_geo_h5 = output_dir / f"copied_{Path(geo_h5_path).name}"
    shutil.copy(geo_h5_path, copied_geo_h5)

    with h5py.File(copied_geo_h5, "r+") as h5file:
        if "Components/Comp001/Geometry/rr" in h5file:
            del h5file["Components/Comp001/Geometry/rr"]
            h5file["Components/Comp001/Geometry/rr"] = deformed_nodes

    copied_in_file = None
    if Path(geo_in_path).exists():
        copied_in_file = output_dir / f"copied_{Path(geo_in_path).name}"
        shutil.copy(geo_in_path, copied_in_file)

    return DeformationResult(
        deformed_nodes=deformed_nodes,
        copied_geo_h5=copied_geo_h5,
        copied_in_file=copied_in_file,
        displacements=deformed_nodes,
    )
