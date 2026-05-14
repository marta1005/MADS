"""Mesh coupling utilities for CFD-CSM force integration.

This module provides utilities for integrating aerodynamic forces from
CFD (DUST) results into structural (Lagrange) loads. It handles box-based
force integration, moment calculation, and DOF mapping.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    import h5py

logger = logging.getLogger(__name__)


@dataclass
class ForceBox:
    """Force integration box.

    A box defines a region of space for extracting aerodynamic loads
    from CFD results and computing resultant forces and moments.

    Attributes:
        box_id: Unique identifier.
        ref_node: Reference node (front-bottom-outer corner).
        face_vector: Face normal vector.
        face_bas: Face base dimensions [front, back].
        face_hei: Face height dimensions [bottom, top].
        span_vec: Span direction vector.
        span_len: Span length.
        grid_point_id: Grid point ID for load application.
        grid_point_coord: Grid point coordinates.
    """

    box_id: int
    ref_node: np.ndarray
    face_vector: np.ndarray
    face_bas: np.ndarray
    face_hei: np.ndarray
    span_vec: np.ndarray
    span_len: float
    grid_point_id: int
    grid_point_coord: np.ndarray
    offset: np.ndarray = field(default_factory=lambda: np.zeros(3))

    corners: np.ndarray | None = None
    faces: np.ndarray | None = None
    cen_in_box: list[np.ndarray] = field(default_factory=list)
    for_in_box: list[np.ndarray] = field(default_factory=list)
    fractions: list[tuple[int, float]] = field(default_factory=list)
    sigma_F: np.ndarray | None = None
    sigma_M: np.ndarray | None = None

    def define_geometry(self) -> tuple[np.ndarray, np.ndarray]:
        """Define box geometry (corners and faces).

        Returns:
            Tuple of (corners, faces).
        """
        height_vec = np.cross(self.face_vector, self.span_vec)

        if self.ref_node[1] < 0:
            self.corners = np.array(
                [
                    self.ref_node,
                    self.ref_node + self.face_bas[0] * self.face_vector,
                    self.ref_node
                    + self.face_bas[0] * self.face_vector
                    + self.span_len * self.span_vec,
                    self.ref_node + self.span_len * self.span_vec,
                    self.ref_node
                    + self.span_len * self.span_vec
                    + self.face_hei[1] * height_vec,
                    self.ref_node
                    + self.span_len * self.span_vec
                    + self.face_hei[1] * height_vec
                    + self.face_bas[1] * self.face_vector,
                    self.ref_node
                    + self.face_hei[0] * height_vec
                    + self.face_bas[0] * self.face_vector,
                    self.ref_node + self.face_hei[0] * height_vec,
                ]
            )
        else:
            self.corners = np.array(
                [
                    self.ref_node,
                    self.ref_node + self.face_bas[0] * self.face_vector,
                    self.ref_node
                    + self.face_bas[0] * self.face_vector
                    - self.span_len * self.span_vec,
                    self.ref_node - self.span_len * self.span_vec,
                    self.ref_node
                    - self.span_len * self.span_vec
                    + self.face_hei[1] * height_vec,
                    self.ref_node
                    - self.span_len * self.span_vec
                    + self.face_hei[1] * height_vec
                    + self.face_bas[1] * self.face_vector,
                    self.ref_node
                    + self.face_hei[0] * height_vec
                    + self.face_bas[0] * self.face_vector,
                    self.ref_node + self.face_hei[0] * height_vec,
                ]
            )

        self.faces = np.array(
            [
                [self.corners[0], self.corners[1], self.corners[6], self.corners[7]],
                [self.corners[1], self.corners[2], self.corners[5], self.corners[6]],
                [self.corners[3], self.corners[2], self.corners[5], self.corners[4]],
                [self.corners[0], self.corners[3], self.corners[4], self.corners[7]],
                [self.corners[4], self.corners[7], self.corners[6], self.corners[5]],
                [self.corners[0], self.corners[1], self.corners[2], self.corners[3]],
            ]
        )

        return self.corners, self.faces

    def find_elements_in_box(
        self,
        elements: np.ndarray,
        nodes: np.ndarray,
        forces: np.ndarray,
        centroids: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Find elements within the box and compute forces.

        Args:
            elements: Element connectivity array.
            nodes: Node coordinates array.
            forces: Force array per element.
            centroids: Element centroid coordinates.

        Returns:
            Tuple of (centroids_in_box, forces_in_box).
        """
        self.cen_in_box = []
        self.for_in_box = []
        self.fractions = []

        min_y_box = np.min(self.corners[:, 1])
        max_y_box = np.max(self.corners[:, 1])

        for i, elem in enumerate(elements):
            el_nodes = nodes[elem - 1]
            min_y = np.min(el_nodes[:, 1])
            max_y = np.max(el_nodes[:, 1])
            el_width = max_y - min_y

            if min_y >= min_y_box and max_y <= max_y_box:
                fraction = 1.0
            elif max_y < min_y_box or min_y > max_y_box:
                fraction = 0.0
            else:
                inter_start = max(min_y, min_y_box)
                inter_end = min(max_y, max_y_box)
                inter_length = inter_end - inter_start
                fraction = inter_length / el_width

            self.fractions.append((i, fraction))
            force = forces[i] * fraction

            if not np.allclose(force, 0):
                self.cen_in_box.append(centroids[i])
                self.for_in_box.append(force)

        return np.array(self.cen_in_box), np.array(self.for_in_box)

    def compute_resultants(self) -> tuple[np.ndarray, np.ndarray]:
        """Compute total forces and moments.

        Returns:
            Tuple of (total_forces, total_moments).
        """
        if not self.for_in_box:
            self.sigma_F = np.zeros(3)
            self.sigma_M = np.zeros(3)
            return self.sigma_F, self.sigma_M

        forces = np.array(self.for_in_box)
        centers = np.array(self.cen_in_box)

        self.sigma_F = np.sum(forces, axis=0)

        x_app = self.grid_point_coord[0]
        y_app = self.grid_point_coord[1]
        z_app = self.grid_point_coord[2]

        arms = np.zeros((len(centers), 3))
        for i, center in enumerate(centers):
            dx = (center[0] - x_app) * 1000
            dy = (center[1] - y_app) * 1000
            dz = (center[2] - z_app) * 1000
            arms[i] = [dx, dy, dz]

        moments = np.cross(arms, forces)
        self.sigma_M = np.sum(moments, axis=0)

        return self.sigma_F, self.sigma_M


@dataclass
class PropellerLoadAggregator:
    """Aggregates loads from multiple propeller blades.

    This class handles time-averaging of loads for rotating propellers.
    """

    def __init__(
        self,
        prop_node_ids: list[int] | None = None,
        n_blades_per_prop: list[int] | None = None,
        comp_start: int = 2,
    ) -> None:
        """Initialize aggregator.

        Args:
            prop_node_ids: Grid point IDs for propellers.
            n_blades_per_prop: Number of blades per propeller.
            comp_start: Starting component number.
        """
        self.prop_node_ids = prop_node_ids or []
        self.n_blades_per_prop = n_blades_per_prop or []
        self.comp_start = comp_start

    def aggregate_blade_loads(
        self,
        h5_path: Path | str,
        result_files: list[str],
        force_path: str,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Aggregate loads from all propeller blades.

        Args:
            h5_path: Path to HDF5 file directory.
            result_files: List of HDF5 result file names.
            force_path: HDF5 path to force data.

        Returns:
            Tuple of (averaged_forces, averaged_moments).
        """
        import h5py

        all_forces = []
        all_moments = []

        for filename in result_files:
            filepath = Path(h5_path) / filename
            if not filepath.exists():
                continue

            with h5py.File(filepath, "r") as h5file:
                forces = np.array(h5file[force_path])
                all_forces.append(forces)

        if not all_forces:
            return np.zeros(3), np.zeros(3)

        return np.mean(all_forces, axis=0), np.zeros(3)


def compute_element_centroids(
    elements: np.ndarray,
    nodes: np.ndarray,
) -> np.ndarray:
    """Compute centroids of elements.

    Args:
        elements: Element connectivity array.
        nodes: Node coordinates array.

    Returns:
        Array of centroids.
    """
    centroids = np.zeros((elements.shape[0], 3))
    for i, elem in enumerate(elements):
        el_nodes = nodes[elem - 1]
        centroids[i] = np.mean(el_nodes, axis=0)
    return centroids


def create_force_cards(
    boxes: list[ForceBox],
    card_id: int = 100000,
) -> str:
    """Create BDF force and moment cards.

    Args:
        boxes: List of force boxes.
        card_id: Card ID for loads.

    Returns:
        BDF card string.
    """
    lines = []
    for box in boxes:
        if box.sigma_F is None or box.sigma_M is None:
            continue

        for direction, value in enumerate(box.sigma_F):
            if abs(value) > 1e-10:
                direction_vec = [0, 0, 0]
                direction_vec[direction] = 1.0
                lines.append(
                    f"FORCE   {card_id}   {box.grid_point_id}   {value:.6f}   "
                    f"{direction_vec[0]}   {direction_vec[1]}   {direction_vec[2]}\n"
                )

        for direction, value in enumerate(box.sigma_M):
            if abs(value) > 1e-10:
                direction_vec = [0, 0, 0]
                direction_vec[direction] = 1.0
                lines.append(
                    f"MOMENT  {card_id}   {box.grid_point_id}   {value:.6f}   "
                    f"{direction_vec[0]}   {direction_vec[1]}   {direction_vec[2]}\n"
                )

    return "".join(lines)


def compute_dof_mapping(
    bdf_path: Path | str,
    boxes: list[ForceBox],
) -> list[list[int]]:
    """Compute DOF mapping from grid points.

    Args:
        bdf_path: Path to NASTRAN BDF file.
        boxes: List of force boxes.

    Returns:
        List of DOF indices for each box.
    """
    try:
        from pyNastran.bdf.bdf import BDF
    except ImportError as e:
        raise ImportError("pyNastran is required for BDF file handling.") from e

    model = BDF(debug=False, log=None)
    model.read_bdf(str(bdf_path), punch=True)
    grid_points = list(model.nodes.keys())

    dofs = []
    for box in boxes:
        try:
            index = grid_points.index(box.grid_point_id)
            dof_start = index * 6 + 1
            dof = list(range(dof_start, dof_start + 6))
            dofs.append(dof)
        except ValueError:
            logger.warning(f"Grid point {box.grid_point_id} not found in BDF")
            dofs.append([0] * 6)

    return dofs


def create_mirrored_boxes(
    boxes: list[ForceBox],
    symmetry_id_offset: int = 1000001,
) -> list[ForceBox]:
    """Create symmetric boxes for full aircraft analysis.

    Args:
        boxes: Original box list.
        symmetry_id_offset: Offset for mirrored box IDs.

    Returns:
        List including original and mirrored boxes.
    """
    mirrored = []
    for box in boxes:
        if np.sign(box.ref_node[1]) == np.sign(box.ref_node[1] + box.span_len):
            continue

        mirrored_box = ForceBox(
            box_id=-box.box_id,
            ref_node=box.ref_node * np.array([1, -1, 1]),
            face_vector=box.face_vector.copy(),
            face_bas=box.face_bas.copy(),
            face_hei=box.face_hei.copy(),
            span_vec=box.span_vec.copy(),
            span_len=box.span_len,
            grid_point_id=box.grid_point_id - symmetry_id_offset,
            grid_point_coord=box.grid_point_coord * np.array([1, -1, 1]),
            offset=box.offset,
        )
        mirrored.append(mirrored_box)

    return mirrored


@dataclass
class ForceIntegrationResult:
    """Result of force integration.

    Attributes:
        dofs: Degrees of freedom mapping.
        forces: Total forces per box.
        moments: Total moments per box.
        prop_index: Propeller indices (if applicable).
    """

    dofs: list[list[int]]
    forces: np.ndarray
    moments: np.ndarray
    prop_index: list[int] | None = None


def integrate_forces(
    boxes_data: list[dict[str, Any]],
    res_h5_path: Path | str,
    geo_h5_path: Path | str,
    force_path: str,
    position_path: str,
    nodes_path: str,
    elements_path: str,
    bdf_path: Path | str | None = None,
    symmetry: bool = False,
    prop_node_ids: list[int] | None = None,
    n_blades_per_prop: list[int] | None = None,
    comp_start_h5: int | None = None,
    comp_start_geo: int | None = None,
) -> ForceIntegrationResult:
    """Integrate aerodynamic forces from CFD to structural model.

    This is the main function for DUST-to-Lagrange force transfer.

    Args:
        boxes_data: List of box definition dictionaries.
        res_h5_path: Path to DUST results HDF5 file.
        geo_h5_path: Path to DUST geometry HDF5 file.
        force_path: HDF5 path to force data.
        position_path: HDF5 path to position/offset data.
        nodes_path: HDF5 path to node coordinates.
        elements_path: HDF5 path to element connectivity.
        bdf_path: Path to NASTRAN BDF file.
        symmetry: Enable symmetric box generation.
        prop_node_ids: Propeller grid point IDs.
        n_blades_per_prop: Blades per propeller.
        comp_start_h5: Component start number for HDF5.
        comp_start_geo: Component start number for geometry.

    Returns:
        ForceIntegrationResult with DOFs, forces, and moments.
    """
    import h5py

    with (
        h5py.File(res_h5_path, "r") as res_file,
        h5py.File(geo_h5_path, "r") as geo_file,
    ):
        forces = np.array(res_file[force_path])
        offset = np.array(res_file[position_path])
        nodes = np.array(geo_file[nodes_path])
        elements = np.array(geo_file[elements_path])

    centroids = compute_element_centroids(elements, nodes)

    boxes = []
    for data in boxes_data:
        box = ForceBox(
            box_id=data["box_id"],
            ref_node=np.array(data["ref_node"]) - offset,
            face_vector=np.array(data["face_vector"]),
            face_bas=np.array(data["face_bas"]),
            face_hei=np.array(data["face_hei"]),
            span_vec=np.array(data["span_vec"]),
            span_len=data["span_len"],
            grid_point_id=data["grid_point_id"],
            grid_point_coord=np.array(data["grid_point_coord"]) - offset,
            offset=offset,
        )
        boxes.append(box)

    if symmetry:
        mirrored = create_mirrored_boxes(boxes)
        boxes.extend(mirrored)

    total_forces = []
    total_moments = []

    for box in boxes:
        box.define_geometry()
        box.find_elements_in_box(elements, nodes, forces, centroids)
        sigma_F, sigma_M = box.compute_resultants()
        total_forces.append(sigma_F)
        total_moments.append(sigma_M)

    dofs = []
    if bdf_path is not None:
        dofs = compute_dof_mapping(bdf_path, boxes)

    prop_index = None
    if prop_node_ids and bdf_path is not None:
        model_dofs = compute_dof_mapping(bdf_path, boxes)
        prop_index = []
        for node_id in prop_node_ids:
            for i, box in enumerate(boxes):
                if box.grid_point_id == node_id:
                    if i < len(model_dofs):
                        prop_index.append(i)
                    break

    return ForceIntegrationResult(
        dofs=dofs,
        forces=np.array(total_forces),
        moments=np.array(total_moments),
        prop_index=prop_index,
    )
