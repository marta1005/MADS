"""Box definition utilities for CFD-CSM force integration.

This module provides utilities for generating force integration box
definitions from NASTRAN BDF files. Boxes are used to extract aerodynamic
loads from CFD results and transfer them to structural FEM.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from pyNastran.bdf.bdf import BDF


@dataclass
class BoxDefinition:
    """Definition of a force integration box.

    Attributes:
        box_id: Unique identifier for the box.
        ref_node: Reference node [x, y, z] for box origin.
        face_vector: Face normal direction vector.
        face_bas: Face base dimensions [front, back].
        face_hei: Face height dimensions [bottom, top].
        span_vec: Span direction vector.
        span_len: Span length.
        grid_point_id: NASTRAN grid point ID for load application.
        grid_point_coord: Grid point coordinates [x, y, z].
    """

    box_id: int
    ref_node: list[float]
    face_vector: list[float]
    face_bas: list[float]
    face_hei: list[float]
    span_vec: list[float]
    span_len: float
    grid_point_id: int
    grid_point_coord: list[float]

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary.

        Returns:
            Dictionary representation.
        """
        return {
            "box_id": self.box_id,
            "ref_node": self.ref_node,
            "face_vector": self.face_vector,
            "face_bas": self.face_bas,
            "face_hei": self.face_hei,
            "span_vec": self.span_vec,
            "span_len": self.span_len,
            "grid_point_id": self.grid_point_id,
            "grid_point_coord": self.grid_point_coord,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> BoxDefinition:
        """Create from dictionary.

        Args:
            data: Dictionary with box data.

        Returns:
            BoxDefinition instance.
        """
        return cls(
            box_id=data["box_id"],
            ref_node=list(data["ref_node"]),
            face_vector=list(data["face_vector"]),
            face_bas=list(data["face_bas"]),
            face_hei=list(data["face_hei"]),
            span_vec=list(data["span_vec"]),
            span_len=data["span_len"],
            grid_point_id=data["grid_point_id"],
            grid_point_coord=list(data["grid_point_coord"]),
        )


@dataclass
class BoxDefinitionConfig:
    """Configuration for automatic box generation.

    Attributes:
        ref_node_x: X-coordinate of reference node.
        ref_node_z: Z-coordinate of reference node.
        face_vector: Face normal direction [x, y, z].
        span_vector: Span direction [x, y, z].
        face_base: Face base dimensions [front, back].
        face_height: Face height dimensions [bottom, top].
        symmetry: Enable symmetric box generation.
    """

    ref_node_x: float = 0.0
    ref_node_z: float = 0.0
    face_vector: list[float] = field(default_factory=lambda: [1.0, 0.0, 0.0])
    span_vector: list[float] = field(default_factory=lambda: [0.0, 1.0, 0.0])
    face_base: list[float] = field(default_factory=lambda: [3.5, 3.5])
    face_height: list[float] = field(default_factory=lambda: [2.0, 2.0])
    symmetry: bool = False


class BoxDefinitionGenerator:
    """Generator for force integration boxes from BDF files.

    This class reads a NASTRAN BDF file and automatically generates
    box definitions based on the configuration and node positions.
    """

    def __init__(
        self,
        bdf_path: str | Path,
        y_positions: list[float] | None = None,
        config: BoxDefinitionConfig | None = None,
    ) -> None:
        """Initialize generator.

        Args:
            bdf_path: Path to NASTRAN BDF file.
            y_positions: Y-coordinates for box references. If None, auto-generates.
            config: Box generation configuration.
        """
        self.bdf_path = Path(bdf_path)
        self.y_positions = y_positions
        self.config = config or BoxDefinitionConfig()
        self._model: BDF | None = None
        self._nodes: dict[int, np.ndarray] = {}
        self._boxes: list[BoxDefinition] = []

    def _read_bdf(self) -> BDF:
        """Read BDF file.

        Returns:
            BDF model.
        """
        if self._model is None:
            try:
                from pyNastran.bdf.bdf import BDF
            except ImportError as e:
                raise ImportError(
                    "pyNastran is required for BDF file handling. "
                    "Install with: pip install pyNastran"
                ) from e
            self._model = BDF(debug=False, log=None)
            self._model.read_bdf(str(self.bdf_path), punch=True)
        return self._model

    def _extract_nodes(
        self,
        start_id: int,
        end_id: int,
    ) -> tuple[list[int], np.ndarray]:
        """Extract nodes within ID range.

        Args:
            start_id: Starting grid point ID.
            end_id: Ending grid point ID.

        Returns:
            Tuple of (sorted node IDs, coordinates array).
        """
        model = self._read_bdf()
        node_ids = []
        coordinates = []

        for nid, grid in model.nodes.items():
            if start_id <= nid <= end_id:
                pos = grid.get_position()
                node_ids.append(nid)
                coordinates.append(pos)

        sorted_indices = np.argsort(node_ids)[::-1]
        node_ids = [node_ids[i] for i in sorted_indices]
        coordinates = np.array(coordinates)[sorted_indices]

        return node_ids, coordinates

    def _compute_midpoints(
        self,
        y_pos: list[float],
    ) -> list[float]:
        """Compute midpoints between y-positions.

        Args:
            y_pos: List of y-positions.

        Returns:
            List of midpoints.
        """
        y_sorted = sorted(y_pos, reverse=True)
        return [(y_sorted[i] + y_sorted[i + 1]) / 2 for i in range(len(y_sorted) - 1)]

    def generate(
        self,
        start_id: int,
        end_id: int,
    ) -> list[BoxDefinition]:
        """Generate box definitions.

        Args:
            start_id: Starting grid point ID.
            end_id: Ending grid point ID.

        Returns:
            List of box definitions.
        """
        node_ids, coordinates = self._extract_nodes(start_id, end_id)

        if not self.y_positions:
            y_coords = sorted(coordinates[:, 1].tolist(), reverse=True)
            self.y_positions = y_coords

        y_pos = sorted(self.y_positions, reverse=True)
        midpoints = self._compute_midpoints(y_pos)

        boxes = []
        spans = []
        y_refs = []

        for i, y_ref_pos in enumerate(y_pos):
            if i == 0:
                y_ref = y_ref_pos + y_ref_pos / 10
                span = midpoints[0] - y_ref_pos - y_ref_pos / 10
            elif i == len(y_pos) - 1:
                y_ref = midpoints[-1]
                if self.config.symmetry:
                    span = (y_ref_pos - midpoints[-1]) * 2
                else:
                    span = y_ref_pos - midpoints[-1]
            else:
                y_ref = midpoints[i - 1]
                span = midpoints[i] - midpoints[i - 1]

            spans.append(span)
            y_refs.append(y_ref + span)

            box = BoxDefinition(
                box_id=i + 1,
                ref_node=[
                    self.config.ref_node_x,
                    round(y_ref, 5),
                    self.config.ref_node_z,
                ],
                face_vector=self.config.face_vector.copy(),
                face_bas=self.config.face_base.copy(),
                face_hei=self.config.face_height.copy(),
                span_vec=self.config.span_vector.copy(),
                span_len=round(abs(span), 5),
                grid_point_id=node_ids[i],
                grid_point_coord=[
                    coordinates[i][0] / 1000,
                    coordinates[i][1] / 1000,
                    coordinates[i][2] / 1000,
                ],
            )
            boxes.append(box)

        self._boxes = boxes
        return boxes

    def export_to_json(self, output_path: str | Path | None = None) -> str:
        """Export box definitions to JSON.

        Args:
            output_path: Output file path. If None, returns JSON string.

        Returns:
            JSON string or file path.
        """
        if not self._boxes:
            raise RuntimeError("No boxes generated. Call generate() first.")

        data = [box.to_dict() for box in self._boxes]
        json_str = json.dumps(data, indent=2)

        if output_path is not None:
            with Path(output_path).open("w") as f:
                f.write(json_str)

        return json_str

    def export_to_text(self, output_path: str | Path | None = None) -> str:
        """Export box definitions to text format.

        Args:
            output_path: Output file path. If None, returns text string.

        Returns:
            Text string or file path.
        """
        if not self._boxes:
            raise RuntimeError("No boxes generated. Call generate() first.")

        lines = []
        for box in self._boxes:
            lines.append("   {")
            for key, value in box.to_dict().items():
                lines.append(f'  "{key}": {value}, ')
            lines.append("   },\n")

        text = "\n".join(lines)

        if output_path is not None:
            with Path(output_path).open("w") as f:
                f.write(text)

        return text


def load_boxes_from_json(path: str | Path) -> list[BoxDefinition]:
    """Load box definitions from JSON file.

    Args:
        path: Path to JSON file.

    Returns:
        List of box definitions.
    """
    with Path(path).open() as f:
        data = json.load(f)
    return [BoxDefinition.from_dict(d) for d in data]


def load_boxes_from_list(boxes_data: list[dict[str, Any]]) -> list[BoxDefinition]:
    """Load box definitions from list of dictionaries.

    Args:
        boxes_data: List of box dictionaries.

    Returns:
        List of box definitions.
    """
    return [BoxDefinition.from_dict(d) for d in boxes_data]
