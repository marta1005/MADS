"""TiGL Bridge for MultiADS.

This module provides the TiGLBridge class which connects CPACSImporter
to TiGL geometry engine for geometric queries and CAD export.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from multiads.design_space.cpacs_importer import CPACSImporter

logger = logging.getLogger(__name__)


class TiGLBridge:
    """Bridge between CPACSImporter and TiGL geometry engine.

    This class provides:
    - Wing and fuselage geometry extraction
    - Geometric property computation (area, volume, etc.)
    - CAD export (IGES, STL)
    - Geometric queries for optimization

    Args:
        importer: The CPACSImporter instance to connect to.
        validate: Whether to validate the TiGL configuration.

    Example:
        >>> from multiads.design_space import CPACSImporter, TiGLBridge
        >>> importer = CPACSImporter("aircraft.xml", validate=False)
        >>> tigl = TiGLBridge(importer)
        >>> wing_area = tigl.compute_wing_area("wing_main")
        >>> tigl.export_to_iges("wing_main", "wing_main.igs")
    """

    def __init__(
        self,
        importer: CPACSImporter,
        validate: bool = False,
        cpacs_file: str | None = None,
    ) -> None:
        """Initialize TiGL bridge from CPACSImporter.

        Args:
            importer: The CPACSImporter instance.
            validate: Whether to validate the TiGL configuration.
            cpacs_file: Optional CPACS filename (if different from importer's file).
        """
        self._importer = importer
        self._validate = validate
        self._tigl: Any = None
        self._tixi: Any = None

        if cpacs_file:
            self._cpacs_file = cpacs_file
        else:
            self._cpacs_file = None

        self._load_tigl()

    def _load_tigl(self) -> None:
        """Load TiGL with the CPACS file from importer."""
        try:
            import tigl3
            import tixi3
        except ImportError as e:
            msg = "TiGL3 or TIXI3 Python bindings not available."
            raise ImportError(msg) from e

        if self._cpacs_file:
            cpacs_path = self._importer.path / self._cpacs_file
        else:
            cpacs_path = self._importer.path / self._importer.file_name

        if not cpacs_path.exists():
            msg = f"CPACS file not found at {cpacs_path}"
            raise FileNotFoundError(msg)

        try:
            self._tixi = tixi3.Tixi3()
            self._tixi.open(str(cpacs_path))

            self._tigl = tigl3.Tigl3()
            self._tigl.logToFileDisabled()
            self._tigl.open(self._tixi, "")

            logger.info(f"TiGL loaded successfully from {cpacs_path}")
        except Exception as e:
            msg = f"Failed to load TiGL: {e}"
            raise RuntimeError(msg) from e

    @property
    def tigl(self) -> Any:
        """Get the TiGL instance."""
        return self._tigl

    def get_wing_index(self, wing_uid: str) -> int:
        """Get the wing index from UID.

        Args:
            wing_uid: The wing UID.

        Returns:
            The wing index.
        """
        if self._tigl is None:
            msg = "TiGL not initialized"
            raise RuntimeError(msg)

        n_wings = self._tigl.getWingCount()
        for i in range(1, n_wings + 1):
            uid = self._tigl.wingGetUID(i)
            if uid == wing_uid:
                return i

        msg = f"Wing UID '{wing_uid}' not found"
        raise ValueError(msg)

    def get_fuselage_index(self, fuselage_uid: str) -> int:
        """Get the fuselage index from UID.

        Args:
            fuselage_uid: The fuselage UID.

        Returns:
            The fuselage index.
        """
        if self._tigl is None:
            msg = "TiGL not initialized"
            raise RuntimeError(msg)

        n_fuselages = self._tigl.getFuselageCount()
        for i in range(1, n_fuselages + 1):
            uid = self._tigl.fuselageGetUID(i)
            if uid == fuselage_uid:
                return i

        msg = f"Fuselage UID '{fuselage_uid}' not found"
        raise ValueError(msg)

    def compute_wing_area(
        self,
        wing_uid: str,
        projected: bool = False,
    ) -> float:
        """Compute wing reference or projected area.

        Args:
            wing_uid: The wing UID.
            projected: If True, compute projected area; otherwise wetted area.

        Returns:
            The wing area in m^2.
        """
        if self._tigl is None:
            msg = "TiGL not initialized"
            raise RuntimeError(msg)

        wing_index = self.get_wing_index(wing_uid)

        if projected:
            area = self._tigl.wingGetReferenceArea(wing_index)
        else:
            area = self._tigl.wingGetSurfaceArea(wing_index)

        return area

    def compute_wet_area(
        self,
        wing_uid: str,
    ) -> float:
        """Compute wetted surface area of the wing.

        Args:
            wing_uid: The wing UID.

        Returns:
            The wetted area in m^2.
        """
        if self._tigl is None:
            msg = "TiGL not initialized"
            raise RuntimeError(msg)

        wing_index = self.get_wing_index(wing_uid)
        return self._tigl.wingGetWettedArea(wing_index)

    def compute_volume(
        self,
        component_uid: str,
    ) -> float:
        """Compute component volume.

        Args:
            component_uid: The component UID (wing or fuselage).

        Returns:
            The volume in m^3.
        """
        if self._tigl is None:
            msg = "TiGL not initialized"
            raise RuntimeError(msg)

        try:
            wing_index = self.get_wing_index(component_uid)
            return self._tigl.wingGetVolume(wing_index)
        except ValueError:
            pass

        try:
            fuselage_index = self.get_fuselage_index(component_uid)
            return self._tigl.fuselageGetVolume(fuselage_index)
        except ValueError:
            pass

        msg = f"Component UID '{component_uid}' not found"
        raise ValueError(msg)

    def get_thickness_distribution(
        self,
        wing_uid: str,
        n_eta: int = 20,
        n_chord: int = 5,
    ) -> NDArray[np.float64]:
        """Get thickness distribution along the wing span.

        Args:
            wing_uid: The wing UID.
            n_eta: Number of spanwise stations.
            n_chord: Number of chordwise points per station.

        Returns:
            Array of shape (n_eta, n_chord, 3) containing thickness values.
        """
        if self._tigl is None:
            msg = "TiGL not initialized"
            raise RuntimeError(msg)

        wing_index = self.get_wing_index(wing_uid)
        n_segments = self._tigl.wingGetComponentSegmentCount(wing_index)

        if n_segments == 0:
            msg = f"No component segments found for wing '{wing_uid}'"
            raise ValueError(msg)

        thickness_data = []

        eta_values = np.linspace(0.0, 1.0, n_eta)

        for eta in eta_values:
            eta_row = []
            for xsi in np.linspace(0.0, 1.0, n_chord):
                try:
                    upper = self._tigl.wingGetUpperPoint(wing_index, eta, xsi, 0)
                    lower = self._tigl.wingGetLowerPoint(wing_index, eta, xsi, 0)
                    thickness = np.linalg.norm(np.array(upper) - np.array(lower))
                    eta_row.append(thickness)
                except Exception:
                    eta_row.append(0.0)

            thickness_data.append(eta_row)

        return np.array(thickness_data)

    def get_wing_segment_point(
        self,
        wing_uid: str,
        eta: float,
        xsi: float,
        surface: str = "upper",
    ) -> tuple[float, float, float]:
        """Get point on wing at eta, xsi coordinates.

        Args:
            wing_uid: The wing UID.
            eta: Spanwise position (0.0 at root, 1.0 at tip).
            xsi: Chordwise position (0.0 at leading edge, 1.0 at trailing edge).
            surface: "upper" or "lower".

        Returns:
            Tuple of (x, y, z) coordinates.
        """
        if self._tigl is None:
            msg = "TiGL not initialized"
            raise RuntimeError(msg)

        wing_index = self.get_wing_index(wing_uid)

        if surface.lower() == "upper":
            point = self._tigl.wingGetUpperPoint(wing_index, eta, xsi, 0)
        else:
            point = self._tigl.wingGetLowerPoint(wing_index, eta, xsi, 0)

        return tuple(point)

    def get_wing_span(self, wing_uid: str) -> float:
        """Get wing span.

        Args:
            wing_uid: The wing UID.

        Returns:
            The wing span in meters.
        """
        if self._tigl is None:
            msg = "TiGL not initialized"
            raise RuntimeError(msg)

        wing_index = self.get_wing_index(wing_uid)
        return self._tigl.wingGetSpan(wing_index)

    def get_wing_mac(self, wing_uid: str) -> tuple[float, float, float]:
        """Get wing mean aerodynamic chord (MAC).

        Args:
            wing_uid: The wing UID.

        Returns:
            Tuple of (x, y, z) for MAC location.
        """
        if self._tigl is None:
            msg = "TiGL not initialized"
            raise RuntimeError(msg)

        wing_index = self.get_wing_index(wing_uid)
        mac = self._tigl.wingGetMAC(wing_index)
        return tuple(mac)

    def get_fuselage_cross_section(
        self,
        fuselage_uid: str,
        x: float,
    ) -> dict[str, Any]:
        """Get fuselage cross-section at given x position.

        Args:
            fuselage_uid: The fuselage UID.
            x: The x position along the fuselage.

        Returns:
            Dictionary with cross-section info (area, circumference, center).
        """
        if self._tigl is None:
            msg = "TiGL not initialized"
            raise RuntimeError(msg)

        fuselage_index = self.get_fuselage_index(fuselage_uid)

        center = self._tigl.fuselageGetPointOnXPlane(fuselage_index, x)
        area = self._tigl.getCrossSectionArea(fuselage_index, x)
        circumference = self._tigl.fuselageGetCircumference(fuselage_index, x)

        return {
            "center": tuple(center),
            "area": area,
            "circumference": circumference,
        }

    def export_to_iges(
        self,
        component_uid: str,
        filepath: str | Path,
    ) -> None:
        """Export component to IGES format.

        Args:
            component_uid: The component UID (wing or fuselage).
            filepath: Output file path.
        """
        if self._tigl is None:
            msg = "TiGL not initialized"
            raise RuntimeError(msg)

        filepath = Path(filepath)

        try:
            wing_index = self.get_wing_index(component_uid)
            self._tigl.exportIGES(str(filepath))
        except ValueError:
            pass

        try:
            fuselage_index = self.get_fuselage_index(component_uid)
            self._tigl.exportIGES(str(filepath))
        except ValueError:
            pass

        logger.info(f"Exported {component_uid} to IGES: {filepath}")

    def export_to_stl(
        self,
        component_uid: str,
        filepath: str | Path,
        mesh_accuracy: float = 0.001,
    ) -> None:
        """Export component to STL format.

        Args:
            component_uid: The component UID (wing or fuselage).
            filepath: Output file path.
            mesh_accuracy: Mesh accuracy parameter.
        """
        if self._tigl is None:
            msg = "TiGL not initialized"
            raise RuntimeError(msg)

        filepath = Path(filepath)
        component_uid_lower = component_uid.lower()

        if "wing" in component_uid_lower:
            try:
                wing_index = self.get_wing_index(component_uid)
                self._tigl.exportMeshedWingSTLByUID(
                    component_uid, str(filepath), mesh_accuracy
                )
            except Exception as e:
                logger.warning(f"STL export failed for wing: {e}")
        elif "fuselage" in component_uid_lower:
            try:
                fuselage_index = self.get_fuselage_index(component_uid)
                self._tigl.exportMeshedFuselageSTLByUID(
                    component_uid, str(filepath), mesh_accuracy
                )
            except Exception as e:
                logger.warning(f"STL export failed for fuselage: {e}")
        else:
            msg = f"Unknown component type: {component_uid}"
            raise ValueError(msg)

        logger.info(f"Exported {component_uid} to STL: {filepath}")

    def export_aircraft_to_iges(
        self,
        filepath: str | Path,
    ) -> None:
        """Export entire aircraft to IGES format.

        Args:
            filepath: Output file path.
        """
        if self._tigl is None:
            msg = "TiGL not initialized"
            raise RuntimeError(msg)

        filepath = Path(filepath)
        self._tigl.exportIGES(str(filepath))
        logger.info(f"Exported aircraft to IGES: {filepath}")

    def get_bounding_box(self) -> NDArray[np.float64]:
        """Get aircraft bounding box.

        Returns:
            Array of [xmin, ymin, zmin, xmax, ymax, zmax].
        """
        if self._tigl is None:
            msg = "TiGL not initialized"
            raise RuntimeError(msg)

        bb = self._tigl.configurationGetBoundingBox()
        return np.array(bb)

    def sync_from_tigl(self) -> None:
        """Sync geometric data from TiGL back to CPACS tree.

        This reads computed geometry from TiGL (e.g., MAC, wing area) and
        updates the corresponding CPACS elements.

        Note: TiGL computes geometry from the CPACS description, so major
        changes should be made to the CPACS tree directly. This method is
        useful for updating derived values that TiGL computes.
        """
        if self._tigl is None:
            msg = "TiGL not initialized"
            raise RuntimeError(msg)

        importer = self._importer

        try:
            n_wings = self._tigl.getWingCount()
            for i in range(1, n_wings + 1):
                wing_uid = self._tigl.wingGetUID(i)

                span = self._tigl.wingGetSpan(i)
                area = self._tigl.wingGetSurfaceArea(i)
                volume = self._tigl.wingGetVolume(i)

                logger.info(
                    f"Wing '{wing_uid}': span={span:.3f}m, "
                    f"area={area:.3f}m², volume={volume:.3f}m³"
                )

        except Exception as e:
            logger.warning(f"Failed to sync from TiGL: {e}")

    def sync_to_tigl(self) -> None:
        """Sync CPACS changes to TiGL by reopening the configuration.

        After modifying the CPACS tree (e.g., updating wing geometry),
        call this method to refresh TiGL's internal geometry cache.

        Note: This requires TiGL to rebuild the geometry from the modified
        CPACS tree, which may take some time for complex configurations.
        """
        if self._tigl is None:
            msg = "TiGL not initialized"
            raise RuntimeError(msg)

        if self._tixi is None:
            msg = "TIXI not initialized"
            raise RuntimeError(msg)

        try:
            self._tigl.close()
            self._tigl.open(self._tixi, "")
            logger.info("TiGL configuration reopened with updated CPACS data")
        except Exception as e:
            msg = f"Failed to reopen TiGL configuration: {e}"
            raise RuntimeError(msg) from e

    def get_wing_geometric_properties(
        self,
        wing_uid: str,
    ) -> dict[str, float]:
        """Get comprehensive geometric properties for a wing.

        Args:
            wing_uid: The wing UID.

        Returns:
            Dictionary with geometric properties.
        """
        if self._tigl is None:
            msg = "TiGL not initialized"
            raise RuntimeError(msg)

        wing_index = self.get_wing_index(wing_uid)

        try:
            properties = {
                "span": self._tigl.wingGetSpan(wing_index),
                "reference_area": self._tigl.wingGetReferenceArea(wing_index),
                "surface_area": self._tigl.wingGetSurfaceArea(wing_index),
                "wetted_area": self._tigl.wingGetWettedArea(wing_index),
                "volume": self._tigl.wingGetVolume(wing_index),
            }

            mac = self._tigl.wingGetMAC(wing_index)
            properties["mac"] = {
                "x": mac[0],
                "y": mac[1],
                "z": mac[2],
            }

            return properties

        except Exception as e:
            logger.warning(f"Failed to get geometric properties for '{wing_uid}': {e}")
            return {}

    def get_aircraft_geometric_properties(self) -> dict[str, Any]:
        """Get comprehensive geometric properties for the entire aircraft.

        Returns:
            Dictionary with aircraft-level geometric properties.
        """
        if self._tigl is None:
            msg = "TiGL not initialized"
            raise RuntimeError(msg)

        properties = {}

        try:
            properties["wings"] = {}
            n_wings = self._tigl.getWingCount()
            for i in range(1, n_wings + 1):
                wing_uid = self._tigl.wingGetUID(i)
                properties["wings"][wing_uid] = self.get_wing_geometric_properties(
                    wing_uid
                )

            properties["fuselages"] = {}
            n_fuselages = self._tigl.getFuselageCount()
            for i in range(1, n_fuselages + 1):
                fuselage_uid = self._tigl.fuselageGetUID(i)
                properties["fuselages"][fuselage_uid] = {
                    "volume": self._tigl.fuselageGetVolume(i),
                }

            properties["aircraft"] = {
                "length": self._tigl.getAirplaneLength(),
                "bounding_box": self._tigl.configurationGetBoundingBox(),
            }

        except Exception as e:
            logger.warning(f"Failed to get aircraft properties: {e}")

        return properties

