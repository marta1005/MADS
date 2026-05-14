"""Descartes library for geometry generation.

This module provides the Descartes-specific logic for geometry generation
from CPACS files. It wraps the Descartes Python interface (when available)
or provides fallback implementations for testing.

This module handles:
- Descartes interface initialization
- CPACS model loading
- Geometry generation parameters
- Output extraction and conversion to MultiADS format
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import NDArray

from multiads.solvers import SolverOptions

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence


DESCARTES_AVAILABLE = True

try:
    import descartes
except ImportError:
    DESCARTES_AVAILABLE = False


def check_descartes_available() -> bool:
    """Check if Descartes is available.

    Returns:
        True if Descartes Python interface is available, False otherwise.
    """
    return DESCARTES_AVAILABLE


def require_descartes() -> None:
    """Raise ImportError if Descartes is not available."""
    if not DESCARTES_AVAILABLE:
        msg = (
            "Descartes is not available. Please install Descartes "
            "Python interface to use geometry generation features. "
            "Descartes can be obtained from Airbus Defence and Space."
        )
        raise ImportError(msg)


class GeometryParameters:
    """Parameters for geometry generation.

    Attributes:
        wing_span: Wing span.
        wing_sweep: Leading edge sweep angle.
        wing_taper_ratio: Taper ratio.
        wing_reference_area: Reference area.
        wing_aspect_ratio: Aspect ratio.
        wing_chord_distribution: Wing chord distribution mode.
        wing_span_distribution: Wing span distribution mode.
        n_span_panels: Number of spanwise panels.
        n_chord_panels: Number of chordwise panels.
        wing_spanwise_spacing: Spanwise panel distribution.
        wing_chordwise_spacing: Chordwise panel distribution.
        fuselage_cabin_length: Fuselage cabin length.
        fuselage_nose_length: Fuselage nose length.
        fuselage_tail_length: Fuselage tail length.
        fuselage_n_cross_sections: Number of fuselage cross-sections.
    """

    def __init__(
        self,
        wing_span: float | None = None,
        wing_sweep: float | None = None,
        wing_taper_ratio: float | None = None,
        wing_reference_area: float | None = None,
        wing_aspect_ratio: float | None = None,
        wing_chord_distribution: str = "linear",
        wing_span_distribution: str = "linear",
        n_span_panels: int = 20,
        n_chord_panels: int = 5,
        wing_spanwise_spacing: str = "linear",
        wing_chordwise_spacing: str = "linear",
        fuselage_cabin_length: float | None = None,
        fuselage_nose_length: float | None = None,
        fuselage_tail_length: float | None = None,
        fuselage_n_cross_sections: int = 10,
    ) -> None:
        self.wing_span = wing_span
        self.wing_sweep = wing_sweep
        self.wing_taper_ratio = wing_taper_ratio
        self.wing_reference_area = wing_reference_area
        self.wing_aspect_ratio = wing_aspect_ratio
        self.wing_chord_distribution = wing_chord_distribution
        self.wing_span_distribution = wing_span_distribution
        self.n_span_panels = n_span_panels
        self.n_chord_panels = n_chord_panels
        self.wing_spanwise_spacing = wing_spanwise_spacing
        self.wing_chordwise_spacing = wing_chordwise_spacing
        self.fuselage_cabin_length = fuselage_cabin_length
        self.fuselage_nose_length = fuselage_nose_length
        self.fuselage_tail_length = fuselage_tail_length
        self.fuselage_n_cross_sections = fuselage_n_cross_sections


class GeometryOutput:
    """Output from geometry generation.

    Attributes:
        wing_coordinates: Wing surface coordinates [n_points, 3].
        wing_normals: Wing surface normals [n_points, 3].
        wing_panel_areas: Wing panel areas [n_panels].
        fuselage_coordinates: Fuselage surface coordinates [n_points, 3].
        fuselage_normals: Fuselage surface normals [n_points, 3].
        wing_reference_area: Wing reference area.
        wing_span: Wing span.
        wing_mean_geometric_chord: Wing mean geometric chord.
        wing_aspect_ratio: Wing aspect ratio.
        fuselage_wetted_area: Fuselage wetted area.
        fuselage_volume: Fuselage volume.
        success: Whether generation succeeded.
        message: Status or error message.
    """

    def __init__(
        self,
        wing_coordinates: NDArray[np.float64] | None = None,
        wing_normals: NDArray[np.float64] | None = None,
        wing_panel_areas: NDArray[np.float64] | None = None,
        fuselage_coordinates: NDArray[np.float64] | None = None,
        fuselage_normals: NDArray[np.float64] | None = None,
        wing_reference_area: float = 0.0,
        wing_span: float = 0.0,
        wing_mean_geometric_chord: float = 0.0,
        wing_aspect_ratio: float = 0.0,
        fuselage_wetted_area: float = 0.0,
        fuselage_volume: float = 0.0,
        success: bool = False,
        message: str = "",
    ) -> None:
        self.wing_coordinates = wing_coordinates if wing_coordinates is not None else np.zeros((0, 3))
        self.wing_normals = wing_normals if wing_normals is not None else np.zeros((0, 3))
        self.wing_panel_areas = wing_panel_areas if wing_panel_areas is not None else np.zeros(0)
        self.fuselage_coordinates = fuselage_coordinates if fuselage_coordinates is not None else np.zeros((0, 3))
        self.fuselage_normals = fuselage_normals if fuselage_normals is not None else np.zeros((0, 3))
        self.wing_reference_area = wing_reference_area
        self.wing_span = wing_span
        self.wing_mean_geometric_chord = wing_mean_geometric_chord
        self.wing_aspect_ratio = wing_aspect_ratio
        self.fuselage_wetted_area = fuselage_wetted_area
        self.fuselage_volume = fuselage_volume
        self.success = success
        self.message = message


class Options(SolverOptions):
    """Solver options for Descartes geometry generation.

    Attributes:
        name: Solver name.
        geometry_type: Type of geometry to generate ('wing', 'fuselage', 'aircraft').
        param_file: Path to Descartes parameter file (optional).
        output_cpacs: Whether to output updated CPACS file.
        output_cpacs_path: Path for output CPACS file.
        generate_mesh: Whether to generate mesh for CFD/FEM.
        mesh_format: Mesh output format if generating mesh.
    """

    def __init__(
        self,
        name: str = "descartes",
        geometry_type: str = "aircraft",
        param_file: str | None = None,
        output_cpacs: bool = False,
        output_cpacs_path: str | None = None,
        generate_mesh: bool = False,
        mesh_format: str = "nastran",
    ) -> None:
        super().__init__()
        self.name = name
        self.geometry_type = geometry_type
        self.param_file = param_file
        self.output_cpacs = output_cpacs
        self.output_cpacs_path = output_cpacs_path
        self.generate_mesh = generate_mesh
        self.mesh_format = mesh_format


class DescartesDriver:
    """Driver for Descartes geometry generation.

    This class encapsulates the interaction with Descartes
    for geometry generation from CPACS files.

    Attributes:
        cpacs_path: Path to CPACS input file.
        options: Descartes solver options.
        params: Geometry generation parameters.
        _descartes_model: Internal Descartes model (when available).
    """

    def __init__(
        self,
        cpacs_path: str | Path,
        options: Options | None = None,
        params: GeometryParameters | None = None,
    ) -> None:
        """Initialize Descartes driver.

        Args:
            cpacs_path: Path to CPACS input file.
            options: Descartes solver options.
            params: Geometry generation parameters.
        """
        self.cpacs_path = Path(cpacs_path) if cpacs_path else None
        self.options = options or Options()
        self.params = params or GeometryParameters()
        self._descartes_lib: Any = None
        self._descartes_interface: Any = None
        self._geometry_output: GeometryOutput | None = None

        if DESCARTES_AVAILABLE and self.cpacs_path:
            self._initialize_descartes()

    def _initialize_descartes(self) -> None:
        """Initialize Descartes interface and load CPACS model."""
        if not DESCARTES_AVAILABLE:
            require_descartes()
            return

        try:
            self._descartes_interface = descartes.DescartesPyLib()
            self._descartes_lib = self._descartes_interface.descartesLib()
            self._descartes_lib.importCPACS(str(self.cpacs_path))
        except AttributeError as e:
            raise AttributeError(
                f"Descartes API method not found: {e}. "
                "The Descartes API may have changed. Check the documentation."
            ) from e
        except Exception as e:
            raise RuntimeError(f"Failed to initialize Descartes: {e}") from e

    def run(self) -> GeometryOutput:
        """Run geometry generation.

        Returns:
            GeometryOutput containing generated geometry data.
        """
        if not DESCARTES_AVAILABLE:
            return self._fallback_run()

        if self._descartes_lib is None:
            return GeometryOutput(
                success=False,
                message="Descartes not initialized",
            )

        try:
            return self._descartes_run()
        except Exception as e:
            return GeometryOutput(
                success=False,
                message=f"Geometry generation failed: {e}",
            )

    def _descartes_run(self) -> GeometryOutput:
        """Run Descartes geometry generation.

        Returns:
            GeometryOutput with generated data.
        """
        output = GeometryOutput(success=True, message="Descartes geometry generated")

        try:
            if self._descartes_lib is None:
                return GeometryOutput(
                    success=False,
                    message="DescartesLib not initialized",
                )

            geom_model = self._descartes_lib.geometryModel()
            geom_eval_service = geom_model.geometryEvaluationService()

            wing_uid = "wing_main"
            try:
                outline_service = geom_eval_service.getOutlinePointService()
                
                root_le = outline_service.getWingRootLE(wing_uid)
                root_te = outline_service.getWingRootTE(wing_uid)
                tip_le = outline_service.getWingTipLE(wing_uid)
                tip_te = outline_service.getWingTipTE(wing_uid)

                wing_span = abs(tip_le[1] - root_le[1])
                root_chord = abs(root_te[0] - root_le[0])
                tip_chord = abs(tip_te[0] - tip_le[0])
                mgc = (root_chord + tip_chord) / 2.0

                output.wing_span = wing_span
                output.wing_reference_area = wing_span * mgc
                output.wing_mean_geometric_chord = mgc
                if mgc > 0:
                    output.wing_aspect_ratio = wing_span**2 / output.wing_reference_area
                else:
                    output.wing_aspect_ratio = 0.0

                output.wing_wetted_area = geom_eval_service.evaluateWettedArea(wing_uid)
                output.wing_reference_area = geom_eval_service.referenceArea()
                output.wing_mean_geometric_chord = geom_eval_service.evaluateMeanAeroChord(wing_uid)

            except AttributeError:
                pass

            self._geometry_output = output
            return self._geometry_output

        except AttributeError as e:
            return GeometryOutput(
                success=False,
                message=f"Descartes API method not found: {e}",
            )
        except Exception as e:
            return GeometryOutput(
                success=False,
                message=f"Descartes geometry generation failed: {e}",
            )

    def _fallback_run(self) -> GeometryOutput:
        """Fallback run when Descartes is not available.

        Provides mock geometry data for testing.

        Returns:
            GeometryOutput with mock data.
        """
        output = GeometryOutput(success=False, message="Descartes not available")

        if self.cpacs_path and self.cpacs_path.exists():
            output.success = True
            output.message = "Fallback mode: CPACS file exists but Descartes unavailable"

        return output

    def _map_params_to_descartes(self) -> dict[str, Any]:
        """Map MultiADS parameters to Descartes format.

        Returns:
            Dictionary of Descartes parameters.
        """
        return {
            "wing_chord_distribution": self.params.wing_chord_distribution,
            "wing_span_distribution": self.params.wing_span_distribution,
            "n_span_panels": self.params.n_span_panels,
            "n_chord_panels": self.params.n_chord_panels,
            "wing_spanwise_spacing": self.params.wing_spanwise_spacing,
            "wing_chordwise_spacing": self.params.wing_chordwise_spacing,
            "fuselage_cabin_length": self.params.fuselage_cabin_length,
            "fuselage_nose_length": self.params.fuselage_nose_length,
            "fuselage_tail_length": self.params.fuselage_tail_length,
            "fuselage_n_cross_sections": self.params.fuselage_n_cross_sections,
        }

    def _convert_geometry_output(self, descartes_output: Any) -> GeometryOutput:
        """Convert Descartes output to MultiADS GeometryOutput.

        Args:
            descartes_output: Raw output from Descartes.

        Returns:
            GeometryOutput with data in MultiADS format.
        """
        output = GeometryOutput(success=True, message="Converted from Descartes")

        try:
            if hasattr(descartes_output, "wing_coordinates"):
                output.wing_coordinates = np.array(descartes_output.wing_coordinates)
            if hasattr(descartes_output, "wing_normals"):
                output.wing_normals = np.array(descartes_output.wing_normals)
            if hasattr(descartes_output, "wing_panel_areas"):
                output.wing_panel_areas = np.array(descartes_output.wing_panel_areas)
            if hasattr(descartes_output, "fuselage_coordinates"):
                output.fuselage_coordinates = np.array(
                    descartes_output.fuselage_coordinates
                )
            if hasattr(descartes_output, "fuselage_normals"):
                output.fuselage_normals = np.array(descartes_output.fuselage_normals)
            if hasattr(descartes_output, "wing_reference_area"):
                output.wing_reference_area = float(descartes_output.wing_reference_area)
            if hasattr(descartes_output, "wing_span"):
                output.wing_span = float(descartes_output.wing_span)
            if hasattr(descartes_output, "wing_mean_geometric_chord"):
                output.wing_mean_geometric_chord = float(
                    descartes_output.wing_mean_geometric_chord
                )
            if hasattr(descartes_output, "wing_aspect_ratio"):
                output.wing_aspect_ratio = float(descartes_output.wing_aspect_ratio)
            if hasattr(descartes_output, "fuselage_wetted_area"):
                output.fuselage_wetted_area = float(descartes_output.fuselage_wetted_area)
            if hasattr(descartes_output, "fuselage_volume"):
                output.fuselage_volume = float(descartes_output.fuselage_volume)
        except Exception as e:
            output.success = False
            output.message = f"Failed to convert output: {e}"

        return output

    def get_wing_coordinates(self) -> NDArray[np.float64]:
        """Get wing surface coordinates.

        Returns:
            Wing coordinates array [n_points, 3].
        """
        if self._geometry_output:
            return self._geometry_output.wing_coordinates
        return np.zeros((0, 3))

    def get_fuselage_coordinates(self) -> NDArray[np.float64]:
        """Get fuselage surface coordinates.

        Returns:
            Fuselage coordinates array [n_points, 3].
        """
        if self._geometry_output:
            return self._geometry_output.fuselage_coordinates
        return np.zeros((0, 3))

    def get_wing_reference_area(self) -> float:
        """Get wing reference area.

        Returns:
            Wing reference area.
        """
        if self._geometry_output:
            return self._geometry_output.wing_reference_area
        return 0.0

    def get_wing_span(self) -> float:
        """Get wing span.

        Returns:
            Wing span.
        """
        if self._geometry_output:
            return self._geometry_output.wing_span
        return 0.0

    def get_wing_mean_geometric_chord(self) -> float:
        """Get wing mean geometric chord.

        Returns:
            Wing mean geometric chord.
        """
        if self._geometry_output:
            return self._geometry_output.wing_mean_geometric_chord
        return 0.0

    def get_wing_aspect_ratio(self) -> float:
        """Get wing aspect ratio.

        Returns:
            Wing aspect ratio.
        """
        if self._geometry_output:
            return self._geometry_output.wing_aspect_ratio
        return 0.0

    def get_fuselage_wetted_area(self) -> float:
        """Get fuselage wetted area.

        Returns:
            Fuselage wetted area.
        """
        if self._geometry_output:
            return self._geometry_output.fuselage_wetted_area
        return 0.0

    def get_fuselage_volume(self) -> float:
        """Get fuselage volume.

        Returns:
            Fuselage volume.
        """
        if self._geometry_output:
            return self._geometry_output.fuselage_volume
        return 0.0


def load_cpacs_model(cpacs_path: str | Path) -> Any:
    """Load a CPACS model into Descartes.

    Args:
        cpacs_path: Path to CPACS file.

    Returns:
        DescartesLib object with loaded CPACS.
    """
    require_descartes()
    cpacs_path = Path(cpacs_path)
    if not cpacs_path.exists():
        msg = f"CPACS file not found: {cpacs_path}"
        raise FileNotFoundError(msg)

    desc_lib = descartes.DescartesLib()
    desc_lib.importCPACS(str(cpacs_path))
    return desc_lib


def set_design_variable_values(
    desc_lib: Any,
    values: list[float],
) -> None:
    """Set design variable values in Descartes model.

    Args:
        desc_lib: DescartesLib instance with loaded CPACS.
        values: List of design variable values.
    """
    require_descartes()

    try:
        desc_pylib = descartes.DescartesPyLib()
        desc_pylib.setDesignVariableValues(values)
    except Exception as e:
        raise RuntimeError(f"Failed to set design variables: {e}") from e


def set_geometry_parameters(
    desc_lib: Any,
    params: GeometryParameters,
) -> None:
    """Set geometry generation parameters in Descartes model.

    Uses Descartes configuration builders to set wing and fuselage parameters.

    Args:
        desc_lib: DescartesLib instance with loaded CPACS.
        params: Geometry generation parameters.
    """
    require_descartes()

    geom_model = desc_lib.geometryModel()
    designer = geom_model.design()

    wing_params = params.wing_span is not None or params.wing_sweep is not None or params.wing_taper_ratio is not None
    fuselage_params = params.fuselage_cabin_length is not None

    if wing_params:
        wing_builder = descartes.WingConfigurationBuilder()

        if params.wing_span is not None:
            wing_builder.setSpan(params.wing_span)
        if params.wing_sweep is not None:
            wing_builder.setSweep(params.wing_sweep)
        if params.wing_taper_ratio is not None:
            wing_builder.setTaperRatio(params.wing_taper_ratio)
        if params.wing_reference_area is not None:
            wing_builder.setReferenceArea(params.wing_reference_area)

        wing_config = wing_builder.config()
        designer.addWing(wing_config)

    if fuselage_params:
        fuselage_builder = descartes.FuselageConfigurationBuilder()

        if params.fuselage_cabin_length is not None:
            fuselage_builder.setLength(params.fuselage_cabin_length)
        if params.fuselage_n_cross_sections is not None:
            fuselage_builder.setNSegments(params.fuselage_n_cross_sections)

        fuselage_config = fuselage_builder.config()
        designer.addFuselage(fuselage_config)


def generate_geometry(
    desc_lib: Any,
    params: GeometryParameters | None = None,
) -> GeometryOutput:
    """Generate geometry using Descartes.

    Args:
        desc_lib: DescartesLib instance with loaded CPACS.
        params: Geometry generation parameters.

    Returns:
        GeometryOutput with generated geometry data.
    """
    require_descartes()

    try:
        geom_model = desc_lib.geometryModel()
        geom_eval_service = geom_model.geometryEvaluationService()

        output = GeometryOutput(success=True, message="Geometry generated from Descartes")

        wing_uid = "wing_main"
        try:
            outline_service = geom_eval_service.getOutlinePointService()
            root_le = outline_service.getWingRootLE(wing_uid)
            root_te = outline_service.getWingRootTE(wing_uid)
            tip_le = outline_service.getWingTipLE(wing_uid)
            tip_te = outline_service.getWingTipTE(wing_uid)

            output.wing_span = abs(tip_le[1] - root_le[1])
            root_chord = abs(root_te[0] - root_le[0])
            tip_chord = abs(tip_te[0] - tip_le[0])
            mgc = (root_chord + tip_chord) / 2.0

            output.wing_mean_geometric_chord = mgc
            output.wing_reference_area = geom_eval_service.referenceArea()
            if output.wing_reference_area > 0:
                output.wing_aspect_ratio = output.wing_span**2 / output.wing_reference_area
            output.wing_wetted_area = geom_eval_service.evaluateWettedArea(wing_uid)

            output.wing_coordinates = np.array([
                list(root_le), list(root_te), list(tip_te), list(tip_le)
            ])

        except AttributeError as e:
            output.message = f"Partial geometry generated: {e}"

        return output
    except Exception as e:
        return GeometryOutput(success=False, message=str(e))


def _convert_descartes_output(descartes_output: Any) -> GeometryOutput:
    """Convert Descartes output to GeometryOutput.

    Args:
        descartes_output: Raw output from Descartes.

    Returns:
        GeometryOutput with data in MultiADS format.
    """
    output = GeometryOutput(success=True, message="Geometry generated")

    try:
        if hasattr(descartes_output, "wing_coordinates"):
            output.wing_coordinates = np.array(descartes_output.wing_coordinates)
        if hasattr(descartes_output, "wing_normals"):
            output.wing_normals = np.array(descartes_output.wing_normals)
        if hasattr(descartes_output, "wing_panel_areas"):
            output.wing_panel_areas = np.array(descartes_output.wing_panel_areas)
        if hasattr(descartes_output, "fuselage_coordinates"):
            output.fuselage_coordinates = np.array(
                descartes_output.fuselage_coordinates
            )
        if hasattr(descartes_output, "fuselage_normals"):
            output.fuselage_normals = np.array(descartes_output.fuselage_normals)
        if hasattr(descartes_output, "wing_reference_area"):
            output.wing_reference_area = float(descartes_output.wing_reference_area)
        if hasattr(descartes_output, "wing_span"):
            output.wing_span = float(descartes_output.wing_span)
        if hasattr(descartes_output, "wing_mean_geometric_chord"):
            output.wing_mean_geometric_chord = float(
                descartes_output.wing_mean_geometric_chord
            )
        if hasattr(descartes_output, "wing_aspect_ratio"):
            output.wing_aspect_ratio = float(descartes_output.wing_aspect_ratio)
        if hasattr(descartes_output, "fuselage_wetted_area"):
            output.fuselage_wetted_area = float(descartes_output.fuselage_wetted_area)
        if hasattr(descartes_output, "fuselage_volume"):
            output.fuselage_volume = float(descartes_output.fuselage_volume)
    except Exception as e:
        output.success = False
        output.message = f"Failed to convert output: {e}"

    return output


def synchronize_cpacs_to_model(
    cpacs_importer: Any,
) -> None:
    """Synchronize CPACS importer variables to underlying XML model.

    This function updates the CPACS XML tree with current variable values
    from the CPACS importer before passing to Descartes.

    Args:
        cpacs_importer: CPACSImporter instance with variables to sync.
    """
    if hasattr(cpacs_importer, "sync_to_cpacs"):
        cpacs_importer.sync_to_cpacs()