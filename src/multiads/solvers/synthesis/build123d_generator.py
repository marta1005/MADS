"""Build123D geometry generator for MultiADS.

This module provides geometry generation using build123d, a Pythonic
CAD library built on OpenCASCADE. It generates aircraft geometries
from CPACS files and MultiADS Assembly components.

The generator:
1. Reads full geometry from CPACS files (wing/fuselage definitions)
2. Overrides specific parameters from Wing/Fuselage components (for optimization)
3. Exports to STEP, IGES, or STL formats for downstream analysis

.. deprecated::
    The original CPACS implementation with incorrect structure has been discontinued.
    This version now uses the official CPACS schema structure from
    https://github.com/DLR-SL/CPACS/tree/develop/examples:
    
    - Fuselage sections use ``positionings`` element (not ``segments``)
    - Wing type attribute: ``type="wing|horizontalStabilizer|verticalStabilizer"``
    - Fuselage profiles: ``pointList`` (real) or ``standardProfile`` (circle/ellipse)
    - Symmetry: ``symmetry="x-z-plane"`` attribute on wings/fuselage
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from multiads.assembly import Fuselage, Wing

try:
    from build123d import (
        Airfoil,
        Box,
        BuildPart,
        BuildSketch,
        Cylinder,
        Edge,
        Face,
        Location,
        Plane,
        Pos,
        Sketch,
        Solid,
        Wire,
        loft,
        make_face,
        export_step,
        export_stl,
        import_step,
    )
    try:
        from ocp_vscode import show
    except ImportError:
        show = None

    BUILD123D_AVAILABLE = True
except ImportError:
    BUILD123D_AVAILABLE = False

from multiads.design_space import CPACSImporter


class Build123DGenerator:
    """Generate aircraft geometries using build123d.

    This class provides methods to generate wing and fuselage geometries
    suitable for aircraft MDO workflows. It prioritizes CPACS data for
    overall geometry but allows overriding specific parameters from
    MultiADS Assembly components during optimization.

    Attributes:
        cpacs_path: Path to CPACS file for full geometry.
        build123d_version: Version of build123d being used.
    """

    def __init__(self, cpacs_path: Path | None = None) -> None:
        """Initialize Build123D generator.

        Args:
            cpacs_path: Path to CPACS file (optional, can be set later).
        """
        self.cpacs_path = cpacs_path
        self._importer: CPACSImporter | None = None
        self._geometry_output: dict[str, Any] = {}

        if BUILD123D_AVAILABLE:
            import build123d

            self.build123d_version = build123d.__version__
        else:
            self.build123d_version = None

    def load_cpacs(self, cpacs_path: Path | None = None) -> None:
        """Load CPACS file for geometry data.

        Args:
            cpacs_path: Path to CPACS file. Uses self.cpacs_path if None.
        """
        path = Path(cpacs_path) if cpacs_path else self.cpacs_path
        if path is None:
            msg = "CPACS path not set"
            raise ValueError(msg)

        self.cpacs_path = path
        self._importer = CPACSImporter(
            file_name=path.name,
            path=path.parent,
            validate=False,
        )

    def _find_first_wing_name(self) -> str | None:
        """Find the first wing name in the CPACS file."""
        if self._importer is None:
            return None

        tree = self._importer._CPACSStructureData__tree
        if tree is None:
            return None

        wings = tree.findall(".//wing")
        if wings:
            return wings[0].get("uID")
        return None

    def _find_wing_by_type(self, wing_type: str) -> str | None:
        """Find wing name by type attribute.

        Args:
            wing_type: Wing type ('wing', 'horizontalStabilizer', 'verticalStabilizer').

        Returns:
            Wing uID or None if not found.
        """
        if self._importer is None:
            return None

        tree = self._importer._CPACSStructureData__tree
        if tree is None:
            return None

        # Search for wing with matching type attribute
        wings = tree.findall(f".//wing[@type='{wing_type}']")
        if wings:
            return wings[0].get("uID")

        # If not found by type, try to find by name pattern
        if wing_type == "horizontalStabilizer":
            for wing in tree.findall(".//wing"):
                name = wing.get("name", "").lower()
                if "horizontal" in name or "stabilizer" in name or "hstab" in name:
                    return wing.get("uID")
        elif wing_type == "verticalStabilizer":
            for wing in tree.findall(".//wing"):
                name = wing.get("name", "").lower()
                if "vertical" in name or "stabilizer" in name or "vstab" in name:
                    return wing.get("uID")

        return None

    def _find_first_fuselage_name(self) -> str | None:
        """Find the first fuselage name in the CPACS file."""
        if self._importer is None:
            return None

        tree = self._importer._CPACSStructureData__tree
        if tree is None:
            return None

        fuselages = tree.findall(".//fuselage")
        if fuselages:
            return fuselages[0].get("uID")
        return None

    def _create_wing_from_cpacs(self, wing_name: str) -> "Wing | None":
        """Manually create Wing component from CPACS data.

        This is a fallback when CPACSImporter.load_wing() fails.

        Args:
            wing_name: uID of the wing in CPACS file.

        Returns:
            Wing component, or None if creation fails.
        """
        if self._importer is None:
            return None

        try:
            from multiads.assembly import Wing, Section, Span, AirfoilNACA4

            tree = self._importer._CPACSStructureData__tree
            if tree is None:
                return None

            # Find wing element
            wing_elem = tree.find(f".//wing[@uID='{wing_name}']")
            if wing_elem is None:
                return None

            # Find sections
            sections = []
            section_elems = wing_elem.findall(".//section")
            for i, sec_elem in enumerate(section_elems):
                sec_uid = sec_elem.get("uID", f"{wing_name}_section_{i}")

                # Get chord
                chord_elem = sec_elem.find(".//chord")
                chord = float(chord_elem.text) if chord_elem is not None else 1.0

                # Get twist
                twist_elem = sec_elem.find(".//rotation/alfa")
                twist = float(twist_elem.text) if twist_elem is not None else 0.0

                # Get airfoil (default to NACA 0012)
                airfoil = AirfoilNACA4(
                    name=f"{sec_uid}_airfoil",
                    m=0, p=0, t=12,
                )

                section = Section(
                    name=sec_uid,
                    airfoil=airfoil,
                    chord=chord,
                    twist=twist,
                )
                sections.append(section)

            # Find spans
            spans = []
            segment_elems = wing_elem.findall(".//segment")
            for i, seg_elem in enumerate(segment_elems):
                seg_uid = seg_elem.get("uID", f"{wing_name}_segment_{i}")

                # Get length
                length_elem = seg_elem.find(".//length")
                length = float(length_elem.text) if length_elem is not None else 10.0

                # Get sweep
                sweep_elem = seg_elem.find(".//dihedral//sweep")
                sweep = float(sweep_elem.text) if sweep_elem is not None else 0.0

                # Get dihedral
                dihedral_elem = seg_elem.find(".//dihedral//dihedral")
                dihedral = float(dihedral_elem.text) if dihedral_elem is not None else 0.0

                span = Span(
                    name=seg_uid,
                    length=length,
                    sweep=sweep,
                    dihed=dihedral,
                )
                spans.append(span)

            if not sections or not spans:
                return None

            wing = Wing(
                name=wing_name,
                sections=sections,
                spans=spans,
            )
            return wing

        except Exception as e:
            print(f"Failed to create wing manually: {e}")
            import traceback
            traceback.print_exc()
            return None

    def generate_wing_from_cpacs(
        self,
        wing_name: str = None,
        override_wing: "Wing" = None,
    ) -> "Solid | None":
        """Generate wing geometry from CPACS with optional overrides.

        Args:
            wing_name: Name of wing in CPACS file. Auto-detected if None.
                Only used when override_wing is not provided.
            override_wing: Optional Wing component to use directly.
                If provided, uses ONLY the sections from override_wing
                (CPACS file is NOT required).

        Returns:
            build123d Solid representing the wing, or None if failed.
        """
        if not BUILD123D_AVAILABLE:
            msg = "build123d not available"
            raise ImportError(msg)

        try:
            # If override_wing is provided, use it directly (no CPACS needed)
            if override_wing is not None:
                sections_data = self._extract_sections_from_wing(override_wing)
                if not sections_data:
                    msg = "No section data extracted from override_wing"
                    raise ValueError(msg)
                wing_solid = self._create_wing_loft(sections_data)
                return wing_solid

            # Otherwise, need CPACS loaded
            if self._importer is None:
                if self.cpacs_path is None:
                    msg = "CPACS path not set"
                    raise ValueError(msg)
                self.load_cpacs()

            # Auto-detect wing name if not provided
            if wing_name is None:
                wing_name = self._find_first_wing_name()
                if wing_name is None:
                    msg = "No wing found in CPACS file"
                    raise ValueError(msg)

            # Load wing from CPACS
            cpacs_wing = self._importer.load_wing(wing_name)
            sections_data = self._extract_sections_from_wing(cpacs_wing)

            if not sections_data:
                msg = "No section data extracted from wing"
                raise ValueError(msg)

            # Generate wing using build123d loft
            wing_solid = self._create_wing_loft(sections_data)

            return wing_solid

        except Exception as e:
            print(f"Wing generation failed: {e}")
            import traceback
            traceback.print_exc()
            return None

    def _extract_sections_from_wing(
        self, wing_component: "Wing"
    ) -> list[dict]:
        """Extract section data from Wing component.

        Note: CPACS stores angles in degrees, but build123d uses radians.
        Position is cumulative along the span.
        """
        sections_data = []

        if not hasattr(wing_component, "sections") or not hasattr(
            wing_component, "spans"
        ):
            return sections_data

        sections = wing_component.sections
        spans = wing_component.spans

        # Calculate cumulative positions along span
        cumulative_pos = 0.0

        for i, section in enumerate(sections):
            airfoil = section.airfoil
            if hasattr(airfoil, "coordinates"):
                coords = airfoil.coordinates(100)
            elif hasattr(airfoil, "camber_distribution"):
                coords = self._generate_airfoil_coords(airfoil, 100)
            else:
                # Skip sections without valid airfoil
                continue

            # Convert degrees to radians for build123d
            twist_deg = float(section.twist) if section.twist else 0.0
            
            # Get span data (if available) for sweep/dihedral
            sweep_deg = 0.0
            dihedral_deg = 0.0
            if i < len(spans):
                span = spans[i]
                sweep_deg = float(span.sweep) if hasattr(span, "sweep") else 0.0
                dihedral_deg = float(span.dihed) if hasattr(span, "dihed") else 0.0

            # Set position (cumulative)
            if i == 0:
                position = 0.0
            else:
                # Use the previous span's length for position
                prev_span_idx = i - 1
                if prev_span_idx < len(spans):
                    length = float(spans[prev_span_idx].length)
                    # If length is 0 or very small, use a default spacing
                    if length < 0.01:
                        length = 10.0  # Default spacing between sections
                    cumulative_pos += length
                    position = cumulative_pos
                else:
                    # No span data, use default spacing
                    cumulative_pos += 10.0
                    position = cumulative_pos

            section_data = {
                "position": position,
                "chord": float(section.chord),
                "twist": np.radians(twist_deg),  # Convert to radians
                "airfoil_coords": coords,
                "sweep": np.radians(sweep_deg),  # Convert to radians
                "dihedral": np.radians(dihedral_deg),  # Convert to radians
            }
            sections_data.append(section_data)

        return sections_data

    def _generate_airfoil_coords(
        self, airfoil: Any, n_points: int = 100
    ) -> np.ndarray:
        """Generate airfoil coordinates from various airfoil types."""
        x = np.linspace(0, 1, n_points)

        if hasattr(airfoil, "naca_code"):
            from multiads.assembly import AirfoilNACA4

            if isinstance(airfoil, AirfoilNACA4):
                coords = self._naca4_coordinates(
                    airfoil.naca_code, n_points
                )
            else:
                coords = np.column_stack((x, np.zeros_like(x)))
        elif hasattr(airfoil, "cst_coeffs_upper") or hasattr(airfoil, "cst_coeffs_lower"):
            # CST airfoil
            coords = self._cst_airfoil_coords(airfoil, n_points)
        elif hasattr(airfoil, "file_path"):
            # File-based airfoil (e.g., Selig format)
            coords = self._file_airfoil_coords(airfoil, n_points)
        else:
            coords = self._naca4_coordinates("0012", n_points)

        return coords

    def _cst_airfoil_coords(
        self, airfoil: Any, n_points: int = 100
    ) -> np.ndarray:
        """Generate airfoil coordinates from CST coefficients."""
        from math import comb
        
        x = np.linspace(0, 1, n_points)

        # Get CST coefficients
        if hasattr(airfoil, "cst_coeffs_upper"):
            upper_coeffs = np.array(airfoil.cst_coeffs_upper)
        else:
            upper_coeffs = np.array([0.1, 0.2, 0.3, 0.4, 0.5])

        if hasattr(airfoil, "cst_coeffs_lower"):
            lower_coeffs = np.array(airfoil.cst_coeffs_lower)
        else:
            lower_coeffs = np.array([-0.1, -0.2, -0.3, -0.4, -0.5])

        # CST class function (shape function)
        def class_function(x, n1=0.5, n2=1.0):
            return x**n1 * (1 - x)**n2

        # CST shape function (Bernstein polynomials)
        def shape_function(x, coeffs):
            N = len(coeffs) - 1
            result = np.zeros_like(x)
            for i, c in enumerate(coeffs):
                # Bernstein polynomial
                result += c * comb(N, i) * x**i * (1 - x)**(N - i)
            return result

        # Calculate y-coordinates
        y_upper = class_function(x) * shape_function(x, upper_coeffs)
        y_lower = class_function(x) * shape_function(x, lower_coeffs)

        # Combine upper and lower surfaces
        x_coords = np.concatenate([x[::-1], x[1:]])
        y_coords = np.concatenate([y_upper[::-1], y_lower[1:]])

        return np.column_stack((x_coords, y_coords))

    def _file_airfoil_coords(
        self, airfoil: Any, n_points: int = 100
    ) -> np.ndarray:
        """Generate airfoil coordinates from file (Selig format)."""
        if not hasattr(airfoil, "file_path"):
            return self._naca4_coordinates("0012", n_points)

        file_path = Path(airfoil.file_path)
        if not file_path.exists():
            return self._naca4_coordinates("0012", n_points)

        # Read Selig format (x, y coordinates)
        coords = []
        with open(file_path, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 2:
                    try:
                        x = float(parts[0])
                        y = float(parts[1])
                        coords.append([x, y])
                    except ValueError:
                        continue

        if not coords:
            return self._naca4_coordinates("0012", n_points)

        return np.array(coords)

    def _naca4_coordinates(self, code: str, n_points: int = 100) -> np.ndarray:
        """Generate NACA 4-digit airfoil coordinates."""
        m = int(code[0]) / 100.0
        p = int(code[1]) / 10.0
        t = int(code[2:]) / 100.0

        x = np.linspace(0, 1, n_points)

        yt = (
            5
            * t
            * (
                0.2969 * np.sqrt(x)
                - 0.1260 * x
                - 0.3516 * x**2
                + 0.2843 * x**3
                - 0.1015 * x**4
            )
        )

        if m > 0 and p > 0:
            yc = np.where(
                x <= p,
                m / p**2 * (2 * p * x - x**2),
                m / (1 - p) ** 2 * (1 - 2 * p + 2 * p * x - x**2),
            )
            dyc = np.where(
                x <= p,
                2 * m / p**2 * (p - x),
                2 * m / (1 - p) ** 2 * (p - x),
            )
        else:
            yc = np.zeros_like(x)
            dyc = np.zeros_like(x)

        theta = np.arctan(dyc)
        xu = x - yt * np.sin(theta)
        yu = yc + yt * np.cos(theta)
        xl = x + yt * np.sin(theta)
        yl = yc - yt * np.cos(theta)

        x_coords = np.concatenate([xu[::-1], xl[1:]])
        y_coords = np.concatenate([yu[::-1], yl[1:]])

        return np.column_stack((x_coords, y_coords))

    def _apply_wing_overrides(
        self, sections_data: list[dict], wing_component: "Wing"
    ) -> list[dict]:
        """Apply MultiADS Wing component overrides to sections_data.

        This allows optimization to modify chord, twist, sweep, dihedral, and positions
        by providing a Wing object with modified parameters.

        Args:
            sections_data: Sections data from CPACS (will be modified in place)
            wing_component: MultiADS Wing component with override parameters

        Returns:
            Updated sections_data with overrides applied
        """
        if not wing_component:
            return sections_data

        # Get sections and spans from the MultiADS Wing component
        sections = wing_component.sections
        spans = wing_component.spans

        if not sections:
            return sections_data

        # Calculate cumulative positions from spans
        cumulative_pos = 0.0

        # Update sections with overrides (match by index)
        for i, section in enumerate(sections):
            # Check if we have corresponding data in sections_data
            if i >= len(sections_data):
                print(f"Warning: More MultiADS sections than CPACS sections")
                break

            # Update chord
            sections_data[i]["chord"] = float(section.chord)

            # Update twist (convert to radians)
            twist_deg = float(section.twist) if section.twist else 0.0
            sections_data[i]["twist"] = np.radians(twist_deg)

            # Update position based on spans (cumulative)
            if i == 0:
                sections_data[i]["position"] = 0.0
            elif i - 1 < len(spans):
                span = spans[i - 1]
                length = float(span.length)
                if length < 0.01:
                    length = 10.0  # Default spacing
                cumulative_pos += length
                sections_data[i]["position"] = cumulative_pos

            # Update airfoil coordinates if airfoil is specified
            airfoil = section.airfoil
            if airfoil:
                if hasattr(airfoil, "coordinates"):
                    coords = airfoil.coordinates(100)
                elif hasattr(airfoil, "camber_distribution"):
                    coords = self._generate_airfoil_coords(airfoil, 100)
                else:
                    coords = None
                
                if coords is not None:
                    sections_data[i]["airfoil_coords"] = coords

        # Update sweep and dihedral from spans
        # Note: sweep/dihedral are stored in sections_data with the section they affect
        for i, span in enumerate(spans):
            # Apply sweep/dihedral to the section AFTER this span
            section_idx = i + 1
            if section_idx < len(sections_data):
                # Update sweep (convert to radians)
                if hasattr(span, "sweep"):
                    sweep_deg = float(span.sweep)
                    sections_data[section_idx]["sweep"] = np.radians(sweep_deg)
                
                # Update dihedral (convert to radians)
                if hasattr(span, "dihed"):
                    dihedral_deg = float(span.dihed)
                    sections_data[section_idx]["dihedral"] = np.radians(dihedral_deg)

        return sections_data

    def _create_wing_loft(
        self, sections_data: list[dict]
    ) -> "Solid | None":
        """Create wing solid by lofting airfoil sections with sweep/dihedral/twist."""
        if not sections_data or len(sections_data) < 2:
            print(f"Need at least 2 sections to loft, got {len(sections_data) if sections_data else 0}")
            return None

        try:
            with BuildPart() as wing_part:
                for i, section in enumerate(sections_data):
                    z_pos = section["position"]
                    sweep = section.get("sweep", 0.0)
                    dihedral = section.get("dihedral", 0.0)
                    twist = section.get("twist", 0.0)
                    chord = section["chord"]

                    # Validate position
                    if z_pos < 0:
                        print(f"Warning: Section {i} has negative position {z_pos}, skipping...")
                        continue

                    # Calculate position with sweep and dihedral
                    x_offset = -np.tan(sweep) * z_pos if sweep != 0.0 else 0.0
                    y_offset = np.tan(dihedral) * z_pos if dihedral != 0.0 else 0.0

                    # Create sketch with twist rotation
                    location = Location((x_offset, y_offset, z_pos))
                    if twist != 0.0:
                        from build123d import Rotation
                        location = location * Rotation(0, 0, twist)

                    with BuildSketch(location) as sketch:
                        coords = section["airfoil_coords"]
                        scaled_coords = np.column_stack([
                            coords[:, 0] * chord,
                            coords[:, 1] * chord
                        ])

                        points = [tuple(p) for p in scaled_coords]
                        if len(points) > 2:
                            wire = Wire.make_polygon(points)
                            from build123d import make_face
                            make_face(wire)

                # Loft all sections
                if len(wing_part.pending_faces) > 1:
                    loft()

            return wing_part.part if wing_part.part else None

        except Exception as e:
            print(f"Wing loft failed: {e}")
            import traceback
            traceback.print_exc()
            return None

    def generate_fuselage_from_cpacs(
        self,
        fuselage_name: str = None,
        override_fuselage: "Fuselage" = None,
    ) -> "Solid | None":
        """Generate fuselage geometry from CPACS with optional overrides.

        Args:
            fuselage_name: Name of fuselage in CPACS file. Auto-detected if None.
            override_fuselage: Optional Fuselage component to override.

        Returns:
            build123d Solid representing the fuselage, or None if failed.
        """
        if not BUILD123D_AVAILABLE:
            msg = "build123d not available"
            raise ImportError(msg)

        try:
            # Load CPACS if not already loaded
            if self._importer is None:
                if self.cpacs_path is None:
                    # No CPACS, use manual creation
                    return self._create_fuselage_manual(override_fuselage)
                self.load_cpacs()

            # Auto-detect fuselage name if not provided
            if fuselage_name is None:
                fuselage_name = self._find_first_fuselage_name()
                if fuselage_name is None:
                    # No fuselage in CPACS, use manual creation
                    return self._create_fuselage_manual(override_fuselage)

            # Try to create fuselage from CPACS sections
            fuselage_solid = self._create_fuselage_from_cpacs_sections(fuselage_name)

            if fuselage_solid is not None:
                return fuselage_solid

            # Fallback: use manual creation with parameters from CPACS
            cpacs_fuselage = self._importer.load_fuselage(fuselage_name)

            # Extract fuselage parameters
            length = float(cpacs_fuselage.length) if hasattr(cpacs_fuselage, "length") else 40.0
            width = float(cpacs_fuselage.maximum_width) if hasattr(cpacs_fuselage, "maximum_width") else 4.0
            height = float(cpacs_fuselage.maximum_height) if hasattr(cpacs_fuselage, "maximum_height") else 4.0

            # Override if Fuselage component provided
            if override_fuselage is not None:
                if hasattr(override_fuselage, "length") and override_fuselage.length is not None:
                    length = float(override_fuselage.length)
                if hasattr(override_fuselage, "maximum_width") and override_fuselage.maximum_width is not None:
                    width = float(override_fuselage.maximum_width)
                elif hasattr(override_fuselage, "width") and override_fuselage.width is not None:
                    width = float(override_fuselage.width)
                if hasattr(override_fuselage, "maximum_height") and override_fuselage.maximum_height is not None:
                    height = float(override_fuselage.maximum_height)
                elif hasattr(override_fuselage, "height") and override_fuselage.height is not None:
                    height = float(override_fuselage.height)

            # Create fuselage using lofted cross-sections
            return self._create_fuselage_loft(length, width, height)

        except Exception as e:
            print(f"Fuselage generation failed: {e}")
            import traceback
            traceback.print_exc()
            return None

    def _create_fuselage_manual(
        self, override_fuselage: "Fuselage" = None
    ) -> "Solid | None":
        """Create fuselage manually (when no CPACS data available)."""
        length = 40.0
        width = 4.0
        height = 4.0

        if override_fuselage is not None:
            if hasattr(override_fuselage, "length") and override_fuselage.length is not None:
                length = float(override_fuselage.length)
            if hasattr(override_fuselage, "maximum_width") and override_fuselage.maximum_width is not None:
                width = float(override_fuselage.maximum_width)
            elif hasattr(override_fuselage, "width") and override_fuselage.width is not None:
                width = float(override_fuselage.width)
            if hasattr(override_fuselage, "maximum_height") and override_fuselage.maximum_height is not None:
                height = float(override_fuselage.maximum_height)
            elif hasattr(override_fuselage, "height") and override_fuselage.height is not None:
                height = float(override_fuselage.height)

        return self._create_fuselage_loft(length, width, height)

    def _create_fuselage_loft(
        self, length: float, width: float, height: float
    ) -> "Solid | None":
        """Create fuselage by lofting circular cross-sections.

        Args:
            length: Fuselage length in meters (along x-axis).
            width: Fuselage maximum width in meters (y-axis).
            height: Fuselage maximum height in meters (z-axis).

        Returns:
            build123d Solid representing the fuselage.
        """
        with BuildPart() as fuselage_part:
            n_sections = 7
            for i in range(n_sections):
                x_pos = (i / (n_sections - 1)) * length

                # Vary radius along length (smaller at nose/tail)
                # Using a smooth cosine transition
                t = i / (n_sections - 1)
                radius_factor = 0.3 + 0.7 * (1 - np.cos(np.pi * t)) / 2

                radius = width * radius_factor

                # Create circular cross-section at this x position
                # Use Plane.YZ to create sketch in y-z plane at each x position
                with BuildSketch(Plane.YZ.offset(x_pos)) as sketch:
                    from build123d import Circle

                    Circle(radius)

            # Loft all sections
            if len(fuselage_part.pending_faces) > 1:
                loft()

        return fuselage_part.part if fuselage_part.part else None

    def _create_fuselage_from_cpacs_sections(
        self, fuselage_uid: str
    ) -> "Solid | None":
        """Create fuselage from CPACS sections and profiles (CPACS 3.5 compliant).

        Args:
            fuselage_uid: The fuselage uID in CPACS.

        Returns:
            build123d Solid representing the fuselage.
        """
        if self._importer is None:
            return None

        tree = self._importer._CPACSStructureData__tree
        if tree is None:
            return None

        # Find fuselage element
        fuselage_elem = tree.find(f".//fuselage[@uID='{fuselage_uid}']")
        if fuselage_elem is None:
            return None

        # Get fuselage sections
        sections = fuselage_elem.findall("sections/section")
        if not sections:
            return None

        # Build a map of section positions from positionings
        section_positions = self._build_fuselage_position_map(fuselage_elem, sections)

        with BuildPart() as fuselage_part:
            for i, section in enumerate(sections):
                section_uid = section.get("uID")
                x_pos = section_positions.get(section_uid, 0.0)

                # Get element and profile
                element = section.find("elements/element")
                if element is not None:
                    profile_uid = element.get("profileUID")
                    if profile_uid:
                        # Get profile
                        profile = tree.find(f".//fuselageProfile[@uID='{profile_uid}']")
                        if profile is not None:
                            # Try to get profile points (pointList or standardProfile)
                            points = self._get_fuselage_profile_points(profile)

                            if points is not None and len(points) > 2:
                                # Profile with points (pointList)
                                with BuildSketch(Plane.YZ.offset(x_pos)) as sketch:
                                    # Convert points to (y, z) coordinates
                                    yz_points = [(p[1], p[2]) for p in points]
                                    wire = Wire.make_polygon(yz_points)
                                    make_face(wire)
                            else:
                                # Circular profile (standardProfile/circle)
                                radius = self._get_fuselage_profile_radius(profile, element)
                                if radius > 0:
                                    with BuildSketch(Plane.YZ.offset(x_pos)) as sketch:
                                        from build123d import Circle
                                        Circle(radius)

            # Loft all sections
            if len(fuselage_part.pending_faces) > 1:
                loft()

        return fuselage_part.part if fuselage_part.part else None

    def _build_fuselage_position_map(
        self, fuselage_elem, sections
    ) -> dict[str, float]:
        """Build a map of section uID to x-position using positionings (CPACS 3.5).

        Args:
            fuselage_elem: XML element of fuselage.
            sections: List of section XML elements.

        Returns:
            Dictionary mapping section uID to x-position.
        """
        positions = {}
        cumulative_pos = 0.0

        # Get positionings
        positionings = fuselage_elem.findall("positionings/positioning")

        if positionings:
            # Build graph of section connections
            connections = {}
            for pos in positionings:
                from_uid = pos.findtext("fromSectionUID")
                to_uid = pos.findtext("toSectionUID")
                length_elem = pos.find("length")

                if to_uid:
                    try:
                        length = float(length_elem.text) if length_elem is not None else 0.0
                    except (ValueError, AttributeError):
                        length = 0.0

                    if from_uid:
                        connections[to_uid] = (from_uid, length)
                    else:
                        # This is a "to" only positioning (absolute position)
                        positions[to_uid] = length

            # Build positions using connections
            # First, find sections with no "from" (root sections)
            for section in sections:
                uid = section.get("uID")
                if uid not in connections and uid not in positions:
                    # Check if section has direct translation
                    trans = section.find("transformation/translation/x")
                    if trans is not None and trans.text:
                        try:
                            positions[uid] = float(trans.text)
                        except ValueError:
                            positions[uid] = cumulative_pos
                    else:
                        positions[uid] = cumulative_pos

            # Now resolve connections
            for to_uid, (from_uid, length) in connections.items():
                if from_uid in positions:
                    positions[to_uid] = positions[from_uid] + length
                else:
                    positions[to_uid] = length
        else:
            # No positionings - use direct translations
            for section in sections:
                uid = section.get("uID")
                trans = section.find("transformation/translation/x")
                if trans is not None and trans.text:
                    try:
                        positions[uid] = float(trans.text)
                    except ValueError:
                        positions[uid] = cumulative_pos
                else:
                    positions[uid] = cumulative_pos
                cumulative_pos += 2.0  # Default spacing

        return positions

    def _get_fuselage_section_position(
        self, fuselage_elem, section_uid: str
    ) -> float:
        """Get x-position of fuselage section from positionings.

        Args:
            fuselage_elem: XML element of fuselage.
            section_uid: uID of the section to find position for.

        Returns:
            X-position in meters (float).
        """
        # Search positionings to find which one references this section
        positionings = fuselage_elem.findall(".//positioning")
        for pos in positionings:
            to_section = pos.findtext("toSectionUID")
            if to_section == section_uid:
                # Get length from this positioning
                length_elem = pos.find("length")
                if length_elem is not None and length_elem.text:
                    try:
                        return float(length_elem.text)
                    except ValueError:
                        pass
            from_section = pos.findtext("fromSectionUID")
            if from_section == section_uid:
                # This is the from section, need to add length
                length_elem = pos.find("length")
                if length_elem is not None and length_elem.text:
                    try:
                        length = float(length_elem.text)
                        # Recursively get the to-section position
                        to_uid = pos.findtext("toSectionUID")
                        if to_uid:
                            to_pos = self._get_fuselage_section_position(fuselage_elem, to_uid)
                            return to_pos - length
                    except ValueError:
                        pass

        # Default: check transformation directly
        section = fuselage_elem.find(f".//section[@uID='{section_uid}']")
        if section is not None:
            trans = section.find("transformation/translation/x")
            if trans is not None and trans.text:
                try:
                    return float(trans.text)
                except ValueError:
                    pass

        return 0.0

    def _get_fuselage_profile_radius(self, profile_elem, element_elem) -> float:
        """Get radius from fuselage profile (circular profile).

        Args:
            profile_elem: XML element of fuselageProfile.
            element_elem: XML element of section element.

        Returns:
            Radius in meters (default 2.0 if not found).
        """
        # Try to get radius from standardProfile/circle (CPACS 3.5)
        circle = profile_elem.find(".//standardProfile/circle/radius")
        if circle is not None and circle.text:
            try:
                return float(circle.text)
            except ValueError:
                pass

        # Try to get radius from profile scaling
        scaling = profile_elem.find(".//scaling")
        if scaling is not None:
            x_scale = scaling.find("x")
            if x_scale is not None and x_scale.text:
                try:
                    return float(x_scale.text)
                except ValueError:
                    pass

        # Try to get from element transformation scaling
        elem_scaling = element_elem.find(".//transformation/scaling")
        if elem_scaling is not None:
            x_scale = elem_scaling.find("x")
            if x_scale is not None and x_scale.text:
                try:
                    return float(x_scale.text)
                except ValueError:
                    pass

        return 2.0  # Default radius

    def _get_fuselage_profile_points(self, profile_elem) -> list | None:
        """Extract points from fuselageProfile pointList or standardProfile.

        Args:
            profile_elem: XML element of fuselageProfile.

        Returns:
            List of (x, y, z) points or None.
        """
        # Try pointList first (real profiles with explicit points)
        point_list = profile_elem.find("pointList")
        if point_list is not None:
            x_elem = point_list.find("x")
            y_elem = point_list.find("y")
            z_elem = point_list.find("z")

            if x_elem is not None and y_elem is not None and z_elem is not None:
                try:
                    x_vals = [float(v) for v in x_elem.text.split(";")]
                    y_vals = [float(v) for v in y_elem.text.split(";")]
                    z_vals = [float(v) for v in z_elem.text.split(";")]

                    points = list(zip(x_vals, y_vals, z_vals))
                    return points
                except (ValueError, AttributeError):
                    pass

        # Try standardProfile/superEllipse (CPACS 3.5)
        super_ellipse = profile_elem.find("standardProfile/superEllipse")
        if super_ellipse is not None:
            return self._generate_superellipse_points(super_ellipse)

        # Try standardProfile/circle (CPACS 3.5)
        circle = profile_elem.find("standardProfile/circle/radius")
        if circle is not None:
            try:
                radius = float(circle.text)
                return self._generate_circle_points(radius)
            except (ValueError, AttributeError):
                pass

        return None

    def _generate_superellipse_points(self, super_ellipse_elem) -> list:
        """Generate points from superEllipse parameters.

        Args:
            super_ellipse_elem: XML element of superEllipse.

        Returns:
            List of (x, y, z) points for a super-ellipse profile.
        """
        try:
            m_lower = float(super_ellipse_elem.findtext("mLower", "2.0"))
            m_upper = float(super_ellipse_elem.findtext("mUpper", "2.0"))
            n_lower = float(super_ellipse_elem.findtext("nLower", "2.0"))
            n_upper = float(super_ellipse_elem.findtext("nUpper", "2.0"))
            lower_height_fraction = float(super_ellipse_elem.findtext("lowerHeightFraction", "0.5"))
        except (ValueError, AttributeError):
            m_lower, m_upper, n_lower, n_upper = 2.0, 2.0, 2.0, 2.0
            lower_height_fraction = 0.5

        import math

        n_points = 50
        points = []

        for i in range(n_points + 1):
            t = i / n_points
            theta = 2 * math.pi * t

            # Super-ellipse equation: |x/a|^n + |y/b|^n = 1
            # For fuselage profile: y is width, z is height
            # We map x to 0 (profile is in y-z plane)
            x = 0.0

            # Parametric form of super-ellipse
            if t <= lower_height_fraction:
                # Lower part
                n = n_lower
                m = m_lower
            else:
                # Upper part
                n = n_upper
                m = m_upper

            # Generate y, z coordinates
            # Using parametric equations for super-ellipse
            cos_theta = math.cos(theta)
            sin_theta = math.sin(theta)

            # Super-ellipse parametric (using sign-preserving power)
            y = math.copysign(abs(cos_theta) ** (2 / n), cos_theta)
            z = math.copysign(abs(sin_theta) ** (2 / m), sin_theta)

            # Scale to unit size (will be scaled by transformation scaling)
            points.append((x, y, z))

        return points

    def _generate_circle_points(self, radius: float, n_points: int = 50) -> list:
        """Generate points for a circular profile.

        Args:
            radius: Circle radius.
            n_points: Number of points to generate.

        Returns:
            List of (x, y, z) points for a circular profile.
        """
        import math

        points = []
        for i in range(n_points + 1):
            theta = 2 * math.pi * i / n_points
            x = 0.0
            y = radius * math.cos(theta)
            z = radius * math.sin(theta)
            points.append((x, y, z))

        return points

    def _create_fuselage_loft_test(self) -> "Solid | None":
        """Test method: Create fuselage with lofted sections."""
        return self.generate_fuselage_from_cpacs()

    def export_step(self, shape, filepath: Path) -> None:
        """Export geometry to STEP file."""
        if not BUILD123D_AVAILABLE:
            return

        filepath.parent.mkdir(parents=True, exist_ok=True)
        export_step(shape, str(filepath))

    def export_iges(self, shape, filepath: Path) -> None:
        """Export geometry to STEP file (renamed to .iges if needed)."""
        if not BUILD123D_AVAILABLE:
            return

        filepath.parent.mkdir(parents=True, exist_ok=True)

        step_path = filepath.with_suffix(".step")
        export_step(shape, str(step_path))

        if filepath.suffix.lower() == ".iges":
            import shutil
            shutil.copy2(step_path, filepath)

    def export_stl(
        self, shape, filepath: Path, tolerance: float = 0.1
    ) -> None:
        """Export geometry to STL file."""
        if not BUILD123D_AVAILABLE:
            return

        filepath.parent.mkdir(parents=True, exist_ok=True)
        export_stl(
            shape,
            str(filepath),
            tolerance=tolerance,
        )

    def get_geometry_metrics(self, shape) -> dict[str, float]:
        """Calculate geometry metrics from shape."""
        if not BUILD123D_AVAILABLE or shape is None:
            return {}

        try:
            metrics = {
                "volume": float(shape.volume),
                "surface_area": float(shape.area),
            }

            bbox = shape.bounding_box()
            if bbox:
                metrics["span"] = float(bbox.max.Z - bbox.min.Z)
                metrics["max_chord"] = float(bbox.max.X - bbox.min.X)
                metrics["max_thickness"] = float(bbox.max.Y - bbox.min.Y)

            return metrics

        except Exception as e:
            print(f"Metrics calculation failed: {e}")
            return {}

    def _ensure_solid(self, result) -> "Solid | None":
        """Extract Solid from boolean operation result (handles Solid or ShapeList).

        Boolean operations like fuse(), cut(), intersect() can return
        either a Solid or a ShapeList containing Solids.

        Args:
            result: Result from boolean operation (Solid or ShapeList).

        Returns:
            Single Solid or None if extraction fails.
        """
        from build123d.topology.three_d import Solid
        from build123d.topology.shape_core import ShapeList

        if isinstance(result, Solid):
            return result
        elif isinstance(result, ShapeList) and len(result) > 0:
            # Return first solid (assumes single combined body)
            return result[0]
        return None

    def _get_wing_position_from_cpacs(self, wing_uid: str) -> dict:
        """Read wing transformation position from CPACS.

        Args:
            wing_uid: The wing uID in CPACS.

        Returns:
            Dictionary with 'x', 'y', 'z' position floats (meters).
        """
        if self._importer is None:
            return {"x": 0.0, "y": 0.0, "z": 0.0}

        tree = self._importer._CPACSStructureData__tree
        if tree is None:
            return {"x": 0.0, "y": 0.0, "z": 0.0}

        wing_elem = tree.find(f".//wing[@uID='{wing_uid}']")
        if wing_elem is None:
            return {"x": 0.0, "y": 0.0, "z": 0.0}

        trans = wing_elem.find("transformation/translation")
        if trans is None:
            return {"x": 0.0, "y": 0.0, "z": 0.0}

        x = trans.find("x")
        y = trans.find("y")
        z = trans.find("z")

        return {
            "x": float(x.text) if x is not None and x.text else 0.0,
            "y": float(y.text) if y is not None and y.text else 0.0,
            "z": float(z.text) if z is not None and z.text else 0.0,
        }

    def _find_junction_edges(
        self, combined_solid: "Solid", junction_y: float, tolerance: float = 0.5
    ) -> list:
        """Find edges near wing-fuselage junction for fillet.

        Args:
            combined_solid: The combined aircraft solid.
            junction_y: The y-coordinate of the junction (fuselage half-width).
            tolerance: Distance tolerance for edge selection.

        Returns:
            List of edges near the junction.
        """
        try:
            all_edges = combined_solid.edges()
            junction_edges = []

            for edge in all_edges:
                try:
                    center = edge.center()
                    # Check if edge is near the junction (y ≈ junction_y)
                    if abs(center.Y - junction_y) < tolerance:
                        junction_edges.append(edge)
                except Exception:
                    continue

            return junction_edges
        except Exception as e:
            print(f"Failed to find junction edges: {e}")
            return []

    def _generate_and_position_wing(
        self,
        wing_name: str,
        override_wing: "Wing",
        fuselage_solid: "Solid",
        wing_type: str = "main",
    ) -> "Solid | None":
        """Generate wing and position it at fuselage side.

        Args:
            wing_name: CPACS wing name (auto-detect if None).
            override_wing: Optional Wing component for parameters.
            fuselage_solid: Fuselage solid for positioning reference.
            wing_type: Type of wing ('main', 'horizontal', 'vertical').

        Returns:
            Positioned wing Solid or None if failed.
        """
        # Generate wing
        wing_part = self.generate_wing_from_cpacs(wing_name, override_wing)
        if wing_part is None:
            return None

        # Convert Part to Solid
        from build123d.topology.composite import Part
        from build123d.topology.three_d import Solid
        
        wing_solid = None
        if isinstance(wing_part, Part):
            solids = wing_part.solids()
            if solids and len(solids) > 0:
                wing_solid = solids[0]
        elif isinstance(wing_part, Solid):
            wing_solid = wing_part
        
        if wing_solid is None:
            print("Failed to extract solid from wing")
            return None

        # Get fuselage bounding box for positioning
        try:
            fuse_bb = fuselage_solid.bounding_box()
            fuse_half_width = (fuse_bb.max.Y - fuse_bb.min.Y) / 2
        except Exception:
            fuse_half_width = 2.5  # Default

        # Read wing position from CPACS if available
        wing_pos = {"x": 0.5, "y": 0.0, "z": 0.3}  # Defaults
        if wing_name is not None:
            cpacs_pos = self._get_wing_position_from_cpacs(wing_name)
            if cpacs_pos["y"] != 0.0:  # CPACS has explicit position
                wing_pos["y"] = cpacs_pos["y"]
            else:
                wing_pos["y"] = fuse_half_width  # At fuselage side
            wing_pos["x"] = cpacs_pos["x"]
            wing_pos["z"] = cpacs_pos["z"]

        # Calculate position based on wing type
        if wing_type == "main":
            # Main wing at fuselage side
            # wing_pos["y"] is the offset from center (0 = at center, >0 = to the side)
            # If CPACS has explicit position, use it as offset
            if wing_pos["y"] != 0.0:  # CPACS has explicit position
                y_pos = wing_pos["y"]  # Use as absolute position
            else:
                y_pos = fuse_half_width  # At fuselage side (half-width)
            x_pos = fuse_bb.min.X + wing_pos["x"] * (fuse_bb.max.X - fuse_bb.min.X)
            z_pos = fuse_bb.min.Z + wing_pos["z"] * (fuse_bb.max.Z - fuse_bb.min.Z)
        elif wing_type == "horizontal":
            # Horizontal stabilizer at tail, centered
            y_pos = wing_pos["y"] if wing_pos["y"] != 0.0 else 0.0
            x_pos = fuse_bb.max.X - 2.0  # Near tail
            z_pos = fuse_bb.min.Z + 0.7 * (fuse_bb.max.Z - fuse_bb.min.Z)
        elif wing_type == "vertical":
            # Vertical stabilizer at tail, centered on fuselage
            y_pos = 0.0  # Centered
            x_pos = fuse_bb.max.X - 2.0  # Near tail
            z_pos = fuse_bb.min.Z + 0.5 * (fuse_bb.max.Z - fuse_bb.min.Z)
        else:
            y_pos = fuse_half_width
            x_pos = 0.0
            z_pos = 0.0

        # Move wing to position
        try:
            from build123d import Location
            positioned = wing_solid.move(Location((x_pos, y_pos, z_pos)))
            return positioned  # Returns a Solid after move
        except Exception as e:
            print(f"Failed to position wing: {e}")
            return wing_solid  # Return unpositioned

    def generate_aircraft(
        self,
        wing_name: str = None,
        fuselage_name: str = None,
        horizontal_stabilizer_name: str = None,
        vertical_stabilizer_name: str = None,
        override_wing: "Wing" = None,
        override_fuselage: "Fuselage" = None,
        override_horizontal: "Wing" = None,
        override_vertical: "Wing" = None,
        symmetry: bool = True,
        fillet_radius: float = 0.3,
        auto_detect_stabilizers: bool = True,
    ) -> "Solid | None":
        """Generate complete aircraft geometry with wing-fuselage integration.

        Creates a unified aircraft solid with C1 continuity (fillet) at junctions.
        Supports main wing, horizontal stabilizer, vertical stabilizer, and symmetry.

        Args:
            wing_name: CPACS main wing name (auto-detect if None).
            fuselage_name: CPACS fuselage name (auto-detect if None).
            horizontal_stabilizer_name: CPACS horizontal stabilizer name.
                Auto-detected if auto_detect_stabilizers is True.
            vertical_stabilizer_name: CPACS vertical stabilizer name.
                Auto-detected if auto_detect_stabilizers is True.
            override_wing: Optional Wing component for main wing.
            override_fuselage: Optional Fuselage component.
            override_horizontal: Optional Wing component for horizontal stabilizer.
            override_vertical: Optional Wing component for vertical stabilizer.
            symmetry: If True, mirror wings for both sides.
            fillet_radius: Radius for C1 fillet at junctions (meters).
            auto_detect_stabilizers: If True, auto-detect stabilizers from CPACS.

        Returns:
            Combined aircraft Solid with C1 continuity, or None if failed.
        """
        if not BUILD123D_AVAILABLE:
            msg = "build123d not available"
            raise ImportError(msg)

        try:
            # Load CPACS if needed
            if self._importer is None:
                if self.cpacs_path is None:
                    msg = "CPACS path not set"
                    raise ValueError(msg)
                self.load_cpacs()

            # 1. Generate fuselage
            print("Generating fuselage...")
            fuselage = self.generate_fuselage_from_cpacs(
                fuselage_name, override_fuselage
            )
            if fuselage is None:
                print("Fuselage generation failed")
                return None

            # Convert Part to Solid if needed
            from build123d.topology.composite import Part
            from build123d.topology.three_d import Solid
            
            if isinstance(fuselage, Part):
                solids = fuselage.solids()
                if solids and len(solids) > 0:
                    fuselage = solids[0]
                else:
                    print("Failed to extract solid from fuselage Part")
                    return None
            
            if not isinstance(fuselage, Solid):
                print("Failed to get valid fuselage solid")
                return None

            # 2. Generate main wing(s)
            print("Generating main wing...")
            main_wing = self._generate_and_position_wing(
                wing_name, override_wing, fuselage, wing_type="main"
            )

            # 3. Generate horizontal stabilizer
            horizontal = None
            h_stab_name = horizontal_stabilizer_name
            if auto_detect_stabilizers and h_stab_name is None:
                h_stab_name = self._find_wing_by_type("horizontalStabilizer")
            if h_stab_name is not None:
                print(f"Generating horizontal stabilizer ({h_stab_name})...")
                horizontal = self._generate_and_position_wing(
                    h_stab_name,
                    override_horizontal,
                    fuselage,
                    wing_type="horizontal",
                )

            # 4. Generate vertical stabilizer
            vertical = None
            v_stab_name = vertical_stabilizer_name
            if auto_detect_stabilizers and v_stab_name is None:
                v_stab_name = self._find_wing_by_type("verticalStabilizer")
            if v_stab_name is not None:
                print(f"Generating vertical stabilizer ({v_stab_name})...")
                vertical = self._generate_and_position_wing(
                    v_stab_name,
                    override_vertical,
                    fuselage,
                    wing_type="vertical",
                )

            # 5. Combine all solids with boolean union (fuse)
            print("Combining solids...")
            combined = fuselage  # fuselage is already a Solid at this point

            if main_wing is not None:
                # Convert wing to Solid if needed
                if isinstance(main_wing, Part):
                    wing_solids = main_wing.solids()
                    if wing_solids and len(wing_solids) > 0:
                        main_wing = wing_solids[0]
                    else:
                        print("Failed to extract solid from wing Part")
                        main_wing = None

            if main_wing is not None and isinstance(main_wing, Solid):
                    # Fuse main wing using OCP directly for reliability
                    print(f"  Fusing main wing...")
                    try:
                        from OCP.TopoDS import TopoDS_Shape
                        from OCP.BRepAlgoAPI import BRepAlgoAPI_Fuse
                        
                        fuse_api = BRepAlgoAPI_Fuse(main_wing.wrapped, combined.wrapped)
                        fuse_api.Build()
                        
                        if fuse_api.IsDone():
                            result_shape = fuse_api.Shape()
                            # Convert back to build123d Solid
                            from build123d.topology.three_d import Solid
                            combined = Solid(result_shape)
                            print(f"  ✓ Fuse succeeded!")
                        else:
                            print(f"  ✗ OCP fuse failed, trying build123d...")
                            fuse_result = combined.fuse(main_wing)
                            combined = self._ensure_solid(fuse_result)
                    except Exception as e:
                        print(f"  Fuse failed: {e}")
                        # Fallback to build123d
                        try:
                            fuse_result = combined.fuse(main_wing)
                            combined = self._ensure_solid(fuse_result)
                        except:
                            pass
                    
                    # Handle symmetry (mirror)
                    if symmetry:
                        print("Adding mirrored wing...")
                        try:
                            from build123d import mirror, Plane
                            wing_left_part = mirror(main_wing, Plane.YZ)
                            # Extract Solid from Part
                            if isinstance(wing_left_part, Part):
                                left_solids = wing_left_part.solids()
                                if left_solids and len(left_solids) > 0:
                                    wing_left_solid = left_solids[0]
                                else:
                                    wing_left_solid = None
                            elif isinstance(wing_left_part, Solid):
                                wing_left_solid = wing_left_part
                            else:
                                wing_left_solid = None
                            
                            if wing_left_solid is not None:
                                # Fuse mirrored wing
                                fuse_api2 = BRepAlgoAPI_Fuse(
                                    wing_left_solid.wrapped, combined.wrapped
                                )
                                fuse_api2.Build()
                                if fuse_api2.IsDone():
                                    result_shape2 = fuse_api2.Shape()
                                    combined = Solid(result_shape2)
                                    print("  ✓ Mirrored wing added!")
                                else:
                                    print("  ✗ Failed to fuse mirrored wing")
                        except Exception as e:
                            print(f"Failed to add mirrored wing: {e}")

            if horizontal is not None:
                print("Fusing horizontal stabilizer...")
                fuse_result = combined.fuse(horizontal)
                combined = self._ensure_solid(fuse_result)

            if vertical is not None:
                print("Fusing vertical stabilizer...")
                fuse_result = combined.fuse(vertical)
                combined = self._ensure_solid(fuse_result)

            if combined is None:
                print("Failed to combine solids")
                return None

            # 6. Apply fillet for C1 continuity at junctions
            if fillet_radius > 0:
                print(f"Applying fillet (radius={fillet_radius}m)...")
                try:
                    # Find junction edges (at fuselage side)
                    fuse_bb = fuselage.bounding_box()
                    junction_y = (fuse_bb.max.Y - fuse_bb.min.Y) / 2
                    junction_edges = self._find_junction_edges(
                        combined, junction_y
                    )

                    if junction_edges:
                        filleted = combined.fillet(fillet_radius, junction_edges)
                        combined = self._ensure_solid(filleted)
                        print(f"Fillet applied to {len(junction_edges)} edges")
                    else:
                        print("No junction edges found, skipping fillet")
                except Exception as e:
                    print(f"Fillet failed (returning unfilleted): {e}")
                    # Return unfilleted solid as per user requirement

            print("Aircraft generation complete!")
            return combined

        except Exception as e:
            print(f"Aircraft generation failed: {e}")
            import traceback
            traceback.print_exc()
            return None

    def visualize_geometry(self, shape, title: str = "Geometry") -> None:
        """Visualize geometry using build123d's show() function.

        Note: This requires ocp_vscode extension in VS Code,
        or will fall back to matplotlib visualization.

        Args:
            shape: build123d Solid or Part to visualize.
            title: Title for the visualization.
        """
        if not BUILD123D_AVAILABLE or shape is None:
            print("Cannot visualize: build123d not available or shape is None")
            return

        try:
            # Try to use build123d's show() function
            from build123d import show

            if show is not None:
                print(f"Visualizing: {title}")
                print("Use VS Code with ocp_vscode extension to view the geometry")
                show(shape)
            else:
                # Fallback: export to STEP and suggest viewing tools
                print(f"Visualization: {title}")
                print("build123d show() not available.")
                print("Export to STEP/STL and use a CAD viewer.")
                print("Suggested viewers: FreeCAD, Gmsh, ParaView")

        except Exception as e:
            print(f"Visualization failed: {e}")

    def plot_airfoil_comparison(
        self, coords1, coords2, label1="Airfoil 1", label2="Airfoil 2"
    ) -> None:
        """Plot two airfoil coordinate sets for comparison.

        Args:
            coords1: First airfoil coordinates (N x 2 array).
            coords2: Second airfoil coordinates (N x 2 array).
            label1: Label for first airfoil.
            label2: Label for second airfoil.
        """
        try:
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots(1, 2, figsize=(12, 5))

            # Plot 1: Overlaid airfoils
            ax[0].plot(coords1[:, 0], coords1[:, 1], "b-", label=label1, linewidth=2)
            ax[0].plot(coords2[:, 0], coords2[:, 1], "r--", label=label2, linewidth=2)
            ax[0].set_aspect("equal", adjustable="box")
            ax[0].set_xlabel("x/c")
            ax[0].set_ylabel("y/c")
            ax[0].set_title("Airfoil Comparison")
            ax[0].legend()
            ax[0].grid(True, alpha=0.3)

            # Plot 2: Difference
            # Interpolate to common x coordinates
            from scipy import interpolate

            # Create common x grid
            x_common = np.linspace(0, 1, 100)

            # Interpolate both airfoils
            f1 = interpolate.interp1d(coords1[:, 0], coords1[:, 1], kind="cubic", fill_value="extrapolate")
            f2 = interpolate.interp1d(coords2[:, 0], coords2[:, 1], kind="cubic", fill_value="extrapolate")

            y1_interp = f1(x_common)
            y2_interp = f2(x_common)
            diff = y1_interp - y2_interp

            ax[1].plot(x_common, diff, "g-", linewidth=2)
            ax[1].set_xlabel("x/c")
            ax[1].set_ylabel("y1 - y2")
            ax[1].set_title("Difference (y1 - y2)")
            ax[1].grid(True, alpha=0.3)
            ax[1].axhline(y=0, color="k", linestyle="-", alpha=0.5)

            plt.suptitle(f"{label1} vs {label2}")
            plt.tight_layout()
            plt.show()

        except ImportError:
            print("matplotlib or scipy not available for plotting.")
            print("Install with: pip install matplotlib scipy")


if __name__ == "__main__":
    if not BUILD123D_AVAILABLE:
        print("build123d not available. Install with: pip install build123d")
    else:
        print(f"build123d version: {BUILD123D_AVAILABLE}")

        generator = Build123DGenerator()

        coords = generator._naca4_coordinates("2412", 50)
        print(f"Generated {len(coords)} airfoil coordinates")
        print(f"First point: {coords[0]}")
        print(f"Last point: {coords[-1]}")
