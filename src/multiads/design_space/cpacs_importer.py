"""CPACS Importer for MultiADS Assembly.

This module provides the CPACSImporter class which inherits from CPACSStructureData
and enables loading aircraft configurations from CPACS files into MultiADS
Assembly components. It also supports creating VariableFloat and VariableFloatNP
objects linked to CPACS elements for optimization workflows.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
from lxml import etree
from numpy.typing import NDArray

from multiads.design_space.cpacs_design_space import CPACSDesignSpace
from multiads.design_space.cpacs_path_template import CPACSPathTemplate
from multiads.design_space.cpacs_structure_data import CPACSStructureData
from multiads.design_space.path_templates_cpacs35 import CPACS_TEMPLATES
from multiads.design_space.schema_validator import SchemaValidator
from multiads.scenario import VariableFloat, VariableFloatNP, Variable

if TYPE_CHECKING:
    from multiads.assembly import (
        Aircraft,
        Fuselage,
        Nacelle,
        Wing,
        CombustionEngine,
        ElectricEngine,
        Section,
        Span,
        Airfoil,
    )


logger = logging.getLogger(__name__)


class CPACSImporter(CPACSStructureData):
    """Import aircraft configurations from CPACS files to MultiADS Assembly.

    This class extends CPACSStructureData to provide:
    - Schema validation (optional, using bundled XSD)
    - Component creation (Wing, Fuselage, Nacelle, Engine, etc.)
    - Variable creation (VariableFloat, VariableFloatNP) linked to CPACS
    - Write-back capability for optimized designs

    Args:
        file_name: The name of the CPACS file.
        path: The path where the CPACS file is located.
            If None, uses the current directory.
        validate: Whether to validate the XML against the schema.
        schema_path: Path to the XSD schema file.
            If None, uses the bundled schema.
        load_tigl: Whether to load TiGL for geometry (experimental).

    Example:
        >>> importer = CPACSImporter("aircraft.xml", validate=True)
        >>> wing = importer.load_wing("wing_main")
        >>> fuselage = importer.load_fuselage("fuselage_1")
        >>> print(importer.variables)
        {'wing_main_root_chord': VariableFloat, ...}

        >>> # After optimization
        >>> importer.sync_to_cpacs()
        >>> importer.write_optimized("aircraft_optimized.xml")
    """

    def __init__(
        self,
        file_name: str,
        path: Path | None = None,
        validate: bool = True,
        schema_path: Path | None = None,
        load_tigl: bool = False,
    ) -> None:
        super().__init__(file_name, path, load_tigl)

        self._schema_validator = SchemaValidator(
            schema_path=schema_path,
            validate=validate,
        )

        if validate:
            self._schema_validator.validate(self._CPACSStructureData__tree)

        self._variables: dict[str, Variable] = {}
        self._unmapped_warnings: list[str] = []

    @property
    def variables(self) -> dict[str, Variable]:
        """Get all created variables mapped to their names."""
        return self._variables

    @property
    def unmapped_warnings(self) -> list[str]:
        """Get list of warnings for unmapped parameters."""
        return self._unmapped_warnings

    def _log_unmapped(self, xpath: str, attribute: str, reason: str) -> None:
        """Log a warning for an unmapped parameter.

        Args:
            xpath: The XPath that could not be mapped.
            attribute: The attribute name that was attempted.
            reason: Reason why mapping failed.
        """
        msg = (
            f"CPACS parameter '{xpath}' could not be mapped to "
            f"Assembly attribute '{attribute}'. {reason}"
        )
        logger.warning(msg)
        self._unmapped_warnings.append(msg)

    def create_variable_float(
        self,
        xpath: str,
        name: str,
        lower_bound: float | None = None,
        upper_bound: float | None = None,
        unit: str | None = None,
    ) -> VariableFloat:
        """Create a VariableFloat from a CPACS XPath.

        Args:
            xpath: The XPath to the XML element.
            name: Variable name in MultiADS.
            lower_bound: Optimization lower bound.
            upper_bound: Optimization upper bound.
            unit: Unit type ('deg' for degrees, 'm' for meters, etc.)

        Returns:
            VariableFloat with the extracted value.

        Raises:
            ValueError: If the XPath returns no elements or multiple elements.
        """
        element = self._get_xml_element(xpath)
        if element is None:
            msg = f"XPath '{xpath}' returned no elements"
            raise ValueError(msg)
        value = self._parse_element_value(element)

        if isinstance(value, np.ndarray) and value.size > 1:
            logger.warning(
                f"Variable '{name}' has multiple values but is created as VariableFloat. "
                f"Consider using create_variable_float_np instead."
            )
            value = value[0] if value.size > 0 else 0.0

        var = VariableFloat(
            name=name,
            value=float(value),
            lb=lower_bound,
            ub=upper_bound,
        )

        var.cpacs = {
            "xpath": xpath,
            "unit": unit,
            "element": name,
        }

        self._variables[name] = var
        return var

    def create_variable_float_np(
        self,
        xpath: str,
        name: str,
        lower_bound: float | None = None,
        upper_bound: float | None = None,
        unit: str | None = None,
    ) -> VariableFloatNP:
        """Create a VariableFloatNP from a CPACS XPath or group.

        Args:
            xpath: The XPath to the XML element or group pattern.
            name: Variable name in MultiADS.
            lower_bound: Optimization lower bound.
            upper_bound: Optimization upper bound.
            unit: Unit type ('deg' for degrees, 'm' for meters, etc.)

        Returns:
            VariableFloatNP with the extracted values.
        """
        elements = self._find_xpath(xpath)

        if len(elements) == 0:
            msg = f"XPath '{xpath}' returned no elements"
            raise ValueError(msg)

        values = [self._parse_element_value(elem) for elem in elements]
        flat_values: list[float] = []
        for v in values:
            if isinstance(v, np.ndarray):
                flat_values.extend(v.flatten().tolist())
            else:
                flat_values.append(float(v))

        var = VariableFloatNP(
            name=name,
            value=np.array(flat_values),
            lb=lower_bound,
            ub=upper_bound,
        )

        var.cpacs = {
            "xpath": xpath,
            "unit": unit,
            "element": name,
            "count": len(elements),
        }

        self._variables[name] = var
        return var

    def _create_variable_from_attribute(
        self,
        wing_uid: str,
        section_uid: str,
        attribute_name: str,
        name: str,
        lower_bound: float | None = None,
        upper_bound: float | None = None,
        unit: str | None = None,
    ) -> VariableFloat:
        """Create a VariableFloat from an XML attribute.

        Args:
            wing_uid: The wing uID.
            section_uid: The section uID.
            attribute_name: The attribute name to extract.
            name: Variable name in MultiADS.
            lower_bound: Optimization lower bound.
            upper_bound: Optimization upper bound.
            unit: Unit type ('deg' for degrees, 'm' for meters, etc.)

        Returns:
            VariableFloat with the extracted value.
        """
        xpath = CPACS_TEMPLATES.wing.WING_SECTION_TRANSFORMATION.get_xpath([wing_uid, section_uid])
        rotation_elem = self._get_single_element(xpath)
        
        value = 0.0
        if rotation_elem is not None:
            attr_value = rotation_elem.get(attribute_name)
            if attr_value:
                value = float(attr_value)

        full_xpath = f"{xpath}/@{attribute_name}"

        var = VariableFloat(
            name=name,
            value=value,
            lb=lower_bound,
            ub=upper_bound,
        )

        var.cpacs = {
            "xpath": full_xpath,
            "unit": unit,
            "element": name,
            "attribute": attribute_name,
        }

        self._variables[name] = var
        return var

    def _parse_element_value(self, element: Any) -> float | np.ndarray:
        """Parse an XML element to extract its numeric value.

        Args:
            element: XML element to parse.

        Returns:
            Numeric value(s) from the element.
        """
        if element is None:
            return np.nan

        text = element.text
        if text is None:
            return np.nan

        text = text.strip()
        if not text:
            return np.nan

        try:
            parts = text.split()
            if len(parts) == 1:
                return float(parts[0])
            return np.array([float(p) for p in parts])
        except ValueError:
            return text

    def load_wing(self, wing_uid: str) -> Wing:
        """Load a wing from CPACS.

        Args:
            wing_uid: The uID of the wing in CPACS.

        Returns:
            Wing component with all sections, spans, and airfoils.
        """
        from multiads.assembly import (
            Wing,
            Section,
            Span,
            AirfoilNACA4,
            MovableSurface,
        )

        wing_xpath = CPACS_TEMPLATES.wing.WING.get_xpath([wing_uid])
        wing_elements = self._find_xpath(wing_xpath)

        if not wing_elements:
            msg = f"Wing with uID '{wing_uid}' not found in CPACS"
            raise ValueError(msg)

        sections = self._load_wing_sections(wing_uid)
        spans = self._load_wing_spans(wing_uid)

        symmetry = self._get_wing_symmetry(wing_uid)
        name = wing_uid

        wing = Wing(
            name=name,
            sections=sections,
            spans=spans,
            symmetry=symmetry,
        )

        logger.info(f"Loaded wing '{wing_uid}' with {len(sections)} sections")
        return wing

    def _load_wing_sections(self, wing_uid: str) -> list[Section]:
        """Load all sections for a wing.

        Args:
            wing_uid: The wing uID.

        Returns:
            List of Section components.
        """
        from multiads.assembly import Section, AirfoilNACA4

        sections_path = CPACS_TEMPLATES.wing.WING_SECTIONS.get_xpath([wing_uid])
        section_elements = self._find_xpath(sections_path)

        sections = []
        for i, section_elem in enumerate(section_elements):
            section_uid = section_elem.get("uID", f"{wing_uid}_section_{i}")

            chord = self._get_section_chord(wing_uid, section_uid)
            twist = self._get_section_twist(wing_uid, section_uid)
            airfoil_uid = self._get_section_airfoil(wing_uid, section_uid)

            airfoil = self._load_airfoil(airfoil_uid) if airfoil_uid else AirfoilNACA4(
                name=f"{section_uid}_airfoil",
                m=4,
                p=4,
                t=12,
            )

            # Create variable for chord using element scaling xpath
            element_scaling_xpath = (
                CPACS_TEMPLATES.wing.WING_SECTION.get_xpath([wing_uid, section_uid])
                + "/elements/element/transformation/scaling/x"
            )
            try:
                chord_var = self.create_variable_float(
                    element_scaling_xpath,
                    name=f"{section_uid}_chord",
                    lower_bound=0.5,
                    upper_bound=10.0,
                    unit="m",
                )
            except ValueError:
                # Skip variable creation if element not found
                pass

            # Create variable for twist using rotation y (which is the twist in CPACS)
            twist_xpath = (
                CPACS_TEMPLATES.wing.WING_SECTION_TRANSFORMATION.get_xpath(
                    [wing_uid, section_uid]
                )
                + "/y"
            )
            try:
                twist_var = self.create_variable_float(
                    twist_xpath,
                    name=f"{section_uid}_twist",
                    lower_bound=-20.0,
                    upper_bound=20.0,
                    unit="deg",
                )
            except ValueError:
                # Skip variable creation if rotation not found
                pass

            section = Section(
                name=section_uid,
                airfoil=airfoil,
                chord=chord,
                twist=twist,
            )
            sections.append(section)

        return sections

    def _load_wing_spans(self, wing_uid: str) -> list[Span]:
        """Load all span segments for a wing.

        Args:
            wing_uid: The wing uID.

        Returns:
            List of Span components.
        """
        from multiads.assembly import Span

        segments_path = CPACS_TEMPLATES.wing.WING_SEGMENTS.get_xpath([wing_uid])
        segment_elements = self._find_xpath(segments_path)

        spans = []
        for i, segment_elem in enumerate(segment_elements):
            segment_uid = segment_elem.get("uID", f"{wing_uid}_segment_{i}")

            # Try to get length from segment, otherwise compute from section positions
            length = self._get_segment_length(wing_uid, segment_uid)
            sweep = self._get_segment_sweep(wing_uid, segment_uid)
            dihed = self._get_segment_dihedral(wing_uid, segment_uid)

            # If length is default (1.0), try to compute from section positions
            if length == 1.0 and sweep == 0.0 and dihed == 0.0:
                length, sweep, dihed = self._compute_segment_geometry(
                    wing_uid, segment_uid, segment_elem
                )

            # Only create variables if the elements exist in CPACS
            length_xpath = segments_path + f"[@uID='{segment_uid}']/length"
            if self._get_single_element(length_xpath) is not None:
                self.create_variable_float(
                    length_xpath,
                    name=f"{segment_uid}_length",
                    lower_bound=0.1,
                    upper_bound=50.0,
                    unit="m",
                )

            sweep_xpath = segments_path + f"[@uID='{segment_uid}']/sweepAngle"
            if self._get_single_element(sweep_xpath) is not None:
                self.create_variable_float(
                    sweep_xpath,
                    name=f"{segment_uid}_sweep",
                    lower_bound=-30.0,
                    upper_bound=45.0,
                    unit="deg",
                )

            span = Span(
                name=segment_uid,
                length=length,
                sweep=sweep,
                dihed=dihed,
            )
            spans.append(span)

        return spans

    def _compute_segment_geometry(
        self, wing_uid: str, segment_uid: str, segment_elem: Any
    ) -> tuple[float, float, float]:
        """Compute segment geometry from section positions.

        Args:
            wing_uid: The wing uID.
            segment_uid: The segment uID.
            segment_elem: The segment XML element.

        Returns:
            Tuple of (length, sweep, dihedral).
        """
        from_element_uid = segment_elem.findtext("fromElementUID")
        to_element_uid = segment_elem.findtext("toElementUID")

        if not from_element_uid or not to_element_uid:
            return 1.0, 0.0, 0.0

        # Get positions of the two sections
        from_pos = self._get_element_position(from_element_uid)
        to_pos = self._get_element_position(to_element_uid)

        if from_pos is None or to_pos is None:
            return 1.0, 0.0, 0.0

        # Calculate length (distance between points along span, which is y-direction)
        length = abs(to_pos[1] - from_pos[1])  # Use y-distance as span length

        # Calculate sweep (x-offset per unit span)
        dx = to_pos[0] - from_pos[0]
        if length > 0:
            sweep_rad = np.arctan2(dx, length)
            sweep = np.degrees(sweep_rad)
        else:
            sweep = 0.0

        # Calculate dihedral (z-offset per unit span)  
        dz = to_pos[2] - from_pos[2]
        if length > 0:
            dihed_rad = np.arctan2(dz, length)
            dihed = np.degrees(dihed_rad)
        else:
            dihed = 0.0

        return length, sweep, dihed

    def _get_element_position(self, element_uid: str) -> tuple[float, float, float] | None:
        """Get the position of an element from its section's transformation.

        Args:
            element_uid: The element uID.

        Returns:
            Tuple of (x, y, z) position or None if not found.
        """
        # Find the section containing this element
        xpath = f".//element[@uID='{element_uid}']"
        element = self._get_single_element(xpath)
        if element is None:
            return None

        # Get the parent section
        section = element.getparent()
        while section is not None and section.tag != "section":
            section = section.getparent()

        if section is None:
            return None

        # Get translation from section transformation
        trans_xpath = f".//section[@uID='{section.get('uID')}']/transformation/translation"
        trans_elem = self._get_single_element(trans_xpath)
        if trans_elem is None:
            return None

        try:
            x = float(trans_elem.findtext("x", "0"))
            y = float(trans_elem.findtext("y", "0"))
            z = float(trans_elem.findtext("z", "0"))
            return (x, y, z)
        except (ValueError, TypeError):
            return None

    def _get_section_chord(self, wing_uid: str, section_uid: str) -> float:
        """Get chord value for a section from element scaling."""
        xpath = CPACS_TEMPLATES.wing.WING_SECTION.get_xpath([wing_uid, section_uid])
        element_scaling_xpath = f"{xpath}/elements/element/transformation/scaling/x"
        chord_elem = self._get_single_element(element_scaling_xpath)
        if chord_elem is not None and len(chord_elem) >= 0 and chord_elem.text is not None:
            try:
                return float(chord_elem.text)
            except (ValueError, TypeError):
                pass
        return 1.0

    def _get_section_twist(self, wing_uid: str, section_uid: str) -> float:
        """Get twist value for a section (in degrees) from rotation/y element."""
        xpath = CPACS_TEMPLATES.wing.WING_SECTION_TRANSFORMATION.get_xpath([wing_uid, section_uid])
        rotation_y_xpath = f"{xpath}/y"
        twist_elem = self._get_single_element(rotation_y_xpath)
        if twist_elem is not None and len(twist_elem) >= 0 and twist_elem.text is not None:
            try:
                return float(twist_elem.text)
            except (ValueError, TypeError):
                pass
        return 0.0

    def _get_section_airfoil(self, wing_uid: str, section_uid: str) -> str | None:
        """Get airfoil uID reference for a section."""
        xpath = CPACS_TEMPLATES.wing.WING_AIRFOIL_REF.get_xpath([wing_uid, section_uid])
        profile_elem = self._get_single_element(xpath)
        if profile_elem is not None and len(profile_elem) >= 0:
            return profile_elem.get("uID")
        return None

    def _get_segment_length(self, wing_uid: str, segment_uid: str) -> float:
        """Get segment length."""
        xpath = CPACS_TEMPLATES.wing.WING_SEGMENT.get_xpath([wing_uid, segment_uid])
        length_elem = self._get_single_element(f"{xpath}/length")
        if length_elem is not None and len(length_elem) >= 0:
            try:
                return float(length_elem.text)
            except (ValueError, TypeError):
                pass
        return 1.0

    def _get_segment_sweep(self, wing_uid: str, segment_uid: str) -> float:
        """Get segment sweep angle (in degrees)."""
        xpath = CPACS_TEMPLATES.wing.WING_SEGMENT.get_xpath([wing_uid, segment_uid])
        sweep_elem = self._get_single_element(f"{xpath}/sweepAngle")
        if sweep_elem is not None and len(sweep_elem) >= 0:
            try:
                return float(sweep_elem.text)
            except (ValueError, TypeError):
                pass
        return 0.0

    def _get_segment_dihedral(self, wing_uid: str, segment_uid: str) -> float:
        """Get segment dihedral angle (in degrees)."""
        xpath = CPACS_TEMPLATES.wing.WING_SEGMENT.get_xpath([wing_uid, segment_uid])
        dihed_elem = self._get_single_element(f"{xpath}/dihedralAngle")
        if dihed_elem is not None and len(dihed_elem) >= 0:
            try:
                return float(dihed_elem.text)
            except (ValueError, TypeError):
                pass
        return 0.0

    def _get_wing_symmetry(self, wing_uid: str) -> bool:
        """Check if wing has symmetry."""
        xpath = CPACS_TEMPLATES.wing.WING.get_xpath([wing_uid])
        wing_elem = self._get_single_element(xpath)
        if wing_elem is not None:
            # Symmetry is an attribute in CPACS 3.5
            symmetry_attr = wing_elem.get("symmetry")
            if symmetry_attr:
                text = symmetry_attr.strip().lower()
                return text in ("true", "1", "mirrorz", "mirrory", "x-z-plane")
        return False

    def _get_single_element(self, xpath: str) -> Any:
        """Get a single element from XPath or None."""
        elements = self._find_xpath(xpath)
        return elements[0] if elements else None

    def _load_airfoil(self, airfoil_uid: str) -> Airfoil:
        """Load an airfoil profile.

        Args:
            airfoil_uid: The airfoil uID.

        Returns:
            Airfoil component.
        """
        from multiads.assembly import AirfoilNACA4, AirfoilFile, AirfoilCST

        cst_lower_path = CPACS_TEMPLATES.wing.PROFILE_CST_LOWER.get_xpath([airfoil_uid])
        cst_upper_path = CPACS_TEMPLATES.wing.PROFILE_CST_UPPER.get_xpath([airfoil_uid])

        lower_elem = self._get_single_element(cst_lower_path)
        upper_elem = self._get_single_element(cst_upper_path)

        if lower_elem is not None and upper_elem is not None:
            lower_vals = self._parse_element_value(lower_elem)
            upper_vals = self._parse_element_value(upper_elem)

            if isinstance(lower_vals, np.ndarray) and isinstance(upper_vals, np.ndarray):
                self.create_variable_float_np(
                    cst_lower_path,
                    name=f"{airfoil_uid}_cst_lower",
                )
                self.create_variable_float_np(
                    cst_upper_path,
                    name=f"{airfoil_uid}_cst_upper",
                )

                return AirfoilCST(
                    name=airfoil_uid,
                    cst_coeffs_lower=lower_vals,
                    cst_coeffs_upper=upper_vals,
                    n_points=100,
                )
            else:
                logger.warning(
                    f"CST coefficients for '{airfoil_uid}' could not be parsed, "
                    f"using NACA 4412"
                )

        naca_path = CPACS_TEMPLATES.wing.PROFILE_NACA.get_xpath([airfoil_uid])
        naca_elem = self._get_single_element(naca_path)

        if naca_elem is not None:
            m = float(naca_elem.get("m", "0"))
            p = float(naca_elem.get("p", "0"))
            t = float(naca_elem.get("t", "12"))
            return AirfoilNACA4(name=airfoil_uid, m=int(m), p=int(p), t=int(t))

        logger.warning(f"Could not load airfoil '{airfoil_uid}', using default NACA 4412")
        self._unmapped_warnings.append(f"Airfoil '{airfoil_uid}' not fully supported")
        return AirfoilNACA4(name=airfoil_uid, m=4, p=4, t=12)

    def load_fuselage(self, fuselage_uid: str) -> Fuselage:
        """Load a fuselage from CPACS.

        Args:
            fuselage_uid: The uID of the fuselage in CPACS.

        Returns:
            Fuselage component.
        """
        from multiads.assembly import Fuselage

        fuselage_xpath = CPACS_TEMPLATES.fuselage.FUSELAGE.get_xpath([fuselage_uid])
        fuselage_elements = self._find_xpath(fuselage_xpath)

        if not fuselage_elements:
            msg = f"Fuselage with uID '{fuselage_uid}' not found in CPACS"
            raise ValueError(msg)

        length = self._get_fuselage_length(fuselage_uid)
        width = self._get_fuselage_width(fuselage_uid)
        height = self._get_fuselage_height(fuselage_uid)
        wetted_area = self._get_fuselage_wetted_area(fuselage_uid)
        cabin_volume = self._get_fuselage_cabin_volume(fuselage_uid)

        # Only create variables if the elements exist in CPACS
        length_xpath = CPACS_TEMPLATES.fuselage.FUSELAGE_GEOMETRY_LENGTH.get_xpath([fuselage_uid])
        if self._get_single_element(length_xpath) is not None:
            self.create_variable_float(
                length_xpath,
                name=f"{fuselage_uid}_length",
                lower_bound=10.0,
                upper_bound=100.0,
                unit="m",
            )

        width_xpath = CPACS_TEMPLATES.fuselage.FUSELAGE_GEOMETRY_WIDTH.get_xpath([fuselage_uid])
        if self._get_single_element(width_xpath) is not None:
            self.create_variable_float(
                width_xpath,
                name=f"{fuselage_uid}_width",
                lower_bound=1.0,
                upper_bound=10.0,
                unit="m",
            )

        height_xpath = CPACS_TEMPLATES.fuselage.FUSELAGE_GEOMETRY_HEIGHT.get_xpath([fuselage_uid])
        if self._get_single_element(height_xpath) is not None:
            self.create_variable_float(
                height_xpath,
                name=f"{fuselage_uid}_height",
                lower_bound=1.0,
                upper_bound=10.0,
                unit="m",
            )

        fuselage = Fuselage(
            name=fuselage_uid,
            wetted_area=wetted_area,
            maximum_width=width,
            maximum_height=height,
            length=length,
            volume_pressurized_cabin=cabin_volume,
        )

        logger.info(f"Loaded fuselage '{fuselage_uid}'")
        return fuselage

    def _get_fuselage_length(self, fuselage_uid: str) -> float:
        """Get fuselage length."""
        xpath = CPACS_TEMPLATES.fuselage.FUSELAGE_GEOMETRY_LENGTH.get_xpath([fuselage_uid])
        elem = self._get_single_element(xpath)
        if elem is not None and len(elem) >= 0 and elem.text is not None:
            try:
                return float(elem.text)
            except (ValueError, TypeError):
                pass
        return 30.0

    def _get_fuselage_width(self, fuselage_uid: str) -> float:
        """Get fuselage maximum width."""
        xpath = CPACS_TEMPLATES.fuselage.FUSELAGE_GEOMETRY_WIDTH.get_xpath([fuselage_uid])
        elem = self._get_single_element(xpath)
        if elem is not None and len(elem) >= 0 and elem.text is not None:
            try:
                return float(elem.text)
            except (ValueError, TypeError):
                pass
        return 5.0

    def _get_fuselage_height(self, fuselage_uid: str) -> float:
        """Get fuselage maximum height."""
        xpath = CPACS_TEMPLATES.fuselage.FUSELAGE_GEOMETRY_HEIGHT.get_xpath([fuselage_uid])
        elem = self._get_single_element(xpath)
        if elem is not None and len(elem) >= 0 and elem.text is not None:
            try:
                return float(elem.text)
            except (ValueError, TypeError):
                pass
        return 5.0

    def _get_fuselage_wetted_area(self, fuselage_uid: str) -> float:
        """Get fuselage wetted area."""
        xpath = CPACS_TEMPLATES.fuselage.FUSELAGE_GEOMETRY_AREA.get_xpath([fuselage_uid])
        elem = self._get_single_element(xpath)
        if elem is not None and len(elem) >= 0 and elem.text is not None:
            try:
                return float(elem.text)
            except (ValueError, TypeError):
                pass
        return 200.0

    def _get_fuselage_cabin_volume(self, fuselage_uid: str) -> float:
        """Get fuselage cabin volume."""
        xpath = CPACS_TEMPLATES.fuselage.FUSELAGE_CABIN_VOLUME.get_xpath([fuselage_uid])
        elem = self._get_single_element(xpath)
        if elem is not None and len(elem) >= 0 and elem.text is not None:
            try:
                return float(elem.text)
            except (ValueError, TypeError):
                pass
        return 1.0

    def load_nacelle(self, nacelle_uid: str) -> Nacelle:
        """Load a nacelle from CPACS.

        Args:
            nacelle_uid: The uID of the nacelle in CPACS.

        Returns:
            Nacelle component.
        """
        from multiads.assembly import Nacelle

        nacelle_xpath = CPACS_TEMPLATES.nacelle.NACELLE.get_xpath([nacelle_uid])
        nacelle_elements = self._find_xpath(nacelle_xpath)

        if not nacelle_elements:
            msg = f"Nacelle with uID '{nacelle_uid}' not found in CPACS"
            raise ValueError(msg)

        length = self._get_nacelle_length(nacelle_uid)
        width = self._get_nacelle_width(nacelle_uid)
        height = self._get_nacelle_height(nacelle_uid)
        wetted_area = self._get_nacelle_wetted_area(nacelle_uid)

        self.create_variable_float(
            CPACS_TEMPLATES.nacelle.NACELLE_GEOMETRY_LENGTH.get_xpath([nacelle_uid]),
            name=f"{nacelle_uid}_length",
            lower_bound=0.5,
            upper_bound=20.0,
            unit="m",
        )

        nacelle = Nacelle(
            name=nacelle_uid,
            wetted_area=wetted_area,
            maximum_width=width,
            maximum_height=height,
            length=length,
            mass=100.0,
            portion_laminar_flow=0.0,
        )

        logger.info(f"Loaded nacelle '{nacelle_uid}'")
        return nacelle

    def _get_nacelle_length(self, nacelle_uid: str) -> float:
        """Get nacelle length."""
        xpath = CPACS_TEMPLATES.nacelle.NACELLE_GEOMETRY_LENGTH.get_xpath([nacelle_uid])
        elem = self._get_single_element(xpath)
        return self._parse_element_value(elem) if elem is not None else 3.0

    def _get_nacelle_width(self, nacelle_uid: str) -> float:
        """Get nacelle maximum width."""
        xpath = CPACS_TEMPLATES.nacelle.NACELLE_GEOMETRY_WIDTH.get_xpath([nacelle_uid])
        elem = self._get_single_element(xpath)
        return self._parse_element_value(elem) if elem is not None else 1.5

    def _get_nacelle_height(self, nacelle_uid: str) -> float:
        """Get nacelle maximum height."""
        xpath = CPACS_TEMPLATES.nacelle.NACELLE_GEOMETRY_HEIGHT.get_xpath([nacelle_uid])
        elem = self._get_single_element(xpath)
        return self._parse_element_value(elem) if elem is not None else 1.5

    def _get_nacelle_wetted_area(self, nacelle_uid: str) -> float:
        """Get nacelle wetted area."""
        xpath = CPACS_TEMPLATES.nacelle.NACELLE_GEOMETRY_AREA.get_xpath([nacelle_uid])
        elem = self._get_single_element(xpath)
        return self._parse_element_value(elem) if elem is not None else 15.0

    def load_engine(self, engine_uid: str) -> CombustionEngine | ElectricEngine:
        """Load an engine from CPACS.

        Args:
            engine_uid: The uID of the engine in CPACS.

        Returns:
            CombustionEngine or ElectricEngine based on engine type.
        """
        from multiads.assembly import CombustionEngine, ElectricEngine

        engine_xpath = CPACS_TEMPLATES.engine.ENGINE.get_xpath([engine_uid])
        engine_elements = self._find_xpath(engine_xpath)

        if not engine_elements:
            msg = f"Engine with uID '{engine_uid}' not found in CPACS"
            raise ValueError(msg)

        mass = self._get_engine_mass(engine_uid)
        engine_type = self._get_engine_type(engine_uid)

        self.create_variable_float(
            CPACS_TEMPLATES.engine.ENGINE_MASS.get_xpath([engine_uid]),
            name=f"{engine_uid}_mass",
            lower_bound=100.0,
            upper_bound=10000.0,
            unit="kg",
        )

        if "electric" in engine_type.lower() or "hybrid" in engine_type.lower():
            engine = ElectricEngine(name=engine_uid, mass=mass)
            logger.info(f"Loaded electric engine '{engine_uid}'")
        else:
            engine = CombustionEngine(name=engine_uid, mass=mass)
            logger.info(f"Loaded combustion engine '{engine_uid}'")

        return engine

    def _get_engine_mass(self, engine_uid: str) -> float:
        """Get engine mass."""
        xpath = CPACS_TEMPLATES.engine.ENGINE_MASS.get_xpath([engine_uid])
        elem = self._get_single_element(xpath)
        return self._parse_element_value(elem) if elem is not None else 500.0

    def _get_engine_type(self, engine_uid: str) -> str:
        """Get engine type description."""
        xpath = CPACS_TEMPLATES.engine.ENGINE_TYPE.get_xpath([engine_uid])
        elem = self._get_single_element(xpath)
        return elem.text.strip() if elem is not None and elem.text else "unknown"

    def load_aircraft(
        self,
        aircraft_uid: str = "Aircraft_1",
        load_wings: bool = True,
        load_fuselages: bool = True,
        load_nacelles: bool = True,
        load_engines: bool = True,
    ) -> Aircraft:
        """Load a complete aircraft from CPACS.

        Args:
            aircraft_uid: The uID of the aircraft in CPACS.
            load_wings: Whether to load wing components.
            load_fuselages: Whether to load fuselage components.
            load_nacelles: Whether to load nacelle components.
            load_engines: Whether to load engine components.

        Returns:
            Aircraft component with all loaded children.
        """
        from multiads.assembly import Aircraft

        aircraft_xpath = CPACS_TEMPLATES.aircraft.AIRCRAFT_MODEL.get_xpath([aircraft_uid])
        aircraft_elements = self._find_xpath(aircraft_xpath)

        if not aircraft_elements:
            msg = f"Aircraft with uID '{aircraft_uid}' not found in CPACS"
            raise ValueError(msg)

        children = []
        ref_area = self._get_aircraft_reference_area()
        ref_length = self._get_aircraft_reference_length()

        if load_wings:
            wing_uids = self._get_component_uids("wings/wing")
            for wing_uid in wing_uids:
                try:
                    wing = self.load_wing(wing_uid)
                    children.append(wing)
                except Exception as e:
                    logger.warning(f"Failed to load wing '{wing_uid}': {e}")

        if load_fuselages:
            fuselage_uids = self._get_component_uids("fuselages/fuselage")
            for fuselage_uid in fuselage_uids:
                try:
                    fuselage = self.load_fuselage(fuselage_uid)
                    children.append(fuselage)
                except Exception as e:
                    logger.warning(f"Failed to load fuselage '{fuselage_uid}': {e}")

        if load_nacelles:
            nacelle_uids = self._get_component_uids("nacelles/nacelle")
            for nacelle_uid in nacelle_uids:
                try:
                    nacelle = self.load_nacelle(nacelle_uid)
                    children.append(nacelle)
                except Exception as e:
                    logger.warning(f"Failed to load nacelle '{nacelle_uid}': {e}")

        if load_engines:
            engine_uids = self._get_component_uids("engines/engine")
            for engine_uid in engine_uids:
                try:
                    engine = self.load_engine(engine_uid)
                    children.append(engine)
                except Exception as e:
                    logger.warning(f"Failed to load engine '{engine_uid}': {e}")

        aircraft = Aircraft(
            name=aircraft_uid,
            s_ref=ref_area,
            l_ref=ref_length,
        )
        aircraft.children = [c.name for c in children]

        logger.info(
            f"Loaded aircraft '{aircraft_uid}' with {len(children)} components"
        )
        return aircraft

    def _get_component_uids(self, relative_path: str) -> list[str]:
        """Get all component uIDs for a given path."""
        xpath = f"./vehicles/aircraft/model/{relative_path}"
        elements = self._find_xpath(xpath)
        return [elem.get("uID") for elem in elements if elem.get("uID")]

    def _get_aircraft_reference_area(self) -> float:
        """Get aircraft reference area."""
        xpath = CPACS_TEMPLATES.aircraft.AIRCRAFT_REF_AREA.get_xpath([])
        elem = self._get_single_element(xpath)
        return self._parse_element_value(elem) if elem is not None else 100.0

    def _get_aircraft_reference_length(self) -> float:
        """Get aircraft reference length."""
        xpath = CPACS_TEMPLATES.aircraft.AIRCRAFT_REF_LENGTH.get_xpath([])
        elem = self._get_single_element(xpath)
        return self._parse_element_value(elem) if elem is not None else 10.0

    def sync_to_cpacs(self) -> None:
        """Sync all mapped variables back to CPACS XML tree.

        Only variables that have CPACS metadata are written back.
        Variables without CPACS mapping are skipped (logged as info).
        """
        for var_name, var in self._variables.items():
            if not var.cpacs:
                logger.info(
                    f"Variable '{var_name}' has no CPACS mapping, skipping write."
                )
                continue

            xpath = var.cpacs.get("xpath")
            if not xpath:
                logger.warning(f"Variable '{var_name}' has empty xpath, skipping.")
                continue

            value = var.value
            if isinstance(value, np.ndarray):
                value = value.flatten()

            unit = var.cpacs.get("unit")
            if unit == "deg" and isinstance(value, (float, np.ndarray)):
                if isinstance(value, np.ndarray):
                    value = np.degrees(value)
                else:
                    value = np.degrees(value)

            self._write_to_xpath(xpath, value)

        logger.info(f"Synced {len(self._variables)} variables to CPACS")

    def _write_to_xpath(self, xpath: str, value: float | np.ndarray) -> None:
        """Write a value to an XML element specified by XPath.

        Args:
            xpath: The XPath to the target element.
            value: The value to write.
        """
        elements = self._find_xpath(xpath)

        if not elements:
            logger.warning(f"No element found for xpath: {xpath}")
            return

        for elem in elements:
            if isinstance(value, np.ndarray) and value.size == 1:
                value = float(value.flatten()[0])
            elif isinstance(value, np.ndarray):
                value_str = " ".join(str(v) for v in value.flatten())
                elem.text = value_str
                return

            elem.text = str(float(value))

    def write_optimized(self, output_path: str | Path) -> None:
        """Write the CPACS XML with all optimized values to a new file.

        Args:
            output_path: Path for the new CPACS file.
        """
        self.sync_to_cpacs()
        output_path = Path(output_path)
        self.write_xml(output_path.as_posix())
        logger.info(f"Wrote optimized CPACS to '{output_path}'")

    def create_design_space(self) -> CPACSDesignSpace:
        """Create a GEMSEO DesignSpace from loaded variables.

        Returns:
            CPACSDesignSpace populated with all mapped variables.
        """
        design_space = CPACSDesignSpace(cpacs_structure_data=self)

        for var_name, var in self._variables.items():
            if var.cpacs and var.cpacs.get("xpath"):
                bounds = (var.lb_np, var.ub_np)
                design_space.add_variable(
                    var_name,
                    size=len(var.value_np),
                    value=var.value_np,
                    lower_bound=bounds[0] if bounds[0] is not None else None,
                    upper_bound=bounds[1] if bounds[1] is not None else None,
                )

        logger.info(f"Created design space with {len(self._variables)} variables")
        return design_space
