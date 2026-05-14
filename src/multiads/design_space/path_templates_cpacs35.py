"""CPACS v3.5 XPath path templates.

This module extends the basic CPACS path templates with comprehensive coverage
for CPACS v3.5 schema, enabling mapping of all major aircraft components to
MultiADS assembly components.
"""

from __future__ import annotations

from collections.abc import Sequence

from multiads.design_space.cpacs_path_template import CPACSPathTemplate


class CPACSWingTemplates:
    """XPath templates for wing-related CPACS elements."""

    WING = CPACSPathTemplate(
        "./vehicles/aircraft/model/wings/wing[@uID='{wing_ID}']",
        ["wing_ID"],
    )

    WING_SECTIONS = CPACSPathTemplate(
        "./vehicles/aircraft/model/wings/wing[@uID='{wing_ID}']/sections/section",
        ["wing_ID"],
    )

    WING_SECTION = CPACSPathTemplate(
        "./vehicles/aircraft/model/wings/wing[@uID='{wing_ID}']/"
        "sections/section[@uID='{section_ID}']",
        ["wing_ID", "section_ID"],
    )

    WING_SECTION_GEOMETRY = CPACSPathTemplate(
        "./vehicles/aircraft/model/wings/wing[@uID='{wing_ID}']/"
        "sections/section[@uID='{section_ID}']",
        ["wing_ID", "section_ID"],
    )

    WING_SECTION_TRANSFORMATION = CPACSPathTemplate(
        "./vehicles/aircraft/model/wings/wing[@uID='{wing_ID}']/"
        "sections/section[@uID='{section_ID}']/transformation/rotation",
        ["wing_ID", "section_ID"],
    )

    WING_SEGMENTS = CPACSPathTemplate(
        "./vehicles/aircraft/model/wings/wing[@uID='{wing_ID}']/segments/segment",
        ["wing_ID"],
    )

    WING_SEGMENT = CPACSPathTemplate(
        "./vehicles/aircraft/model/wings/wing[@uID='{wing_ID}']/"
        "segments/segment[@uID='{segment_ID}']",
        ["wing_ID", "segment_ID"],
    )

    WING_POSITIONINGS = CPACSPathTemplate(
        "./vehicles/aircraft/model/wings/wing[@uID='{wing_ID}']/positionings/positioning",
        ["wing_ID"],
    )

    WING_POSITIONING = CPACSPathTemplate(
        "./vehicles/aircraft/model/wings/wing[@uID='{wing_ID}']/"
        "positionings/positioning[@uID='{position_ID}']",
        ["wing_ID", "position_ID"],
    )

    WING_AIRFOIL_REF = CPACSPathTemplate(
        "./vehicles/aircraft/model/wings/wing[@uID='{wing_ID}']/"
        "sections/section[@uID='{section_ID}']/elements/element/profiles/profile",
        ["wing_ID", "section_ID"],
    )

    PROFILE_CST_LOWER = CPACSPathTemplate(
        "./vehicles/profiles/wingAirfoils/wingAirfoil[@uID='{airfoil_ID}']/cst2D/lowerB",
        ["airfoil_ID"],
    )

    PROFILE_CST_UPPER = CPACSPathTemplate(
        "./vehicles/profiles/wingAirfoils/wingAirfoil[@uID='{airfoil_ID}']/cst2D/upperB",
        ["airfoil_ID"],
    )

    PROFILE_CST_COEFFICIENTS = CPACSPathTemplate(
        "./vehicles/profiles/wingAirfoils/wingAirfoil[@uID='{airfoil_ID}']/cst2D",
        ["airfoil_ID"],
    )

    PROFILE_NACA = CPACSPathTemplate(
        "./vehicles/profiles/wingAirfoils/wingAirfoil[@uID='{airfoil_ID}']/naca",
        ["airfoil_ID"],
    )


class CPACSFuselageTemplates:
    """XPath templates for fuselage-related CPACS elements."""

    FUSELAGE = CPACSPathTemplate(
        "./vehicles/aircraft/model/fuselages/fuselage[@uID='{fuselage_ID}']",
        ["fuselage_ID"],
    )

    FUSELAGE_GEOMETRY = CPACSPathTemplate(
        "./vehicles/aircraft/model/fuselages/fuselage[@uID='{fuselage_ID}']/geometry",
        ["fuselage_ID"],
    )

    FUSELAGE_GEOMETRY_LENGTH = CPACSPathTemplate(
        "./vehicles/aircraft/model/fuselages/fuselage[@uID='{fuselage_ID}']/"
        "geometry/length",
        ["fuselage_ID"],
    )

    FUSELAGE_GEOMETRY_WIDTH = CPACSPathTemplate(
        "./vehicles/aircraft/model/fuselages/fuselage[@uID='{fuselage_ID}']/"
        "geometry/width",
        ["fuselage_ID"],
    )

    FUSELAGE_GEOMETRY_HEIGHT = CPACSPathTemplate(
        "./vehicles/aircraft/model/fuselages/fuselage[@uID='{fuselage_ID}']/"
        "geometry/height",
        ["fuselage_ID"],
    )

    FUSELAGE_GEOMETRY_AREA = CPACSPathTemplate(
        "./vehicles/aircraft/model/fuselages/fuselage[@uID='{fuselage_ID}']/"
        "geometry/wettedArea",
        ["fuselage_ID"],
    )

    FUSELAGE_CABIN = CPACSPathTemplate(
        "./vehicles/aircraft/model/fuselages/fuselage[@uID='{fuselage_ID}']/"
        "cabinLayout/cabin",
        ["fuselage_ID"],
    )

    FUSELAGE_CABIN_VOLUME = CPACSPathTemplate(
        "./vehicles/aircraft/model/fuselages/fuselage[@uID='{fuselage_ID}']/"
        "cabinLayout/cabin/volume",
        ["fuselage_ID"],
    )


class CPACSNacelleTemplates:
    """XPath templates for nacelle-related CPACS elements."""

    NACELLE = CPACSPathTemplate(
        "./vehicles/aircraft/model/nacelles/nacelle[@uID='{nacelle_ID}']",
        ["nacelle_ID"],
    )

    NACELLE_GEOMETRY = CPACSPathTemplate(
        "./vehicles/aircraft/model/nacelles/nacelle[@uID='{nacelle_ID}']/geometry",
        ["nacelle_ID"],
    )

    NACELLE_GEOMETRY_LENGTH = CPACSPathTemplate(
        "./vehicles/aircraft/model/nacelles/nacelle[@uID='{nacelle_ID}']/"
        "geometry/length",
        ["nacelle_ID"],
    )

    NACELLE_GEOMETRY_WIDTH = CPACSPathTemplate(
        "./vehicles/aircraft/model/nacelles/nacelle[@uID='{nacelle_ID}']/"
        "geometry/width",
        ["nacelle_ID"],
    )

    NACELLE_GEOMETRY_HEIGHT = CPACSPathTemplate(
        "./vehicles/aircraft/model/nacelles/nacelle[@uID='{nacelle_ID}']/"
        "geometry/height",
        ["nacelle_ID"],
    )

    NACELLE_GEOMETRY_AREA = CPACSPathTemplate(
        "./vehicles/aircraft/model/nacelles/nacelle[@uID='{nacelle_ID}']/"
        "geometry/wettedArea",
        ["nacelle_ID"],
    )


class CPACSEngineTemplates:
    """XPath templates for engine-related CPACS elements."""

    ENGINE = CPACSPathTemplate(
        "./vehicles/aircraft/model/engines/engine[@uID='{engine_ID}']",
        ["engine_ID"],
    )

    ENGINE_PROPULSOR = CPACSPathTemplate(
        "./vehicles/aircraft/model/engines/engine[@uID='{engine_ID}']/"
        "propulsor",
        ["engine_ID"],
    )

    ENGINE_MASS = CPACSPathTemplate(
        "./vehicles/aircraft/model/engines/engine[@uID='{engine_ID}']/"
        "properties/mass",
        ["engine_ID"],
    )

    ENGINE_TYPE = CPACSPathTemplate(
        "./vehicles/aircraft/model/engines/engine[@uID='{engine_ID}']/"
        "type",
        ["engine_ID"],
    )

    ENGINE_NETWORK = CPACSPathTemplate(
        "./vehicles/aircraft/model/engines/engine[@uID='{engine_ID}']/"
        "reference/network",
        ["engine_ID"],
    )


class CPACSMaterialTemplates:
    """XPath templates for material-related CPACS elements."""

    MATERIALS = CPACSPathTemplate(
        "./vehicles/aircraft/model/materials",
        [],
    )

    MATERIAL = CPACSPathTemplate(
        "./vehicles/aircraft/model/materials/material[@uID='{material_ID}']",
        ["material_ID"],
    )

    MATERIAL_DENSITY = CPACSPathTemplate(
        "./vehicles/aircraft/model/materials/material[@uID='{material_ID}']/"
        "mechanical/density",
        ["material_ID"],
    )

    MATERIAL_YOUNG = CPACSPathTemplate(
        "./vehicles/aircraft/model/materials/material[@uID='{material_ID}']/"
        "mechanical/youngModulus",
        ["material_ID"],
    )


class CPACSAircraftTemplates:
    """XPath templates for aircraft-level CPACS elements."""

    AIRCRAFT = CPACSPathTemplate(
        "./vehicles/aircraft",
        [],
    )

    AIRCRAFT_MODEL = CPACSPathTemplate(
        "./vehicles/aircraft/model[@uID='{aircraft_ID}']/",
        ["aircraft_ID"],
    )

    AIRCRAFT_MASS = CPACSPathTemplate(
        "./vehicles/aircraft/model[@uID='{aircraft_ID}']/"
        "analyses/massBreakdown/massSegment/mass",
        ["aircraft_ID"],
    )

    AIRCRAFT_REF_AREA = CPACSPathTemplate(
        "./vehicles/aircraft/model/analyses/aeroMap/aeroPerformanceMap/"
        "reference/area",
        [],
    )

    AIRCRAFT_REF_LENGTH = CPACSPathTemplate(
        "./vehicles/aircraft/model/analyses/aeroMap/aeroPerformanceMap/"
        "reference/length",
        [],
    )


class CPACSMappingRegistry:
    """Registry of all CPACS v3.5 path templates.

    Provides a unified interface for accessing all templates and
    generating XPath queries for aircraft components.
    """

    def __init__(self) -> None:
        self.wing = CPACSWingTemplates()
        self.fuselage = CPACSFuselageTemplates()
        self.nacelle = CPACSNacelleTemplates()
        self.engine = CPACSEngineTemplates()
        self.material = CPACSMaterialTemplates()
        self.aircraft = CPACSAircraftTemplates()

    def get_template(self, component: str, attribute: str) -> CPACSPathTemplate | None:
        """Get a specific template by component and attribute name.

        Args:
            component: Component type (wing, fuselage, nacelle, engine, etc.)
            attribute: Attribute name within the component

        Returns:
            The path template if found, None otherwise.
        """
        component_map = {
            "wing": self.wing,
            "fuselage": self.fuselage,
            "nacelle": self.nacelle,
            "engine": self.engine,
            "material": self.material,
            "aircraft": self.aircraft,
        }

        templates = component_map.get(component.lower())
        if templates is None:
            return None

        return getattr(templates, attribute, None)

    def get_all_templates(self) -> dict[str, dict[str, CPACSPathTemplate]]:
        """Get all templates organized by component type.

        Returns:
            Dictionary mapping component names to their templates.
        """
        return {
            "wing": {k: v for k, v in vars(self.wing).items() if isinstance(v, CPACSPathTemplate)},
            "fuselage": {k: v for k, v in vars(self.fuselage).items() if isinstance(v, CPACSPathTemplate)},
            "nacelle": {k: v for k, v in vars(self.nacelle).items() if isinstance(v, CPACSPathTemplate)},
            "engine": {k: v for k, v in vars(self.engine).items() if isinstance(v, CPACSPathTemplate)},
            "material": {k: v for k, v in vars(self.material).items() if isinstance(v, CPACSPathTemplate)},
            "aircraft": {k: v for k, v in vars(self.aircraft).items() if isinstance(v, CPACSPathTemplate)},
        }


CPACS_TEMPLATES = CPACSMappingRegistry()
