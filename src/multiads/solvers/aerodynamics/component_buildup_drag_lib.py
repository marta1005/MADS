from dataclasses import dataclass

import numpy as np
from assembly import Environment, Section
from assembly import Fuselage as Base_Fuselage
from assembly import Nacelle as Base_Nacelle
from assembly import Wing as Base_Wing
from typing_extensions import Self
from utilities.aerodynamic_utilities import calculate_reynolds_number
from utilities.algebra import cosd

# functions


def calculate_cutoff_reynolds_number(
    mach: float, characteristic_length: float, surface_material: str
) -> float:
    if surface_material == "camouflage_paint_on_aluminum":
        roughness = 10.15e-6
    elif surface_material == "smooth_paint":
        roughness = 6.34e-6
    elif surface_material == "production_sheet_metal":
        roughness = 4.05e-6
    elif surface_material == "polished_sheet_metal":
        roughness = 1.52e-6
    elif surface_material == "smooth_molded_composite":
        roughness = 0.52e-6
    else:
        raise ValueError(
            "Each provided component must have be an 'options' field for this solver, with a valid input for 'surface_material'.",
        )

    if mach < 0.8:
        # subsonic flow regime
        re_cutoff = 38.21 * (characteristic_length / roughness) ** 1.053
    else:
        # supersonic flow regime
        re_cutoff = 44.62 * (characteristic_length / roughness) ** 1.053 * mach**1.16

    return re_cutoff


def calculate_friction_coefficient(
    reynolds_number: float, reynolds_cutoff: float, mach: float
) -> list[float, float]:
    if reynolds_number < reynolds_cutoff:
        c_f_turb = 0.455 / (
            np.log10(reynolds_number) ** 2.58 * (1 + 0.144 * mach**2) ** 0.65
        )
        c_f_laminar = 1.328 / np.sqrt(reynolds_number)
    else:
        c_f_turb = 0.455 / (
            np.log10(reynolds_cutoff) ** 2.58 * (1 + 0.144 * mach**2) ** 0.65
        )
        c_f_laminar = 1.328 / np.sqrt(reynolds_cutoff)

    return [c_f_turb, c_f_laminar]


@dataclass
class Options:
    wing_name_serving_for_reference_area: str = None  # the name of the wing that is to be used as a reference area for the aerodynamic coefficients


@dataclass
class Wing:
    name: str  # name of the wing
    mac: float  # mean aerodynamic chord [m]
    area: float  # wing area [m^2] # TODO @Tim: how does this work if area itself is not a variable, but depends on variables (e.g. span), -->test if automatic update works correctly! Maybe all attributes of the wing need to be included in the "required variables" list?
    wetted_area: float  # wetted area of the wing [m**2]
    sections: list[Section]  # information about the wing sections
    portion_laminar_flow: float = (
        None  # fraction of laminar flow to turbulent flow on wing [0, 1]
    )
    rel_chord_position_max_thickness: float = (
        None  # relative chord position of the maximum thickness
    )
    sweep_at_max_thickness_line: float = (
        None  # sweep angle along the maximum thickness line [deg]
    )
    overall_thick_to_chord_ratio: float = (
        None  # thickness to chord ratio of the wing [-]
    )
    surface_material: str = None  # surface of the wing (is set in the options field)

    @classmethod
    def from_component(cls, comp: Base_Wing) -> Self:
        wing = Wing(
            name=comp.name,
            mac=comp.mac,
            area=comp.area,
            wetted_area=comp.wetted_area,
            sections=comp.sections,
            portion_laminar_flow=comp.portion_laminar_flow,
            rel_chord_position_max_thickness=comp.overall_rel_chord_position_max_thick,
            sweep_at_max_thickness_line=comp.mean_sweep_at_max_thick_lines,
            overall_thick_to_chord_ratio=comp.overall_thick_to_chord_ratio,
        )
        if "options" in vars(comp):
            for k, v in comp.options.get("component_buildup_drag", {}).items():
                if k in vars(wing):
                    setattr(wing, k, v)
        return wing


@dataclass
class Fuselage:
    wetted_area: float = None  # fuselage wetted area in [m**2]
    maximum_width: float = None  # maximum width of the fuselage in [m]
    maximum_height: float = None  # maximum height of the fuselage in [m]
    length: float = None  # length of the fuselage in [m]
    portion_laminar_flow: float = (
        None  # fraction of laminar flow to turbulent flow on the fuselage [0, 1]
    )
    surface_material: str = (
        None  # surface of the fuselage (is set in the options field)
    )

    @classmethod
    def from_component(cls, comp: Base_Fuselage) -> Self:
        fuselage = Fuselage(
            wetted_area=comp.wetted_area,
            maximum_width=comp.maximum_width,
            maximum_height=comp.maximum_height,
            length=comp.length,
            portion_laminar_flow=comp.portion_laminar_flow,
        )
        if "options" in vars(comp):
            for k, v in comp.options.get("component_buildup_drag", {}).items():
                if k in vars(fuselage):
                    setattr(fuselage, k, v)
        return fuselage


@dataclass
class Nacelle:
    wetted_area: float = None  # nacelle wetted area in [m**2]
    maximum_width: float = None  # maximum width of the nacelle in [m]
    maximum_height: float = None  # maximum height of the nacelle in [m]
    length: float = None  # length of the nacelle in [m]
    portion_laminar_flow: float = (
        None  # fraction of laminar flow to turbulent flow on the nacelle [0, 1]
    )
    surface_material: str = None  # surface of the nacelle (is set in the options field)

    @classmethod
    def from_component(cls, comp: Base_Nacelle) -> Self:
        nacelle = Nacelle(
            wetted_area=comp.wetted_area,
            maximum_width=comp.maximum_width,
            maximum_height=comp.maximum_height,
            length=comp.length,
            portion_laminar_flow=comp.portion_laminar_flow,
        )
        if "options" in vars(comp):
            for k, v in comp.options.get("component_buildup_drag", {}).items():
                if k in vars(nacelle):
                    setattr(nacelle, k, v)
        return nacelle


class Driver:
    def __init__(
        self,
        environment: Environment,
        wings: list[Base_Wing],
        fuselages: list[Base_Fuselage],
        nacelles: list[Base_Nacelle],
        options: Options,
    ) -> None:
        self.options = options
        self.density = environment.density
        self.velocity = environment.speed
        self.dynamic_viscosity = environment.dyn_viscosity
        self.mach = environment.mach
        self.reference_area = None

        if wings is not None:
            self.mapind_wing = {
                w.name: i for i, w in enumerate(wings)
            }  # TODO @Tim: only if wings have been provided. also at other locations...
            self.wings = [Wing.from_component(w) for w in wings]
            self.wing_cd0 = [None for _ in wings]

        if fuselages is not None:
            self.mapind_fus = {f.name: i for i, f in enumerate(fuselages)}
            self.fuselages = [
                Fuselage.from_component(f) for f in fuselages
            ]  # TODO @Tim: only if fuselages have been provided. also at other locations...
            self.fus_cD0 = [None for _ in fuselages]

        if nacelles is not None:
            self.mapind_nac = {n.name: i for i, n in enumerate(nacelles)}
            self.nacelles = [
                Nacelle.from_component(f) for f in nacelles
            ]  # TODO @Tim: only if nacelles have been provided. also at other locations...
            self.nacelle_cD0 = [None for _ in nacelles]

    def run(self) -> None:
        if hasattr(self, "wings"):
            # identify wing reference area
            for wing in self.wings:
                if wing.name == self.options.wing_name_serving_for_reference_area:
                    self.reference_area = wing.area

            for i, wing in enumerate(self.wings):
                wing_cd0 = self._run_wing(
                    wing,
                    self.reference_area,
                    self.density,
                    self.velocity,
                    self.dynamic_viscosity,
                    self.mach,
                )
                self.wing_cd0[i] = wing_cd0

        else:
            raise ValueError("At least one wing must be provided!")

        if hasattr(self, "fuselages"):
            for i, fuselage in enumerate(self.fuselages):
                fus_cD0 = self._run_fuselage(
                    fuselage,
                    self.reference_area,
                    self.density,
                    self.velocity,
                    self.dynamic_viscosity,
                    self.mach,
                )
                self.fus_cD0[i] = fus_cD0

        if hasattr(self, "nacelles"):
            for i, nacelles in enumerate(self.nacelles):
                nacelle_cD0 = self._run_nacelle(
                    nacelles,
                    self.reference_area,
                    self.density,
                    self.velocity,
                    self.dynamic_viscosity,
                    self.mach,
                )
                self.nacelle_cD0[i] = nacelle_cD0

    @staticmethod
    # run the model
    def _run_wing(
        wing: Wing,
        reference_area: float,
        density: float,
        velocity: float,
        dynamic_viscosity: float,
        mach: float,
    ) -> float:
        """ """

        # determine cutoff reynolds number
        re_cutoff = calculate_cutoff_reynolds_number(
            mach=mach,
            characteristic_length=wing.mac,
            surface_material=wing.surface_material,
        )

        # determine actual reynolds number
        re = calculate_reynolds_number(
            characteristic_length=wing.mac,
            airspeed=velocity,
            density=density,
            dynamic_viscosity=dynamic_viscosity,
        )

        # calculate skin friction coefficient
        [c_f_turb, c_f_laminar] = calculate_friction_coefficient(
            reynolds_number=re, reynolds_cutoff=re_cutoff, mach=mach
        )

        # weigh portions of laminar and turbulent flow
        c_f_weighted = (
            wing.portion_laminar_flow * c_f_laminar
            + (1 - wing.portion_laminar_flow) * c_f_turb
        )

        # calculate form factor
        ff = (
            1
            + 0.6
            / wing.rel_chord_position_max_thickness
            * wing.overall_thick_to_chord_ratio
            + 100 * wing.overall_thick_to_chord_ratio**4
        ) * (1.34 * mach**0.18 * cosd(wing.sweep_at_max_thickness_line) ** 0.28)

        # skin friction and pressure drag factor
        f_friction_pressure = c_f_weighted * ff

        # drag coefficient
        wing_cd0 = f_friction_pressure * wing.wetted_area / reference_area

        return wing_cd0

    @staticmethod
    def _run_fuselage(
        fuselage: Fuselage,
        reference_area: float,
        density: float,
        velocity: float,
        dynamic_viscosity: float,
        mach: float,
    ) -> float:
        """ """

        # determine cutoff reynolds number
        re_cutoff = calculate_cutoff_reynolds_number(
            mach=mach,
            characteristic_length=fuselage.length,
            surface_material=fuselage.surface_material,
        )

        # determine actual reynolds number
        re = calculate_reynolds_number(
            characteristic_length=fuselage.length,
            airspeed=velocity,
            density=density,
            dynamic_viscosity=dynamic_viscosity,
        )

        # calculate skin friction coefficient
        [c_f_turb, c_f_laminar] = calculate_friction_coefficient(
            reynolds_number=re, reynolds_cutoff=re_cutoff, mach=mach
        )

        # weigh portions of laminar and turbulent flow
        c_f_weighted = (
            fuselage.portion_laminar_flow * c_f_laminar
            + (1 - fuselage.portion_laminar_flow) * c_f_turb
        )

        # fineness ratio (modification to Raymer to account for non-circular shapes (e.g. elliptical)
        if fuselage.maximum_height > fuselage.maximum_width:
            fineness = fuselage.length / fuselage.maximum_height
        else:
            fineness = fuselage.length / fuselage.maximum_width

        # calculate form factor
        ff = 1 + 60 / fineness**3 + fineness / 400

        # skin friction and pressure drag factor
        f_friction_pressure = c_f_weighted * ff

        # drag coefficient
        fuselage_cd0 = f_friction_pressure * fuselage.wetted_area / reference_area

        return fuselage_cd0

    @staticmethod
    def _run_nacelle(
        nacelle: Nacelle,
        reference_area: float,
        density: float,
        velocity: float,
        dynamic_viscosity: float,
        mach: float,
    ) -> float:
        """ """

        # determine cutoff reynolds number
        re_cutoff = calculate_cutoff_reynolds_number(
            mach=mach,
            characteristic_length=nacelle.length,
            surface_material=nacelle.surface_material,
        )

        # determine actual reynolds number
        re = calculate_reynolds_number(
            characteristic_length=nacelle.length,
            airspeed=velocity,
            density=density,
            dynamic_viscosity=dynamic_viscosity,
        )

        # calculate skin friction coefficient
        [c_f_turb, c_f_laminar] = calculate_friction_coefficient(
            reynolds_number=re, reynolds_cutoff=re_cutoff, mach=mach
        )

        # weigh portions of laminar and turbulent flow
        c_f_weighted = (
            nacelle.portion_laminar_flow * c_f_laminar
            + (1 - nacelle.portion_laminar_flow) * c_f_turb
        )

        # fineness ratio (modification to Raymer to account for non-circular shapes (e.g. elliptical)
        if nacelle.maximum_height > nacelle.maximum_width:
            fineness = nacelle.length / nacelle.maximum_height
        else:
            fineness = nacelle.length / nacelle.maximum_width

        # calculate form factor
        ff = 1 + (0.35 / fineness)

        # skin friction and pressure drag factor
        f_friction_pressure = c_f_weighted * ff

        # drag coefficient
        nacelle_cd0 = f_friction_pressure * nacelle.wetted_area / reference_area

        return nacelle_cd0

    def retrieve_cd0(self, component_names: list[str]) -> list[float]:
        component_cd0 = []
        for component_number, component_name in enumerate(component_names):
            if hasattr(self, "wings"):
                if component_name in self.mapind_wing.keys():
                    wing_id = self.mapind_wing[component_name]
                    component_cd0.append(self.wing_cd0[wing_id])
            if hasattr(self, "fuselages"):
                if component_name in self.mapind_fus.keys():
                    fuselage_id = self.mapind_fus[component_name]
                    component_cd0.append(self.fus_cD0[fuselage_id])
            if hasattr(self, "nacelles"):
                if component_name in self.mapind_nac.keys():
                    nacelle_id = self.mapind_nac[component_name]
                    component_cd0.append(self.nacelle_cd0[nacelle_id])

        return component_cd0
