from __future__ import annotations

from copy import deepcopy
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy.interpolate import interp1d
from typing_extensions import Self

from multiads import assembly
from multiads.assembly import MADSComponent
from multiads.solvers import SolverOptions

def _aero_lf_options(comp: MADSComponent) -> dict[str, Any]:
    options = {}
    if hasattr(comp, "options") and isinstance(comp.options, dict):
        options = comp.options.get("aero_lf", {})
    return deepcopy(options)


def _buildup_drag_options(comp: MADSComponent) -> dict[str, Any]:
    options = {}
    if hasattr(comp, "options") and isinstance(comp.options, dict):
        options = comp.options.get("component_buildup_drag", {})
    return deepcopy(options)


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
        re_cutoff = 38.21 * (characteristic_length / roughness) ** 1.053
    else:
        re_cutoff = 44.62 * (characteristic_length / roughness) ** 1.053 * mach**1.16

    return re_cutoff


def calculate_friction_coefficient(
    reynolds_number: float, reynolds_cutoff: float, mach: float
) -> tuple[float, float]:
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

    return c_f_turb, c_f_laminar


def cosd(deg: float) -> float:
    return np.cos(np.radians(deg))

class Options(SolverOptions):
    def __init__(
        self,
        *,
        wing_name_serving_for_reference_area: str | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.wing_name_serving_for_reference_area = wing_name_serving_for_reference_area

class Section:
    def __init__(
        self,
        chord: float,
        twist: float,
        camber_distribution: NDArray[np.float64],
        camber_to_chord_ratio: float,
        thickness_to_chord_ratio: float,
        rel_chord_position_max_camber: float,
        alpha0_rad: float = 0.0,
    ) -> None:
        self.chord = chord
        self.twist = twist
        self.camber_distribution = camber_distribution
        self.camber_to_chord_ratio = camber_to_chord_ratio
        self.thickness_to_chord_ratio = thickness_to_chord_ratio
        self.rel_chord_position_max_camber = rel_chord_position_max_camber
        self.alpha0_rad = alpha0_rad

    @classmethod
    def from_component(cls, comp: assembly.Section) -> Self:
        return cls(
            chord=comp.chord,
            twist=comp.twist,
            camber_distribution=comp.camber_distribution,
            camber_to_chord_ratio=comp.camber_to_chord_ratio,
            thickness_to_chord_ratio=comp.thickness_to_chord_ratio,
            rel_chord_position_max_camber=comp.rel_chord_position_max_camber,
            alpha0_rad=0.0,
        )


class Span:
    TYPES = ["uniform"]

    def __init__(
        self,
        length: float,
        n_elem: int,
        elem_type: str = "uniform",
    ) -> None:
        self.length = length
        self.n_elem = n_elem
        self.elem_type = elem_type

        if self.elem_type.strip() not in Span.TYPES:
            raise ValueError(
                f"Span-wise distribution of elements '{self.elem_type.strip()}' unkown",
            )

    @classmethod
    def from_component(cls, comp: assembly.Span) -> Self:
        opts = _aero_lf_options(comp)
        if density := opts.pop("panelDensity", None):
            opts["n_elem"] = int(max(1, np.ceil(comp.length * density)))

        return cls(
            length=comp.length,
            n_elem=opts.get("n_elem", 1),
            elem_type=opts.get("elem_type", "uniform"),
        )


class WingOptions(assembly.ComponentOptions):
    def __init__(
        self,
        alpha: float = 2.0,
        mac: float = 0.4,
        ar: float = 15.0,
        area: float = 20.0,
        phi_50: float = 1.0,
        cd0: float = 0.0,
        oswald_factor: float | None = None,
        k2: float = 1.0,
        k_polhamus: float = 1.0,
        sections: list[Section] | None = None,
        spans: list[Span] | None = None,
        symmetry: bool = False,
    ) -> None:
        self.alpha = alpha
        self.mac = mac
        self.ar = ar
        self.area = area
        self.phi_50 = phi_50
        self.cd0 = cd0
        self.oswald_factor = oswald_factor
        self.k2 = k2
        self.k_polhamus = k_polhamus
        self.sections = sections if sections is not None else []
        self.spans = spans if spans is not None else []
        self.symmetry = symmetry

class Wing:
    def __init__(
        self,
        alpha: float = 2.0,
        mac: float = 1.0,
        ar: float = 15.0,
        area: float = 20.0,
        phi_50: float = 1.0,
        cd0: float = 0.0,
        oswald_factor: float | None = None,
        k2: float = 1.0,
        k_polhamus: float = 1.0,
        sections: list[Section] | None = None,
        spans: list[Span] | None = None,
        symmetry: bool = False,
    ) -> None:
        self.alpha = alpha
        self.mac = mac
        self.ar = ar
        self.area = area
        self.phi_50 = phi_50
        self.cd0 = cd0
        self.oswald_factor = oswald_factor
        self.k2 = k2
        self.k_polhamus = k_polhamus
        self.sections = sections if sections is not None else []
        self.spans = spans if spans is not None else []
        self.symmetry = symmetry

    @classmethod
    def from_component(cls, comp: assembly.Wing) -> Self:
        comp_opts = _aero_lf_options(comp)
        sections = [Section.from_component(s) for s in comp.sections]
        spans = [Span.from_component(s) for s in comp.spans]

        try:
            area = comp.area
        except:
            if opts := next((o for o in comp.options if type(o) is WingOptions), None):
                area = opts.area
        
        try:
            mac = comp.mac
        except:
            if opts := next((o for o in comp.options if type(o) is WingOptions), None):
                mac = opts.mac
        
        try:
            ar = comp.aspect_ratio
        except:
            if opts := next((o for o in comp.options if type(o) is WingOptions), None):
                ar = opts.ar

        try:
            phi_50 = (
                comp.sweep_at_chord_station(0.5)
                if hasattr(comp, "sweep_at_chord_station")
                else 0.0
            )
        except (IndexError, AttributeError):
            if opts := next((o for o in comp.options if type(o) is WingOptions), None):
                phi_50 = opts.phi_50

        return cls(
            alpha=comp.alpha,
            mac=mac,
            ar=ar,
            area=area,
            phi_50=phi_50,
            sections=sections,
            spans=spans,
            symmetry=comp.symmetry,
        )


class Fuselage:
    def __init__(
        self,
        name: str,
        wetted_area: float,
        maximum_width: float,
        maximum_height: float,
        length: float,
        portion_laminar_flow: float = 0.0,
        surface_material: str = "smooth_paint",
    ) -> None:
        self.name = name
        self.wetted_area = wetted_area
        self.maximum_width = maximum_width
        self.maximum_height = maximum_height
        self.length = length
        self.portion_laminar_flow = portion_laminar_flow
        self.surface_material = surface_material

    @classmethod
    def from_component(cls, comp: assembly.Fuselage) -> Self:
        opts = _buildup_drag_options(comp)
        fuselage = cls(
            name=comp.name,
            wetted_area=comp.wetted_area,
            maximum_width=comp.maximum_width,
            maximum_height=comp.maximum_height,
            length=comp.length,
            portion_laminar_flow=getattr(comp, "portion_laminar_flow", 0.0),
        )
        for k, v in opts.items():
            if k in ["surface_material"]:
                setattr(fuselage, k, v)
        return fuselage


class Nacelle:
    def __init__(
        self,
        name: str,
        wetted_area: float,
        maximum_width: float,
        maximum_height: float,
        length: float,
        portion_laminar_flow: float = 0.0,
        surface_material: str = "smooth_paint",
    ) -> None:
        self.name = name
        self.wetted_area = wetted_area
        self.maximum_width = maximum_width
        self.maximum_height = maximum_height
        self.length = length
        self.portion_laminar_flow = portion_laminar_flow
        self.surface_material = surface_material

    @classmethod
    def from_component(cls, comp: assembly.Nacelle) -> Self:
        opts = _buildup_drag_options(comp)
        nacelle = cls(
            name=comp.name,
            wetted_area=comp.wetted_area,
            maximum_width=comp.maximum_width,
            maximum_height=comp.maximum_height,
            length=comp.length,
            portion_laminar_flow=getattr(comp, "portion_laminar_flow", 0.0),
        )
        for k, v in opts.items():
            if k in ["surface_material"]:
                setattr(nacelle, k, v)
        return nacelle


class Driver:
    def __init__(
        self,
        environment: assembly.Environment,
        wings: list[assembly.Wing],
        fuselages: list[assembly.Fuselage] | None = None,
        nacelles: list[assembly.Nacelle] | None = None,
        options: Options | None = None,
    ) -> None:
        self.environment = environment
        self.options = options or Options()
        self.density = environment.density
        self.soundSpeed = environment.sound_speed
        self.viscosity = environment.dyn_viscosity
        self.velocity = environment.speed
        self.mach = environment.mach

        self.mapind = {w.name: i for i, w in enumerate(wings)}
        self.Cl = [0.0 for _ in wings]
        self.Cd = [0.0 for _ in wings]

        self.wings = [Wing.from_component(w) for w in wings]
        for w in self.wings:
            w.alpha += environment.alpha

        self.fuselages: list[Fuselage] = []
        if fuselages:
            self.fuselages = [Fuselage.from_component(f) for f in fuselages]

        self.nacelles: list[Nacelle] = []
        if nacelles:
            self.nacelles = [Nacelle.from_component(n) for n in nacelles]

        self.wing_cd0: list[float] = [0.0 for _ in wings]
        self.fuselage_cd0: list[float] = (
            [0.0 for _ in self.fuselages] if self.fuselages else []
        )
        self.nacelle_cd0: list[float] = (
            [0.0 for _ in self.nacelles] if self.nacelles else []
        )

        self.reference_area: float | None = None

    def run(self) -> None:
        if not self.wings:
            raise ValueError("At least one wing must be provided!")

        if self.options.wing_name_serving_for_reference_area:
            for wing in self.wings:
                if (
                    getattr(wing, "name", None)
                    == self.options.wing_name_serving_for_reference_area
                ):
                    self.reference_area = getattr(wing, "area", 0.0)
                    break
        else:
            self.reference_area = (
                getattr(self.wings[0], "area", 0.0) if self.wings else 0.0
            )

        for i, wing in enumerate(self.wings):
            cl, cd = self._run_wing(
                wing,
                self.density,
                self.velocity,
                self.viscosity,
                self.soundSpeed,
            )
            self.Cl[i] = cl
            self.Cd[i] = cd

            wing_cd0 = self._compute_wing_cd0(wing)
            self.wing_cd0[i] = wing_cd0

        for i, fuselage in enumerate(self.fuselages):
            self.fuselage_cd0[i] = self._compute_fuselage_cd0(fuselage)

        for i, nacelle in enumerate(self.nacelles):
            self.nacelle_cd0[i] = self._compute_nacelle_cd0(nacelle)

    def _run_wing(
        self,
        wing: Wing,
        density: float,
        velocity: float,
        viscosity: float,
        soundSpeed: float,
    ) -> tuple[float, float]:
        mean_chord = wing.mac if wing.mac else 1.0
        aspect_ratio = wing.ar if wing.ar else 1.0
        cd0 = getattr(wing, "cd0", 0.0)
        k1 = getattr(wing, "k1", 1.0)

        if len(wing.sections) < 2 or not wing.spans:
            alpha = wing.alpha
            alpha_0 = 0.0
            phi_50 = wing.phi_50

            cL_alpha = (2.0 * np.pi * aspect_ratio) / (
                2.0 + np.sqrt(((aspect_ratio**2) * (1.0 + (np.tan(phi_50) ** 2)) + 4.0))
            )

            Cl = cL_alpha * (alpha - alpha_0)
            Cd = cd0 + k1 * Cl**2

            return Cl, Cd

        sum_all_areas = 0.0
        sum_all_alpha0_rad_span_weighted = 0.0
        sum_all_twist_rad_span_weighted = 0.0

        for i, span in enumerate(wing.spans):
            chord_section_inner = wing.sections[i].chord
            chord_section_outer = wing.sections[i + 1].chord
            twist_section_inner = wing.sections[i].twist
            twist_section_outer = wing.sections[i + 1].twist

            area_span = 0.5 * span.length * (chord_section_inner + chord_section_outer)

            x_camber_inner = wing.sections[i].camber_distribution[:, 0]
            y_camber_inner = wing.sections[i].camber_distribution[:, 1]
            x_camber_outer = wing.sections[i + 1].camber_distribution[:, 0]
            y_camber_outer = wing.sections[i + 1].camber_distribution[:, 1]

            n = 1000
            theta_inner = np.linspace(0, np.pi, n)
            x_theta_inner = 0.5 * (1 - np.cos(theta_inner))

            interp_camber = interp1d(
                x_camber_inner, y_camber_inner, kind="cubic", fill_value="extrapolate"
            )
            z_inner = interp_camber(x_theta_inner)

            dzdx_inner = np.gradient(z_inner, x_theta_inner)

            integrand_inner = dzdx_inner * np.cos(theta_inner)
            alpha0_rad_inner = -1 / np.pi * np.trapezoid(integrand_inner, theta_inner)

            theta_outer = np.linspace(0, np.pi, n)
            x_theta_outer = 0.5 * (1 - np.cos(theta_outer))

            interp_camber = interp1d(
                x_camber_outer, y_camber_outer, kind="cubic", fill_value="extrapolate"
            )
            z_outer = interp_camber(x_theta_outer)

            dzdx_outer = np.gradient(z_outer, x_theta_outer)

            integrand_outer = dzdx_outer * np.cos(theta_outer)
            alpha0_rad_outer = -1 / np.pi * np.trapezoid(integrand_outer, theta_outer)

            alpha0_rad_span = (alpha0_rad_outer + alpha0_rad_inner) / 2

            twist_rad_span = (
                np.radians(twist_section_outer) + np.radians(twist_section_inner)
            ) / 2

            sum_all_alpha0_rad_span_weighted += alpha0_rad_span * area_span
            sum_all_twist_rad_span_weighted += twist_rad_span * area_span
            sum_all_areas += area_span

            wing.sections[i].alpha0_rad = alpha0_rad_inner
            wing.sections[i + 1].alpha0_rad = alpha0_rad_outer

        alpha0_rad_weighted = sum_all_alpha0_rad_span_weighted / sum_all_areas
        twist_rad_weighted = sum_all_twist_rad_span_weighted / sum_all_areas

        reynolds_number = density * velocity * mean_chord / viscosity

        beta = np.sqrt(1.0 - (velocity / soundSpeed) ** 2)

        alpha = np.radians(wing.alpha)
        phi_50 = np.radians(wing.phi_50)

        cL_alpha = (2.0 * np.pi * aspect_ratio) / (
            2.0
            + np.sqrt(
                ((aspect_ratio**2 * beta**2) / (wing.k_polhamus**2))
                * (1.0 + ((np.tan(phi_50) ** 2) / beta**2))
                + 4.0,
            )
        )

        factor_cd0 = 1.0
        factor_cL_alpha = 1.0
        factor_alpha_zero_lift = 1.0
        factor_mean_twist = 1.0
        factor_k1 = 1.0
        factor_k2 = 1.0

        oswald = wing.oswald_factor if wing.oswald_factor else 0.8
        k1 = 1 / (np.pi * wing.ar * oswald)

        Cl = (
            factor_cL_alpha
            * cL_alpha
            * (
                alpha
                + twist_rad_weighted * factor_mean_twist
                - alpha0_rad_weighted * factor_alpha_zero_lift
            )
        )

        cd_polar = (
            wing.cd0 * factor_cd0 + factor_k1 * k1 * Cl**2 + (factor_k2 - wing.k2) * Cl
        )

        return Cl, cd_polar

    def _compute_wing_cd0(self, wing: Wing) -> float:
        if self.reference_area is None:
            return 0.0

        re = self.density * self.velocity * wing.mac / self.viscosity

        re_cutoff = calculate_cutoff_reynolds_number(
            mach=self.mach,
            characteristic_length=wing.mac,
            surface_material="smooth_paint",
        )

        c_f_turb, c_f_laminar = calculate_friction_coefficient(re, re_cutoff, self.mach)

        portion_laminar = getattr(wing, "portion_laminar_flow", 0.0)
        c_f_weighted = portion_laminar * c_f_laminar + (1 - portion_laminar) * c_f_turb

        rel_chord_max_thick = getattr(wing, "rel_chord_position_max_thickness", 0.35)
        thick_to_chord = getattr(wing, "thickness_to_chord_ratio", 0.12)
        sweep_at_max_thick = getattr(wing, "sweep_at_max_thickness_line", 0.0)

        ff = (
            1 + 0.6 / rel_chord_max_thick * thick_to_chord + 100 * thick_to_chord**4
        ) * (1.34 * self.mach**0.18 * cosd(sweep_at_max_thick) ** 0.28)

        wetted_area = getattr(wing, "wetted_area", wing.area * 2)
        return c_f_weighted * ff * wetted_area / self.reference_area

    def _compute_fuselage_cd0(self, fuselage: Fuselage) -> float:
        if self.reference_area is None:
            return 0.0

        re = self.density * self.velocity * fuselage.length / self.viscosity

        re_cutoff = calculate_cutoff_reynolds_number(
            mach=self.mach,
            characteristic_length=fuselage.length,
            surface_material=fuselage.surface_material,
        )

        c_f_turb, c_f_laminar = calculate_friction_coefficient(re, re_cutoff, self.mach)

        c_f_weighted = (
            fuselage.portion_laminar_flow * c_f_laminar
            + (1 - fuselage.portion_laminar_flow) * c_f_turb
        )

        if fuselage.maximum_height > fuselage.maximum_width:
            fineness = fuselage.length / fuselage.maximum_height
        else:
            fineness = fuselage.length / fuselage.maximum_width

        ff = 1 + 60 / fineness**3 + fineness / 400

        return c_f_weighted * ff * fuselage.wetted_area / self.reference_area

    def _compute_nacelle_cd0(self, nacelle: Nacelle) -> float:
        if self.reference_area is None:
            return 0.0

        re = self.density * self.velocity * nacelle.length / self.viscosity

        re_cutoff = calculate_cutoff_reynolds_number(
            mach=self.mach,
            characteristic_length=nacelle.length,
            surface_material=nacelle.surface_material,
        )

        c_f_turb, c_f_laminar = calculate_friction_coefficient(re, re_cutoff, self.mach)

        c_f_weighted = (
            nacelle.portion_laminar_flow * c_f_laminar
            + (1 - nacelle.portion_laminar_flow) * c_f_turb
        )

        if nacelle.maximum_height > nacelle.maximum_width:
            fineness = nacelle.length / nacelle.maximum_height
        else:
            fineness = nacelle.length / nacelle.maximum_width

        ff = 1 + (0.35 / fineness)

        return c_f_weighted * ff * nacelle.wetted_area / self.reference_area

    @property
    def total_cd0(self) -> float:
        return sum(self.wing_cd0) + sum(self.fuselage_cd0) + sum(self.nacelle_cd0)

    def coefficients(self, wing_names: list[str]) -> tuple[float, float]:
        if not wing_names:
            wing_names = list(self.mapind.keys())

        cl, cd = 0.0, 0.0
        for name in wing_names:
            try:
                i = self.mapind[name]
                cl += self.Cl[i]
                cd += self.Cd[i]
            except KeyError:
                pass

        return cl, cd

    def forces(self, wing_names: list[str]) -> tuple[float, float]:
        if not wing_names:
            wing_names = list(self.mapind.keys())

        q = 0.5 * self.density * self.velocity**2
        l, d = 0.0, 0.0
        for name in wing_names:
            try:
                i = self.mapind[name]
                l += self.wings[i].area * self.Cl[i]
                d += self.wings[i].area * self.Cd[i]
            except KeyError:
                pass

        return q * l, q * d

    def lift(self, wing_names: list[str] | None = None) -> float:
        if wing_names is None:
            wing_names = list(self.mapind.keys())

        q = 0.5 * self.density * self.velocity**2
        total_lift = 0.0

        for name in wing_names:
            try:
                i = self.mapind[name]
                cl = self.Cl[i]
                total_lift += q * self.wings[i].area * cl
            except KeyError:
                pass

        return total_lift

    def drag(self, wing_names: list[str] | None = None) -> float:
        if wing_names is None:
            wing_names = list(self.mapind.keys())

        q = 0.5 * self.density * self.velocity**2
        total_drag = 0.0
        cd0 = self.total_cd0

        for name in wing_names:
            try:
                i = self.mapind[name]
                cl = self.Cl[i]
                cd_coefficient = self.Cd[i]
                ar = self.wings[i].ar or 1.0
                oswald = self.wings[i].oswald_factor or 0.8
                k1 = 1 / (np.pi * ar * oswald)
                k2 = getattr(self.wings[i], "k2", 1.0)
                cd_total = cd0 + k1 * cl**2 + (k2 - 1.0) * cl
                total_drag += q * self.wings[i].area * cd_total
            except KeyError:
                pass

        return total_drag

    def efficiency(self, wing_names: list[str] | None = None) -> float:
        l = self.lift(wing_names)
        d = self.drag(wing_names)
        if d == 0:
            return 0.0
        return l / d

    def moments(
        self, wing_names: list[str] | None = None
    ) -> tuple[float, float, float]:
        if wing_names is None:
            wing_names = list(self.mapind.keys())

        thick_corr_fac = 0.05

        q = 0.5 * self.density * self.velocity**2
        m_y = 0.0
        for name in wing_names:
            try:
                i = self.mapind[name]

                sum_all_areas = 0.0
                sum_all_cm_y_span_weighted = 0.0

                for k, span in enumerate(self.wings[i].spans):
                    chord_section_inner = self.wings[i].sections[k].chord
                    chord_section_outer = self.wings[i].sections[k + 1].chord
                    area_span = (
                        0.5 * span.length * (chord_section_inner + chord_section_outer)
                    )

                    camber_to_chord_ratio_in = (
                        self.wings[i].sections[k].camber_to_chord_ratio
                    )
                    thickness_to_chord_ratio_in = (
                        self.wings[i].sections[k].thickness_to_chord_ratio
                    )
                    rel_chord_position_max_camber_in = (
                        self.wings[i].sections[k].rel_chord_position_max_camber
                    )

                    camber_to_chord_ratio_out = (
                        self.wings[i].sections[k + 1].camber_to_chord_ratio
                    )
                    thickness_to_chord_ratio_out = (
                        self.wings[i].sections[k + 1].thickness_to_chord_ratio
                    )
                    rel_chord_position_max_camber_out = (
                        self.wings[i].sections[k + 1].rel_chord_position_max_camber
                    )

                    cm_y_inner = (
                        -np.pi
                        / 2
                        * camber_to_chord_ratio_in
                        * (0.5 - rel_chord_position_max_camber_in)
                        - thick_corr_fac * thickness_to_chord_ratio_in
                    )

                    cm_y_outer = (
                        -np.pi
                        / 2
                        * camber_to_chord_ratio_out
                        * (0.5 - rel_chord_position_max_camber_out)
                        - thick_corr_fac * thickness_to_chord_ratio_out
                    )

                    cm_y_span = (cm_y_outer + cm_y_inner) / 2

                    sum_all_cm_y_span_weighted += cm_y_span * area_span
                    sum_all_areas += area_span

                cm_y_weighted = sum_all_cm_y_span_weighted / sum_all_areas
                m_y += cm_y_weighted * q * self.wings[i].area * self.wings[i].mac

            except KeyError:
                pass

        return 0.0, m_y, 0.0

    def moment_y(self, wing_names: list[str] | None = None) -> float:
        _, my, _ = self.moments(wing_names)
        return my

    def spanloads(self, wing_names: list[str] | None = None) -> list[list[float]]:
        if wing_names is None:
            wing_names = list(self.mapind.keys())

        y_cen = np.array([])
        y_span = np.array([])
        fx = np.array([])
        fy = np.array([])
        fz = np.array([])
        mo = np.array([])

        for name in wing_names:
            try:
                i = self.mapind[name]

                wing_accumlated_half_span = 0.0

                for k, span in enumerate(self.wings[i].spans):
                    element_length = span.length / span.n_elem

                    y_cen = np.append(
                        y_cen,
                        wing_accumlated_half_span
                        + (np.arange(span.n_elem) + 0.5) * element_length,
                    )

                    y_span = np.append(y_span, np.full(span.n_elem, element_length))

                    chord_section_inner = self.wings[i].sections[k].chord
                    chord_section_outer = self.wings[i].sections[k + 1].chord

                    twist_section_inner = self.wings[i].sections[k].twist
                    twist_section_outer = self.wings[i].sections[k + 1].twist

                    alpha0_rad_section_inner = self.wings[i].sections[k].alpha0_rad
                    alpha0_rad_section_outer = self.wings[i].sections[k + 1].alpha0_rad

                    wing_accumlated_half_span += span.length

                lift, drag = self.forces([name])

                _, m_y, _ = self.moments([name])

                spanwise_lift_half_wing = (
                    4
                    * lift
                    / (np.pi * 2 * wing_accumlated_half_span)
                    * np.sqrt(1 - (2 * y_cen / (2 * wing_accumlated_half_span)) ** 2)
                )

                if self.wings[i].symmetry:
                    negative_part_y_cen = -np.flip(y_cen)
                    negative_part_y_span = np.flip(y_span)

                    spanwise_lift_other_half_wing = np.flip(spanwise_lift_half_wing)
                    spanwise_drag_other_half_wing = np.flip(
                        -drag * y_span / wing_accumlated_half_span
                    )

                    y_cen = np.concatenate((negative_part_y_cen, y_cen))
                    y_span = np.concatenate((negative_part_y_span, y_span))
                    fz = np.concatenate(
                        (spanwise_lift_other_half_wing, spanwise_lift_half_wing)
                    )
                    fx = np.concatenate(
                        (
                            spanwise_drag_other_half_wing,
                            -drag * y_span / wing_accumlated_half_span,
                        )
                    )
                    fy = np.zeros_like(fz)
                    mo = np.zeros_like(fz)
                else:
                    raise ValueError(
                        "Only symmetric wings are currently supported by this solver",
                    )
            except KeyError:
                pass

        return [
            y_cen.tolist(),
            y_span.tolist(),
            fx.tolist(),
            fy.tolist(),
            fz.tolist(),
            mo.tolist(),
        ]
