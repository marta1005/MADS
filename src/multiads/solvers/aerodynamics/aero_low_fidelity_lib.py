from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any

import assembly
import numpy as np
from assembly import MADSComponent
from numpy.typing import NDArray
from scipy.interpolate import interp1d
from typing_extensions import Self


def _aero_lf_options(comp: MADSComponent) -> dict[str, Any]:
    options = {}
    if hasattr(comp, "options"):
        options = comp.options.get("aero_lf", {})
    return deepcopy(options)


@dataclass
class Section:
    chord: float
    twist: float
    camber_distribution: NDArray[np.float64]  # camber distribution [x,y]
    camber_to_chord_ratio: float
    thickness_to_chord_ratio: float
    rel_chord_position_max_camber: float
    alpha0_rad: float = 0.0  # zero lift angle of attack [rad]

    @classmethod
    def from_component(cls, comp: assembly.Section) -> Self:
        return Section(
            chord=comp.chord,
            twist=comp.twist,
            camber_distribution=comp.camber_distribution,
            camber_to_chord_ratio=comp.camber_to_chord_ratio,
            thickness_to_chord_ratio=comp.thickness_to_chord_ratio,
            rel_chord_position_max_camber=comp.rel_chord_position_max_camber,
            alpha0_rad=0.0,  # Initialize alpha0_rad to 0.0
        )


@dataclass
class Span:
    length: float  # length of the span [m]
    n_elem: int
    elem_type: str = "uniform"

    TYPES = [
        "uniform",
        # "cosine", # not considered yet
        # "cosineIB", # not considered yet
        # "cosineOB", # not considered yet
        # "equalarea", # not considered yet
    ]

    def __post_init__(self) -> None:
        if self.elem_type.strip() not in Span.TYPES:
            raise ValueError(
                f"Span-wise distribution of elements '{self.elem_type.strip()}' unkown",
            )

    @classmethod
    def from_component(cls, comp: assembly.Span) -> Self:
        opts = _aero_lf_options(comp)
        if density := opts.pop("panelDensity", None):
            opts["n_elem"] = int(max(1, np.ceil(comp.length * density)))

        return Span(comp.length, **opts)


@dataclass
class Wing:
    alpha: float  # angle of attack [deg]
    mac: float  # mean aerodynamic chord [m]
    ar: float  # aspect ratio [-]
    area: float  # wing area [m^2] # TODO @Tim: how does this work if area itself is not a variable, but depends on variables (e.g. span), -->test if automatic update works correctly! Maybe all attributes of the wing need to be included in the "required variables" list?
    phi_50: float  # sweep at 50% of the chord [deg]
    cd0: float = 0.0  # cd_0 polar parameter [-]
    oswald_factor: float = None
    k2: float = 1.0  # k2 polar parameter [-]
    k_polhamus: float = 1.0  # airfoil/pressure impact [-] # TODO @Tim: ???
    sections: list[Section] = field(default_factory=list)
    spans: list[Span] = field(default_factory=list)
    symmetry: bool = False

    @classmethod
    def from_component(cls, comp: assembly.Wing) -> Self:
        opts = _aero_lf_options(comp)
        sections = [Section.from_component(s) for s in comp.sections]
        spans = [Span.from_component(s) for s in comp.spans]
        
        return Wing(
            alpha=comp.alpha,
            mac=comp.mac,
            ar=comp.aspect_ratio,
            area=comp.area,
            # phi_50=comp.sweep_at_chord_station(0.5),
            # cd0=comp.cd0,
            sections=sections,
            spans=spans,
            symmetry=comp.symmetry,
            **opts,
        )


@dataclass
class aero_lf_options:
    pass


class Driver:
    def __init__(
        self,
        environment: assembly.Environment = field(default_factory=assembly.Environment),
        wings: list[Wing] = field(default_factory=list),
        options: aero_lf_options = field(default_factory=aero_lf_options),
    ) -> None:
        self.environment = environment
        self.options = options
        self.density = environment.density
        self.soundSpeed = environment.sound_speed
        self.viscosity = environment.dyn_viscosity
        self.velocity = environment.speed
        self.mapind = {w.name: i for i, w in enumerate(wings)}
        self.Cl = [0.0 for _ in wings]
        self.Cd = [0.0 for _ in wings]

        self.wings = [Wing.from_component(w) for w in wings]
        for w in self.wings:
            w.alpha += environment.alpha

    def run(self) -> None:
        for i, wing in enumerate(self.wings):
            cl, cd = self._run(
                wing,
                self.options,
                self.density,
                self.velocity,
                self.viscosity,
                self.soundSpeed,
            )
            self.Cl[i] = cl
            self.Cd[i] = cd

    @staticmethod
    def _run(
        wing: Wing,
        options: aero_lf_options,
        density: float,
        velocity: float,
        viscosity: float,
        soundSpeed: float,
    ) -> tuple[float, float]:
        # recover derived geometrical parameters
        mean_chord = wing.mac
        aspect_ratio = wing.ar

        ### estimate zero lift angle of attack based on thin airfoil theory for each Section ###

        # initialise
        sum_all_areas = 0.0
        sum_all_alpha0_rad_span_weighted = 0.0
        sum_all_twist_rad_span_weighted = 0.0

        for i, span in enumerate(wing.spans):
            # calculate area, taper ratio, and reference chord (MAC) of each span
            chord_section_inner = wing.sections[i].chord
            chord_section_outer = wing.sections[i + 1].chord
            twist_section_inner = wing.sections[
                i
            ].twist  # twist of the section in [deg]
            twist_section_outer = wing.sections[
                i + 1
            ].twist  # twist of the section in [deg]

            area_span = 0.5 * span.length * (chord_section_inner + chord_section_outer)

            # Split camber line into x and y coordinates
            x_camber_inner = wing.sections[i].camber_distribution[:, 0]
            y_camber_inner = wing.sections[i].camber_distribution[:, 1]
            x_camber_outer = wing.sections[i + 1].camber_distribution[:, 0]
            y_camber_outer = wing.sections[i + 1].camber_distribution[:, 1]

            # Interpolate camber line to theta-based x grid
            ################# inner section
            n = 1000
            theta_inner = np.linspace(0, np.pi, n)
            x_theta_inner = 0.5 * (1 - np.cos(theta_inner))  # mapping x = (1 - cosθ)/2

            # Interpolate camber to x_theta grid
            interp_camber = interp1d(
                x_camber_inner, y_camber_inner, kind="cubic", fill_value="extrapolate"
            )
            z_inner = interp_camber(x_theta_inner)

            # Compute dz/dx
            dzdx_inner = np.gradient(z_inner, x_theta_inner)

            # Apply thin airfoil theory integral
            integrand_inner = dzdx_inner * np.cos(theta_inner)
            alpha0_rad_inner = -1 / np.pi * np.trapz(integrand_inner, theta_inner)

            ################# outer section
            n = 1000
            theta_outer = np.linspace(0, np.pi, n)
            x_theta_outer = 0.5 * (1 - np.cos(theta_outer))  # mapping x = (1 - cosθ)/2

            # Interpolate camber to x_theta grid
            interp_camber = interp1d(
                x_camber_outer, y_camber_outer, kind="cubic", fill_value="extrapolate"
            )
            z_outer = interp_camber(x_theta_outer)

            # Compute dz/dx
            dzdx_outer = np.gradient(z_outer, x_theta_outer)

            # Apply thin airfoil theory integral
            integrand_outer = dzdx_outer * np.cos(theta_outer)
            alpha0_rad_outer = -1 / np.pi * np.trapz(integrand_outer, theta_outer)

            # calculate mean alpha0 of the span
            alpha0_rad_span = (alpha0_rad_outer + alpha0_rad_inner) / 2

            # calculate mean twist angle of the span
            twist_rad_span = (
                np.radians(twist_section_outer) + np.radians(twist_section_inner)
            ) / 2

            # compute area weighted sum of the alpha0_rad_span
            sum_all_alpha0_rad_span_weighted += alpha0_rad_span * area_span

            # compute area weighted sum of the twist angles
            sum_all_twist_rad_span_weighted += twist_rad_span * area_span

            # compute total area
            sum_all_areas += area_span

            # Update alpha0_rad for the inner and outer sections
            wing.sections[i].alpha0_rad = alpha0_rad_inner
            wing.sections[i + 1].alpha0_rad = alpha0_rad_outer

        # compute area weighted alpha0_rad for the entire wing
        alpha0_rad_weighted = sum_all_alpha0_rad_span_weighted / sum_all_areas

        # compute area weighted twist_rad_ for the entire wing
        twist_rad_weighted = sum_all_twist_rad_span_weighted / sum_all_areas

        # Reynolds number # ! review reference lenght scale
        reynolds_number = density * velocity * mean_chord / viscosity

        # Compressibility factor
        beta = np.sqrt(1.0 - (velocity / soundSpeed) ** 2)

        # degrees to radians
        alpha = np.radians(wing.alpha)
        phi_50 = np.radians(wing.phi_50)

        # compute dCl/dalpha
        cL_alpha = (2.0 * np.pi * aspect_ratio) / (
            2.0
            + np.sqrt(
                ((aspect_ratio**2 * beta**2) / (wing.k_polhamus**2))
                * (1.0 + ((np.tan(phi_50) ** 2) / beta**2))
                + 4.0,
            )
        )

        ###### RE-Number is out of the region for which the correction factor formulas are valid!!!
        # compute correction factors
        # factor_cL_alpha, factor_alpha_zero_lift, factor_k1, factor_k2 = self._correction_factors(reynolds_number, aspect_ratio)
        factor_cd0 = 1.0  # no correction if value is 1
        factor_cL_alpha = 1.0  # no correction if value is 1
        factor_alpha_zero_lift = 1.0  # no correction if value is 1
        factor_mean_twist = 1.0  # no correction if value is 1
        factor_k1 = 1.0  # no correction if value is 1
        factor_k2 = 1.0  # no correction if value is 1
        k1 = 1 / (np.pi * wing.ar * wing.oswald_factor)

        # lift and drag coefficients
        Cl = (
            factor_cL_alpha
            * cL_alpha
            * (
                alpha
                + twist_rad_weighted * factor_mean_twist
                - alpha0_rad_weighted * factor_alpha_zero_lift
            )
        )
        Cd = wing.cd0 * factor_cd0 + factor_k1 * k1 * Cl**2 + (factor_k2 - wing.k2) * Cl

        return Cl, Cd

    def coefficients(self, wing_names: list[str]) -> tuple[float, float]:
        if not wing_names:
            wing_names = self.mapind.keys()

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
            wing_names = self.mapind.keys()

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

    def moments(self, wing_names: list[str]) -> tuple[float, float, float]:
        # thickness correction factor: [0.05 - 0.1]
        thick_corr_fac = 0.05

        if not wing_names:
            wing_names = self.mapind.keys()

        q = 0.5 * self.density * self.velocity**2
        m_y = 0.0
        for name in wing_names:
            try:
                i = self.mapind[name]

                ### estimate pitching moment coefficient for each Section ###

                # initialise
                sum_all_areas = 0.0
                sum_all_cm_y_span_weighted = 0.0

                for k, span in enumerate(self.wings[i].spans):
                    # calculate area, taper ratio, and reference chord (MAC) of each span
                    chord_section_inner = self.wings[i].sections[k].chord
                    chord_section_outer = self.wings[i].sections[k + 1].chord
                    area_span = (
                        0.5 * span.length * (chord_section_inner + chord_section_outer)
                    )

                    # inner airfoil geometric characteristics
                    camber_to_chord_ratio_in = (
                        self.wings[i].sections[k].camber_to_chord_ratio
                    )
                    thickness_to_chord_ratio_in = (
                        self.wings[i].sections[k].thickness_to_chord_ratio
                    )
                    rel_chord_position_max_camber_in = (
                        self.wings[i].sections[k].rel_chord_position_max_camber
                    )

                    # outer airfoil geometric characteristics
                    camber_to_chord_ratio_out = (
                        self.wings[i].sections[k + 1].camber_to_chord_ratio
                    )
                    thickness_to_chord_ratio_out = (
                        self.wings[i].sections[k + 1].thickness_to_chord_ratio
                    )
                    rel_chord_position_max_camber_out = (
                        self.wings[i].sections[k + 1].rel_chord_position_max_camber
                    )

                    # estimate cm_y at 1/4 line of inner section
                    cm_y_inner = (
                        -np.pi
                        / 2
                        * camber_to_chord_ratio_in
                        * (0.5 - rel_chord_position_max_camber_in)
                        - thick_corr_fac * thickness_to_chord_ratio_in
                    )

                    # estimate cm_y at 1/4 line of outer section
                    cm_y_outer = (
                        -np.pi
                        / 2
                        * camber_to_chord_ratio_out
                        * (0.5 - rel_chord_position_max_camber_out)
                        - thick_corr_fac * thickness_to_chord_ratio_out
                    )

                    # calculate mean cm_y_span of the span
                    cm_y_span = (cm_y_outer + cm_y_inner) / 2

                    # compute area weighted sum of the cm_y_span
                    sum_all_cm_y_span_weighted += cm_y_span * area_span

                    # compute total area
                    sum_all_areas += area_span

                # compute area weighted cm_y for the entire wing
                cm_y_weighted = sum_all_cm_y_span_weighted / sum_all_areas

                # averaged pitching moment
                m_y = cm_y_weighted * q * self.wings[i].area * self.wings[i].mac

            except KeyError:
                pass

        return 0.0, m_y, 0.0

    def spanloads(self, wing_names: list[str]) -> list[list[float]]:
        if not wing_names:
            wing_names = self.mapind.keys()

        for name in wing_names:
            try:
                i = self.mapind[name]

                # initialize wing_total_span
                wing_accumlated_half_span = 0.0
                y_cen = np.array([])  # initialize y_cen as NumPy array
                y_span = np.array([])  # initialize y_span as NumPy array
                fz = np.array([])  # initialize fz (--> spanwise lift) as NumPy array
                fx = np.array([])  # initialize fz (--> spanwise drag) as NumPy array
                chord = np.array([])  # initialize local chord as NumPy array
                twist_rad = np.array(
                    []
                )  # initialize local twist angle in [rad] as NumPy array
                alpha0_rad = np.array(
                    []
                )  # initialize local zero lift angle of attack in [rad]as NumPy array

                for k, span in enumerate(self.wings[i].spans):
                    # length of the elements (as in DUST) in this span
                    element_length = span.length / span.n_elem

                    # y coordinates of centers of the elements (as in DUST; projected y position)
                    y_cen = np.append(
                        y_cen,
                        wing_accumlated_half_span
                        + (np.arange(span.n_elem) + 0.5) * element_length,
                    )

                    # create an array containing the element_length for each element in this span (uniform element length)
                    y_span = np.append(y_span, np.full(span.n_elem, element_length))

                    # create an array containing the local chord (linear interpolation)
                    chord_section_inner = self.wings[i].sections[k].chord
                    chord_section_outer = self.wings[i].sections[k + 1].chord
                    chord = np.append(
                        chord,
                        np.linspace(
                            chord_section_inner, chord_section_outer, span.n_elem
                        ),
                    )

                    # create an array containing the local twist angle (linear interpolation)
                    twist_section_inner = self.wings[i].sections[k].twist
                    twist_section_outer = self.wings[i].sections[k + 1].twist
                    twist_rad = np.append(
                        twist_rad,
                        np.linspace(
                            np.radians(twist_section_inner),
                            np.radians(twist_section_outer),
                            span.n_elem,
                        ),
                    )

                    # create an array containing the local zero lift angle (linear interpolation)
                    alpha0_rad_section_inner = self.wings[i].sections[k].alpha0_rad
                    alpha0_rad_section_outer = self.wings[i].sections[k + 1].alpha0_rad
                    alpha0_rad = np.append(
                        alpha0_rad,
                        np.linspace(
                            alpha0_rad_section_inner,
                            alpha0_rad_section_outer,
                            span.n_elem,
                        ),
                    )

                    # sum up currently accumulated wing span
                    wing_accumlated_half_span += span.length

                # get integral lift and drag of this wing
                lift, drag = self.forces([name])

                # get integral moments of this wing
                _, m_y, _ = self.moments([name])

                # compute elliptical lift distribution
                spanwise_lift_half_wing = (
                    4
                    * lift
                    / (np.pi * 2 * wing_accumlated_half_span)
                    * np.sqrt(1 - (2 * y_cen / (2 * wing_accumlated_half_span)) ** 2)
                )

                # add global AoA of the wing to the twist distribution
                aoa_wing_total = twist_rad + np.radians(self.wings[i].alpha)

                # generate an array with effective angles of attack at each spanwise location
                # (subtract zero-lift angle of attack from total AoA)
                aoa_eff = aoa_wing_total - alpha0_rad

                # weigh the spanwise lift according to the distributions of aoa_eff
                # (meaning global AoA + local twist - local alpha0)
                spanwise_lift_half_wing_weighted = spanwise_lift_half_wing * aoa_eff

                #                ####################################################################################################
                #                # weigh the spanwise lift according to the local chord (TO BE TESTED). Divide by mac to maintain
                #                # unit consistency
                #                spanwise_lift_half_wing_weighted = spanwise_lift_half_wing_weighted * chord / self.wings[i].mac
                #                ####################################################################################################

                # compute scaling factor to preserve integral lift while weighting the spanwise lift distribution
                # (just the half of the integral wing, as here the half wing is considered)
                scaling_fac_lift = (
                    0.5 * lift / np.sum(spanwise_lift_half_wing_weighted * y_span)
                )

                # rescale spanwise_lift_half_wing_weighted to match the integral lift value
                spanwise_lift_half_wing = (
                    scaling_fac_lift * spanwise_lift_half_wing_weighted
                )

                # compute spanwise m_y on the half wing
                spanwise_m_y_half_wing = m_y * 0.5 / y_span

                # weigh the spanwise moment m_y according to the local chord. Divide by mac to maintain unit consistency
                spanwise_m_y_half_wing_weighted = (
                    spanwise_m_y_half_wing * chord / self.wings[i].mac
                )

                # compute scaling factor to preserve integral m_y while weighting the spanwise moment distribution
                sacling_fac_m_y = (
                    0.5 * m_y / np.sum(spanwise_m_y_half_wing_weighted * y_span)
                )

                # rescale spanwise_m_y_half_wing_weighted to match the integral m_y value
                spanwise_m_y_half_wing_weighted = (
                    sacling_fac_m_y * spanwise_m_y_half_wing_weighted
                )

                # compute elliptical drag distribution
                spanwise_drag_half_wing = (
                    -4
                    * drag
                    / (np.pi * 2 * wing_accumlated_half_span)
                    * np.sqrt(1 - (2 * y_cen / (2 * wing_accumlated_half_span)) ** 2)
                )

                if self.wings[i].symmetry == True:
                    # Create the negative part of the sequences
                    negative_part_y_cen = -np.flip(y_cen)
                    negative_part_y_span = np.flip(y_span)

                    # Mirror the spanwise lift, drag and moment distribution for the other half-wing
                    spanwise_lift_other_half_wing = np.flip(spanwise_lift_half_wing)
                    spanwise_drag_other_half_wing = np.flip(spanwise_drag_half_wing)
                    spanwise_m_y_other_half_wing_weighted = np.flip(
                        spanwise_m_y_half_wing_weighted
                    )

                    # Concatenate the negative part with the original array
                    y_cen = np.concatenate((negative_part_y_cen, y_cen))
                    y_span = np.concatenate((negative_part_y_span, y_span))
                    fz = np.concatenate(
                        (spanwise_lift_other_half_wing, spanwise_lift_half_wing)
                    )
                    fx = np.concatenate(
                        (spanwise_drag_other_half_wing, spanwise_drag_half_wing)
                    )
                    spanwise_m_y_weighted = np.concatenate(
                        (
                            spanwise_m_y_other_half_wing_weighted,
                            spanwise_m_y_half_wing_weighted,
                        )
                    )

                    # fy (-> lateral force on wing) is assumed to be equal along the wingspan and 0
                    fy = np.zeros_like(fz)

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
            spanwise_m_y_weighted.tolist(),
        ]

    # def _correction_factors(self, reynolds_number, aspect_ratio):
    #     cL_alpha = (
    #         0.2923 * reynolds_number**0.1196 * aspect_ratio ** (-0.1231)
    #         - 2.662 * 10 ** (-7) * reynolds_number
    #         + 0.008293 * aspect_ratio
    #     )
    #     alpha_cL0 = 0.8007 + 1.954 * 10**7 * reynolds_number ** (-1.427)
    #     k1 = (
    #         7325 * reynolds_number ** (-0.7374) * aspect_ratio ** (0.3261)
    #         + 8.214 * 10 ** (-7) * reynolds_number
    #         + 0.032 * aspect_ratio
    #     )
    #     k2 = 0.9331 * reynolds_number ** (0.004296) * aspect_ratio ** (0.0003579)

    #     return cL_alpha, alpha_cL0, k1, k2
