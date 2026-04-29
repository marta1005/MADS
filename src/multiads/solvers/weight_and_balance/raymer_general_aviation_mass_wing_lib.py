from dataclasses import dataclass

import utilities.units as u
from assembly import Configuration as Base_Configuration
from assembly import Environment, Section, Span
from assembly import Wing as Base_Wing
from typing_extensions import Self
from utilities.algebra import cosd, softmax


@dataclass
class Options:
    pass


@dataclass
class Wing:
    ar: float  # aspect ratio [-]
    tr: float  # taper ratio [-]
    area: float  # wing area [m*22]
    sweep_25: float  # mean wing sweep at 25% chord line [deg]
    spans: list[Span]  # wing span information
    sections: list[Section]  # information about the wing section
    fuel_mass_in_wing: float  # fuel mass carried by the wing [kg]

    @classmethod
    def from_component(cls, comp: Base_Wing) -> Self:
        wing = Wing(
            ar=comp.aspect_ratio,
            tr=comp.taper_ratio,
            area=comp.area,
            sweep_25=comp.sweep_at_chord_station(
                0.25
            ),  # TODO @Tim: how is this handled? This calls a function inside the assembly. The function itself depends on parameters that themselves can be "Variables". How is the updating of these variables arranged (as they are not explicitly listed in the "required inputs" list in the solver interface)?
            spans=comp.spans,  # TODO @Tim: is this really needed?
            sections=comp.sections,
            fuel_mass_in_wing=comp.fuel_mass_in_wing,
        )
        if "options" in vars(comp):
            for k, v in comp.options.get(
                "raymer_general_aviation_mass_wing",
                {},
            ).items():
                if k in vars(wing):
                    setattr(wing, k, v)
        return wing


@dataclass
class Configuration:
    m_tom: float  # maximum takeoff mass in [kg]
    ultimate_load_factor: float  # ultimate load factor [-]

    @classmethod
    def from_component(cls, comp: Base_Configuration) -> Self:
        configuration = Configuration(
            m_tom=comp.m_tom,
            ultimate_load_factor=comp.ultimate_load_factor,
        )
        if "options" in vars(comp):
            for k, v in comp.options.get(
                "raymer_general_aviation_mass_wing",
                {},
            ).items():
                if k in vars(configuration):
                    setattr(configuration, k, v)
        return configuration


class Driver:
    def __init__(
        self,
        environment: Environment,
        wings: list[Base_Wing],
        configuration: Configuration,
        options: Options,
    ) -> None:
        self.options = options
        self.density = environment.density
        self.velocity = environment.speed
        self.mapind = {w.name: i for i, w in enumerate(wings)}
        self.configuration = configuration
        self.wing_mass = [0.0 for _ in wings]

        self.wings = [Wing.from_component(w) for w in wings]

    def run(self) -> None:
        for i, wing in enumerate(self.wings):
            wing_mass = self._run(
                wing,
                self.density,
                self.velocity,
                self.options,
                self.configuration,
            )
            self.wing_mass[i] = wing_mass

    @staticmethod
    def _run(
        wing: Wing,
        density: float,
        velocity: float,
        options: Options,
        configuration: Configuration,
    ) -> float:
        """Computes the mass of a wing of a general aviation aircraft, according to Raymer's Aircraft Design: A Conceptual
        Approach, p.575 (6th ed.).

        Args:
            wing: The wing object.

            m_tom: The design takeoff gross weight of the entire aircraft [kg].

            ultimate_load_factor: The ultimate load factor of the aircraft.

            fuel_mass_in_wing: The mass of fuel in the wing [kg]. If there is no fuel in the wing, set this to 0.

                Note: Model extrapolates strangely for infinitesimally-small-but-nonzero fuel masses; don't let an
                optimizer land here.

            cruise_op_point: The cruise operating point of the aircraft.

            use_advanced_composites: Whether to use advanced composites for the wing. If True, the wing mass is modified
            accordingly.

        Returns: The mass of the wing [kg].

        """
        # run the model
        if wing.fuel_mass_in_wing > 0:
            fuel_weight_factor = softmax(
                (wing.fuel_mass_in_wing / u.lbm) ** 0.0035,
                1,
                hardness=1000,
            )
        else:
            fuel_weight_factor = 1

        # select section # TODO @Tim: here, a mean thickness_to_chord_ratio of the overall wing is required
        section = wing.sections[0]
        thickness_to_chord_ratio = section.thickness_to_chord_ratio

        dynamic_pressure = 0.5 * density * velocity**2

        wing_mass = (
            0.036
            * (wing.area / u.foot**2) ** 0.758
            * fuel_weight_factor
            * (wing.ar / cosd(wing.sweep_25) ** 2) ** 0.6
            * (dynamic_pressure / u.psf) ** 0.006
            * wing.tr**0.04
            * (100 * thickness_to_chord_ratio / cosd(wing.sweep_25)) ** -0.3
            * (configuration.m_tom / u.lbm * configuration.ultimate_load_factor) ** 0.49
        ) * u.lbm

        return wing_mass

    def retrieve_wing_mass(self, wing_names: list[str]) -> float:
        if not wing_names:
            wing_names = self.mapind.keys()

        wing_mass = 0.0
        for name in wing_names:
            try:
                i = self.mapind[name]
                wing_mass += self.wing_mass[i]
            except KeyError:
                pass

        return wing_mass
