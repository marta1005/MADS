from dataclasses import dataclass, field
from typing_extensions import Self

import numpy as np

from assembly import Fuselage as Base_Fuselage
from assembly import Wing as Base_Wing
from assembly import Configuration as Base_Configuration


@dataclass
class Options:
    wing_names_contributing_to_total_span: list[str] = field(default_factory=dict) # list of the names of wings that are
    # to be considered for the effective span that is provided to the semi-empirical formula (relevant for biplanes:
    # here, the spans of all listed wings will be summed)


@dataclass
class Fuselage:
    length: float  # length of the fuselage in [m]

    @classmethod
    def from_component(cls, comp: Base_Fuselage) -> Self:
        fuselage = Fuselage(
            length=comp.length,
        )
        if "options" in vars(comp):
            for k, v in comp.options.get("raymer_general_aviation_mass_controls", {}).items():
                if k in vars(fuselage):
                    setattr(fuselage, k, v)
        return fuselage


@dataclass
class Wing:
    name: str  # name of the wing
    span: float  # overall wing span [m]

    @classmethod
    def from_component(cls, comp: Base_Wing) -> Self:
        wing = Wing(
            span=comp.span,
            name = comp.name,
        )
        if "options" in vars(comp):
            for k, v in comp.options.get("raymer_general_aviation_mass_wing", {}).items():
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
            for k, v in comp.options.get("raymer_general_aviation_mass_controls", {}).items():
                if k in vars(configuration):
                    setattr(configuration, k, v)
        return configuration

class Driver:
    def __init__(
            self,
            fuselage: Fuselage,
            wings: list[Base_Wing],
            configuration: Configuration,
            options: Options,
    ):
        self.options = options
        self.fuselage = fuselage
        self.configuration = configuration
        self.flight_controls_mass = 0.0
        self.wings = [Wing.from_component(w) for w in wings]

    def run(self):
        """
            Calculates the flight controls mass of a general aviation aircraft according to Raymer p.576 (6th ed.).

        Args:
            length:                 total fuselage length [m]
            wing_span:              wingspan [m]
            m_tom:                  maximum takeoff weight of the aircraft [kg]
            ultimate_load_factor:   ultimate load of the aircraft [-]

        Returns:
            flight_controls_mass: flight controls mass [kg]
        """

        # unit conversion factors
        kg2lbs = 2.20462
        m2ft = 3.28084

        # initialize
        effective_span=0.0

        # obtain effective wing span (in case of a biplane, the multiple wing spans need to be summed)
        for wing in self.wings:
            if wing.name in self.options.wing_names_contributing_to_total_span:
                effective_span += wing.span

        # input data conversion
        self.fuselage.length = self.fuselage.length * m2ft
        effective_span = effective_span * m2ft
        self.configuration.m_tom = self.configuration.m_tom * kg2lbs

        # calculation
        flight_controls_mass = 0.053 * (self.fuselage.length ** 1.536) * (effective_span ** 0.371) * (
                    self.configuration.ultimate_load_factor * self.configuration.m_tom * 10 ** (-4)) ** 0.8

        # result conversion
        self.flight_controls_mass = flight_controls_mass / kg2lbs

        return self.flight_controls_mass
