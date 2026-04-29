from dataclasses import dataclass
from typing_extensions import Self

import numpy as np

from assembly import Fuselage as Base_Fuselage
from assembly import Configuration as Base_Configuration
from assembly import Environment



@dataclass
class Options:
    length_tail_moment_arm: float = 0.0 # [m]

@dataclass
class Fuselage:
    wetted_area: float # fuselage wetted area in [m**2]
    maximum_width: float # maximum width of the fuselage in [m]
    maximum_height: float # maximum height of the fuselage in [m]
    length: float # length of the fuselage in [m]
    volume_pressurized_cabin: float # volume of the pressurized cabin in [m**3]
    maximum_fuselage_pressure_differential: float # maximum fuselage pressure differential in [Pa]

    @classmethod
    def from_component(cls, comp: Base_Fuselage) -> Self:
        fuselage = Fuselage(
            wetted_area=comp.wetted_area,
            maximum_width=comp.maximum_width,
            maximum_height=comp.maximum_height,
            length=comp.length,
            volume_pressurized_cabin=comp.volume_pressurized_cabin,
            maximum_fuselage_pressure_differential=comp.maximum_fuselage_pressure_differential,
        )
        if "options" in vars(comp):
            for k, v in comp.options.get("raymer_general_aviation_mass_fuselage", {}).items():
                if k in vars(fuselage):
                    setattr(fuselage, k, v)
        return fuselage

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
            for k, v in comp.options.get("raymer_general_aviation_mass_fuselage", {}).items():
                if k in vars(configuration):
                    setattr(configuration, k, v)
        return configuration

class Driver:
    def __init__(
            self,
            environment: Environment,
            fuselage: Fuselage,
            configuration: Configuration,
            options: Options,
    ):
        self.options = options
        self.density = environment.density
        self.velocity = environment.speed
        self.fuselage = fuselage
        self.configuration = configuration
        self.fuselage_mass = 0.0

    def run(self):
        """
        Calculates the fuselage structure mass of a general aviation aircraft according to Raymer p.576 (6th ed.).

        Args:
            wetted_area:                            fuselage wetted area in [m^2]
            ultimate_load_factor:	                ultimate load factor of the aircraft [-]
            m_tom:	                                maximum takeoff mass of the aircraft in [kg]
            maximum_width:	                        width of the fuselage in [m]
            maximum_height:	                        height of the fuselage in [m]
            length:	                                length of the fuselage in [m]
            length_tail_moment_arm:	                length of tail arm in [m]
            dynamic_pressure:                       dynamic pressure at cruise in [Pa]
            volume_pressurized_cabin:               volume of pressurized cabin in [m^3]
            maximum_fuselage_pressure_differential: Maximum fuselage pressure differential in [Pa]

        Returns:
            fuselage_mass: fuselage structure mass in [kg]
        """

        # unit conversion factors
        m2ft = 3.28084
        kg2lbs = 2.20462
        m32ft3 = 35.3147
        m22ft2 = 10.7639
        Pa2lb_ft2 = 0.02088547
        Pa2psi = 0.000145038

        # input data conversion
        fuselage_diameter = (self.fuselage.maximum_height + self.fuselage.maximum_width) / 2 * m2ft
        # take average as diameter
        self.fuselage.maximum_fuselage_pressure_differential = self.fuselage.maximum_fuselage_pressure_differential * Pa2psi
        self.options.length_tail_moment_arm = self.options.length_tail_moment_arm * m2ft
        self.fuselage.volume_pressurized_cabin = self.fuselage.volume_pressurized_cabin * m32ft3
        self.fuselage.length = self.fuselage.length * m2ft
        weight = self.configuration.m_tom * kg2lbs
        self.fuselage.wetted_area = self.fuselage.wetted_area * m22ft2

        dynamic_pressure = 0.5 * self.density * self.velocity ** 2
        dynamic_pressure = dynamic_pressure * Pa2lb_ft2

        # calculation
        self.fuselage_mass = 0.052 * (self.fuselage.wetted_area ** 1.086) * ((self.configuration.ultimate_load_factor * weight) ** 0.177) * (
                self.options.length_tail_moment_arm ** (-0.051)
        ) * ((self.fuselage.length / fuselage_diameter) ** (-0.072)) * (dynamic_pressure ** 0.241) + 11.9 * (
                             (self.fuselage.volume_pressurized_cabin * self.fuselage.maximum_fuselage_pressure_differential) ** 0.271
                     )

        # result conversion
        self.fuselage_mass = self.fuselage_mass / kg2lbs

        return self.fuselage_mass
