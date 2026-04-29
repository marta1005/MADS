import copy
from dataclasses import dataclass, field
from typing_extensions import Self
import numpy as np

from collections.abc import Sequence
from pathlib import Path
from typing import Any

from multiads.solvers import SolverOptions
from multiads.assembly import Propeller, Environment
class Options(SolverOptions):
    """

   ....

    """
    def __init__(
        self,
        *,
        altitude: float = 100.0,
        print_file: bool = False,
        # Jacobian
        jac_approx_type: str = "finite_differences",
        jac_approx_step: float = 1e-03,
        jac_approx_n_processes: int = 1,
        jac_approx_use_threading: bool = False,
        **kwargs: Any,  # noqa: ANN401
    ) -> None:
        super().__init__(**kwargs)
        self.altitude = altitude
        self.print_file = print_file
        # jacobian
        self.jac_approx_type = jac_approx_type
        self.jac_approx_step = jac_approx_step
        self.jac_approx_n_processes = jac_approx_n_processes
        self.jac_approx_use_threading = jac_approx_use_threading

class SimplestPropeller:
    """__________ SIMPLE PROPELLER __________
    The simplest propeller model available, the thrust is calculated as the
    efficiency factor times the shaft power divided by the flight speed.

    Args:
      flight_speed: Flight speed of the aircraft in m/s.
      shaft_power: Shaft power of the motor in Watts.

    Returns:
      thrust: Thrust of the propeller in N.
    """

    def __init__(self, flight_speed):
        self.flight_speed = np.array(flight_speed)

    def calculate_thrust(self, shaft_power):
        # ------------------------ COMPUTE THE RESULTANT THRUST  ------------------------#
        # This thrust calculation is quite simplistic, the efficency fator is one
        efficiency_factor = 1
        thrust = efficiency_factor * shaft_power / self.flight_speed
        return thrust


class MtipPropeller:
    """_______________________ CONSTANT SPEED PROPELLER _______________________
    A more sphisticated propeller model, it is defined with a constant speed
    refered to the blade tip mach number at sea level. The thurst is calculated
    with the standard coeficients and a propeller maps, adjustements are made
    to keep the MDA robust

    Args:
      flight_speed: Flight speed of the aircraft in m/s.
      diameter: Diameter of the selected propeller in meters.
      Mtip: Mach number at the blade tip at sea level.
      propmap: Propeller map, J-Cp map of the propeller.
      density: Air density in kg/m^3.
      shaft_power: Shaft power of the motor in Watts.

    Returns:
      thrust: Thrust of the propeller in N.
    """

    def __init__(
        self, flight_speed, diameter, Mtip, propmap, density, low_speed_correction
    ):
        self.flight_speed = np.array(flight_speed)
        self.diameter = diameter
        self.Mtip = Mtip
        self.propmap = propmap
        self.density = density
        self.low_speed_correction = low_speed_correction

    @staticmethod
    def calculate_prop_power(
        thrust,
        diameter,
        flight_speed,
        Mtip,
        rpm,
        density,
        propmap=None,
        low_speed_correction=False,
    ):
        """_______________________ CONSTANT SPEED PROPELLER _______________________
        A more sphisticated propeller model, it is defined with a constant speed
        refered to the blade tip mach number at sea level. The thurst is calculated
        with the standard coeficients and a propeller maps, adjustements are made
        to keep the MDA robust

        Args:
        flight_speed: Flight speed of the aircraft in m/s.
        diameter: Diameter of the selected propeller in meters.
        Mtip: Mach number at the blade tip at sea level.
        propmap: Propeller map, J-Cp map of the propeller.
        density: Air density in kg/m^3.
        thrust: Thrust of the propeller in N.


        Returns:
        thrust: Thrust of the propeller in N.
        shaft_power: Shaft power of the motor in Watts.

        """

        # ---------------- MAKE THE MDA ROBUST IN CASE OF NEGATIVE POWER ----------------#
        # GEMSEO this has to be done in code base, here we just make the power
        # values positive if they are negative. And we add a slope to keep the direction of
        # the gradient.
        # WARNING: This approaches are more effective when a lower bound is applied over the
        # state variable, this fix that we do here is just so the MDA does not break.
        # if np.any(shaft_power < 1):
        #  shaft_power = np.where(shaft_power<1, 100-0.0002*(1-shaft_power), shaft_power)
        #  shaft_power=shaft_power.flatten()
        #  print('WARNING: Negative power values, forcing positive values into the MDA')

        # ------------------------ CALCULATE OPERATING MAP POINTS ------------------------#
        # Here we calculate the operating points of the propeller that we use to enter the
        # propeller map. This is done by calculating the advance ratio and the power coefficient
        # J = self.flight_speed * np.pi / (self.Mtip * 340)
        if thrust is None:
            print(f"warning :: Propeller thrust computed with semi-empirical model")
            sqrt_prop_pitch = np.sqrt(flight_speed/(rpm*0.0166667))*39.37  # inches
            thrust = (4.392*10**-8 * rpm * (2*diameter/2.0/39.3701)**3.5 / sqrt_prop_pitch* (4.3*10**-4 * rpm * sqrt_prop_pitch**2 -flight_speed ))

        Ct = thrust / (density * Mtip**2 * 340**2 * diameter**2 / (np.pi**2))

        J = flight_speed * np.pi / (Mtip * 340)

        # ------------------------ INTERPOLATE THE PROP MAP ------------------------#
        # Here we need to interpolate in the propeller map, from the previous fix we have ensured
        # that there is no point outside the interpolation range. From the propeller map surrrogate
        # generation we specificed that any value outisde the propeller map is set to a np.nan value.
        # This means that the extrapolation wont happen but wont generate an error.
        # We need to define how to handle the extended values, for this we agan perform a similar process
        # as before. The most critical part will be the cp values, as the propeller map is defined only till
        # cp_max=3.209 with an end value of eta=0.15.
        # Here again we fill the space with the addition of a small slope, a constraint will also help
        # to move the map to a feasible region.

        # nominal power coefficient
        Cp = Ct * J

        # exit velocity from actuator disk
        A_disk = np.pi * (diameter / 2) ** 2
        ue = np.sqrt(
            flight_speed**2
            * (thrust / (A_disk * flight_speed**2 * density / 2) + 1)
        )

        # intoduce propeller efficiency
        if propmap != None:
            eta_interp = propmap(np.array([Cp, J]).T)
            eta_interp = np.where(Cp < 3.2, eta_interp, 0.38 - 0.01 * Cp)
            eta = eta_interp
        else:
            eta = 2 / (1 + ue / flight_speed)  # Ct * J / Cp

        if low_speed_correction:
            Ct_max_static = (
                Cp * 0.5 * (Cp) ** (2 / 3) * (np.pi / 2) ** (1 / 3) / ((Cp) ** 1.27)
            )
            Ct = np.where(Ct > Ct_max_static, Ct_max_static, Ct)
            eta = Ct * J / Cp

        # ------------------------ COMPUTE THE RESULTANT THRUST  ------------------------#
        # Here we calculate the operating points of the propeller that we use to enter the
        # propeller map. This is done by calculating the advance ratio and the power coefficient.
        #
        # If the low speed correction is activated the thrust coefficient is corrected to not to be
        # greater than the maximum value imposed by the momentum theory given the Cp value at that
        # operating point.
        # The equation is adjusted to better mach the propeller maps given by Hamilton.
        shaft_power = 0.5*Cp*density*np.pi*(diameter/2)**2*flight_speed**3

        #  thrust= Ct*self.density*self.Mtip**2*340**2*self.diameter**2/(np.pi**2)

        return thrust, eta, J, shaft_power

    @staticmethod
    def calculate_propeller_weight(diameter=0.0):

        propeller_weight = 0.01 * (np.pi * diameter**2) * 683 / (np.pi * 5.3**2)
        return propeller_weight


class Driver:
    def __init__(
        self,
        propellers: Sequence[Propeller],
        environment: Environment,
        options: Options,
    ) -> None:
        self.options = options
        self.density = environment.density
        self.soundSpeed = environment.sound_speed
        self.speed = environment.speed

        # components
        self.propellers = propellers
        # enumerate component
        self.mapind = {w.name: i for i, w in enumerate(propellers)}
        # properties
        self.eta: Sequence[float] = []          # efficiency
        self.thrust: Sequence[float] = []
        self.shaft_power: Sequence[float] = []
        self.mass: Sequence[float] = []

    def run(self) -> None:
        for props in self.propellers:
            thrust, eta, shaft_power, mass = self._run(
                props, self.speed, self.density, self.soundSpeed
            )
            # append the quantities of interest
            self.eta.append(eta)
            self.thrust.append(thrust)
            self.shaft_power.append(shaft_power)
            self.mass.append(mass)

    @staticmethod
    def _run(props, speed, density, soundSpeed):

        Mtip = (props.rpm * 2 * np.pi / 60 * props.r_tip) / soundSpeed
        
        thrust, eta, J, shaft_power = MtipPropeller.calculate_prop_power(
            props.thrust,    # thrust coefficient als if None
            2 * props.r_tip,
            speed,
            Mtip,
            props.rpm,
            density,
            propmap=None,
            low_speed_correction=False,
        )

        mass = MtipPropeller.calculate_propeller_weight(2 * props.r_tip)

        return thrust, eta, shaft_power, mass

    def compute_total_thrust(self, props_names):
        if not props_names:
            props_names = self.mapind.keys()

        thrust = 0.0
        for name in props_names:
            try:
                i = self.mapind[name]
                thrust += self.thrust[i]
            except KeyError:
                pass

        return thrust

    def compute_efficiency(self, props_names):
        if not props_names:
            props_names = self.mapind.keys()

        efficiency = 0.0
        for name in props_names:
            try:
                i = self.mapind[name]
                efficiency += self.eta[i]
            except KeyError:
                pass

        # debug
        # print(efficiency)

        return efficiency / float(len(props_names))

    def compute_shaft_power(self, props_names):
        if not props_names:
            props_names = self.mapind.keys()
            for name in props_names:
                try:
                    i = self.mapind[name]
                    shaft_power = self.shaft_power[i]
                except KeyError:
                    pass
        else:
            shaft_power = 0.0
            i = self.mapind[props_names]
            shaft_power = self.shaft_power[i]

        return shaft_power

    def compute_mass(self, props_names):
        if not props_names:
            props_names = self.mapind.keys()

        mass = 0.0
        for name in props_names:
            try:
                i = self.mapind[name]
                mass += self.mass[i]
            except KeyError:
                pass

        return mass


if __name__ == "__main__":

    Test1 = True

    """
  ################################## TESTING THE MODELS ##################################

  """
    import matplotlib.pyplot as plt

    if Test1:
        flight_speed = 100  # m/s
        diameter = 2.5  # meters
        Mtip = 0.8  # Mach number
        propmap = None  # Placeholder, not used in this example
        density = 1.225
        propeller = MtipPropeller(flight_speed, diameter, Mtip, propmap, density)
        power_values = np.linspace(-1000, 1000, 100)
        thrust_values = propeller.calculate_thrust(power_values)

        print("Thrust values:", thrust_values)
        print("Power values:", power_values)
        print("Thrust values shape:", thrust_values.shape)
        print("Power values shape:", power_values.shape)

        # Plot thrust as a function of power
        plt.figure()
        plt.plot(power_values, thrust_values, label="Thrust vs Power")
        plt.xlabel("Shaft Power (W)")
        plt.ylabel("Thrust (N)")
        plt.title("Thrust as a Function of Shaft Power")
        plt.legend()
        plt.grid(True)
        plt.show()
