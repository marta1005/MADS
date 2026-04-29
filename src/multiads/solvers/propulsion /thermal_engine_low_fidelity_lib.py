# -*- coding: utf-8 -*-
"""

# Inputs
#   : max_power_rating # P = M*omega [w]

    
# parameters (within the model that can be moved to inputs)
    : throttles      # in this case defined within the model
    : psfcs          # in this case defined by the model
    : time_steps     # this can be included from the mission

# Outputs
#   : engine_mass
#   : cumulative_fuel_consumed
#   : co2_emission_estimation

"""

import numpy as np
from matplotlib import pyplot as plt
from scipy.interpolate import UnivariateSpline


from scipy.integrate import simpson

# MADS integration
import copy
from typing_extensions import Self
import numpy as np
from dataclasses import dataclass, field
from numpy.typing import NDArray



from collections.abc import Sequence
from pathlib import Path
from typing import Any

from multiads.solvers import SolverOptions
from multiads.assembly import PropulsionSystem, Environment

class Options(SolverOptions):
    """

   ....

    """
    def __init__(
        self,
        *,
        segment_duration: float = 100.0, # [s]
        throttle_map: NDArray[np.float64] | None = None,
        # Jacobian
        jac_approx_type: str = "finite_differences",
        jac_approx_step: float = 1e-03,
        jac_approx_n_processes: int = 1,
        jac_approx_use_threading: bool = False,
        **kwargs: Any,  # noqa: ANN401
    ) -> None:
        super().__init__(**kwargs)
        self.segment_duration = segment_duration, # duration of the segment ins [s]
        self.throttle_map = throttle_map
        # jacobian
        self.jac_approx_type = jac_approx_type
        self.jac_approx_step = jac_approx_step
        self.jac_approx_n_processes = jac_approx_n_processes
        self.jac_approx_use_threading = jac_approx_use_threading

@staticmethod
def engine_mass_estimation(max_power_rating):
    engine_mass = max_power_rating / (9.81 * 0.379)
    # max power rating in kW
    # engine mass in kg
    # 9.81 is gravitational constant g in SI Units
    # Correlation from Howe et. al ADD, Aircraft Conceptual Design Synthesis
    # Chapter 6 - Basic lift, drag and mass representations
    return engine_mass


@staticmethod
def psfc_throttleMAP():
    """
    Provides throttle vs PSFC relationship data
    PW127 gas turbine PSFC vs throttle (realistic distribution data taken from the article
    Feasibility studies on regional aircraft retrofitted with hybrid-electric powertrains
    Quiben Figueroa, Raul, Cavallaro Rauno, Cini Andrea)
    """
    throttles = [
        -0.001,
        0.05,
        0.1,
        0.15,
        0.2,
        0.25,
        0.3,
        0.35,
        0.4,
        0.45,
        0.50,
        0.55,
        0.60,
        0.65,
        0.70,
        0.75,
        0.80,
        0.85,
        0.90,
        0.95,
        1.00,
        3,
    ]
    psfcs = [
        0.450,
        0.425,
        0.401,
        0.376,
        0.352,
        0.327,
        0.303,
        0.285,
        0.276,
        0.271,
        0.270,
        0.272,
        0.275,
        0.279,
        0.284,
        0.289,
        0.293,
        0.298,
        0.305,
        0.320,
        0.350,
        0.450,
    ]
    return throttles, psfcs


# Step 3: Define a function for power
@staticmethod
def calculate_power(throttle, max_power_rating=1 * 10**4):  # max power can be changed
    """
    Simulates power as a linear function of throttle.
    max_power: Maximum power output at full throttle (in kW).
    """
    return throttle * max_power_rating


# Step 4: Define a function for fuel flow based on PSFC and Power
@staticmethod
def fuel_flow(throttle,psfc_spline):
    """
    Calculate fuel flow given throttle.
    Fuel Flow = PSFC * Power
    """
    power = calculate_power(throttle)  # Power depends on throttle
    return psfc_spline(throttle) * power  # PSFC (kg/kWh) * Power (kW)


@staticmethod
def co2_emission_estimation(mass_fuel):
    mass_co2 = (3.14 + 0.61) * mass_fuel
    # kg co2 emissions per kg of fuel
    # Correlation from Hoelzen et Al.
    # Conceptual Design of Operation Strategies for Hybrid Electric Aircraft. 2017

    return mass_co2


class Driver:
    def __init__(
        self,
        engines: Sequence[PropulsionSystem],
        environment: Environment,
        options: Options,
    ) -> None:
        self.options = options
        self.density = environment.density
        self.soundSpeed = environment.sound_speed
        self.viscosity = environment.dyn_viscosity
        self.velocity = environment.speed
        
        # map engine to index
        self.mapind = {e.name: i for i, e in enumerate(engines)}

        # instatiate the engine from the base class
        self.engines = engines

        # initialise main engine QOI
        self.mass: Sequence[float] = [] 
        self.fuel_consumption: Sequence[float] = [] 
        self.co2_emission: Sequence[float] = [] 

    def run(self) -> None:
        for engine in self.engines:
            (
                mass,
                fuel_consumption,
                co2_emission,
            ) = self._run(engine, self.options)
            self.mass.append(mass)
            self.fuel_consumption.append(fuel_consumption)
            self.co2_emission.append(co2_emission)

    @staticmethod
    def _run(engine, options) -> tuple[float, float, float]:
        """ """
        # setup options
        # (shaft_power, efficiency, fuel_power, power_rating, voltage_rating):
        time_steps = options.segment_duration

        # recover updated values
        max_power_rating = engine.power

        # step 0 estimate mass from power
        # max_power_rating = 1 * 10**4  # Example. This is an input
        engine_mass = engine_mass_estimation(max_power_rating)

        # Step 1: Get throttle and PSFC data
        if options.throttle_map.any():
            throttles = options.throttle_map[0,:]
            psfcs = options.throttle_map[1,:]
        else:
            throttles, psfcs = psfc_throttleMAP()

        # Step 2: Interpolate PSFC using Univariate Spline
        psfc_spline = UnivariateSpline(throttles, psfcs, s=0)

        # Generate fine throttle values for the curve
        throttle_fine = np.linspace(0, 1, 1000)
        psfc_fine = psfc_spline(throttle_fine)

        # Step 5: Calculate cumulative fuel consumption using integration
        time_steps = np.linspace(
            0, options.segment_duration, len(throttle_fine)
        )  # Simulate over 3600 seconds
        fuel_flows = fuel_flow(throttle_fine,psfc_spline)  # Compute fuel flow at each throttle step

        # Integrate fuel flow over time to calculate total fuel consumed
        cumulative_fuel_consumed = simpson(np.reshape(fuel_flows,(1,1000)), x=np.reshape(time_steps,(1,1000)),axis=-1)   
        
        # cumulative consumption of co2
        cumulative_co2_emission = co2_emission_estimation(cumulative_fuel_consumed)

        #return output
        return engine_mass, cumulative_fuel_consumed[0], cumulative_co2_emission

    def engine_mass(self) -> tuple[float]:
        if not engine_names:
            engine_names = self.mapind.keys()

        engine_mass = 0.0
        for name in engine_names:
            try:
                i = self.mapind[name]
                engine_mass += self.engine_mass[i]
            except KeyError:
                pass

        return engine_mass

    def cumulative_fuel_consumed(self) -> tuple[float]:
        if not engine_names:
            engine_names = self.mapind.keys()

        cumulative_fuel_consumed = 0.0
        for name in engine_names:
            try:
                i = self.mapind[name]
                cumulative_fuel_consumed += self.cumulative_fuel_consumed[i]
            except KeyError:
                pass

        return cumulative_fuel_consumed

    def cumulative_co2_emission(self) -> tuple[float]:
        if not engine_names:
            engine_names = self.mapind.keys()

        cumulative_co2_emission = 0.0
        for name in engine_names:
            try:
                i = self.mapind[name]
                cumulative_co2_emission += self.cumulative_co2_emission[i]
            except KeyError:
                pass

        return cumulative_co2_emission


# -----------------------
# if __name__ == __main__:
if __name__ == "__main__":

    # step 0 estimate mass from power
    max_power_rating = 1 * 10**4  # Example. This is an input
    engine_mass = engine_mass_estimation(max_power_rating)

    # Step 1: Get throttle and PSFC data
    throttles, psfcs = psfc_throttleMAP()

    # Step 2: Interpolate PSFC using Univariate Spline
    psfc_spline = UnivariateSpline(throttles, psfcs, s=0)

    # Generate fine throttle values for the curve
    throttle_fine = np.linspace(0, 1, 1000)
    psfc_fine = psfc_spline(throttle_fine)

    # Step 5: Calculate cumulative fuel consumption using integration
    time_steps = np.linspace(0, 3600, len(throttle_fine))  # Simulate over 3600 seconds
    fuel_flows = fuel_flow(throttle_fine)  # Compute fuel flow at each throttle step

    # Integrate fuel flow over time to calculate total fuel consumed
    cumulative_fuel_consumed = simpson(fuel_flows, time_steps)

    cumulative_co2_emission = co2_emission_estimation(cumulative_fuel_consumed)

    # Step 6: Plot the results
    plt.figure(figsize=(22, (22 * 9 / 16)))
    plt.plot(throttles, psfcs, "o", label="Data Points")
    plt.plot(throttle_fine, psfc_fine, "-", label="Interpolated Curve")
    plt.xlim(0, 1)
    plt.xlabel("Throttle", fontsize=20)
    plt.ylabel("PSFC (kg/kW/hr)", fontsize=20)
    plt.title("Throttle vs PSFC", fontsize=20)
    plt.legend(fontsize=20)

    # Plot Cumulative Fuel Consumed
    plt.figure(figsize=(22, (22 * 9 / 16)))
    plt.plot(
        time_steps,
        np.cumsum(fuel_flows) * np.gradient(time_steps),
        label="Cumulative Fuel Consumed",
    )
    plt.xlabel("Time (s)", fontsize=20)
    plt.ylabel("Fuel Consumed (kg)", fontsize=20)
    plt.title("Cumulative Fuel Consumption Over Time", fontsize=20)
    plt.legend(fontsize=20)

    # plt.tight_layout()
    plt.show()

    # Step 7: Output total fuel consumed
    print(f"Total Fuel Consumed over 3600 seconds: {cumulative_fuel_consumed:.2f} kg")
