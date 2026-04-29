import copy
from dataclasses import dataclass, field
from typing_extensions import Self
import numpy as np

from collections.abc import Sequence
from pathlib import Path
from typing import Any

from multiads.solvers import SolverOptions
from multiads.assembly import FuelCell, Environment, PowerManagSyst


class Options(SolverOptions):
    """
    A simple fuel cell system analyzer that computes:
      - Electrical power output (W)
      - Hydrogen fuel flow (kg/s)
      - Waste heat produced (W)
      - Fuel cell system weight (kg)

    The computation uses the given inputs and a set of default parameters.

            Parameters:
            activation_area (float): Total activation area (cm^2)
            current_density (float or np.array): Current density (A/cm^2)

    """
    def __init__(
        self,
        *,
        activation_area: float = 100.0,
        current_density: float = 0.5,
        delta_G: float = 230000.0,  # Free enthalpy of gaseous H2 [J/mol]
        n_e: float = 2,  # Number of exchanged electrons
        F: float = 96485.3399,  # Faraday constant [A*s/mol]
        R: float = 8.314472,  # Molar gas constant [J/K/mol]
        alpha_a: float = 2,  # Anode transfer coefficient
        alpha_c: float = 0.5,  # Cathode transfer coefficient
        M_H2: float = 2.01588,  # Hydrogen molar weight [g/mol]
        T: float = 273.15 + 70.0,  # Fuel cell temperature [K]
        p_H2: float = 2.5,  # Hydrogen partial pressure [bar]
        p_O2: float = 2,  # Oxygen partial pressure [bar]
        r: float = 0.238,  # Cell resistance [ohms*cm**2]
        i_0a: float = 0.2,  # Anode exchange current density [A/cm**2]
        i_0c: float = 0.0001,  # Cathode exchange current density [A/cm**2]
        i_n: float = 0.002,  # Internal current density [A/cm**2]
        i_la: float = 1,  # Anode limiting current density [A/cm**2]
        i_lc: float = 1,  # Cathode limiting current density [A/cm**2]
        stoic: float = 1.1,  # Stoichiometric ratio
        sp: float = 2.5,  # Specific power [kW/kg]
        percentage: float = 0.85,  # Percentage of limiting current density for max power
        # Jacobian
        jac_approx_type: str = "finite_differences",
        jac_approx_step: float = 1e-03,
        jac_approx_n_processes: int = 1,
        jac_approx_use_threading: bool = False,
        **kwargs: Any,  # noqa: ANN401
    ) -> None:
        super().__init__(**kwargs)
        self.activation_area = activation_area
        self.current_density = current_density
        self.delta_G = delta_G
        self.n_e = n_e
        self.F = F
        self.R = R
        self.alpha_a = alpha_a
        self.alpha_c = alpha_c
        self.M_H2 = M_H2
        self.T = T
        self.p_H2 = p_H2
        self.p_O2 = p_O2
        self.r = r
        self.i_0a = i_0a
        self.i_0c = i_0c
        self.i_n = i_n
        self.i_la = i_la
        self.i_lc = i_lc
        self.stoic = stoic
        self.sp = sp
        self.percentage = percentage
        self.jac_approx_type = jac_approx_type
        self.jac_approx_step = jac_approx_step
        self.jac_approx_n_processes = jac_approx_n_processes
        self.jac_approx_use_threading = jac_approx_use_threading


class Driver:
    def __init__(
        self,
        fuelcells: Sequence[FuelCell],
        powermanag_sys: PowerManagSyst,
        environment: Environment,
        options: Options,
    ) -> None:
        self.options = options
        # recover environment variables
        self.density = environment.density
        self.soundSpeed = environment.sound_speed
        self.viscosity = environment.dyn_viscosity
        self.velocity = environment.velocity
        
        # components
        self.fuelcells = fuelcells
        self.pwr_mng = powermanag_sys
        
        # properties
        self.mass: Sequence[float] = []
        self.h2_fuel_flow: Sequence[float] = []
        self.waste_heat: Sequence[float] = []
        self.power: Sequence[float] = []
        
        # mapping if needed
        self.mapind = {fc.name: i for i, fc in enumerate(self.fuelcells)}

    def run(self) -> None:
        for fuel_cell in self.fuelcells:
            (
                power,
                h2_fuel_flow,
                waste_heat,
                mass,
            ) = self._run(fuel_cell, self.pwr_mng, self.options)

            # store properties per engine
            self.mass.append(mass)
            self.waste_heat.append(waste_heat)
            self.h2_fuel_flow.append(h2_fuel_flow)
            self.power.append(power)

    @staticmethod
    def _run(fuel_cell, pwr_mng, options) -> tuple[float, float, float]:
        """
        Compute the fuel cell outputs.

        Parameters:
            NA (float): Total activation area (cm^2)
            i (float or np.array): Current density (A/cm^2)

        Returns:
            dict: A dictionary with keys:
                - 'elec_power_out': Electrical power output (W)
                - 'h2_fuel_flow': Hydrogen fuel flow (kg/s)
                - 'heat_out': Waste heat produced (W)
                - 'fcs_weight': Fuel cell system weight (kg)
        """
        # Compute the term used in power and heat calculations
        term = (
            -options.delta_G / (options.n_e * options.F)
            + options.R
            * options.T
            / (options.n_e * options.F)
            * np.ma.log(options.p_H2 * options.p_O2**0.5)
            - options.R
            * options.T
            / (options.alpha_c * options.F)
            * np.ma.log(
                (options.current_density + options.i_n) / (2 * options.i_0c)
                + np.sqrt(
                    ((options.current_density + options.i_n) / (2 * options.i_0c)) ** 2
                    + 1
                )
            )
            - options.R
            * options.T
            / (options.alpha_a * options.F)
            * np.ma.log(
                (options.current_density + options.i_n) / (2 * options.i_0a)
                + np.sqrt(
                    ((options.current_density + options.i_n) / (2 * options.i_0a)) ** 2
                    + 1
                )
            )
            - options.r * (options.current_density + options.i_n)
            + options.R
            * options.T
            / (4 * options.F)
            * np.ma.log(1 - (options.current_density + options.i_n) / options.i_lc)
            + options.R
            * options.T
            / (2 * options.F)
            * np.ma.log(1 - (options.current_density + options.i_n) / options.i_la)
        )

        # split in case the input is provide about electrical power
        if pwr_mng.electrical_power_from_fc == None:
            elec_power_out = options.activation_area * options.current_density * term
            # activation area  X current density
            activarea_currdens = options.activation_area * options.current_density
            # Compute hydrogen fuel flow (kg/s)
            h2_fuel_flow = (
                (
                    options.current_density
                    * options.activation_area
                    / (2 * options.F)
                    * options.M_H2
                )
                * options.stoic
                / 1000
            )
        else:
            # recover fuel cell power
            elec_power_out = fuel_cell.power

            # activation area  X current density
            activarea_currdens = (
                fuel_cell.power / term
            )  # options.activation_area * options.current_density

            # debug
            # print(pwr_mng.electrical_power_from_fc, term, fuel_cell.fc_power)

            # update activation area
            options.activation_area = activarea_currdens / options.current_density

            # update fuel flow
            h2_fuel_flow = (
                (activarea_currdens / (2 * options.F) * options.M_H2)
                * options.stoic
                / 1000
            )

        # Compute waste heat produced (W)
        heat_out = elec_power_out * (1.481 / term - 1)

        # Maximum power calculation for weight determination
        maximum_power = (
            options.activation_area
            * options.percentage
            * options.i_la
            * (
                -options.delta_G / (options.n_e * options.F)
                + options.R
                * options.T
                / (options.n_e * options.F)
                * np.ma.log(options.p_H2 * options.p_O2**0.5)
                - options.R
                * options.T
                / (options.alpha_c * options.F)
                * np.ma.log(
                    (options.percentage * options.i_la + options.i_n)
                    / (2 * options.i_0c)
                    + np.sqrt(
                        (
                            (options.percentage * options.i_la + options.i_n)
                            / (2 * options.i_0c)
                        )
                        ** 2
                        + 1
                    )
                )
                - options.R
                * options.T
                / (options.alpha_a * options.F)
                * np.ma.log(
                    (options.percentage * options.i_la + options.i_n)
                    / (2 * options.i_0a)
                    + np.sqrt(
                        (
                            (options.percentage * options.i_la + options.i_n)
                            / (2 * options.i_0a)
                        )
                        ** 2
                        + 1
                    )
                )
                - options.r * (options.percentage * options.i_la + options.i_n)
                + options.R
                * options.T
                / (4 * options.F)
                * np.ma.log(
                    1 - (options.percentage * options.i_la + options.i_n) / options.i_lc
                )
                + options.R
                * options.T
                / (2 * options.F)
                * np.ma.log(
                    1 - (options.percentage * options.i_la + options.i_n) / options.i_la
                )
            )
        )
        fcs_weight = maximum_power / (1000 * options.sp)

        return elec_power_out, h2_fuel_flow, heat_out, fcs_weight

    # [to be connected with the model to feed the jacobian]
    def _get_sensitivity(self) -> float:
        """
        compute the adjoint of the shaft power

        """
        pass

    # -------------------------------------------------------------------------
    # Compute outputs
    # -------------------------------------------------------------------------   
    def mass_fuel_cell(self, fuelcell_names: list[str]) -> tuple[float, float]:
        if not fuelcell_names:
            fuelcell_names = self.mapind.keys()

        fuelcell_total_mass = 0.0
        for name in fuelcell_names:
            try:
                i = self.mapind[name]
                fuelcell_total_mass += self.weights[i]
            except KeyError:
                pass

        return fuelcell_total_mass

    def h2flow_fuel_cell(self, fuelcell_names: list[str]) -> tuple[float, float]:
        if not fuelcell_names:
            fuelcell_names = self.mapind.keys()

        h2flow_total = 0.0
        for name in fuelcell_names:
            try:
                i = self.mapind[name]
                h2flow_total += self.h2_fuel_flow[i]
            except KeyError:
                pass

        return h2flow_total


if __name__ == "__main__":

    import sys
    import os

    sys.path.insert(
        0,
        os.path.abspath(os.path.join(os.path.dirname(__file__), "../../src")),
    )
    print(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../src")))

    # Sample input values
    NA = 100.0  # Total activation area (cm^2)
    i = 0.5  # Current density (A/cm^2); this can also be a numpy array for multiple values

    # Instantiate the fuel cell analyzer
    fuel_cell = Driver()

    # Compute the outputs
    results = fuel_cell.run(NA, i)

    # Display the results
    print("Electric Power Output (W):", results["elec_power_out"])
    print("Hydrogen Fuel Flow (kg/s):", results["h2_fuel_flow"])
    print("Heat Output (W):", results["heat_out"])
    print("Fuel Cell System Weight (kg):", results["fcs_weight"])
