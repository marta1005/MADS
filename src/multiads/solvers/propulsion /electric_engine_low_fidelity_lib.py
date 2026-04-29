import copy
from dataclasses import dataclass, field
from typing_extensions import Self
import numpy as np
from assembly import Environment as Base_Environment
from assembly import PropulsionSystem as Base_PropulsionSystem
from assembly import PowerManagSyst as Base_PowerManagSyst
from assembly import PowerManagSyst as Base_Propeller


@dataclass
class Options:
    """
    Analyze a simple electric motor.

    Default values inside the function definition can be changed

    Parameters:
        throttle (float or np.ndarray): Power control setting (0 to 1).
        elec_power_rating (float): Rated electrical power (W).
        tech_level (str): Technology level, either 'SOA' or 'HTS'.
        efficiency_SOA (float): Efficiency for SOA technology.
        efficiency_HTS (float): Efficiency for HTS technology.
        weight_inc_SOA (float): Weight increase per watt for SOA.
        weight_inc_HTS (float): Weight increase per watt for HTS.
        weight_base (float): Base weight (kg).
        cost_inc (float): Cost increment per watt (USD/W).
        cost_base (float): Base cost (USD).

    Returns:
        dict: Dictionary containing computed outputs:
            - shaft_power_out: Shaft power output (W)
            - heat_out: Waste heat produced (W)
            - elec_load: Electrical load consumed (W)
            - component_cost: Motor component cost (USD)
            - component_weight: Motor component weight (kg)
            - component_sizing_margin: Fraction of rated power (same as throttle)


    """

    tech_level: str = "SOA"  # technology type # Technology level ('SOA' or 'HTS')
    efficiency_SOA: float = 0.96
    efficiency_HTS: float = 0.98
    weight_base: float = 0.0  # reference weight for the engine as baseline [kg]
    elec_power_rating: float = 1000000.0  # Rated electrical power in Watts
    efficiency: float = 0.8  # [n.d.] efficiency of the engine shaft to engine
    voltage_rating: float = 800.0  # [V] voltage
    fuel_power: float = 500.0  # [W] the fuel equivaletn power in Watt
    efficiency_gear_box: float = 0.98  # efficiency gear box
    jac_approx_type: str = "finite_differences"
    jac_approx_step: float = 1e-03
    jac_approx_n_processes: int = 1
    jac_approx_use_threading: bool = False


@dataclass
class PropulsionSystem:
    type: str  # engine type (see assembly) ['Electric','Hybrid','Thermal']
    engine_power: float  # [W] shaft power in Watt
    shaft_power: float  # [W] shaft power in Watt

    @classmethod
    def from_component(cls, comp: Base_PropulsionSystem) -> Self:
        engine = PropulsionSystem(
            type=comp.type,
            engine_power=comp.engine_power,  # [W] shaft power in Watt
            shaft_power=comp.shaft_power,  # [W] shaft power in Watt
        )
        if "options" in vars(comp):
            for k, v in comp.options.get("e_eng_lf", {}).items():
                if k in vars(engine):
                    setattr(engine, k, v)
        return engine
    
@dataclass
class PowerManagSyst:
    hybrid_factor_engine: float = 0.0  # combustion to electric hybridization - set to 0 if pure electric

    @classmethod
    def from_component(cls, comp: Base_PowerManagSyst) -> Self:
        if comp is not None:
            pwr_mng = PowerManagSyst(
                hybrid_factor_engine=comp.hybrid_factor_engine,  # combustion to electric hybridization - set to 0 if pure electric
            )
            if "options" in vars(comp):
                for k, v in comp.options.get("power_mng_lf", {}).items():
                    if k in vars(pwr_mng):
                        setattr(pwr_mng, k, v)
        else:
            # initialise dummy component if not specified
            pwr_mng = Base_PowerManagSyst(name="",type="Electric")
            pwr_mng.hybrid_factor_engine = PowerManagSyst.hybrid_factor_engine
             
        return pwr_mng


class Driver:
    def __init__(
        self,
        environment: Base_Environment,
        engines: list[Base_PropulsionSystem],
        pwr_mng: Base_PowerManagSyst,
        options: field(default_factory=Options),
    ) -> None:
        self.options = options
        self.density = environment.density
        self.soundSpeed = environment.sound_speed
        self.viscosity = environment.dyn_viscosity
        self.velocity = environment.velocity
        self.mapind = {e.name: i for i, e in enumerate(engines)}

        # Instatiate the engine from the base class
        self.engines = [PropulsionSystem.from_component(e) for e in engines]        
        # Power Management
        #if hasattr(pwr_mng,"name"):
        self.pwr_mng = PowerManagSyst.from_component(pwr_mng)
        #else:
        #   self.pwr_mng = None
        #    self.pwr_mng.hybrid_factor_engine = 0.0
               
        # initialise main engine QOI
        self.electric_engine_power = [0.0 for _ in engines]
        self.electric_engine_mass = [0.0 for _ in engines]
        self.electric_engine_voltage = [0.0 for _ in engines]
        self.engine_temperature = [0.0 for _ in engines]

    def run(self) -> None:
        for i, engine in enumerate(self.engines):
            
            # debug
            #print("engine shaft power", engine.shaft_power)
            #print(engine.type)
            
            (
                electric_engine_power,
                electric_engine_voltage,
                engine_temperature,
                electric_engine_mass,
            ) = self._run(engine, self.pwr_mng, self.options)

            # store properties per engine
            self.electric_engine_power[i] = electric_engine_power
            self.electric_engine_mass[i] = electric_engine_mass
            self.electric_engine_voltage[i] = electric_engine_voltage
            self.engine_temperature[i] = engine_temperature
        
        # debug
        #print(
        #        "electric engine:",
        #        self.electric_engine_power,
        #        self.electric_engine_mass,
        #        self.engine_temperature,
        #)

    @staticmethod
    def _run(engine, pwr_mng, options) -> tuple[float, float, float]:
        """
        Calculate the electric engine power, voltage, and temperature from the shaft power for a hybrid engine, with a power rating and voltage rating.

        :param shaft_power: The power at the shaft in watts.
        :param efficiency: The efficiency of the electric engine as a decimal (e.g. 0.8 for 80%).
        :param fuel_power: The power provided by the fuel engine in watts.
        :param power_rating: The maximum power rating of the electric engine in watts.
        :param voltage_rating: The maximum voltage rating of the electric engine in volts.
        :return: The power of the electric engine in watts, the voltage in volts, and the temperature in degrees Celsius.
        """

        # Select parameters based on the technology level

        # P. Wheeler et Al. "Electric/Hybrid-Electric Aircraft Propulsion Systems"
        # doi: 10.1109/JPROC.2021.3073291

        if options.tech_level == "SOA":
            eta_m = options.efficiency_SOA
            weight_inc = (
                1.96 * 0.453592 * (options.elec_power_rating / 1000) ** (0.8997)
            )  # 0.453592 is conversion to kg
        elif options.tech_level == "HTS":
            eta_m = options.efficiency_HTS
            weight_inc = (
                2.28 * 0.453592 * (options.elec_power_rating / 1000) ** (0.6616)
            )  # 0.453592 is conversion to kg
        else:
            raise ValueError("tech_level must be 'SOA' or 'HTS'")
        # setup options
        # (shaft_power, efficiency, fuel_power, power_rating, voltage_rating):
        efficiency = options.efficiency * options.efficiency_gear_box
        voltage_rating = options.voltage_rating
        power_rating = options.elec_power_rating
        fuel_power = options.fuel_power

        # recover updated values
        shaft_power = engine.shaft_power

        # compute quantities
        if engine.type == "Electric":
            electric_engine_power = shaft_power / efficiency
        elif engine.type == "Hybrid":
            electric_engine_power = (1-pwr_mng.hybrid_factor_engine) * shaft_power / efficiency
        else:
            ValueError(f"Engine type {engine.type} not implemented ")

        electric_engine_power = min(
                electric_engine_power, power_rating
            )  # limit the power to the power rating
        
        electric_engine_voltage = voltage_rating * (
            electric_engine_power / power_rating
        )  # calculate the voltage based on the power rating
        engine_temperature = (
            electric_engine_power * 0.01 + 25
        )  # calculate the temperature based on the power
        engine_temperature = min(
            engine_temperature, 100
        )  # limit the temperature to 100 degrees Celsius
        # compute throttle settings
        throttle = electric_engine_power / (options.elec_power_rating)
        # compute heat
        heat_out = throttle * options.elec_power_rating * (1.0 - eta_m)
        # Compute mass increment comare to reference
        electric_engine_weight = weight_inc + options.weight_base

        return (
            electric_engine_power,
            electric_engine_voltage,
            engine_temperature,
            electric_engine_weight,
        )

    # [SM to be connected with the model to feed the jacobian]
    def _get_sensitivity(self) -> float:
        """
        compute the adjoint of the shaft power

        """
        # Calculate the sensitivity of the output with respect to each input
        sensitivity = {}
        derivatives = []

        # compute derivatives - anlytically derived
        d_electric_engine_power_d_shaft_power = 1 / efficiency
        d_electric_engine_power_d_efficiency = -(shaft_power / (efficiency**2))
        d_electric_engine_power_d_fuel_power = -(shaft_power / (total_power**2)) * (
            shaft_power / efficiency
        )
        d_electric_engine_voltage_d_shaft_power = voltage_rating * (
            d_electric_engine_power_d_shaft_power / power_rating
        )
        d_electric_engine_voltage_d_efficiency = voltage_rating * (
            d_electric_engine_power_d_efficiency / power_rating
        )
        d_electric_engine_voltage_d_fuel_power = voltage_rating * (
            d_electric_engine_power_d_fuel_power / power_rating
        )
        d_engine_temperature_d_shaft_power = 0.01
        d_engine_temperature_d_efficiency = 0
        d_engine_temperature_d_fuel_power = 0
        d_electric_engine_power_d_power_rating = 0
        d_electric_engine_voltage_d_power_rating = voltage_rating * (
            (shaft_power / total_power) * (1 / efficiency)
            - electric_engine_power / power_rating
        )
        d_engine_temperature_d_power_rating = 0
        d_electric_engine_power_d_voltage_rating = 0
        d_electric_engine_voltage_d_voltage_rating = (
            electric_engine_power / power_rating
        )
        d_engine_temperature_d_voltage_rating = 0

        sensitivity = {
            "electric_engine_power": {
                "shaft_power": d_electric_engine_power_d_shaft_power,
                "efficiency": d_electric_engine_power_d_efficiency,
                "fuel_power": d_electric_engine_power_d_fuel_power,
                "power_rating": d_electric_engine_power_d_power_rating,
                "voltage_rating": d_electric_engine_power_d_voltage_rating,
            },
            "electric_engine_voltage": {
                "shaft_power": d_electric_engine_voltage_d_shaft_power,
                "efficiency": d_electric_engine_voltage_d_efficiency,
                "fuel_power": d_electric_engine_voltage_d_fuel_power,
                "power_rating": d_electric_engine_voltage_d_power_rating,
                "voltage_rating": d_electric_engine_voltage_d_voltage_rating,
            },
            "engine_temperature": {
                "shaft_power": d_engine_temperature_d_shaft_power,
                "efficiency": d_engine_temperature_d_efficiency,
                "fuel_power": d_engine_temperature_d_fuel_power,
                "power_rating": d_engine_temperature_d_voltage_rating,
            },
        }
        # return values
        return sensitivity

    def power_electric_engine(self, engine_names: list[str]) -> tuple[float, float]:
        if not engine_names:
            engine_names = self.mapind.keys()

        electric_engine_power = 0.0
        for name in engine_names:
            try:
                i = self.mapind[name]
                electric_engine_power += self.electric_engine_power[i]
            except KeyError:
                pass

        return electric_engine_power

    def mass_electric_engine(self, engine_names: list[str]) -> tuple[float, float]:
        if not engine_names:
            engine_names = self.mapind.keys()

        electric_engine_mass = 0.0
        for name in engine_names:
            try:
                i = self.mapind[name]
                electric_engine_mass += self.electric_engine_mass[i]
            except KeyError:
                pass

        return electric_engine_mass


if __name__ == "__main__":

    import sys
    import os

    sys.path.insert(
        0,
        os.path.abspath(os.path.join(os.path.dirname(__file__), "../../src")),
    )
    print(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../src")))

    EE = Driver()

    engines = Base_PropulsionSystem
    engines.shaft_power = 4000.0
    engines.name = "Eng"
    engines.type = "Hybrid"
    engines = [engines]
    engines = [PropulsionSystem.from_component(w) for w in engines]

    options = Options()
	
    pw_mng = PowerManagSyst()
    pw_mng.hybrid_factor_engine = 0.0

    (
        electric_engine_power,
        electric_engine_voltage,
        engine_temperature,
        electric_engine_mass,
    ) = EE._run(engines, pw_mng, options)
    print(electric_engine_power, electric_engine_mass)
