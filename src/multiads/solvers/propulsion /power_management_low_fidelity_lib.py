# -*- coding: utf-8 -*-
"""

# Inputs
#   : ....

    
# parameters (within the model that can be moved to inputs)
    : ...      # ....
    : ...      # ...
    : ....     # ...

# Outputs
#   : ...
#   : ...
#   : ...

"""

import numpy as np
from matplotlib import pyplot as plt
from scipy.interpolate import UnivariateSpline

## SM6
# -----------------------
# from scipy.integrate import simps
from scipy.integrate import simpson


# MADS integration
import copy
from dataclasses import dataclass, field
from typing_extensions import Self
import numpy as np
from assembly import Environment as Base_Environment
from assembly import PropulsionSystem as Base_PropulsionSystem
from assembly import PowerManagSyst as Base_PowerManagSyst
from assembly import (
    ThermalSystem as Base_ThermalSystem,
)


@dataclass
class Options:
    number_of_engines: int = 3  # number of engines
    efficiency_inverter: float = 0.98  # efficiency  inverter
    efficiency_gear_box: float = 0.98  # efficiency  gear box
    inverter_power = float = 800  # [W]
    jac_approx_type: str = "finite_differences"
    jac_approx_step: float = 1e-04
    jac_approx_n_processes: int = 1
    jac_approx_use_threading: bool = False


@dataclass
class PropulsionSystem:
    engine_power: float  # [W] shaft power in Watt
    type: str  # engine type (see assembly) ['hydrogen','battery','fuelcell','kerosene-hydrogen','kerosene-battery','kerosene-fuelcell','hydrogen-battery','hydrogen-fuelcell','fuelcell-battery']

    @classmethod
    def from_component(cls, comp: Base_PropulsionSystem) -> Self:
        engine = PropulsionSystem(engine_power=comp.engine_power, type=comp.type)
        if "options" in vars(comp):
            for k, v in comp.options.get("power_mng_lf", {}).items():
                if k in vars(engine):
                    setattr(engine, k, v)
        return engine


@dataclass
class ThermalSystem:
    ecs_power: float  # power consumde by the ecs

    @classmethod
    def from_component(cls, comp: Base_ThermalSystem) -> Self:
        ecs = ThermalSystem(
            ecs_power=comp.ecs_power,  # poer consumption [W] Range [ 0 - inf]
        )
        if "options" in vars(comp):
            for k, v in comp.options.get("power_mng_lf", {}).items():
                if k in vars(ecs):
                    setattr(ecs, k, v)
        return ecs


@dataclass
class PowerManagSyst:
    hybrid_factor_engine: (
        float  # combustion to electric hybridization - set to 0 if pure electric
    )
    hybrid_factor_electric_power: (
        float  # battery/Fuel Cell hybridzytion - set to 1 if only battery
    )

    @classmethod
    def from_component(cls, comp: Base_PowerManagSyst) -> Self:
        pwr_mng = PowerManagSyst(
            hybrid_factor_engine=comp.hybrid_factor_engine,  # combustion to electric hybridization - set to 0 if pure electric
            hybrid_factor_electric_power=comp.hybrid_factor_electric_power,    # battery/Fuel Cell hybridzytion - set to 1 if only battery
        )
        if "options" in vars(comp):
            for k, v in comp.options.get("power_mng_lf", {}).items():
                if k in vars(pwr_mng):
                    setattr(pwr_mng, k, v)
        return pwr_mng


class Driver:
    def __init__(
        self,
        environment: Base_Environment,
        engines: list[Base_PropulsionSystem],
        pwr_mng: list[Base_PowerManagSyst],
        ecs: list[Base_ThermalSystem],
        options: Options,
    ) -> None:
        self.options = options
        self.density = environment.density
        self.soundSpeed = environment.sound_speed
        self.viscosity = environment.dyn_viscosity
        self.velocity = environment.velocity
        self.mapind = {e.name: i for i, e in enumerate(engines)}
        self.mapind2 = {cs.name: i for i, cs in enumerate(ecs)}

        # instatiate the engine from the base class
        ## Engine
        self.engines = [PropulsionSystem.from_component(e) for e in engines]
        ## ECS
        self.ecs = [ThermalSystem.from_component(e) for e in ecs]
        ## power Management
        self.pwr_mng = PowerManagSyst.from_component(pwr_mng)

        # initialise QOI
        self.hybrid_factor_engine = [0.0 for _ in engines]
        self.hybrid_factor_power = 0.0
        self.electric_engine_power = [0.0 for _ in engines]
        self.comb_engine_power = [0.0 for _ in engines]
        self.inverter_power = [0.0 for _ in engines]

    def run(self) -> None:
        # loop over the engines
        for i, engine in enumerate(self.engines):
            (
                el_engine_power,
                inv_power,
                comb_eng_power,
                hybrid_factor_eng,
            ) = self._run(engine, self.pwr_mng, self.ecs, self.options)

            # introduce hybridization factor to compute power from power source
            if engine.type == "Electric":
                self.hybrid_factor_engine[i] = 0
            else:
                self.hybrid_factor_engine[i] = hybrid_factor_eng
            # collect power needs
            self.electric_engine_power[i] = el_engine_power
            self.comb_engine_power[i] = comb_eng_power
            self.inverter_power[i] = inv_power

    @staticmethod
    def _run(engine, pwr_mng, ecs, options) -> tuple[float, float, float]:
        """_summary_

        Args:
            engine (_type_): _description_
            options (_type_): _description_

        Parameters:
            efficiency_gear_box

        Returns:
            tuple[float, float, float]: _description_
        """
        # efficiency gear box accounting in engine
        efficiency_gear_box = options.efficiency_gear_box
        # efficieny of inverter
        efficiency_inverter = options.efficiency_inverter

        if engine.type == "Electric":
            # get engine power # ger_box efficiency in engine model
            engine_power = engine.engine_power
            electric_engine_power = engine.engine_power
            comb_engine_power = 0.0
        else:
            engine_power = engine.engine_power
            # in case hybridize include it
            electric_engine_power = engine_power * (1 - pwr_mng.hybrid_factor_engine)
            # combustion engine power
            comb_engine_power = engine_power - electric_engine_power

            # debug
            #print(
            #    "power_mangement",
            #    electric_engine_power,
            #    comb_engine_power,
            #    pwr_mng.hybrid_factor_engine,
            #)

        # power at inverter
        inverter_power = options.inverter_power / efficiency_inverter

        # debug
        # print("engine power", engine_power)
        # print("electric engine power", electric_engine_power)
        # print("combustion engine power", comb_engine_power)
        # print("inverter power", inverter_power)

        return (
            electric_engine_power,
            inverter_power,
            comb_engine_power,
            pwr_mng.hybrid_factor_engine,
        )

    def total_electrical_power(self, engine_names=None, ecs_names=None) -> tuple[float]:
        if not engine_names:
            engine_names = self.mapind.keys()

        self.total_electrical_power = 0.0
        for name in engine_names:
            try:
                i = self.mapind[name]
                self.total_electrical_power += (
                    self.electric_engine_power[i] + self.inverter_power[i]
                )
                # debug
                # print(
                #    self.electric_engine_power[i],
                #    self.inverter_power[i],
                #    self.total_electrical_power,
                # )
                # print(name)

            except KeyError:
                pass

        if not ecs_names:
            ecs_names = self.mapind2.keys()

        for name in ecs_names:
            try:
                i = self.mapind2[name]
                self.total_electrical_power += self.ecs[i].ecs_power

            except KeyError:
                pass

        return self.total_electrical_power

    def total_combustion_power(self, engine_names=None) -> tuple[float]:
        if not engine_names:
            engine_names = self.mapind.keys()

        self.total_combustion_power = 0.0
        for name in engine_names:
            try:
                i = self.mapind[name]
                self.total_combustion_power += self.comb_engine_power[i]
            except KeyError:
                pass

        return self.total_combustion_power

    def electrical_power_from_battery(self) -> tuple[float]:
        self.electrical_power_from_battery = 0.0
        hybrid_factor_power = self._compute_hybrid_global_factor()

        # print(self.total_electrical_power, self.hybrid_factor_power)
        self.electrical_power_from_battery = (
            self.total_electrical_power * hybrid_factor_power
        )*self.pwr_mng.hybrid_factor_electric_power
        return self.electrical_power_from_battery

    def electrical_power_from_fc(self) -> tuple[float]:
        self.electrical_power_to_fc = 0.0
        hybrid_factor_power = self._compute_hybrid_global_factor()

        self.electrical_power_from_fc = (
            self.total_electrical_power * hybrid_factor_power)*(1-self.pwr_mng.hybrid_factor_electric_power)

        return self.electrical_power_from_fc

    # hybridization factor bewten combustion and electric
    def _compute_hybrid_global_factor(self) -> tuple[float]:
        self.hybrid_factor_power = self.total_electrical_power / (
            self.total_electrical_power + self.total_combustion_power
        )
        return self.hybrid_factor_power


if __name__ == "__main__":
    pass
