"""Interface to the model of the Environmental Control System.

# [To be update]
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

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from typing_extensions import Self

from multiads.solvers import SolverOptions
from multiads.solvers.thermal.models import ecs_block_model

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence
    from typing import Any

    from multiads.assembly import Environment
    from multiads.assembly import ThermalSystem as Base_ThermalSystem


class Options(SolverOptions):
    def __init__(
        self,
        *,
        fuel_tank_volume: float = 37820,  # Fuel Tank Volume [l]
        crew: int = 5,  # Number of crew members
        passengers: int = 120,  # Number of passengers
        operational_engines: int = 2,  # Number of operational engines
        n_engines: int = 2,  # Number of total engines
        temp_out: float = 55.0,  #  Outpflow temperature [ºC]
        temp_in: float = 26.0,  # Inflow temperature [ºC]
        cp: int = 1006,  # Thermal capaicty at constant pressure
        **kwargs: Any,  # noqa: ANN401
    ) -> None:
        super().__init__(**kwargs)
        self.fuel_tank_volume = fuel_tank_volume
        self.crew = crew
        self.passengers = passengers
        self.operational_engines = operational_engines
        self.n_engines = n_engines
        self.temp_out = temp_out
        self.temp_in = temp_in
        self.cp = cp


class ThermalSystem:
    def __init__(
        self,
        name: str,
        fuel_tank_volume: float,  # Fuel Tank Volume [l]
        q_total: float,  # Equipment Heat dissipation [W]
    ) -> None:
        self.name = name
        self.fuel_tank_volume = fuel_tank_volume
        self.q_total = q_total

    @classmethod
    def from_component(cls, comp: Base_ThermalSystem) -> Self:
        return cls(
            name=comp.name,
            fuel_tank_volume=comp.fuel_tank_volume,
            q_total=comp.q_total,
        )


def compute_environment_control(
    height: float,
    speed: float,
    fuel_tank_volume: float,
    crew: int,
    passengers: int,
    operational_engines: int,
    n_engines: int,
    q_total: float,
    temp_out: float,
    temp_in: float,
    cp: float,
) -> tuple[float, float, float, float, float]:
    # Definition of Ambient static conditions
    pressure, temperature, density = ecs_block_model.ISA_calc(height)

    #  Human Ventilation data
    human_ventilation_kgs = ecs_block_model.humanVent(
        crew,
        passengers,
        operational_engines,
        n_engines,
        margin=10,
    )

    #  Equipment ventilation data
    equipment_ventilation_kgs = ecs_block_model.equipVent(
        q_total,
        temp_out,
        temp_in,
        cp,
    )

    # Bleed needed by ECS
    obogs_kgs = ecs_block_model.OBOGS_bleed(
        pressure,
        temperature,
        crew,
        height,
    )
    obiggs_kgs = ecs_block_model.OBIGGS_bleed(fuel_tank_volume, margin=10)

    ecs_bleed = (
        obogs_kgs + obiggs_kgs + equipment_ventilation_kgs + human_ventilation_kgs
    )
    ram_drag = ecs_block_model.inlet_drag(speed, density, ecs_bleed)
    ecs_w, ecs_vol = ecs_block_model.Weight_Volume()
    ecs_power = ecs_block_model.Elect_ECS_Power()

    return ecs_bleed, ram_drag, ecs_w, ecs_vol, ecs_power


class Driver:
    def __init__(
        self,
        environment: Environment,
        ecs: Sequence[Base_ThermalSystem],
        options: Options,
    ) -> None:
        self.options = options
        self.height = environment.height
        self.speed = environment.speed

        # Instatiate the engine from the base class
        self.ecs = [ThermalSystem.from_component(e) for e in ecs]

        # Initialise main engine QOI
        self.ecs_mass: dict[str, float] = {}
        self.ecs_ram_drag: dict[str, float] = {}
        self.ecs_power: dict[str, float] = {}
        self.ecs_volume: dict[str, float] = {}

    def run(self) -> None:
        for ecs in self.ecs:
            _, ecs_ram_drag, ecs_weight, ecs_vol, ecs_power = (
                compute_environment_control(
                    height=self.height,
                    speed=self.speed,
                    fuel_tank_volume=ecs.fuel_tank_volume,
                    crew=self.options.crew,
                    passengers=self.options.passengers,
                    operational_engines=self.options.operational_engines,
                    n_engines=self.options.n_engines,
                    q_total=ecs.q_total,
                    temp_out=self.options.temp_out,
                    temp_in=self.options.temp_in,
                    cp=self.options.cp,
                )
            )
            name = ecs.name
            self.ecs_mass[name] = ecs_weight
            self.ecs_ram_drag[name] = ecs_ram_drag
            self.ecs_power[name] = ecs_power
            self.ecs_volume[name] = ecs_vol

    def total_mass(self, ecs_names: Iterable[str] | None = None) -> np.float64:
        if not ecs_names:
            ecs_names = [e.name for e in self.ecs]
        return np.sum([self.ecs_mass.get(name, 0.0) for name in ecs_names])

    def cumulative_ram_drag(self, ecs_names: Iterable[str] | None = None) -> np.float64:
        if not ecs_names:
            ecs_names = [e.name for e in self.ecs]
        return np.sum([self.ecs_ram_drag.get(name, 0.0) for name in ecs_names])

    def cumulative_power(self, ecs_names: Iterable[str] | None = None) -> np.float64:
        if not ecs_names:
            ecs_names = [e.name for e in self.ecs]
        return np.sum([self.ecs_power.get(name, 0.0) for name in ecs_names])

    def cumulative_volume(self, ecs_names: Iterable[str] | None = None) -> np.float64:
        if not ecs_names:
            ecs_names = [e.name for e in self.ecs]
        return np.sum([self.ecs_volume.get(name, 0.0) for name in ecs_names])


if __name__ == "__main__":
    ecs_bleed, ram_drag, ecs_w, ecs_vol, ecs_power = compute_environment_control(
        height=6705,  # Altitude
        speed=250.8508,  # TAS
        fuel_tank_volume=37820,
        crew=5,
        passengers=120,
        operational_engines=1,
        n_engines=2,
        q_total=15000,
        temp_out=55,
        temp_in=26,
        cp=1006,
    )
    print(ecs_bleed, ram_drag, ecs_w, ecs_vol, ecs_power)  # noqa: T201
