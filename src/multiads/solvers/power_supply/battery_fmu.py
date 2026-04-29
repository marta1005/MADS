from __future__ import annotations

from typing import TYPE_CHECKING

from multiads.assembly import (
    Battery,
    MADSComponent,
    copy_components,
    flatten_components,
)
from multiads.scenario import BaseVariable, InnerVariableFloat
from multiads.solvers import BaseSolver
from multiads.solvers.power_supply.battery_fmu_lib import Driver, Options

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from numpy.typing import NDArray


class BatteryFMU(BaseSolver):
    def __init__(self, options: Options | None = None) -> None:
        super().__init__()
        self.options: Options = options or Options()
        self.driver: Driver | None = None
        self.batteries: Sequence[Battery] | None = None
        self.outputs_map: dict[str, InnerVariableFloat] | None = None

    def parse_variables(
        self,
        components: Sequence[MADSComponent],
    ) -> Sequence[MADSComponent]:
        # Filter batteries
        _components = copy_components(components)
        c_flat = flatten_components(_components)
        self.batteries = [c for c in c_flat if isinstance(c, Battery)]

        # Extract variables
        self.inputs: list[BaseVariable] = []
        for batt in self.batteries:
            if v := batt.variables.get("power"):
                self.inputs.append(v)
            if v := batt.variables.get("nominal_voltage"):
                self.inputs.append(v)
            if v := batt.variables.get("flight_time"):
                self.inputs.append(v)

        self.outputs: list[InnerVariableFloat] = []
        for batt in self.batteries:
            self.outputs.extend(
                [
                    InnerVariableFloat(f"{batt.name}.mass", 0.0),
                    InnerVariableFloat(f"{batt.name}.volume", 0.0),
                    InnerVariableFloat(f"{batt.name}.voltage", 0.0),
                    InnerVariableFloat(f"{batt.name}.capacity", 0.0),
                ],
            )

        self.outputs_map = {v.name: v for v in self.outputs}

        return self.batteries

    def _run(self) -> None:
        if self.batteries is None:
            msg = f"Components in solver '{type(self).__name__}' not initialized."
            raise RuntimeError(msg)

        self.driver = Driver(
            batteries=self.batteries,
            options=self.options,
        )
        self.driver.run()

    def compute_output(self) -> None:
        if self.driver is None:
            msg = f"The driver of solver '{type(self).__name__}' is not initialized."
            raise RuntimeError(msg)

        if self.outputs is None or self.outputs_map is None:
            msg = f"The outputs of solver '{type(self).__name__}' are not initialized"
            raise RuntimeError(msg)

        masses = self.driver.weights
        volumes = self.driver.volumes
        voltages = self.driver.voltages
        capacities = self.driver.capacities

        if self.batteries is None:
            msg = f"Components in solver '{type(self).__name__}' not initialized."
            raise RuntimeError(msg)

        for i, batt in enumerate(self.batteries):
            self.outputs_map[f"{batt.name}.mass"].value = masses[i]
            self.outputs_map[f"{batt.name}.volume"].value = volumes[i]
            self.outputs_map[f"{batt.name}.voltage"].value = voltages[i]
            self.outputs_map[f"{batt.name}.capacity"].value = capacities[i]

    def compute_sensitivities(
        self,
        input_names: Sequence[str],
        inputs: Sequence[BaseVariable],
        output_names: Sequence[str],
        outputs: Sequence[BaseVariable],
    ) -> Mapping[str, NDArray]:
        return {}
