from __future__ import annotations

from typing import TYPE_CHECKING

from multiads.assembly import (
    Environment,
    ThermalSystem,
    copy_components,
    flatten_components,
)
from multiads.scenario import InnerVariableFloat
from multiads.solvers import BaseSolver
from multiads.solvers.thermal.ecs_low_fidelity_lib import Driver, Options

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from numpy.typing import NDArray

    from multiads.assembly import MADSComponent
    from multiads.scenario import BaseVariable


class ECSLowFidelity(BaseSolver):
    """Model of the Environmental Control System.

    Raises:
        ValueError: _description_

    Returns:
        _type_: _description_

    """

    def __init__(self, options: Options | None = None) -> None:
        super().__init__()
        self.options: Options = options or Options()
        self.driver: Driver | None = None
        self.environment: Environment | None = None
        self.ecs: Sequence[ThermalSystem] | None = None
        self.outputs_map: dict[str, InnerVariableFloat] | None = None

    def parse_variables(
        self,
        components: Sequence[MADSComponent],
    ) -> Sequence[MADSComponent]:
        # Filter components
        _components = copy_components(components)
        comp_flat = flatten_components(_components)
        self.environment = next(c for c in comp_flat if isinstance(c, Environment))
        self.ecs = [c for c in comp_flat if isinstance(c, ThermalSystem)]

        # Extract variables
        self.inputs: list[BaseVariable] = []
        if v := self.environment.variables.get("height"):
            self.inputs.append(v)
        if v := self.environment.variables.get("speed"):
            self.inputs.append(v)

        for ecs in self.ecs:
            if v := ecs.variables.get("fuel_tank_volume"):
                self.inputs.append(v)
            if v := ecs.variables.get("q_total"):
                self.inputs.append(v)

        self.outputs = [
            InnerVariableFloat("ecs_mass", 0.0),
            InnerVariableFloat("ram_drag", 0.0),
            InnerVariableFloat("ecs_power", 0.0),
            InnerVariableFloat("ecs_volume", 0.0),
        ]

        self.outputs_map = {v.name: v for v in self.outputs}

        return [self.environment, *self.ecs]

    def _run(self) -> None:
        if self.environment is None or self.ecs is None:
            msg = f"Components in solver '{type(self).__name__}' not initialized."
            raise RuntimeError(msg)

        self.driver = Driver(
            environment=self.environment,
            ecs=self.ecs,
            options=self.options,
        )
        self.driver.run()

    def compute_output(self) -> None:
        if self.driver is None:
            msg = f"The driver of solver '{type(self).__name__}' is not initialized."
            raise RuntimeError(msg)

        if self.outputs is None or self.outputs_map is None:
            msg = f"The outputs of solver '{type(self).__name__} are not initialized"
            raise RuntimeError(msg)

        self.outputs_map["ecs_mass"].value = self.driver.total_mass()
        self.outputs_map["ram_drag"].value = self.driver.cumulative_ram_drag()
        self.outputs_map["ecs_power"].value = self.driver.cumulative_power()
        self.outputs_map["ecs_volume"].value = self.driver.cumulative_volume()

    def compute_sensitivities(
        self,
        input_names: Sequence[str],
        inputs: Sequence[BaseVariable],
        output_names: Sequence[str],
        outputs: Sequence[BaseVariable],
    ) -> Mapping[str, NDArray]:
        return {}
