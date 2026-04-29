from __future__ import annotations

from typing import TYPE_CHECKING

from multiads.assembly import (
    Environment,
    ThermalSystem,
    Wing,
    copy_components,
    flatten_components,
)
from multiads.scenario import InnerVariableFloat
from multiads.solvers import BaseSolver
from multiads.solvers.thermal.ips_low_fidelity_lib import Driver, Options

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from numpy.typing import NDArray

    from multiads.assembly import MADSComponent
    from multiads.scenario import BaseVariable


class IPSLowFidelity(BaseSolver):
    """Model of the Ice Protection System.

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
        self.ips: Sequence[ThermalSystem] | None = None
        self.outputs_map: dict[str, InnerVariableFloat] | None = None

    def parse_variables(
        self,
        components: Sequence[MADSComponent],
    ) -> Sequence[MADSComponent]:
        # Filter components
        _components = copy_components(components)
        comp_flat = flatten_components(_components)
        self.environment = next(c for c in comp_flat if isinstance(c, Environment))
        self.wings = [c for c in comp_flat if isinstance(c, Wing)]
        self.ips = [c for c in comp_flat if isinstance(c, ThermalSystem)]

        # Extract variables
        self.inputs: list[BaseVariable] = []
        if v := self.environment.variables.get("height"):
            self.inputs.append(v)
        if v := self.environment.variables.get("speed"):
            self.inputs.append(v)

        for wing in self.wings:
            for sec in wing.sections:
                # Design variables
                if v := sec.variables.get("chord"):
                    self.inputs.append(v)
            for span in wing.spans:
                if l := span.variables.get("length"):
                    self.inputs.append(l)

        self.outputs = [
            InnerVariableFloat(f"{self.ips.name}.mass", 0.0),
            InnerVariableFloat(f"{self.ips.name}.power", 0.0),
            InnerVariableFloat(f"{self.ips.name}.volume", 0.0),
        ]

        self.outputs_map = {v.name: v for v in self.outputs}

        return [self.environment, *self.ips]

    def _run(self) -> None:
        if self.environment is None or self.ecs is None:
            msg = f"Components in solver '{type(self).__name__}' not initialized."
            raise RuntimeError(msg)

        self.driver = Driver(
            environment=self.environment,
            ips=self.ips,
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

        self.outputs_map[f"{self.ips.name}.mass"].value = self.driver.total_mass()
        self.outputs_map[f"{self.ips.name}.power"].value = self.driver.cumulative_power()
        self.outputs_map[f"{self.ips.name}.volume"].value = self.driver.cumulative_volume()
        

    def compute_sensitivities(
        self,
        input_names: Sequence[str],
        inputs: Sequence[BaseVariable],
        output_names: Sequence[str],
        outputs: Sequence[BaseVariable],
    ) -> Mapping[str, NDArray]:
        return {}
