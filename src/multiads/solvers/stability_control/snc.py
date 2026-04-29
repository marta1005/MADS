from __future__ import annotations

from typing import TYPE_CHECKING, Any

from numpy.typing import NDArray

from multiads.assembly import (
    Aircraft,
    Environment,
    MADSComponent,
    copy_components,
    flatten_components,
)
from multiads.scenario import BaseVariable, InnerVariable, InnerVariableFloat
from multiads.scenario.aero_derivatives import AeroDerivativesVariable
from multiads.scenario.mass_properties import MassPropertiesVariable
from multiads.scenario.stability_control_properties import StabilityControlVariable
from multiads.solvers import BaseSolver
from multiads.solvers.stability_control.snc_lib import Driver, Options

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from numpy.typing import NDArray


class LinearStabilityControl(BaseSolver):
    def __init__(self, options: Options | None = None) -> None:
        super().__init__()
        self.options: Options = options or Options()
        self.driver: Driver | None = None
        self.aircraft: Aircraft | None = None
        self.environment: Environment | None = None
        self.inputs_map: dict[str, BaseVariable] | None = None
        self.outputs_map: dict[str, InnerVariable] | None = None

    def parse_variables(
        self,
        components: Sequence[MADSComponent],
        *argv: Any,  # noqa: ANN401, ARG002
    ) -> Sequence[MADSComponent]:
        # Filter components
        _components = copy_components(components)
        components_flat = flatten_components(_components)
        ## select aircraft
        try:
            ac = (c for c in components_flat if isinstance(c, Aircraft))
            self.aircraft = next(ac)
        except StopIteration:
            msg = f"An aircraft must be provided to solver '{type(self).__name__}'."
            raise ValueError(msg) from None

        ## environment
        try:
            env = (c for c in components_flat if isinstance(c, Environment))
            self.environment = next(env)
        except StopIteration:
            msg = f"An environment must be provided to solver '{type(self).__name__}'."
            raise ValueError(msg) from None

        # Extract variables
        inputs: list[BaseVariable] = []

        dvars = ["height", "speed", "alpha", "beta", "gamma"]
        inputs.extend(v for k, v in self.environment.variables.items() if k in dvars)

        dvars = ["mass", "global_pos"]
        inputs.extend(v for k, v in self.aircraft.variables.items() if k in dvars)

        # Create inner variables
        inputs.extend(
            [
                AeroDerivativesVariable.empty(
                    f"{self.aircraft.name}.aero_derivatives",
                ),
                MassPropertiesVariable.from_num_elements(
                    f"{self.aircraft.name}.mass_properties",
                    1,
                ),
                StabilityControlVariable.empty(
                    f"{self.aircraft.name}.stability_control_properties",
                ),
            ],
        )

        outputs = [
            InnerVariableFloat(
                f"{self.aircraft.name}.static_margin",
                0.0,
            ),
            InnerVariableFloat(
                f"{self.aircraft.name}.omega_short_period",
                0.0,
            ),
            InnerVariableFloat(
                f"{self.aircraft.name}.eta_short_period",
                0.0,
            ),
            InnerVariableFloat(
                f"{self.aircraft.name}.omega_phugoid",
                0.0,
            ),
            InnerVariableFloat(
                f"{self.aircraft.name}.eta_phugoid",
                0.0,
            ),
            InnerVariableFloat(
                f"{self.aircraft.name}.time_roll",
                0.0,
            ),
            InnerVariableFloat(
                f"{self.aircraft.name}.time_pitch",
                0.0,
            ),
            InnerVariableFloat(
                f"{self.aircraft.name}.time_yaw",
                0.0,
            ),
            InnerVariableFloat(  # spiral mode charateristic
                f"{self.aircraft.name}.roll_subs_time",
                0.0,
            ),
            InnerVariableFloat(  # spiral subsidance charateristic
                f"{self.aircraft.name}.spiral_subs_time",
                0.0,
            ),
            InnerVariableFloat(  # spiral time charateristic
                f"{self.aircraft.name}.spiral_char_time",
                0.0,
            ),
        ]

        # Create a map of inputs and outputs
        self.inputs = inputs
        self.outputs = outputs
        self.inputs_map = {v.name: v for v in self.inputs}
        self.outputs_map = {v.name: v for v in self.outputs}

        return [self.aircraft]

    def _run(self) -> None:
        if self.aircraft is None or self.environment is None or self.inputs_map is None:
            msg = f"Components in solver '{type(self).__name__}' not initialized."
            raise RuntimeError(msg)

        prefix = f"{self.aircraft.name}"
        mass_properties = self.inputs_map[f"{prefix}.mass_properties"]
        aero_derivatives = self.inputs_map[f"{prefix}.aero_derivatives"]
        stability_properties = self.inputs_map[f"{prefix}.stability_control_properties"]

        # Run S&C driver
        self.driver = Driver(
            environment=self.environment,
            massproperties=mass_properties,  # type: ignore[invalid-argument-type]
            aeroderivatives=aero_derivatives,  # type: ignore[invalid-argument-type]
            stability_controlproperties=stability_properties,  # type: ignore[invalid-argument-type]
            options=self.options,
        )
        self.driver.run()

    def compute_output(self) -> None:
        if self.aircraft is None:
            msg = f"Components in solver '{type(self).__name__}' not initialized."
            raise RuntimeError(msg)

        if self.driver is None:
            msg = f"The driver of solver '{type(self).__name__}' is not initialized."
            raise RuntimeError(msg)

        if self.outputs is None or self.outputs_map is None:
            msg = f"The outputs of solver '{type(self).__name__} are not initialized"
            raise RuntimeError(msg)

        prefix = self.aircraft.name

        self.outputs_map[f"{prefix}.static_margin"].value = self.driver.static_margin
        self.outputs_map[f"{prefix}.omega_short_period"].value = self.driver.omega_s
        self.outputs_map[f"{prefix}.eta_short_period"].value = self.driver.xi_s
        self.outputs_map[f"{prefix}.omega_phugoid"].value = self.driver.omega_p
        self.outputs_map[f"{prefix}.eta_phugoid"].value = self.driver.xi_p
        self.outputs_map[f"{prefix}.time_roll"].value = self.driver.Tr
        self.outputs_map[f"{prefix}.time_pitch"].value = self.driver.Ts
        self.outputs_map[f"{prefix}.time_yaw"].value = self.driver.T2
        self.outputs_map[f"{prefix}.roll_subs_time"].value = self.driver.t_roll
        self.outputs_map[f"{prefix}.spiral_subs_time"].value = self.driver.t_pitch
        self.outputs_map[f"{prefix}.spiral_char_time"].value = self.driver.t_yaw

    def compute_sensitivities(
        self,
        input_names: Sequence[str],  # noqa: ARG002
        inputs: Sequence[BaseVariable],  # noqa: ARG002
        output_names: Sequence[str],  # noqa: ARG002
        outputs: Sequence[BaseVariable],  # noqa: ARG002
    ) -> Mapping[str, NDArray]:
        return {}
