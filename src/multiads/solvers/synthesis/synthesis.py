from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import NDArray

from multiads.assembly import (
    Aircraft,
    Environment,
    MADSComponent,
    Wing,
    copy_components,
    flatten_components,
)
from multiads.scenario import BaseVariable, InnerVariable, InnerVariableFloat
from multiads.scenario.aero_derivatives import AeroDerivativeReferenceFrame, AeroDerivativesVariable
from multiads.scenario.mass_properties import MassPropertiesVariable
from multiads.solvers import BaseSolver
from multiads.solvers.synthesis.synthesis_lib import (
    Aircraft as DSAircraft,
    Driver,
    Options,
    Wing as WSWing,
)

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence


def body_AeroDerivatives_alpha(name: str, driver: Driver, **kwargs: Any) -> NDArray[np.float64]:
    result = driver.body_AeroDerivatives_alpha()
    return np.asarray(result)


def body_AeroDerivatives_beta(name: str, driver: Driver, **kwargs: Any) -> NDArray[np.float64]:
    result = driver.body_AeroDerivatives_beta()
    return np.asarray(result)


def body_AeroDerivatives_p(name: str, driver: Driver, **kwargs: Any) -> NDArray[np.float64]:
    result = driver.body_AeroDerivatives_p()
    return np.asarray(result)


def body_AeroDerivatives_q(name: str, driver: Driver, **kwargs: Any) -> NDArray[np.float64]:
    result = driver.body_AeroDerivatives_q()
    return np.asarray(result)


def body_AeroDerivatives_r(name: str, driver: Driver, **kwargs: Any) -> NDArray[np.float64]:
    result = driver.body_AeroDerivatives_r()
    return np.asarray(result)


def body_AeroDerivatives_alpha_dot(name: str, driver: Driver, **kwargs: Any) -> NDArray[np.float64]:
    result = driver.body_AeroDerivatives_alpha_dot()
    return np.asarray(result)


def body_AeroDerivatives_beta_dot(name: str, driver: Driver, **kwargs: Any) -> NDArray[np.float64]:
    result = driver.body_AeroDerivatives_beta_dot()
    return np.asarray(result)


def body_AeroDerivatives_pitching(name: str, driver: Driver, **kwargs: Any) -> NDArray[np.float64]:
    result = driver.body_AeroDerivatives_pitching()
    return np.asarray(result)


def body_AeroDerivatives_yawing(name: str, driver: Driver, **kwargs: Any) -> NDArray[np.float64]:
    result = driver.body_AeroDerivatives_yawing()
    return np.asarray(result)


def body_AeroDerivatives_delta_aileron(name: str, driver: Driver, **kwargs: Any) -> NDArray[np.float64]:
    result = driver.body_AeroDerivatives_delta_aileron()
    return np.asarray(result)


def body_AeroDerivatives_delta_elevator(name: str, driver: Driver, **kwargs: Any) -> NDArray[np.float64]:
    result = driver.body_AeroDerivatives_delta_elevator()
    return np.asarray(result)


def body_AeroDerivatives_delta_rudder(name: str, driver: Driver, **kwargs: Any) -> NDArray[np.float64]:
    result = driver.body_AeroDerivatives_delta_rudder()
    return np.asarray(result)


def thetas_new(name: str, driver: Driver, **kwargs: Any) -> NDArray[np.float64]:
    result = driver.thetas_deformed()
    return np.asarray(result)


def dihedrals_new(name: str, driver: Driver, **kwargs: Any) -> NDArray[np.float64]:
    result = driver.dihedrals_deformed()
    return np.asarray(result)


def spans_new(name: str, driver: Driver, **kwargs: Any) -> NDArray[np.float64]:
    result = driver.spans_deformed()
    return np.asarray(result)


IMPLEMENTED_OUTPUTS = {
    "body_AeroDerivatives_alpha": body_AeroDerivatives_alpha,
    "body_AeroDerivatives_beta": body_AeroDerivatives_beta,
    "body_AeroDerivatives_p": body_AeroDerivatives_p,
    "body_AeroDerivatives_q": body_AeroDerivatives_q,
    "body_AeroDerivatives_r": body_AeroDerivatives_r,
    "body_AeroDerivatives_alpha_dot": body_AeroDerivatives_alpha_dot,
    "body_AeroDerivatives_beta_dot": body_AeroDerivatives_beta_dot,
    "body_AeroDerivatives_pitching": body_AeroDerivatives_pitching,
    "body_AeroDerivatives_yawing": body_AeroDerivatives_yawing,
    "body_AeroDerivatives_delta_aileron": body_AeroDerivatives_delta_aileron,
    "body_AeroDerivatives_delta_elevator": body_AeroDerivatives_delta_elevator,
    "body_AeroDerivatives_delta_rudder": body_AeroDerivatives_delta_rudder,
    "thetas_new": thetas_new,
    "spans_new": spans_new,
    "dihedrals_new": dihedrals_new,
}


class DesignSynthesis(BaseSolver):
    def __init__(self, options: Options | None = None) -> None:
        super().__init__()
        self.options: Options = options or Options()
        self.driver: Driver | None = None
        self.aircraft: Aircraft | None = None
        self.wings: list[Wing] | None = None
        self.environment: Environment | None = None
        self.inputs_map: dict[str, BaseVariable] | None = None
        self.outputs_map: dict[str, InnerVariable] | None = None

    def parse_variables(
        self,
        components: Sequence[MADSComponent],
        *argv: Any,  # noqa: ANN401, ARG002
    ) -> Sequence[MADSComponent]:
        _components = copy_components(components)
        components_flat = flatten_components(_components)

        try:
            ac = (c for c in components_flat if isinstance(c, Aircraft))
            self.aircraft = next(ac)
        except StopIteration:
            msg = f"An aircraft must be provided to solver '{type(self).__name__}'."
            raise ValueError(msg) from None

        try:
            env = (c for c in components_flat if isinstance(c, Environment))
            self.environment = next(env)
        except StopIteration:
            msg = f"An environment must be provided to solver '{type(self).__name__}'."
            raise ValueError(msg) from None

        self.wings = [c for c in components_flat if isinstance(c, Wing)]

        inputs: list[BaseVariable] = []

        dvars = ["height", "speed", "alpha", "beta", "gamma"]
        inputs.extend(v for k, v in self.environment.variables.items() if k in dvars)

        if self.aircraft:
            dvars = ["mass", "global_pos"]
            inputs.extend(v for k, v in self.aircraft.variables.items() if k in dvars)

            prefix = self.aircraft.name
            inputs.extend(
                [
                    AeroDerivativesVariable.zeros(f"{prefix}.aero_derivatives"),
                    MassPropertiesVariable.from_num_elements(f"{prefix}.mass_properties", 1),
                ],
            )

        outputs: list[InnerVariable] = []

        if self.aircraft:
            for out_name in IMPLEMENTED_OUTPUTS.keys():
                full_name = f"{self.aircraft.name}.{out_name}"
                outputs.append(InnerVariableFloat(full_name, 0.0))

        self.inputs = inputs
        self.outputs = outputs
        self.inputs_map = {v.name: v for v in self.inputs}
        self.outputs_map = {v.name: v for v in self.outputs}

        return [self.environment, self.aircraft, *self.wings]

    def _run(self) -> None:
        if self.aircraft is None or self.environment is None or self.inputs_map is None:
            msg = f"Components in solver '{type(self).__name__}' not initialized."
            raise RuntimeError(msg)

        prefix = self.aircraft.name
        aero_derivatives = self.inputs_map[f"{prefix}.aero_derivatives"]
        mass_properties = self.inputs_map[f"{prefix}.mass_properties"]

        aircraft_ds = DSAircraft.from_variables(
            name=self.aircraft.name,
            aerodynamicproperties=aero_derivatives,
            massproperties=mass_properties,
        )

        # wing rotation and displacement for mesh deformation
        wing_displ_z = [0.0]
        wing_rot_y = [0.0]

        prefix = f"{self.aircraft.name}"
        if hasattr(self.inputs_map, "wing_displ_z") and hasattr(self.inputs_map, "wing_displ_z"):
            wing_displ_z = self.inputs_map[f"{prefix}.wing_displ_z"]
            wing_rot_y = self.inputs_map[f"{prefix}.wing_rot_y"]

        # recover wing from component
        wings_list = [
            WSWing.from_component(w, wing_displ_z=wing_displ_z, wing_rot_y=wing_rot_y)
            for w in self.wings
        ] if self.wings else []

        self.driver = Driver(
            aircraft=[aircraft_ds],
            wings=wings_list,
            environment=self.environment,
            options=self.options,
        )

        self.driver.compute_aircraft_properties()
        self.driver.find_nodes()
        self.driver.find_deform()

        if aero_derivatives is not None and isinstance(aero_derivatives, AeroDerivativesVariable):
            aero_derivatives.reference_frame = AeroDerivativeReferenceFrame.BODY

    def compute_output(self) -> None:
        if self.driver is None:
            msg = f"The driver of solver '{type(self).__name__}' is not initialized."
            raise RuntimeError(msg)

        if self.outputs is None or self.outputs_map is None:
            msg = f"The outputs of solver '{type(self).__name__}' are not initialized"
            raise RuntimeError(msg)

        if self.aircraft is None:
            return

        for out in self.outputs:
            out_name = out.name
            if "." in out_name:
                base_name = out_name.split(".", 1)[1]
            else:
                base_name = out_name

            if base_name in IMPLEMENTED_OUTPUTS:
                out_func = IMPLEMENTED_OUTPUTS[base_name]
                out.value = out_func(out_name, self.driver)

    def compute_sensitivities(
        self,
        input_names: Sequence[str],  # noqa: ARG002
        inputs: Sequence[BaseVariable],  # noqa: ARG002
        output_names: Sequence[str],  # noqa: ARG002
        outputs: Sequence[BaseVariable],  # noqa: ARG002
    ) -> Mapping[str, NDArray]:
        return {}
