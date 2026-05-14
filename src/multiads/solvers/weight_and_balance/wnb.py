from __future__ import annotations

from typing import TYPE_CHECKING, Any

from multiads.assembly import (
    Aircraft,
    Fuselage,
    MADSComponent,
    Propeller,
    Wing,
    copy_components,
    flatten_components,
)
from multiads.scenario.mass_properties import MassPropertiesVariable
from multiads.solvers import BaseSolver
from multiads.solvers.weight_and_balance.wnb_lib import Driver, Options

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from numpy.typing import NDArray

    from multiads.scenario import BaseVariable


class WB(BaseSolver):
    def __init__(self, options: Options | None = None) -> None:
        super().__init__()
        self.options: Options = options or Options()
        self.driver: Driver | None = None
        self.wings: Sequence[Wing] | None = None
        self.propellers: Sequence[Propeller] | None = None
        self.fuselage: Fuselage | None = None
        self.aircraft: Aircraft | None = None
        self.outputs_map: dict[str, MassPropertiesVariable] | None = None

    def parse_variables(
        self,
        components: Sequence[MADSComponent],
        *argv: Any,  # noqa: ANN401, ARG002
    ) -> Sequence[MADSComponent]:
        # Filter components
        _components = copy_components(components)
        components_flat = flatten_components(_components)

        self.propellers = [c for c in components_flat if type(c) is Propeller]
        self.wings = [c for c in components_flat if type(c) is Wing]

        self.fuselage = next(
            (c for c in components_flat if isinstance(c, Fuselage)),
            None,
        )

        try:
            self.aircraft = next(c for c in components_flat if isinstance(c, Aircraft))
        except StopIteration:
            msg = f"An aircraft must be provided to solver '{type(self).__name__}'."
            raise ValueError(msg) from None

        # Extract variables
        variables = ["mass", "global_pos"]
        inputs: list[BaseVariable] = []

        for wing in self.wings:
            inputs.extend(v for k, v in wing.variables.items() if k in variables)
        for prop in self.propellers:
            inputs.extend(v for k, v in prop.variables.items() if k in variables)
        if fs := self.fuselage:
            inputs.extend(v for k, v in fs.variables.items() if k in variables)

        inputs.extend(v for k, v in self.aircraft.variables.items() if k in variables)

        def update_outputs(
            components: Sequence[MADSComponent],
            outputs: list[MassPropertiesVariable],
        ) -> None:
            outputs.extend(
                MassPropertiesVariable.from_num_elements(
                    f"{c.name}.mass_properties",
                    1,
                )
                for c in components
            )

        # Create inner variables
        outputs: list[MassPropertiesVariable] = []
        update_outputs(self.wings, outputs)
        update_outputs(self.propellers, outputs)
        update_outputs([self.aircraft], outputs)

        if self.fuselage:
            update_outputs([self.fuselage], outputs)

        self.inputs = inputs
        self.outputs = outputs
        self.outputs_map = {v.name: v for v in self.outputs}

        if self.fuselage:
            return [*self.wings, *self.propellers, self.fuselage, self.aircraft]
        return [*self.wings, *self.propellers, self.aircraft]

    def _run(self) -> None:
        self.driver = Driver(
            propellers=self.propellers,
            wings=self.wings,
            fuselage=self.fuselage,
            options=self.options,
        )
        self.driver.run()

    def compute_output(self) -> None:
        if self.driver is None or self.aircraft is None:
            msg = f"The driver of solver '{type(self).__name__}' is not initialized."
            raise RuntimeError(msg)

        if self.outputs is None or self.outputs_map is None:
            msg = f"The outputs of solver '{type(self).__name__} are not initialized"
            raise RuntimeError(msg)

        # assign to aircraft the mass properties
        properties = self.driver.global_massVector
        properties_wing = self.driver.wing_mass_vector
        properties_fuselage = self.driver.fuselage_mass_vector
        properties_prop = self.driver.prop_mass_vector

        # assign mass properties to the right component
        if f := self.fuselage:
            self.outputs_map[f"{f.name}.mass_properties"].value = properties_fuselage
        for wing in self.wings:
            self.outputs_map[f"{wing.name}.mass_properties"].value = properties_wing
        for prop in self.propellers:
            self.outputs_map[f"{prop.name}.mass_properties"].value = properties_prop

        self.outputs_map[f"{self.aircraft.name}.mass_properties"].value = properties

    def compute_sensitivities(
        self,
        input_names: Sequence[str],  # noqa: ARG002
        inputs: Sequence[BaseVariable],  # noqa: ARG002
        output_names: Sequence[str],  # noqa: ARG002
        outputs: Sequence[BaseVariable],  # noqa: ARG002
    ) -> Mapping[str, NDArray]:
        return {}
