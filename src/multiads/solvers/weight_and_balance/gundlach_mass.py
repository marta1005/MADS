from __future__ import annotations

from typing import TYPE_CHECKING, Any

from multiads.assembly import (
    Fuselage as AssemblyFuselage,
    MADSComponent,
    Propeller as AssemblyPropeller,
    copy_components,
    flatten_components,
)
from multiads.scenario import VariableFloat
from multiads.solvers import BaseSolver
from multiads.solvers.weight_and_balance.gundlach_mass_lib import (
    Driver,
    Fuselage,
    Options,
    Propeller,
)

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from numpy.typing import NDArray


class GundlachMass(BaseSolver):
    def __init__(self, options: Options | None = None) -> None:
        super().__init__()
        self.options: Options = options or Options()
        self.driver: Driver | None = None
        self.propeller_component: AssemblyPropeller | None = None
        self.fuselage_component: AssemblyFuselage | None = None
        self.mass_variables: dict[str, VariableFloat] = {}

    def parse_variables(
        self,
        components: Sequence[MADSComponent],
        *argv: Any,
    ) -> Sequence[MADSComponent]:
        _components = copy_components(components)
        components_flat = flatten_components(_components)

        self.propeller_component = next(
            (c for c in components_flat if isinstance(c, AssemblyPropeller)),
            None,
        )

        self.fuselage_component = next(
            (c for c in components_flat if isinstance(c, AssemblyFuselage)),
            None,
        )

        self.mass_variables = {}

        if self.propeller_component is not None:
            self.mass_variables[
                f"{self.propeller_component.name}.mass"
            ] = VariableFloat(name=f"{self.propeller_component.name}.mass", value=0.0)

        if self.fuselage_component is not None:
            self.mass_variables["fuselage.mass"] = VariableFloat(
                name="fuselage.mass",
                value=0.0,
            )

        self.inputs: list[VariableFloat] = []
        self.outputs = list(self.mass_variables.values())

        result_components = []
        if self.propeller_component is not None:
            result_components.append(self.propeller_component)
        if self.fuselage_component is not None:
            result_components.append(self.fuselage_component)

        return result_components

    def _run(self) -> None:
        from multiads.solvers.weight_and_balance.gundlach_mass_lib import (
            Fuselage as LibFuselage,
            Propeller as LibPropeller,
        )

        propeller_lib = None
        fuselage_lib = None

        if self.propeller_component is not None:
            propeller_lib = LibPropeller.from_component(self.propeller_component)

        if self.fuselage_component is not None:
            fuselage_lib = LibFuselage.from_component(self.fuselage_component)

        self.driver = Driver(
            propeller=propeller_lib,
            fuselage=fuselage_lib,
            options=self.options,
        )
        self.driver.run()

    def compute_output(self) -> None:
        if self.driver is None:
            msg = "Driver not initialized. Call _run first."
            raise RuntimeError(msg)

        if (
            self.propeller_component is not None
            and self.driver.propeller_mass is not None
        ):
            self.mass_variables[
                f"{self.propeller_component.name}.mass"
            ].value = self.driver.propeller_mass

        if self.fuselage_component is not None and self.driver.fuselage_mass is not None:
            self.mass_variables["fuselage.mass"].value = self.driver.fuselage_mass

    def compute_sensitivities(
        self,
        input_names: Sequence[str],
        inputs: Sequence[VariableFloat],
        output_names: Sequence[str],
        outputs: Sequence[VariableFloat],
    ) -> Mapping[str, NDArray]:
        return {}
