from __future__ import annotations

from typing import TYPE_CHECKING, Any

from multiads.assembly import (
    Fuselage,
    MADSComponent,
    Wing,
    copy_components,
    flatten_components,
)
from multiads.scenario import VariableFloat
from multiads.solvers import BaseSolver
from multiads.solvers.weight_and_balance.raymer_mass_lib import Driver, Options

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from numpy.typing import NDArray


class RaymerMass(BaseSolver):
    def __init__(self, options: Options | None = None) -> None:
        super().__init__()
        self.options: Options = options or Options()
        self.driver: Driver | None = None
        self.wings: Sequence[Wing] | None = None
        self.fuselage: Fuselage | None = None
        self.environment: Any = None
        self.mass_variables: dict[str, VariableFloat] = {}

    def parse_variables(
        self,
        components: Sequence[MADSComponent],
        *argv: Any,
    ) -> Sequence[MADSComponent]:
        _components = copy_components(components)
        components_flat = flatten_components(_components)

        self.wings = [c for c in components_flat if isinstance(c, Wing)]
        self.fuselage = next(
            (c for c in components_flat if isinstance(c, Fuselage)),
            None,
        )

        env_components = [
            c
            for c in components_flat
            if "environment" in c.name.lower() or type(c).__name__ == "Environment"
        ]
        self.environment = env_components[0] if env_components else None

        if not self.wings:
            msg = "At least one Wing component must be provided."
            raise ValueError(msg)

        if self.fuselage is None:
            msg = "A Fuselage component must be provided."
            raise ValueError(msg)

        if self.environment is None:
            msg = "An Environment component must be provided."
            raise ValueError(msg)

        self.options.density = self.environment.density
        self.options.velocity = self.environment.speed

        self.inputs: list[VariableFloat] = []
        for wing in self.wings:
            for key, var in wing.variables.items():
                if key == "mass":
                    self.inputs.append(var)

        for key, var in self.fuselage.variables.items():
            if key == "mass":
                self.inputs.append(var)

        self.mass_variables = {}
        for wing in self.wings:
            self.mass_variables[f"{wing.name}.mass"] = VariableFloat(
                name=f"{wing.name}.mass",
                value=0.0,
            )

        self.mass_variables["fuselage.mass"] = VariableFloat(
            name="fuselage.mass",
            value=0.0,
        )

        self.outputs = list(self.mass_variables.values())

        return list(self.wings) + [self.fuselage]

    def _run(self) -> None:
        if self.wings is None or self.fuselage is None:
            msg = "Components not initialized. Call parse_variables first."
            raise RuntimeError(msg)

        if self.options.m_tom is None:
            msg = "m_tom must be set in Options."
            raise ValueError(msg)

        if self.options.ultimate_load_factor is None:
            msg = "ultimate_load_factor must be set in Options."
            raise ValueError(msg)

        from multiads.solvers.weight_and_balance.raymer_mass_lib import Fuselage as RMFuselage
        from multiads.solvers.weight_and_balance.raymer_mass_lib import Wing as RMWing

        wings_lib = [RMWing.from_component(w) for w in self.wings]
        fuselage_lib = RMFuselage.from_component(self.fuselage)

        self.driver = Driver(
            wings=wings_lib,
            fuselage=fuselage_lib,
            options=self.options,
        )
        self.driver.run()

    def compute_output(self) -> None:
        if self.driver is None:
            msg = "Driver not initialized. Call _run first."
            raise RuntimeError(msg)

        for i, wing in enumerate(self.wings):
            var_name = f"{wing.name}.mass"
            if var_name in self.mass_variables:
                self.mass_variables[var_name].value = self.driver.wing_masses[i]

        self.mass_variables["fuselage.mass"].value = self.driver.fuselage_mass

    def compute_sensitivities(
        self,
        input_names: Sequence[str],
        inputs: Sequence[VariableFloat],
        output_names: Sequence[str],
        outputs: Sequence[VariableFloat],
    ) -> Mapping[str, NDArray]:
        return {}
