from typing import Any

import numpy as np

from assembly import Environment, Section, Span, Wing, Configuration
from scenario import Variable
from solvers import SolverFactory, BaseSolver
import solvers.weight_and_balance.raymer_general_aviation_mass_wing_lib as mass_lf


def wing_mass(driver, wing_names: list[str]):
    wing_mass = driver.retrieve_wing_mass(wing_names)
    return np.atleast_1d(wing_mass)


# def wing_mass(driver, wing_names: list[str]) -> NDArray[float]:
#    f = forces(driver, wing_names)
#    return np.atleast_1d(f[0])


@SolverFactory.register("weight_and_balance", "raymer_general_aviation_mass_wing")
class RaymerGeneralAviationMassWing(BaseSolver):

    required_variables = {
        Environment: ["density", "speed"],
        Wing: ["sections", "spans", "fuel_mass_in_wing"],
        Section: Section.attributes(),
        Span: Span.attributes(),
        Configuration: ["m_tom", "ultimate_load_factor"],
    }

    implemented_outputs = {
        "mass_wing": wing_mass,
    }

    def __init__(self) -> None:
        # options of the solver in separated structure
        self.options = mass_lf.Options()
        # simulation driver
        self.driver = None
        # components
        self.environment = None
        self.wings = None
        self.configuration = None

    def get_state(self):
        return [self.environment, self.wings, self.configuration]

    def set_state(self, components):
        # extract items
        self.environment = next(
            filter(lambda x: isinstance(x, Environment), components)
        )
        self.wings = list(filter(lambda x: isinstance(x, Wing), components))
        self.configuration = next(
            filter(lambda x: isinstance(x, Configuration), components)
        )
        self.configuration = mass_lf.Configuration.from_component(self.configuration)

    def run(self):
        """
        Purpose: run the low fi mass model
        """
        # initialise driver object
        self.driver = mass_lf.Driver(
            environment=self.environment,
            wings=self.wings,
            configuration=self.configuration,
            options=self.options,
        )

        # run model
        self.driver.run()

    def compute_output(
        self,
        requested_outputs: list[str],
        mapped_outputs: list[Variable],
    ) -> dict[str, Any]:
        """_summary_: post-process data if needed and collect results"""
        outputs = {}
        for out in requested_outputs:
            if output_var := next((o for o in mapped_outputs if o.name == out), False):
                # Callback options
                wing_names = output_var.options.get(
                    "raymer_general_aviation_mass_wing", {}
                ).get("wings", [])
                out_type = output_var.output_type

                # Callback
                try:
                    out_function = RaymerGeneralAviationMassWing.implemented_outputs[
                        out_type
                    ]
                    outputs[out] = out_function(self.driver, wing_names)
                except KeyError:
                    raise ValueError(
                        f"'{type(self).__name__}' cannot compute '{out_type}"
                    )

        return outputs
