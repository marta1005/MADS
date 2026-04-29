from typing import Any

import numpy as np

from assembly import Environment, Fuselage, Configuration
from scenario import Variable
from solvers import SolverFactory, BaseSolver
import solvers.weight_and_balance.raymer_general_aviation_mass_fuselage_lib as mass_lf


def fuselage_mass(name, driver, **kwargs):
    return np.asarray([driver.fuselage_mass])


@SolverFactory.register("weight_and_balance", "raymer_general_aviation_mass_fuselage")
class RaymerGeneralAviationMassFuselage(BaseSolver):

    required_variables = {
        Environment: ["density", "speed"],
        Fuselage: ["wetted_area",
                   "maximum_width",
                   "maximum_height",
                   "length",
                   "volume_pressurized_cabin",
                   "maximum_fuselage_pressure_differential",
                   ],
        Configuration: ["m_tom", "ultimate_load_factor"],
    }

    implemented_outputs = {
        "mass_fuselage": fuselage_mass,
    }

    def __init__(self) -> None:
        # options of the solver in separated structure
        self.options = mass_lf.Options()
        # simulation driver
        self.driver = None
        # components
        self.environment = None
        self.fuselage = None
        self.configuration = None

    def get_state(self):
        return [self.environment, self.fuselage, self.configuration]

    def set_state(self, components):
        # extract items
        self.environment = next(filter(lambda x: isinstance(x, Environment), components))
        self.fuselage = next(filter(lambda x: isinstance(x, Fuselage), components))
        self.fuselage = mass_lf.Fuselage.from_component(self.fuselage)
        self.configuration = next(filter(lambda x: isinstance(x, Configuration), components))
        self.configuration = mass_lf.Configuration.from_component(self.configuration)

    def run(self):
        """
        Purpose: run the low fi mass model
        """
        # initialise driver object
        self.driver = mass_lf.Driver(
            environment=self.environment,
            fuselage=self.fuselage,
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
                out_options = output_var.options.get("raymer_general_aviation_mass_fuselage", {})
                out_type = output_var.output_type

                # Callback
                try:
                    out_function = RaymerGeneralAviationMassFuselage.implemented_outputs[out_type]
                    outputs[out] = out_function(out, self.driver, **out_options)
                except KeyError:
                    raise ValueError(
                        f"'{type(self).__name__}' cannot compute '{out_type}"
                    )

        return outputs
