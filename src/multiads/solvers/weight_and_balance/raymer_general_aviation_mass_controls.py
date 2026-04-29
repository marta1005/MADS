from typing import Any

import numpy as np

from assembly import Fuselage, Configuration, Wing, Section, Span
from scenario import Variable
from solvers import SolverFactory, BaseSolver
import solvers.weight_and_balance.raymer_general_aviation_mass_controls_lib as mass_lf


def flight_controls_mass(name, driver, **kwargs):
    return np.asarray([driver.flight_controls_mass])

@SolverFactory.register("weight_and_balance", "raymer_general_aviation_mass_controls")
class RaymerGeneralAviationMassControls(BaseSolver):

    required_variables = {
        Fuselage: ["length"],
        Configuration: ["m_tom", "ultimate_load_factor"],
        Wing: ["sections", "spans"],
        Section: Section.attributes(),
        Span: Span.attributes(),
    }

    implemented_outputs = {
        "mass_controls": flight_controls_mass,
    }

    def __init__(self) -> None:
        # options of the solver in separated structure
        self.options = mass_lf.Options()
        # simulation driver
        self.driver = None
        # components
        self.fuselage = None
        self.wings = None
        self.configuration = None

    def get_state(self):
        return [self.fuselage, self.wings, self.configuration]

    def set_state(self, components):
        # extract items
        self.fuselage = next(filter(lambda x: isinstance(x, Fuselage), components))
        self.fuselage = mass_lf.Fuselage.from_component(self.fuselage)
        self.wings = list(filter(lambda x: isinstance(x, Wing), components))
        self.configuration = next(filter(lambda x: isinstance(x, Configuration), components))
        self.configuration = mass_lf.Configuration.from_component(self.configuration)

    def run(self):
        """
        Purpose: run the low fi mass model
        """
        # initialise driver object
        self.driver = mass_lf.Driver(
            fuselage=self.fuselage,
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
                out_options = output_var.options.get("raymer_general_aviation_mass_controls", {})
                out_type = output_var.output_type

                # Callback
                try:
                    out_function = RaymerGeneralAviationMassControls.implemented_outputs[out_type]
                    outputs[out] = out_function(out, self.driver, **out_options)
                except KeyError:
                    raise ValueError(
                        f"'{type(self).__name__}' cannot compute '{out_type}"
                    )

        return outputs
