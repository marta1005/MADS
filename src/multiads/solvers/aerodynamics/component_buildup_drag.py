from typing import Any

import numpy as np

from assembly import Environment, Section, Span, Wing, Fuselage, Nacelle
from scenario import Variable
from solvers import SolverFactory, BaseSolver
import solvers.aerodynamics.component_buildup_drag_lib as buildup_drag_lib


def cd0(driver, component_names: list[str]):
    cd0_wing = driver.retrieve_cd0(component_names)
    return np.atleast_1d(cd0_wing)

#def cd0_fuselage(driver, fuselage_names: list[str]):
#    cd0_fuselage = driver.retrieve_cd0_fuselage(fuselage_names)
#    return np.atleast_1d(cd0_fuselage)
#
#def cd0_nacelle(driver, nacelle_names: list[str]):
#    cd0_nacelle = driver.retrieve_cd0_nacelle(nacelle_names)
#    return np.atleast_1d(cd0_nacelle)


@SolverFactory.register("aerodynamics", "component_buildup_drag")
class ComponentBuildupDrag(BaseSolver):

    required_variables = {
        Environment: ["density", "speed", "mach", "dyn_viscosity"],
        Wing: ["sections", "spans"],
        Section: Section.attributes(),
        Span: Span.attributes(),
        Fuselage: Fuselage.attributes(),
        Nacelle: Nacelle.attributes(),
    }

    implemented_outputs = {
        "cd0": cd0,
        #"cd0_fuselage": cd0_fuselage,
        #"cd0_nacelle": cd0_nacelle,
    }

    def __init__(self) -> None:
        # options of the solver in separated structure
        self.options = buildup_drag_lib.Options()
        # simulation driver
        self.driver = None
        # components
        self.environment = None
        self.wings = None
        self.fuselages = None

    def get_state(self):
        return [self.environment, self.wings, self.fuselages, self.nacelles]

    def set_state(self, components):
        # extract items
        self.environment = next(
            filter(lambda x: isinstance(x, Environment), components)
        )
        self.wings = list(filter(lambda x: isinstance(x, Wing), components))
        self.fuselages = list(filter(lambda x: isinstance(x, Fuselage), components))
        self.nacelles = list(filter(lambda x: isinstance(x, Nacelle), components))

    def run(self):
        """
        Purpose: run the low fi mass model
        """
        # initialise driver object
        self.driver = buildup_drag_lib.Driver(
            environment=self.environment,
            wings=self.wings,
            fuselages=self.fuselages,
            nacelles=self.nacelles,
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
                component_names = output_var.options.get(
                    "component_buildup_drag", {}
                ).get("associated_component_names", [])
                out_type = output_var.output_type

                # Callback
                try:
                    out_function = ComponentBuildupDrag.implemented_outputs[
                        out_type
                    ]
                    outputs[out] = out_function(self.driver, component_names)
                except KeyError:
                    raise ValueError(
                        f"'{type(self).__name__}' cannot compute '{out_type}"
                    )

        return outputs
