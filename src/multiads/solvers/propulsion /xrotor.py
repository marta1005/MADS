from typing import Any

import numpy as np

from assembly import MADSComponent, Environment, Propeller, Wing
from scenario import Variable
from solvers import SolverFactory, BaseSolver
import solvers.propulsion.xrotor_lib as xr


def thrust(name, driver, **kwargs):
    f = driver.thrust
    return np.atleast_1d(f)


def efficiency(name, driver, **kwargs):
    e = driver.efficiency
    return np.atleast_1d(e)


def shaft_power(name, driver, **kwargs):
    f = driver.shaft_power
    return np.atleast_1d(f)


@SolverFactory.register("propulsion", "xrotor")
class XROTOR(BaseSolver):

    implemented_outputs = {
        "thrust": thrust,
        "prop_efficiency": efficiency,
        "shaft_power": shaft_power,
    }

    def __init__(self) -> None:
        # options of the solver in separated holde
        self.options = xr.Options()
        # simulation driver
        self.driver = None
        # components
        self.environment = None
        self.propeller = None

    def get_state(self):
        return [self.environment, self.propeller]

    def set_state(self, components: list[MADSComponent]):
        # instatiante propeller
        self.propeller = next(filter(lambda x: isinstance(x, Propeller), components))
        self.propeller = xr.Propeller.from_component(self.propeller)

        # instatiate environment
        self.environment = filter(lambda x: isinstance(x, Environment), components)
        self.environment = next(self.environment)

    def run(self):
        """
        Purpose: run the dust framework
        """
        # Run Xrotor
        self.driver = xr.Driver(
            environment=self.environment,
            propeller=self.propeller,
            options=self.options,
        )

        # prepare and run XROTOR
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
                out_options = output_var.options.get("xrotor", {})
                out_type = output_var.output_type

                # Callback
                try:
                    out_function = XROTOR.implemented_outputs[out_type]
                    outputs[out] = out_function(out, self.driver, **out_options)
                except KeyError:
                    raise ValueError(
                        f"'{type(self).__name__}' cannot compute '{out_type}"
                    )

        return outputs
