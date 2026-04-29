import numpy as np
from numpy.typing import NDArray

from assembly import Environment, PropulsionSystem, PowerManagSyst, ThermalSystem

from solvers import SolverFactory, BaseSolver
from scenario import MADSScenario, Variable
import solvers.propulsion.power_management_low_fidelity_lib as pm_lf

from typing import Any


def total_electrical_power(driver, **kwargs: Any) -> NDArray[np.float_]:
    ep = driver.total_electrical_power()
    return np.atleast_1d(ep)


def total_combustion_power(driver, **kwargs: Any) -> NDArray[np.float_]:
    ep = driver.total_combustion_power()
    return np.atleast_1d(ep)


def electrical_power_from_fc(driver, **kwargs: Any) -> NDArray[np.float_]:
    ep = driver.electrical_power_from_fc()
    return np.atleast_1d(ep)


def electrical_power_from_battery(driver, **kwargs: Any) -> NDArray[np.float_]:
    ep = driver.electrical_power_from_battery()
    return np.atleast_1d(ep)


@SolverFactory.register("propulsion", "power_management_low_fidelity")
class PowerManagementLowFidelity(BaseSolver):
    """_summary_

    Raises:
        ValueError: _description_

    Returns:
        _type_: _description_
    """

    required_variables = {
        Environment: Environment.attributes(),
        PropulsionSystem: PropulsionSystem.attributes(),
        PowerManagSyst: [
            "hybrid_factor_engine",
            "hybrid_factor_electric_power",
        ],
        ThermalSystem: ["ecs_power"],
    }

    implemented_outputs = {
        "total_electrical_power": total_electrical_power,
        "total_combustion_power": total_combustion_power,
        "electrical_power_from_fc": electrical_power_from_fc,
        "electrical_power_from_battery": electrical_power_from_battery,
    }

    def __init__(self) -> None:
        # options of the solver in separated holde
        self.options = pm_lf.Options()
        # simulation driver
        self.driver = None
        # components
        self.environment = None
        self.ecs = None
        self.engine = None

    def get_state(self):
        return [self.environment, self.engine]

    def set_state(self, components):
        self.engines = list(
            filter(lambda x: isinstance(x, PropulsionSystem), components)
        )
        
        self.environment = next(
            filter(lambda x: isinstance(x, Environment), components)
        )

        self.powermanag_sys = next(
            filter(lambda x: isinstance(x, PowerManagSyst), components)
        )
        
        # check if component is present
        if filter(lambda x: isinstance(x, ThermalSystem), components):
            self.ecs = list(filter(lambda x: isinstance(x, ThermalSystem), components))



    def run(self):
        """
        Purpose: run the low-fi aero framework
        """
        # initialise driver object
        self.driver = pm_lf.Driver(
            environment=self.environment,
            engines=self.engines,
            pwr_mng=self.powermanag_sys,
            ecs=self.ecs,
            options=self.options,
        )

        # prepare and run model
        self.driver.run()

    def compute_output(
        self,
        requested_outputs: list[str],
        mapped_outputs: list[Variable],
    ) -> dict[str, NDArray[np.float_]]:
        """_summary_: post-process data if needed and collect results"""
        outputs = {}

        # debug
        # print(requested_outputs)

        # for out in requested_outputs:
        #    for o in mapped_outputs:
        #        print(o.name)
        #        print(o.name == out)

        for out in requested_outputs:
            if output_var := next((o for o in mapped_outputs if o.name == out), False):
                # Callback options
                out_options = output_var.options.get("power_mng_lf", {})
                out_type = output_var.output_type

                # Callback
                try:
                    out_function = PowerManagementLowFidelity.implemented_outputs[
                        out_type
                    ]
                    outputs[out] = out_function(self.driver, **out_options)
                except KeyError:
                    raise ValueError(
                        f"'{type(self).__name__}' cannot compute '{out_type}"
                    )

        return outputs
