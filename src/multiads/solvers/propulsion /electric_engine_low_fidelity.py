import numpy as np
from numpy.typing import NDArray

from assembly import Environment, PropulsionSystem, PowerManagSyst, Propeller

from solvers import SolverFactory, BaseSolver
from scenario import MADSScenario, Variable
import solvers.propulsion.electric_engine_low_fidelity_lib as ee_lf


def engine_power(driver, engine_names: list[str]) -> NDArray[np.float_]:
    ep = driver.power_electric_engine(engine_names)
    return np.atleast_1d(ep)


def engine_mass(driver, engine_names: list[str]) -> NDArray[np.float_]:
    em = driver.electric_engine_mass(engine_names)
    return np.atleast_1d(em)


def voltage_rating(driver, engine_names) -> NDArray[np.float_]:
    pass


def engine_temperature(driver, engine_names) -> NDArray[np.float_]:
    pass


@SolverFactory.register("propulsion", "electric_engine_low_fidelity")
class ElectricEngineLowFidelity(BaseSolver):
    """_summary_

    Raises:
        ValueError: _description_

    Returns:
        _type_: _description_
    """

    required_variables = {
        Environment: Environment.attributes(),
        PropulsionSystem: ["engine_power", "shaft_power"],
        PowerManagSyst: [
            "hybrid_factor_engine",
        ],
        }

    implemented_outputs = {
        "engine_power": engine_power,
        "electric_engine_power": engine_power,
        "mass": engine_mass,
        "voltage_rating": voltage_rating,
        "engine_temperature": engine_temperature,
    }

    def __init__(self) -> None:
        # options of the solver in separated holde
        self.options = ee_lf.Options()
        # simulation driver
        self.driver = None
        # components
        self.environment = None
        self.engine = None
        self.powermanag_sys = None
        self.prop = None

    def get_state(self):
        return [self.environment, self.engine]

    def set_state(self, components):
        self.engines = list(
            filter(lambda x: isinstance(x, PropulsionSystem), components)
        )
        self.environment = next(
            filter(lambda x: isinstance(x, Environment), components)
        )
        # check if component is present
        if any(isinstance(x, PowerManagSyst) for x in components):
            print(filter(lambda x: isinstance(x, PowerManagSyst), components))
            self.powermanag_sys = next(
                filter(lambda x: isinstance(x, PowerManagSyst), components)
            )

    def run(self):
        """
        Purpose: run the low-fi aero framework
        """
        # initialise driver object
        self.driver = ee_lf.Driver(
            environment=self.environment,
            engines=self.engines,
            pwr_mng=self.powermanag_sys,
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
        for out in requested_outputs:
            if output_var := next((o for o in mapped_outputs if o.name == out), False):
                # Callback options
                propulsion_names = output_var.options.get("e_eng_lf", {}).get(
                    "propulsion", []
                )
                out_type = output_var.output_type

                # Callback
                try:
                    out_function = ElectricEngineLowFidelity.implemented_outputs[
                        out_type
                    ]
                    outputs[out] = out_function(self.driver, propulsion_names)
                except KeyError:
                    raise ValueError(
                        f"'{type(self).__name__}' cannot compute '{out_type}"
                    )

        return outputs

    # [TODO][SM][review formulation and coupling with the discipline]
    def compute_sensitivities(
        self, input_names, inputs, requested_outputs, mapped_outputs
    ):
        """compute sensitivities of the inputs with respect to the outputs

        Args:
            input_names ([type]): [description]
            inputs ([type]): [description]
            requested_outputs ([type]): [description]
            mapped_outputs ([type]): [description]

        Raises:
            ValueError: [description]

        Returns:
            [type]: [description]
        """
        sensitivity = self.driver._get_sensitivity()
        jac = {}
        for out in requested_outputs:
            for inp in inputs:
                # create dictionary key based on output
                jac.setdefault(out, {})
                try:
                    # create key for the input from the sensitivity matrix
                    jac.setdefault(out, {}).update(inp, sensitivity[out][inp])
                except KeyError:
                    raise ValueError(
                        f"'The Jacobian for the Shaft2Engine Low Fidelity' is not computed for input '{inp}' and output '{out}' "
                    )
        return jac
