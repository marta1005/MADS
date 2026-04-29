import numpy as np
from numpy.typing import NDArray

from assembly import Environment, MADSComponent, Section, Span, Wing
from scenario import Variable
from solvers import SolverFactory, BaseSolver
import solvers.aerodynamics.aero_low_fidelity_lib as aero_lf


def forces(driver, wing_names: list[str]) -> NDArray[np.float_]:
    f = driver.forces(wing_names)
    return np.atleast_1d([f[1], 0.0, f[0]])


def lift(driver, wing_names: list[str]) -> NDArray[np.float_]:
    f = forces(driver, wing_names)
    return np.atleast_1d(f[2])


def drag(driver, wing_names: list[str]) -> NDArray[np.float_]:
    f = forces(driver, wing_names)
    return np.atleast_1d(f[0])


def thrust(driver, wing_names: list[str]) -> NDArray[np.float_]:
    f = forces(driver, wing_names)
    return np.atleast_1d(-f[0])


def coefficients(driver, wing_names: list[str]) -> NDArray[np.float_]:
    coeff = driver.coefficients(wing_names)
    return np.atleast_1d(coeff)


def lift_coefficient(driver, wing_names: list[str]) -> NDArray[np.float_]:
    coeff = coefficients(driver, wing_names)
    return np.atleast_1d(coeff[0])


def drag_coefficient(driver, wing_names: list[str]) -> NDArray[np.float_]:
    coeff = coefficients(driver, wing_names)
    return np.atleast_1d(coeff[1])


def spanwise_loads(driver, wing_names: list[str]) -> NDArray[np.float_]:
    spanloads = driver.spanloads(wing_names)

    matrix = driver.environment.matrixWindBody
    #m = np.dot(matrix, forces_analysis.m[-1])


    return np.array(spanloads).flatten()


def moments(driver, wing_names: list[str]) -> NDArray[np.float_]:
    m = driver.moments(wing_names)
    return np.atleast_1d(m)


def moment_y(driver, wing_names: list[str]) -> NDArray[np.float_]:
    m = driver.moments(wing_names)
    return np.asarray([m[1]])


@SolverFactory.register("aerodynamics", "aero_lf")
class AeroLowFidelity(BaseSolver):

    required_variables = {
        Environment: Environment.attributes(),
        Wing: ["sections", "spans", "alpha","cd0"],
        Section: Section.attributes(),
        Span: Span.attributes(),
    }

    implemented_outputs = {
        "forces": forces,
        "lift": lift,
        "drag": drag,
        "thrust": thrust,
        "moments": moments,
        "my": moment_y,
        "span_load": spanwise_loads,
        "coefficients": coefficients,
        "cl": lift_coefficient,
        "cd": drag_coefficient,
    }

    def __init__(self) -> None:
        # options of the solver in separated structure
        self.options = aero_lf.aero_lf_options()
        # simulation driver
        self.driver = None
        # components
        self.environment = None
        self.wings = None

    def get_state(self):
        return [self.environment, *self.wings]

    def set_state(self, components: list[MADSComponent]):
        self.wings = list(filter(lambda x: isinstance(x, Wing), components))
        self.environment = next(
            filter(lambda x: isinstance(x, Environment), components)
        )

    def run(self):
        """
        Purpose: run the low fi aero framework
        """
        # initialise driver object
        self.driver = aero_lf.Driver(
            environment=self.environment,
            wings=self.wings,
            options=self.options,
        )

        # run model
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
                wing_names = output_var.options.get("aero_lf", {}).get("wings", [])
                out_type = output_var.output_type

                # Callback
                try:
                    out_function = AeroLowFidelity.implemented_outputs[out_type]
                    outputs[out] = out_function(self.driver, wing_names)
                except KeyError:
                    raise ValueError(
                        f"'{type(self).__name__}' cannot compute '{out_type}"
                    )

        return outputs
