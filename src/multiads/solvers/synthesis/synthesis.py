from typing import Any

import numpy as np
from numpy.typing import NDArray

from assembly import MADSComponent, Aircraft, Wing, Environment, MassProperties, AerodynamicProperties
from scenario import Variable
from solvers import SolverFactory, BaseSolver
import solvers.synthesis.synthesis_lib as dl


def body_AeroDerivatives_alpha(name: str, driver: dl.DSdriver, **kwargs: Any) -> NDArray[np.float_]:
    body_AeroDerivatives_alpha = driver.body_AeroDerivatives_alpha()
    return np.array(body_AeroDerivatives_alpha)


def body_AeroDerivatives_beta(name: str, driver: dl.DSdriver, **kwargs: Any) -> NDArray[np.float_]:
    body_AeroDerivatives_beta = driver.body_AeroDerivatives_beta()
    return np.array(body_AeroDerivatives_beta)


def body_AeroDerivatives_p(name: str, driver: dl.DSdriver, **kwargs: Any) -> NDArray[np.float_]:
    body_AeroDerivatives_p = driver.body_AeroDerivatives_p()
    return np.array(body_AeroDerivatives_p)


def body_AeroDerivatives_q(name: str, driver: dl.DSdriver, **kwargs: Any) -> NDArray[np.float_]:
    body_AeroDerivatives_q = driver.body_AeroDerivatives_q()
    return np.array(body_AeroDerivatives_q)


def body_AeroDerivatives_r(name: str, driver: dl.DSdriver, **kwargs: Any) -> NDArray[np.float_]:
    body_AeroDerivatives_r = driver.body_AeroDerivatives_r()
    return np.array(body_AeroDerivatives_r)


def body_AeroDerivatives_alpha_dot(name: str, driver: dl.DSdriver, **kwargs: Any) -> NDArray[np.float_]:
    body_AeroDerivatives_alpha_dot = driver.body_AeroDerivatives_alpha_dot()
    return np.array(body_AeroDerivatives_alpha_dot)


def body_AeroDerivatives_beta_dot(name: str, driver: dl.DSdriver, **kwargs: Any) -> NDArray[np.float_]:
    body_AeroDerivatives_beta_dot = driver.body_AeroDerivatives_beta_dot()
    return np.array(body_AeroDerivatives_beta_dot)


def body_AeroDerivatives_pitching(name: str, driver: dl.DSdriver, **kwargs: Any) -> NDArray[np.float_]:
    body_AeroDerivatives_pitching = driver.body_AeroDerivatives_pitching()
    return np.array(body_AeroDerivatives_pitching)


def body_AeroDerivatives_yawing(name: str, driver: dl.DSdriver, **kwargs: Any) -> NDArray[np.float_]:
    body_AeroDerivatives_yawing = driver.body_AeroDerivatives_yawing()
    return np.array(body_AeroDerivatives_yawing)


def body_AeroDerivatives_delta_aileron(name: str, driver: dl.DSdriver, **kwargs: Any) -> NDArray[np.float_]:
    body_AeroDerivatives_delta_aileron = driver.body_AeroDerivatives_delta_aileron()
    return np.array(body_AeroDerivatives_delta_aileron)


def body_AeroDerivatives_delta_elevator(name: str, driver: dl.DSdriver, **kwargs: Any) -> NDArray[np.float_]:
    body_AeroDerivatives_delta_elevator = driver.body_AeroDerivatives_delta_elevator()
    return np.array(body_AeroDerivatives_delta_elevator)


def body_AeroDerivatives_delta_rudder(name: str, driver: dl.DSdriver, **kwargs: Any) -> NDArray[np.float_]:
    body_AeroDerivatives_delta_rudder = driver.body_AeroDerivatives_delta_rudder()
    return np.array(body_AeroDerivatives_delta_rudder)


def thetas_new(name: str, driver: dl.DSdriver, **kwargs: Any) -> NDArray[np.float_]:
    thetas_new = driver.thetas_deformed()
    return np.asarray(thetas_new)


def dihedrals_new(name: str, driver: dl.DSdriver, **kwargs: Any) -> NDArray[np.float_]:
    dihedrals_new = driver.dihedrals_deformed()
    print(dihedrals_new)
    return np.asarray(dihedrals_new)


def spans_new(name: str, driver: dl.DSdriver, **kwargs: Any) -> NDArray[np.float_]:
    spans_new = driver.spans_deformed()
    return np.asarray(spans_new)




@SolverFactory.register("design_synthesis", "synthesis")
class DS(BaseSolver):

    required_variables = {
        Environment: Environment.attributes(),
        MassProperties: MassProperties.attributes(),
        AerodynamicProperties: AerodynamicProperties.attributes(),
        Aircraft: Aircraft.attributes(),
        Wing: Wing.attributes(),
    }

    implemented_outputs = {
        "body_AeroDerivatives_alpha": body_AeroDerivatives_alpha,
        "body_AeroDerivatives_beta": body_AeroDerivatives_beta,
        "body_AeroDerivatives_p": body_AeroDerivatives_p,
        "body_AeroDerivatives_q": body_AeroDerivatives_q,
        "body_AeroDerivatives_r": body_AeroDerivatives_r,
        "body_AeroDerivatives_alpha_dot": body_AeroDerivatives_alpha_dot,
        "body_AeroDerivatives_beta_dot": body_AeroDerivatives_beta_dot,
        "body_AeroDerivatives_pitching": body_AeroDerivatives_pitching,
        "body_AeroDerivatives_yawing": body_AeroDerivatives_yawing,
        "body_AeroDerivatives_delta_aileron": body_AeroDerivatives_delta_aileron,
        "body_AeroDerivatives_delta_elevator": body_AeroDerivatives_delta_elevator,
        "body_AeroDerivatives_delta_rudder": body_AeroDerivatives_delta_rudder,
        "thetas_new": thetas_new,
        "spans_new": spans_new,
        "dihedrals_new": dihedrals_new,
    }

    def __init__(self) -> None:
        # options of the solver in separated holde
        self.options = dl.DSoptions()
        # simulation driver
        self.driver = None
        # components
        self.aircraft = None
        self.wings = None
        self.environment = None

    def get_state(self) -> list[MADSComponent]:
        return [self.environment, *self.aircraft, *self.wings]

    def set_state(self, components: list[MADSComponent]) -> None:

        self.aircraft = filter(lambda x: isinstance(x, Aircraft), components)
        self.aircraft = list(map(lambda x: dl.Aircraft.from_component(x), self.aircraft))
        
        self.wings = filter(lambda x: isinstance(x, Wing), components)
        self.wings = list(map(lambda x: dl.Wing.from_component(x), self.wings))

        self.environment = filter(lambda x: isinstance(x, Environment), components)
        self.environment = next(self.environment)

    def run(self):
        # Run Synthesis Driver
        self.driver = dl.DSdriver(
            environment=self.environment,
            aircraft=self.aircraft,
            wings=self.wings
        )
        self.driver.compute_aircraft_properties()
        self.driver.find_nodes()
        self.driver.find_deform()

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
                out_options = output_var.options.get("synthesis", {})
                out_type = output_var.output_type

                # Callback
                try:
                    out_function = DS.implemented_outputs[out_type]
                    outputs[out] = out_function(out, self.driver, **out_options)
                except KeyError:
                    raise ValueError(
                        f"'{type(self).__name__}' cannot compute '{out_type}'"
                    )

                # Debug
                #print(f"{out} = {outputs[out]}")

        return outputs
