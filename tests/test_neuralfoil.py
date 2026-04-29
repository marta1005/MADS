import copy

import numpy as np
from numpy.testing import assert_allclose
from numpy.typing import NDArray

from multiads.assembly import AirfoilNACA4, Environment, Section, Span, Wing
from multiads.disciplines import UserDefined
from multiads.disciplines.aerodynamics import Aerodynamics
from multiads.scenario import (
    InnerVariableFloat,
    MADSScenario,
    VariableFloat,
)
from multiads.scenario.polars import PolarVariable
from multiads.solvers.aerodynamics.neuralfoil import Neuralfoil, Options


def test_neuralfoil_mdo() -> None:
    # Variables
    chord_1 = VariableFloat(name="chord_1", value=1.0, lb=0.5, ub=2.0)
    chord_2 = VariableFloat(name="chord_2", value=0.7, lb=0.3, ub=1.0)

    # Components
    env = Environment(name="env", height=0.0, speed=100.0)

    wing = Wing(
        name="wing",
        sections=[
            Section(
                name="sec_1",
                airfoil=AirfoilNACA4("airfoil_1", 1, 2, 24),
                chord=chord_1,
                twist=0.0,
            ),
            Section(
                name="sec_2",
                airfoil=AirfoilNACA4("airfoil_2", 2, 2, 12),
                chord=chord_2,
                twist=-1.0,
            ),
        ],
        spans=[
            Span(name="span_1", length=10.0, sweep=5.0, dihed=2.0),
        ],
    )

    # Disciplines
    solver = Neuralfoil(options=Options())
    polars = Aerodynamics(name="Polars", components=[env, wing], solver=solver)

    # IO of user defined discipline
    p1 = copy.deepcopy(
        next(o for o in polars.solver.outputs if o.name == "sec_1.polar"),  # type: ignore[not-iterable]
    )
    p2 = copy.deepcopy(
        next(o for o in polars.solver.outputs if o.name == "sec_2.polar"),  # type: ignore[not-iterable]
    )
    eff = InnerVariableFloat(name="max_efficiency", value=0.0)

    # User defined function
    def objective_f(
        polar_1: NDArray[np.float64],
        polar_2: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        p1 = PolarVariable("p1", polar_1)
        p2 = PolarVariable("p2", polar_2)

        max_eff_1 = np.max(p1.cl / p1.cd)
        max_eff_2 = np.max(p2.cl / p2.cd)

        return np.array([max_eff_1**2 + max_eff_2**2])

    objective = UserDefined(
        name="Max. Efficiency",
        inputs=[p1, p2],
        outputs=[eff],
        expression=objective_f,
    )

    # Scenario
    scenario = MADSScenario()
    scenario.fill_parameter_space([chord_1, chord_2])
    scenario.create_scenario(
        disciplines=[polars, objective],
        formulation="DisciplinaryOpt",
        objective_name="max_efficiency",
        scenario_type="DOE",
    )

    scenario.execute(algo_name="PYDOE_FULLFACT", n_samples=3**2)

    # Post process
    data = scenario.to_dataset()
    data_c1 = data["designs"]["chord_1"][0]
    data_c2 = data["designs"]["chord_2"][0]
    data_eff = data["functions"]["max_efficiency"][0]

    assert_allclose(data_c1, [0.5, 1.25, 2.0, 0.5, 1.25, 2.0, 0.5, 1.25, 2.0])
    assert_allclose(data_c2, [0.3, 0.3, 0.3, 0.65, 0.65, 0.65, 1.0, 1.0, 1.0])
    assert_allclose(
        data_eff,
        [
            25240.292164,
            29463.860539,
            31430.344914,
            28389.484456,
            32613.052831,
            34579.537206,
            30535.505869,
            34759.074243,
            36725.558619,
        ],
    )
