import numpy as np
from numpy.testing import assert_allclose

from multiads.assembly import Aircraft, AirfoilNACA4, Section, Span, Wing
from multiads.disciplines import UserDefined
from multiads.disciplines.weight_and_balance import WeightAndBalance
from multiads.scenario import (
    MADSScenario,
    VariableFloat,
    VariableFloatNP,
)
from multiads.solvers.weight_and_balance.wnb import WB
from multiads.solvers.weight_and_balance.wnb import Options as WBOptions


def test_wnb() -> None:
    # Design variables
    mass_wing = VariableFloat(name="mass", value=800.0, lb=0.05, ub=1000.0)
    mass_xpos = VariableFloat(name="xpos", value=5.0, lb=0.0, ub=10.0)

    # Coupling variables
    mass_pos = VariableFloatNP(name="pos", value=np.zeros(3))

    wing = Wing(
        name="wing",
        mass=mass_wing,
        global_pos=mass_pos,
        sections=[
            Section(
                name="sec_1",
                airfoil=AirfoilNACA4("airfoil_1", 1, 2, 24),
                chord=2.0,
                twist=0.0,
            ),
            Section(
                name="sec_2",
                airfoil=AirfoilNACA4("airfoil_2", 2, 2, 12),
                chord=1.0,
                twist=-1.0,
            ),
        ],
        spans=[
            Span(name="span_1", length=10.0, sweep=5.0, dihed=2.0),
        ],
    )

    aircraft = Aircraft(name="aircraft")

    # Disciplines
    geometry = UserDefined(
        name="Geometry",
        inputs=[mass_xpos],
        outputs=[mass_pos],
        expression="[xpos[0], 1.0, 0.5]",
    )

    solver = WB(
        options=WBOptions(
            non_linear_inertia_factor=1.0,
            inertia_vector=np.array(
                [100.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            ),
        ),
    )

    meas_weight = WeightAndBalance(
        name="Weight",
        components=[aircraft, wing],
        solver=solver,
    )

    # Scenario
    scenario = MADSScenario()
    scenario.fill_parameter_space([mass_wing, mass_xpos])
    scenario.create_scenario(
        disciplines=[geometry, meas_weight],
        formulation="DisciplinaryOpt",
        objective_name=f"{wing.name}.mass_properties",
        scenario_type="DOE",
    )

    scenario.execute(algo_name="PYDOE_FULLFACT", n_samples=6)

    # Post process
    data = scenario.to_dataset()
    data_pw = data["designs"]["mass"][0]
    data_mass = data["functions"][f"{wing.name}.mass_properties"][1]
    data_iy = data["functions"][f"{wing.name}.mass_properties"][6]

    assert_allclose(data_pw, [0.05, 1000.0, 0.05, 1000.0])
    assert_allclose(data_mass, [0.05, 1000.0, 0.05, 1000.0])
    assert_allclose(data_iy, [25.0, 25.0, 10025.0, 10025.0])
