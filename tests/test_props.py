import copy

import numpy as np
from numpy.typing import NDArray

import os, sys

sys.path.insert(
        0,
        os.path.abspath(os.path.join(os.path.dirname(__file__), "../src/")),
    )

from multiads.assembly import AirfoilNACA4, Section, Span, Wing, Propeller
from multiads.disciplines import UserDefined
from multiads.disciplines.aerodynamics import Aerodynamics
from multiads.disciplines.propulsion import Propulsion
from multiads.scenario import InnerVariable, MADSScenario, VariableFloat
from multiads.solvers.propulsion.propeller_low_fidelity import PropellerLowFidelity 
from multiads.solvers.propulsion.propeller_low_fidelity import Options as PROptions

class TestModels:
    def test_propeller(self) -> None:

        thrust_in = VariableFloat(
            name="thrust",
            value=np.array([1000.0]),
            lb=np.array([500.0]),
            ub=np.array([500.0]),
        )
        rpm_in = VariableFloat(name="rpm", value=np.array([1900.0]), lb=1000.0, ub=3000.0)
        
        # varaible is internal to prop
        # prop_shaft_power = InnerVariable(name="shaft_power", value=np.array([1900.0]))

        speed = 150.0

        blade = Wing(
            name="blade",
            xc_ref=0.5,
            options={
                "dust": {
                    "n_elem": 10,
                    "elem_type": "uniform",
                }
            },
            sections=[
                Section(
                    "blade_root_sec",
                    AirfoilNACA4("blade_root_foil", 1, 2, 30),
                    chord=0.30,
                    twist=27.0,
                ),
                Section(
                    "blade_mid_sec",
                    AirfoilNACA4("blade_mid_foil", 2, 4, 15),
                    chord=0.30,
                    twist=12.0,
                ),
                Section(
                    "blade_tip_sec",
                    AirfoilNACA4("blade_tip_foil", 2, 4, 10),
                    chord=0.15,
                    twist=5.0,
                ),
            ],
            spans=[
                Span(
                    "blade_span_in",
                    length=0.8,
                    sweep=0.0,
                    dihed=0.0,
                    options={
                        "dust": {
                            "n_elem": 8,
                            "elem_type": "uniform",
                        }
                    },
                ),
                Span(
                    "blade_span_out",
                    length=0.6,
                    sweep=0.0,
                    dihed=0.0,
                    options={
                        "dust": {
                            "n_elem": 8,
                            "elem_type": "uniform",
                        }
                    },
                ),
            ],
        )

        # static variabels
        r_tip_in  = 2.0
        n_blades_in = 6.0
        pitch_in = 19

        prop = Propeller(name="propeller",
                        blade=blade,
                        r_tip=r_tip_in,
                        n_blades=n_blades_in,
                        pitch=pitch_in,
                        rpm=rpm_in,
                        thrust=thrust_in,
                    )

        sqrt_prop_pitch = np.sqrt(19)
        # introdue relation between rpm and thrust
        disc_coupling = UserDefined(
                "Thrust_rpm_coupling",
            inputs=[rpm_in],
            outputs=[thrust_in],
            expression=f"(4.392*10**-8 * rpm * (2*{r_tip_in}/39.3701)**3.5 / {sqrt_prop_pitch} * (4.3*10**-4 * rpm * {sqrt_prop_pitch}**2 -{speed} ))",
        )

        # Disciplines
        solver = PropellerLowFidelity(
            options=PROptions(
                altitude = 1500.0,
            ),
        )
        disc_propulsion = Propulsion(name="Propeller-Prop",components=[prop],solver=solver)
        
        # Scenario
        scenario = MADSScenario()
        scenario.fill_parameter_space([rpm_in])
        scenario.create_scenario(
            disciplines=[disc_propulsion,disc_coupling],
            formulation="DisciplinaryOpt",
            objective_name=f"{prop.name}.shaft_power",
            scenario_type="DOE",
        )

        # scenario.scenario.xdsmize()
        scenario.scenario.execute(algo_name="PYDOE_FULLFACT", n_samples=3**2)
        
        # Post process
        data = scenario.scenario.to_dataset()
        data_rpm = data["designs"]["rpm"][0]
        data_power = data["functions"][f"{prop.name}.shaft_power"][0]
        
        assert np.all(
            np.isclose(data_rpm, [1000.0,1250.0,1500.0,1750.0,2000.0,2250.0,2500.0,2750.0,3000.0]),
        )
        
        try:
            assert True
            assert np.all(
                np.isclose(data_power.values,
                           [672705.188583, 
                            344425.056554, 
                            199320.055876, 
                            125519.335479,
                            84088.148573,
                            59057.794334,
                            43053.132069,
                            32346.455349,
                            24915.006985,
                           ],
                        ),
                    )
        except AssertionError:
            import traceback
            _, _, tb = sys.exc_info()             
            traceback.print_tb(tb) 
            
            
if __name__ == "__main__":
   
    test = TestModels()
    test.test_propeller()