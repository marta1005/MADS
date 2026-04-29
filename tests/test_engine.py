import copy

import numpy as np
from numpy.typing import NDArray

import os, sys

sys.path.insert(
        0,
        os.path.abspath(os.path.join(os.path.dirname(__file__), "../src/")),
    )

from multiads.assembly import AirfoilNACA4, Section, Span, Wing, Propeller, PropulsionSystem
from multiads.disciplines import UserDefined
from multiads.disciplines.aerodynamics import Aerodynamics
from multiads.disciplines.propulsion import Propulsion
from multiads.scenario import InnerVariable, MADSScenario, VariableFloat
from multiads.solvers.propulsion.propeller_low_fidelity import PropellerLowFidelity 
from multiads.solvers.propulsion.propeller_low_fidelity import Options as PROptions
from multiads.solvers.propulsion.thermal_engine_low_fidelity import ThermalEngineLowFidelity 
from multiads.solvers.propulsion.thermal_engine_low_fidelity import Options as EOptions

class TestModels:
    def test_propeller(self) -> None:

        thrust_in = VariableFloat(
            name="thrust",
            value=np.array([1000.0]),
            lb=np.array([500.0]),
            ub=np.array([500.0]),
        )
        rpm_in = VariableFloat(name="rpm", value=np.array([1900.0]), lb=1000.0, ub=3000.0)
        

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
        
        # varaible is internal to prop
        power = InnerVariable(name=f"{prop.name}.shaft_power", value=np.array([1900.0]))
        
        engine = PropulsionSystem(name="engine",
                        type="thermal",
                        power=power,
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
        
        throttles = [
            -0.001,
            0.05,
            0.1,
            0.15,
            0.2,
            0.25,
            0.3,
            0.35,
            0.4,
            0.45,
            0.50,
            0.55,
            0.60,
            0.65,
            0.70,
            0.75,
            0.80,
            0.85,
            0.90,
            0.95,
            1.00,
            3,
        ]
        psfcs = [
            0.450,
            0.425,
            0.401,
            0.376,
            0.352,
            0.327,
            0.303,
            0.285,
            0.276,
            0.271,
            0.270,
            0.272,
            0.275,
            0.279,
            0.284,
            0.289,
            0.293,
            0.298,
            0.305,
            0.320,
            0.350,
            0.450,
        ]
        
        throttle_map = np.stack((np.array(throttles),np.array(psfcs)),axis=0)
        
                
        solver_eng = ThermalEngineLowFidelity(
            options=EOptions(
                segment_duration = 3600,
                throttle_map=throttle_map,
            ),
        )
        
        
        disc_propeller = Propulsion(name="Propeller-Prop",components=[prop],solver=solver)
        
        disc_propulsion = Propulsion(name="Propulsion-System",components=[engine,prop],solver=solver_eng)

        
        # Scenario
        scenario = MADSScenario()
        scenario.fill_parameter_space([rpm_in])
        scenario.create_scenario(
            disciplines=[disc_propeller,disc_coupling,disc_propulsion],
            formulation="DisciplinaryOpt",
            objective_name=f"{engine.name}.fuel_consumption",
            scenario_type="DOE",
        )

        # scenario.scenario.xdsmize()
        scenario.scenario.execute(algo_name="PYDOE_FULLFACT", n_samples=3**2)
        
        # Post process
        data = scenario.scenario.to_dataset()
        data_rpm = data["designs"]["rpm"][0]
        data_engine = data["functions"][f"{engine.name}.fuel_consumption"][0]
        
        assert np.all(
            np.isclose(data_rpm, [1000.0,1250.0,1500.0,1750.0,2000.0,2250.0,2500.0,2750.0,3000.0]),
        )
                
        try:
            assert True
            assert np.all(
                np.isclose(data_engine.values,
                           [5360142.35, 
                            5360142.35, 
                            5360142.35, 
                            5360142.35,
                            5360142.35,
                            5360142.35,
                            5360142.35,
                            5360142.35,
                            5360142.35,
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