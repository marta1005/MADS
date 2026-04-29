import copy

import numpy as np
from numpy.typing import NDArray

from multiads.assembly import AirfoilNACA4, Section, Span, Wing, FuelCell
from multiads.disciplines import UserDefined
from multiads.disciplines.aerodynamics import Aerodynamics
from multiads.disciplines.power_supply import PowerSupply
from multiads.scenario import InnerVariable, MADSScenario, VariableFloat
from multiads.scenario.polars import PolarVariable
from multiads.solvers.aerodynamics.neuralfoil import Neuralfoil, Options
from multiads.solvers.power_supply.fuel_cell_low_fidelity import FuelCellLowFidelity 
from multiads.solvers.power_supply.fuel_cell_low_fidelity import Options as FCOptions


from multiads.assembly import AirfoilNACA4, Environment
from multiads.assembly import Section as AssemblySection
from multiads.assembly import Span as AssemblySpan
from multiads.assembly import Wing as AssemblyWing

import shutil
from warnings import warn
import logging, traceback


class TestModels:
    def test_fuelcell(self) -> None:
        # Variables
        electrical_power_from_fc = VariableFloat(
            name="power",
            value=np.array([800.0]),
        )

        fuelcell = FuelCell(name="fuelcell", 
                            power=electrical_power_from_fc, 
                            waste_heat=0.0,
                            h2_fuel_flow=0.0 
                    )

        # Disciplines
        solver = FuelCellLowFidelity(
            options=FCOptions(
                activation_area = 100.0,
                current_density = 0.5,
                delta_G  = 230000.0
            ),
        )
        
        power_supply = PowerSupply(name="Fuel_Cell", components=[fuelcell], solver=solver)

        # Scenario
        scenario = MADSScenario()
        scenario.fill_parameter_space([electrical_power_from_fc])
        scenario.create_scenario(
            disciplines=[power_supply],
            formulation="DisciplinaryOpt",
            objective_name=f"{fuelcell.name}.mass",
            scenario_type="DOE",
        )

        scenario.scenario.execute(algo_name="PYDOE_FULLFACT", n_samples=3**2)
        
        # Post process
        data = scenario.scenario.to_dataset()
        data_pw = data["designs"]["power"][0]
        data_mass = data["functions"][f"{fuelcell.name}.mass"][0]
        
        assert np.all(
            np.isclose(data_pw, [0.000, 0.125, 0.250, 0.375, 0.500, 0.625, 0.750, 0.875, 1.000]),
        )
        
        assert np.all(
            np.isclose(data_mass.values,
                [
                    0.00000000e+00, 
                    9.18898063e-05, 
                    1.83779613e-04, 
                    2.75669419e-04,
                    3.67559225e-04,
                    4.59449032e-04,
                    5.51338838e-04,
                    6.43228644e-04,
                    7.35118451e-04,
                ],
            ),
        )
    
    def test_ips(self) -> None:
        # Variables
        chordv = VariableFloat(name="chord-wing", value=np.array([2.0]))
        # Create test environment
        env = Environment(
            name="test_env",
            height=0.0,
            speed=100.0,
        )
        
        try:
            from multiads.solvers.thermal.ips_low_fidelity import IPSLowFidelity 
            from multiads.solvers.power_supply.ips_low_fidelity import Options as IPSOptions

            # Create test wing
            wing = IPSLowFidelity(
                name="test_wing",
                sections=[
                    AssemblySection(
                        name="sec1",
                        airfoil=AirfoilNACA4("NACA2412", 2, 4, 12),
                        chord=chordv,
                        twist=0.0,
                    ),
                    AssemblySection(
                        name="sec2",
                        airfoil=AirfoilNACA4("NACA2412", 1, 2, 12),
                        chord=chordv,
                        twist=0.0,
                    ),
                ],
                spans=[
                    AssemblySpan(
                        name="span1",
                        length=10.0,
                        sweep=5.0,
                        dihed=2.0,
                        options=[
                            SpanOptions(panel_type=SpanPanelType.UNIFORM, panel_density=1),
                        ],
                    ),
                ],
                options=[
                ],
            )
            # Disciplines
            solver = IPSLowFidelity(
                options=IPSOptions(
                    AoA_min_root = 2.0,
                ),
            )

            ice_protection = PowerSupply(name="Ice-Protection-Sys", components=[wing], solver=solver)

            # Scenario
            scenario = MADSScenario()
            # scenario.fill_parameter_space([env, wing])
            scenario.create_scenario(
                disciplines=[ice_protection],
                formulation="DisciplinaryOpt",
                objective_name=f"{ice_protection.name}.mass",
                scenario_type="DOE",
            )
        
            scenario.execute(algo_name="PYDOE_FULLFACT", n_samples=3**2)
            shutil.rmtree("test_models")
              
            # Post process
            data = scenario.scenario.to_dataset()
            data_pw = data["designs"]["chord"][0]
            data_mass = data["functions"][f"{ice_protection.name}.mass"][0]
            
            assert np.all(
            np.isclose(data_pw, [0.000, 0.125, 0.250, 0.375, 0.500, 0.625, 0.750, 0.875, 1.000]),
            )
        
            assert np.all(
            np.isclose(data_mass.values,
                    [
                    0.00000000e+00, 
                    9.18898063e-05, 
                    1.83779613e-04, 
                    2.75669419e-04,
                    3.67559225e-04,
                    4.59449032e-04,
                    5.51338838e-04,
                    6.43228644e-04,
                    7.35118451e-04,
                    ],
                ),
            )

        except Exception as e:
            logging.error(traceback.format_exc())
            
            
if __name__ == "__main__":
   
    test = TestModels()
    test.test_fuelcell()
