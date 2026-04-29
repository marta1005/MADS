import logging


import logging
import os, sys
import gemseo
import numpy as np

# Path to the src directory
sys.path.insert(
    0,
    os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")),
)

sys.path.insert(
    0,
    os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")),
)
sys.path.insert(
    0,
    os.path.abspath(os.path.join(os.path.dirname(__file__), "../../pyxrotor")),
)
sys.path.insert(
    0,
    os.path.abspath(os.path.join(os.path.dirname(__file__), "../src/solvers/thermal")),
)

sys.path.insert(
    0,
    os.path.abspath(
        os.path.join(
            os.path.dirname(__file__), "../src/solvers/weight_and_balance/ampet"
        )
    ),
)

import gemseo
import numpy as np
from gemseo.algos.opt.scipy_local.settings.slsqp import SLSQP_Settings

# from gemseo.formulations.disciplinary_opt_settings import DisciplinaryOpt_Settings
from gemseo.formulations.mdf_settings import MDF_Settings


from scenario import MADSScenario, Variable

from disciplines.power_supply import PowerSupply


from assembly import Assembly, Battery, Environment

import utilities.units as unit


## SM6
# -----------------------------------------------------
# from objectives.mads_objectives import mads_objectives

from disciplines.disciplines import UserDefined


# Logger
gemseo.configure_logger(
    level=logging.INFO,
    # level=logging.DEBUG,
    filename="gemseo.log",
    filemode="w",
)

# Parameters
altitude = 25000.0 * unit.ft
speed = 154.835
alpha = 0.0
beta = 0.0

chord_root = 3.0

# Parameters
# density = 0.54895
# pressure = 37601
# sound = 309.67
# viscosity = 1.554e-8
# velocity_ampl = 154.835
# alpha_global = 0.0
# beta_global = 0.0

chord_root = 3.0

r_tip_in = 1.5
n_blades_in = 6
rpm_in = 1200.0
pitch_in = 43.0

r_tip_out = 1.5
n_blades_out = 3
rpm_out = 1200.0
pitch_out = 43.0

# reference values
design_mass_TOGW = 32000.0  # weight required
wing_mass = 1500.0
weight_str = design_mass_TOGW - wing_mass
thrust_req = 18000.0  # Thrust Required
A_ref = 13.5  # Area reference
l_ref = 3.5  # reference len

# environment
env = Environment("env", altitude, speed, alpha, beta)


# battery
batt_power = Variable("batt_power", np.array([300.0]), lb=5.0, ub=500.0)
batt_nominal_voltage = Variable(
    "batt_nominal_voltage", np.array([540.0]), lb=28.0, ub=540.0
)
flight_time = Variable("flight_time", np.array([40.0]), lb=10.0, ub=100.0)


batt_weight = Variable(
    "BATT_WEIGHT",
    value=np.nan,  # np.array([900.0]),
    output_type="batt_weight",
)


batt_volume = Variable(
    "BATT_VOL",
    value=np.nan,  # np.array([700.0]),
    output_type="batt_volume",
)

batt_capacity = Variable(
    "BATT_CAP",
    value=np.array(
        [
            np.nan,  # 0.0,
            np.nan,  # 18.52,
            np.nan,  # 55.56,
            np.nan,  # 83.33,
            np.nan,  # 111.1,
            np.nan,  # 138.9,
            np.nan,  # 166.7,
            np.nan,  # 194.4,
            np.nan,  # 222.2,
            np.nan,  # 250.0,
            np.nan,  # 277.8,
            np.nan,  # 305.6,
            np.nan,  # 333.3,
            np.nan,  # 361.1,
            np.nan,  # 370.4
        ]
    ),
    output_type="capacity",
)

batt_voltage = Variable(
    "BATT_VOLT",
    value=np.array(
        [
            np.nan,  # 624.0,
            np.nan,  # 600.0,
            np.nan,  # 594.0,
            np.nan,  # 588.0,
            np.nan,  # 576.0,
            np.nan,  # 564.0,
            np.nan,  # 552.0,
            np.nan,  # 540.0,
            np.nan,  # 528.0,
            np.nan,  # 516.0,
            np.nan,  # 504.0,
            np.nan,  # 492.0,
            np.nan,  # 480.0,
            np.nan,  # 468.0,
            np.nan,  # 408.0
        ]
    ),
    output_type="voltage",
)


# reference parameters
design_mass_TOGW = Variable("weight_req", np.array([weight_str + wing_mass]))


def Coupling(span_0, xc_ref):
    span_1 = 13.5 - span_0
    return span_1


# define new coupling
def Coupling_AR(chord_tip, chord_root, span_0, span_1):
    AR = (span_0 + span_1) / (chord_tip + chord_root) / 2
    return AR


def Coupling_Weight(wing_mass):
    weight_req = 32000.0 - wing_mass
    return weight_req


# mass
# wing_mass = Variable(
#     name="Mass_wing", value=np.array([wing_mass]), output_type="Mass_wing"
# )


# All variables
variables = {
    batt_power,
    batt_nominal_voltage,
    flight_time,
}


battery = Battery(
    name="battery",
    batt_power=batt_power.value,
    batt_nominal_voltage=batt_nominal_voltage.value,
    batt_flight_time=flight_time,  # VariableMarker("batt_flight_time"),
)


# Define assembly
assembly = Assembly(
    "system",
    [env, battery],
    variables,
)

# Battery discipline
battery_outputs = {
    "batt_weight": batt_weight,
    "batt_volume": batt_volume,
    "batt_capacity_output": batt_capacity,
    "batt_voltage_output": batt_voltage,
}

disc_battery = PowerSupply(
    "Battery",
    inputs=variables,
    outputs=[batt_weight, batt_volume, batt_capacity],  # [battery_outputs],
    components=assembly.components,
    solver_name="battery_fmu",
)

constraints = [
    {"name": batt_volume.name, "type": "ineq", "value": 300.0},
    #    {"name": batt_capacity.name, "type": "eq"},
]


# discipline list
disciplines = [
    # disc_coupling,
    # disc_aerodynamics,
    disc_battery,
    # disc_mass,
    # disc_obj_fun1,
    # disc_obj_fun2,
    # disc_obj_fun3,
]

# Design space
mads_scenario = MADSScenario()
mads_scenario.fill_parameter_space(variables)
# objective_name = {"Eff", "Trimming", "inv_weight"}

# Scenario
mads_scenario.create_scenario(
    disciplines=disciplines,
    formulation=MDF_Settings(main_mda_name="MDAGaussSeidel"),
    objective_name="BATT_VOL",
    maximize_objective=False,
)
mads_scenario.add_constraints(constraints)
# mads_scenario.add_constraints(constraint2)


mads_scenario.scenario.set_differentiation_method(
    "finite_differences",
    step=1e-3,
)

# preprocess architecture
mads_scenario.scenario.xdsmize(show_html=False)
if len(disciplines) > 1:
    gemseo.generate_n2_plot(disciplines, save=True, show=False)

# set optimization inputs back up
mads_scenario.scenario.set_optimization_history_backup("my_data.h5")

slsqp_settings = SLSQP_Settings(max_iter=10, eq_tolerance=0.1, ineq_tolerance=0.1)
mads_scenario.scenario.execute(slsqp_settings)


# Database
database = mads_scenario.scenario.formulation.optimization_problem.database

# ! Postprocess
from gemseo import execute_post

# postprocess
out_file = "opti_results.hdf5"
mads_scenario.scenario.save_optimization_history(out_file)
# to_be_postprocess = out_file

# execute postprocess
execute_post(mads_scenario.scenario, post_name="OptHistoryView", save=True, show=False)
execute_post(
    mads_scenario.scenario,
    post_name="ConstraintsHistory",
    constraint_names=["BATT_VOL"],
    save=True,
    show=False,
)
execute_post(mads_scenario.scenario, post_name="ObjConstrHist", save=True, show=False)
execute_post(mads_scenario.scenario, post_name="RadarChart", save=True, show=False)
execute_post(
    mads_scenario.scenario, post_name="GradientSensitivity", save=True, show=False
)
execute_post(
    mads_scenario.scenario, post_name="ParallelCoordinates", save=True, show=False
)
execute_post(
    mads_scenario.scenario,
    post_name="BasicHistory",
    variable_names=["flight_time"],
    save=True,
    show=False,
)
