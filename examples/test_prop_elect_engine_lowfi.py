import logging
import os, sys

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

import gemseo
import numpy as np

from assembly import (
    Environment,
    Section,
    Span,
    AirfoilNACA4,
    AirfoilFile,
    Wing,
    Propeller,
    Assembly,
    ThermalSystem,
    PropulsionSystem,
)
from disciplines.aerodynamics import Aerodynamics
from disciplines.propulsion import Propulsion
from disciplines.thermal import Thermal
from disciplines.disciplines import UserDefined
from scenario import MADSScenario, Variable
import utilities.units as unit

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

# propulsion specifications
rpm_ref = 2300.0  # rpm
r_tip_in = 1.5
n_blades_in = 8
rpm_in = 1200.0
pitch_in = 60  # 43.0

r_tip_out = 1.5
n_blades_out = 3
rpm_out = 1200.0
pitch_out = 43.0


# Environment here to compute atmosphere parameters
env = Environment("env", altitude, speed, alpha, beta)

# reference values
weight_req = 32000.0
thrust_req = 18000.0
wing_span = 13.5

# engine power
engine_power = 3 * 10**9  # Watt

# Optimization variables
chord_tip = Variable("chord_tip", 1.5, lb=0.5, ub=3.0)
span_0 = Variable("span_0", 4.7, lb=2.0, ub=10.0)
xc_ref = Variable("xc_ref", 0.5, lb=0.0, ub=1.0)
angle_root = Variable("angle_root", 3.0, lb=0.0, ub=20.0)
angle_tip = Variable("angle_tip", 0.0, lb=-10.0, ub=30.0)

# Coupled variables
span_1 = Variable("span_1", 8.8, lb=5.0, ub=10.0)
# efficiency = Variable(name="Eff", value=50.0)

# -----------------------------------------------------------------------
# define Prop Variables
# -----------------------------------------------------------------------
# define rpm as design variale
rpm = Variable("rpm", 1900.0, lb=1000.0, ub=3000.0)
thrust_trim = Variable("thrust_trim", 150000.0, lb=10000.0, ub=30000.0)
eff_prop_target = Variable("eff_prop_target", 0.5, lb=0.0, ub=1.0)

# Output variable for propulsion
thrust = Variable(
    name="thrust",
    value=12000.0,
    lb=5000.0,
    ub=30000.0,
    output_type="thrust",
)
prop_efficiency = Variable(
    name="prop_efficiency",
    value=0.8,
    lb=0.3,
    ub=0.99,
    output_type="prop_efficiency",
)

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

prop_0 = Propeller(
    name="propeller",
    blade=blade,
    r_tip=r_tip_in,
    n_blades=n_blades_in,
    pitch=pitch_in,
    rpm=rpm,
    thrust=thrust,
    options={
        "prop_lf": {
            "altitude": 20000.0,
        },
    },
)

# -----------------------------------------------------------------------
# define ECS Variables
# -----------------------------------------------------------------------
ecs_mass = Variable(
    "ecs_mass",
    15.0,
    output_type="ecs_mass",
    lb=0.0,
    ub=300.0,
    options={"ecs_lf": {"ecs": ["ecs_0"]}},
)
ecs_drag = Variable(
    "RAM_drag",
    15.0,
    output_type="RAM_drag",
    lb=0.0,
    ub=300.0,
    options={"ecs_lf": {"ecs": ["ecs_0"]}},
)

ecs_power = Variable(
    "ecs_power",
    150.0,
    output_type="ecs_power",
    lb=0.0,
    ub=30000.0,
    options={"ecs_lf": {"ecs": ["ecs_0"]}},
)

ecs_0 = ThermalSystem(
    name="ecs_0",
    fuel_tank_volume=37820,
    Q_total=15000,
    ecs_mass=ecs_mass,
    RAM_drag=ecs_drag,
    ecs_power=ecs_power,
    options={},
)

# -------------------------------------------------------------


# -----------------------------------------------------------------------
# define eelctric engine Variables
# -----------------------------------------------------------------------


# define rpm as design variale
electric_engine_power = Variable(
    "electric_engine_power",
    2000.0,
    lb=1000.0,
    ub=3000.0,
    output_type="electric_engine_power",
)
shaft_power = Variable(
    "shaft_power",
    200.0,
    lb=100.0,
    ub=500.0,
    output_type="shaft_power",
)

engine1 = PropulsionSystem(
    "E1",
    type="Hybrid",
    engine_power=engine_power,
    shaft_power=shaft_power,
    electric_engine_power=electric_engine_power,
    options={
        "e_eng_lf": {},
    },
)
engine2 = PropulsionSystem(
    "E2",
    type="Hybrid",
    engine_power=engine_power,
    shaft_power=shaft_power,
    electric_engine_power=electric_engine_power,
    options={
        "e_eng_lf": {},
    },
)

# -------------------------------------------

# All variables that are included in the design space - XROTOR
variables = [
    # Optimization
    chord_tip,
    span_0,
    xc_ref,
    angle_root,
    angle_tip,
    ecs_drag,
    ecs_mass,
    # Internal
    span_1,
    thrust,
    rpm,
    shaft_power,
    electric_engine_power,
]

# Components and assembly
root_section = Section(
    "root_sec",
    airfoil=AirfoilFile("root_foil", filename="naca653218.dat"),
    chord=chord_root,
    twist=angle_root,
)
tip_section = Section(
    "tip_sec",
    airfoil=AirfoilFile("tip_foil", filename="naca653218.dat"),
    chord=chord_tip,
    twist=angle_tip,
)

# Define assembly
# --------------------------------------------------------------------
assembly = Assembly(
    "system",
    [env, prop_0, ecs_0, engine1, engine2],
    variables,
)


# Propeller - Propulsion discipline: Propeller and Engine
# -----------------------------------------------------------
disc_propeller = Propulsion(
    "Propeller",
    inputs=variables,
    outputs=[prop_efficiency, shaft_power],
    components=assembly.components,
    solver_name="propeller_low_fidelity",
    options={},
)

disc_powertrain = Propulsion(
    "PowerTrain",
    inputs=variables,
    outputs=[electric_engine_power],
    components=assembly.components,
    solver_name="electric_engine_low_fidelity",
)

# -----------------------------------------------------------

# ECS - environment control
# -----------------------------------------------------------
disc_ecs = Thermal(
    "ECS",
    inputs=variables,
    outputs=[ecs_drag, ecs_mass, ecs_power],
    components=assembly.components,
    solver_name="ecs_low_fidelity",
    options={},
)

# -----------------------------------------------------------
# define pitch angle
sqrt_prop_pitch = np.sqrt(43)

# introdue relation between rpm and thrust
disc_obj_coupling = UserDefined(
    "Thrust_rpm_coupling",
    inputs=[rpm],
    outputs=[thrust],
    expression=f"(4.392*10**-8 * rpm * (2*{r_tip_in})**3.5 / {sqrt_prop_pitch} * (4.3*10**-4 * rpm * {sqrt_prop_pitch}**2 -{speed} ))",
)

# -----------------------------------------------------

disc_obj_fun3 = UserDefined(
    "Thrust trim",
    inputs=[thrust, ecs_drag],
    outputs=[thrust_trim],
    expression=f"(thrust - {thrust_req} - RAM_drag) / {thrust_req}",
)

disc_obj_fun4 = UserDefined(
    "Efficiency prop target",
    inputs=[prop_efficiency],
    outputs=[eff_prop_target],
    expression="prop_efficiency - 0.8",
)

constraints = [
    {"name": eff_prop_target.name, "type": "ineq"},
    {"name": thrust_trim.name, "type": "eq"},
]

# Discipline list
disciplines = [
    disc_propeller,
    disc_powertrain,
    disc_ecs,
    disc_obj_fun3,
    disc_obj_fun4,
    disc_obj_coupling,
]

# Design space
mads_scenario = MADSScenario()
mads_scenario.fill_parameter_space(variables)

# Scenario
mads_scenario.create_scenario(
    disciplines,
    formulation="DisciplinaryOpt",
    scenario_type="DOE",
    objective_name=prop_efficiency.name,
    maximize_objective=True,
)
mads_scenario.add_constraints(constraints)

# ECS add observable
mads_scenario.add_observable(ecs_power.name)
mads_scenario.add_observable(ecs_drag.name)
mads_scenario.add_observable(thrust.name)

mads_scenario.scenario.set_differentiation_method(
    "finite_differences",
    step=1e-3,
)

# preprocess architecture
mads_scenario.scenario.xdsmize(show_html=False)
gemseo.generate_n2_plot(disciplines, save=True, show=False)

# set optimization inputs back up
# mads_scenario.scenario.set_optimization_history_backup("my_data.h5")

mads_scenario.scenario.execute(
    input_data={
        # "max_iter": 30,
        "n_samples": 30,
        "algo": "LHS",  # "SLSQP",
        "algo_options": {
            "eq_tolerance": 0.5,
            "ineq_tolerance": 0.5,
        },
    }
)

# Database
database = mads_scenario.scenario.formulation.opt_problem.database

# ! Postprocess
from gemseo import execute_post

# postprocess
out_file = "opti_results.hdf5"
mads_scenario.scenario.save_optimization_history(out_file)
to_be_postprocess = out_file

# execute postprocess
execute_post(to_be_postprocess, "OptHistoryView", save=True, show=False)
execute_post(
    to_be_postprocess,
    "ConstraintsHistory",
    constraint_names=[thrust_trim.name],
    save=True,
    show=False,
)
execute_post(
    to_be_postprocess,
    "BasicHistory",
    variable_names=[rpm.name, thrust_trim.name],
    save=True,
    show=False,
)


execute_post(to_be_postprocess, "ObjConstrHist", save=True, show=False)
execute_post(to_be_postprocess, "Correlations", save=True, show=False)
execute_post(to_be_postprocess, "ObjConstrHist", save=True, show=False)
execute_post(
    to_be_postprocess,
    "RadarChart",
    constraint_names=[thrust_trim.name, eff_prop_target.name],
    save=True,
    show=False,
)

# only with optimisation
# execute_post(to_be_postprocess, "GradientSensitivity", save=True, show=False)


from gemseo.post.dataset.scatter_plot_matrix import ScatterMatrix

# execute_post(to_be_postprocess, "ParallelCoordinates", save=True, show=False)

from gemseo.algos.database import Database
from gemseo.datasets.dataset import Dataset

database = Database.from_hdf(out_file)
names = database.get_function_names()
hist = database.get_history(function_names=())

with open("output.txt", "w") as txt_file:
    for line in names:
        # txt_file.write(" ".join(line) + "\n") #
        txt_file.write(f"{line} ")  #
    txt_file.write(f"\n")  #
with open("output.txt", "ab") as txt_file:
    np.savetxt(txt_file, hist[0], newline="\n")
# database
# dataset = Dataset()
# dataset.get_view(variable_names=[thrust.name, rpm.name, ecs_drag.name], components=[0])
# ScatterMatrix(dataset, classifier="specy", kde=True).execute(save=True, show=False)

execute_post(
    to_be_postprocess,
    "ScatterPlotMatrix",
    variable_names=[thrust.name, rpm.name, ecs_drag.name, prop_efficiency.name],
    save=False,
    show=True,
    kde=True,
)


# import pandas as pd
# import sys

# fpath = sys.argv[1]
# if len(sys.argv) > 2:
#    key = sys.argv[2]
#    df = pd.read_hdf(out_file, key=key)
# else:
#    df = pd.read_hdf(out_file)
#
# df.to_csv(sys.stdout, index=False)
