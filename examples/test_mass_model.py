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

import gemseo
import numpy as np

from assembly import (
    Environment,
    Section,
    Span,
    AirfoilFile,
    Wing,
    Configuration,
    Assembly,
)
from disciplines.aerodynamics import Aerodynamics
from disciplines.weight_and_balance import WeightAndBalance
from disciplines.disciplines import UserDefined
from scenario import MADSScenario, Variable
import utilities.units as unit


# Logger
gemseo.configure_logger(
    level=logging.INFO,
    # level = logging.DEBUG,
    filename="gemseo.log",
    filemode="w",
)

# Parameters
altitude = 25000.0 * unit.ft
speed = 154.835
alpha = 0.0
beta = 0.0

chord_root = 3.0

# Reference values
design_mass_TOGW = 32000.0
weight_str = 30500.0
wing_span = 13.5

# Optimization variables
chord_tip = Variable("chord_tip", 1.5, lb=0.5, ub=3.0)
span_0 = Variable("span_0", 4.7, lb=2.0, ub=10.0)
angle_root = Variable("angle_root", 3.0, lb=0.0, ub=20.0)
angle_tip = Variable("angle_tip", 0.0, lb=-10.0, ub=30.0)

# Coupled variables
span_1 = Variable("span_1", 8.8)
efficiency = Variable(name="Eff", value=50.0)
lift_const = Variable(name="Lift_const", value=1.0)

# Output variables
lift = Variable(
    name="L",
    value=300000.0,
    output_type="lift",
)
drag = Variable(
    name="D",
    value=10000.0,
    output_type="drag",
)
wing_mass = Variable(
    name="mass_wing",
    value=1500.0,
    output_type="mass_wing",
)

# All variables
variables = [
    # Optimization
    chord_tip,
    span_0,
    angle_root,
    angle_tip,
    # Internal
    span_1,
    lift,
    drag,
    wing_mass,
]

# Components and assembly
env = Environment("env", altitude, speed, alpha, beta)

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

wing = Wing(
    name="wing",
    fuel_mass_in_wing=800.0,
    symmetry=True,
    options={
        "dust": {
            "n_elem": 25,
            "elem_type": "uniform",
        },
        "aero_lf": {
            "cd0": 0.01,
            "k2": 0.0,
            "oswald_factor": 1 / 1.02,
        },
    },
    sections=[
        root_section,
        root_section,
        tip_section,
    ],
    spans=[
        Span(
            "span_in",
            length=span_0,
            sweep=0.0,
            dihed=0.0,
            options={
                "dust": {
                    "n_elem": 35,
                }
            },
        ),
        Span(
            "span_out",
            length=span_1,
            sweep=0.0,
            dihed=0.0,
            options={
                "dust": {
                    "n_elem": 35,
                    "elem_type": "uniform",
                }
            },
        ),
    ],
)

configuration = Configuration(
    "configuration",
    m_tom=design_mass_TOGW,
    ultimate_load_factor=2.0,
)

# Define assembly
assembly = Assembly(
    "system",
    [env, wing, configuration],
    variables,
)

# Aerodynamics discipline
disc_aerodynamics = Aerodynamics(
    "Aerodynamics",
    inputs=variables,
    outputs=[lift, drag],
    components=assembly.components,
    solver_name="aero_lf",
    options={"k_polhamus": 0.4},
)

# Mass estimation discipline
disc_mass = WeightAndBalance(
    "Weight and Balance",
    inputs=variables,
    outputs=[wing_mass],
    components=assembly.components,
    solver_name="raymer_general_aviation_mass_wing",
    options={},
)


# Coupling discipline
disc_coupling = UserDefined(
    "Coupling",
    inputs=[span_0],
    outputs=[span_1],
    expression=f"{wing_span} - span_0",
)


def Trimming(L, wing_mass):
    total_weight = (wing_mass + weight_str) * 9.81
    return (L - total_weight) / total_weight


disc_obj_fun2 = UserDefined(
    "Trimming",
    inputs=[lift, wing_mass],
    outputs=[lift_const],
    expression=Trimming,
)

disc_obj_fun1 = UserDefined(
    "Efficiency",
    inputs=[lift, drag],
    outputs=[efficiency],
    expression="L / D",
)

constraints = [{"name": lift_const.name, "type": "eq"}]

# discipline list
disciplines = [
    disc_coupling,
    disc_aerodynamics,
    disc_mass,
    disc_obj_fun1,
    disc_obj_fun2,
]

# Design space
mads_scenario = MADSScenario()
mads_scenario.fill_parameter_space(variables)

# Scenario
mads_scenario.create_scenario(
    disciplines,
    formulation="DisciplinaryOpt",
    objective_name=efficiency.name,
    maximize_objective=True,
)
mads_scenario.add_constraints(constraints)

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
        "max_iter": 10,
        "algo": "SLSQP",
        "algo_options": {
            "eq_tolerance": 0.1,
            "ineq_tolerance": 0.1,
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
    constraint_names=[lift_const.name],
    save=True,
    show=False,
)
execute_post(to_be_postprocess, "ObjConstrHist", save=True, show=False)
execute_post(to_be_postprocess, "ObjConstrHist", save=True, show=False)
execute_post(to_be_postprocess, "RadarChart", save=True, show=False)
execute_post(to_be_postprocess, "GradientSensitivity", save=True, show=False)
execute_post(to_be_postprocess, "ParallelCoordinates", save=True, show=False)
execute_post(
    to_be_postprocess,
    "BasicHistory",
    variable_names=[span_0.name],
    save=True,
    show=False,
)
