import logging
import os, sys

sys.path.insert(
    0,
    os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")),
)

import gemseo
import numpy as np

from assembly import Environment, Section, Span, AirfoilFile, Wing, Assembly
from disciplines.aerodynamics import Aerodynamics
from disciplines.disciplines import UserDefined
from disciplines.static_stability import StaticStability
from scenario import MADSScenario, Variable
import utilities.units as unit


# Logger
gemseo.configure_logger(
    # level=logging.INFO,
    level=logging.DEBUG,
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
weight_req = 32000.0  # weight required
wing_span = 13.5  # fixed wing semi-span
A_ref = 13.5  # reference area
l_ref = 3.5  # reference length

# Optimization variables
chord_tip = Variable("chord_tip", 1.5, lb=0.5, ub=3.0)
span_0 = Variable("span_0", 4.7, lb=2.0, ub=10.0)
xc_ref = Variable("xc_ref", 0.5, lb=0.0, ub=1.0)
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
    options={
        "dust": {
            "start_res": 90,
            "end_res": 100,
        }
    },
)
drag = Variable(
    name="D",
    value=10000.0,
    output_type="drag",
    options={
        "dust": {
            "start_res": 90,
            "end_res": 100,
            "components": ["wing"],
        }
    },
)
dCm = Variable(
    name="dCm",
    value=0.5,
    output_type="dCm",
    options={
        "dust": {
            "start_res": 90,
            "end_res": 100,
            "components": ["wing"],
        },
        "parameters": {
            "A_ref": A_ref,
            "l_ref": l_ref,
        },
    },
)

# All variables
variables = [
    # Optimization
    chord_tip,
    span_0,
    xc_ref,
    angle_root,
    angle_tip,
    # Internal
    span_1,
    lift,
    drag,
    dCm,
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
    xc_ref=xc_ref,
    symmetry=True,
    options={
        "dust": {
            "n_elem": 25,
            "elem_type": "uniform",
        }
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

# Define assembly
assembly = Assembly(
    "system",
    [env, wing],
    variables,
)

# Aerodynamics discipline
disc_aerodynamics = Aerodynamics(
    "Aerodynamics",
    inputs=variables,
    outputs=[lift, drag],
    components=assembly.components,
    solver_name="dust",
    options={
        "t_end": 0.5,
        "dt": 0.005,
        "n_threads": 30,
        "n_wake_particles": 100000,
        "particles_box_min": [-4.0, -15.0, -5.0],
        "particles_box_max": [6.0, 15.0, 5.0],
    },
)

# Aerodynamics discipline
disc_SnC = StaticStability(
    "Stability",
    inputs=variables,
    outputs=[dCm],
    components=assembly.components,
    solver_name="dust",
    options={
        "t_end": 0.5,
        "dt": 0.005,
        "n_threads": 30,
        "n_wake_particles": 100000,
        "particles_box_min": [-4.0, -15.0, -5.0],
        "particles_box_max": [6.0, 15.0, 5.0],
        "dAoA": 0.03,
    },
)

# Coupling discipline
disc_coupling = UserDefined(
    "Coupling",
    inputs=[span_0],
    outputs=[span_1],
    expression=f"{wing_span} - span_0",
)

disc_obj_fun2 = UserDefined(
    "Trimming",
    inputs=[lift],
    outputs=[lift_const],
    expression=f"(L - ({weight_req} * 9.81)) / ({weight_req} * 9.81)",
)

disc_obj_fun1 = UserDefined(
    "Efficiency",
    inputs=[lift, drag],
    outputs=[efficiency],
    expression="L / D",
)

constraints = [
    {"name": lift_const.name, "type": "eq"},
    {"name": dCm.name, "type": "ineq", "positive": True},
]

# Discipline list
disciplines = [
    disc_coupling,
    disc_aerodynamics,
    disc_SnC,
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

mads_scenario.add_observable(dCm.name)

mads_scenario.scenario.set_differentiation_method(
    "finite_differences",
    step=1e-3,
)

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
    variable_names=[dCm.name, span_0.name],
    save=True,
    show=False,
)
