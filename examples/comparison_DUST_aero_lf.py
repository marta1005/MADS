import logging
import pickle
import os, sys

sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
)
sys.path.insert(
    0,
    os.path.abspath(os.path.join(os.path.dirname(__file__), "../../pyxrotor")),
)
sys.path.insert(
    0,
    os.path.abspath(
        os.path.join(
            os.path.dirname(__file__), "../src/solvers/weight_and_balance/ampet"
        )
    ),
)

from utilities.gemseo_utilities import *

import gemseo
import numpy as np

from gemseo import create_surrogate

from scenario import MADSScenario, Variable
from disciplines.aerodynamics import Aerodynamics
from disciplines.disciplines import UserDefined

from assembly import (
    Wing,
    Section,
    Span,
    AirfoilFile,
    Assembly,
    Environment,
)

# Logger
gemseo.configure_logger(
    level=logging.INFO,
    filename="gemseo.log",
    filemode="w",
)

# settings
n_samples = 200

# Requirements
airspeed_design = 160 / 3.6  # m/s

# Parameters
chord_root = 3.0
wing_span = 13.5

# Variables
alpha = Variable(
    "alpha",
    1.0,
    lb=-3.0,
    ub=10.0,
)   # [deg]

# Aerodynamic assumptions
K_polhamus = 1.0  # [-]
oswald_factor = 1 / 1.02

# Assumptions / inputs for mission analysis
altitude_cruise = 6000 * 0.3048  # m

# Dust Variables
nPanelsInner = 20  # 35
nPanelsOuter = 20  # 35
nPanelsTotal = int(2 * (nPanelsInner + nPanelsOuter))
nPanelsChord = 15  # 25
dustDeltaT = 0.005
dustNTempSteps = 100  # 100
dustNStep0Calc = 90  # 90
dustTf = dustDeltaT * dustNTempSteps

# Optimization variables
chord_tip = Variable("chord_tip", 1.5, lb=0.5, ub=3.0)
span_0 = Variable("span_0", 4.7, lb=2.0, ub=10.0)
angle_root = Variable("angle_root", 3.0, lb=0.0, ub=20.0)
angle_tip = Variable("angle_tip", 0.0, lb=-10.0, ub=30.0)

# Coupled variables
span_1 = Variable("span_1", 8.8)

# Output variables
#cD0 = Variable(
#    "cD0",
#    0.0,
#    output_type="cd0",
#    options={"component_buildup_drag": {"associated_component_names": ["wing"]}},
#)
### new for comparison of aero_lf and DUST (DUST does not consider friction drag)
cD0 = 0.0

cL = Variable(
    "cL",
    np.arange(0, 1.3, 0.1),
    output_type="cl",
    options={"aero_lf": {"wings": ["wing"]}},
)
cD = Variable(
    "cD",
    np.arange(0, 1.3, 0.1),
    output_type="cd",
    options={"aero_lf": {"wings": ["wing"]}},
)
lift = Variable(
    "lift",
    100.0,
    output_type="lift",
    options={
        "dust": {
            "start_res": dustNStep0Calc,
            "end_res": dustNTempSteps,
            "components": ["wing"],
        }
    },
)
drag = Variable(
    "drag",
    100.0,
    output_type="drag",
    options={
        "dust": {
            "start_res": dustNStep0Calc,
            "end_res": dustNTempSteps,
            "components": ["wing"],
        }
    },
)

m_y = Variable(
    "m_y",
    100.0,
    output_type="my",
    options={
        "dust": {
            "start_res": dustNStep0Calc,
            "end_res": dustNTempSteps,
            "components": ["wing"],
        }
    },
)

m_y_lf = Variable(
    "m_y_lf",
    10.0,
    output_type="my",
    options={"aero_lf": {"wings": ["wing"]}},
)

spanload = Variable(
    name="spanload",
    value=np.zeros((5, nPanelsTotal)).flatten(),
    output_type="span_load",
    options={
        "dust": {
            "start_res": dustNStep0Calc,
            "end_res": dustNTempSteps,
            "components": ["wing"],
        },
        "aero_lf": {
            "wings": ["wing"],
        },
    },
)

spanload_lf = Variable(
    name="spanload_lf",
    value=np.zeros((5, nPanelsTotal)).flatten(),
    output_type="span_load",
    options={
        "aero_lf": {
            "wings": ["wing"],
        },
    },
)

lift_lf = Variable(
    "lift_lf",
    10.0,
    output_type="lift",
    options={"aero_lf": {"wings": ["wing"]}},
)
drag_lf = Variable(
    "drag_lf",
    10.0,
    output_type="drag",
    options={"aero_lf": {"wings": ["wing"]}},
)

# All variables
variables = [
    # Optimization
    alpha,
    chord_tip,
    span_0,
    angle_root,
    angle_tip,
    # Internal
    span_1,
    cL,
    cD,
    #cD0,
    lift,
    drag,
    spanload,
    spanload_lf,
    lift_lf,
    drag_lf,
    m_y_lf,
    m_y,
]

# Components and assembly
env = Environment(
    "env",
    height=altitude_cruise,
    alpha=0.0,
    beta=0.0,
    speed=airspeed_design,
)

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
    xc_ref=0.25,
    symmetry=True,
    alpha=alpha,
    portion_laminar_flow=0.1,
    cd0=cD0,
    options={
        "dust": {
            "n_elem": nPanelsChord,
            "elem_type": "uniform",
        },
        "aero_lf": {
            "k2": 1.0,  # initial guess for the  k2-factor in the aerodynamic drag polar [-]
            "k_polhamus": 1.0,  # Airfoil/Pressure impact [-]
            "oswald_factor": oswald_factor,
        },
        "component_buildup_drag": {
            "surface_material": "camouflage_paint_on_aluminum",
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
                },
                "aero_lf": {
                    "n_elem": 35,
                    "elem_type": "uniform",
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
                },
                "aero_lf": {
                    "n_elem": 35,
                    "elem_type": "uniform",
                }
            },
        ),
    ],
)

assembly = Assembly(
    "system",
    [
        wing,
        env,
    ],
    variables,
)

# Coupling discipline
disc_coupling = UserDefined(
    "Coupling",
    inputs=[span_0],
    outputs=[span_1],
    expression=f"{wing_span} - span_0",
)

disc_aerodynamics = Aerodynamics(
    "Aerodynamics (wings)",
    inputs=variables,
    outputs=[
        cL,
        cD,
        lift_lf,
        drag_lf,
        spanload_lf,
        m_y_lf,
    ],
    components=[wing, env],
    solver_name="aero_lf",
)

#disc_drag = Aerodynamics(
#    "Aerodynamics (drag)",
#    inputs=variables,
#    outputs=[cD0],
#    components=assembly.components,
#    solver_name="component_buildup_drag",
#    options={"wing_name_serving_for_reference_area": "wing"},
#)

disc_dust = Aerodynamics(
    "Aerodynamics (DUST)",
    inputs=variables,
    outputs=[
        lift,
        drag,
        spanload,
        m_y,
    ],
    components=assembly.components,
    solver_name="dust",
    options={
        "t_end": dustTf,
        "dt": dustDeltaT,
        "n_threads": 30,
        "n_wake_particles": 100000,
        "particles_box_min": [-4.0, -15.0, -5.0],
        "particles_box_max": [6.0, 15.0, 5.0],
        "box_length": 10.0,
        "n_box": [1, 3, 1 ],
        "octree_origin": [-4.0, -15.0, -5.0],
        "n_octree_levels": 2, #5, #--> 2 faster,
        "min_octree_part": 2, #5, #--> 2 faster,
        "multipole_degree": 2, #5, #--> 2 faster,

    },
)

disciplines = [
    disc_coupling, #### TK how does this work with surroate DoE process?
    #disc_drag,
    disc_aerodynamics,
    #disc_dust,
]

# Parameter space
mads_scenario = MADSScenario()
mads_scenario.fill_parameter_space(variables)

# Scenario
mads_scenario.create_scenario(
    disciplines,
    formulation='DisciplinaryOpt',
    # objective_name=["cL", "lift_lf", "cD", "drag_lf", "cD0", "lift", "drag", "spanload"],
    #objective_name=["cL", "lift_lf", "cD", "drag_lf", "cD0", "spanload_lf"],
    objective_name=["cL"],
    scenario_type="DOE",
)

mads_scenario.scenario.add_observable("lift_lf")
mads_scenario.scenario.add_observable("cD")
mads_scenario.scenario.add_observable("drag_lf")
mads_scenario.scenario.add_observable("lift")
mads_scenario.scenario.add_observable("drag")
mads_scenario.scenario.add_observable("spanload_lf")
mads_scenario.scenario.add_observable("spanload")
mads_scenario.scenario.add_observable("m_y")
mads_scenario.scenario.add_observable("m_y_lf")

mads_scenario.scenario.execute(
    input_data={
        "algo": "LHS",
        "n_samples": n_samples,
    }
)
mission_dataset = mads_scenario.scenario.to_dataset(opt_naming=False)

# save dataset to csv
mission_dataset.to_csv(path_or_buf="./dataset_export_DUST_aero_lf.csv")
