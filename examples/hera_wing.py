import copy
import logging
from pathlib import Path

import gemseo
import numpy as np

import multiads.solvers.aerodynamics.neuralfoil as nf
import multiads.utilities.units as unit
from multiads.assembly import (
    AirfoilFile,
    Environment,
    Section,
    Span,
    Wing,
)
from multiads.disciplines import UserDefined
from multiads.disciplines.aerodynamics import Aerodynamics
from multiads.scenario import (
    Constraint,
    ConstraintType,
    MADSScenario,
    VariableFloat,
)
from multiads.solvers.aerodynamics import dust, dust_lib

# Logger
gemseo.configure_logger(
    level=logging.INFO,
    filename="gemseo.log",
    filemode="w",
)

# Parameters
altitude = 25000.0 * unit.ft
speed = 300.0 * unit.knot

chord_root = 3.0
wing_span = 13.5
weight_req = 32000.0

# %% Optimization variables
chord_tip = VariableFloat("chord_tip", 1.5, lb=0.5, ub=3.0)
span_0 = VariableFloat("span_0", 4.7, lb=2.0, ub=10.0)
xc_ref = VariableFloat("xc_ref", 0.5, lb=0.0, ub=1.0)
angle_root = VariableFloat("angle_root", 3.0, lb=0.0, ub=20.0)
angle_tip = VariableFloat("angle_tip", 0.0, lb=-10.0, ub=30.0)

variables = [chord_tip, span_0, xc_ref, angle_root, angle_tip]

# Inner variables
span_1 = VariableFloat("span_1", 8.8)
efficiency = VariableFloat("wing_efficiency", 0.0)
lift_const = VariableFloat("lift_const", 1.0)

# %% Components and assembly
env = Environment(name="env", height=altitude, speed=speed)

foil_file = Path(__file__).parent.parent / "assets" / "naca653218.dat"
root_section = Section(
    name="root_sec",
    airfoil=AirfoilFile(name="root_foil", filename=foil_file),
    chord=chord_root,
    twist=angle_root,
    options=[dust_lib.SectionOptions(polar=True)],
)
kink_section = Section(
    name="kink_sec",
    airfoil=AirfoilFile(name="kink_foil", filename=foil_file),
    chord=chord_root,
    twist=angle_root,
    options=[dust_lib.SectionOptions(polar=True)],
)
tip_section = Section(
    name="tip_sec",
    airfoil=AirfoilFile(name="tip_foil", filename=foil_file),
    chord=chord_tip,
    twist=angle_tip,
    options=[dust_lib.SectionOptions(polar=True)],
)

span_in = Span(
    name="span_in",
    length=span_0,
    sweep=0.0,
    dihed=0.0,
    options=[
        dust_lib.SpanOptions(panel_type=dust_lib.SpanPanelType.UNIFORM, num_panels=35),
    ],
)
span_out = Span(
    name="span_out",
    length=span_1,
    sweep=0.0,
    dihed=0.0,
    options=[
        dust_lib.SpanOptions(panel_type=dust_lib.SpanPanelType.UNIFORM, num_panels=35),
    ],
)

wing = Wing(
    name="wing",
    xc_ref=xc_ref,
    symmetry=True,
    sections=[root_section, kink_section, tip_section],
    spans=[span_in, span_out],
    options=[
        dust_lib.WingOptions(
            discretization_method=dust_lib.WingMethod.VORTEX_LATTICE,
            panel_type=dust_lib.WingPanelType.UNIFORM,
            num_panels=25,
            output_options=dust_lib.OutputOptions(
                compute_loads=True,
                loads_start=90,
                loads_end=100,
                loads_avg=True,
            ),
        ),
    ],
)

# %% Discipline> Geometric coupling
disc_geometry = UserDefined(
    name="Geometry",
    inputs=[span_0],
    outputs=[span_1],
    expression=f"{wing_span} - span_0",
)

# Discipline> Aerodynamic polars
nf_solver = nf.Neuralfoil(options=nf.Options())
disc_polars = Aerodynamics(name="Polars", components=[env, wing], solver=nf_solver)

# Discipline> Aerodynamics
dust_solver = dust.DUST(
    options=dust_lib.Options(
        t_end=0.5,
        dt=0.005,
        n_threads=30,
        n_wake_particles=100000,
        particles_box_min=np.array([-4.0, -15.0, -5.0]),
        particles_box_max=np.array([6.0, 15.0, 5.0]),
        output_options=dust_lib.OutputOptions(
            compute_loads=True,
            loads_start=90,
            loads_end=100,
            loads_avg=True,
        ),
    ),
)
disc_aerodynamics = Aerodynamics(
    name="Aerodynamics",
    components=[env, wing],
    solver=dust_solver,
)

# Discipline> Objectives and constraints
lift, drag = copy.deepcopy(dust_solver.get_output_variables(["lift", "wing.drag"]))


def obj_const(lift, drag):  # noqa: ANN001, ANN201
    eff = lift / drag
    const = (lift - weight_req * 9.81) / (weight_req * 9.81)
    return eff, const


disc_obj_const = UserDefined(
    name="Objectives & Constraints",
    inputs=[lift, drag],
    outputs=[efficiency, lift_const],
    expression=obj_const,
)
constraints = [
    Constraint(output_name=lift_const.name, constraint_type=ConstraintType.EQ),
]

# Discipline list
disciplines = [
    disc_geometry,
    disc_polars,
    disc_aerodynamics,
    disc_obj_const,
]

# %% Design space
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

mads_scenario.set_differentiation_method(
    "finite_differences",
    step=1e-3,
)

# %% Run
mads_scenario.xdsmize()

mads_scenario.execute(
    algo_name="NLOPT_COBYLA",
    max_iter=20,
    eq_tolerance=0.1,
    ineq_tolerance=0.1,
)

# %% Postprocess
out_file = "opti_results.h5"
mads_scenario.save_optimization_history(out_file)

gemseo.execute_post(out_file, post_name="OptHistoryView", save=True, show=False)
gemseo.execute_post(
    out_file,
    post_name="ConstraintsHistory",
    constraint_names=[lift_const.name],
    save=True,
    show=False,
)
gemseo.execute_post(out_file, post_name="ObjConstrHist", save=True, show=False)
gemseo.execute_post(out_file, post_name="RadarChart", save=True, show=False)
gemseo.execute_post(out_file, post_name="ParallelCoordinates", save=True, show=False)
