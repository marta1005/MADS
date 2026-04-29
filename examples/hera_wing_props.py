import copy
import logging
from pathlib import Path

import gemseo
import numpy as np

import multiads.solvers.aerodynamics.neuralfoil as nf
import multiads.utilities.units as unit
from multiads.assembly import (
    AirfoilFile,
    AirfoilNACA4,
    Blade,
    Environment,
    Propeller,
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
    VariableFloatNP,
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
prop_offset = 0.5

r_tip_in = 1.5
n_blades_in = 6
rpm_in = 1200.0
pitch_in = 43.0

r_tip_out = 1.5
n_blades_out = 3
rpm_out = 1200.0
pitch_out = 43.0

weight_req = 32000.0

# %% Optimization variables
chord_tip = VariableFloat("chord_tip", 1.5, lb=0.5, ub=3.0)
span_0 = VariableFloat("span_0", 4.7, lb=2.0, ub=10.0)
xc_ref = VariableFloat("xc_ref", 0.5, lb=0.0, ub=1.0)
angle_root = VariableFloat("angle_root", 3.0, lb=0.0, ub=20.0)
angle_tip = VariableFloat("angle_tip", 0.0, lb=-10.0, ub=30.0)
prop_in_y = VariableFloat("prop_in_y", 0.5, lb=0.0, ub=1.0)
prop_mid_y = VariableFloat("prop_mid_y", 0.5, lb=0.0, ub=1.0)
prop_out_y = VariableFloat("prop_out_y", 0.5, lb=0.0, ub=1.0)

variables = [
    chord_tip,
    span_0,
    xc_ref,
    angle_root,
    angle_tip,
    prop_in_y,
    prop_mid_y,
    prop_out_y,
]

# Inner variables
span_1 = VariableFloat("span_1", 8.8)
efficiency = VariableFloat("wing_efficiency", 0.0)
lift_const = VariableFloat("lift_const", 1.0)

prop0_pos = VariableFloatNP("prop0_pos", np.array([-2.0, 3.0, 0.0]))
prop1_pos = VariableFloatNP("prop1_pos", np.array([-2.0, 8.0, 0.0]))
prop2_pos = VariableFloatNP("prop2_pos", np.array([-2.0, 12.0, 0.0]))
prop3_pos = VariableFloatNP("prop3_pos", np.array([-2.0, -3.0, 0.0]))
prop4_pos = VariableFloatNP("prop4_pos", np.array([-2.0, -8.0, 0.0]))
prop5_pos = VariableFloatNP("prop5_pos", np.array([-2.0, -12.0, 0.0]))

# %% Components and assembly
env = Environment(name="env", height=altitude, speed=speed)

airfoil_file = Path(__file__).parent.parent / "assets" / "naca653218.dat"
wing = Wing(
    name="wing",
    xc_ref=xc_ref,
    symmetry=True,
    sections=[
        Section(
            name="root_sec",
            airfoil=AirfoilFile(name="root_foil", filename=airfoil_file),
            chord=chord_root,
            twist=angle_root,
            options=[dust_lib.SectionOptions(polar=True)],
        ),
        Section(
            name="kink_sec",
            airfoil=AirfoilFile(name="kink_foil", filename=airfoil_file),
            chord=chord_root,
            twist=angle_root,
            options=[dust_lib.SectionOptions(polar=True)],
        ),
        Section(
            name="tip_sec",
            airfoil=AirfoilFile(name="tip_foil", filename=airfoil_file),
            chord=chord_tip,
            twist=angle_tip,
            options=[dust_lib.SectionOptions(polar=True)],
        ),
    ],
    spans=[
        Span(
            name="span_in",
            length=span_0,
            sweep=0.0,
            dihed=0.0,
            options=[
                dust_lib.SpanOptions(
                    panel_type=dust_lib.SpanPanelType.UNIFORM,
                    num_panels=35,
                ),
            ],
        ),
        Span(
            name="span_out",
            length=span_1,
            sweep=0.0,
            dihed=0.0,
            options=[
                dust_lib.SpanOptions(
                    panel_type=dust_lib.SpanPanelType.UNIFORM,
                    num_panels=35,
                ),
            ],
        ),
    ],
    options=[
        dust_lib.WingOptions(
            discretization_method=dust_lib.WingMethod.VORTEX_LATTICE,
            panel_type=dust_lib.WingPanelType.UNIFORM,
            num_panels=25,
            output_options=dust_lib.OutputOptions(
                compute_loads=True,
                loads_start=80,
                loads_end=90,
                loads_avg=True,
            ),
        ),
    ],
)

blades = [
    Blade(
        name=f"blade_{i}",
        xc_ref=0.5,
        mirror=i > 2,
        sections=[
            Section(
                name=f"blade_{i}.sec_0",
                airfoil=AirfoilNACA4(name=f"blade_{i}.airfoil_0", m=1, p=2, t=30),
                chord=0.3,
                twist=27.0,
                options=[dust_lib.SectionOptions(polar=True)],
            ),
            Section(
                name=f"blade_{i}.sec_1",
                airfoil=AirfoilNACA4(name=f"blade_{i}.airfoil_1", m=2, p=4, t=15),
                chord=0.3,
                twist=12.0,
                options=[dust_lib.SectionOptions(polar=True)],
            ),
            Section(
                name=f"blade_{i}.sec_2",
                airfoil=AirfoilNACA4(name=f"blade_{i}.airfoil_2", m=2, p=4, t=10),
                chord=0.15,
                twist=5.0,
                options=[dust_lib.SectionOptions(polar=True)],
            ),
        ],
        spans=[
            Span(
                name=f"blade_{i}.span_0",
                length=0.8,
                sweep=0.0,
                dihed=0.0,
                options=[
                    dust_lib.SpanOptions(
                        panel_type=dust_lib.SpanPanelType.UNIFORM,
                        num_panels=8,
                    ),
                ],
            ),
            Span(
                name=f"blade_{i}.span_1",
                length=0.6,
                sweep=0.0,
                dihed=0.0,
                options=[
                    dust_lib.SpanOptions(
                        panel_type=dust_lib.SpanPanelType.UNIFORM,
                        num_panels=8,
                    ),
                ],
            ),
        ],
        options=[
            dust_lib.WingOptions(
                discretization_method=dust_lib.WingMethod.VORTEX_LATTICE,
                panel_type=dust_lib.WingPanelType.UNIFORM,
                num_panels=10,
            ),
        ],
    )
    for i in range(6)
]

prop_0 = Propeller(
    name="prop_0",
    blade=blades[0],
    r_tip=r_tip_in,
    n_blades=n_blades_in,
    pitch=pitch_in,
    rpm=rpm_in,
    pos=prop0_pos,
)
prop_1 = Propeller(
    name="prop_1",
    blade=blades[1],
    r_tip=r_tip_out,
    n_blades=n_blades_out,
    pitch=pitch_out,
    rpm=rpm_out,
    pos=prop1_pos,
)
prop_2 = Propeller(
    name="prop_2",
    blade=blades[2],
    r_tip=r_tip_out,
    n_blades=n_blades_out,
    pitch=pitch_out,
    rpm=rpm_out,
    pos=prop2_pos,
)

prop_3 = Propeller(
    name="prop_3",
    blade=blades[3],
    r_tip=r_tip_in,
    n_blades=n_blades_in,
    pitch=pitch_in,
    rpm=rpm_in,
    reverse=True,
    pos=prop3_pos,
)
prop_4 = Propeller(
    name="prop_4",
    blade=blades[4],
    r_tip=r_tip_out,
    n_blades=n_blades_out,
    pitch=pitch_out,
    rpm=rpm_out,
    reverse=True,
    pos=prop4_pos,
)
prop_5 = Propeller(
    name="prop_5",
    blade=blades[5],
    r_tip=r_tip_out,
    n_blades=n_blades_out,
    pitch=pitch_out,
    rpm=rpm_out,
    reverse=True,
    pos=prop5_pos,
)

propellers = [prop_0, prop_1, prop_2, prop_3, prop_4, prop_5]


# %% Discipline> Geometric coupling
def geometry(span_in, xc, y_prop0, y_prop1, y_prop2):  # noqa: ANN001, ANN201
    lb = r_tip_in + 0.05
    ub = wing_span - 3.0 * r_tip_out - r_tip_in - 0.2
    p0y = lb + y_prop0[0] * (ub - lb)

    lb = p0y + r_tip_in + 0.1 + r_tip_out
    ub = wing_span - 2.0 * r_tip_out - 0.1
    p1y = lb + y_prop1[0] * (ub - lb)

    lb = p1y + 2.0 * r_tip_out + 0.1
    ub = wing_span
    p2y = lb + y_prop2[0] * (ub - lb)

    prop_off = chord_root * xc[0] + prop_offset

    prop0_pos = np.array([-prop_off, p0y, 0.0])
    prop1_pos = np.array([-prop_off, p1y, 0.0])
    prop2_pos = np.array([-prop_off, p2y, 0.0])
    prop3_pos = np.array([-prop_off, -p0y, 0.0])
    prop4_pos = np.array([-prop_off, -p1y, 0.0])
    prop5_pos = np.array([-prop_off, -p2y, 0.0])

    span_1 = wing_span - span_in

    return span_1, prop0_pos, prop1_pos, prop2_pos, prop3_pos, prop4_pos, prop5_pos


disc_geometry = UserDefined(
    name="Geometry",
    inputs=[span_0, xc_ref, prop_in_y, prop_mid_y, prop_out_y],
    outputs=[span_1, prop0_pos, prop1_pos, prop2_pos, prop3_pos, prop4_pos, prop5_pos],
    expression=geometry,
)

# Discipline> Aerodynamic polars
nf_solver = nf.Neuralfoil(options=nf.Options())
disc_polars = Aerodynamics(
    name="Polars",
    components=[env, wing, *propellers],
    solver=nf_solver,
)

# Discipline> Aerodynamics
dust_solver = dust.DUST(
    options=dust_lib.Options(
        steps_per_turn=30,
        n_turns=3,
        n_threads=30,
        n_wake_particles=100000,
        particles_box_min=np.array([-4.0, -15.0, -5.0]),
        particles_box_max=np.array([6.0, 15.0, 5.0]),
        output_options=dust_lib.OutputOptions(
            compute_loads=True,
            loads_start=80,
            loads_end=90,
            loads_avg=True,
        ),
    ),
)
disc_aerodynamics = Aerodynamics(
    name="Aerodynamics",
    components=[env, wing, *propellers],
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
