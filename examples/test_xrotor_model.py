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
    AirfoilNACA4,
    AirfoilFile,
    Wing,
    Propeller,
    Assembly,
)
from disciplines.aerodynamics import Aerodynamics
from disciplines.propulsion import Propulsion
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


geomdata = np.array(
    [
        # r/R     c/R     t/c     beta
        [0.2000, 0.0759, 0.7977, 45.6410],
        [0.3000, 0.1062, 0.3635, 41.8157],
        [0.3500, 0.1226, 0.2405, 39.4593],
        [0.4000, 0.1382, 0.1748, 36.8433],
        [0.4500, 0.1480, 0.1366, 34.1065],
        [0.5000, 0.1520, 0.1182, 31.8362],
        [0.5500, 0.1503, 0.1078, 29.9828],
        [0.6000, 0.1461, 0.1015, 28.3983],
        [0.6500, 0.1393, 0.0955, 27.0601],
        [0.7000, 0.1313, 0.0915, 25.9288],
        [0.7500, 0.1224, 0.0884, 24.9254],
        [0.8000, 0.1121, 0.0855, 24.0828],
        [0.8500, 0.1010, 0.0839, 23.3286],
        [0.9000, 0.0886, 0.0818, 22.6380],
        [0.9500, 0.0752, 0.0812, 22.1021],
    ]
)

# Environment here to compute atmosphere parameters
env = Environment("env", altitude, speed, alpha, beta)

case = {
    "conditions": {
        # Standard atmosphere at sea level
        "rho": env.density,
        "vso": env.pressure,
        "rmu": env.dyn_viscosity,
        "alt": altitude,
        # Operating conditions in accordance with NASA Report No. 640
        # adv = V / (Omega * R) = V / (RPM * pi/30 * R)
        "vel": speed,
        "adv": 0.70,
    },
    "disk": {
        "n_blds": n_blades_in,
        "blade": {
            "geometry": {
                # Geometry of the NASA Report No. 640, 5868-9, 2 blades, blade angle 25 deg at 0.75R propeller
                "r_hub": 0.305,
                "r_tip": r_tip_in,
                "r_wake": 0.0,
                "rake": 0.0,
                "radii": geomdata[:, 0],
                "chord": geomdata[:, 1],
                "twist": geomdata[:, 3],
                "ubody": np.zeros_like(geomdata[:, 0]),
            },
            "polars": {
                0.0: np.array(
                    [
                        # Clark-y at Re = 1,000,000, M = 0.3 (ncrit = 0.01)
                        # alpha     cl      cd       cm
                        [-10.00, -0.6912, 0.0334, -0.0951],
                        [-9.75, -0.7003, 0.0277, -0.0954],
                        [-9.50, -0.6899, 0.0246, -0.0947],
                        [-9.25, -0.6723, 0.0226, -0.0939],
                        [-9.00, -0.6511, 0.0210, -0.0930],
                        [-8.75, -0.6276, 0.0198, -0.0921],
                        [-8.50, -0.6027, 0.0188, -0.0913],
                        [-8.25, -0.5767, 0.0179, -0.0905],
                        [-8.00, -0.5500, 0.0172, -0.0898],
                        [-7.75, -0.5227, 0.0166, -0.0891],
                        [-7.50, -0.4950, 0.0160, -0.0885],
                        [-7.25, -0.4670, 0.0156, -0.0878],
                        [-7.00, -0.4387, 0.0151, -0.0873],
                        [-6.75, -0.4102, 0.0147, -0.0867],
                        [-6.50, -0.3816, 0.0144, -0.0862],
                        [-6.25, -0.3528, 0.0141, -0.0857],
                        [-6.00, -0.3240, 0.0138, -0.0852],
                        [-5.75, -0.2950, 0.0135, -0.0848],
                        [-5.50, -0.2660, 0.0133, -0.0844],
                        [-5.25, -0.2370, 0.0130, -0.0840],
                        [-5.00, -0.2079, 0.0128, -0.0836],
                        [-4.75, -0.1788, 0.0127, -0.0833],
                        [-4.50, -0.1497, 0.0125, -0.0830],
                        [-4.25, -0.1205, 0.0124, -0.0827],
                        [-4.00, -0.0914, 0.0122, -0.0824],
                        [-3.75, -0.0623, 0.0121, -0.0821],
                        [-3.50, -0.0331, 0.0120, -0.0818],
                        [-3.25, -0.0040, 0.0119, -0.0816],
                        [-3.00, 0.0251, 0.0119, -0.0813],
                        [-2.75, 0.0542, 0.0118, -0.0811],
                        [-2.50, 0.0832, 0.0117, -0.0809],
                        [-2.25, 0.1123, 0.0117, -0.0807],
                        [-2.00, 0.1413, 0.0117, -0.0804],
                        [-1.75, 0.1703, 0.0116, -0.0802],
                        [-1.50, 0.1992, 0.0116, -0.0800],
                        [-1.25, 0.2282, 0.0116, -0.0798],
                        [-1.00, 0.2570, 0.0116, -0.0796],
                        [-0.75, 0.2859, 0.0116, -0.0794],
                        [-0.50, 0.3147, 0.0116, -0.0792],
                        [-0.25, 0.3434, 0.0117, -0.0790],
                        [0.00, 0.3721, 0.0117, -0.0788],
                        [0.25, 0.4007, 0.0117, -0.0786],
                        [0.50, 0.4293, 0.0118, -0.0784],
                        [0.75, 0.4578, 0.0119, -0.0782],
                        [1.00, 0.4863, 0.0119, -0.0779],
                        [1.25, 0.5147, 0.0120, -0.0777],
                        [1.50, 0.5430, 0.0121, -0.0774],
                        [1.75, 0.5712, 0.0122, -0.0772],
                        [2.00, 0.5994, 0.0123, -0.0769],
                        [2.25, 0.6274, 0.0124, -0.0766],
                        [2.50, 0.6554, 0.0125, -0.0763],
                        [2.75, 0.6833, 0.0126, -0.0759],
                        [3.00, 0.7111, 0.0127, -0.0756],
                        [3.25, 0.7387, 0.0129, -0.0752],
                        [3.50, 0.7662, 0.0130, -0.0748],
                        [3.75, 0.7936, 0.0132, -0.0744],
                        [4.00, 0.8208, 0.0133, -0.0739],
                        [4.25, 0.8479, 0.0135, -0.0734],
                        [4.50, 0.8748, 0.0137, -0.0729],
                        [4.75, 0.9015, 0.0139, -0.0723],
                        [5.00, 0.9280, 0.0141, -0.0717],
                        [5.25, 0.9542, 0.0143, -0.0711],
                        [5.50, 0.9802, 0.0145, -0.0704],
                        [5.75, 1.0060, 0.0148, -0.0696],
                        [6.00, 1.0314, 0.0150, -0.0688],
                        [6.25, 1.0564, 0.0153, -0.0679],
                        [6.50, 1.0811, 0.0156, -0.0670],
                        [6.75, 1.1053, 0.0159, -0.066],
                        [7.00, 1.1290, 0.0162, -0.0648],
                        [7.25, 1.1514, 0.0165, -0.0635],
                        [7.50, 1.1719, 0.0168, -0.0618],
                        [7.75, 1.1921, 0.0172, -0.0600],
                        [8.00, 1.2133, 0.0176, -0.0585],
                        [8.25, 1.2342, 0.0181, -0.0571],
                        [8.50, 1.2546, 0.0185, -0.0555],
                        [8.75, 1.2743, 0.0191, -0.0539],
                        [9.00, 1.2934, 0.0196, -0.0523],
                        [9.25, 1.3118, 0.0202, -0.0506],
                        [9.50, 1.3295, 0.0209, -0.0489],
                        [9.75, 1.3465, 0.0216, -0.0472],
                        [10.00, 1.3628, 0.0224, -0.0454],
                        [10.25, 1.3783, 0.0232, -0.0437],
                        [10.50, 1.3932, 0.0241, -0.0420],
                        [10.75, 1.4073, 0.0251, -0.0403],
                        [11.00, 1.4205, 0.0262, -0.0386],
                        [11.25, 1.4330, 0.0274, -0.0370],
                        [11.50, 1.4445, 0.0287, -0.0354],
                        [11.75, 1.4551, 0.0302, -0.0339],
                        [12.00, 1.4647, 0.0317, -0.0325],
                        [12.25, 1.4733, 0.0334, -0.0312],
                        [12.50, 1.4807, 0.0353, -0.0299],
                        [12.75, 1.4869, 0.0373, -0.0288],
                        [13.00, 1.4919, 0.0395, -0.0278],
                        [13.25, 1.4955, 0.0420, -0.0270],
                        [13.50, 1.4979, 0.0446, -0.0263],
                        [13.75, 1.4989, 0.0474, -0.0258],
                        [14.00, 1.4985, 0.0505, -0.0255],
                        [14.25, 1.4967, 0.0539, -0.0254],
                        [14.50, 1.4934, 0.0576, -0.0256],
                        [14.75, 1.4885, 0.0615, -0.0260],
                        [15.00, 1.4822, 0.0657, -0.0266],
                        [15.25, 1.4742, 0.0703, -0.0275],
                        [15.50, 1.4647, 0.0751, -0.0286],
                        [15.75, 1.4540, 0.0802, -0.0299],
                        [16.00, 1.4421, 0.0856, -0.0315],
                        [16.25, 1.4294, 0.0912, -0.0333],
                        [16.50, 1.4162, 0.0969, -0.0353],
                        [16.75, 1.4029, 0.1028, -0.0374],
                        [17.00, 1.3895, 0.1087, -0.0398],
                        [17.25, 1.3764, 0.1147, -0.0423],
                        [17.50, 1.3638, 0.1207, -0.0450],
                        [17.75, 1.3516, 0.1267, -0.0478],
                        [18.00, 1.3402, 0.1327, -0.0509],
                        [18.25, 1.3295, 0.1387, -0.0541],
                        [18.50, 1.3194, 0.1447, -0.0574],
                        [18.75, 1.3100, 0.1507, -0.0609],
                        [19.00, 1.3012, 0.1567, -0.0646],
                        [19.25, 1.2931, 0.1627, -0.0683],
                        [19.50, 1.2855, 0.1686, -0.0722],
                        [19.75, 1.2787, 0.1746, -0.0761],
                        [20.00, 1.2727, 0.1805, -0.0801],
                    ]
                )
            },
        },
    },
    "settings": {"free": True, "duct": False, "wind": False},
}
# reference values
weight_req = 32000.0
thrust_req = 18000.0
wing_span = 13.5

# Optimization variables
chord_tip = Variable("chord_tip", 1.5, lb=0.5, ub=3.0)
span_0 = Variable("span_0", 4.7, lb=2.0, ub=10.0)
xc_ref = Variable("xc_ref", 0.5, lb=0.0, ub=1.0)
angle_root = Variable("angle_root", 3.0, lb=0.0, ub=20.0)
angle_tip = Variable("angle_tip", 0.0, lb=-10.0, ub=30.0)

# Coupled variables
span_1 = Variable("span_1", 8.8, lb=5.0, ub=10.0)
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

# -----------------------------------------------------------------------
# define XRotor
# -----------------------------------------------------------------------
# define rpm as design variale
rpm = Variable("rpm", 2000.0, lb=1000.0, ub=3000.0)
thrust_trim = Variable("thrust_trim", 150000.0, lb=10000.0, ub=30000.0)
eff_prop_target = Variable("eff_prop_target", 0.8, lb=0.3, ub=0.99)

# Output variable for propulsion
thrust = Variable(
    name="thrust",
    value=12000.0,
    output_type="thrust",
)
prop_efficiency = Variable(
    name="prop_efficiency",
    value=0.5,
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
    options={
        "prop_xrotor": {
            "thrust": thrust_req,
            "adv": 0.7,
            "geomdata": geomdata,
            "case": case,
        },
    },
)

# define new coupling fro trimming
# def Coupling_Thrust(thrust):
#    thrust_trim = thrust-18000.0
#    return thrust_trim

# -----------------------------------------------------------------------

# All variables that are included in the design space - XROTOR
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
    thrust,
    rpm,
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
wing = Wing(
    name="wing",
    xc_ref=xc_ref,
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
    solver_name="aero_lf",
)

# XROTOR - Propulsion discipline
# -----------------------------------------------------------
disc_propulsion = Propulsion(
    "Propulsion",
    inputs=variables,
    outputs=[thrust, prop_efficiency],
    components=assembly.components,
    solver_name="xrotor",
    options={"free": True, "rake": False, "altitude": altitude / unit.ft},
)

# -----------------------------------------------------------

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

disc_obj_fun3 = UserDefined(
    "Thrust trim",
    inputs=[thrust],
    outputs=[thrust_trim],
    expression=f"(thrust - {thrust_req}) / {thrust_req}",
)

disc_obj_fun4 = UserDefined(
    "Efficiency prop target",
    inputs=[prop_efficiency],
    outputs=[eff_prop_target],
    expression="prop_efficiency - 0.8",
)

constraints = [
    {"name": lift_const.name, "type": "eq"},
    {"name": eff_prop_target.name, "type": "ineq"},
    {"name": thrust_trim.name, "type": "eq"},
]

# Discipline list
disciplines = [
    disc_coupling,
    disc_aerodynamics,
    disc_propulsion,
    disc_obj_fun1,
    disc_obj_fun2,
    disc_obj_fun3,
    disc_obj_fun4,
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
        "max_iter": 30,
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
execute_post(to_be_postprocess, "RadarChart", save=True, show=False)
execute_post(to_be_postprocess, "GradientSensitivity", save=True, show=False)
execute_post(to_be_postprocess, "ParallelCoordinates", save=True, show=False)
