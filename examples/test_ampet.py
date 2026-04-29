import logging
import os, sys

sys.path.insert(
    0,
    os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")),
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

from assembly import (
    Environment,
    Section,
    Span,
    AirfoilFile,
    Wing,
    Assembly,
    InterfaceFloatMatrix,
)
from disciplines.aerodynamics import Aerodynamics
from disciplines.weight_and_balance import WeightAndBalance
from disciplines.disciplines import UserDefined
from postprocessing.optimizationPostProcessing import optimizationResults
from scenario import MADSScenario, Variable
import utilities.units as unit


# Logger
logger = gemseo.configure_logger(
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

ACap0 = 10.0e-3
chord_root = 3.0
RibsSeparation = 0.475
PanelLength = 0.1
yWingFusInterface = 1.1
nEngines = 1
# yEngines = [3.5]
mEngines = np.array([988.650])
wingX0 = 0.0
wingY0 = 0.0
wingZ0 = 0.0
wing_span = 13.5
wingProfile = "naca653218.dat"


# Dust Variables
nPanelsInner = 10  # 35
nPanelsOuter = 10  # 35
nPanelsTotal = int(2 * (nPanelsInner + nPanelsOuter))
nPanelsChord = 8  # 25
dustDeltaT = 0.005
dustNTempSteps = 35  # 100
dustNStep0Calc = 30  # 90
dustTf = dustDeltaT * dustNTempSteps

SFC = 0.5 / (3600.0 * 9.81)  # 0.5 lb/(h*lbf)

mtow = 32000.0
zfw_WingExcluded = 24000.0

# Optimization variables

alphaF = Variable("alphaF", -5.0, lb=-15.0, ub=5.0)
chord_tip = Variable("chord_tip", 1.5, lb=0.5, ub=3.0)
span_0 = Variable("span_0", 4.7, lb=2.0, ub=10.0)
xc_ref = 0.25  # Variable("xc_ref", 0.5, lb=0.0, ub=1.0)
angle_root = Variable("angle_root", 3.0, lb=0.0, ub=20.0)
angle_tip = Variable("angle_tip", 0.0, lb=-10.0, ub=30.0)
# ACapRoot = 0.01316719739419805
ACapRoot = Variable("ACapRoot", ACap0, lb=1.0e-5, ub=20.0e-3)
# AStringerRoot = 0.0004412478551108304
AStringerRoot = Variable("AStringerRoot", ACap0 / 10.0, lb=1.0e-5, ub=20.0e-3)

# Coupling Variables

span_1 = Variable("span_1", 8.8)

wingSurfPlus = wing_span * chord_root
wingSurfMinus = 8.8 * (chord_root - 1.5)
wingSurf = 2.0 * (wingSurfPlus - wingSurfMinus)

with open("pareto_data.txt", "w") as myfile:
    myfile.write("valid(0:No/1:Yes);InitialEff;FinalEff;WingMass(kg);Range(km)\n")

yEngines = Variable(
    name="yEngines",
    value=np.array([3.5]),
)
lift = Variable(
    name="L",
    value=300000.0,
    output_type="lift",
    options={
        "dust": {
            "start_res": dustNStep0Calc,
            "end_res": dustNTempSteps,
            "components": ["wing"],
        }
    },
)
liftF = Variable(
    name="LF",
    value=300000.0,
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
    name="D",
    value=10000.0,
    output_type="drag",
    options={
        "dust": {
            "start_res": dustNStep0Calc,
            "end_res": dustNTempSteps,
            "components": ["wing"],
        }
    },
)
dragF = Variable(
    name="DF",
    value=10000.0,
    output_type="drag",
    options={
        "dust": {
            "start_res": dustNStep0Calc,
            "end_res": dustNTempSteps,
            "components": ["wing"],
        }
    },
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
        }
    },
)
mass = Variable(
    name="mass",
    output_type="mass",
    value=300000.0,
)

# mass = 6185.791359028178

fuel = Variable(
    name="fuel",
    value=6000.0,
)
# fuel = mtow - mass - zfw_WingExcluded

efficiency = Variable(
    name="Eff",
    value=5.0,
)
efficiencyF = Variable(
    name="EffF",
    value=5.0,
)


# Constraints variables

lift_const = Variable(
    name="Lift_const",
    value=1.0,
)
lift_constF = Variable(
    name="Lift_constF",
    value=1.0,
)
minRFC = Variable(
    name="minRFC",
    output_type="minRFC",
    value=1.0,
)
minRFT = Variable(
    name="minRFT",
    output_type="minRFT",
    value=1.0,
)

# Output Variables

rangeVar = Variable(
    name="range",
    value=50.0,
)

# All variables
variables = [
    # Optimization
    ACapRoot,
    AStringerRoot,
    chord_tip,
    span_0,
    span_1,
    # xc_ref,
    angle_root,
    angle_tip,
    yEngines,
    alphaF,
    # Internal
    lift,
    liftF,
    drag,
    dragF,
    spanload,
    mass,
    minRFC,
    minRFT,
    fuel,
    efficiency,
    efficiencyF,
]

# Components and assembly

env = Environment("env_0", altitude, speed, alpha, beta)
envF = Environment("env_1", altitude, speed, alphaF, beta)

rho = env.density
cd0 = 0.022
dragFR = 0.5 * rho * speed * speed * wingSurf * cd0

root_section = Section(
    "root_sec",
    airfoil=AirfoilFile("root_foil", filename=wingProfile),
    chord=chord_root,
    twist=angle_root,
)
tip_section = Section(
    "tip_sec",
    airfoil=AirfoilFile("tip_foil", filename=wingProfile),
    chord=chord_tip,
    twist=angle_tip,
)
wing = Wing(
    name="wing",
    xc_ref=xc_ref,
    symmetry=True,
    ribsMaxSeparation=RibsSeparation,
    panelMaxLength=PanelLength,
    pos=np.array([wingX0, wingY0, wingZ0]),
    ACapRoot=ACapRoot,
    AStringerRoot=AStringerRoot,
    yWingFuselageInterface=yWingFusInterface,
    nEngines=nEngines,
    yEngines=yEngines,
    mEngines=mEngines,
    options={
        "dust": {
            "n_elem": nPanelsChord,
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
            "span_0",
            length=span_0,
            sweep=0.0,
            dihed=0.0,
            options={
                "dust": {
                    "n_elem": nPanelsInner,
                }
            },
        ),
        Span(
            "span_1",
            length=span_1,
            sweep=0.0,
            dihed=0.0,
            options={
                "dust": {
                    "n_elem": nPanelsOuter,
                    "elem_type": "uniform",
                }
            },
        ),
    ],
)

aeroForces = InterfaceFloatMatrix("DustLoads", matrix=spanload)

# define assembly
assembly = Assembly(
    "assembly",
    [env, wing, aeroForces],
    variables,
)
assemblyF = Assembly(
    "assembly",
    [envF, wing],
    variables,
)

# disciplines
disc_aerodynamics = Aerodynamics(
    "Aerodynamics 0",
    inputs=variables,
    outputs=[lift, drag, spanload],
    components=assembly.components,
    solver_name="dust",
    options={
        "t_end": dustTf,
        "dt": dustDeltaT,
        "n_threads": 30,
        "n_wake_particles": 100000,
        "particles_box_min": [-4.0, -15.0, -5.0],
        "particles_box_max": [6.0, 15.0, 5.0],
    },
)

disc_aerodynamicsF = Aerodynamics(
    "Aerodynamics 1",
    inputs=variables,
    outputs=[liftF, dragF],
    components=assemblyF.components,
    solver_name="dust",
    options={
        "t_end": dustTf,
        "dt": dustDeltaT,
        "n_threads": 30,
        "n_wake_particles": 100000,
        "particles_box_min": [-4.0, -15.0, -5.0],
        "particles_box_max": [6.0, 15.0, 5.0],
    },
)

disc_mass = WeightAndBalance(
    "AMPET",
    inputs=variables,
    components=assembly.components,
    outputs=[mass, minRFC, minRFT],
    solver_name="ampet",
)

# coupling


def couplingFunction(span0):
    # span1 = 2.0*(SurfSemiWing-(span0[0]*chord_root))/(chord_root+chordTip[0])
    span1 = wing_span - span0[0]
    chordTip = chord_root - wingSurfMinus / span1
    return ([span1], [chordTip], span0 - 0.75)  # wing_span - span0


disc_geo_coupling = UserDefined(
    "Coupling",
    inputs=[span_0],
    outputs=[span_1, chord_tip, yEngines],
    expression=couplingFunction,  # f"{wing_span} - span_0",
)

# objective

def objectiveFunction(l0, d0, lf, df, mWing, minRFC, minRFT, alphaF):

    eff0 = l0[0] / (d0[0] + dragFR)
    effF = lf[0] / (df[0] + dragFR)
    averageEff = (eff0 + effF) / 2.0

    mZFW = zfw_WingExcluded + mWing[0]
    mFuel = mtow - mZFW

    trimming0 = (l0[0] - 9.81 * mtow) / (9.81 * mtow)
    trimmingF = (lf[0] - 9.81 * mZFW) / (9.81 * mZFW)

    range = speed * averageEff / (9.81 * SFC) * np.log(mtow / mZFW)

    logger.info("   NEW ITERATION")
    logger.info("      Initial Efficiency  = " + str(eff0))
    logger.info(
        "         Initial Lift (N) = "
        + str(l0[0])
        + " (L/W="
        + str(l0[0] / (9.81 * mtow))
        + ")"
    )
    logger.info("         Initial Drag (N) = " + str(d0[0]))
    logger.info("      Final Efficiency    = " + str(effF))
    logger.info("         Alpha (Deg)      = " + str(alphaF[0]))
    logger.info(
        "         Final Lift (N)   = "
        + str(lf[0])
        + " (L/W="
        + str(lf[0] / (9.81 * mZFW))
        + ")"
    )
    logger.info("         Final Drag (N)   = " + str(df[0]))
    logger.info("      Wing Mass (kg)      = " + str(mWing[0]))
    logger.info("         minRFComp        = " + str(minRFC[0]))
    logger.info("         minRFTrac        = " + str(minRFT[0]))
    logger.info("      Range (km)          = " + str(range / 1000.0))

    valid = 1

    if not (
        (abs(trimming0) < 0.1)
        and (abs(trimmingF) < 0.1)
        and (minRFC[0] >= 1.4)
        and (minRFT[0] >= 1.4)
    ):
        valid = 0
        logger.info("   ERROR! SET NOT VALID!")

        if trimming0 < -0.1:
            logger.info("      Lift at beginning is too low")
        elif trimming0 > 0.1:
            logger.info("      Lift at beginning is too high")

        if trimmingF < -0.1:
            logger.info("      Lift at ending is too low")
        elif trimmingF > 0.1:
            logger.info("      Lift at ending is too high")

        if minRFC[0] < 1.5:
            logger.info("      Compression Reserve Factor is too low")

        if minRFT[0] < 1.5:
            logger.info("      Traction Reserve Factor is too low")

    with open("pareto_data.txt", "a") as myfile:
        myfile.write(
            str(valid)
            + ";"
            + str(eff0).replace(".", ",")
            + ";"
            + str(effF).replace(".", ",")
            + ";"
            + str(mWing[0]).replace(".", ",")
            + ";"
            + str(range / 1000.0).replace(".", ",")
            + "\n"
        )
    return [[-eff0], [trimming0], [effF], [trimmingF], [-range]]


disc_post_fun = UserDefined(
    "Post Analysis",
    inputs=[lift, drag, liftF, dragF, mass, minRFC, minRFT, alphaF],
    outputs=[efficiency, lift_const, efficiencyF, lift_constF, rangeVar],
    expression=objectiveFunction,  # f"{speed}*((efficiency+efficiencyF)/2.0)/(9.81*{SFC}) * log({mtow}/({mtow}-fuel))",
)

# Design space
mads_scenario = MADSScenario()
mads_scenario.fill_parameter_space(variables)

# discipline list
disciplines = [
    disc_geo_coupling,
    disc_aerodynamics,
    # disc_lift_check,
    # disc_aero_coupling,
    disc_mass,
    # disc_fuel_coupling,
    disc_aerodynamicsF,
    # disc_lift_checkF,
    # disc_aero_couplingF,
    # disc_obj_fun,
    disc_post_fun,
]


# Scenario
mads_scenario.create_scenario(
    disciplines,
    formulation="DisciplinaryOpt",
    objective_name=[rangeVar.name],
    maximize_objective=True,
)

constraints = [
    {"name": minRFC.name, "value": 1.5, "type": "ineq", "positive": True},
    {"name": minRFT.name, "value": 1.5, "type": "ineq", "positive": True},
    {"name": lift_const.name, "type": "eq"},
    {"name": lift_constF.name, "type": "eq"},
]
mads_scenario.add_constraints(constraints)
mads_scenario.add_observable(mass.name)
mads_scenario.add_observable(efficiency.name)


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
        "max_iter": 20,
        "algo": "NLOPT_COBYLA",
        # "algo": "SLSQP",
        "algo_options": {
            "eq_tolerance": 0.1,
            "ineq_tolerance": 0.1,
        },
    }
)


maximizeObservable = [False, True] # For plotting, the optimal in mass (first Observable) is minimal and in Efficiency (second Observable) is maximal
optimizationResults(mads_scenario.scenario, maximizeObservable=maximizeObservable, save=True, show=False)

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
    constraint_names=[
        minRFT.name,
        minRFC.name,
        lift_const.name,
        lift_constF.name,
    ],
    save=True,
    show=False,
)
# execute_post(to_be_postprocess, 'ConstraintsHistory', constraint_names=['minRFTraction','minRFCompression','Lift_const'],save=True, show=False)
# execute_post(to_be_postprocess, 'ConstraintsHistory', constraint_names=['Lift_const'],save=True, show=False)
execute_post(to_be_postprocess, "ObjConstrHist", save=True, show=False)
# execute_post(to_be_postprocess, 'RadarChart', save=True, show=False)
# execute_post(to_be_postprocess, 'GradientSensitivity', save=True, show=True)
execute_post(
    to_be_postprocess,
    "ParetoFront",
    objectives=[mass.name, efficiency.name],
    save=True,
    show=False,
)
# execute_post(to_be_postprocess, "ParetoFront", objectives=["chord_tip", "span_0","angle_root","angle_tip","ACapRoot","AStringerRoot"], save=True, show=False)

# execute_post(to_be_postprocess, 'ParallelCoordinates',save=True, show=False)
# execute_post(to_be_postprocess, 'BasicHistory',variable_names=['span_0'],save=True, show=False)
