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
from scenario import MADSScenario, Variable
import utilities.units as unit


# Logger
logger = gemseo.configure_logger(
    level=logging.INFO,
    # level=logging.DEBUG,
    # filename="gemseo.log",
    # filemode="w",
)

# Parameters

altitude = 25000.0 * unit.ft
speed = 154.835
beta = 0.0

alphaBeginCruise = 0.0
alphaEndCruiseInit = -5.0

chordRootInit = 3.0

ACapInit = 10.0e-3
chordTipInit = 1.5
chordTipMin = 0.0

span0Init = 4.7
span1Init = 8.8

wingSpan = span0Init + span1Init

wingSurfPlus = wingSpan * chordRootInit
wingSurfMinus = span1Init * (chordRootInit - chordTipInit) / 2.0
wingSurf = wingSurfPlus - wingSurfMinus

distanceKinkEngine = 0.75
yEnginesInit = span0Init - distanceKinkEngine


xcRef = 0.25
RibsSeparation = 0.475
PanelLength = 0.1
yWingFusInterface = 1.1
nEngines = 1
yEngines = np.array([yEnginesInit])
mEngines = np.array([988.650])
wingX0 = 0.0
wingY0 = 0.0
wingZ0 = 0.0
wingProfile = "naca653218.dat"

# Dust Variables
nPanelsInner = 20  # 20, 35
nPanelsOuter = 35  # 35
nPanelsTotal = int(2 * (nPanelsInner + nPanelsOuter))
nPanelsChord = 25  # 25
dustDeltaT = 0.005
dustNTempSteps = 100  # 100
dustNStep0Calc = 90  # 90
dustTf = dustDeltaT * dustNTempSteps

SFC = 0.5 / (3600.0 * 9.81)  # 0.5 lb/(h*lbf)

panelsDensityInner = nPanelsInner / span0Init
panelsDensityOuter = nPanelsOuter / span1Init

mtow = 32000.0
zfwWingExcluded = 24000.0

spanLower = 0.75 * wingSpan  # yEngines[0]
spanUpper = 1.5 * wingSpan

chordLower = 0.2


# Optimization variables

span0 = Variable("span0", span0Init)
chordRoot = Variable("chordRoot", chordRootInit)
angleRoot = Variable("angleRoot", 3.0, lb=0.0, ub=20.0)
angleTip = Variable("angleTip", 0.0, lb=-10.0, ub=30.0)
ACapRoot = Variable("ACapRoot", ACapInit, lb=1.0e-5, ub=20.0e-3)
AStringerRoot = Variable("AStringerRoot", ACapInit / 10.0, lb=1.0e-5, ub=20.0e-3)
alphaEndCruise = Variable("alphaEndCruise", alphaEndCruiseInit, lb=-15.0, ub=5.0)

# Geometric Coupling Variables

chordTip = Variable("chordTip", chordTipInit)
span1 = Variable("span1", span1Init)
checkGeo = Variable("checkGeometry", 0.0)
# yEngines = Variable("yEngines",np.array([yEnginesInit]),isArray = True)
newIteration = Variable("newIteration", 1.0)

liftBeginCruise = Variable(
    name="LBeginCruise",
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
dragBeginCruise = Variable(
    name="DBeginCruise",
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

effBeginCruise = Variable("EffBeginCruise", 5.0)

mass = Variable(name="mass", value=300000.0, output_type="mass")

liftEndCruise = Variable(
    name="LEndCruise",
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
dragEndCruise = Variable(
    name="DEndCruise",
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
effEndCruise = Variable("EffEndCruise", 5.0)

# Constrains variables

liftConstBeginCruise = Variable("LiftConstBeginCruise", 1.0)
liftConstEndCruise = Variable("LiftConstEndCruise", 1.0)

minRFC = Variable(name="minRFC", value=1.0, output_type="minRFC")
minRFT = Variable(name="minRFT", value=1.0, output_type="minRFT")

# Output Variables

rangeVar = Variable("range", 50.0)

# All variables
variables = [
    # Optimization
    ACapRoot,
    AStringerRoot,
    chordRoot,
    span0,
    angleRoot,
    angleTip,
    alphaEndCruise,
    # Internal
    checkGeo,
    span1,
    chordTip,
    # yEngines,
    liftBeginCruise,
    dragBeginCruise,
    spanload,
    effBeginCruise,
    mass,
    liftEndCruise,
    dragEndCruise,
    effEndCruise,
    minRFC,
    minRFT,
    liftConstBeginCruise,
    liftConstEndCruise,
]

# Components and assembly

envBeginCruise = Environment("env_0", altitude, speed, alphaBeginCruise, beta, 0.0)
envEndCruise = Environment("env_1", altitude, speed, alphaEndCruise, beta, 0.0)

rootSection = Section(
    "root_sec",
    airfoil=AirfoilFile("root_foil", filename=wingProfile),
    chord=chordRoot,
    twist=angleRoot,
)
tipSection = Section(
    "tip_sec",
    airfoil=AirfoilFile("tip_foil", filename=wingProfile),
    chord=chordTip,
    twist=angleTip,
)
wing = Wing(
    name="wing",
    xc_ref=xcRef,
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
        rootSection,
        rootSection,
        tipSection,
    ],
    spans=[
        Span(
            "span0",
            length=span0,
            sweep=0.0,
            dihed=0.0,
            options={
                "dust": {
                    # "n_elem": nPanelsInner,
                    "panelDensity": panelsDensityInner,
                }
            },
        ),
        Span(
            "span1",
            length=span1,
            sweep=0.0,
            dihed=0.0,
            options={
                "dust": {
                    # "n_elem": nPanelsOuter,
                    "panelDensity": panelsDensityOuter,
                    "elem_type": "cosineOB",
                }
            },
        ),
    ],
)

aeroForces = InterfaceFloatMatrix("int_matrix", matrix=spanload)

# define assembly
assemblyBeginCruise = Assembly(
    "assemblyBeginCruise",
    [envBeginCruise, wing, aeroForces],
    variables,
)

assemblyEndCruise = Assembly(
    "assemblyFinalCruise",
    [envEndCruise, wing],
    variables,
)


def GenerateSamples(nSpan, nKink, nChord):

    hSpan = spanUpper - spanLower
    hYkink = 0.9 - 0.1

    nSamples = nChord * nKink * nSpan

    samples = np.zeros((nSamples, 3))

    dSpan = hSpan / (nSpan - 1)
    dKink = hYkink / (nKink - 1)

    for iSpan in range(nSpan):
        span = spanLower + dSpan * iSpan

        for iKink in range(nKink):
            yKink = 0.1 + dKink * iKink

            span0 = span * yKink
            span1 = span - span0

            chordMin = wingSurf / span
            chordMax = (2.0 * wingSurf / span - chordLower * (1.0 - yKink)) / (
                1.0 + yKink
            )

            dChord = (chordMax - chordMin) / (nChord - 1)

            for iChord in range(nChord):
                chord = chordMin + dChord * iChord
                iSample = int(iSpan * nChord * nKink + iKink * nChord + iChord)

                samples[iSample] = np.array([chord, span0, span1])

    return [nSamples, samples]


# Geometry Coupling


def wingGeoAnalysis(span0, span1, chordRoot):

    wingSpan = span0 + span1

    wingSurfPlus = wingSpan * chordRoot
    wingSurfMinus = wingSurfPlus - wingSurf

    chordTip = chordRoot - 2.0 * wingSurfMinus / span1

    return chordTip


def couplingGeometryFunction(span0, span1, chordRoot):

    coupledChordTip = wingGeoAnalysis(span0[0], span1[0], chordRoot[0])

    # if (coupledChordTip < chordTipLower):
    #    coupledChordTip = chordTipLower

    return [coupledChordTip]  # ,[coupledYEngines])


def aeroAnalysisFunction(l, d, mWing=[0.0]):
    eff = l[0] / d[0]
    if mWing[0] == 0.0:
        mass = mtow
    else:
        mass = zfwWingExcluded + mWing[0]
    trimming = (l[0] - 9.81 * mass) / (9.81 * mass)
    return [[eff], [trimming]]


def objectiveFunction(
    l0, d0, lf, df, mWing, minRFC, minRFT, alphaF, span0, span1, chordRoot, chordTip
):

    eff0 = l0[0] / d0[0]
    effF = lf[0] / df[0]
    averageEff = (eff0 + effF) / 2.0

    mZFW = zfwWingExcluded + mWing[0]
    mFuel = mtow - mZFW

    trimming0 = (l0[0] - 9.81 * mtow) / (9.81 * mtow)
    trimmingF = (lf[0] - 9.81 * mZFW) / (9.81 * mZFW)

    range = speed * averageEff / (9.81 * SFC) * np.log(mtow / mZFW)

    # [coupledSpan1,coupledChordTip] = wingGeoAnalysis(span0[0],chordRoot[0])

    checkGeo = -1.0
    # if (coupledChordTip<chordTipLower):
    #    checkGeo = 1.0

    logger.info("   NEW ITERATION")
    logger.info("      Span0            = " + str(span0[0]))
    logger.info("      Span1            = " + str(span1[0]))
    logger.info("      ChordRoot        = " + str(chordRoot[0]))
    logger.info("      ChordTip         = " + str(chordTip[0]))
    logger.info("      Begin Cruise Eff = " + str(eff0))
    logger.info(
        "         Lift (N)      = "
        + str(l0[0])
        + " (L/W="
        + str(l0[0] / (9.81 * mtow))
        + ")"
    )
    logger.info("         Drag (N)      = " + str(d0[0]))
    logger.info("      End Cruise Eff   = " + str(effF))
    logger.info("         Alpha (Deg)   = " + str(alphaF[0]))
    logger.info(
        "         Lift (N)      = "
        + str(lf[0])
        + " (L/W="
        + str(lf[0] / (9.81 * mZFW))
        + ")"
    )
    logger.info("         Drag (N)      = " + str(df[0]))
    logger.info("      Wing Mass (kg)   = " + str(mWing[0]))
    logger.info("         minRFComp     = " + str(minRFC[0]))
    logger.info("         minRFTrac     = " + str(minRFT[0]))
    logger.info("      Range (km)       = " + str(range / 1000.0))

    valid = 1

    if not (
        (abs(trimming0) < 0.1)
        and (abs(trimmingF) < 0.1)
        and (minRFC[0] >= 1.4)
        and (minRFT[0] >= 1.4)
        and (checkGeo < 0.0)
    ):
        valid = 0
        logger.info("   ERROR! SET NOT VALID!")

        if checkGeo > 0.0:
            logger.info("      Geometry set not compatible")

        if trimming0 < -0.1:
            logger.info("      Lift at beginning is too low")
        elif trimming0 > 0.1:
            logger.info("      Lift at beginning is too high")

        if trimmingF < -0.1:
            logger.info("      Lift at ending is too low")
        elif trimmingF > 0.1:
            logger.info("      Lift at ending is too high")

        if minRFC[0] < 1.4:
            logger.info("      Compression Reserve Factor is too low")

        if minRFT[0] < 1.4:
            logger.info("      Traction Reserve Factor is too low")

    with open("pareto_data.txt", "a") as myfile:
        myfile.write(
            str(valid)
            + ";"
            + str(span0[0]).replace(".", ",")
            + ";"
            + str(span1[0]).replace(".", ",")
            + ";"
            + str(chordRoot[0]).replace(".", ",")
            + ";"
            + str(chordTip[0]).replace(".", ",")
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

    if valid == 0:
        range = 0.0

    return [range]

# Design space
mads_scenario = MADSScenario()
mads_scenario.fill_parameter_space(variables)

###############################################################################################
# Aerodynamics Cruise Begining
###############################################################################################

disc_geo_coupling = UserDefined(
    "Geometry Coupling",
    inputs=[span0, span1, chordRoot],
    outputs=[chordTip],
    expression=couplingGeometryFunction,
)

disc_aerodynamics_BeginCruise = Aerodynamics(
    "Aerodynamics 0",
    inputs=variables,
    outputs=[liftBeginCruise, dragBeginCruise, spanload],
    components=assemblyBeginCruise.components,
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

disc_aero_BeginCruise_Analysis = UserDefined(
    "Aero Begin Cruise Analysis",
    inputs=[liftBeginCruise, dragBeginCruise],
    outputs=[effBeginCruise, liftConstBeginCruise],
    expression=aeroAnalysisFunction,
)

aero_BeginCruise_scenario = gemseo.create_scenario(
    [disc_geo_coupling, disc_aerodynamics_BeginCruise, disc_aero_BeginCruise_Analysis],
    "DisciplinaryOpt",
    effBeginCruise.name,
    mads_scenario.parameter_space.filter([angleRoot.name, angleTip.name], copy=True),
    name="Aero Begin Cruise Scenario",
    maximize_objective=True,
)


aero_BeginCruise_scenario.add_constraint(
    output_name="LiftConstBeginCruise",
    constraint_name="Begin Cruise Trimming",
    value=0.0,
    constraint_type="eq",
)

###############################################################################################
# Wing Mass Estimation
###############################################################################################

disc_mass = WeightAndBalance(
    "Mass",
    inputs=variables,
    components=assemblyBeginCruise.components,
    outputs=[mass, minRFC, minRFT],
    solver_name="ampet",
)

mass_scenario = gemseo.create_scenario(
    [disc_mass],
    "DisciplinaryOpt",
    mass.name,
    mads_scenario.parameter_space.filter([ACapRoot.name, AStringerRoot.name], copy=True),
    name="MassScenario",
    maximize_objective=False,
)

mass_scenario.add_constraint(
    output_name=minRFC.name,
    value=1.5,
    constraint_type="ineq",
    positive=True,
)

mass_scenario.add_constraint(
    output_name=minRFT.name,
    value=1.5,
    constraint_type="ineq",
    positive=True,
)

###############################################################################################
# Aerodynamics Cruise Ending
###############################################################################################

disc_aerodynamics_EndCruise = Aerodynamics(
    "Aerodynamics 1",
    inputs=variables,
    outputs=[liftEndCruise, dragEndCruise],
    components=assemblyEndCruise.components,
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

disc_aero_EndCruise_Analysis = UserDefined(
    "Aero End Cruise Analysis",
    inputs=[liftEndCruise, dragEndCruise, mass],
    outputs=[effEndCruise, liftConstEndCruise],
    expression=aeroAnalysisFunction,
)

aero_EndCruise_scenario = gemseo.create_scenario(
    [disc_aerodynamics_EndCruise, disc_aero_EndCruise_Analysis],
    "DisciplinaryOpt",
    effEndCruise.name,
    mads_scenario.parameter_space.filter([alphaEndCruise.name], copy=True),
    name="Aero End Cruise Scenario",
    maximize_objective=True,
)

aero_EndCruise_scenario.add_constraint(
    output_name=liftConstEndCruise.name,
    value=0.0,
    constraint_type="eq",
)

###############################################################################################
# objective
###############################################################################################

disc_range_Analysis = UserDefined(
    "Range Analysis",
    inputs=[
        liftBeginCruise,
        dragBeginCruise,
        liftEndCruise,
        dragEndCruise,
        mass,
        minRFC,
        minRFT,
        alphaEndCruise,
        span0,
        span1,
        chordRoot,
        chordTip,
    ],
    outputs=[rangeVar],
    expression=objectiveFunction,
)

sub_disciplines = [
    aero_BeginCruise_scenario,
    mass_scenario,
    aero_EndCruise_scenario,
    disc_range_Analysis,
]

# Scenario
mads_scenario.scenario = gemseo.create_scenario(
    sub_disciplines,
    "BiLevel",  #'DisciplinaryOpt',#'BiLevel', #'MDF',
    rangeVar.name,
    mads_scenario.parameter_space.filter(
        [
            ACapRoot.name,
            AStringerRoot.name,
            chordRoot.name,
            span0.name,
            span1.name,
            angleRoot.name,
            angleTip.name,
            alphaEndCruise.name,
        ],
        copy=True,
    ),
    # inner_mda_name='MDAGaussSeidel',
    maximize_objective=True,
    scenario_type="DOE",
    reset_x0_before_opt=True,
)


mads_scenario.scenario.set_differentiation_method(
    "finite_differences",
    step=1e-3,
)


# mads_scenario.scenario.add_constraint(
#    output_name="chordTip",
#    constraint_name="Geometry Check",
#    value=chordTipLower,
#    constraint_type="ineq",
#    positive=True
# )

# mads_scenario.scenario.add_constraint(
#    output_name="checkGeometry",
#    constraint_name="Geometry Check",
#    value=0.0,
#    constraint_type="ineq",
# )


algo_options = {
    "xtol_rel": 1e-7,
    "xtol_abs": 1e-7,
    "ftol_rel": 1e-4,
    "ftol_abs": 1e-4,
    "ineq_tolerance": 1e-2,
}

for sub_sc in sub_disciplines[0:3]:
    sub_sc.default_inputs = {
        "max_iter": 40,
        "algo": "NLOPT_COBYLA",
        "algo_options": algo_options,
    }

# preprocess architecture
mads_scenario.scenario.xdsmize(show_html=False)
gemseo.generate_n2_plot(sub_disciplines, save=True, show=False)

# set optimization inputs back up
# mads_scenario.scenario.set_optimization_history_backup("my_data.h5")

with open("pareto_data.txt", "w") as myfile:
    myfile.write(
        "valid(0:No/1:Yes);Span0;span1;ChordRoot;InitialEff;FinalEff;WingMass(kg);Range(km)\n"
    )

logger.info("DUST Parameters")
logger.info("    N Panels Inner  = " + str(nPanelsInner))
logger.info("    N Panels Outer  = " + str(nPanelsOuter))
logger.info("    N Panels Chord  = " + str(nPanelsChord))
logger.info("    N Time Step     = " + str(dustNTempSteps))
logger.info("    N Time Pre Calc = " + str(dustNStep0Calc))

nSpan = 6
nKink = 9
nChord = 6

[nSamples, samples] = GenerateSamples(nSpan, nKink, nChord)


mads_scenario.scenario.execute(
    input_data={
        ##"max_iter": 500,
        ##"algo": "NLOPT_COBYLA", #"SLSQP",
        ##"algo_options": {
        ##    "eq_tolerance": 0.1,
        ##    "ineq_tolerance": 0.1,
        ##},
        ##"n_samples": 100,
        ##"algo": "lhs",
        ##"algo_options": {"n_processes": 1}
        "n_samples": nSamples,
        "algo": "CustomDOE",
        "algo_options": {"samples": samples},
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
