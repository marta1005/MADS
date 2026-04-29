import logging
import os
import sys

sys.path.insert(
    0,
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")),
)
sys.path.insert(
    0,
    os.path.abspath(os.path.join(os.path.dirname(__file__), "../../pyxrotor")),
)
sys.path.insert(
    0,
    os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),
            "../src/solvers/weight_and_balance/ampet",
        ),
    ),
)

import inspect

import gemseo
import numpy as np
from assembly import (
    AerodynamicPolarData,
    AirfoilFile,
    AllVariables,
    Assembly,
    CombustionEngine,
    Configuration,
    ElectricEngine,
    ElectronicSpeedController,
    Environment,
    FlightControlSystem,
    FuelTank,
    Fuselage,
    MissionAssembly,
    MissionSegment,
    PropulsionSystem,
    Section,
    Span,
    Wing,
)
from disciplines.aerodynamics import Aerodynamics
from disciplines.disciplines import UserDefined
from disciplines.mission import Mission
from disciplines.propulsion import Propulsion
from disciplines.weight_and_balance import WeightAndBalance
from scenario import MADSScenario, Variable
from utilities.gemseo_utilities import *

# Logger
gemseo.configure_logger(
    level=logging.INFO,
    filename="gemseo.log",
    filemode="w",
)

############################ Missing disciplines / TODO @ TK
# - include cD of the fuselage depending on angle of attack
# - stability
# - estimate fuselage dimension via semi-empirical formula
# - implement different fuselage masss estimation formula
# - implement transition hover - cruise in mission simulation?
# - take into account generator
# - implement parallel hybrid propulsion system capability
# - check assumptions
# - check sensitivity wrt SFC...
# - implememnt capability to analyze also the lift-and-cruise configuration


# TODO @Tim:
# in order to allow flexible ad-hoc switching of parameters, which are currently inputted as "options" to the solvers, to make them (Random)Variables:
# at run-time make a check which of the (Bass)Variables in this file are associated with a component/attribute in the assembly.
# all unaffiliated BaseVariables will be defined in a ghost class that is created at run-time in the assembly to store all unaffiliated variables.
# these parameters will still remain in the "options" for the resepctive solver. In the solver, a check is performed if any of the options is a BaseVariable
# if so, they are added to the required variables list, and are retrieved from the ghost class --> ensure grammar update etc....
# ensure that the option Variables are correctly updated!


# settings
n_samples = 200
m_tom_cutoff = 100000

# Requirements
airspeed_design = 160 / 3.6  # m/s
ultimate_load_factor = 4.0  # ultimate load of the aircraft [-]
m_payload = 250.0  # [kg]
angle_of_attack_max_allowed = 10  # [deg]

# initial assumptions
initial_mass = 2000

# Design assumptions
rotorRadius_assumption = 2 * np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5])
wing_aspect_ratio_lower_wing = 8.9  # [-]
wing_aspect_ratio_upper_wing = 8.9  # [-]
wing_area_lower_wing = 5.8  # m^2
wing_area_upper_wing = 5.8  # m^2
wing_total_span_lower_wing = np.sqrt(
    wing_area_lower_wing * wing_aspect_ratio_lower_wing,
)
wing_half_span_upper_wing = Variable(
    "wing_half_span_upper_wing",
    3.0,
    lb=2.0,
    ub=5.0,
)

wing_chord_lower_wing = wing_area_lower_wing / wing_total_span_lower_wing  # m
wing_chord_upper_wing = 1.0
wing_sweep_25 = 0.0  # wing sweep at 25% chord line [deg]
number_of_wings = 2

# Aerodynamic assumptions
K_polhamus = 1.0  # [-]
alphaRangeDeg = np.arange(-3.0, 11.0, 1)  # [rad]
oswald_factor = 1 / 1.02
configuration_aerodynamics_reference_area = wing_area_lower_wing
fuselage_reference_area = (
    wing_area_lower_wing  # wing_area_lower_wing is used as a reference area
)
interference_factor = 1.15  # estimation

# Fuselage geometry assumptions
fuselage_length = 3.037  # [m]
fuselage_wetted_area = 12.4  # fuselage wetted area [m^2]
fuselage_width = 1.309  # width of the fuselage [m]
fuselage_height = 1.041  # height of the fuselage [m]
tail_arm_length = 2.0  # length of the tail arm [m]
fuselage_volume_pressurized = (
    0.0  # volume of pressurized cabin [m^3] (0 because of low altitude)
)
fuselage_maximum_pressure_differential = 0.0  # [Pa] (0 because of low altitude)

# Assumptions / inputs propulsion system sizing
propulsion_system_architecture = "serial"  # so far, only serial is implemented!
engine_combustion_efficiency = 0.5  # [-]
engine_electric_efficiency = 0.8  # [-]
power_specific_engine_mass_combustion = 7.5e-04  # power specific engine mass combustion system [kg/W] (from Rotax 916: mass: 85.5kg, peak power: 117kW)
power_specific_engine_volume_combustion = (
    1.4e-06  # power specific engine volume combustion [m**3/W]
)
fuelTankEmptyMassFraction = 0.15  # [-]
fuelDensity = 800.0  # [kg/m^3]
fuelReserve = 1.3  # factor to take into account fuel reserve
batteryReserve = 1.3  # factor to take into account fuel reserve
number_of_engines_combustion = 1
number_of_engines_electric = len(rotorRadius_assumption)
escEfficiency = 0.95  # efficiency of the electronic speed controller (ESC) [-]
method_components = "feger"  # method used for the component mass estimation of the electric propulsion system
escMassDensity = 2700.0  # mass density of the ESC [kg/m^3]
batteryMassDensity = 2200.0  # battery mass density [kg/m^3]
powerSpecificESCMass = 0.0001  # power specific ESC mass [kg/W]
power_specific_engine_mass_electric = (
    0.00027  # power specific engine mass (electric motor) [kg/W]
)
motor_mass_density_electric = 4000  # electric motor mass density [kg/m^3]
thrustSafetyFactor = 1.2  # -
propellerEfficiency = 0.6  # [-]
methodRotor = "gundlach"  # calculation method for the rotor mass estimation
number_of_blades = 2  # number of blades per rotor [-]

# Assumptions / inputs for mission analysis
energy_fraction = {}
n_v = -1.0
altitude_start = 6000 * 0.3048  # m
altitude_start_cruise = altitude_start + 300 * 0.3048  # m
altitude_landing = altitude_start
propellant_type = "kerosene-battery"
eta_prop1_hover = 0.3  # efficiency of prop system 1 during hover [-]
eta_prop1_cruise = 0.3  # efficiency of prop system 1 during cruise [-]
eta_prop1_loiter = 0.3  # efficiency of prop system 1 during cruise [-]
eta_prop2_hover = 0.45  # efficiency of prop system 2 during hover [-]
eta_prop2_cruise = 0.45  # # efficiency of prop system 2 during cruise [-]
eta_prop2_loiter = 0.45  # # efficiency of prop system 2 during cruise [-]
Easterisk = (
    7.2 * 10**5
)  # specific energy of battery [Ws/kg]; value taken from Evolito: 181.82 * 3600 (old: 7.2*10**5)
SFC_hover = (
    1.2 * 10**-7
)  # specific fuel consumption during hover [kg/Ws] [Raymer: 6.8*10**-8] [Gundlach: two-stroke: 1.27*10**-7 - 3.38*10**-7; four-stroke: 8.44*10**-8]
SFC_cruise = (
    1.2 * 10**-7
)  # specific fuel consumption during cruise [kg/Ws] [Raymer: 6.8*10**-8] [Gundlach: two-stroke: 1.27*10**-7 - 3.38*10**-7; four-stroke: 8.44*10**-8]
SFC_loiter = (
    1.2 * 10**-7
)  # specific fuel consumption during loiter [kg/Ws] [Raymer: 6.8*10**-8] [Gundlach: two-stroke: 1.27*10**-7 - 3.38*10**-7; four-stroke: 8.44*10**-8]
transition_time = 60
hover_time_takeoff_landing = 15.0  # s
hover_time_after_takeoff = 90.0  # s
endurance_hover_start = (
    hover_time_takeoff_landing + hover_time_after_takeoff + transition_time
)  # s
endurance_hover_landing = transition_time + hover_time_takeoff_landing  # s
endurance_loiter = (
    120.0 + 30
)  # s (120s: direct requirement, 30s: estimated derived time for descend to hover altitude)
range = 250000.0  # m

# Optimization variables
hybridization_hover = Variable(
    "hybridization_hover",
    0.8,
    lb=0.3,
    ub=0.999,
    options={"roskam": {"components": ["mission"]}},
)
hybridization_cruise = Variable(
    "hybridization_cruise",
    0.005,
    lb=0.001,
    ub=0.5,
    options={"roskam": {"components": ["mission"]}},
)

# Output variables
cD0_lower_wing = Variable(
    "cD0_lower_wing",
    0.005,
    output_type="cd0",
    options={"component_buildup_drag": {"associated_component_names": ["lower_wing"]}},
)
cD0_upper_wing = Variable(
    "cD0_upper_wing",
    0.005,
    output_type="cd0",
    options={"component_buildup_drag": {"associated_component_names": ["upper_wing"]}},
)
cD0_fuselage = Variable(
    "cD0_fuselage",
    0.005,
    output_type="cd0",
    options={"component_buildup_drag": {"associated_component_names": ["fuselage"]}},
)
cL_array_lower_wing = Variable(
    "cL_array_lower_wing",
    np.arange(0, 1.3, 0.1),
    output_type="cl",
    options={"aero_lf": {"wings": ["lower_wing"]}},
)
cD_array_lower_wing = Variable(
    "cD_array_lower_wing",
    np.arange(0, 1.3, 0.1),
    output_type="cd",
    options={"aero_lf": {"wings": ["lower_wing"]}},
)
cL_array_upper_wing = Variable(
    "cL_array_upper_wing",
    np.arange(0, 1.3, 0.1),
    output_type="cl",
    options={"aero_lf": {"wings": ["upper_wing"]}},
)
cD_array_upper_wing = Variable(
    "cD_array_upper_wing",
    np.arange(0, 1.3, 0.1),
    output_type="cd",
    options={"aero_lf": {"wings": ["upper_wing"]}},
)

m_tom = Variable("m_tom", 2000.0, output_type="m_tom")  # kg
cL_array_configuration = Variable(
    "cL_array_configuration",
    np.arange(0, 1.3, 0.1),
    output_type="cL_array_configuration",
)  # [-] ############### TODO @Tim: ANPASSEN !!!!!
cD_array_configuration = Variable(
    "cD_array_configuration",
    np.arange(0, 1.3, 0.1),
    output_type="cD_array_configuration",
)  # [-] ############### TODO @Tim: ANPASSEN !!!!!

fuel_used_total = Variable("fuel_used_total", 100.0, output_type="fuel_used_total")
mass_lower_wing = Variable(
    "mass_lower_wing",
    20.0,
    output_type="mass_wing",
    options={"raymer_general_aviation_mass_wing": {"wings": ["lower_wing"]}},
)
mass_upper_wing = Variable(
    "mass_upper_wing",
    20.0,
    output_type="mass_wing",
    options={"raymer_general_aviation_mass_wing": {"wings": ["upper_wing"]}},
)
m_fuselage = Variable("m_fuselage", 50.0, output_type="mass_fuselage")
m_controls = Variable("m_controls", 15.0, output_type="mass_controls")
battery_mass_used_total = Variable(
    "battery_mass_used_total",
    50.0,
    output_type="battery_mass_used_total",
)
# required_engine_power_combustion = Variable("required_engine_power_combustion", value=np.array([10.0]),output_type="required_engine_power_combustion")
engine_mass_combustion = Variable(
    "engine_mass_combustion",
    10.0,
    output_type="engine_mass_combustion",
)
# zHeight_engine = Variable(name="zHeight_engine_combustion", value=np.array([10.0]),output_type="zHeight_engine")
# yWidth_engine = Variable(name="yWidth_engine_combustion", value=np.array([10.0]),output_type="yWidth_engine")
# xLength_engine = Variable(name="xLength_engine_combustion", value=np.array([10.0]),output_type="xLength_engine")
tank_mass_empty_combustion = Variable(
    "tank_mass_empty_combustion",
    10.0,
    output_type="tank_mass_empty_combustion",
)

# zHeight_fuelTank = Variable(name="zHeight_fuelTank_combustion", value=np.array([10.0]),output_type="zHeight_fuelTank")
# yWidth_fuelTank = Variable(name="yWidth_fuelTank_combustion", value=np.array([10.0]),output_type="yWidth_fuelTank")
# xLength_fuelTank = Variable(name="xLength_fuelTank_combustion", value=np.array([10.0]),output_type="xLength_fuelTank")
rotor_mass = Variable(
    "rotor_mass",
    np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]),
    output_type="rotor_mass",
)
# required_engine_power_electric = Variable(name="required_engine_power_electric", np.array([10.0]),output_type="required_engine_power_electric")
engine_mass_electric = Variable(
    "engine_mass_electric",
    10.0,
    output_type="engine_mass_electric",
)
# motor_x_length = Variable(name="motor_x_length_electric", value=np.array([10.0]),output_type="motor_x_length")
# motor_y_width = Variable(name="motor_y_width_electric", value=np.array([10.0]),output_type="motor_y_width")
# motor_z_height = Variable(name="motor_z_height_electric", value=np.array([10.0]),output_type="motor_z_height")
esc_mass = Variable("esc_mass", 10.0, output_type="esc_mass_electric")
# esc_x_length = Variable(name="esc_x_length_electric", value=np.array([10.0]),output_type="esc_x_length")
# esc_y_width = Variable(name="esc_y_width_electric", value=np.array([10.0]),output_type="esc_y_width")
# esc_z_height = Variable(name="esc_z_height_electric", value=np.array([10.0]),output_type="esc_z_height")
# battery_x_length = Variable(name="battery_x_length_electric", value=np.array([10.0]),output_type="battery_x_length")
# battery_y_width = Variable(name="battery_y_width_electric", value=np.array([10.0]),output_type="battery_y_width")
# battery_z_height = Variable(name="battery_z_height_electric", value=np.array([10.0]),output_type="battery_z_height")
fuel_mass_wing = Variable("fuel_mass_wing", 50.0, output_type="fuel_mass_wing")

angle_of_attack_cruise_1 = Variable(
    "angle_of_attack_cruise_1",
    5.0,
    output_type="angle_of_attack",
    options={"roskam": {"segment_name": ["cruise_1"]}},
)
angle_of_attack_loiter_1 = Variable(
    "angle_of_attack_loiter_1",
    5.0,
    output_type="angle_of_attack",
    options={"roskam": {"segment_name": ["loiter_1"]}},
)
lift_to_drag_ratio_cruise_1 = Variable(
    "lift_to_drag_ratio_cruise_1",
    10.0,
    output_type="lift_to_drag_ratio",
    options={"roskam": {"segment_name": ["cruise_1"]}},
)
lift_to_drag_ratio_loiter_1 = Variable(
    "lift_to_drag_ratio_loiter_1",
    10.0,
    output_type="lift_to_drag_ratio",
    options={"roskam": {"segment_name": ["loiter_1"]}},
)
lift_coefficient_cruise_1 = Variable(
    "lift_coefficient_cruise_1",
    1.0,
    output_type="lift_coefficient",
    options={"roskam": {"segment_name": ["cruise_1"]}},
)
lift_coefficient_loiter_1 = Variable(
    "lift_coefficient_loiter_1",
    1.0,
    output_type="lift_coefficient",
    options={"roskam": {"segment_name": ["loiter_1"]}},
)


objective = Variable("objective", 1.0)
angle_of_attack_constraint_cruise = Variable("angle_of_attack_constraint_cruise", 1.0)
angle_of_attack_constraint_loiter = Variable("angle_of_attack_constraint_loiter", 1.0)
lift_coefficient_constraint_cruise = Variable("lift_coefficient_constraint_cruise", 1.0)
lift_coefficient_constraint_loiter = Variable("lift_coefficient_constraint_loiter", 1.0)

# All variables
variables = [
    # Optimization
    hybridization_hover,
    hybridization_cruise,
    wing_half_span_upper_wing,
    # Internal
    m_tom,
    cL_array_lower_wing,
    cD_array_lower_wing,
    cL_array_upper_wing,
    cD_array_upper_wing,
    cL_array_configuration,
    cD_array_configuration,
    fuel_used_total,
    fuel_mass_wing,
    mass_lower_wing,
    mass_upper_wing,
    m_fuselage,
    battery_mass_used_total,
    engine_mass_combustion,
    tank_mass_empty_combustion,
    rotor_mass,
    engine_mass_electric,
    esc_mass,
    lift_to_drag_ratio_cruise_1,
    lift_to_drag_ratio_loiter_1,
    m_controls,
    angle_of_attack_cruise_1,
    angle_of_attack_loiter_1,
    lift_coefficient_cruise_1,
    lift_coefficient_loiter_1,
    cD0_lower_wing,
    cD0_upper_wing,
]

all_variables = AllVariables("all variables", rotor_mass=rotor_mass)

# Components and assembly
env = Environment(
    "env",
    height=altitude_start_cruise,
    alpha=0.0,
    beta=0.0,
    speed=airspeed_design,
)

# Compute mach number for polhamus equation
# Compute density in current altitude
speed_of_sound = env.sound_speed
density_cruise = env.density
dynamic_viscosity_cruise = env.dyn_viscosity
mach = airspeed_design / speed_of_sound
dynamic_pressure_cruise = 0.5 * env.density * airspeed_design**2

reynolds_number_cruise_lower_wing = (
    density_cruise * airspeed_design * wing_chord_lower_wing / dynamic_viscosity_cruise
)
reynolds_number_cruise_upper_wing = (
    density_cruise * airspeed_design * wing_chord_upper_wing / dynamic_viscosity_cruise
)

section_lower = Section(
    "lower_sec",
    airfoil=AirfoilFile("lower_foil", filename="naca653218.dat"),  ### to be updated
    chord=wing_chord_lower_wing,
    twist=0.0,
)
wing_lower = Wing(
    name="lower_wing",
    symmetry=True,
    alpha=alphaRangeDeg,
    fuel_mass_in_wing=fuel_mass_wing,
    portion_laminar_flow=0.1,
    cd0=cD0_lower_wing,
    options={
        "dust": {
            "n_elem": 25,
            "elem_type": "uniform",
        },
        "aero_lf": {  # TODO @Tim: these are not implemented as options in the lofi aero tool
            "k2": 1.0,  # initial guess for the  k2-factor in the aerodynamic drag polar [-]
            "k_polhamus": 1.0,  # Airfoil/Pressure impact [-]
            "oswald_factor": oswald_factor,
        },
        "component_buildup_drag": {
            "surface_material": "camouflage_paint_on_aluminum",
        },
    },
    sections=[
        section_lower,
        section_lower,
    ],
    spans=[
        Span(
            "lower_span",
            length=wing_total_span_lower_wing / 2,
            sweep=wing_sweep_25,
            dihed=0.0,
            options={
                "aero_lf": {
                    "n_elem": 35,
                    "elem_type": "uniform",
                },
            },
        ),
    ],
)

section_upper = Section(
    "upper_sec",
    airfoil=AirfoilFile("upper_foil", filename="naca653218.dat"),  ### to be updated
    chord=wing_chord_lower_wing,
    twist=0.0,
)
wing_upper = Wing(
    name="upper_wing",
    symmetry=True,
    alpha=alphaRangeDeg,
    fuel_mass_in_wing=fuel_mass_wing,
    portion_laminar_flow=0.1,
    cd0=cD0_upper_wing,
    options={
        "dust": {
            "n_elem": 25,
            "elem_type": "uniform",
        },
        "aero_lf": {  ## TODO @Tim: these are not implemented as options in the lofi aero tool
            "k2": 1.0,  # initial guess for the  k2-factor in the aerodynamic drag polar [-]
            "k_polhamus": 1.0,  # Airfoil/Pressure impact [-]
            "oswald_factor": oswald_factor,
        },
        "component_buildup_drag": {
            "surface_material": "camouflage_paint_on_aluminum",
        },
    },
    sections=[
        section_upper,
        section_upper,
    ],
    spans=[
        Span(
            "upper_span",
            length=wing_half_span_upper_wing,
            sweep=wing_sweep_25,
            dihed=0.0,
            options={
                "aero_lf": {
                    "n_elem": 35,
                    "elem_type": "uniform",
                },
            },
        ),
    ],
)

mission = MissionAssembly(
    name="mission",
    fuel_used_total=fuel_used_total,
    battery_mass_used_total=battery_mass_used_total,
    h2o_used_total=0.0,
    mission_segments=[
        MissionSegment(
            name="hover_1",
            type="hover",
            range=0.0,
            mass_start=m_tom,
            mass_end=2000.0,  ###### just a dummy value as initial value
            endurance=endurance_hover_start,
            airspeed_start=0.0,
            airspeed_end=0.0,
            altitude_start=altitude_start,
            altitude_end=altitude_start_cruise,
            lift_drag_ratio=0.0,
            angle_of_attack=0.0,
            fuel_used=0.0,  # initial value will be changed during calculation
            battery_mass_used=0.0,  # initial value will be changed during calculation
            h2o_used=0.0,  # initial value will be changed during calculation
            hybridization=hybridization_hover,
        ),
        MissionSegment(
            name="cruise_1",
            type="cruise",
            range=range,
            mass_start=2000.0,  ###### just a dummy value as initial value
            mass_end=1800.0,  ###### just a dummy value as initial value
            endurance=500.0,  ###### just a dummy value as initial value
            airspeed_start=airspeed_design,
            airspeed_end=airspeed_design,
            altitude_start=altitude_start_cruise,
            altitude_end=altitude_start_cruise,  # initial value will be changed during calculation
            lift_drag_ratio=lift_to_drag_ratio_cruise_1,
            angle_of_attack=angle_of_attack_cruise_1,
            fuel_used=0.0,  # initial value will be changed during calculation
            battery_mass_used=0.0,  # initial value will be changed during calculation
            h2o_used=0.0,  # initial value will be changed during calculation
            hybridization=hybridization_cruise,
        ),
        MissionSegment(
            name="loiter_1",
            type="loiter",
            range=range,
            mass_start=initial_mass,  ###### just a dummy value as initial value
            mass_end=initial_mass * 0.8,  ###### just a dummy value as initial value
            endurance=endurance_loiter,  ###### just a dummy value as initial value
            airspeed_start=airspeed_design,
            airspeed_end=airspeed_design,
            altitude_start=altitude_landing,
            altitude_end=altitude_landing,  # initial value will be changed during calculation
            lift_drag_ratio=lift_to_drag_ratio_loiter_1,
            angle_of_attack=angle_of_attack_loiter_1,
            fuel_used=0.0,  # initial value will be changed during calculation
            battery_mass_used=0.0,  # initial value will be changed during calculation
            h2o_used=0.0,  # initial value will be changed during calculation
            hybridization=hybridization_cruise,
        ),
        MissionSegment(
            name="hover_2",
            type="hover",
            range=range,
            mass_start=initial_mass,  ###### just a dummy value as initial value
            mass_end=initial_mass * 0.8,  ###### just a dummy value as initial value
            endurance=endurance_hover_landing,  ###### just a dummy value as initial value
            airspeed_start=0.0,
            airspeed_end=0.0,
            altitude_start=altitude_landing,  # initial value will be changed during calculation
            altitude_end=altitude_landing,  # initial value will be changed during calculation
            lift_drag_ratio=0.0,
            angle_of_attack=0.0,
            fuel_used=0.0,  # initial value will be changed during calculation
            battery_mass_used=0.0,  # initial value will be changed during calculation
            h2o_used=0.0,  # initial value will be changed during calculation
            hybridization=hybridization_hover,
        ),
    ],
)

propulsionSystem = PropulsionSystem(
    name="propulsionSystem",
    type="kerosene-battery",
    engine_power=0.0,
    shaft_power=0.0,
    electric_engine_power=0.0,
    hybrid_factor_engine=None,
    hybrid_factor_power=None,
    total_electrical_power=None,
    total_combustion_power=None,
    electrical_power_from_fc=None,
    electrical_power_from_battery=None,
)

fuselage = Fuselage(
    "fuselage",
    wetted_area=fuselage_wetted_area,
    maximum_width=fuselage_width,
    maximum_height=fuselage_height,
    length=fuselage_length,
    volume_pressurized_cabin=fuselage_volume_pressurized,
    maximum_fuselage_pressure_differential=fuselage_maximum_pressure_differential,
    portion_laminar_flow=0.0,
    options={
        "component_buildup_drag": {
            "surface_material": "camouflage_paint_on_aluminum",
        },
    },
)

fuel_tank = FuelTank(
    "fuel_tank",
    structure_mass=tank_mass_empty_combustion,
)

configuration = Configuration(
    "configuration",
    m_tom=m_tom,
    ultimate_load_factor=ultimate_load_factor,
    aerodynamic_polar=AerodynamicPolarData(
        "aero polar data",
        alpha=alphaRangeDeg,
        lift_coefficient=cL_array_configuration,
        drag_coefficient=cD_array_configuration,
    ),
)

flight_control_system = FlightControlSystem(
    "flight_control_system",
    mass=m_controls,
)

electronic_speed_controller = ElectronicSpeedController(
    "electronic_speed_controller",
    mass=esc_mass,
)

electric_engine = ElectricEngine(
    "electric_engine",
    mass=engine_mass_electric,
)

combustion_engine = CombustionEngine(
    "combustion_engine",
    mass=engine_mass_combustion,
)

assembly = Assembly(
    "system",
    [
        wing_lower,
        wing_upper,
        mission,
        propulsionSystem,
        fuselage,
        fuel_tank,
        flight_control_system,
        electronic_speed_controller,
        electric_engine,
        combustion_engine,
        configuration,
        env,
        all_variables,
    ],  # TK: all_variables here is a dummy component that contains all variables not yet linked to a "real" component in the assembly
    variables,
)

disc_aerodynamics = Aerodynamics(
    "Aerodynamics (wings)",
    inputs=variables,
    outputs=[
        cL_array_lower_wing,
        cL_array_upper_wing,
        cD_array_lower_wing,
        cD_array_upper_wing,
    ],
    components=[wing_lower, wing_upper, env],
    solver_name="aero_lf",
    options={},
)

disc_drag = Aerodynamics(
    "Aerodynamics (drag)",
    inputs=variables,
    outputs=[cD0_lower_wing, cD0_upper_wing, cD0_fuselage],
    components=assembly.components,
    solver_name="component_buildup_drag",
    options={"wing_name_serving_for_reference_area": "lower_wing"},
)

# disc_propulsion_rubber_engine_uav_sizing
disc_propulsion_rubber_engine_uav_sizing = Propulsion(
    "Propulsion",
    inputs=variables,
    outputs=[
        engine_mass_combustion,
        tank_mass_empty_combustion,
        rotor_mass,
        engine_mass_electric,
        esc_mass,
    ],
    options={
        "thrustSafetyFactor": thrustSafetyFactor,
        "rotorRadius": rotorRadius_assumption,
        "propellerEfficiency": propellerEfficiency,
        "methodRotor": methodRotor,
        "number_of_blades": number_of_blades,
        "engine_combustion_efficiency": engine_combustion_efficiency,
        "engine_electric_efficiency": engine_electric_efficiency,
        "power_specific_engine_mass_combustion": power_specific_engine_mass_combustion,
        "fuelTankEmptyMassFraction": fuelTankEmptyMassFraction,
        "fuelDensity": fuelDensity,
        "number_of_engines_combustion": number_of_engines_combustion,
        "escEfficiency": escEfficiency,
        "method_components": method_components,
        "power_specific_engine_volume_combustion": power_specific_engine_volume_combustion,
        "number_of_engines_electric": number_of_engines_electric,
        "escMassDensity": escMassDensity,
        "batteryMassDensity": batteryMassDensity,
        "powerSpecificESCMass": powerSpecificESCMass,
        "power_specific_engine_mass_electric": power_specific_engine_mass_electric,
        "motor_mass_density_electric": motor_mass_density_electric,
        "propulsion_system_architecture": propulsion_system_architecture,
    },
    components=assembly.components,
    solver_name="rubber_engine_uav_sizing",
)

# disc_w_and_b_mass_wing_general_aviation_raymer
disc_w_and_b_mass_wing_general_aviation_raymer = WeightAndBalance(
    "Mass (wings)",
    inputs=variables,
    outputs=[mass_lower_wing, mass_upper_wing],
    components=assembly.components,
    solver_name="raymer_general_aviation_mass_wing",
    options={"mass_fuel_in_wing": fuel_mass_wing},
)

# disc_w_and_b_mass_fuselage_general_aviation_raymer
disc_w_and_b_mass_fuselage_general_aviation_raymer = WeightAndBalance(
    "Mass (fuselage)",
    inputs=variables,
    outputs=[m_fuselage],
    components=assembly.components,
    solver_name="raymer_general_aviation_mass_fuselage",
    options={
        "length_tail_moment_arm": tail_arm_length,
    },  # TODO @Tim: put in the assembly once coordinate systems logic is established
)

# disc_w_and_b_mass_controls_general_aviation_raymer
disc_w_and_b_mass_controls_general_aviation_raymer = WeightAndBalance(
    "Mass (controls)",
    inputs=variables,
    outputs=[m_controls],
    components=assembly.components,
    solver_name="raymer_general_aviation_mass_controls",
    options={"wing_names_contributing_to_total_span": ["lower_wing", "upper_wing"]},
)

# disc_mission_roskam
roskam_outputs = [
    fuel_used_total,
    battery_mass_used_total,
    lift_to_drag_ratio_cruise_1,
    lift_to_drag_ratio_loiter_1,
    angle_of_attack_cruise_1,
    angle_of_attack_loiter_1,
    lift_coefficient_cruise_1,
    lift_coefficient_loiter_1,
]

disc_mission_roskam = Mission(
    "Mission",
    inputs=variables,
    outputs=roskam_outputs,
    components=assembly.components,
    solver_name="roskam",
    options={
        "wing_name_serving_for_reference_area": "lower_wing",
        "energy_fraction": {},
        "n_v": n_v,
        "fuelReserve": fuelReserve,  # TODO @Tim: put in assembly
        "batteryReserve": batteryReserve,  # TODO @Tim: put in assembly
        "kwargs": {
            "thrustSafetyFactor": thrustSafetyFactor,  # TODO @Tim: put in assembly
            "rotorRadius": rotorRadius_assumption,  # TODO @Tim: put in assembly
            "eta_prop1_hover": eta_prop1_hover,  # TODO @Tim: put in assembly
            "eta_prop2_hover": eta_prop2_hover,  # TODO @Tim: put in assembly
            "SFC_hover": SFC_hover,  # TODO @Tim: put in assembly
            "SFC_cruise": SFC_cruise,  # TODO @Tim: put in assembly
            "SFC_loiter": SFC_loiter,  # TODO @Tim: put in assembly
            "eta_prop1_cruise": eta_prop1_cruise,  # TODO @Tim: put in assembly
            "eta_prop2_cruise": eta_prop2_cruise,  # TODO @Tim: put in assembly
            "eta_prop1_loiter": eta_prop1_loiter,  # TODO @Tim: put in assembly
            "eta_prop2_loiter": eta_prop2_loiter,  # TODO @Tim: put in assembly
            "Easterisk": Easterisk,  # TODO @Tim: put in assembly
        },
    },
)


def total_mass(
    battery_mass_used_total,
    fuel_used_total,
    mass_lower_wing,
    mass_upper_wing,
    m_fuselage,
    rotor_mass,
    engine_mass_electric,
    engine_mass_combustion,
    esc_mass,
    m_controls,
    tank_mass_empty_combustion,
):
    rotor_mass_total = sum(rotor_mass)
    m_tom = np.atleast_1d(
        battery_mass_used_total
        + fuel_used_total
        + mass_lower_wing
        + mass_upper_wing
        + m_fuselage
        + m_payload
        + rotor_mass_total
        + number_of_engines_combustion * engine_mass_combustion
        + number_of_engines_electric * (engine_mass_electric + esc_mass)
        + m_controls
        + tank_mass_empty_combustion,
    )

    fuel_mass_wing = np.atleast_1d(fuel_used_total / number_of_wings)

    # feasibility check must be INSIDE inner convergence loop
    if m_tom > m_tom_cutoff:
        m_tom = np.atleast_1d(
            m_tom_cutoff + battery_mass_used_total / m_tom + fuel_used_total / m_tom,
        )

    return (
        m_tom,
        fuel_mass_wing,
    )  # TODO @Tim: change once propulsion discipline has been updated


# Coupling discipline
disc_coupling_sum_up_masses = UserDefined(
    "Total_mass",
    expression=total_mass,
    inputs=[
        globals()[name]
        for name in list(inspect.signature(total_mass).parameters.keys())
    ],  # TODO @Tim: this needs to be moved to UserDefined()
    outputs=[
        m_tom,
        fuel_mass_wing,
    ],  # TODO @Tim: this should not be hardcoded but put somewhere in UserDefined()
)


def configuration_aerodynamics(
    cL_array_lower_wing,
    cD_array_lower_wing,
    cL_array_upper_wing,
    cD_array_upper_wing,
    cD0_fuselage,
):
    # wing_area_lower_wing is used as a reference area for the configuration aerodynamics
    cL_array_configuration = np.atleast_1d(
        cL_array_lower_wing
        * wing_area_lower_wing
        / configuration_aerodynamics_reference_area
        + cL_array_upper_wing
        * wing_area_upper_wing
        / configuration_aerodynamics_reference_area,
    )
    cD_array_configuration = np.atleast_1d(
        interference_factor
        * (
            cD_array_lower_wing
            * wing_area_lower_wing
            / configuration_aerodynamics_reference_area
            + cD_array_upper_wing
            * wing_area_upper_wing
            / configuration_aerodynamics_reference_area
            + cD0_fuselage
            * fuselage_reference_area
            / configuration_aerodynamics_reference_area
        ),
    )

    return (cL_array_configuration, cD_array_configuration)


# Coupling discipline
disc_coupling_configuration_aerodynamics = UserDefined(
    "Config aero",
    expression=configuration_aerodynamics,
    inputs=[
        globals()[name]
        for name in list(
            inspect.signature(configuration_aerodynamics).parameters.keys(),
        )
    ],  # TODO @Tim this
    outputs=[cL_array_configuration, cD_array_configuration],
)


# Objective/constraints discipline
def constraint_function(
    angle_of_attack_cruise_1,
    angle_of_attack_loiter_1,
    cL_array_configuration,
    lift_coefficient_cruise_1,
    lift_coefficient_loiter_1,
):
    # limit for angle of attack during cruise
    angle_of_attack_constraint_cruise = (
        angle_of_attack_cruise_1 - angle_of_attack_max_allowed
    )  # this must be < 0 (this is ensured by setting the constraint type to ineq)
    angle_of_attack_constraint_loiter = (
        angle_of_attack_loiter_1 - angle_of_attack_max_allowed
    )  # this must be < 0 (this is ensured by setting the constraint type to ineq)
    lift_coefficient_constraint_cruise = lift_coefficient_cruise_1 - max(
        cL_array_configuration,
    )  # this must be < 0 (this is ensured by setting the constraint type to ineq)
    lift_coefficient_constraint_loiter = lift_coefficient_loiter_1 - max(
        cL_array_configuration,
    )  # this must be < 0 (this is ensured by setting the constraint type to ineq)

    return (
        angle_of_attack_constraint_cruise,
        angle_of_attack_constraint_loiter,
        lift_coefficient_constraint_cruise,
        lift_coefficient_constraint_loiter,
    )  # ,battery_energy_feasible_constraint,fuel_energy_feasible_constraint


disc_objective = UserDefined(
    "Constraints",
    expression=constraint_function,
    inputs=[
        globals()[name]
        for name in list(inspect.signature(constraint_function).parameters.keys())
    ],
    # TODO @Tim this
    outputs=[
        angle_of_attack_constraint_cruise,
        angle_of_attack_constraint_loiter,
        lift_coefficient_constraint_cruise,
        lift_coefficient_constraint_loiter,
    ],
)

# Constraints
constraints = [
    {"name": angle_of_attack_constraint_cruise.name, "type": "ineq"},
    {"name": angle_of_attack_constraint_loiter.name, "type": "ineq"},
    {"name": lift_coefficient_constraint_cruise.name, "type": "ineq"},
    {"name": lift_coefficient_constraint_loiter.name, "type": "ineq"},
]

disciplines = [
    disc_drag,
    disc_aerodynamics,
    disc_coupling_configuration_aerodynamics,
    disc_propulsion_rubber_engine_uav_sizing,
    disc_w_and_b_mass_wing_general_aviation_raymer,
    disc_w_and_b_mass_fuselage_general_aviation_raymer,
    disc_w_and_b_mass_controls_general_aviation_raymer,
    disc_mission_roskam,
    disc_coupling_sum_up_masses,
    disc_objective,
]

gemseo.generate_n2_plot(
    disciplines,
    save=True,
    show=False,
)

# Parameter space
mads_scenario = MADSScenario()
mads_scenario.fill_parameter_space(variables)

# Scenario
mads_scenario.create_scenario(
    disciplines,
    #    formulation='DisciplinaryOpt', #'MDF',
    formulation="MDF",  #'MDF',
    inner_mda_name="MDAGaussSeidel",
    objective_name=m_tom.name,
    scenario_type="DOE",
    maximize_objective=False,
)

mads_scenario.add_constraints(constraints)

mads_scenario.scenario.xdsmize(show_html=True)

mads_scenario.scenario.set_differentiation_method(
    "finite_differences",
    step=1e-2,
)

# variables_to_be_observed = list(variables.keys())
# variables_to_be_observed.remove("m_tom")
# mads_scenario.scenario.add_observable(variables_to_be_observed)

# add internal variables that shall be observed: this allows to see their history afterwards. Needs to be one by one
# because the gemseo method add_observable() has an upper limit for the input arguments
for v in variables:
    if v.name not in [
        "m_tom",
        "angle_of_attack_constraint_cruise",
        "angle_of_attack_constraint_loiter",
        "hybridization_hover",
        "hybridization_cruise",
        "lift_coefficient_constraint_cruise",
        "lift_coefficient_constraint_loiter",
        "wing_half_span_upper_wing",
    ]:
        mads_scenario.scenario.add_observable(v.name)

mads_scenario.scenario.execute(
    input_data={
        "max_iter": 10,
        "algo": "LHS",  # "SLSQP",
        "n_samples": n_samples,
        "algo_options": {
            "eq_tolerance": 0.1,
            "ineq_tolerance": 0.001,
        },
    },
)

# create a dataset including all internal (observable) variables
opt_problem = mads_scenario.scenario.formulation.opt_problem
dataset = opt_problem.to_dataset("dataset_cargoUAS_LF")
# show dataset (e.g. the variable engine_mass_combustion)
dataset.get_view(variable_names=[engine_mass_combustion.name])

# save dataset to csv
dataset.to_csv(path_or_buf=r".\dataset_export_tim.csv")

# make plot of the LHS sampling result
# ----------
# Tricontour
# ----------
# Directly supply the unordered, irregularly spaced coordinates
# to tricontour.
import matplotlib

matplotlib.use("Qt5Agg")
import matplotlib.pyplot as plt
import numpy as np

# Make exemplary plots for visualization of uncertainty capabilities
x = dataset.get_view(variable_names=[hybridization_hover.name]).values.flatten()
y = dataset.get_view(variable_names=[hybridization_cruise.name]).values.flatten()
z = dataset.get_view(variable_names=[m_tom.name]).values.flatten()

# cut-off infeasible designs at 5000 kg to focus the plots on the interesting solutions
z[z > 5000] = 5000

z_constraint_angle_of_attack_constraint_cruise = (
    (dataset.get_view(variable_names=[angle_of_attack_constraint_cruise.name])).values
).flatten()
z_constraint_angle_of_attack_constraint_loiter = (
    (dataset.get_view(variable_names=[angle_of_attack_constraint_loiter.name])).values
).flatten()
z_constraint_lift_coefficient_constraint_cruise = (
    (dataset.get_view(variable_names=[lift_coefficient_constraint_cruise.name])).values
).flatten()
z_constraint_lift_coefficient_constraint_loiter = (
    (dataset.get_view(variable_names=[lift_coefficient_constraint_loiter.name])).values
).flatten()

# make constraints quasi-boolean: -1 (feasible) and 0 (infeasible)
z_constraint_angle_of_attack_constraint_cruise[
    z_constraint_angle_of_attack_constraint_cruise < 0
] = -1
z_constraint_angle_of_attack_constraint_loiter[
    z_constraint_angle_of_attack_constraint_loiter < 0
] = -1
z_constraint_lift_coefficient_constraint_cruise[
    z_constraint_lift_coefficient_constraint_cruise < 0
] = -1
z_constraint_lift_coefficient_constraint_loiter[
    z_constraint_lift_coefficient_constraint_loiter < 0
] = -1

z_constraint_angle_of_attack_constraint_cruise[
    z_constraint_angle_of_attack_constraint_cruise > 0
] = 0
z_constraint_angle_of_attack_constraint_loiter[
    z_constraint_angle_of_attack_constraint_loiter > 0
] = 0
z_constraint_lift_coefficient_constraint_cruise[
    z_constraint_lift_coefficient_constraint_cruise > 0
] = 0
z_constraint_lift_coefficient_constraint_loiter[
    z_constraint_lift_coefficient_constraint_loiter > 0
] = 0

fig, ax = plt.subplots()

ax.tricontour(x, y, z, levels=14, linewidths=0.5, colors="k")
cntr2 = ax.tricontourf(x, y, z, levels=14, cmap="RdBu_r")

ax.tricontour(
    x,
    y,
    z_constraint_angle_of_attack_constraint_cruise,
    levels=0,
    linewidths=2.0,
    colors="k",
    hatches=["-"],
)
# cntr2_2 = ax.tricontourf(x, y, -z_constraint_angle_of_attack_constraint_cruise, levels=0, cmap=plt.colormaps.get_cmap("gray_r"),alpha=.25, hatches=['/',"",""])

fig.colorbar(cntr2, ax=ax)
ax.plot(x, y, "ko", ms=3)
ax.set(xlim=(hybridization_hover.lb, 1), ylim=(0, hybridization_cruise.ub))
ax.set_title("m_tom obtained by LHS (%d points)" % n_samples)
ax.set_xlabel("hybridization_hover")
ax.set_ylabel("hybridization_cruise")

plt.subplots_adjust(hspace=0.5)
plt.show()


## Export components
# with open("solution.pickle", "wb") as f:
#    x_opt = mads_scenario.scenario.optimization_result.x_opt
#    x_opt = dict(zip(
#        mads_scenario.scenario.get_optim_variable_names(),
#        [np.array([x]) for x in x_opt],
#    ))
#
#    disc_coupling.evaluate(x_opt)
#    x_opt.update(disc_coupling.get_output_data())
#
#    assembly.variables = variables
#    assembly.parse_variables()
#    assembly.update_from_values(x_opt)
#
#    pickle.dump(assembly.components, f)

# Post Processing
mads_scenario.scenario.post_process(
    "OptHistoryView",
    save=True,
    show=False,
)


mads_scenario.scenario.post_process(
    "Correlations",
    save=True,
    show=False,
)

# plot history of objective function(s)
mads_scenario.scenario.post_process(
    "BasicHistory",
    variable_names=[m_tom.name],
    save=True,
    show=False,
    file_name="basic_history_objectives",
)

# plot history of constraint(s)
mads_scenario.scenario.post_process(
    "BasicHistory",
    variable_names=[
        angle_of_attack_constraint_cruise.name,
        angle_of_attack_constraint_loiter.name,
        lift_coefficient_constraint_cruise.name,
        lift_coefficient_constraint_loiter.name,
    ],
    save=True,
    show=False,
    file_name="basic_history_constraints",
)

mads_scenario.scenario.post_process(
    "VariableInfluence",
    fig_size=(20, 20),
    save=True,
    show=True,
)

# Robutness
mads_scenario.scenario.post_process("Robustness", save=True, show=False)
