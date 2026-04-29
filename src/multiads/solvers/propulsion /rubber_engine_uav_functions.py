import numpy as np

from solvers.weight_and_balance.gundlach import *

from utilities.gemseo_utilities import *


def rubber_engine_sizing_combustion_uav(
    power_specific_engine_mass_combustion,
    power_specific_engine_volume_combustion,
    fuelTankEmptyMassFraction,
    fuelDensity,
    methodRotor,
    fuel_mass_total,
    number_of_blades,
    rotorRadius,
    number_of_engines_combustion,
    required_engine_power_total_combustion,
    real_power_per_rotor,
):
    """
    Parameters:
    - estimatedGlideRatio               estimated glide ratio                                 [double]    [1 x n]     [-]
    - thrustSafetyFactor                factor to increase design thrust to leave a           [double]    [1 x 1]     [-]
                                        safety margin
    - energySafetyFactor                factor to increase design energy to leave a           [double]    [1 x 1]     [-]
                                        safety margin
    - propellerEfficiency               (estimated) propeller efficiency: propulsive power    [double]    [1 x 1]     [-]
                                        per shaft power
    - power_specific_engine_mass_combustion           (estimated) power specific engine mass: mass per      [double]    [1 x 1]     [kg/W]
                                        input power
    - power_specific_engine_volume_combustion         (estimated) power specific engine volume: volume per  [double]    [1 x 1]     [m^3/W]
    -				                    input power
    - PSFC                              power specific fuel consumption of the engine: fuel   [double]    [1 x 1]     [kg/sW]
                                        consumption per input energy
    - m_tom                             maximum take-off mass                                 [double]    [1 x 1]     [kg]
    - airspeed                          airspeed of the configuration                         [double]    [1 x 1]     [m/s]
    - fuelTankEmptyMassFraction         estimated fraction of empty tank mass and fuel mass   [double]    [1 x 1]     [-]
    - fuelDensity                       fuel density                                          [double]    [1 x 1]     [kg/m^3]
    - methodRotor                       method for the rotor mass estimation                  [string]    ['gundlach' or 'seren']
    - fuel_mass_total                   total fuel mass consumed in the mission               [double]    [1 x 1]     [-]
    - number_of_blades                  number of blades per rotor                            [double]    [1 x 1]     [-]
    - rotorRadius                       radius of each rotor                                  [double]    [1 x 1]     [m]
    Returns:
    - requiredEnginePower              required engine input power                           [double]    [1 x 1]     [W]
    - engineMass                       engine mass                                           [double]    [1 x 1]     [kg]
    - zHeight_engine                   engine size in z direction                            [double]    [1 x 1]     [m]
    - yWidth_engine                    engine size in y direction                            [double]    [1 x 1]     [m]
    - xLength_engine                   engine size in x direction                            [double]    [1 x 1]     [m]
    - tankMassEmpty                    empty tank mass                                       [double]    [1 x 1]     [kg]
    - zHeight_fuelTank                 fuel tank size in z direction                         [double]    [1 x 1]     [m]
    - yWidth_fuelTank                  fuel tank size in y direction                         [double]    [1 x 1]     [m]
    - xLength_fuelTank                 fuel tank size in x direction                         [double]    [1 x 1]     [m]
    - rotorMass                        propeller mass                                        [double]    [1 x 1]     [kg]

    Description:
    - This function estimates the dimensions and masses of the engine, the fuel tank, and the rotors.

    Limitations:
    -

    Assumptions:
    - engine geometry is assumed to be cubical fuel tank geometry assumption: length = 2*width = 2*height
    """

    # find out which one from cruise and hover is driving the design
    required_engine_power_total = max(required_engine_power_total_combustion)
    realPowerPerRotor = np.amax(real_power_per_rotor, axis=0)

    # Calculate engine mass and volume
    engine_mass = (
        required_engine_power_total
        * power_specific_engine_mass_combustion
        / number_of_engines_combustion
    )  # [kg]
    engineVolume = (
        required_engine_power_total
        * power_specific_engine_volume_combustion
        / number_of_engines_combustion
    )  # [m^3]
    zHeight_engine = engineVolume ** (
        1 / 3
    )  # [m] engine geometry is assumed to be cubical
    yWidth_engine = zHeight_engine  # [m]
    xLength_engine = zHeight_engine  # [m]

    # Calculate fuel volume, and fuel tank mass
    fuelVolume = fuel_mass_total / fuelDensity  # [m^3]
    tank_mass_empty = fuelTankEmptyMassFraction * fuel_mass_total  # [kg]
    zHeight_fuelTank = (fuelVolume / 2) ** (
        1 / 3
    )  # [m] assumption: length = 2*width = 2*height
    yWidth_fuelTank = zHeight_fuelTank  # [m]
    xLength_fuelTank = 2 * zHeight_fuelTank  # [m]

    # Calculate rotor mass
    if methodRotor == "gundlach":
        rotor_mass = calculate_mass_propeller_uav_gundlach(
            15, number_of_blades, 2 * rotorRadius, realPowerPerRotor
        )  # [kg]
    elif methodRotor == "seren":
        rotor_mass = 0.0136 + 1.219 * (rotorRadius) ** 2.56  # [kg]

    outputs = {
        "required_engine_power_combustion": np.atleast_1d(
            required_engine_power_total / number_of_engines_combustion
        ),
        "engine_mass_combustion": np.atleast_1d(engine_mass),
        "zHeight_engine_combustion": np.atleast_1d(zHeight_engine),
        "yWidth_engine_combustion": np.atleast_1d(yWidth_engine),
        "xLength_engine_combustion": np.atleast_1d(xLength_engine),
        "tank_mass_empty_combustion": np.atleast_1d(tank_mass_empty),
        "zHeight_fuelTank_combustion": np.atleast_1d(zHeight_fuelTank),
        "yWidth_fuelTank_combustion": np.atleast_1d(yWidth_fuelTank),
        "xLength_fuelTank_combustion": np.atleast_1d(xLength_fuelTank),
        "rotor_mass": np.atleast_1d(rotor_mass),
    }

    return outputs


def rubber_engine_sizing_electric_uav(
    escEfficiency,
    methodRotor,
    method_components,
    battery_mass_total,
    number_of_blades,
    rotorRadius,
    number_of_engines_electric,
    escMassDensity,
    batteryMassDensity,
    powerSpecificESCMass,
    power_specific_engine_mass_electric,
    motor_mass_density_electric,
    required_engine_power_total_electric,
    real_power_per_rotor,
):
    """ """

    # find out which one from cruise and hover is driving the design
    required_engine_power_total = max(required_engine_power_total_electric)
    realPowerPerRotor = np.amax(real_power_per_rotor, axis=0)

    # find required ESC power
    required_esc_power_individual = (
        required_engine_power_total / number_of_engines_electric / escEfficiency
    )

    # Calculate engine mass and volume
    if method_components == "feger":
        engine_mass = (
            required_engine_power_total
            / number_of_engines_electric
            * power_specific_engine_mass_electric
        )  # [kg]
    elif method_components == "seren":
        engine_mass = (
            4.57 * 10 ** (-4) * required_esc_power_individual ** (0.899)
        )  # [kg]

    motor_x_length = (4 / np.pi * engine_mass / motor_mass_density_electric) ** (
        1 / 3
    )  # [m] % assumption: cylindrical, diamter = length, motorMassDensity = 4000kg/m^3 or 3800 kg/m^3. cylinder volume = pi * xlength_motor^3 / 4
    motor_y_width = motor_x_length  # [m]
    motor_z_height = motor_x_length  # [m]

    # Calculate rotor mass
    if methodRotor == "gundlach":
        rotor_mass = calculate_mass_propeller_uav_gundlach(
            15, number_of_blades, 2 * rotorRadius, realPowerPerRotor
        )  # [kg]
    elif methodRotor == "seren":
        rotor_mass = 0.0136 + 1.219 * (rotorRadius) ** 2.56  # [kg]

    # Calculate ESC mass
    if method_components == "feger":
        esc_mass = required_esc_power_individual * powerSpecificESCMass  # [kg]
    elif method_components == "seren":
        esc_mass = 1.18 * 10 ** (-4) * required_esc_power_individual ** (0.828)  # [kg]

    # assumption for ESC dimensions: 4 units long, 2 units wide, 0.7 units thick --> (5.6 in total); escMassDensity ~2700kg/m^3
    ESCLengthUnit = (esc_mass / escMassDensity / 5.6) ** (1 / 3)  # [m]
    esc_x_length = 4 * ESCLengthUnit  # [m]
    esc_y_width = 2 * ESCLengthUnit  # [m]
    esc_z_height = 0.7 * ESCLengthUnit  # [m]

    # assumption for battery dimensions: 20 units long, 7 units wide, 5 units thick (700 in total), batteryMassDensity ~2200kg/m^3
    batteryLengthUnit = (battery_mass_total / batteryMassDensity / 700) ** (
        1 / 3
    )  # [m]
    battery_x_length = 20 * batteryLengthUnit  # [m]
    battery_y_width = 7 * batteryLengthUnit  # [m]
    battery_z_height = 5 * batteryLengthUnit  # [m]

    # Gather outputs
    outputs = {
        "required_engine_power_electric": np.atleast_1d(
            required_engine_power_total / number_of_engines_electric
        ),
        "engine_mass_electric": np.atleast_1d(engine_mass),
        "motor_x_length_electric": np.atleast_1d(motor_x_length),
        "motor_y_width_electric": np.atleast_1d(motor_y_width),
        "motor_z_height_electric": np.atleast_1d(motor_z_height),
        "esc_mass_electric": np.atleast_1d(esc_mass),
        "esc_x_length_electric": np.atleast_1d(esc_x_length),
        "esc_y_width_electric": np.atleast_1d(esc_y_width),
        "esc_z_height_electric": np.atleast_1d(esc_z_height),
        "battery_x_length_electric": np.atleast_1d(battery_x_length),
        "battery_y_width_electric": np.atleast_1d(battery_y_width),
        "battery_z_height_electric": np.atleast_1d(battery_z_height),
        "rotor_mass": np.atleast_1d(rotor_mass),
    }

    return outputs


def power_demand_mission_segments(
    mission_segment_types,
    m_tom,
    estimatedGlideRatio,
    thrustSafetyFactor,
    rotorRadius,
    airspeed,
    propellerEfficiency,
    hybridization,
    engine_combustion_efficiency,
    engine_electric_efficiency,
    density_air,
):
    # initialize
    realPowerPerRotor = []
    requiredEnginePower_total_combustion = []
    requiredEnginePower_total_electric = []

    index = 0

    for mission_segment_type in mission_segment_types:
        if mission_segment_type == "cruise" or mission_segment_type == "loiter":

            # Compute required thrust for the segment
            thrust_total = 9.81 * m_tom * 1 / estimatedGlideRatio[index]  # [N]

            # Add safety factor
            thrust_total = thrustSafetyFactor * thrust_total  # [N]

            # Compute thrust per rotor
            thrustPerRotor = thrust_total / len(rotorRadius)  # [N]

            # ideal power per rotor
            idealPowerPerRotor = thrustPerRotor * airspeed[index]

            # real power per rotor
            realPowerPerRotor.append(
                np.repeat(idealPowerPerRotor / propellerEfficiency, len(rotorRadius))
            )  # [W]

            # Calculate required combustion engine power
            requiredEnginePower_total_combustion.append(
                (1 - hybridization[index])
                * sum(realPowerPerRotor[index])
                / engine_combustion_efficiency
            )  # [W] required engine input power

            # Calculate required electric engine power
            requiredEnginePower_total_electric.append(
                (hybridization[index])
                * sum(realPowerPerRotor[index])
                / engine_electric_efficiency
            )  # [W] required engine input power

            # increase index
            index += 1

        elif mission_segment_type == "hover":

            thrust_total = 9.81 * m_tom  # [N]

            # Apply safety factor
            thrust_total = thrustSafetyFactor * thrust_total

            # Compute thrust per rotor
            thrustPerRotor = thrust_total / len(rotorRadius)  # [N]

            # Compute induced velocity per rotor
            inducedVelocityPerRotor = np.sqrt(
                thrustPerRotor / (2 * density_air[index] * np.pi * rotorRadius**2)
            )  # [m/s]

            # Compute ideal power per rotor
            idealPowerPerRotor = thrustPerRotor * inducedVelocityPerRotor  # [W]

            # Compute real power per rotor
            realPowerPerRotor.append(idealPowerPerRotor / propellerEfficiency)  # [W]

            # Calculate required combustion engine power
            requiredEnginePower_total_combustion.append(
                (1 - hybridization[index])
                * sum(realPowerPerRotor[index])
                / engine_combustion_efficiency
            )  # [W] required engine input power

            # Calculate required electric engine power
            requiredEnginePower_total_electric.append(
                (hybridization[index])
                * sum(realPowerPerRotor[index])
                / engine_electric_efficiency
            )  # [W] required engine input power

            # increase index
            index += 1

    return (
        np.array(requiredEnginePower_total_combustion),
        np.array(requiredEnginePower_total_electric),
        realPowerPerRotor,
    )
