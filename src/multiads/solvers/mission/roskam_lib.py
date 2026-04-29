import math

import numpy as np
from ambiance import Atmosphere


# Breguet
def calculate_fuel_fraction_endurance_maneuvering_fuelcell(
    m_start,
    t,
    T_max,
    V,
    eta_prop,
    eta_fc,
    U_thermo,
    mue,
    retainH2O,
):
    """Parameters
    m_start    start mass of flight phase                 [kg]    [double]    [1 x n] or [n x 1]
    t          flight endurance                           [s]     [double]    [1 x n] or [n x 1]
    T_max      maximum thrust                             [N]     [double]    [1 x n] or [n x 1]
    V          flight speed                               [m/s]   [double]    [1 x n] or [n x 1]
    eta_prop   efficiency of powertrain until fuelcell    [-]     [double]    [1 x n] or [n x 1]
    eta_fc     efficiency of fuelcell (U_cell/U_thermo)   [-]     [double]    [1 x n] or [n x 1]
    U_thermo   theoretical voltage of fuelcell            [V]     [double]    [1 x n] or [n x 1]
    mue		   factor to account for BoP power            [-]     [double]    [1 x n] or [n x 1]
    retainH2O  flag if water produced is stored on board  [-]     [logical]   [1 x n] or [n x 1]

    Returns
    -------
    f_H2       hydrogen fraction of fuel cell aircraft    [-]     [double]    [1 x n] or [n x 1]
    f_H2O      water fraction of fuel cell aircraft       [-]     [double]    [1 x n] or [n x 1]

    Description: This function calculates the flight endurance of a fuel cell
                 aircraft for the maneuvering phase.

    Limitations: -

    Assumptions: Fuel utilization coefficient of 0.95 (Larminie & Dicks,
                 Fuel Cell Systems Explained, 2nd Ed., 2003, p.35)
                 Maximum thrust

    Example: f_H2 = calculate_fuel_fraction_endurance_maneuvering_fuelcell(70000, 8565, 120000, 250, 0.8, 0.6, 1.2, 0.3, false)
             f_H2, f_H2O = calculate_fuel_fraction_endurance_maneuvering_fuelcell(70000, 8565, 135000, 250, 0.85, 0.65, 1.1, 0.4, true)

    """
    # Calculation
    K = (
        -1.05 * 10**-8 / 0.95
    )  # from Larminie & Dicks, Fuel Cell Systems Explained, 2nd Ed., 2003, App. 2.4
    f_H2 = (
        m_start + ((K * (1 + mue) * V * T_max * t) / (eta_prop * eta_fc * U_thermo))
    ) / m_start

    if retainH2O:
        K = (
            9.34 * 10**-8
        )  # from Larminie & Dicks, Fuel Cell Systems Explained, 2nd Ed., 2003, App. 2.5
        f_H2O = (
            m_start + ((K * (1 + mue) * V * T_max * t) / (eta_prop * eta_fc * U_thermo))
        ) / m_start
    else:
        f_H2O = 1

    return (f_H2, f_H2O)


def calculate_fuel_fraction_range_breguet_fc1(
    R,
    SFC,
    c_L,
    c_D,
    V,
    n_v,
):  # input: (range,specific fuel consumption,lift coefficient,drag coefficient,airspeed,power dependency regarding airspeed)
    """Parameters
    R           flight range                        [m]     [double]    [1 x n] or [n x 1]
    SFC         specific fuel consumption           [kg/Ns] [double]    [1 x n] or [n x 1]
                or     							    [kg/Ws] [double]    [1 x n] or [n x 1]
    c_L         lift coefficient                    [-]     [double]    [1 x n] or [n x 1]
    c_D         drag coefficient                    [-]     [double]    [1 x n] or [n x 1]
    V           airspeed                            [m/s]   [double]    [1 x n] or [n x 1]
    n_v         power dependency regarding airspeed [-]     [double]    [1 x n] or [n x 1]

    Returns
    -------
    ff         fuel fraction of ESM                 [-]     [double]    [1 x n] or [n x 1]
    Description:  This function calculates the fuel fraction defined by
                flight range according to Breguet Flight Condition 1.
    Limitations:  -
    Assumptions:  -
    Example:      ff = calculate_fuel_fraction_range_breguet_fc1(1000000,1.2*10^-5,0.6,0.15,90,0)

    """
    # DEFINES
    g = 9.81  # earth gravity [m/s^2]

    # Calculation
    ff = math.exp((-1 * R) / ((V ** (n_v + 1) / (SFC * g)) * (c_L / c_D)))

    return ff


def calculate_fuel_fraction_endurance_maneuvering(
    m_start,
    t,
    SFC,
    T_max,
    V,
    n_v,
):  # input: (start mass,time,specific fuel consumption,maximum thrust,airspeed,power dependency regarding airspeed)
    """Parameters
    m_start     start mass of maneuvering           [kg]    [double]    [1 x n] or [n x 1]
    t           flight endurance                    [s]     [double]    [1 x n] or [n x 1]
    SFC         specific fuel consumption           [kg/Ns] [double]    [1 x n] or [n x 1]
                or     							   [kg/Ws] [double]    [1 x n] or [n x 1]
    T_max       maximum thrust                      [N]     [double]    [1 x n] or [n x 1]
    V           airspeed                            [m/s]   [double]    [1 x n] or [n x 1]
    n_v         power dependency regarding airspeed [-]     [double]    [1 x n] or [n x 1]

    Returns
    -------
    ff         fuel fraction                       [-]     [double]    [1 x n] or [n x 1]

    Description:  This function calculates the fuel fraction defined by
                flight endurance of a maneuvering/combat phase.

    Limitations:  -

    Assumptions:  -

    Example:    ff = calculate_fuel_fraction_endurance_maneuvering(8000,1200,1.2*10^-5,15000,100,0)
                ff = calculate_fuel_fraction_endurance_maneuvering(8000,1200,1.1*10^-7,15000,100,-1)

    """
    # Calculation
    ff = (-1 * SFC * T_max * t * V**-n_v + m_start) / m_start

    return ff


def calculate_fuel_fraction_endurance_breguet_fc1_fuelcell(
    t,
    c_L,
    c_D,
    V,
    eta_prop,
    eta_fc,
    U_thermo,
    mue,
    retainH2O,
):
    """Parameters
    t           flight endurance							[s]     [double]    [1 x n] or [n x 1]
    c_L         lift coefficient							[-]     [double]    [1 x n] or [n x 1]
    c_D         drag coefficient							[-]     [double]    [1 x n] or [n x 1]
    V           flight speed								[m/s]   [double]    [1 x n] or [n x 1]
    eta_prop    efficiency of powertrain until fuelcell   [-]     [double]    [1 x n] or [n x 1]
    eta_fc      efficiency of fuelcell (U_cell/U_thermo)  [-]     [double]    [1 x n] or [n x 1]
    U_thermo    theoretical voltage of fuelcell			[V]     [double]    [1 x n] or [n x 1]
    mue		  factor to account for BoP power			[-]     [double]    [1 x n] or [n x 1]
    retainH2O   flag if water produced is stored on board	[-]     [logical]   [1 x n] or [n x 1]

    Returns
    -------
    f_H2       hydrogen fraction of fuel cell aircraft  	[-]     [double]    [1 x n] or [n x 1]
    f_H2O      water fraction of fuel cell aircraft     	[-]     [double]    [1 x n] or [n x 1]

    Description:  This function calculates the hydrogen fraction defined by
                flight endurance of a fuel cell aircraft according to Breguet
                    Flight Condition 1.

    Limitations:  -

    Assumptions:  Fuel utilization coefficient of 0.95 (Larminie & Dicks,
                    Fuel Cell Systems Explained, 2nd Ed., 2003, p.35)

    Example:      [f_H2,f_H2O] = calculate_fuel_fraction_endurance_breguet_fc1_fuelcell(8695,0.6,0.018,230,0.85,0.65,1.1,0.4,true)
                    f_H2 = calculate_fuel_fraction_endurance_breguet_fc1_fuelcell(8695,0.6,0.018,230,0.85,0.65,1.1,0.4,false)

    """
    # DEFINES
    g = 9.81  # earth gravity [m/s^2]

    K = (
        -1.05 * 10**-8 / 0.95
    )  # from Larminie & Dicks, Fuel Cell Systems Explained, 2nd Ed., 2003, App. 2.4

    # Calculation
    f_H2 = math.exp(
        (t * V * K * (1 + mue) * g) / (eta_prop * eta_fc * U_thermo * c_L / c_D),
    )

    if retainH2O:
        K = (
            9.34 * 10**-8
        )  # from Larminie & Dicks, Fuel Cell Systems Explained, 2nd Ed., 2003, App. 2.5
        f_H2O = math.exp(
            (t * V * K * (1 + mue) * g) / (eta_prop * eta_fc * U_thermo * c_L / c_D),
        )
    else:
        f_H2O = 1

    return (f_H2, f_H2O)


def calculate_fuel_fraction_endurance_breguet_fc1(
    t,
    SFC,
    c_L,
    c_D,
    V,
    n_v,
):  # (endurance,specific fuel consumption,lift coefficient,drag coefficient,airspeed,power dependency regarding airspeed)
    """Parameters
    t           flight endurance                    [s]     [double]    [1 x n] or [n x 1]
    SFC         specific fuel consumption           [kg/Ns] [double]    [1 x n] or [n x 1]
                or     							  [kg/Ws] [double]    [1 x n] or [n x 1]
    c_L         lift coefficient                    [-]     [double]    [1 x n] or [n x 1]
    c_D         drag coefficient                    [-]     [double]    [1 x n] or [n x 1]
    V           airspeed                            [m/s]   [double]    [1 x n] or [n x 1]
    n_v         power dependency regarding airspeed [-]     [double]    [1 x n] or [n x 1]

    Returns
    -------
    ff         fuel fraction of ESM                [-]     [double]    [1 x n] or [n x 1]

    Description:  This function calculates the fuel fraction defined by
                flight endurance according to Breguet Flight Condition 1.

    Limitations:  -

    Assumptions:  -

    Example:    ff = calculate_fuel_fraction_endurance_breguet_fc1(3600,1.2*10^-5,0.6,0.15,90,0)
                ff = calculate_fuel_fraction_endurance_breguet_fc1(3600,1.1*10^-7,0.6,0.15,90,-1)

    """
    # DEFINES
    g = 9.81  # earth gravity [m/s^2]
    # Calculation
    ff = math.exp((-1 * t) / ((V**n_v / (SFC * g)) * (c_L / c_D)))

    return ff


def calculate_battery_mass_range(
    m_TOM,
    R,
    Easterisk,
    eta,
    c_L,
    c_D,
):  # input: (take-off mass,range,specific energy,overall efficiency,lift coefficient,drag coefficient)
    """Parameters
    m_TOM       take-off mass                       [kg]    [double]    [1 x n] or [n x 1]
    R           flight range                        [m]     [double]    [1 x n] or [n x 1]
    Easterisk   specific energy of battery          [Ws/kg] [double]    [1 x n] or [n x 1]
    eta         overall efficiency of the electric
                        powertrain						  [-]     [double]    [1 x n] or [n x 1]
    c_L         lift coefficient                    [-]     [double]    [1 x n] or [n x 1]
    c_D         drag coefficient                    [-]     [double]    [1 x n] or [n x 1]

    Returns
    -------
    m_bat      battery mass                        [kg]    [double]    [1 x n] or [n x 1]

    Description:  This function calculates the battery mass according to a
                defined fight range for an electric cruise flight according to
                Hepperle.

    Limitations:  -

    Assumptions:  -

    Example:      m_bat = calculate_battery_mass_range(16200,100000,720000,0.7,16)
                m_bat = calculate_battery_mass_range(16200,100000,720000,0.7,0.8,0.05)

    """
    # DEFINES
    g = 9.81  # earth gravity [m/s^2]
    # Calculation
    c_L_c_D = c_L / c_D  # lift-to-drag ratio
    m_bat = R * (1 / c_L_c_D) * ((m_TOM * g) / (Easterisk * eta))  # battery mass

    return m_bat


def calculate_battery_mass_endurance(
    m_TOM,
    t,
    V,
    Easterisk,
    eta,
    c_L,
    c_D,
):  # input: (take-off mass,endurance,airspeed,specific energy,overall efficiency,lift coefficient,drag coefficient)
    """Parameters
    m_TOM       take-off mass                       [kg]    [double]    [1 x n] or [n x 1]
    t           flight endurance                    [s]     [double]    [1 x n] or [n x 1]
    V           airspeed                            [m/s]   [double]    [1 x n] or [n x 1]
    Easterisk   specific energy of battery          [Ws/kg] [double]    [1 x n] or [n x 1]
    eta         overall efficiency of the electric
                powertrain						  [-]     [double]    [1 x n] or [n x 1]
    c_L         lift coefficient                [-]     [double]    [1 x n] or [n x 1]
    c_D         drag coefficient                    [-]     [double]    [1 x n] or [n x 1]

    Returns
    -------
    m_Bat      battery mass                        [kg]    [double]    [1 x n] or [n x 1]

    Description:  This function calculates the battery mass according to a
                defined fight endurance for an electric cruise flight
                according to Hepperle.

    Limitations:  -

    Assumptions:  -

    Example:    m_Bat = calculate_battery_mass_endurance(16200,3600,90,720000,0.7,16)
                m_Bat = calculate_battery_mass_endurance(16200,3600,90,720000,0.7,0.8,0.05)

    """
    # DEFINES
    g = 9.81  # earth gravity [m/s^2]
    # Calculation
    c_L_c_D = c_L / c_D  # lift-to-drag ratio

    m_Bat = t * V * (1 / c_L_c_D) * ((m_TOM * g) / (Easterisk * eta))  # battery mass

    return m_Bat


def calculate_fuel_fraction_range_breguet_fc1_fuelcell(
    R,
    c_L,
    c_D,
    eta_prop,
    eta_fc,
    U_thermo,
    mue,
    retainH2O,
):
    """Parameters
        R           flight range								[m]     [double]    [1 x n] or [n x 1]
        c_L         lift coefficient							[-]     [double]    [1 x n] or [n x 1]
        c_D         drag coefficient							[-]     [double]    [1 x n] or [n x 1]
        eta_prop    efficiency of powertrain until fuelcell   [-]     [double]    [1 x n] or [n x 1]
        eta_fc      efficiency of fuelcell (U_cell/U_thermo)  [-]     [double]    [1 x n] or [n x 1]
        U_thermo    theoretical voltage of fuelcell			[V]     [double]    [1 x n] or [n x 1]
        mue		  factor to account for BoP power			[-]     [double]    [1 x n] or [n x 1]
        retainH2O   flag if water produced is stored on board	[-]     [logical]   [1 x n] or [n x 1]

    Returns
    -------
        f_H2       hydrogen fraction of fuel cell aircraft  	[-]     [double]    [1 x n] or [n x 1]
        f_H2O      water fraction of fuel cell aircraft     	[-]     [double]    [1 x n] or [n x 1]

        Description:  This function calculates the hydrogen fraction defined by
                                flight range of a fuel cell aircraft according to Breguet
                                        Flight Condition 1.

        Limitations:  -

        Assumptions:  Fuel utilization coefficient of 0.95 (Larminie & Dicks,
                                        Fuel Cell Systems Explained, 2nd Ed., 2003, p.35)

        Example:      [f_H2,f_H2O] = calculate_fuel_fraction_range_breguet_fc1_fuelcell(2*10^6,0.6,0.018,0.85,0.65,1.1,0.4,true)
                                        f_H2 = calculate_fuel_fraction_range_breguet_fc1_fuelcell(2*10^6,0.6,0.018,0.85,0.65,1.1,0.4,false)

    """
    # DEFINES
    g = 9.81  # earth gravity [m/s^2]

    K = (
        -1.05 * 10**-8 / 0.95
    )  # from Larminie & Dicks, Fuel Cell Systems Explained, 2nd Ed., 2003, App. 2.4

    # Calculation
    f_H2 = math.exp(
        (R * K * (1 + mue) * g) / (eta_prop * eta_fc * U_thermo * c_L / c_D),
    )

    if retainH2O:
        K = (
            9.34 * 10**-8
        )  # from Larminie & Dicks, Fuel Cell Systems Explained, 2nd Ed., 2003, App. 2.5
        f_H2O = math.exp(
            (R * K * (1 + mue) * g) / (eta_prop * eta_fc * U_thermo * c_L / c_D),
        )
    else:
        f_H2O = 1
    return (f_H2, f_H2O)


# VTOL
def calculate_fuel_mass_hover_endurance(
    mass,
    thrustSafetyFactor,
    density,
    rotorRadius,
    powerTrainEfficiency,
    requiredTime,
    SFC_hover,
):
    ########################## LIMITATION ##################################################################
    # Only for short hover times since no mass change due to fuel consumption is taken into account!
    # Equal distribution of the thrust among all rotors is assumed during the hover!
    ########################################################################################################

    # Compute required thrust
    totalThrust = 9.81 * mass  # [N]

    # Apply safety factor
    totalThrust = thrustSafetyFactor * totalThrust

    # Compute thrust per rotor
    thrustPerRotor = totalThrust / len(rotorRadius)  # [N]

    # Compute induced velocity per rotor
    inducedVelocityPerRotor = np.sqrt(
        thrustPerRotor / (2 * density * np.pi * rotorRadius**2),
    )  # [m/s]

    # Compute ideal power per rotor
    idealPowerPerRotor = thrustPerRotor * inducedVelocityPerRotor  # [W]

    # Compute required fuel consumption
    requiredFuelMass = (
        sum(idealPowerPerRotor) / powerTrainEfficiency * SFC_hover * requiredTime
    )  # [kg]

    return requiredFuelMass


def calculate_battery_mass_hover(
    mass,
    thrustSafetyFactor,
    density,
    rotorRadius,
    powerTrainEfficiency,
    requiredTime,
    Easterisk,
):
    ########################## LIMITATION ##################################################################
    # Only for short hover times since no mass change due to fuel consumption is taken into account!
    # Equal distribution of the thrust among all rotors is assumed during the hover!
    ########################################################################################################

    # Compute required thrust
    totalThrust = 9.81 * mass  # [N]

    # Apply safety factor
    totalThrust = thrustSafetyFactor * totalThrust

    # Compute thrust per rotor
    thrustPerRotor = totalThrust / len(rotorRadius)  # [N]

    # Compute induced velocity per rotor
    inducedVelocityPerRotor = np.sqrt(
        thrustPerRotor / (2 * density * np.pi * rotorRadius**2),
    )  # [m/s]

    # Compute ideal power per rotor
    idealPowerPerRotor = thrustPerRotor * inducedVelocityPerRotor  # [W]

    # Compute required battery power
    requiredBatteryPower = sum(idealPowerPerRotor) / powerTrainEfficiency  # [W]

    requiredBatteryEnergy = requiredBatteryPower * requiredTime  # [J], required energy

    requiredBatteryMass = requiredBatteryEnergy / Easterisk  # [kg]

    return requiredBatteryMass


# Roskam
def calculate_mission_roskam_customized(
    mission,
    energy_fraction,
    propellant_type,
    S_ref,
    n_v,
    c_L_array,
    c_D_array,
    alpha_array,
    fuelReserve,
    batteryReserve,
    **kwargs,
):
    """Parameters
    mission               mission containing all mission segments [-]         [cell]      [n x 1]
    energy_fraction       dictionary containing energy fractions according to Roskam with 6 keys:
                                            engineStart,taxi,takeOff,climb, descent, landing		[-]         [struct]    [1 x 1]
    S_ref                 reference area for the configuration aerodynamics      [m^2]       [double]    [1 x 1]
    n_v                   power dependency regarding airspeed                     [-]         [double]    [1 x 1]
    c_L_array                   lift coefficient array								   	[-]         [double]    [n x 1]
    c_D_array                   drag coefficient array								   	[-]         [double]    [n x 1]
    alpha_array         range of angles of attack assoicated with c_L_array and c_D_array [deg]
    SFC_cruise		    cruise specific fuel consumption						[kg/Ns]     [double]    [1 x 1]
                                                                                                                                           or      	[kg/Ws]     [double]    [1 x 1]
    SFC_hover		    hover specific fuel consumption						[kg/Ns]     [double]    [1 x 1]
                                                                                                                                           or      	[kg/Ws]     [double]    [1 x 1]
    SFC_loiter		    loiter specific fuel consumption						[kg/Ns]     [double]    [1 x 1]
                                                                                                                                           or      	[kg/Ws]     [double]    [1 x 1]
    SFC_maneuvering       maneuvering specific fuel consumption                   [kg/Ns]     [double]    [1 x 1]
                                                                                                                                           or      	[kg/Ws]     [double]    [1 x 1]
    T_maneuvering         max available thrust for maneuvering                    [N]         [double]    [1 x 1]
    eta_prop1_cruise      overall efficiency of powertrain 1					   	[-]         [double]    [1 x 1]
    Easterisk             specific energy of battery (optional)                   [Ws/kg]     [double]    [1 x 1]
    eta_prop2_cruise      overall efficiency of powertrain 2 during
                                            cruise (optional)									   	[-]         [double]    [1 x 1]
    eta_prop2_loiter      overall efficiency of powertrain 2 during
                                            loiter (optional)									   	[-]         [double]    [1 x 1]
    eta_prop2_maneuvering overall efficiency of powertrain 2 during
                                            maneuvering (optional)								   	[-]         [double]    [1 x 1]
    eta_fuelcell_voltage  efficiency of fuelcell (U_cell/U_thermo)				[-]         [double]    [1 x 1]
    eta_fuelcell_LHV      efficiency of fuelcell (P_out/(LHV*m_dot_H2))			[-]         [double]    [1 x 1]
    U_thermo			    theoretical voltage of fuelcell						   	[V]         [double]    [1 x 1]
    mue				    factor to account for BoP power						   	[-]         [double]    [1 x 1]
    retainH2O			    flag if produced H2O of fuel cell is retained on board 	[-]         [logical]   [1 x 1]

    Returns
    -------
    energy1MassTotal     mass of energy 1 (e.g. kerosene) of complete mission	[kg]        [double]    [1 x 1]
    energy2MassTotal     mass of energy 2 (e.g. battery) of complete mission		[kg]        [double]    [1 x 1]
    mission              mission containing all mission segments [-]         [cell]      [n x 1]
    fuelReserve          fuel reserve as a portion of the trip fuel            [double]    [1 x 1]     [%]
    batteryReserve       battery reserve as a portion of the total battery mass            [double]    [1 x 1]     [%]

    Description:  This function calculates the mass fraction and also battery
                  weight for a defined mission using a previoulsy calculated
                                 aerodynamic polar. The mission segments can be
                  calculated separately according to a defined degree of
                  energy hybridization (electric energy divided by total energy).
                  This can range from:
                  DoH = 0:        fuel burn (0% electric energy)
                  0 <= DoH <= 1:  hybrid
                  DoH = 1:        fully electric (100% electric energy)
                  The mission segments have to be according to the definitions
                  of Roskam.
                                 This function has a different eta for cruise, loiter and maneuvering.

    Limitations:  Valid for missions defined according to Roskam and calculated
                  with Breguet flight condition 1 (constant airspeed and lift
                                 coefficient).
                  ########################## LIMITATION ##################################################################
                  # If hover with kerosene is used: only suited for short hover times since no mass change due to fuel consumption is taken into account!
                  ########################################################################################################

    """
    # Analyze Input Data
    mission_segments = mission.mission_segments
    n_segment = len(mission_segments)  # number of mission segments

    if "thrustSafetyFactor" in kwargs:
        thrustSafetyFactor = kwargs["thrustSafetyFactor"]
    if "rotorRadius" in kwargs:
        rotorRadius = kwargs["rotorRadius"]
    if "eta_prop1_hover" in kwargs:
        eta_prop1_hover = kwargs["eta_prop1_hover"]
    if "eta_prop2_hover" in kwargs:
        eta_prop2_hover = kwargs["eta_prop2_hover"]
    if "SFC_hover" in kwargs:
        SFC_hover = kwargs["SFC_hover"]
    if "SFC_cruise" in kwargs:
        SFC_cruise = kwargs["SFC_cruise"]
    if "SFC_loiter" in kwargs:
        SFC_loiter = kwargs["SFC_loiter"]
    if "SFC_maneuvering" in kwargs:
        SFC_maneuvering = kwargs["SFC_maneuvering"]
    if "T_maneuvering" in kwargs:
        T_maneuvering = kwargs["T_maneuvering"]
    if "eta_prop1_cruise" in kwargs:
        eta_prop1_cruise = kwargs["eta_prop1_cruise"]
    if "Easterisk" in kwargs:
        Easterisk = kwargs["Easterisk"]
    if "eta_prop2_cruise" in kwargs:
        eta_prop2_cruise = kwargs["eta_prop2_cruise"]
    if "eta_prop2_loiter" in kwargs:
        eta_prop2_loiter = kwargs["eta_prop2_loiter"]
    if "eta_prop2_maneuvering" in kwargs:
        eta_prop2_maneuvering = kwargs["eta_prop2_maneuvering"]
    if "fuelcell" in propellant_type:
        eta_fuelcell_voltage = kwargs["eta_fuelcell_voltage"]
        eta_fuelcell_LHV = kwargs["eta_fuelcell_LHV"]
        U_thermo = kwargs["U_thermo"]
        mue = kwargs["mue"]
        retainH2O = kwargs["retainH2O"]

    # DEFINES
    g = 9.81  # earth gravity [m/s^2]
    energy_fraction1 = np.ones(
        n_segment,
    )  # predefine (fuelFraction_i = massEndSegment_i/massStartSegment_i = 1 - (massFuelUsedSegment_i/massStartSegment_i))
    energy_fraction2 = np.ones(
        n_segment,
    )  # predefine (batteryFraction_i = 1 - (batteryMassMissionSegment_i/massStartSegment_i))
    energy_mass1_mission_segment = np.zeros(n_segment)  # predefine
    energy_mass2_mission_segment = np.zeros(n_segment)  # predefine

    LHV_kerosene = 43.1 * 10**6  # [Ws/kg] lower heating value kerosene
    LHV_hydrogen = 120 * 10**6  # [Ws/kg] lower heating value hydrogen
    eta_kerosene = 0.3  # [-] approximate overall efficiency of engines with EIS 1980 (about state of technology on which Roskams Fuel Fractions are based)

    if propellant_type == "kerosene":
        LHV1 = LHV_kerosene
        mass_change_type1 = 1  # change (lighter)
        LHV2 = 1
        mass_change_type2 = 0  # same
    elif propellant_type == "hydrogen":
        LHV1 = LHV_hydrogen
        mass_change_type1 = 1  # change (lighter)
        LHV2 = 1
        mass_change_type2 = 0  # same
    elif propellant_type == "battery":
        LHV1 = Easterisk
        mass_change_type1 = 0  # same
        LHV2 = 1
        mass_change_type2 = 0  # same
    elif propellant_type == "fuelcell":
        LHV1 = LHV_hydrogen
        mass_change_type1 = 1  # change
        LHV2 = 1
        mass_change_type2 = 0  # same
    elif propellant_type == "kerosene-hydrogen":
        LHV1 = LHV_kerosene
        mass_change_type1 = 1  # change (lighter)
        LHV2 = LHV_hydrogen
        mass_change_type2 = 1  # change (lighter)
    elif propellant_type == "kerosene-battery":
        LHV1 = LHV_kerosene
        mass_change_type1 = 1  # change (lighter)
        LHV2 = Easterisk
        mass_change_type2 = 0  # same
    elif propellant_type == "kerosene-fuelcell":
        LHV1 = LHV_kerosene
        mass_change_type1 = 1  # change (lighter)
        LHV2 = LHV_hydrogen
        mass_change_type2 = 1  # change
    elif propellant_type == "hydrogen-battery":
        LHV1 = LHV_hydrogen
        mass_change_type1 = 1  # change (lighter)
        LHV2 = Easterisk
        mass_change_type2 = 0  # same
    elif propellant_type == "hydrogen-fuelcell":
        LHV1 = LHV_hydrogen
        mass_change_type1 = 1  # change (lighter)
        LHV2 = LHV_hydrogen
        mass_change_type2 = 1  # change
    elif propellant_type == "fuelcell-battery":
        LHV1 = LHV_hydrogen
        mass_change_type1 = 1  # change
        LHV2 = Easterisk
        mass_change_type2 = 0  # same

    # Mission calculation
    for ii in range(n_segment):
        segment_type = mission_segments[ii].type
        if segment_type in [
            "engineStart",
            "taxi",
            "takeOff",
            "climb",
            "descent",
            "landing",
        ]:
            if ii != 0:
                mass_start = mission_segments[ii - 1].mass_end
            else:
                mass_start = mission_segments[ii].mass_start

            mission_segments[ii].mass_start = mass_start
            energy_fraction1[ii] = 1 - (
                (1 - energy_fraction[segment_type])
                * (LHV_kerosene * eta_kerosene)
                / (LHV1 * eta_prop1_cruise)
                * (1 - mission_segments[ii].hybridization)
            )
            energy_fraction2[ii] = 1 - (
                (1 - energy_fraction[segment_type])
                * (LHV_kerosene * eta_kerosene)
                / (LHV1 * eta_prop1_cruise)
                * (LHV1 * eta_prop1_cruise)
                / (LHV2 * eta_prop2_cruise)
                * mission_segments[ii].hybridization
            )
            mass_energy1 = mass_start * (1 - energy_fraction1[ii])
            mass_energy2 = mass_start * (1 - energy_fraction2[ii])
            energy_mass1_mission_segment[ii] = mass_energy1
            energy_mass2_mission_segment[ii] = mass_energy2
            mission_segments[ii].usedFuel = mass_energy1
            mission_segments[ii].usedBatteryEnergy = mass_energy2

            if "fuelcell" in propellant_type:
                if retainH2O and propellant_type.startswith("fuelcell"):
                    mass_H2O = (
                        8.45 * mass_energy1
                    )  # 8.45 from Larminie & Dicks fraction mH2O/mH2
                elif retainH2O and propellant_type.endswith("fuelcell"):
                    mass_H2O = (
                        8.45 * mass_energy2
                    )  # 8.45 from Larminie & Dicks fraction mH2O/mH2
                else:
                    mass_H2O = 0
            else:
                mass_H2O = 0

            mass_end = (
                mass_start
                - mass_energy1 * mass_change_type1
                - mass_energy2 * mass_change_type2
                + mass_H2O
            )
            mission_segments[ii].mass_end = mass_end
            if ii != 0:
                mission_segments[ii].altitude_start = mission_segments[
                    ii - 1
                ].altitude_end

        elif segment_type == "hover":
            # Only kerosene and battery are supported for this mission segment
            if (
                propellant_type.endswith("kerosene") == False
                and propellant_type.endswith("battery") == False
            ):
                raise ValueError(
                    "So far, only kerosene and battery are supported for this mission segment",
                )

            if ii != 0:
                mass_start = mission_segments[ii - 1].mass_end
            else:
                mass_start = mission_segments[ii].mass_start

            endurance_energy1 = mission_segments[ii].endurance * (
                1 - mission_segments[ii].hybridization
            )
            endurance_energy2 = (
                mission_segments[ii].endurance * mission_segments[ii].hybridization
            )

            altitude_start = mission_segments[ii].altitude_start

            # Compute density in current altitude
            atmosphere = Atmosphere(altitude_start)
            rho_start = atmosphere.density

            # Energy1
            if propellant_type.startswith("kerosene"):
                energy_mass1_mission_segment[ii] = calculate_fuel_mass_hover_endurance(
                    mass_start,
                    thrustSafetyFactor,
                    rho_start,
                    rotorRadius,
                    eta_prop1_hover,
                    endurance_energy1,
                    SFC_hover,
                )
                mass_energy1 = energy_mass1_mission_segment[ii]
            elif propellant_type.startswith(
                "battery",
            ):  # battery is only first, if it's battery-only, so battery mass can be calculated with massStart only as the mass wouldn't change
                energy_mass1_mission_segment[ii] = calculate_battery_mass_hover(
                    mass_start,
                    thrustSafetyFactor,
                    rho_start,
                    rotorRadius,
                    eta_prop1_hover,
                    endurance_energy1,
                    Easterisk,
                )
                mass_energy1 = energy_mass1_mission_segment[ii]
                energy_fraction1[ii] = 1 - mass_energy1 / mass_start

            mass_end1 = mass_start - mass_energy1 * mass_change_type1

            # Energy2
            # in case no hybrid aircraft is calculated, energyFraction2 will return 1, because rangeEnergy2 = 0. However, calculation still needed to proceed with script
            if propellant_type.endswith("kerosene"):
                energy_mass2_mission_segment[ii] = calculate_fuel_mass_hover_endurance(
                    mass_start,
                    thrustSafetyFactor,
                    rho_start,
                    rotorRadius,
                    eta_prop2_hover,
                    endurance_energy2,
                    SFC_hover,
                )
                mass_energy2 = energy_mass2_mission_segment[ii]
            elif propellant_type.endswith("battery"):
                energy_mass2_mission_segment[ii] = calculate_battery_mass_hover(
                    mass_start,
                    thrustSafetyFactor,
                    rho_start,
                    rotorRadius,
                    eta_prop2_hover,
                    endurance_energy2,
                    Easterisk,
                )
                mass_energy2 = energy_mass2_mission_segment[ii]
                energy_fraction2[ii] = 1 - mass_energy2 / ((mass_start + mass_end1) / 2)

            mass_end = (
                mass_start
                - mass_energy1 * mass_change_type1
                - mass_energy2 * mass_change_type2
            )

            # Save results
            mission_segments[ii].lift_coefficient = 0
            mission_segments[ii].lift_drag_ratio = 0
            mission_segments[ii].mass_start = mass_start  # [kg]
            mission_segments[ii].mass_end = mass_end  # [kg]
            mission_segments[ii].fuel_used = mass_energy1  # [kg]
            mission_segments[ii].battery_mass_used = mass_energy2  # [kg]
            mission_segments[ii].airspeed_end = mission_segments[
                ii
            ].airspeed_start  # no airspeed change during hover
            mission_segments[
                ii
            ].altitude_end = altitude_start  # [m] no altitude change during hover
            mission_segments[ii].range = (
                mission_segments[ii].endurance * mission_segments[ii].airspeed_start
            )  # [m]
            mission_segments[ii].angle_of_attack = 0

        elif segment_type == "cruise":
            if ii != 0:
                mass_start = mission_segments[ii - 1].mass_end
            else:
                mass_start = mission_segments[ii].mass_start

            range_energy1 = mission_segments[ii].range * (
                1 - mission_segments[ii].hybridization
            )
            range_energy2 = (
                mission_segments[ii].range * mission_segments[ii].hybridization
            )

            altitude_start = mission_segments[ii].altitude_start  # [m]

            # Compute density in current altitude
            atmosphere = Atmosphere(altitude_start)
            rho_start = atmosphere.density

            c_L_cruise = (2 * mass_start * g) / (
                rho_start * mission_segments[ii].airspeed_start ** 2 * S_ref
            )  # Breguet FC1 = constant cL
            # if c_L_cruise > max(c_L_array) or np.isnan(c_L_cruise):
            #
            #    # in case the lift coefficient is higher than possible, do not fix to a value but keep some dependency to
            #    # enable gradient computation
            #    lift_to_drag_ratio_all_mission_segments = {"lift_to_drag_ratio_" + segment.name + "_segment_" + str(index): 0.1 for index,segment in enumerate(mission_segments, start=1)}
            #    angle_of_attack_all_mission_segments = {"angle_of_attack_" + segment.name + "_segment_" + str(index): c_L_cruise + 10 for index,segment in enumerate(mission_segments, start=1)}
            #
            #    return mass_start,mass_start,lift_to_drag_ratio_all_mission_segments,angle_of_attack_all_mission_segments
            #
            #    #raise ValueError('the available lift is not sufficient to lift the configuration!')
            # else:
            c_D_cruise = np.interp(
                c_L_cruise,
                c_L_array,
                c_D_array,
            )  # careful! constant extrapolation if out of range! Check c_L afterwards!
            alpha_cruise = np.interp(
                c_L_cruise,
                c_L_array,
                alpha_array,
            )  # careful! constant extrapolation if out of range! Check alpha afterwards!

            # Energy1
            if propellant_type.startswith("kerosene") or propellant_type.startswith(
                "hydrogen",
            ):
                energy_fraction1[ii] = calculate_fuel_fraction_range_breguet_fc1(
                    range_energy1,
                    SFC_cruise,
                    c_L_cruise,
                    c_D_cruise,
                    mission_segments[ii].airspeed_start,
                    n_v,
                )
                mass_energy1 = mass_start * (1 - energy_fraction1[ii])
                energy_mass1_mission_segment[ii] = mass_energy1
                mass_H2O_1 = 0
            elif propellant_type.startswith("fuelcell"):
                eta_prop = eta_prop1_cruise / eta_fuelcell_LHV
                energy_fraction1[ii] = (
                    calculate_fuel_fraction_range_breguet_fc1_fuelcell(
                        range_energy1,
                        c_L_cruise,
                        c_D_cruise,
                        eta_prop,
                        eta_fuelcell_voltage,
                        U_thermo,
                        mue,
                        retainH2O,
                    )
                )
                mass_energy1 = mass_start * (1 - energy_fraction1[ii])
                energy_mass1_mission_segment[ii] = mass_energy1
                if retainH2O:
                    mass_H2O_1 = (
                        8.45 * mass_energy1
                    )  # 8.45 from Larminie & Dicks fraction mH2O/mH2
                else:
                    mass_H2O_1 = 0
            elif propellant_type.startswith(
                "battery",
            ):  # battery is only first, if it's battery-only, so battery mass can be calculated with massStart only as the mass wouldn't change
                energy_mass1_mission_segment[ii] = calculate_battery_mass_range(
                    mass_start,
                    range_energy1,
                    Easterisk,
                    eta_prop1_cruise,
                    c_L_cruise,
                    c_D_cruise,
                )
                mass_energy1 = energy_mass1_mission_segment[ii]
                energy_fraction1[ii] = 1 - mass_energy1 / mass_start
                mass_H2O_1 = 0

            mass_end1 = mass_start - mass_energy1 * mass_change_type1 + mass_H2O_1

            # Energy2
            # in case no hybrid aircraft is calculated, energyFraction2 will return 1, because rangeEnergy2 = 0. However, calculation still needed to proceed with script
            if propellant_type.endswith("kerosene") or propellant_type.endswith(
                "hydrogen",
            ):
                energy_fraction2[ii] = calculate_fuel_fraction_range_breguet_fc1(
                    range_energy2,
                    SFC_cruise,
                    c_L_cruise,
                    c_D_cruise,
                    mission_segments[ii].airspeed_start,
                    n_v,
                )
                mass_energy2 = mass_start * (1 - energy_fraction2[ii])
                energy_mass2_mission_segment[ii] = mass_energy2
                mass_H2O_2 = 0
            elif propellant_type.endswith("fuelcell"):
                eta_prop = eta_prop2_cruise / eta_fuelcell_LHV
                energy_fraction2[ii] = (
                    calculate_fuel_fraction_range_breguet_fc1_fuelcell(
                        range_energy2,
                        c_L_cruise,
                        c_D_cruise,
                        eta_prop,
                        eta_fuelcell_voltage,
                        U_thermo,
                        mue,
                        retainH2O,
                    )
                )
                mass_energy2 = mass_start * (1 - energy_fraction2[ii])
                energy_mass2_mission_segment[ii] = mass_energy2
                if retainH2O:
                    mass_H2O_2 = (
                        8.45 * mass_energy2
                    )  # 8.45 from Larminie & Dicks fraction mH2O/mH2
                else:
                    mass_H2O_2 = 0

            elif propellant_type.endswith("battery"):
                energy_mass2_mission_segment[ii] = calculate_battery_mass_range(
                    (mass_start + mass_end1) / 2,
                    range_energy2,
                    Easterisk,
                    eta_prop2_cruise,
                    c_L_cruise,
                    c_D_cruise,
                )
                mass_energy2 = energy_mass2_mission_segment[ii]
                energy_fraction2[ii] = 1 - mass_energy2 / ((mass_start + mass_end1) / 2)
                mass_H2O_2 = 0

            mass_end = (
                mass_start
                - mass_energy1 * mass_change_type1
                - mass_energy2 * mass_change_type2
                + mass_H2O_1
                + mass_H2O_2
            )
            rho_end = (2 * mass_end * g) / (
                c_L_cruise * mission_segments[ii].airspeed_start ** 2 * S_ref
            )

            # Calculate altitude at the end of the segment (changes according to breguet flight condition 1)
            altitude_end = Atmosphere.from_density(rho_end).h

            # Save results
            mission_segments[ii].lift_coefficient = c_L_cruise
            mission_segments[ii].lift_drag_ratio = c_L_cruise / c_D_cruise
            mission_segments[ii].mass_start = mass_start  # [kg]
            mission_segments[ii].mass_end = mass_end  # [kg]
            mission_segments[ii].fuel_used = mass_energy1  # [kg]
            mission_segments[ii].battery_mass_used = mass_energy2  # [kg]
            mission_segments[ii].airspeed_end = mission_segments[
                ii
            ].airspeed_start  # Breguet FC1 = constant velocity
            mission_segments[ii].altitude_end = float(altitude_end)  # [m]
            mission_segments[ii].endurance = (
                mission_segments[ii].range / mission_segments[ii].airspeed_start
            )  # [s]
            mission_segments[ii].angle_of_attack = alpha_cruise

        elif segment_type == "maneuvering":
            if ii != 0:
                mass_start = mission_segments[ii - 1].mass_end
            else:
                mass_start = mission_segments[ii].mass_start

            endurance_energy1 = mission_segments[ii].endurance * (
                1 - mission_segments[ii].hybridization
            )
            endurance_energy2 = (
                mission_segments[ii].endurance * mission_segments[ii].hybridization
            )

            altitude_start = mission_segments[ii].altitude_start

            # Compute density in current altitude
            atmosphere = Atmosphere(altitude_start)
            rho_start = atmosphere.density

            c_L_maneuvering = (2 * mass_start * g) / (
                rho_start * mission_segments[ii].airspeed_start ** 2 * S_ref
            )  # Breguet FC1 = constant cL
            # if c_L_maneuvering > max(c_L_array):
            #    raise ValueError('the available lift is not sufficient to lift the configuration!')
            # else:
            c_D_maneuvering = np.interp(
                c_L_maneuvering,
                c_L_array,
                c_D_array,
            )  # careful! constant extrapolation if out of range! Check c_L afterwards!
            alpha_maneuvering = np.interp(
                c_L_maneuvering,
                c_L_array,
                alpha_array,
            )  # careful! constant extrapolation if out of range! Check alpha afterwards!

            # Energy1
            if propellant_type.startswith("kerosene") or propellant_type.startswith(
                "hydrogen",
            ):
                energy_fraction1[ii] = calculate_fuel_fraction_endurance_maneuvering(
                    mass_start,
                    endurance_energy1,
                    SFC_maneuvering,
                    T_maneuvering,
                    mission_segments[ii].airspeed_start,
                )
                mass_energy1 = mass_start * (1 - energy_fraction1[ii])
                energy_mass1_mission_segment[ii] = mass_energy1
                mass_H2O_1 = 0
            elif propellant_type.startswith("fuelcell"):
                eta_prop = eta_prop1_cruise / eta_fuelcell_LHV
                energy_fraction1[ii] = (
                    calculate_fuel_fraction_endurance_maneuvering_fuelcell(
                        mass_start,
                        endurance_energy1,
                        T_maneuvering,
                        mission_segments[ii].airspeed_start,
                        eta_prop,
                        eta_fuelcell_voltage,
                        U_thermo,
                        mue,
                        retainH2O,
                    )
                )
                mass_energy1 = mass_start * (1 - energy_fraction1[ii])
                energy_mass1_mission_segment[ii] = mass_energy1
                if retainH2O:
                    mass_H2O_1 = (
                        8.45 * mass_energy1
                    )  # 8.45 from Larminie & Dicks fraction mH2O/mH2
                else:
                    mass_H2O_1 = 0
            elif propellant_type.startswith(
                "battery",
            ):  # battery is only first, if it's battery-only, so battery mass can be calculated with massStart only as the mass wouldn't change
                energy_mass1_mission_segment[ii] = calculate_battery_mass_endurance(
                    mass_start,
                    endurance_energy1,
                    mission_segments[ii].airspeed_start,
                    Easterisk,
                    eta_prop1_cruise,
                    c_L_maneuvering,
                    c_D_maneuvering,
                )
                mass_energy1 = energy_mass1_mission_segment[ii]
                energy_fraction1[ii] = 1 - mass_energy1 / mass_start
                mass_H2O_1 = 0

            mass_end1 = mass_start - mass_energy1 * mass_change_type1 + mass_H2O_1

            # Energy2
            # in case no hybrid aircraft is calculated, energyFraction2 will return 1, because rangeEnergy2 = 0. However, calculation still needed to proceed with script
            if propellant_type.endswith("kerosene") or propellant_type.endswith(
                "hydrogen",
            ):
                energy_fraction2[ii] = calculate_fuel_fraction_endurance_maneuvering(
                    mass_start,
                    endurance_energy2,
                    SFC_maneuvering,
                    T_maneuvering,
                    mission_segments[ii].airspeed_start,
                    n_v,
                )
                mass_energy2 = mass_start * (1 - energy_fraction2[ii])
                energy_mass2_mission_segment[ii] = mass_energy2
                mass_H2O_2 = 0
            elif propellant_type.endswith("fuelcell"):
                eta_prop = eta_prop2_maneuvering / eta_fuelcell_LHV
                energy_fraction2[ii] = (
                    calculate_fuel_fraction_endurance_maneuvering_fuelcell(
                        mass_start,
                        endurance_energy2,
                        T_maneuvering,
                        mission_segments[ii].airspeed_start,
                        eta_prop,
                        eta_fuelcell_voltage,
                        U_thermo,
                        mue,
                        retainH2O,
                    )
                )
                mass_energy2 = mass_start * (1 - energy_fraction2[ii])
                energy_mass2_mission_segment[ii] = mass_energy2
                if retainH2O:
                    mass_H2O_2 = (
                        8.45 * mass_energy2
                    )  # 8.45 from Larminie & Dicks fraction mH2O/mH2
                else:
                    mass_H2O_2 = 0
            elif propellant_type.endswith("battery"):
                energy_mass2_mission_segment[ii] = calculate_battery_mass_endurance(
                    (mass_start + mass_end1) / 2,
                    endurance_energy2,
                    mission_segments[ii],
                    Easterisk,
                    eta_prop2_maneuvering,
                    c_L_maneuvering,
                    c_D_maneuvering,
                )
                mass_energy2 = energy_mass2_mission_segment[ii]
                energy_fraction2[ii] = 1 - mass_energy2 / ((mass_start + mass_end1) / 2)
                mass_H2O_2 = 0

            mass_end = (
                mass_start
                - mass_energy1 * mass_change_type1
                - mass_energy2 * mass_change_type2
                + mass_H2O_1
                + mass_H2O_2
            )

            # Save results
            mission_segments[ii].lift_coefficient = c_L_maneuvering
            mission_segments[ii].lift_drag_ratio = c_L_maneuvering / c_D_maneuvering
            mission_segments[ii].mass_start = mass_start  # [kg]
            mission_segments[ii].mass_end = mass_end  # [kg]
            mission_segments[ii].fuel_used = mass_energy1  # [kg]
            mission_segments[ii].battery_mass_used = mass_energy2  # [kg]
            mission_segments[ii].airspeed_end = mission_segments[
                ii
            ].airspeed_start  # airspeed @ end of maneuvering is constant
            mission_segments[ii].altitude_end = mission_segments[
                ii
            ].altitude_start  # altitude @ end of maneuvering is constant
            mission_segments[ii].angle_of_attack = alpha_maneuvering

        elif segment_type == "loiter":
            if ii != 0:
                mass_start = mission_segments[ii - 1].mass_end
            else:
                mass_start = mission_segments[ii].mass_start

            endurance_energy1 = mission_segments[ii].endurance * (
                1 - mission_segments[ii].hybridization
            )
            endurance_energy2 = (
                mission_segments[ii].endurance * mission_segments[ii].hybridization
            )

            altitude_start = mission_segments[ii].altitude_start

            # Compute density in current altitude
            atmosphere = Atmosphere(altitude_start)
            rho_start = atmosphere.density

            c_L_loiter = (2 * mass_start * g) / (
                rho_start * mission_segments[ii].airspeed_start ** 2 * S_ref
            )  # Breguet FC1 = constant cL
            # if c_L_loiter > max(c_L_array):
            #    raise ValueError('the available lift is not sufficient to lift the configuration!')
            # else:
            c_D_loiter = np.interp(
                c_L_loiter,
                c_L_array,
                c_D_array,
            )  # careful! constant extrapolation if out of range! Check c_L afterwards!
            alpha_loiter = np.interp(
                c_L_loiter,
                c_L_array,
                alpha_array,
            )  # careful! constant extrapolation if out of range! Check alpha afterwards!

            # Energy1
            if propellant_type.startswith("kerosene") or propellant_type.startswith(
                "hydrogen",
            ):
                energy_fraction1[ii] = calculate_fuel_fraction_endurance_breguet_fc1(
                    endurance_energy1,
                    SFC_loiter,
                    c_L_loiter,
                    c_D_loiter,
                    mission_segments[ii].airspeed_start,
                    n_v,
                )
                mass_energy1 = mass_start * (1 - energy_fraction1[ii])
                energy_mass1_mission_segment[ii] = mass_energy1
                mass_H2O_1 = 0
            elif propellant_type.startswith("fuelcell"):
                eta_prop = eta_prop1_cruise / eta_fuelcell_LHV
                energy_fraction1[ii] = (
                    calculate_fuel_fraction_endurance_breguet_fc1_fuelcell(
                        endurance_energy1,
                        c_L_loiter,
                        c_D_loiter,
                        mission_segments[ii].airspeed_start,
                        eta_prop,
                        eta_fuelcell_voltage,
                        U_thermo,
                        mue,
                        retainH2O,
                    )
                )
                mass_energy1 = mass_start * (1 - energy_fraction1[ii])
                energy_mass1_mission_segment[ii] = mass_energy1
                if retainH2O:
                    mass_H2O_1 = (
                        8.45 * mass_energy1
                    )  # 8.45 from Larminie & Dicks fraction mH2O/mH2
                else:
                    mass_H2O_1 = 0
            elif propellant_type.startswith(
                "battery",
            ):  # battery is only first, if it's battery-only, so battery mass can be calculated with massStart only as the mass wouldn't change
                energy_mass1_mission_segment[ii] = calculate_battery_mass_endurance(
                    mass_start,
                    endurance_energy1,
                    mission_segments[ii].airspeed_start,
                    Easterisk,
                    eta_prop1_cruise,
                    c_L_loiter,
                    c_D_loiter,
                )
                mass_energy1 = energy_mass1_mission_segment[ii]
                energy_fraction1[ii] = 1 - mass_energy1 / mass_start
                mass_H2O_1 = 0

            mass_end1 = mass_start - mass_energy1 * mass_change_type1 + mass_H2O_1

            # Energy2
            # in case no hybrid aircraft is calculated, energyFraction2 will return 1, because rangeEnergy2 = 0. However, calculation still needed to proceed with script
            if propellant_type.endswith("kerosene") or propellant_type.endswith(
                "hydrogen",
            ):
                energy_fraction2[ii] = calculate_fuel_fraction_endurance_breguet_fc1(
                    endurance_energy2,
                    SFC_loiter,
                    c_L_loiter,
                    c_D_loiter,
                    mission_segments[ii].airspeed_start,
                    n_v,
                )
                mass_energy2 = mass_start * (1 - energy_fraction2[ii])
                energy_mass2_mission_segment[ii] = mass_energy2
                mass_H2O_2 = 0
            elif propellant_type.endswith("fuelcell"):
                eta_prop = eta_prop2_cruise / eta_fuelcell_LHV
                energy_fraction2[ii] = (
                    calculate_fuel_fraction_endurance_breguet_fc1_fuelcell(
                        endurance_energy2,
                        c_L_loiter,
                        c_D_loiter,
                        mission_segments[ii].airspeed_start,
                        eta_prop,
                        eta_fuelcell_voltage,
                        U_thermo,
                        mue,
                        retainH2O,
                    )
                )
                mass_energy2 = mass_start * (1 - energy_fraction2[ii])
                energy_mass2_mission_segment[ii] = mass_energy2
                if retainH2O:
                    mass_H2O_2 = (
                        8.45 * mass_energy2
                    )  # 8.45 from Larminie & Dicks fraction mH2O/mH2
                else:
                    mass_H2O_2 = 0
            elif propellant_type.endswith("battery"):
                energy_mass2_mission_segment[ii] = calculate_battery_mass_endurance(
                    (mass_start + mass_end1) / 2,
                    endurance_energy2,
                    mission_segments[ii].airspeed_start,
                    Easterisk,
                    eta_prop2_loiter,
                    c_L_loiter,
                    c_D_loiter,
                )
                mass_energy2 = energy_mass2_mission_segment[ii]
                energy_fraction2[ii] = 1 - mass_energy2 / ((mass_start + mass_end1) / 2)
                mass_H2O_2 = 0

            mass_end = (
                mass_start
                - mass_energy1 * mass_change_type1
                - mass_energy2 * mass_change_type2
                + mass_H2O_1
                + mass_H2O_2
            )
            rho_end = (2 * mass_end * g) / (
                c_L_loiter * mission_segments[ii].airspeed_start ** 2 * S_ref
            )

            # Calculate altitude at the end of the segment (changes according to breguet flight condition 1)
            altitude_end = Atmosphere.from_density(rho_end).h

            # Save results
            mission_segments[ii].lift_coefficient = c_L_loiter
            mission_segments[ii].lift_drag_ratio = c_L_loiter / c_D_loiter
            mission_segments[ii].mass_start = mass_start  # [kg]
            mission_segments[ii].mass_end = mass_end  # [kg]
            mission_segments[ii].fuel_used = mass_energy1  # [kg]
            mission_segments[ii].battery_mass_used = mass_energy2  # [kg]
            mission_segments[ii].airspeed_end = mission_segments[
                ii
            ].airspeed_start  # Breguet FC1 = constant velocity
            mission_segments[ii].altitude_end = float(altitude_end)  # [m]
            mission_segments[ii].range = (
                mission_segments[ii].endurance * mission_segments[ii].airspeed_start
            )  # [m]
            mission_segments[ii].angle_of_attack = alpha_loiter

    # initialize fuel_used_total
    fuel_used_total = 0
    for segment in mission_segments:
        fuel_used_total += segment.fuel_used

    fuel_used_total = fuel_used_total * fuelReserve

    # initialize battery_mass_used_total
    battery_mass_used_total = 0
    for segment in mission_segments:
        battery_mass_used_total += segment.battery_mass_used

    battery_mass_used_total = battery_mass_used_total * batteryReserve

    lift_to_drag_ratio_all_mission_segments = {
        segment.name: segment.lift_drag_ratio
        for index, segment in enumerate(mission_segments, start=1)
    }
    angle_of_attack_all_mission_segments = {
        segment.name: segment.angle_of_attack
        for index, segment in enumerate(mission_segments, start=1)
    }
    lift_coefficient_all_mission_segments = {
        segment.name: segment.lift_coefficient
        for index, segment in enumerate(mission_segments, start=1)
    }

    return (
        fuel_used_total,
        battery_mass_used_total,
        lift_to_drag_ratio_all_mission_segments,
        angle_of_attack_all_mission_segments,
        lift_coefficient_all_mission_segments,
    )


###########################################################
from dataclasses import dataclass

from assembly import (  # TODO @Tim Environment will become an attribute of the MissionSegment
    AerodynamicPolarData,
    MissionSegment,
)
from assembly import Configuration as Base_Configuration
from assembly import MissionAssembly as Base_MissionAssembly
from assembly import PropulsionSystem as Base_PropulsionSystem
from assembly import Wing as Base_Wing
from typing_extensions import Self


@dataclass
class Options:
    wing_name_serving_for_reference_area: str = None  # the name of the wing that is to be used as a reference area for the aerodynamic coefficients
    energy_fraction = None
    n_v = None
    fuelReserve = None
    batteryReserve = None
    kwargs = None


@dataclass
class Wing:
    name: str  # name of the wing
    area: float  # wing area [m*22]

    @classmethod
    def from_component(cls, comp: Base_Wing) -> Self:
        wing = Wing(
            name=comp.name,
            area=comp.area,
        )
        if "options" in vars(comp):
            for k, v in comp.options.get(
                "raymer_general_aviation_mass_wing", {}
            ).items():
                if k in vars(wing):
                    setattr(wing, k, v)
        return wing


@dataclass
class Configuration:
    aerodynamic_polar: (
        AerodynamicPolarData  # aerodynamic polar data of the entire configuration
    )

    @classmethod
    def from_component(cls, comp: Base_Configuration) -> Self:
        configuration = Configuration(
            aerodynamic_polar=comp.aerodynamic_polar,
        )
        if "options" in vars(comp):
            for k, v in comp.options.get("roskam", {}).items():
                if k in vars(configuration):
                    setattr(configuration, k, v)
        return configuration


@dataclass
class MissionAssembly:
    mission_segments: list[MissionSegment]  # list of mission segments
    fuel_used_total: float = (
        None  # total amount of fuel required to complete the entire mission [kg]
    )
    battery_mass_used_total: float = (
        None  # battery mass required to complete the entire mission [kg]
    )
    h2o_used_total: float = None  # H2 required to complete the entire mission [kg]

    @classmethod
    def from_component(cls, comp: Base_MissionAssembly) -> Self:
        mission_assembly = MissionAssembly(
            mission_segments=comp.mission_segments,
            fuel_used_total=comp.fuel_used_total,
            battery_mass_used_total=comp.battery_mass_used_total,
            h2o_used_total=comp.h2o_used_total,
        )
        if "options" in vars(comp):
            for k, v in comp.options.get("roskam", {}).items():
                if k in vars(mission_assembly):
                    setattr(mission_assembly, k, v)
        return mission_assembly


@dataclass
class PropulsionSystem:
    type: str  # allowed types: ['hydrogen','battery','fuelcell','kerosene-hydrogen','kerosene-battery','kerosene-fuelcell','hydrogen-battery','hydrogen-fuelcell','fuelcell-battery']

    @classmethod
    def from_component(cls, comp: Base_PropulsionSystem) -> Self:
        propulsion_system = PropulsionSystem(
            type=comp.type,
        )
        if "options" in vars(comp):
            for k, v in comp.options.get("roskam", {}).items():
                if k in vars(propulsion_system):
                    setattr(propulsion_system, k, v)
        return propulsion_system


class Driver:
    def __init__(
        self,
        wings: list[Base_Wing],
        mission_assembly: MissionAssembly,
        configuration: Configuration,
        propulsion_system: PropulsionSystem,
        options: Options,
    ) -> None:
        self.options = options
        self.wings = [Wing.from_component(w) for w in wings]
        self.mission_assembly = mission_assembly
        self.configuration = configuration
        self.propulsion_system = propulsion_system
        self.fuel_used_total = None
        self.battery_mass_used_total = None
        self.lift_to_drag_ratio_all_mission_segments = {
            segment.name: 0.0 for segment in mission_assembly.mission_segments
        }
        self.angle_of_attack_all_mission_segments = {
            segment.name: 0.0 for segment in mission_assembly.mission_segments
        }
        self.lift_coefficient_all_mission_segments = {
            segment.name: 0.0 for segment in mission_assembly.mission_segments
        }

    def run(self) -> tuple[float, float, dict, dict, dict]:
        # identify wing reference area
        for wing in self.wings:
            if wing.name == self.options.wing_name_serving_for_reference_area:
                reference_area = wing.area
                # stop loop once the wing has been identified
                break

        (
            self.fuel_used_total,
            self.battery_mass_used_total,
            self.lift_to_drag_ratio_all_mission_segments,
            self.angle_of_attack_all_mission_segments,
            self.lift_coefficient_all_mission_segments,
        ) = calculate_mission_roskam_customized(
            self.mission_assembly,
            self.options.energy_fraction,
            self.propulsion_system.type,
            reference_area,
            self.options.n_v,
            self.configuration.aerodynamic_polar.lift_coefficient,
            self.configuration.aerodynamic_polar.drag_coefficient,
            self.configuration.aerodynamic_polar.alpha,
            self.options.fuelReserve,
            self.options.batteryReserve,
            **self.options.kwargs,
        )

        return (
            self.fuel_used_total,
            self.battery_mass_used_total,
            self.lift_to_drag_ratio_all_mission_segments,
            self.angle_of_attack_all_mission_segments,
            self.lift_coefficient_all_mission_segments,
        )

    def retrieve_lift_to_drag_ratio(self, segment_name: list[str]) -> float:
        lift_to_drag_ratio = self.lift_to_drag_ratio_all_mission_segments[segment_name]

        return lift_to_drag_ratio

    def retrieve_angle_of_attack(self, segment_name: list[str]) -> float:
        angle_of_attack = self.angle_of_attack_all_mission_segments[segment_name]

        return angle_of_attack

    def retrieve_lift_coefficient(self, segment_name: list[str]) -> float:
        lift_coefficient = self.lift_coefficient_all_mission_segments[segment_name]

        return lift_coefficient
