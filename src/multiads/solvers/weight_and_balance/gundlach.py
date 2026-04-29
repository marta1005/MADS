def calculate_mass_propeller_uav_gundlach(F_prop, n_blade, diameter, power):
    """
    Parameters:
    F_prop            multiplication factor (see assumptions)   [double]    [1 x 1]     [-]
    n_blade           number of blades                          [double]    [1 x 1]     [-]
    diameter          propeller diameter                        [double]    [1 x 1]     [m]
    power             maximum shaft power of motor              [double]    [1 x 1]     [W]

    Returns:
    m_propeller      propeller mass                            [double]    [1 x 1]     [kg]
    Description:	This function calculates the propeller mass of a UAV.
                Method according to Jay Gundlach, Designing Unmanned Aircraft
                Systems: A Comprehensive Approach, 2012, p.213. 
    Limitations:  - 
    Assumptions:  n_prop is assumed to be equal the number of motors. A F_prop
                value of 15 is recommended for plastic or composite propellers
                for engines with less than 50 shp (= 37.3kW) shaft power. 
                Roskam recommends a KProp value of 24.0 for turboprops 
                above 1,500 shp and 31.92 for engines below 1,500 shp.
    """

    # defines
    m2ft = 3.28084
    W2hp = 0.00134102
    lb2kg = 0.453592

    # unit conversion of inputs
    power = power * W2hp
    diameter = diameter * m2ft
    
    # calculation
    m_propeller = F_prop * n_blade**0.391* (diameter* power/ 1000)**0.782

    # unit conversion of output
    m_propeller = m_propeller*lb2kg

    return m_propeller

def calculateMassFuselageUAVGundlach(F_mg, F_ng, F_press, F_vt, F_mat, length, W_carried, loadFactorMax, Veq_max):
    """
    Description:
    This function calculates the fuselage structure mass of a   
    UAV. Method according to Jay Gundlach, Designing Unmanned Aircraft
    Systems: A Comprehensive Approach, 2012, p.208.
    
    - F_mg                   main gear on fuselage factor       [double]                [1 x 1]     [-]
            = 1 if no main gear on fuselage
            = 1.07 if main gear on fuselage
    - F_ng                   nose gear on fuselage factor       [double]                [1 x 1]     [-]
            = 1 if no nose gear on fuselage
            = 1.04 if nose gear on fuselage
    - F_press                pressure factor                    [double]                [1 x 1]     [-]
            = 1 if unpressurized
            = 1.08 if pressurized
    - F_vt                   vertical tail on fuselage factor   [double]                [1 x 1]     [-]
            = 1 if VT weight not included
            = 1.07 if VT weight included
    - F_mat                  material factor                    [double]                [1 x 1]     [-]
            = 1 if carbon fiber or metal
            = 2 if fiberglass or unknown
            = 2.187 if wood
    - length                total fuselage length               [double]                [1 x 1]     [m]
    - W_carried             mass of components carried within
                            the structure                       [double]                [1 x 1]     [kg]
    - loadFactorMax         ultimum load factor                 [double]                [1 x 1]     [g]
    - Veq_max               maximum equivalent sepeed           [double]                [1 x 1]     [m/s]
    
    Returns:
    - m_fuselage            structure mass of fuselage          [double]                [1 x 1]     [kg]

    Limitations:
    Semimonocoque or composite shell fuselage for subsonic or transonic
    UAS weighing between 1 to 800,000 lb (0.5 kg to 363 t). 
    
    Assumptions:
    It is based on a curve fit to 197 fuselages, ranging from hand-launched
    gliders to the largest cargo aircraft. The reference fuselages 
    have a fineness ratio (length-to-diameter ratio) of at least 4:1.
    Most of the data are from manned sailplanes. This equation includes
    the nacelle weight for any engines contained within the fuselage.
    """

    # unit conversion factors
    m2ft = 3.28084
    ms2ktn = 1.94384
    lb2kg = 0.453592

    # unit conversion of inputs
    W_carried = W_carried / lb2kg
    Veq_max = Veq_max * ms2ktn
    length = length * m2ft

    # calculation
    m_fuselage = 0.5257 * F_mg * F_ng * F_press * F_vt * F_mat * (length**(0.3796)) * ((W_carried * loadFactorMax)**0.4863) * Veq_max**2

    # unit conversion of output
    m_fuselage = m_fuselage*lb2kg
    
    return m_fuselage