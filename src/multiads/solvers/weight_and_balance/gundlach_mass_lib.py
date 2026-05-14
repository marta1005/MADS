from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from multiads.solvers import SolverOptions

if TYPE_CHECKING:
    from multiads.assembly import Fuselage, Propeller


def calculate_mass_propeller_uav_gundlach(
    F_prop: float,
    n_blade: float,
    diameter: float,
    power: float,
) -> float:
    """Calculate propeller mass for UAV.

    Method according to Jay Gundlach, Designing Unmanned Aircraft
    Systems: A Comprehensive Approach, 2012, p.213.

    Args:
        F_prop: Multiplication factor (recommended 15 for plastic/composite props)
        n_blade: Number of blades
        diameter: Propeller diameter [m]
        power: Maximum shaft power of motor [W]

    Returns:
        Propeller mass [kg]
    """
    m2ft = 3.28084
    W2hp = 0.00134102
    lb2kg = 0.453592

    power_hp = power * W2hp
    diameter_ft = diameter * m2ft

    m_propeller = F_prop * n_blade**0.391 * (diameter_ft * power_hp / 1000) ** 0.782
    m_propeller = m_propeller * lb2kg

    return m_propeller


def calculateMassFuselageUAVGundlach(
    F_mg: float,
    F_ng: float,
    F_press: float,
    F_vt: float,
    F_mat: float,
    length: float,
    W_carried: float,
    loadFactorMax: float,
    Veq_max: float,
) -> float:
    """Calculate fuselage structure mass for UAV.

    Method according to Jay Gundlach, Designing Unmanned Aircraft
    Systems: A Comprehensive Approach, 2012, p.208.

    Args:
        F_mg: Main gear on fuselage factor (1 if no main gear, 1.07 if main gear)
        F_ng: Nose gear on fuselage factor (1 if no nose gear, 1.04 if nose gear)
        F_press: Pressure factor (1 if unpressurized, 1.08 if pressurized)
        F_vt: Vertical tail factor (1 if VT weight not included, 1.07 if included)
        F_mat: Material factor (1 for carbon/metal, 2 for fiberglass/unknown, 2.187 for wood)
        length: Total fuselage length [m]
        W_carried: Mass of components carried within the structure [kg]
        loadFactorMax: Ultimate load factor [g]
        Veq_max: Maximum equivalent speed [m/s]

    Returns:
        Fuselage structure mass [kg]
    """
    m2ft = 3.28084
    ms2ktn = 1.94384
    lb2kg = 0.453592

    W_carried_lb = W_carried / lb2kg
    Veq_max_ktn = Veq_max * ms2ktn
    length_ft = length * m2ft

    m_fuselage = (
        0.5257
        * F_mg
        * F_ng
        * F_press
        * F_vt
        * F_mat
        * (length_ft**0.3796)
        * ((W_carried_lb * loadFactorMax) ** 0.4863)
        * Veq_max_ktn**2
    )

    m_fuselage = m_fuselage * lb2kg

    return m_fuselage


class Options(SolverOptions):
    def __init__(
        self,
        *,
        F_mg: float = 1.0,
        F_ng: float = 1.0,
        F_press: float = 1.0,
        F_vt: float = 1.0,
        F_mat: float = 1.0,
        F_prop: float = 15.0,
        propeller_diameter: float = 0.0,
        propeller_power: float = 0.0,
        fuselage_load_factor: float = 0.0,
        fuselage_Veq_max: float = 0.0,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.F_mg = F_mg
        self.F_ng = F_ng
        self.F_press = F_press
        self.F_vt = F_vt
        self.F_mat = F_mat
        self.F_prop = F_prop
        self.propeller_diameter = propeller_diameter
        self.propeller_power = propeller_power
        self.fuselage_load_factor = fuselage_load_factor
        self.fuselage_Veq_max = fuselage_Veq_max


@dataclass
class Fuselage:
    length: float
    W_carried: float
    loadFactorMax: float
    Veq_max: float

    @classmethod
    def from_component(cls, comp: Fuselage) -> Fuselage:
        fuselage = cls(
            length=comp.length,
            W_carried=comp.mass if hasattr(comp, "mass") else 0.0,
            loadFactorMax=0.0,
            Veq_max=0.0,
        )
        if hasattr(comp, "options") and comp.options:
            for k, v in comp.options.get("gundlach_mass", {}).items():
                if hasattr(fuselage, k):
                    setattr(fuselage, k, v)
        return fuselage


@dataclass
class Propeller:
    n_blade: float
    diameter: float
    power: float

    @classmethod
    def from_component(cls, comp: Propeller) -> Propeller:
        diameter = 0.0
        if hasattr(comp, "r_tip"):
            diameter = float(comp.r_tip) * 2

        propeller = cls(
            n_blade=float(getattr(comp, "n_blades", 2)),
            diameter=diameter,
            power=0.0,
        )
        if hasattr(comp, "options") and comp.options:
            for k, v in comp.options.get("gundlach_mass", {}).items():
                if hasattr(propeller, k):
                    setattr(propeller, k, v)
        return propeller


class Driver:
    def __init__(
        self,
        propeller: Propeller | None,
        fuselage: Fuselage | None,
        options: Options,
    ) -> None:
        self.propeller = propeller
        self.fuselage = fuselage
        self.options = options
        self.propeller_mass: float | None = None
        self.fuselage_mass: float | None = None

    def run(self) -> None:
        if self.propeller is not None:
            self.propeller_mass = self._compute_propeller_mass(self.propeller)

        if self.fuselage is not None:
            self.fuselage_mass = self._compute_fuselage_mass(self.fuselage)

    def _compute_propeller_mass(self, propeller: Propeller) -> float:
        diameter = propeller.diameter if propeller.diameter > 0 else self.options.propeller_diameter
        power = propeller.power if propeller.power > 0 else self.options.propeller_power

        if diameter <= 0 or power <= 0:
            return 0.0

        return calculate_mass_propeller_uav_gundlach(
            self.options.F_prop,
            propeller.n_blade,
            diameter,
            power,
        )

    def _compute_fuselage_mass(self, fuselage: Fuselage) -> float:
        load_factor = fuselage.loadFactorMax if fuselage.loadFactorMax > 0 else self.options.fuselage_load_factor
        Veq_max = fuselage.Veq_max if fuselage.Veq_max > 0 else self.options.fuselage_Veq_max

        if load_factor <= 0 or Veq_max <= 0 or fuselage.W_carried <= 0:
            return 0.0

        return calculateMassFuselageUAVGundlach(
            self.options.F_mg,
            self.options.F_ng,
            self.options.F_press,
            self.options.F_vt,
            self.options.F_mat,
            fuselage.length,
            fuselage.W_carried,
            load_factor,
            Veq_max,
        )
