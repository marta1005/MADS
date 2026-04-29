# ruff: noqa: N806
from typing import Any
from warnings import warn

import numpy as np
from scipy.optimize import fsolve

from multiads.assembly import Environment
from multiads.scenario.aero_derivatives import AeroDerivativesVariable
from multiads.scenario.mass_properties import MassPropertiesVariable
from multiads.scenario.stability_control_properties import StabilityControlVariable
from multiads.solvers import SolverOptions


class Options(SolverOptions):
    def __init__(
        self,
        *,
        aero_property_type: str = "body",  # property type ["global", "body","wind"]
        gravitation: float = 9.81,  # [m/s^2]
        area_ref: float = 1.0,  # reference area [m^2]
        length_ref: float = 1.0,  # reference length [m]
        delta_aileron: float = 0.0,  # aileron deflection [degrees]
        delta_elevator: float = 0.0,  # elevator deflection [degrees]
        delta_rudder: float = 0.0,  # rudder deflection [degrees]
        maneuver_target_roll: float = 0.0,  # target roll attitude [degrees]
        maneuver_target_pitch: float = 0.0,  # target pitch attitude [degrees]
        maneuver_target_yaw: float = 0.0,  # target yaw attitude [degrees]
        t0: float = 1.0,  # initial guess for maneuvering time computation [s]
        **kwargs: Any,  # noqa: ANN401
    ) -> None:
        super().__init__(**kwargs)
        self.aero_property_type = aero_property_type
        self.gravitation = gravitation
        self.area_ref = area_ref
        self.length_ref = length_ref
        self.delta_aileron = delta_aileron
        self.delta_elevator = delta_aileron
        self.delta_elevator = delta_elevator
        self.delta_rudder = delta_rudder
        self.maneuver_target_roll = maneuver_target_roll
        self.maneuver_target_pitch = maneuver_target_pitch
        self.maneuver_target_yaw = maneuver_target_yaw
        self.t0 = t0


class Driver:
    """Computes Global S&C properties of the aircraft.

    Static stability:
    - static margin

    Dynamic stability:
    - short period frequency and damping ratio
    - phugoid frequency and damping ratio
    - roll subsidence time
    - spiral mode time
    - dutch roll frequency and damping ratio

    Maneuvering performance:
    - maneuvering time for roll maneuver
    - maneuvering time for pitch-up maneuver
    - maneuvering time for yawing maneuver

    The driver uses the aerodynamic properties of the aircraft to compute these values.
    """

    def __init__(
        self,
        environment: Environment,
        massproperties: MassPropertiesVariable,
        aeroderivatives: AeroDerivativesVariable,
        stability_controlproperties: StabilityControlVariable,
        options: Options,
    ) -> None:
        self.options = options
        # recover environment variables
        self.density = environment.density
        self.soundSpeed = environment.sound_speed
        self.viscosity = environment.dyn_viscosity
        self.velocity = environment.velocity

        # inner variables mapping
        self.massproperties = massproperties
        self.aeroderivatives = aeroderivatives
        self.stability_controlproperties = stability_controlproperties

        # properties
        self.static_margin: float = 0.0
        self.omega_s: float = 0.0
        self.xi_s: float = 0.0
        self.omega_p: float = 0.0
        self.xi_p: float = 0.0
        self.Tr: float = 0.0
        self.Ts: float = 0.0
        self.T2: float = 0.0
        self.omega_d: float = 0.0
        self.xi_d: float = 0.0
        self.t_roll: float = 0.0
        self.t_pitch: float = 0.0
        self.t_yaw: float = 0.0

    # compute aircraft SnC properties
    def run(self) -> None:
        # aerodynamic properties
        aero_prop = self.aeroderivatives

        # select if aerodynmic properties are in a specific reference frame
        # so far the alignement is done in design_synthesis
        # note that the point of rotation of the modes is close to the cg
        if self.options.aero_property_type != "body":  # "global" or "wind"
            msg = (
                "Unimplemented type of aerodynamic properties: "
                f"{self.options.aero_property_type}"
            )
            raise ValueError(msg)

        # static stability
        cg = self.massproperties.cg[0]
        self.static_margin = (
            100
            * (self.stability_controlproperties.neutral_point[0] - cg[0])
            / self.options.length_ref
        )

        # dynamic stability (Michael V. Cook - Flight Dynamics Principles - 2nd edition)
        Ue = self.velocity[0]  # axial velocity
        V = np.linalg.norm(self.velocity)  # norm of the velocity
        rho = self.density
        A_ref = self.options.area_ref
        l_ref = self.options.length_ref
        adim = 0.5 * rho * V**2 * A_ref
        g = self.options.gravitation

        # collect mass properties from the component
        m = self.massproperties.mass  # mass_vector[0]
        inertia = self.massproperties.inertia_vector[0]
        Ix = inertia[0]
        Iy = inertia[1]
        Iz = inertia[2]
        Ixz = inertia[4]

        # longitudinal dynamics
        lift = self.stability_controlproperties.lift
        drag = self.stability_controlproperties.drag
        Zw = -aero_prop.fz[0] * adim / Ue  # dF_y /d\alpha
        Mw = aero_prop.my[0] * adim * l_ref / Ue  # dM_y /d\alpha
        Mw_dot = aero_prop.my[5] * adim * l_ref / Ue  # dM_y /d\alpha_dot
        Mq = aero_prop.my[3] * adim * l_ref * (l_ref / V)  # dF_My /dq

        # Short period - frequency and damping ratio
        if (Mq * Zw / (Iy * m) - Mw * Ue / Iy) < 0:
            warn("Negative arg in short period frequency", stacklevel=2)

        self.omega_s = np.sqrt(np.abs(Mq * Zw / (Iy * m) - Mw * Ue / Iy))
        self.xi_s = -(Mq / Iy + Zw / m + Mw_dot * Ue / Iy) / (2 * self.omega_s)

        # Phugoid - frequency and damping ratio
        self.omega_p = g * np.sqrt(2) / Ue
        self.xi_p = (1 / np.sqrt(2)) / (lift / drag)

        # lateral dynamics
        Lv = -aero_prop.mx[1] * adim * l_ref / Ue
        Nv = -aero_prop.mz[1] * adim * l_ref / Ue
        Yv = aero_prop.fy[1] * adim / Ue
        Lp = aero_prop.mx[2] * adim * l_ref * (l_ref / V)
        Np = aero_prop.mz[2] * adim * l_ref * (l_ref / V)
        Lr = aero_prop.mx[4] * adim * l_ref * (l_ref / V)
        Nr = aero_prop.mz[4] * adim * l_ref * (l_ref / V)

        # Roll subsidance - characteristic time
        self.Tr = -(Ix * Iz - Ixz**2) / (Iz * Lp + Ixz * Np)

        # Spiral mode - characteristic time
        self.Ts = -Ue * (Lv * Np - Lp * Nv) / (g * (Lr * Nv - Lv * Nr))
        self.T2 = self.Ts * np.log(2)

        # Dutch roll - frequency and damping ratio
        if (Nr * Yv / (Iz * m) + V * Nv / Iz) < 0:
            warn("Negative arg in dutch roll frequency", stacklevel=2)

        self.omega_d = np.sqrt(np.abs(Nr * Yv / (Iz * m) + V * Nv / Iz))
        self.xi_d = -(Nr / Iz + Yv / m) / (2 * self.omega_d)

        # Maneuvering time - Rolling maneuver
        L_delta = 2 * aero_prop.mx[9] * adim * l_ref
        Lp = -aero_prop.mx[2] * adim * l_ref * (l_ref / V)

        # delta_aileron from options
        delta_a = self.options.delta_aileron * np.pi / 180.0

        def phi(t: float) -> float:
            return (
                t * L_delta * delta_a / Lp
                - (1 - np.exp(-t * Lp / Ix)) * L_delta * delta_a * Ix / (Lp**2)
            ) * 180 / np.pi - self.options.maneuver_target_roll

        self.t_roll = fsolve(phi, self.options.t0)[0]

        # Maneuvering time - Pitch-up maneuver
        M_delta = aero_prop.my[10] * adim * l_ref
        Mq_Mad = aero_prop.my[7] * adim * l_ref * (l_ref / V)
        M_alpha = aero_prop.my[0] * adim * l_ref

        # option on delta_elevator
        delta_e = self.options.delta_elevator * np.pi / 180.0

        k = M_delta * delta_e
        a = Iy
        b = -Mq_Mad
        c = -M_alpha

        r1 = (-b + np.sqrt(b**2 - 4 * a * c)) / (2 * a)
        r2 = (-b - np.sqrt(b**2 - 4 * a * c)) / (2 * a)
        yp = k / c

        C1 = (yp * r2) / (r1 + r2)
        C2 = (-yp * r1) / (r1 - r2)

        def theta(t: float) -> float:
            return (
                -self.options.maneuver_target_pitch
                + (C1 * np.exp(r1 * t) + C2 * np.exp(r2 * t) + yp) * 180 / np.pi
            )

        self.t_pitch = fsolve(theta, self.options.t0)[0]

        # Maneuvering time - Yawing maneuver
        N_delta = aero_prop.mz[11] * adim * l_ref
        Nr_Nbd = aero_prop.mz[8] * adim * l_ref * (l_ref / V)
        N_beta = aero_prop.mz[1] * adim * l_ref

        # option on delta_rudder
        delta_r = self.options.delta_rudder * np.pi / 180.0

        k = N_delta * delta_r
        a = Iz
        b = -Nr_Nbd
        c = -N_beta

        r1 = (-b + np.sqrt(b**2 - 4 * a * c)) / (2 * a)
        r2 = (-b - np.sqrt(b**2 - 4 * a * c)) / (2 * a)
        yp = k / c

        C1 = (yp * r2) / (r1 + r2)
        C2 = (-yp * r1) / (r1 - r2)

        def psi(t: float) -> float:
            return (
                -self.options.maneuver_target_yaw
                + (C1 * np.exp(r1 * t) + C2 * np.exp(r2 * t) + yp) * 180 / np.pi
            )

        self.t_yaw = fsolve(psi, self.options.t0)[0]
