# -*- coding: utf-8 -*-
"""
# [To be update]
# Inputs
#   : max_power_rating # P = M*omega [w]

    
# parameters (within the model that can be moved to inputs)
    : throttles      # in this case defined within the model
    : psfcs          # in this case defined by the model
    : time_steps     # this can be included from the mission

# Outputs
#   : engine_mass
#   : cumulative_fuel_consumed
#   : co2_emission_estimation

"""

# for test purposes [SM]
# ------------------------------------------------------------------------
# import sys
# import os
#
# sys.path.insert(
#    0,
#    os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")),
# )
# print(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
# -------------------------------------------------------------------------
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from matplotlib import pyplot as plt
from scipy.interpolate import UnivariateSpline

# SM6
# ----------------------------------------------------------
# from scipy.integrate import simps
from scipy.integrate import simpson

from typing_extensions import Self

from multiads.solvers import SolverOptions
# from multiads.solvers.thermal.models.ips_block_model import IPS_BlockModel
from multiads.solvers.thermal.models import ips_block_model

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence
    from typing import Any

    from multiads.assembly import Environment
    from multiads.assembly import ThermalSystem as Base_ThermalSystem
    from multiads.assembly import Wing as  Base_Wing
    


class Options(SolverOptions):
    def __init__(
        self,
        *,
        AoA_min_root: float = 0.0,  # Wing/HTP/VTP root minimum angle of attack [deg] ---- Range [ 0.0 - 8.0 ]
        AoA_min_tip: float = 0.0,  # Wing/HTP/VTP tip minimum angle of attack [deg] ---- Range [ 0.0 - 8.0 ]
        AoA_max_root: float = 7.0,  # Wing/HTP/VTP root maximum angle of attack [deg] ---- Range [ 0.0 - 8.0 ]
        AoA_max_tip: float = 7.0,  # Wing/HTP/VTP tip minimum angle of attack [deg] ---- Range [ 0.0 - 8.0 ]
        wing_span_protect: float = 13.0460,  # semi-span length to protect [m] ------------ Range [ 0.0 - inf ]
        vtp_span_protect: float = 0.0,  # semi-span length to protect [m] ------------ Range [ 0.0 - inf ]
        htp_span_protect: float = 0.0,  # semi-span length to protect [m] ------------ Range [ 0.0 - inf ]
        chord_root: float = 4.56400, # Wing root chord [m] ------------------------ Range [ 2.5 - 5.0 ]
        ips_mode: int = 1,  # IPS technology selection mode -------------------------------------- Range [0 - 4] - [11,22,33,44]
        # [0] – No Ice Protection System
        # [1] – Thermoelectrical inductive de-icing system ON | [11] - Thermoelectrical inductive de-icing system OFF
        # [2] – Thermoelectrical resistive de-icing system ON | [22] - Thermoelectrical resistive de-icing system OFF
        # [3] – Pneumatic boots ice protection system ON      | [33] - Pneumatic boots ice protection system OFF
        # [4] – Hot air anti-icing system ON                  | [44] - Hot air anti-icing system OFF
        # jacobian settings
        jac_approx_type: str = "finite_differences",
        jac_approx_step: float = 1e-04,
        jac_approx_n_processes: int = 1,
        jac_approx_use_threading: bool = False,
        **kwargs: Any,  # noqa: ANN401
    ) -> None:
        super().__init__(**kwargs)
        self.AoA_min_root = AoA_min_root
        self.AoA_min_tip = AoA_min_tip
        self.AoA_max_tip = AoA_max_tip
        self.wing_span_protect = wing_span_protect
        self.vtp_span_protect = vtp_span_protect
        self.htp_span_protect = htp_span_protect
        self.chord_root = chord_root
        self.ips_mode = ips_mode
        # jacobian
        self.jac_approx_type = jac_approx_type
        self.jac_approx_step = jac_approx_step
        self.jac_approx_n_processes = jac_approx_n_processes
        self.jac_approx_use_threading = jac_approx_use_threading

class ThermalSystem:
    def __init__(
        self,
        name: str,
    ) -> None:
        self.name = name

    @classmethod
    def from_component(cls, comp: Base_ThermalSystem) -> Self:
        return cls(
            name=comp.name,
        )

class Wing:
    name: str  # name of the wing
    chord: float

    @classmethod
    def from_component(cls, comp: Base_Wing) -> Self:
        wing = Wing(
            name=comp.name,
            chord=comp.chord,
        )
        if "options" in vars(comp):
            for k, v in comp.options.get("IPSOptions", {}).items():
                if k in vars(wing):
                    setattr(wing, k, v)
        return wing


# Step 1: compute Weigh of Environment control
class _IPS:
    def __init__(self):
        return

    @staticmethod
    def compute_ice_protection(wings, options):

        wings = [Wing.from_component(w) for w in wings]

        for w, wing in enumerate(wings):
            impLimit_Root = IPS_BlockModel.impLimits(
                wing.name,  # "Wing",
                wing.chord,  # chord_WingRoot
                options.AoA_min_root,  # options.AoA_min_root,
                options.AoA_max_root,  # options.AoA_max_root,
            )
            impLimit_Tip = IPS_BlockModel.impLimits(
                wing.name,  # "Wing",
                wing.chord,  # chord_WingTip
                options.AoA_min_tip,
                options.AoA_max_tip,
            )

        ## Wing geometry and impingement limits
        # impLimit_WingRoot = IPS_BlockModel.impLimits(
        #    "Wing",
        #    self.chord_WingRoot,
        #    self.AoA_min_root_Wing,
        #    self.AoA_max_root_Wing,
        # )
        # impLimit_WingTip = IPS_BlockModel.impLimits(
        #   "Wing",
        #   IPS_data.chord_WingTip,
        #   IPS_data.AoA_min_tip_Wing,
        #   IPS_data.AoA_max_tip_Wing,
        # )
        # HTP geometry and impingement limits airfoil_data
        impLimit_HTPRoot = IPS_BlockModel.impLimits(
            "HTP",
            0.0,  # IPS_data.chord_HTPRoot,
            0.0,  # IPS_data.AoA_min_root_htp,
            0.0,  # IPS_data.AoA_max_root_htp,
        )
        impLimit_HTPTip = IPS_BlockModel.impLimits(
            "HTP",
            0.0,  # IPS_data.chord_HTPTip,
            0.0,  # IPS_data.AoA_min_tip_htp,
            0.0,  # IPS_data.AoA_max_tip_htp,
        )
        # VTP geometry and impingement limits
        impLimit_VTPRoot = IPS_BlockModel.impLimits(
            "VTP",
            0.0,  # IPS_data.chord_VTPRoot,
            0.0,  # IPS_data.AoA_min_root_vtp,
            0.0,  # IPS_data.AoA_max_root_vtp,
        )
        impLimit_VTPTip = IPS_BlockModel.impLimits(
            "VTP",
            0.0,  # IPS_data.chord_VTPTip,
            0.0,  # IPS_data.AoA_min_tip_vtp,
            0.0,  # IPS_data.AoA_max_tip_vtp,
        )

        Bleed_IPS, Power_IPS, IPS_W, IPS_V = IPS_BlockModel.IPS(
            impLimit_Root,
            impLimit_Tip,
            impLimit_HTPRoot,
            impLimit_HTPTip,
            impLimit_VTPRoot,
            impLimit_VTPTip,
            options.wing_span_protect,
            0.0,  # IPS_data.leadingEdge_HTP,
            0.0,  # IPS_data.leadingEdge_VTP,
            options.ips_mode,  # IPS_data.mode,
        )
        # Bleed_IPS, Power_IPS, IPS_W, IPS_V = IPS_BlockModel.IPS(
        #    impLimit_Root,
        #    impLimit_Tip,
        #    options.wing_span_protect,
        #    options.ips_mode,
        # )

        #######################################################################################
        # OUTPUTS_IPS                                                                         #
        #######################################################################################

        return Bleed_IPS, Power_IPS, IPS_W, IPS_V


class Driver:
    def __init__(
        self,
        environment: Environment,
        ips: list[Base_ThermalSystem],  # [To be update]
        wings: list[Base_Wing],
        options: Options,
    ) -> None:
        self.options = options
        self.height = environment.height
        self.density = environment.density
        self.soundSpeed = environment.sound_speed
        self.viscosity = environment.dyn_viscosity
        self.speed = environment.speed
        

        # instatiate the engine from the base class
        self.ips = [ThermalSystem.from_component(i) for i in ips]
        self.wings = [Wing.from_component(w) for w in wings]

        # initialise main engine QOI
        self.ips_mass : dict[str, float] = {}
        self.ips_volume : dict[str, float] = {}
        self.ips_power : dict[str, float] = {}

    def run(self) -> None:
        for ips in self.ips:
            (ips_bleed, ips_w, ips_vol, ips_power) = self._run(
                ips, self.wings, self.options, self.speed, self.height
            )
            name = ips.name
            self.ips_mass[name] = ips_w
            self.ips_volume[name] = ips_vol
            self.ips_power[name] = ips_power

    @staticmethod
    def _run(
        ips, wings, options, speed: float, height: float
    ) -> tuple[float, float, float]:
        """ """
        # get all the variables needed by the model
        height = ips.height

        IPS = _IPS()
        (
            ips_bleed,
            ips_w,
            ips_vol,
            ips_power,
        ) = IPS.compute_ice_protection(wings, options)

        return ips_bleed, ips_w, ips_vol, ips_power

    def total_mass(self) -> tuple[float]:
        if not ips_names:
            ips_names = [ip.name for ip in self.ips]
        return  np.sum([self.ips_mass.get(name, 0.0) for name in ips_names])

    def cumulative_power(self) -> tuple[float]:
        if not ips_names:
            ips_names = [ip.name for ip in self.ips]
        return np.sum([self.ips_power.get(name, 0.0) for name in ips_names])
    
    def cumulative_volume(self) -> tuple[float]:
        if not ips_names:
            ips_names = [ip.name for ip in self.ips]

        volume = 0.0
        for name in ips_names:
            try:
                i = self.mapind[name]
                volume += self.ips_volume[i]
            except KeyError:
                pass

        return self.ips_total_power


if __name__ == "__main__":

    import sys
    import os

    sys.path.insert(
        0,
        os.path.abspath(os.path.join(os.path.dirname(__file__), "../../src")),
    )
    print(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../src")))

    import models.ips_block_model
    # import IPS_BlockModel

    IPS = _IPS()

    wings = Base_Wing
    wings.chord = 4.0
    wings.name = "Wing"
    wings = [wings]
    wings = [Wing.from_component(w) for w in wings]

    options = Options()

    Bleed_IPS, Power_IPS, IPS_W, IPS_V = IPS.compute_ice_protection(
        wings,
        options,
    )
    print(Bleed_IPS, Power_IPS, IPS_W, IPS_V)
