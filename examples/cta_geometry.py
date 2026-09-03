from __future__ import annotations

import csv
import json
import logging
import re
from math import atan2, degrees, radians, sin, tan
from pathlib import Path
from typing import Mapping

import gemseo
import numpy as np
from scipy.stats import qmc

from multiads.assembly import AirfoilCST, Section, Span, Wing, WingGeometrySpec
from multiads.cases.cta_laws import (
    build_cta_resolved_station_factory,
    build_cta_span_station_array,
)
from multiads.disciplines import UserDefined
from multiads.disciplines.geometry import Geometry
from multiads.scenario import VariableFloat
from multiads.solvers.synthesis import (
    CSTGeometrySolver,
    PyGeoExportSolver,
    build_resolved_surface_mesh,
    quintic_c2_transition,
    write_resolved_surface_mesh_npz,
)
from multiads.utilities.internal_volume_constraints import (
    CadReferenceFrame,
    evaluate_internal_volume_constraints,
    load_internal_volume_constraint_set,
)


gemseo.configure_logger(
    level=logging.INFO,
    filename="gemseo.log",
    filemode="w",
)


REPO_ROOT = Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------------------
# Official CTA/BWB data
# ---------------------------------------------------------------------------
CTA_CST_N1 = 0.35
CTA_CST_N2 = 1.10
CTA_NUM_CHORDWISE_SURFACE_POINTS = 241
CTA_NUM_AIRFOIL_POINTS = 2 * CTA_NUM_CHORDWISE_SURFACE_POINTS - 1
CTA_NUM_BASE_STATIONS = 41
CTA_SECTION_CURVE_N_CTL = 41
CTA_K_SPAN = 4

CTA_SECTION_ORDER = ("s0", "s1", "s1a", "s2", "s3", "s4", "s4a", "s4b", "s5")
CTA_OUTER_SECTION_ORDER = ("s4", "s4a", "s4b", "s5")
CTA_S1A_BLEND = (3.765 - 1.9) / (5.694 - 1.9)

CTA_BASELINE_CHORDS_M = {
    "s0": 41.203,
    "s1": 39.246,
    "s2": 30.139,
    "s3": 13.927,
    "s5": 0.8,
}
CTA_BASELINE_Y_M = {
    "s0": 0.0,
    "s1": 1.9,
    "s1a": 3.765,
    "s2": 5.694,
    "s3": 8.041,
}
CTA_BASELINE_LE_X_M = {
    "s0": 3.513099,
    "s1": 5.418099,
    "s2": 14.300099,
    "s3": 19.299046611133836,
}
CTA_BASELINE_LE_Z_M = {
    "s0": 0.25865,
    "s3": 0.67693,
    "s4": 0.72992,
    "s5": 1.97538,
}
CTA_BASELINE_TWIST_DEG = {
    "s0": 0.778,
    "s3": -0.249,
    "s4": 0.483,
    "s5": 3.177,
}

CTA_PROFILE_DATA = {
    "s0": {
        "upper": (
            0.18181997367146724,
            0.1709935828122251,
            0.1455488788794331,
            0.14465565111615697,
            0.1858364950933948,
            0.24688319276374468,
        ),
        "lower": (
            0.09022353282726805,
            0.12382620498018837,
            0.15591644781925368,
            0.21753638510529727,
            0.26350547424055293,
            0.2664047176730415,
        ),
        "te": 0.0017858758327404455,
    },
    "s1": {
        "upper": (
            0.18181997367146724,
            0.1709935828122251,
            0.1455488788794331,
            0.14465565111615697,
            0.1858364950933948,
            0.24688319276374468,
        ),
        "lower": (
            0.09022353282726805,
            0.12382620498018837,
            0.15591644781925368,
            0.21753638510529727,
            0.26350547424055293,
            0.2664047176730415,
        ),
        "te": 0.0017858758327404453,
    },
    "s1a": {
        "upper": (
            0.1700908354261385,
            0.1643753055064036,
            0.14805657488716,
            0.1544418885010697,
            0.19590214626011704,
            0.252186884737471,
        ),
        "lower": (
            0.0898668933613078,
            0.11886322464635057,
            0.13378061679706021,
            0.17967520723399505,
            0.24218543023729194,
            0.29117574867750917,
        ),
        "te": 0.0018185084815651514,
    },
    "s2": {
        "upper": (
            0.15795919592198882,
            0.15752991305443062,
            0.1506503258785301,
            0.16456395387130923,
            0.2063132138743836,
            0.25767258008187743,
        ),
        "lower": (
            0.08949801532922344,
            0.1137299329230406,
            0.11088516476391751,
            0.1405147717842674,
            0.22013376005912008,
            0.3167968311480819,
        ),
        "te": 0.001852260963915209,
    },
    "s3": {
        "upper": (
            0.15253546316853694,
            0.2387366087679237,
            0.24824699180387344,
            0.20673681995442905,
            0.16032373313699053,
            0.1184220073950578,
        ),
        "lower": (
            0.1114625681666329,
            0.14152962008931408,
            0.16230417597184305,
            0.18010566020175103,
            0.17187638770567706,
            0.14132985823466146,
        ),
        "te": 0.0030467512248412913,
    },
    "s4": {
        "upper": (
            0.0919011193097254,
            0.11833457813052178,
            0.156522349087152,
            0.17526797173908723,
            0.1566348341118798,
            0.11600648886691159,
        ),
        "lower": (
            0.05054878264316995,
            0.07848032323305172,
            0.10838842759698347,
            0.12912560835240508,
            0.12119807395342273,
            0.09136101004815043,
        ),
        "te": 0.002362304664394475,
    },
    "s4a": {
        "upper": (
            0.07396713249170112,
            0.10482952253942893,
            0.14258197086577254,
            0.16170691330802373,
            0.14474934466260062,
            0.10556421271045296,
        ),
        "lower": (
            0.06542443182640965,
            0.08707088540586227,
            0.11610685359958275,
            0.13561715527244314,
            0.12479536745742442,
            0.09165229184099036,
        ),
        "te": 0.002353649618446621,
    },
    "s4b": {
        "upper": (
            0.05603314567367683,
            0.0913244669483361,
            0.12864159264439307,
            0.1481458548769602,
            0.1328638552133214,
            0.09512193655399435,
        ),
        "lower": (
            0.08030008100964936,
            0.09566144757867284,
            0.12382527960218204,
            0.1421087021924812,
            0.1283926609614261,
            0.09194357363383027,
        ),
        "te": 0.00233211894786872,
    },
    "s5": {
        "upper": (
            0.03809915885565254,
            0.07781941135724325,
            0.11470121442301358,
            0.13458479644589671,
            0.12097836576404221,
            0.08467966039753574,
        ),
        "lower": (
            0.0951757301928891,
            0.10425200975148341,
            0.13154370560478132,
            0.1486002491125193,
            0.1319899544654278,
            0.0922348554266702,
        ),
        "te": 0.0021855582271015795,
    },
}

_CTA_BASELINE_ABS_TE_S4 = (CTA_BASELINE_CHORDS_M["s3"] * 0.5578) * CTA_PROFILE_DATA["s4"]["te"]
_CTA_BASELINE_ABS_TE_S5 = CTA_BASELINE_CHORDS_M["s5"] * CTA_PROFILE_DATA["s5"]["te"]
CTA_BASELINE_ABSOLUTE_TE_M = {
    "s0": CTA_BASELINE_CHORDS_M["s0"] * CTA_PROFILE_DATA["s0"]["te"],
    "s1": CTA_BASELINE_CHORDS_M["s1"] * CTA_PROFILE_DATA["s0"]["te"],
    "s1a": (
        CTA_BASELINE_CHORDS_M["s1"]
        + CTA_S1A_BLEND * (CTA_BASELINE_CHORDS_M["s2"] - CTA_BASELINE_CHORDS_M["s1"])
    )
    * CTA_PROFILE_DATA["s1a"]["te"],
    "s2": CTA_BASELINE_CHORDS_M["s2"] * CTA_PROFILE_DATA["s2"]["te"],
    "s3": CTA_BASELINE_CHORDS_M["s3"] * CTA_PROFILE_DATA["s3"]["te"],
    "s4": _CTA_BASELINE_ABS_TE_S4,
    "s4a": (2.0 * _CTA_BASELINE_ABS_TE_S4 + _CTA_BASELINE_ABS_TE_S5) / 3.0,
    "s4b": (_CTA_BASELINE_ABS_TE_S4 + 2.0 * _CTA_BASELINE_ABS_TE_S5) / 3.0,
    "s5": _CTA_BASELINE_ABS_TE_S5,
}


CTA_CFD_JUNE_14_VARIABLE_INFO = {
    "delta_c0_m": (0.0, -2.203, 1.797),
    "delta_c3_m": (0.0, -0.927, 2.073),
    "delta_c5_m": (0.0, 0.0, 1.0),
    "taper_ratio_midwing": (0.5578, 0.45, 0.60),
    "rspan_midwing": (0.142, 0.13, 0.21),
    "span_wing_m": (31.4585, 28.0, 35.0),
    "sweep_midwing_deg": (34.6, 15.0, 45.0),
    "sweep_outwing_deg": (24.7, 22.0, 40.0),
    "twist_s4_deg": (0.483, -1.517, 2.483),
    "twist_s4a_deg": (1.381, -0.619, 3.381),
    "twist_s4b_deg": (2.279, 0.279, 4.279),
    "twist_s5_deg": (3.177, 1.177, 5.177),
    "thickness_s4": (0.1027, 0.08, 0.16),
    "thickness_s5": (0.095, 0.08, 0.13),
}


def _complete_design_values(values: Mapping[str, float] | None = None) -> dict[str, float]:
    design_values = {
        name: float(info[0])
        for name, info in CTA_CFD_JUNE_14_VARIABLE_INFO.items()
    }
    if values:
        for key, value in values.items():
            if key in design_values:
                design_values[key] = float(np.ravel(value)[0])
    return design_values


def derive_cfd_geometry_values(values: Mapping[str, float] | None = None) -> dict[str, dict[str, float]]:
    """Derive official CTA geometry parameters from the 14 CFD variables."""

    design = _complete_design_values(values)
    chords = {
        "s0": CTA_BASELINE_CHORDS_M["s0"] + design["delta_c0_m"],
        "s1": CTA_BASELINE_CHORDS_M["s1"] + design["delta_c0_m"],
        "s2": CTA_BASELINE_CHORDS_M["s2"] + design["delta_c0_m"],
        "s3": CTA_BASELINE_CHORDS_M["s3"] + design["delta_c3_m"],
        "s5": CTA_BASELINE_CHORDS_M["s5"] + design["delta_c5_m"],
    }
    chords["s1a"] = chords["s1"] + CTA_S1A_BLEND * (chords["s2"] - chords["s1"])
    chords["s4"] = design["taper_ratio_midwing"] * chords["s3"]
    chords["s4a"] = (2.0 * chords["s4"] + chords["s5"]) / 3.0
    chords["s4b"] = (chords["s4"] + 2.0 * chords["s5"]) / 3.0

    y = dict(CTA_BASELINE_Y_M)
    y["s4"] = y["s3"] + design["rspan_midwing"] * design["span_wing_m"]
    y["s5"] = y["s3"] + design["span_wing_m"]
    y["s4a"] = (2.0 * y["s4"] + y["s5"]) / 3.0
    y["s4b"] = (y["s4"] + 2.0 * y["s5"]) / 3.0

    le_x = {
        "s0": CTA_BASELINE_LE_X_M["s0"],
        "s1": CTA_BASELINE_LE_X_M["s1"],
        "s2": CTA_BASELINE_LE_X_M["s2"],
        "s3": CTA_BASELINE_LE_X_M["s3"],
    }
    le_x["s1a"] = le_x["s1"] + CTA_S1A_BLEND * (le_x["s2"] - le_x["s1"])
    x4_ref = le_x["s3"] + 0.50 * chords["s3"] + tan(radians(design["sweep_midwing_deg"])) * (y["s4"] - y["s3"])
    le_x["s4"] = x4_ref - 0.50 * chords["s4"]
    x5_ref = le_x["s4"] + 0.25 * chords["s4"] + tan(radians(design["sweep_outwing_deg"])) * (y["s5"] - y["s4"])
    le_x["s5"] = x5_ref - 0.25 * chords["s5"]
    le_x["s4a"] = (2.0 * le_x["s4"] + le_x["s5"]) / 3.0
    le_x["s4b"] = (le_x["s4"] + 2.0 * le_x["s5"]) / 3.0

    le_z, twist = _derive_z_and_twist(design, chords, y)
    trailing_edge_thickness = _derive_trailing_edge_thickness(chords)
    outer_thickness = {
        "s4": design["thickness_s4"],
        "s4a": (2.0 * design["thickness_s4"] + design["thickness_s5"]) / 3.0,
        "s4b": (design["thickness_s4"] + 2.0 * design["thickness_s5"]) / 3.0,
        "s5": design["thickness_s5"],
    }

    return {
        "chords_m": chords,
        "spanwise_y_m": y,
        "leading_edge_x_m": le_x,
        "leading_edge_z_m": le_z,
        "twist_deg": twist,
        "trailing_edge_thickness": trailing_edge_thickness,
        "outer_thickness": outer_thickness,
    }


def _derive_z_and_twist(
    design: Mapping[str, float],
    chords: Mapping[str, float],
    y: Mapping[str, float],
) -> tuple[dict[str, float], dict[str, float]]:
    le_z: dict[str, float] = {}
    twist: dict[str, float] = {}
    zle0 = CTA_BASELINE_LE_Z_M["s0"]
    zle3 = CTA_BASELINE_LE_Z_M["s3"]
    zte0 = zle0 + CTA_BASELINE_CHORDS_M["s0"] * sin(radians(CTA_BASELINE_TWIST_DEG["s0"]))
    zte3 = zle3 + CTA_BASELINE_CHORDS_M["s3"] * sin(radians(CTA_BASELINE_TWIST_DEG["s3"]))
    zle_slope_3 = (zle3 - zle0) / max(y["s3"] - y["s0"], 1.0e-12)
    zte_slope_3 = (zte3 - zte0) / max(y["s3"] - y["s0"], 1.0e-12)

    for name in ("s0", "s1", "s1a", "s2", "s3"):
        yy = y[name]
        zle = quintic_c2_transition(yy, y["s0"], y["s3"], zle0, zle3, 0.0, zle_slope_3)
        zte = quintic_c2_transition(yy, y["s0"], y["s3"], zte0, zte3, 0.0, zte_slope_3)
        le_z[name] = zle
        twist[name] = degrees(np.arcsin(np.clip((zte - zle) / max(chords[name], 1.0e-12), -0.95, 0.95)))

    qz4 = CTA_BASELINE_LE_Z_M["s4"] + 0.25 * (CTA_BASELINE_CHORDS_M["s3"] * 0.5578) * sin(
        radians(CTA_BASELINE_TWIST_DEG["s4"])
    )
    qz5 = CTA_BASELINE_LE_Z_M["s5"] + 0.25 * CTA_BASELINE_CHORDS_M["s5"] * sin(
        radians(CTA_BASELINE_TWIST_DEG["s5"])
    )
    for name in CTA_OUTER_SECTION_ORDER:
        eta = np.clip((y[name] - y["s4"]) / max(y["s5"] - y["s4"], 1.0e-12), 0.0, 1.0)
        qz = (1.0 - eta) * qz4 + eta * qz5
        twist_name = f"twist_{name}_deg"
        twist[name] = float(design[twist_name])
        le_z[name] = float(qz - 0.25 * chords[name] * sin(radians(twist[name])))

    return le_z, twist


def _derive_trailing_edge_thickness(chords: Mapping[str, float]) -> dict[str, float]:
    te: dict[str, float] = {}
    te["s0"] = CTA_BASELINE_ABSOLUTE_TE_M["s0"] / max(chords["s0"], 1.0e-12)
    te["s1"] = CTA_BASELINE_ABSOLUTE_TE_M["s1"] / max(chords["s1"], 1.0e-12)
    te["s2"] = CTA_BASELINE_ABSOLUTE_TE_M["s2"] / max(chords["s2"], 1.0e-12)
    te["s1a"] = CTA_BASELINE_ABSOLUTE_TE_M["s1a"] / max(chords["s1a"], 1.0e-12)
    te["s3"] = CTA_BASELINE_ABSOLUTE_TE_M["s3"] / max(chords["s3"], 1.0e-12)
    te["s4"] = CTA_BASELINE_ABSOLUTE_TE_M["s4"] / max(chords["s4"], 1.0e-12)
    te["s5"] = CTA_BASELINE_ABSOLUTE_TE_M["s5"] / max(chords["s5"], 1.0e-12)
    te["s4a"] = CTA_BASELINE_ABSOLUTE_TE_M["s4a"] / max(chords["s4a"], 1.0e-12)
    te["s4b"] = CTA_BASELINE_ABSOLUTE_TE_M["s4b"] / max(chords["s4b"], 1.0e-12)
    return te


CTA_BASELINE_GEOMETRY = derive_cfd_geometry_values()


# ---------------------------------------------------------------------------
# Variables
# ---------------------------------------------------------------------------
def _design_variable(name: str) -> VariableFloat:
    baseline, lower_bound, upper_bound = CTA_CFD_JUNE_14_VARIABLE_INFO[name]
    return VariableFloat(f"cta_{name}", baseline, lb=lower_bound, ub=upper_bound)


delta_c0_m = _design_variable("delta_c0_m")
delta_c3_m = _design_variable("delta_c3_m")
delta_c5_m = _design_variable("delta_c5_m")
taper_ratio_midwing = _design_variable("taper_ratio_midwing")
rspan_midwing = _design_variable("rspan_midwing")
span_wing_m = _design_variable("span_wing_m")
sweep_midwing_deg = _design_variable("sweep_midwing_deg")
sweep_outwing_deg = _design_variable("sweep_outwing_deg")
twist_s4_deg = _design_variable("twist_s4_deg")
twist_s4a_deg = _design_variable("twist_s4a_deg")
twist_s4b_deg = _design_variable("twist_s4b_deg")
twist_s5_deg = _design_variable("twist_s5_deg")
thickness_s4 = _design_variable("thickness_s4")
thickness_s5 = _design_variable("thickness_s5")

CTA_CFD_JUNE_14_DESIGN_VARIABLES = [
    delta_c0_m,
    delta_c3_m,
    delta_c5_m,
    taper_ratio_midwing,
    rspan_midwing,
    span_wing_m,
    sweep_midwing_deg,
    sweep_outwing_deg,
    twist_s4_deg,
    twist_s4a_deg,
    twist_s4b_deg,
    twist_s5_deg,
    thickness_s4,
    thickness_s5,
]

cst_n1 = VariableFloat("cta_cst_n1", CTA_CST_N1)
cst_n2 = VariableFloat("cta_cst_n2", CTA_CST_N2)


def _section_variable(group: str, section_id: str, value: float) -> VariableFloat:
    return VariableFloat(f"cta_{group}_{section_id}", value)


_baseline = CTA_BASELINE_GEOMETRY

chord_c0 = _section_variable("chord", "c0", _baseline["chords_m"]["s0"])
chord_s1 = _section_variable("chord", "s1", _baseline["chords_m"]["s1"])
chord_s1a = _section_variable("chord", "s1a", _baseline["chords_m"]["s1a"])
chord_s2 = _section_variable("chord", "s2", _baseline["chords_m"]["s2"])
chord_c3 = _section_variable("chord", "c3", _baseline["chords_m"]["s3"])
chord_c4 = _section_variable("chord", "c4", _baseline["chords_m"]["s4"])
chord_s4a = _section_variable("chord", "s4a", _baseline["chords_m"]["s4a"])
chord_s4b = _section_variable("chord", "s4b", _baseline["chords_m"]["s4b"])
chord_c5 = _section_variable("chord", "c5", _baseline["chords_m"]["s5"])
chord_s0 = chord_c0
chord_s3 = chord_c3
chord_s4 = chord_c4
chord_s5 = chord_c5

y_s0 = _section_variable("y", "s0", _baseline["spanwise_y_m"]["s0"])
y_s1 = _section_variable("y", "s1", _baseline["spanwise_y_m"]["s1"])
y_s1a = _section_variable("y", "s1a", _baseline["spanwise_y_m"]["s1a"])
y_s2 = _section_variable("y", "s2", _baseline["spanwise_y_m"]["s2"])
y_s3 = _section_variable("y", "s3", _baseline["spanwise_y_m"]["s3"])
y_s4 = _section_variable("y", "s4", _baseline["spanwise_y_m"]["s4"])
y_s4a = _section_variable("y", "s4a", _baseline["spanwise_y_m"]["s4a"])
y_s4b = _section_variable("y", "s4b", _baseline["spanwise_y_m"]["s4b"])
y_s5 = _section_variable("y", "s5", _baseline["spanwise_y_m"]["s5"])
y_c0 = y_s0
y_c3 = y_s3
y_c4 = y_s4
y_c5 = y_s5

le_x_s0 = _section_variable("le_x", "s0", _baseline["leading_edge_x_m"]["s0"])
le_x_s1 = _section_variable("le_x", "s1", _baseline["leading_edge_x_m"]["s1"])
le_x_s1a = _section_variable("le_x", "s1a", _baseline["leading_edge_x_m"]["s1a"])
le_x_s2 = _section_variable("le_x", "s2", _baseline["leading_edge_x_m"]["s2"])
le_x_s3 = _section_variable("le_x", "s3", _baseline["leading_edge_x_m"]["s3"])
le_x_s4 = _section_variable("le_x", "s4", _baseline["leading_edge_x_m"]["s4"])
le_x_s4a = _section_variable("le_x", "s4a", _baseline["leading_edge_x_m"]["s4a"])
le_x_s4b = _section_variable("le_x", "s4b", _baseline["leading_edge_x_m"]["s4b"])
le_x_s5 = _section_variable("le_x", "s5", _baseline["leading_edge_x_m"]["s5"])
le_x_c0 = le_x_s0
le_x_c3 = le_x_s3
le_x_c4 = le_x_s4
le_x_c5 = le_x_s5

le_z_s0 = _section_variable("le_z", "s0", _baseline["leading_edge_z_m"]["s0"])
le_z_s1 = _section_variable("le_z", "s1", _baseline["leading_edge_z_m"]["s1"])
le_z_s1a = _section_variable("le_z", "s1a", _baseline["leading_edge_z_m"]["s1a"])
le_z_s2 = _section_variable("le_z", "s2", _baseline["leading_edge_z_m"]["s2"])
le_z_s3 = _section_variable("le_z", "s3", _baseline["leading_edge_z_m"]["s3"])
le_z_s4 = _section_variable("le_z", "s4", _baseline["leading_edge_z_m"]["s4"])
le_z_s4a = _section_variable("le_z", "s4a", _baseline["leading_edge_z_m"]["s4a"])
le_z_s4b = _section_variable("le_z", "s4b", _baseline["leading_edge_z_m"]["s4b"])
le_z_s5 = _section_variable("le_z", "s5", _baseline["leading_edge_z_m"]["s5"])

twist_s0 = _section_variable("twist", "s0", _baseline["twist_deg"]["s0"])
twist_s1 = _section_variable("twist", "s1", _baseline["twist_deg"]["s1"])
twist_s1a = _section_variable("twist", "s1a", _baseline["twist_deg"]["s1a"])
twist_s2 = _section_variable("twist", "s2", _baseline["twist_deg"]["s2"])
twist_s3 = _section_variable("twist", "s3", _baseline["twist_deg"]["s3"])
twist_s4 = twist_s4_deg
twist_s4a = twist_s4a_deg
twist_s4b = twist_s4b_deg
twist_s5 = twist_s5_deg

te_s0 = _section_variable("te", "s0", _baseline["trailing_edge_thickness"]["s0"])
te_s1 = _section_variable("te", "s1", _baseline["trailing_edge_thickness"]["s1"])
te_s1a = _section_variable("te", "s1a", _baseline["trailing_edge_thickness"]["s1a"])
te_s2 = _section_variable("te", "s2", _baseline["trailing_edge_thickness"]["s2"])
te_s3 = _section_variable("te", "s3", _baseline["trailing_edge_thickness"]["s3"])
te_s4 = _section_variable("te", "s4", _baseline["trailing_edge_thickness"]["s4"])
te_s4a = _section_variable("te", "s4a", _baseline["trailing_edge_thickness"]["s4a"])
te_s4b = _section_variable("te", "s4b", _baseline["trailing_edge_thickness"]["s4b"])
te_s5 = _section_variable("te", "s5", _baseline["trailing_edge_thickness"]["s5"])

SECTION_VARIABLES = {
    "s0": (chord_s0, y_s0, le_x_s0, le_z_s0, twist_s0, te_s0),
    "s1": (chord_s1, y_s1, le_x_s1, le_z_s1, twist_s1, te_s1),
    "s1a": (chord_s1a, y_s1a, le_x_s1a, le_z_s1a, twist_s1a, te_s1a),
    "s2": (chord_s2, y_s2, le_x_s2, le_z_s2, twist_s2, te_s2),
    "s3": (chord_s3, y_s3, le_x_s3, le_z_s3, twist_s3, te_s3),
    "s4": (chord_s4, y_s4, le_x_s4, le_z_s4, twist_s4, te_s4),
    "s4a": (chord_s4a, y_s4a, le_x_s4a, le_z_s4a, twist_s4a, te_s4a),
    "s4b": (chord_s4b, y_s4b, le_x_s4b, le_z_s4b, twist_s4b, te_s4b),
    "s5": (chord_s5, y_s5, le_x_s5, le_z_s5, twist_s5, te_s5),
}

CTA_DERIVED_GEOMETRY_VARIABLES = [
    chord_c0,
    chord_s1,
    chord_s1a,
    chord_s2,
    chord_c3,
    chord_c4,
    chord_s4a,
    chord_s4b,
    chord_c5,
    y_s0,
    y_s1,
    y_s1a,
    y_s2,
    y_s3,
    y_s4,
    y_s4a,
    y_s4b,
    y_s5,
    le_x_s0,
    le_x_s1,
    le_x_s1a,
    le_x_s2,
    le_x_s3,
    le_x_s4,
    le_x_s4a,
    le_x_s4b,
    le_x_s5,
    le_z_s0,
    le_z_s1,
    le_z_s1a,
    le_z_s2,
    le_z_s3,
    le_z_s4,
    le_z_s4a,
    le_z_s4b,
    le_z_s5,
    twist_s0,
    twist_s1,
    twist_s1a,
    twist_s2,
    twist_s3,
    te_s0,
    te_s1,
    te_s1a,
    te_s2,
    te_s3,
    te_s4,
    te_s4a,
    te_s4b,
    te_s5,
]

CTA_FIXED_GEOMETRY_PARAMETERS = [cst_n1, cst_n2]


def _unique_variables(items: list[VariableFloat]) -> list[VariableFloat]:
    seen: set[int] = set()
    out: list[VariableFloat] = []
    for item in items:
        item_id = id(item)
        if item_id in seen:
            continue
        seen.add(item_id)
        out.append(item)
    return out


variables = _unique_variables(
    [
        *CTA_CFD_JUNE_14_DESIGN_VARIABLES,
        *CTA_FIXED_GEOMETRY_PARAMETERS,
        *CTA_DERIVED_GEOMETRY_VARIABLES,
    ]
)


# ---------------------------------------------------------------------------
# Components
# ---------------------------------------------------------------------------
def _build_section(section_id: str) -> Section:
    chord, span_y, le_x, le_z, twist, te = SECTION_VARIABLES[section_id]
    profile = CTA_PROFILE_DATA[section_id]
    return Section(
        name=f"cta_{section_id}",
        airfoil=AirfoilCST(
            name=f"cta_{section_id}_airfoil",
            upper_coefficients=profile["upper"],
            lower_coefficients=profile["lower"],
            n1=cst_n1,
            n2=cst_n2,
            trailing_edge_thickness=te,
            thickness_factor=1.0,
        ),
        chord=chord,
        twist=twist,
        spanwise_y_m=span_y,
        leading_edge_x_m=le_x,
        leading_edge_z_m=le_z,
        metadata={
            "cta_section_id": section_id,
            "cta_label": section_id.upper(),
        },
    )


sections = [_build_section(section_id) for section_id in CTA_SECTION_ORDER]


def _build_span(index: int, section_in: Section, section_out: Section) -> Span:
    y0 = float(section_in.spanwise_y_m)
    y1 = float(section_out.spanwise_y_m)
    dy = max(y1 - y0, 1.0e-12)
    return Span(
        name=f"cta_span_{index:02d}",
        length=dy,
        sweep=degrees(atan2(float(section_out.leading_edge_x_m - section_in.leading_edge_x_m), dy)),
        dihed=degrees(atan2(float(section_out.leading_edge_z_m - section_in.leading_edge_z_m), dy)),
        start_y_m=SECTION_VARIABLES[CTA_SECTION_ORDER[index]][1],
        end_y_m=SECTION_VARIABLES[CTA_SECTION_ORDER[index + 1]][1],
        leading_edge_mode="section_positions",
    )


spans = [
    _build_span(index, sections[index], sections[index + 1])
    for index in range(len(sections) - 1)
]

wing = Wing(
    name="cta_wing",
    sections=sections,
    spans=spans,
    case_name="CTA",
    symmetry=True,
    mirror=False,
    geometry=WingGeometrySpec(
        chordwise_points=CTA_NUM_AIRFOIL_POINTS,
        spanwise_stations=CTA_NUM_BASE_STATIONS,
        station_distribution="le_te",
        # build_cta_span_station_array already injects and rounds the official
        # anchor stations, matching the standalone BWB generator.
        include_anchor_stations=False,
        spanwise_law="pchip",
        section_law="pchip",
        field_laws={
            "chord": "pchip",
            "twist": "pchip",
            "leading_edge_x": "pchip",
            "leading_edge_z": "pchip",
            "airfoil": "pchip",
        },
        field_scopes={
            "chord": "global",
            "twist": "global",
            "leading_edge_x": "global",
            "leading_edge_z": "global",
            "airfoil": "global",
        },
        resolved_station_factory=build_cta_resolved_station_factory,
        span_station_factory=build_cta_span_station_array,
    ),
)
wing.cta_thickness_s4 = thickness_s4
wing.cta_thickness_s5 = thickness_s5


# ---------------------------------------------------------------------------
# Disciplines
# ---------------------------------------------------------------------------
disc_geometry = Geometry(
    name="Geometry",
    components=[wing],
    solver=CSTGeometrySolver(),
)


# ---------------------------------------------------------------------------
# DOE / DUST campaign helpers
# ---------------------------------------------------------------------------
CTA_DOE_DESIGN_VARIABLES = list(CTA_CFD_JUNE_14_DESIGN_VARIABLES)

CTA_INTERNAL_VOLUME_CONSTRAINTS_PATH = (
    REPO_ROOT / "assets" / "cta" / "internal_volume_constraints_set1.csv"
)
TRIANGLE_RESOLUTION = 8
ENABLE_INTERNAL_BOX_EVALUATION = True

GEOMETRY_METRIC_VARIABLES = [
    disc_geometry.solver.metric_outputs["cta_wing.span_m"],
    disc_geometry.solver.metric_outputs["cta_wing.planform_area_m2"],
    disc_geometry.solver.metric_outputs["cta_wing.enclosed_volume_m3"],
    disc_geometry.solver.metric_outputs["cta_wing.root_chord_m"],
    disc_geometry.solver.metric_outputs["cta_wing.tip_chord_m"],
    disc_geometry.solver.metric_outputs["cta_wing.mean_aerodynamic_chord_m"],
]

cta_planform_mapping_outputs = [
    chord_c0,
    chord_s1,
    chord_s1a,
    chord_s2,
    chord_c3,
    chord_c4,
    chord_s4a,
    chord_s4b,
    chord_c5,
    y_s4,
    y_s4a,
    y_s4b,
    y_s5,
    le_x_s1a,
    le_x_s3,
    le_x_s4,
    le_x_s4a,
    le_x_s4b,
    le_x_s5,
    le_z_s0,
    le_z_s1,
    le_z_s1a,
    le_z_s2,
    le_z_s3,
    le_z_s4,
    le_z_s4a,
    le_z_s4b,
    le_z_s5,
    twist_s0,
    twist_s1,
    twist_s1a,
    twist_s2,
    twist_s3,
    te_s0,
    te_s1,
    te_s1a,
    te_s2,
    te_s3,
    te_s4,
    te_s4a,
    te_s4b,
    te_s5,
]

geometry_valid = VariableFloat("bwb_geometry_valid", 1.0)
geometry_failure_code = VariableFloat("bwb_geometry_failure_code", 0.0)
packaging_valid = VariableFloat("bwb_packaging_valid", 1.0)
packaging_min_margin = VariableFloat("bwb_packaging_min_margin_m", 0.0)
validation_result_variables = [
    geometry_valid,
    geometry_failure_code,
    packaging_valid,
    packaging_min_margin,
]


def to_scalar(value) -> float:  # noqa: ANN001
    return float(np.ravel(value)[0])


def _cta_planform_design_mapping(  # noqa: PLR0913
    delta_c0_m,
    delta_c3_m,
    delta_c5_m,
    taper_ratio_midwing,
    rspan_midwing,
    span_wing_m,
    sweep_midwing_deg,
    sweep_outwing_deg,
    twist_s4_deg,
    twist_s4a_deg,
    twist_s4b_deg,
    twist_s5_deg,
    thickness_s4,
    thickness_s5,
):  # noqa: ANN001, ANN201
    derived = derive_cfd_geometry_values(
        {
            "delta_c0_m": to_scalar(delta_c0_m),
            "delta_c3_m": to_scalar(delta_c3_m),
            "delta_c5_m": to_scalar(delta_c5_m),
            "taper_ratio_midwing": to_scalar(taper_ratio_midwing),
            "rspan_midwing": to_scalar(rspan_midwing),
            "span_wing_m": to_scalar(span_wing_m),
            "sweep_midwing_deg": to_scalar(sweep_midwing_deg),
            "sweep_outwing_deg": to_scalar(sweep_outwing_deg),
            "twist_s4_deg": to_scalar(twist_s4_deg),
            "twist_s4a_deg": to_scalar(twist_s4a_deg),
            "twist_s4b_deg": to_scalar(twist_s4b_deg),
            "twist_s5_deg": to_scalar(twist_s5_deg),
            "thickness_s4": to_scalar(thickness_s4),
            "thickness_s5": to_scalar(thickness_s5),
        },
    )
    chords = derived["chords_m"]
    y = derived["spanwise_y_m"]
    le_x = derived["leading_edge_x_m"]
    le_z = derived["leading_edge_z_m"]
    twist = derived["twist_deg"]
    te = derived["trailing_edge_thickness"]

    return (
        chords["s0"],
        chords["s1"],
        chords["s1a"],
        chords["s2"],
        chords["s3"],
        chords["s4"],
        chords["s4a"],
        chords["s4b"],
        chords["s5"],
        y["s4"],
        y["s4a"],
        y["s4b"],
        y["s5"],
        le_x["s1a"],
        le_x["s3"],
        le_x["s4"],
        le_x["s4a"],
        le_x["s4b"],
        le_x["s5"],
        le_z["s0"],
        le_z["s1"],
        le_z["s1a"],
        le_z["s2"],
        le_z["s3"],
        le_z["s4"],
        le_z["s4a"],
        le_z["s4b"],
        le_z["s5"],
        twist["s0"],
        twist["s1"],
        twist["s1a"],
        twist["s2"],
        twist["s3"],
        te["s0"],
        te["s1"],
        te["s1a"],
        te["s2"],
        te["s3"],
        te["s4"],
        te["s4a"],
        te["s4b"],
        te["s5"],
    )


def _load_cta_internal_volume_constraints():
    return load_internal_volume_constraint_set(
        CTA_INTERNAL_VOLUME_CONSTRAINTS_PATH,
        reference_frame=CadReferenceFrame(
            name="CTA CAD reference",
            offset_x_m=0.0,
            offset_y_m=0.0,
            offset_z_m=0.0,
            mirror_about_symmetry_plane=True,
        ),
        name="CTA B359-V0 internal volume boxes",
    )


CTA_INTERNAL_VOLUME_CONSTRAINTS = (
    _load_cta_internal_volume_constraints()
    if ENABLE_INTERNAL_BOX_EVALUATION
    else None
)

all_boxes_fit = VariableFloat("bwb_all_boxes_fit", 1.0)
internal_boxes_min_margin = VariableFloat("bwb_internal_boxes_min_margin_m", 0.0)
box_result_variables: list[VariableFloat] = []
if CTA_INTERNAL_VOLUME_CONSTRAINTS is not None:
    box_result_variables.extend([all_boxes_fit, internal_boxes_min_margin])
    _mlg1_spec = next(
        s for s in CTA_INTERNAL_VOLUME_CONSTRAINTS.surfaces if s.sub_category == "MLG_1"
    )
    mlg1_vertex_variables = [
        VariableFloat(f"bwb_mlg1_vertex_{i + 1:02d}_margin_m", 0.0)
        for i in range(len(_mlg1_spec.vertices_xyz_m))
    ]
    box_result_variables.extend(mlg1_vertex_variables)


def _evaluate_internal_boxes_from_last_geometry(*_geometry_metrics):  # noqa: ANN002, ANN201
    if CTA_INTERNAL_VOLUME_CONSTRAINTS is None:
        msg = "CTA internal volume constraints are disabled."
        raise RuntimeError(msg)

    resolved_wing = disc_geometry.components[0]
    geometry = getattr(resolved_wing, "geometry_state", None)
    if geometry is None:
        msg = "CTA geometry must be resolved before evaluating internal volume boxes."
        raise RuntimeError(msg)

    result = evaluate_internal_volume_constraints(
        geometry,
        CTA_INTERNAL_VOLUME_CONSTRAINTS,
        triangle_resolution=TRIANGLE_RESOLUTION,
    )
    mlg1_result = next(r for r in result.surface_results if r.sub_category == "MLG_1")
    return (
        1.0 if result.satisfied else 0.0,
        result.minimum_margin_m,
        *mlg1_result.vertex_margins_m,
    )


def _validate_cta_geometry(*values):  # noqa: ANN002, ANN201
    span_m, area_m2, volume_m3, root_chord_m, tip_chord_m, mac_m = [
        to_scalar(value)
        for value in values[:6]
    ]
    metrics = np.asarray(
        [span_m, area_m2, volume_m3, root_chord_m, tip_chord_m, mac_m],
        dtype=float,
    )
    positive_metrics = metrics[[0, 1, 3, 4, 5]]
    if not np.all(np.isfinite(metrics)) or np.any(positive_metrics <= 0.0):
        return 0.0, 1.0, 0.0, 0.0

    if ENABLE_INTERNAL_BOX_EVALUATION:
        boxes_fit = to_scalar(values[6])
        min_margin_m = to_scalar(values[7])
        if boxes_fit < 0.5:
            return 0.0, 2.0, 0.0, min_margin_m
        return 1.0, 0.0, 1.0, min_margin_m

    return 1.0, 0.0, 1.0, 0.0


disc_planform_mapping = UserDefined(
    name="CTAPlanformDesignMapping",
    inputs=CTA_DOE_DESIGN_VARIABLES,
    outputs=cta_planform_mapping_outputs,
    expression=_cta_planform_design_mapping,
)

disc_internal_boxes = (
    UserDefined(
        name="CTAInternalBoxFit",
        inputs=[disc_geometry.solver.metric_outputs["cta_wing.enclosed_volume_m3"]],
        outputs=box_result_variables,
        expression=_evaluate_internal_boxes_from_last_geometry,
    )
    if CTA_INTERNAL_VOLUME_CONSTRAINTS is not None
    else None
)

validation_inputs = list(GEOMETRY_METRIC_VARIABLES)
if CTA_INTERNAL_VOLUME_CONSTRAINTS is not None:
    validation_inputs.extend([all_boxes_fit, internal_boxes_min_margin])

disc_geometry_validation = UserDefined(
    name="CTAGeometryValidation",
    inputs=validation_inputs,
    outputs=validation_result_variables,
    expression=_validate_cta_geometry,
)


def design_space_rows() -> list[dict[str, float | str]]:
    return [
        {
            "name": variable.name,
            "baseline": float(variable.value),
            "lower_bound": float(variable.lb),
            "upper_bound": float(variable.ub),
        }
        for variable in CTA_DOE_DESIGN_VARIABLES
    ]


def baseline_sample() -> np.ndarray:
    return np.asarray(
        [[float(variable.value) for variable in CTA_DOE_DESIGN_VARIABLES]],
        dtype=float,
    )


def _unit_samples(method: str, n_samples: int, n_variables: int, seed: int) -> np.ndarray:
    if method == "sobol":
        sampler = qmc.Sobol(d=n_variables, scramble=True, seed=seed)
        return sampler.random(n_samples)
    if method == "halton":
        sampler = qmc.Halton(d=n_variables, scramble=True, seed=seed)
        return sampler.random(n_samples)
    if method == "lhs":
        sampler = qmc.LatinHypercube(d=n_variables, seed=seed)
        return sampler.random(n_samples)
    if method == "random":
        rng = np.random.default_rng(seed)
        return rng.random((n_samples, n_variables))
    msg = f"Unsupported sampling method: {method}"
    raise ValueError(msg)


def generate_design_samples(
    method: str,
    n_samples: int,
    seed: int,
) -> list[dict[str, float | int]]:
    lower = np.asarray([float(variable.lb) for variable in CTA_DOE_DESIGN_VARIABLES], dtype=float)
    upper = np.asarray([float(variable.ub) for variable in CTA_DOE_DESIGN_VARIABLES], dtype=float)
    unit = _unit_samples(method, n_samples, len(CTA_DOE_DESIGN_VARIABLES), seed)
    values = qmc.scale(unit, lower, upper)

    rows: list[dict[str, float | int]] = []
    for index, sample in enumerate(values):
        row: dict[str, float | int] = {"case_id": index}
        for variable, value in zip(CTA_DOE_DESIGN_VARIABLES, sample):
            row[variable.name] = float(value)
        rows.append(row)
    return rows


def write_design_samples_csv(
    path: str | Path,
    *,
    method: str,
    n_samples: int,
    seed: int,
) -> Path:
    rows = generate_design_samples(method, int(n_samples), int(seed))
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["case_id", *(variable.name for variable in CTA_DOE_DESIGN_VARIABLES)]
    with output_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return output_path


def variable_csv_keys(variable_name: str) -> tuple[str, ...]:
    if variable_name.startswith("cta_"):
        return (variable_name, variable_name[4:])
    return (variable_name,)


def load_design_sample_slice(
    path: str | Path,
    *,
    sample_start: int = 0,
    sample_count: int | None = None,
) -> np.ndarray:
    sample_path = Path(path).expanduser().resolve()
    if not sample_path.exists():
        raise FileNotFoundError(f"Sample CSV does not exist: {sample_path}")

    with sample_path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))

    start = max(0, int(sample_start))
    if sample_count is None:
        selected_rows = rows[start:]
    else:
        selected_rows = rows[start : start + max(0, int(sample_count))]

    if not selected_rows:
        msg = (
            f"No DOE samples selected from {sample_path} with "
            f"sample_start={sample_start} and sample_count={sample_count}."
        )
        raise ValueError(msg)

    samples: list[list[float]] = []
    for row_index, row in enumerate(selected_rows, start=start):
        sample: list[float] = []
        for variable in CTA_DOE_DESIGN_VARIABLES:
            value: float | None = None
            for key in variable_csv_keys(variable.name):
                raw_value = row.get(key)
                if raw_value not in (None, ""):
                    value = float(raw_value)
                    break
            if value is None:
                msg = (
                    f"Missing design variable '{variable.name}' in sample CSV row "
                    f"{row_index}. Accepted column names: {variable_csv_keys(variable.name)}."
                )
                raise ValueError(msg)
            sample.append(value)
        samples.append(sample)

    return np.asarray(samples, dtype=float)


def value_for_variable(row: dict[str, str], variable_name: str) -> float:
    for column in row:
        if column == variable_name or column.endswith(f".{variable_name}"):
            return float(row[column])
        negated_name = f"-{variable_name}"
        if column == negated_name or column.endswith(f".{negated_name}"):
            return -float(row[column])
    msg = f"Could not find variable '{variable_name}' in flattened DOE dataset."
    raise KeyError(msg)


def _summary_json(
    geometry_outputs: Mapping[str, np.ndarray],
    export_outputs: Mapping[str, np.ndarray],
    output_dir: Path,
    mesh_path: Path,
    export_state,
) -> dict[str, object]:
    return {
        "case": "CTA official nine-section baseline",
        "section_order": list(CTA_SECTION_ORDER),
        "design_space": {
            variable.name: {
                "baseline": float(variable.value),
                "lower_bound": float(variable.lb),
                "upper_bound": float(variable.ub),
            }
            for variable in CTA_CFD_JUNE_14_DESIGN_VARIABLES
        },
        "geometry_outputs": {
            key: float(value[0])
            for key, value in geometry_outputs.items()
            if key.startswith("cta_wing.")
        },
        "export_outputs": {
            key: float(value[0])
            for key, value in export_outputs.items()
            if ".pygeo_" in key
        },
        "artifacts": {
            "output_dir": str(output_dir),
            "resolved_mesh_npz": str(mesh_path),
            "iges": export_state.iges_path,
            "station_airfoils": export_state.profiles_dir,
        },
    }


def main() -> None:
    input_data = {variable.name: variable.value_np.copy() for variable in variables}
    geometry_outputs = disc_geometry.execute(input_data=input_data)
    resolved_wing = disc_geometry.components[0]
    if not isinstance(resolved_wing, Wing):
        msg = "Geometry discipline did not return a Wing component."
        raise RuntimeError(msg)
    if resolved_wing.geometry_state is None:
        msg = "CTA geometry was not written back to the wing component."
        raise RuntimeError(msg)

    repo_root = Path(__file__).resolve().parents[1]
    output_dir = repo_root / "outputs" / "CTA_case" / "export" / "geometry"
    profiles_dir = output_dir / "station_airfoils"
    mesh_path = output_dir / "cta_resolved_mesh.npz"

    mesh = build_resolved_surface_mesh(resolved_wing.geometry_state)
    write_resolved_surface_mesh_npz(mesh_path, mesh)

    if resolved_wing.geometry is None:
        resolved_wing.geometry = WingGeometrySpec()
    resolved_wing.geometry.out_dir = str(profiles_dir)
    resolved_wing.geometry.iges_path = str(output_dir / "cta.igs")
    resolved_wing.geometry.export_all_resolved_stations = True
    resolved_wing.geometry.blunt_trailing_edge = True
    resolved_wing.geometry.symmetric = False
    resolved_wing.geometry.tip_style = "rounded"
    resolved_wing.geometry.section_curve_n_ctl = CTA_SECTION_CURVE_N_CTL
    resolved_wing.geometry.k_span = CTA_K_SPAN

    disc_export = Geometry(
        name="GeometryExport",
        components=[resolved_wing],
        solver=PyGeoExportSolver(),
    )
    export_outputs = disc_export.execute(input_data=input_data)
    exported_wing = disc_export.components[0]
    if not isinstance(exported_wing, Wing) or exported_wing.export_state is None:
        msg = "CTA export was not written back to the wing component."
        raise RuntimeError(msg)

    output_dir.mkdir(parents=True, exist_ok=True)
    summary = _summary_json(
        geometry_outputs,
        export_outputs,
        output_dir,
        mesh_path,
        exported_wing.export_state,
    )
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")

    print("Resolved CTA geometry outputs:")
    for key, value in geometry_outputs.items():
        if not key.startswith(f"{resolved_wing.name}."):
            continue
        print(f"  {key} = {float(value[0]):.6f}")

    print("CTA export outputs:")
    for key, value in export_outputs.items():
        if ".pygeo_" not in key:
            continue
        print(f"  {key} = {float(value[0]):.6f}")

    export_state = exported_wing.export_state
    print(f"Resolved mesh: {mesh_path}")
    print(f"IGES exported: {export_state.iges_path}")
    print(f"Airfoil .dat written to: {export_state.profiles_dir}/")


if __name__ == "__main__":
    main()
