from __future__ import annotations

import copy
import logging
from math import atan2, degrees
from pathlib import Path

import gemseo
import numpy as np

from multiads.assembly import AirfoilCST, Section, Span, Wing
from multiads.cases.cta_laws import (
    build_cta_resolved_station_factory,
    build_cta_span_station_array,
)
from multiads.disciplines.geometry import Geometry
from multiads.scenario import VariableFloat
from multiads.solvers.synthesis import CSTGeometrySolver, PyGeoExportSolver


gemseo.configure_logger(
    level=logging.INFO,
    filename="gemseo.log",
    filemode="w",
)


# ---------------------------------------------------------------------------
# CTA explicit data
# ---------------------------------------------------------------------------
CTA_CST_N1 = 0.35
CTA_CST_N2 = 1.10

CTA_C0_UPPER_CST = (
    0.18181997367146724,
    0.1709935828122251,
    0.1455488788794331,
    0.14465565111615697,
    0.1858364950933948,
    0.24688319276374468,
)
CTA_C0_LOWER_CST = (
    0.09022353282726805,
    0.12382620498018837,
    0.15591644781925368,
    0.21753638510529727,
    0.26350547424055293,
    0.2664047176730415,
)

CTA_BODY_HELPER_01_UPPER_CST = (
    0.12619522422663712,
    0.14952468554576231,
    0.12503039626141826,
    0.12044132971230177,
    0.16316655264586524,
    0.22636699854257636,
)
CTA_BODY_HELPER_01_LOWER_CST = (
    0.17182923149026957,
    0.16206498688017337,
    0.18400445979451005,
    0.24107952529396867,
    0.28467752187549944,
    0.29608094417761976,
)

CTA_BODY_HELPER_02_UPPER_CST = (
    0.15795919592198882,
    0.15752991305443062,
    0.1506503258785301,
    0.16456395387130923,
    0.2063132138743836,
    0.25767258008187743,
)
CTA_BODY_HELPER_02_LOWER_CST = (
    0.08949801532922344,
    0.1137299329230406,
    0.11088516476391751,
    0.1405147717842674,
    0.22013376005912008,
    0.3167968311480819,
)

CTA_C3_UPPER_CST = (
    0.15253546316853694,
    0.2387366087679237,
    0.24824699180387344,
    0.20673681995442905,
    0.16032373313699053,
    0.1184220073950578,
)
CTA_C3_LOWER_CST = (
    0.1114625681666329,
    0.14152962008931408,
    0.16230417597184305,
    0.18010566020175103,
    0.17187638770567706,
    0.14132985823466146,
)

CTA_C4_UPPER_CST = (
    0.0919011193097254,
    0.11833457813052178,
    0.156522349087152,
    0.17526797173908723,
    0.1566348341118798,
    0.11600648886691159,
)
CTA_C4_LOWER_CST = (
    0.05054878264316995,
    0.07848032323305172,
    0.10838842759698347,
    0.12912560835240508,
    0.12119807395342273,
    0.09136101004815043,
)

CTA_C5_UPPER_CST = (
    0.03809915885565254,
    0.07781941135724325,
    0.11470121442301358,
    0.13458479644589671,
    0.12097836576404221,
    0.08467966039753574,
)
CTA_C5_LOWER_CST = (
    0.0951757301928891,
    0.10425200975148341,
    0.13154370560478132,
    0.1486002491125193,
    0.1319899544654278,
    0.0922348554266702,
)

CTA_INTERPOLATION = {
    "spanwise_law": "pchip",
    "section_law": "pchip",
    "field_laws": {
        "chord": "pchip",
        "twist": "pchip",
        "leading_edge_x": "pchip",
        "leading_edge_z": "pchip",
        "airfoil": "pchip",
    },
    "field_scopes": {
        "chord": "global",
        "twist": "global",
        "leading_edge_x": "global",
        "leading_edge_z": "global",
        "airfoil": "global",
    },
    "metadata": {
        "resolved_station_factory": build_cta_resolved_station_factory,
        "span_station_factory": build_cta_span_station_array,
    },
}

CTA_SAMPLING = {
    "station_distribution": "le_te",
}

CTA_B1_FIXED_M = 8.041
CTA_SPAN_OUTER_BASE_M = 31.4585
CTA_SPAN_OUTER_BOUNDS_M = (28.0, 35.0)
CTA_B2_SPAN_RATIO_BASE = 0.1419998
CTA_B2_SPAN_RATIO_BOUNDS = (0.13, 0.21)
CTA_CHORD_C0_BASE_M = 41.17952274
CTA_CHORD_C0_BOUNDS_M = (39.0, 43.0)
CTA_CHORD_C3_BASE_M = 13.9269627
CTA_CHORD_C3_BOUNDS_M = (13.0, 16.0)
CTA_CHORD_C4_C3_RATIO_BASE = 0.5578
CTA_CHORD_C4_C3_RATIO_BOUNDS = (0.45, 0.60)
CTA_CHORD_C4_BASE_M = CTA_CHORD_C3_BASE_M * CTA_CHORD_C4_C3_RATIO_BASE
CTA_CHORD_C5_BASE_M = 0.8
CTA_CHORD_C5_BOUNDS_M = (0.8, 1.8)
CTA_SWEEP_S2_BASE_DEG = 34.6
CTA_SWEEP_S2_BOUNDS_DEG = (15.0, 45.0)
CTA_SWEEP_S3_BASE_DEG = 24.7
CTA_SWEEP_S3_BOUNDS_DEG = (22.0, 33.0)
CTA_MED_3_TE_SWEEP_BASE_DEG = 0.0
CTA_MED_3_TE_SWEEP_BOUNDS_DEG = (-10.0, 25.0)
CTA_TWIST_C0_C3_BASE_DEG = 1.0
CTA_TWIST_C0_C3_BOUNDS_DEG = (0.2, 2.0)
CTA_TWIST_C4_BASE_DEG = 0.8
CTA_TWIST_C4_BOUNDS_DEG = (0.1, 1.5)
CTA_TWIST_C5_BASE_DEG = 0.6
CTA_TWIST_C5_BOUNDS_DEG = (0.1, 1.0)
CTA_LE_X_C0_BASE_M = 3.513099
CTA_LE_X_BODY_HELPER_01_BASE_M = 5.418099
CTA_LE_X_BODY_HELPER_02_BASE_M = 14.279277199459
CTA_LE_X_C3_BASE_M = 19.299046611133836
CTA_LE_X_C4_BASE_M = 25.459944426354326
CTA_LE_X_C5_BASE_M = 39.61671597215417
CTA_ACTIVE_CAMBER_COEFF_INDICES = (1, 2, 3)
CTA_CAMBER_MODE_BOUNDS = (-0.08, 0.08)
CTA_CAMBER_MODE_STATIONS = ("c0", "c3", "c4", "c5")
CTA_BASE_COEFFICIENTS = {
    "c0": (CTA_C0_UPPER_CST, CTA_C0_LOWER_CST),
    "c3": (CTA_C3_UPPER_CST, CTA_C3_LOWER_CST),
    "c4": (CTA_C4_UPPER_CST, CTA_C4_LOWER_CST),
    "c5": (CTA_C5_UPPER_CST, CTA_C5_LOWER_CST),
}
CTA_CAMBER_MODE_KEYS = tuple(
    (station, coeff_idx)
    for station in CTA_CAMBER_MODE_STATIONS
    for coeff_idx in CTA_ACTIVE_CAMBER_COEFF_INDICES
)
CTA_CAMBER_COEFFICIENT_KEYS = tuple(
    (station, side, coeff_idx)
    for station in CTA_CAMBER_MODE_STATIONS
    for side in ("upper", "lower")
    for coeff_idx in CTA_ACTIVE_CAMBER_COEFF_INDICES
)


# ---------------------------------------------------------------------------
# Variables
# ---------------------------------------------------------------------------
# Internal MADS variables consumed by the actual Section/Wing objects.
chord_c0 = VariableFloat("cta_chord_c0", CTA_CHORD_C0_BASE_M)
chord_c3 = VariableFloat("cta_chord_c3", CTA_CHORD_C3_BASE_M)
chord_c4 = VariableFloat("cta_chord_c4", CTA_CHORD_C4_BASE_M)
chord_c5 = VariableFloat("cta_chord_c5", CTA_CHORD_C5_BASE_M)
camber_coefficient_variables = {
    key: VariableFloat(
        f"cta_{key[0]}_{key[1]}_cst_c{key[2]}",
        CTA_BASE_COEFFICIENTS[key[0]][0 if key[1] == "upper" else 1][key[2]],
    )
    for key in CTA_CAMBER_COEFFICIENT_KEYS
}

# Inner variables: fixed in the current explicit CTA example. They remain regular
# VariableFloat instances so the example still follows the current MADS component
# update mechanism used in the HERA workflows.
cst_n1 = VariableFloat("cta_cst_n1", CTA_CST_N1)
cst_n2 = VariableFloat("cta_cst_n2", CTA_CST_N2)

y_c0 = VariableFloat("cta_y_c0", 0.0)
y_body_helper_01 = VariableFloat("cta_y_body_helper_01", 1.9)
y_body_helper_02 = VariableFloat("cta_y_body_helper_02", 5.694)
y_c3 = VariableFloat("cta_y_c3", 8.041)
y_c4 = VariableFloat("cta_y_c4", 12.5081007083)
y_c5 = VariableFloat("cta_y_c5", 39.4995)

le_x_c0 = VariableFloat("cta_le_x_c0", CTA_LE_X_C0_BASE_M)
le_x_body_helper_01 = VariableFloat("cta_le_x_body_helper_01", CTA_LE_X_BODY_HELPER_01_BASE_M)
le_x_body_helper_02 = VariableFloat("cta_le_x_body_helper_02", CTA_LE_X_BODY_HELPER_02_BASE_M)
le_x_c3 = VariableFloat("cta_le_x_c3", CTA_LE_X_C3_BASE_M)
le_x_c4 = VariableFloat("cta_le_x_c4", CTA_LE_X_C4_BASE_M)
le_x_c5 = VariableFloat("cta_le_x_c5", CTA_LE_X_C5_BASE_M)

le_z_c0 = VariableFloat("cta_le_z_c0", 0.25865)
le_z_body_helper_01 = VariableFloat("cta_le_z_body_helper_01", 1.03474)
le_z_body_helper_02 = VariableFloat("cta_le_z_body_helper_02", 0.48145)
le_z_c3 = VariableFloat("cta_le_z_c3", 0.67693)
le_z_c4 = VariableFloat("cta_le_z_c4", 0.72992)
le_z_c5 = VariableFloat("cta_le_z_c5", 1.97538)

chord_body_helper_01 = VariableFloat("cta_chord_body_helper_01", 39.27452274)
chord_body_helper_02 = VariableFloat("cta_chord_body_helper_02", 30.413344540541)

twist_c0 = VariableFloat("cta_twist_c0", 0.778)
twist_body_helper_01 = VariableFloat("cta_twist_body_helper_01", -0.342)
twist_body_helper_02 = VariableFloat("cta_twist_body_helper_02", 0.371)
twist_c3 = VariableFloat("cta_twist_c3", -0.249)
twist_c4 = VariableFloat("cta_twist_c4", 0.483)
twist_c5 = VariableFloat("cta_twist_c5", 3.177)

te_c0 = VariableFloat("cta_te_c0", 0.0017858758327404453)
te_body_helper_01 = VariableFloat("cta_te_body_helper_01", 0.0018239263113698934)
te_body_helper_02 = VariableFloat("cta_te_body_helper_02", 0.001852260963915209)
te_c3 = VariableFloat("cta_te_c3", 0.0030467512248412917)
te_c4 = VariableFloat("cta_te_c4", 0.002362304664394475)
te_c5 = VariableFloat("cta_te_c5", 0.0021855582271015795)
thickness_factor_c0 = VariableFloat("cta_thickness_factor_c0", 1.0)
thickness_factor_c3 = VariableFloat("cta_thickness_factor_c3", 1.0)
thickness_factor_c4 = VariableFloat("cta_thickness_factor_c4", 1.0)
thickness_factor_c5 = VariableFloat("cta_thickness_factor_c5", 1.0)
thickness_factor_body_helper_01 = VariableFloat("cta_thickness_factor_body_helper_01", 1.0)
thickness_factor_body_helper_02 = VariableFloat("cta_thickness_factor_body_helper_02", 1.0)
med_3_te_sweep_internal = VariableFloat("cta_med_3_te_sweep_deg", CTA_MED_3_TE_SWEEP_BASE_DEG)

inner_variables = [
    cst_n1,
    cst_n2,
    chord_c0,
    chord_c3,
    chord_c4,
    chord_c5,
    y_c0,
    y_body_helper_01,
    y_body_helper_02,
    y_c3,
    y_c4,
    y_c5,
    le_x_c0,
    le_x_body_helper_01,
    le_x_body_helper_02,
    le_x_c3,
    le_x_c4,
    le_x_c5,
    le_z_c0,
    le_z_body_helper_01,
    le_z_body_helper_02,
    le_z_c3,
    le_z_c4,
    le_z_c5,
    chord_body_helper_01,
    chord_body_helper_02,
    twist_c0,
    twist_body_helper_01,
    twist_body_helper_02,
    twist_c3,
    twist_c4,
    twist_c5,
    te_c0,
    te_body_helper_01,
    te_body_helper_02,
    te_c3,
    te_c4,
    te_c5,
    thickness_factor_c0,
    thickness_factor_c3,
    thickness_factor_c4,
    thickness_factor_c5,
    thickness_factor_body_helper_01,
    thickness_factor_body_helper_02,
    med_3_te_sweep_internal,
]

variables = [*inner_variables, *camber_coefficient_variables.values()]


def _cst_coefficients_with_camber_modes(
    station: str,
    side: str,
    baseline_coefficients: tuple[float, ...],
) -> tuple[float | VariableFloat, ...]:
    coefficients: list[float | VariableFloat] = list(baseline_coefficients)
    for coeff_idx in CTA_ACTIVE_CAMBER_COEFF_INDICES:
        coefficients[coeff_idx] = camber_coefficient_variables[(station, side, coeff_idx)]
    return tuple(coefficients)


def _camber_mode_coefficient_mapping(*mode_values):  # noqa: ANN002, ANN201
    modes = {
        key: float(np.ravel(value)[0])
        for key, value in zip(CTA_CAMBER_MODE_KEYS, mode_values, strict=True)
    }
    outputs = []
    for station, side, coeff_idx in CTA_CAMBER_COEFFICIENT_KEYS:
        upper_base, lower_base = CTA_BASE_COEFFICIENTS[station]
        scale = 0.5 * (upper_base[coeff_idx] + lower_base[coeff_idx])
        delta = modes[(station, coeff_idx)] * scale
        if side == "upper":
            outputs.append(upper_base[coeff_idx] + delta)
        else:
            outputs.append(lower_base[coeff_idx] - delta)
    return tuple(outputs)


map_cta_camber_modes_to_coefficients = _camber_mode_coefficient_mapping


# ---------------------------------------------------------------------------
# Components
# ---------------------------------------------------------------------------
section_c0 = Section(
    name="cta_c0",
    airfoil=AirfoilCST(
        name="cta_c0_airfoil",
        upper_coefficients=_cst_coefficients_with_camber_modes("c0", "upper", CTA_C0_UPPER_CST),
        lower_coefficients=_cst_coefficients_with_camber_modes("c0", "lower", CTA_C0_LOWER_CST),
        n1=cst_n1,
        n2=cst_n2,
        trailing_edge_thickness=te_c0,
        thickness_factor=thickness_factor_c0,
    ),
    chord=chord_c0,
    twist=twist_c0,
    spanwise_y_m=y_c0,
    leading_edge_x_m=le_x_c0,
    leading_edge_z_m=le_z_c0,
    metadata={"cta_label": "C0"},
)
section_body_helper_01 = Section(
    name="cta_body_helper_01",
    airfoil=AirfoilCST(
        name="cta_body_helper_01_airfoil",
        upper_coefficients=CTA_BODY_HELPER_01_UPPER_CST,
        lower_coefficients=CTA_BODY_HELPER_01_LOWER_CST,
        n1=cst_n1,
        n2=cst_n2,
        trailing_edge_thickness=te_body_helper_01,
        thickness_factor=thickness_factor_body_helper_01,
    ),
    chord=chord_body_helper_01,
    twist=twist_body_helper_01,
    spanwise_y_m=y_body_helper_01,
    leading_edge_x_m=le_x_body_helper_01,
    leading_edge_z_m=le_z_body_helper_01,
    metadata={"cta_label": "BodyHelper01"},
)
section_body_helper_02 = Section(
    name="cta_body_helper_02",
    airfoil=AirfoilCST(
        name="cta_body_helper_02_airfoil",
        upper_coefficients=CTA_BODY_HELPER_02_UPPER_CST,
        lower_coefficients=CTA_BODY_HELPER_02_LOWER_CST,
        n1=cst_n1,
        n2=cst_n2,
        trailing_edge_thickness=te_body_helper_02,
        thickness_factor=thickness_factor_body_helper_02,
    ),
    chord=chord_body_helper_02,
    twist=twist_body_helper_02,
    spanwise_y_m=y_body_helper_02,
    leading_edge_x_m=le_x_body_helper_02,
    leading_edge_z_m=le_z_body_helper_02,
    metadata={"cta_label": "BodyHelper02"},
)
section_c3 = Section(
    name="cta_c3",
    airfoil=AirfoilCST(
        name="cta_c3_airfoil",
        upper_coefficients=_cst_coefficients_with_camber_modes("c3", "upper", CTA_C3_UPPER_CST),
        lower_coefficients=_cst_coefficients_with_camber_modes("c3", "lower", CTA_C3_LOWER_CST),
        n1=cst_n1,
        n2=cst_n2,
        trailing_edge_thickness=te_c3,
        thickness_factor=thickness_factor_c3,
    ),
    chord=chord_c3,
    twist=twist_c3,
    spanwise_y_m=y_c3,
    leading_edge_x_m=le_x_c3,
    leading_edge_z_m=le_z_c3,
    metadata={"cta_label": "C3"},
)
section_c4 = Section(
    name="cta_c4",
    airfoil=AirfoilCST(
        name="cta_c4_airfoil",
        upper_coefficients=_cst_coefficients_with_camber_modes("c4", "upper", CTA_C4_UPPER_CST),
        lower_coefficients=_cst_coefficients_with_camber_modes("c4", "lower", CTA_C4_LOWER_CST),
        n1=cst_n1,
        n2=cst_n2,
        trailing_edge_thickness=te_c4,
        thickness_factor=thickness_factor_c4,
    ),
    chord=chord_c4,
    twist=twist_c4,
    spanwise_y_m=y_c4,
    leading_edge_x_m=le_x_c4,
    leading_edge_z_m=le_z_c4,
    metadata={"cta_label": "C4"},
)
section_c5 = Section(
    name="cta_c5",
    airfoil=AirfoilCST(
        name="cta_c5_airfoil",
        upper_coefficients=_cst_coefficients_with_camber_modes("c5", "upper", CTA_C5_UPPER_CST),
        lower_coefficients=_cst_coefficients_with_camber_modes("c5", "lower", CTA_C5_LOWER_CST),
        n1=cst_n1,
        n2=cst_n2,
        trailing_edge_thickness=te_c5,
        thickness_factor=thickness_factor_c5,
    ),
    chord=chord_c5,
    twist=twist_c5,
    spanwise_y_m=y_c5,
    leading_edge_x_m=le_x_c5,
    leading_edge_z_m=le_z_c5,
    metadata={"cta_label": "C5"},
)

span_00 = Span(
    name="cta_span_00",
    length=float(y_body_helper_01.value - y_c0.value),
    sweep=degrees(atan2(float(le_x_body_helper_01.value - le_x_c0.value), float(y_body_helper_01.value - y_c0.value))),
    dihed=degrees(atan2(float(le_z_body_helper_01.value - le_z_c0.value), float(y_body_helper_01.value - y_c0.value))),
    start_y_m=y_c0,
    end_y_m=y_body_helper_01,
    metadata={"leading_edge_mode": "section_positions"},
)
span_01 = Span(
    name="cta_span_01",
    length=float(y_body_helper_02.value - y_body_helper_01.value),
    sweep=degrees(
        atan2(
            float(le_x_body_helper_02.value - le_x_body_helper_01.value),
            float(y_body_helper_02.value - y_body_helper_01.value),
        )
    ),
    dihed=degrees(
        atan2(
            float(le_z_body_helper_02.value - le_z_body_helper_01.value),
            float(y_body_helper_02.value - y_body_helper_01.value),
        )
    ),
    start_y_m=y_body_helper_01,
    end_y_m=y_body_helper_02,
    metadata={"leading_edge_mode": "section_positions"},
)
span_02 = Span(
    name="cta_span_02",
    length=float(y_c3.value - y_body_helper_02.value),
    sweep=degrees(atan2(float(le_x_c3.value - le_x_body_helper_02.value), float(y_c3.value - y_body_helper_02.value))),
    dihed=degrees(atan2(float(le_z_c3.value - le_z_body_helper_02.value), float(y_c3.value - y_body_helper_02.value))),
    start_y_m=y_body_helper_02,
    end_y_m=y_c3,
    metadata={"leading_edge_mode": "section_positions"},
)
span_03 = Span(
    name="cta_span_03",
    length=float(y_c4.value - y_c3.value),
    sweep=degrees(atan2(float(le_x_c4.value - le_x_c3.value), float(y_c4.value - y_c3.value))),
    dihed=degrees(atan2(float(le_z_c4.value - le_z_c3.value), float(y_c4.value - y_c3.value))),
    start_y_m=y_c3,
    end_y_m=y_c4,
    metadata={"leading_edge_mode": "section_positions"},
)
span_04 = Span(
    name="cta_span_04",
    length=float(y_c5.value - y_c4.value),
    sweep=degrees(atan2(float(le_x_c5.value - le_x_c4.value), float(y_c5.value - y_c4.value))),
    dihed=degrees(atan2(float(le_z_c5.value - le_z_c4.value), float(y_c5.value - y_c4.value))),
    start_y_m=y_c4,
    end_y_m=y_c5,
    metadata={"leading_edge_mode": "section_positions"},
)

wing = Wing(
    name="cta_wing",
    sections=[
        section_c0,
        section_body_helper_01,
        section_body_helper_02,
        section_c3,
        section_c4,
        section_c5,
    ],
    spans=[span_00, span_01, span_02, span_03, span_04],
    case_name="CTA",
    symmetry=True,
    mirror=False,
    metadata={
        "interpolation": copy.deepcopy(CTA_INTERPOLATION),
        "sampling": copy.deepcopy(CTA_SAMPLING),
    },
)
wing.cta_med_3_te_sweep_deg = med_3_te_sweep_internal


# ---------------------------------------------------------------------------
# Disciplines
# ---------------------------------------------------------------------------
disc_geometry = Geometry(
    name="Geometry",
    components=[wing],
    solver=CSTGeometrySolver(),
)


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
    output_dir = repo_root / "outputs" / "cta_geometry_export"
    profiles_dir = output_dir / "station_airfoils"

    resolved_wing.metadata = copy.deepcopy(resolved_wing.metadata)
    sampling = dict(resolved_wing.metadata.get("sampling", {}))
    sampling["airfoil_distribution_mode"] = "all"
    resolved_wing.metadata["sampling"] = sampling
    resolved_wing.metadata["export"] = {
        "out_dir": str(profiles_dir),
        "iges_path": str(output_dir / "cta.igs"),
        "meshing_iges_path": str(output_dir / "cta_meshing_xy_frame.igs"),
        "frame_only_iges_path": str(output_dir / "cta_xy_frame_only.igs"),
        "blunt_trailing_edge": True,
        "symmetric": False,
        "tip_style": "rounded",
        "include_xy_symmetry_frame": True,
        "write_frame_only": True,
        "section_curve_n_ctl": 18,
        "k_span": 4,
        "metadata": {
            "xy_frame": {
                "x_min_m": -400.0,
                "x_max_m": 400.0,
                "y_min_m": -400.0,
                "y_max_m": 400.0,
            }
        },
    }

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
    print(f"IGES exported: {export_state.iges_path}")
    print(f"Meshing IGES exported: {export_state.meshing_iges_path}")
    print(f"Frame-only IGES exported: {export_state.frame_only_iges_path}")
    print(f"Airfoil .dat written to: {export_state.profiles_dir}/")


if __name__ == "__main__":
    main()
