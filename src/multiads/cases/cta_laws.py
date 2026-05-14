"""CTA-specific interpolation laws implemented natively in MADS."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from scipy.interpolate import PchipInterpolator

from multiads.solvers.synthesis.geometry import (
    ResolvedStation,
    WingGeometryConfig,
    build_control_point_planform,
    resolve_anchor_section,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    from multiads.assembly import Section, Wing


CTA_LINEAR_START_INDEX = 4
CTA_LOWER_FRONT_XC_SIGMA = 0.22
CTA_NOSE_HELPER_1_X_M = 5.418099
CTA_NOSE_HELPER_1_Y_M = 1.9
CTA_C1_LE_HELPER_X_M = 14.300099
CTA_C1_Y_M = 5.694
CTA_MED_3_TE_HELPER_FRACTION = 0.78
CTA_MED_3_TE_SWEEP_DEG = 0.0
CTA_NUM_BASE_STATIONS = 101
CTA_NUM_ROOT_BLEND_STATIONS = 25
CTA_PLANFORM_CONTINUITY_ORDER = 2
CTA_PLANFORM_BLEND_FRACTION = 0.24
CTA_PLANFORM_MIN_LINEAR_CORE_FRACTION = 0.58
CTA_PLANFORM_TE_BLEND_FRACTION = 0.44
CTA_PLANFORM_TE_MIN_LINEAR_CORE_FRACTION = 0.08
CTA_PLANFORM_LE_LINEAR_START_INDEX = 4
CTA_PLANFORM_SYMMETRY_BLEND_Y = 1.9
CTA_PLANFORM_SECTION_Y = np.asarray([0.0, 8.041, 12.5081007083, 39.4995], dtype=float)
CTA_PLANFORM_SECTION_LE_X = np.asarray(
    [3.513099, 19.299046611133836, 25.459944426354326, 39.61671597215417],
    dtype=float,
)
CTA_PLANFORM_SECTION_CHORD = np.asarray(
    [41.17952274, 13.9269627, 7.76845979406, 0.8],
    dtype=float,
)
CTA_LOWER_FRONT_TARGET_Y = np.asarray(
    [0.0, 0.95, 1.9, 5.694, 8.041, 12.5081007083],
    dtype=float,
)
CTA_LOWER_FRONT_TARGET_Z_M = np.asarray(
    [
        -2.368597705825217,
        -2.3450000000000000,
        -2.313970660484542,
        -1.1961949471727737,
        -0.27848207538035497,
        0.41810949398542224,
    ],
    dtype=float,
)
CTA_LOWER_FRONT_XC = np.asarray(
    [
        0.42825368900441024,
        0.38965128248924935,
        0.38965128248924935,
        0.40245483899193585,
        0.3454915028125263,
        0.37692335348550343,
    ],
    dtype=float,
)


def cosine_spacing(n: int) -> np.ndarray:
    beta = np.linspace(0.0, np.pi, int(n))
    return 0.5 * (1.0 - np.cos(beta))


def build_scalar_interpolant(
    y_sections: np.ndarray,
    values: np.ndarray,
    interpolation: str,
    linear_start_index: int | None = None,
) -> Callable[[float], float]:
    y_sections = np.asarray(y_sections, dtype=float)
    values = np.asarray(values, dtype=float)

    if linear_start_index is not None and values.size > 2:
        linear_start_index = int(linear_start_index)
        if not (1 <= linear_start_index < values.size - 1):
            msg = (
                "linear_start_index must lie inside [1, point_count - 2], "
                f"got {linear_start_index} for {values.size} points"
            )
            raise ValueError(msg)
        prefix_interpolant = build_scalar_interpolant(
            y_sections[: linear_start_index + 1],
            values[: linear_start_index + 1],
            interpolation,
            linear_start_index=None,
        )
        y_linear = y_sections[linear_start_index:]
        values_linear = values[linear_start_index:]

        def hybrid_interpolant(yy: float) -> float:
            y_val = float(yy)
            if y_val <= float(y_sections[linear_start_index]):
                return float(prefix_interpolant(y_val))
            return float(np.interp(y_val, y_linear, values_linear))

        return hybrid_interpolant

    interpolation_name = interpolation.lower()
    if interpolation_name == "linear" or values.size == 2:
        return lambda yy: float(np.interp(float(yy), y_sections, values))
    if interpolation_name == "pchip":
        interpolant = PchipInterpolator(y_sections, values)
        return lambda yy: float(interpolant(float(yy)))

    msg = f"Unsupported CTA interpolation '{interpolation}'."
    raise ValueError(msg)


def _build_profile_interpolants(
    anchor_y: np.ndarray,
    anchor_profiles: np.ndarray,
    *,
    interpolation: str,
    linear_start_index: int | None,
) -> list[Callable[[float], float]]:
    return [
        build_scalar_interpolant(
            anchor_y,
            anchor_profiles[:, idx],
            interpolation,
            linear_start_index=linear_start_index,
        )
        for idx in range(anchor_profiles.shape[1])
    ]


def _evaluate_profile_interpolants(
    interpolants: list[Callable[[float], float]],
    y_value: float,
) -> np.ndarray:
    return np.asarray([interpolant(float(y_value)) for interpolant in interpolants], dtype=float)


@dataclass(frozen=True, slots=True)
class CTAResolvedLawSet:
    x_over_c: np.ndarray
    planform: object
    chord_fun: Callable[[float], float]
    base_twist_fun: Callable[[float], float]
    leading_edge_x_fun: Callable[[float], float]
    base_leading_edge_z_fun: Callable[[float], float]
    upper_profile_interpolants: list[Callable[[float], float]]
    lower_profile_interpolants: list[Callable[[float], float]]


def _build_surface_xyz(
    x_over_c: np.ndarray,
    z_over_c: np.ndarray,
    chord_m: float,
    spanwise_y_m: float,
    twist_deg: float,
    leading_edge_x_m: float,
    leading_edge_z_m: float,
) -> np.ndarray:
    x_local = chord_m * np.asarray(x_over_c, dtype=float)
    z_local = chord_m * np.asarray(z_over_c, dtype=float)
    theta = np.radians(twist_deg)
    c = np.cos(theta)
    s = np.sin(theta)

    xyz = np.zeros((x_local.size, 3), dtype=float)
    xyz[:, 0] = leading_edge_x_m + x_local * c - z_local * s
    xyz[:, 1] = spanwise_y_m
    xyz[:, 2] = leading_edge_z_m + x_local * s + z_local * c
    return xyz


def _lower_min_z_world(
    *,
    x_over_c: np.ndarray,
    lower_z_over_c: np.ndarray,
    chord_m: float,
    twist_deg: float,
    leading_edge_z_m: float,
) -> float:
    theta = np.radians(float(twist_deg))
    x_local = np.asarray(x_over_c, dtype=float) * float(chord_m)
    lower_local = np.asarray(lower_z_over_c, dtype=float) * float(chord_m)
    z_world = (
        float(leading_edge_z_m)
        + x_local * np.sin(theta)
        + lower_local * np.cos(theta)
    )
    return float(np.min(z_world))


def _apply_lower_front_guidance(
    *,
    y_value: float,
    x_over_c: np.ndarray,
    lower_z_over_c: np.ndarray,
    chord_m: float,
    twist_deg: float,
    leading_edge_z_m: float,
    target_lower_fun: Callable[[float], float],
    x_c_min_fun: Callable[[float], float],
) -> np.ndarray:
    if y_value < CTA_LOWER_FRONT_TARGET_Y[0] - 1.0e-12 or y_value > CTA_LOWER_FRONT_TARGET_Y[-1] + 1.0e-12:
        return lower_z_over_c
    theta = np.radians(float(twist_deg))
    cos_theta = float(np.cos(theta))
    if abs(chord_m * cos_theta) <= 1.0e-12:
        return lower_z_over_c

    x_air_arr = np.asarray(x_over_c, dtype=float)
    lower_arr = np.asarray(lower_z_over_c, dtype=float)
    current_lower = _lower_min_z_world(
        x_over_c=x_air_arr,
        lower_z_over_c=lower_arr,
        chord_m=chord_m,
        twist_deg=twist_deg,
        leading_edge_z_m=leading_edge_z_m,
    )
    target_lower = float(target_lower_fun(float(y_value)))
    delta_local = float((target_lower - current_lower) / (chord_m * cos_theta))
    if abs(delta_local) <= 1.0e-12:
        return lower_arr

    x_c_center = float(np.clip(x_c_min_fun(float(y_value)), 0.0, 1.0))
    sigma = float(max(CTA_LOWER_FRONT_XC_SIGMA, 1.0e-3))
    weight = np.exp(-0.5 * ((x_air_arr - x_c_center) / sigma) ** 2)
    return lower_arr + delta_local * weight


def _build_cta_law_set(
    component: Wing,
    anchor_sections: tuple[Section, ...],
    config: WingGeometryConfig,
) -> CTAResolvedLawSet:
    anchor_stations = tuple(resolve_anchor_section(section, config) for section in anchor_sections)
    anchor_y = np.asarray([station.spanwise_y_m for station in anchor_stations], dtype=float)
    x_over_c = np.asarray(anchor_stations[0].x_over_c, dtype=float)
    section_y, section_le_x, section_chord = _cta_planform_arrays_from_sections(anchor_sections)
    planform = build_cta_planform(
        section_y=section_y,
        section_le_x=section_le_x,
        section_chord=section_chord,
        med_3_te_sweep_deg=float(
            getattr(component, "cta_med_3_te_sweep_deg", CTA_MED_3_TE_SWEEP_DEG)
        ),
    )
    leading_edge_x_fun = lambda yy: float(planform.le_x(float(yy)))
    chord_fun = lambda yy: float(planform.te_x(float(yy)) - planform.le_x(float(yy)))

    base_twist_fun = build_scalar_interpolant(
        anchor_y,
        np.asarray([station.twist_deg for station in anchor_stations], dtype=float),
        "pchip",
        linear_start_index=CTA_LINEAR_START_INDEX,
    )

    base_leading_edge_z_fun = build_scalar_interpolant(
        anchor_y,
        np.asarray([station.leading_edge_z_m for station in anchor_stations], dtype=float),
        "pchip",
        linear_start_index=CTA_LINEAR_START_INDEX,
    )

    anchor_upper = np.asarray([station.upper_z_over_c for station in anchor_stations], dtype=float)
    anchor_lower = np.asarray([station.lower_z_over_c for station in anchor_stations], dtype=float)
    upper_profile_interpolants = _build_profile_interpolants(
        anchor_y,
        anchor_upper,
        interpolation="pchip",
        linear_start_index=CTA_LINEAR_START_INDEX,
    )
    lower_profile_interpolants = _build_profile_interpolants(
        anchor_y,
        anchor_lower,
        interpolation="pchip",
        linear_start_index=CTA_LINEAR_START_INDEX,
    )

    return CTAResolvedLawSet(
        x_over_c=x_over_c,
        planform=planform,
        chord_fun=chord_fun,
        base_twist_fun=base_twist_fun,
        leading_edge_x_fun=leading_edge_x_fun,
        base_leading_edge_z_fun=base_leading_edge_z_fun,
        upper_profile_interpolants=upper_profile_interpolants,
        lower_profile_interpolants=lower_profile_interpolants,
    )


def _cta_planform_arrays_from_sections(
    anchor_sections: tuple[Section, ...],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    section_by_label = {
        str(section.metadata.get("cta_label")): section
        for section in anchor_sections
        if "cta_label" in section.metadata
    }
    required_labels = ("C0", "C3", "C4", "C5")
    if not all(label in section_by_label for label in required_labels):
        return (
            CTA_PLANFORM_SECTION_Y.copy(),
            CTA_PLANFORM_SECTION_LE_X.copy(),
            CTA_PLANFORM_SECTION_CHORD.copy(),
        )
    return (
        np.asarray([float(section_by_label[label].spanwise_y_m) for label in required_labels], dtype=float),
        np.asarray([float(section_by_label[label].leading_edge_x_m) for label in required_labels], dtype=float),
        np.asarray([float(section_by_label[label].chord) for label in required_labels], dtype=float),
    )


def build_cta_planform(
    *,
    section_y: np.ndarray | None = None,
    section_le_x: np.ndarray | None = None,
    section_chord: np.ndarray | None = None,
    med_3_te_sweep_deg: float | None = None,
):
    section_y = CTA_PLANFORM_SECTION_Y if section_y is None else np.asarray(section_y, dtype=float)
    section_le_x = CTA_PLANFORM_SECTION_LE_X if section_le_x is None else np.asarray(section_le_x, dtype=float)
    section_chord = CTA_PLANFORM_SECTION_CHORD if section_chord is None else np.asarray(section_chord, dtype=float)
    med_3_te_sweep_deg = (
        CTA_MED_3_TE_SWEEP_DEG
        if med_3_te_sweep_deg is None
        else float(med_3_te_sweep_deg)
    )
    leading_edge_points = np.asarray(
        [
            (section_le_x[0], section_y[0]),
            (CTA_NOSE_HELPER_1_X_M, CTA_NOSE_HELPER_1_Y_M),
            (CTA_C1_LE_HELPER_X_M, CTA_C1_Y_M),
            (section_le_x[1], section_y[1]),
            (section_le_x[2], section_y[2]),
            (section_le_x[3], section_y[3]),
        ],
        dtype=float,
    )
    te_root = float(section_le_x[0] + section_chord[0])
    te_c3 = float(section_le_x[1] + section_chord[1])
    te_c4 = float(section_le_x[2] + section_chord[2])
    te_c5 = float(section_le_x[3] + section_chord[3])
    trailing_edge_points = np.asarray(
        [
            (te_root, 0.0),
            (te_root, CTA_C1_Y_M),
            (te_c3, section_y[1]),
            (te_c4, section_y[2]),
            (te_c5, section_y[3]),
        ],
        dtype=float,
    )
    if abs(med_3_te_sweep_deg) > 1.0e-12:
        y_med3 = float(
            section_y[1]
            + CTA_MED_3_TE_HELPER_FRACTION * (section_y[2] - section_y[1])
        )
        te_med3 = float(
            te_c3 + np.tan(np.radians(med_3_te_sweep_deg)) * (y_med3 - section_y[1])
        )
        trailing_edge_points = np.insert(
            trailing_edge_points,
            4,
            np.asarray([[te_med3, y_med3]], dtype=float),
            axis=0,
        )
    te_linear_start_index = 4 if abs(med_3_te_sweep_deg) > 1.0e-12 else 3
    te_exact_segments = (0, te_linear_start_index)

    return build_control_point_planform(
        leading_edge_points=leading_edge_points,
        trailing_edge_points=trailing_edge_points,
        root_le_x=float(section_le_x[0]),
        root_te_x=te_root,
        continuity_order=CTA_PLANFORM_CONTINUITY_ORDER,
        blend_fraction=CTA_PLANFORM_BLEND_FRACTION,
        min_linear_core_fraction=CTA_PLANFORM_MIN_LINEAR_CORE_FRACTION,
        te_blend_fraction=CTA_PLANFORM_TE_BLEND_FRACTION,
        te_min_linear_core_fraction=CTA_PLANFORM_TE_MIN_LINEAR_CORE_FRACTION,
        le_linear_start_index=CTA_PLANFORM_LE_LINEAR_START_INDEX,
        te_linear_start_index=te_linear_start_index,
        le_exact_segments=(),
        te_exact_segments=te_exact_segments,
        le_spline_bridge=None,
        te_spline_bridge=None,
        symmetry_blend_y=CTA_PLANFORM_SYMMETRY_BLEND_Y,
        body_le_fixed_points=((CTA_NOSE_HELPER_1_X_M, CTA_NOSE_HELPER_1_Y_M), (CTA_C1_LE_HELPER_X_M, CTA_C1_Y_M)),
    )


def build_cta_span_station_array(
    *,
    component: Wing,
    anchor_sections: tuple[Section, ...],
    config: WingGeometryConfig,
) -> np.ndarray:
    del config
    anchor_y = np.asarray([float(section.spanwise_y_m) for section in anchor_sections], dtype=float)
    half_span = float(anchor_y[-1])
    base_stations = np.linspace(0.0, half_span, CTA_NUM_BASE_STATIONS, dtype=float)
    root_blend_stations = np.array([], dtype=float)
    if CTA_PLANFORM_SYMMETRY_BLEND_Y > 1.0e-12:
        root_blend_stations = float(CTA_PLANFORM_SYMMETRY_BLEND_Y) * cosine_spacing(CTA_NUM_ROOT_BLEND_STATIONS)
    section_y, section_le_x, section_chord = _cta_planform_arrays_from_sections(anchor_sections)
    planform = build_cta_planform(
        section_y=section_y,
        section_le_x=section_le_x,
        section_chord=section_chord,
        med_3_te_sweep_deg=float(
            getattr(component, "cta_med_3_te_sweep_deg", CTA_MED_3_TE_SWEEP_DEG)
        ),
    )
    planform_helper_stations = np.unique(
        np.concatenate(
            [
                np.asarray(planform.leading_edge_points[:, 1], dtype=float),
                np.asarray(planform.trailing_edge_points[:, 1], dtype=float),
            ]
        )
    )
    span_stations = np.unique(
        np.round(
            np.concatenate(
                [
                    base_stations,
                    root_blend_stations,
                    anchor_y,
                    planform_helper_stations,
                ]
            ),
            decimals=12,
        )
    )
    return span_stations.astype(float)


def build_cta_resolved_station_factory(
    *,
    component: Wing,
    anchor_sections: tuple[Section, ...],
    config: WingGeometryConfig,
) -> Callable[[np.ndarray], tuple[ResolvedStation, ...]]:
    """Return a station resolver based on the native CTA interpolation laws."""

    law_set = _build_cta_law_set(component, anchor_sections, config)
    anchor_name_map = {
        round(float(section.spanwise_y_m), 10): section.name
        for section in anchor_sections
    }
    root_lower_z_over_c = _evaluate_profile_interpolants(
        law_set.lower_profile_interpolants,
        0.0,
    )
    root_lower_z_m = _lower_min_z_world(
        x_over_c=law_set.x_over_c,
        lower_z_over_c=root_lower_z_over_c,
        chord_m=float(law_set.chord_fun(0.0)),
        twist_deg=float(law_set.base_twist_fun(0.0)),
        leading_edge_z_m=float(law_set.base_leading_edge_z_fun(0.0)),
    )
    lower_target_z_m = root_lower_z_m + (
        CTA_LOWER_FRONT_TARGET_Z_M - CTA_LOWER_FRONT_TARGET_Z_M[0]
    )
    target_lower_fun = build_scalar_interpolant(
        CTA_LOWER_FRONT_TARGET_Y,
        lower_target_z_m,
        "pchip",
    )
    lower_x_c_fun = build_scalar_interpolant(
        CTA_LOWER_FRONT_TARGET_Y,
        CTA_LOWER_FRONT_XC,
        "pchip",
    )

    def resolver(sample_y: np.ndarray) -> tuple[ResolvedStation, ...]:
        stations: list[ResolvedStation] = []
        for idx, y_value in enumerate(np.asarray(sample_y, dtype=float)):
            upper_z_over_c = _evaluate_profile_interpolants(law_set.upper_profile_interpolants, float(y_value))
            lower_z_over_c = _evaluate_profile_interpolants(law_set.lower_profile_interpolants, float(y_value))
            chord_m = float(law_set.chord_fun(float(y_value)))
            twist_deg = float(law_set.base_twist_fun(float(y_value)))
            leading_edge_x_m = float(law_set.leading_edge_x_fun(float(y_value)))
            leading_edge_z_m = float(law_set.base_leading_edge_z_fun(float(y_value)))
            lower_z_over_c = _apply_lower_front_guidance(
                y_value=float(y_value),
                x_over_c=law_set.x_over_c,
                lower_z_over_c=lower_z_over_c,
                chord_m=chord_m,
                twist_deg=twist_deg,
                leading_edge_z_m=leading_edge_z_m,
                target_lower_fun=target_lower_fun,
                x_c_min_fun=lower_x_c_fun,
            )
            upper_surface_xyz_m = _build_surface_xyz(
                law_set.x_over_c,
                upper_z_over_c,
                chord_m,
                float(y_value),
                twist_deg,
                leading_edge_x_m,
                leading_edge_z_m,
            )
            lower_surface_xyz_m = _build_surface_xyz(
                law_set.x_over_c,
                lower_z_over_c,
                chord_m,
                float(y_value),
                twist_deg,
                leading_edge_x_m,
                leading_edge_z_m,
            )
            y_key = round(float(y_value), 10)
            name = anchor_name_map.get(y_key, f"cta_station_{idx:03d}")
            stations.append(
                ResolvedStation(
                    name=name,
                    spanwise_y_m=float(y_value),
                    chord_m=chord_m,
                    twist_deg=twist_deg,
                    leading_edge_x_m=leading_edge_x_m,
                    leading_edge_z_m=leading_edge_z_m,
                    x_over_c=law_set.x_over_c,
                    upper_z_over_c=upper_z_over_c,
                    lower_z_over_c=lower_z_over_c,
                    upper_surface_xyz_m=upper_surface_xyz_m,
                    lower_surface_xyz_m=lower_surface_xyz_m,
                    metadata={"source": "cta_native_laws"},
                )
            )
        return tuple(stations)

    return resolver
