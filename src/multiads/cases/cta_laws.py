"""CTA-specific interpolation laws implemented natively in MADS.

The CTA case is resolved from the official nine-section table:

    S0, S1, S1a, S2, S3, S4, S4a, S4b, S5

This module contains geometry laws only.  It does not know anything about
DUST, launchers, wake models, panels or post-processing.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from scipy.interpolate import PchipInterpolator

from multiads.solvers.synthesis.geometry import (
    ResolvedStation,
    WingGeometryConfig,
    build_control_point_planform,
    quintic_c2_transition,
    resolve_anchor_section,
)
from multiads.utilities.cst import evaluate_cst_surface

if TYPE_CHECKING:
    from collections.abc import Callable

    from multiads.assembly import Section, Wing


CTA_SECTION_ORDER: tuple[str, ...] = ("s0", "s1", "s1a", "s2", "s3", "s4", "s4a", "s4b", "s5")
CTA_OUTER_LINEAR_START_INDEX = CTA_SECTION_ORDER.index("s4")
CTA_NUM_BASE_STATIONS = 41
CTA_NUM_ROOT_BLEND_STATIONS = 25
CTA_PLANFORM_CONTINUITY_ORDER = 2
CTA_PLANFORM_BLEND_FRACTION = 0.24
CTA_PLANFORM_MIN_LINEAR_CORE_FRACTION = 0.58
CTA_PLANFORM_TE_BLEND_FRACTION = 0.24
CTA_PLANFORM_TE_MIN_LINEAR_CORE_FRACTION = 0.58
CTA_PLANFORM_SYMMETRY_BLEND_Y = 1.9
CTA_ROOT_FRONT_BLEND_FRACTION = 0.74

CTA_PLANFORM_SECTION_Y = np.asarray(
    [0.0, 1.9, 3.765, 5.694, 8.041, 12.508106999999999, 21.505238, 30.502368999999998, 39.4995],
    dtype=float,
)
CTA_PLANFORM_SECTION_LE_X = np.asarray(
    [
        3.513099,
        5.418099,
        9.784184925144967,
        14.300099,
        19.299046611133836,
        25.45995701373766,
        30.178881631546307,
        34.89780624935495,
        39.6167308671636,
    ],
    dtype=float,
)
CTA_PLANFORM_SECTION_CHORD = np.asarray(
    [
        41.203,
        39.246,
        34.76931180811808,
        30.139,
        13.927,
        7.768480599999999,
        5.445653733333333,
        3.1228268666666668,
        0.8,
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
    root_blend_y: float | None = None,
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

        interpolant: Callable[[float], float] = hybrid_interpolant
    else:
        interpolation_name = interpolation.lower()
        if interpolation_name == "linear" or values.size == 2:
            interpolant = lambda yy: float(np.interp(float(yy), y_sections, values))
        elif interpolation_name == "pchip":
            pchip = PchipInterpolator(y_sections, values)
            interpolant = lambda yy: float(pchip(float(yy)))
        else:
            msg = f"Unsupported CTA interpolation '{interpolation}'."
            raise ValueError(msg)

    if root_blend_y is None or float(root_blend_y) <= 1.0e-12:
        return interpolant

    blend_y = float(root_blend_y)
    root_value = float(interpolant(0.0))
    join_value = float(interpolant(blend_y))
    eps = max(blend_y * 1.0e-4, 1.0e-6)
    y_left = max(blend_y - eps, 0.0)
    y_right = blend_y + eps
    join_slope = float((interpolant(y_right) - interpolant(y_left)) / max(y_right - y_left, 1.0e-12))

    def root_blended(yy: float) -> float:
        y_val = float(yy)
        if y_val >= blend_y:
            return float(interpolant(y_val))
        return quintic_c2_transition(
            y=y_val,
            y0=0.0,
            y1=blend_y,
            x0=root_value,
            x1=join_value,
            dx0=0.0,
            dx1=join_slope,
        )

    return root_blended


def build_positive_interpolant(
    y_sections: np.ndarray,
    values: np.ndarray,
    interpolation: str,
    linear_start_index: int | None = None,
    floor: float = 1.0e-8,
) -> Callable[[float], float]:
    """Interpolate strictly positive section quantities like the BWB core."""

    positive_values = np.maximum(np.asarray(values, dtype=float), float(floor))
    log_interpolant = build_scalar_interpolant(
        y_sections,
        np.log(positive_values),
        interpolation,
        linear_start_index=linear_start_index,
    )
    return lambda yy: float(np.exp(log_interpolant(float(yy))))


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
    upper_coeff_interpolants: list[Callable[[float], float]]
    lower_coeff_interpolants: list[Callable[[float], float]]
    trailing_edge_thickness_absolute_fun: Callable[[float], float]
    cst_n1: float
    cst_n2: float
    outer_thickness_fun: Callable[[float], float]
    outer_start_y_m: float


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
    )
    leading_edge_x_fun = lambda yy: float(planform.le_x(float(yy)))
    chord_fun = lambda yy: float(planform.te_x(float(yy)) - planform.le_x(float(yy)))
    root_blend_y = float(CTA_ROOT_FRONT_BLEND_FRACTION * anchor_y[1])

    base_twist_fun = build_scalar_interpolant(
        anchor_y,
        np.asarray([station.twist_deg for station in anchor_stations], dtype=float),
        "pchip",
        linear_start_index=CTA_OUTER_LINEAR_START_INDEX,
        root_blend_y=root_blend_y,
    )

    base_leading_edge_z_fun = build_scalar_interpolant(
        anchor_y,
        np.asarray([station.leading_edge_z_m for station in anchor_stations], dtype=float),
        "pchip",
        linear_start_index=CTA_OUTER_LINEAR_START_INDEX,
        root_blend_y=root_blend_y,
    )

    anchor_upper_coeffs = np.asarray([section.airfoil.upper_coefficients for section in anchor_sections], dtype=float)
    anchor_lower_coeffs = np.asarray([section.airfoil.lower_coefficients for section in anchor_sections], dtype=float)
    upper_coeff_interpolants = _build_profile_interpolants(
        anchor_y,
        anchor_upper_coeffs,
        interpolation="pchip",
        linear_start_index=CTA_OUTER_LINEAR_START_INDEX,
    )
    lower_coeff_interpolants = _build_profile_interpolants(
        anchor_y,
        anchor_lower_coeffs,
        interpolation="pchip",
        linear_start_index=CTA_OUTER_LINEAR_START_INDEX,
    )
    anchor_chords = np.asarray([float(section.chord) for section in anchor_sections], dtype=float)
    anchor_trailing_edge_thickness_m = anchor_chords * np.asarray(
        [section.airfoil.trailing_edge_thickness for section in anchor_sections],
        dtype=float,
    )
    trailing_edge_thickness_absolute_fun = build_positive_interpolant(
        anchor_y,
        anchor_trailing_edge_thickness_m,
        "pchip",
        linear_start_index=CTA_OUTER_LINEAR_START_INDEX,
    )
    outer_thickness_fun = _outer_thickness_target_fun(component, anchor_y)

    return CTAResolvedLawSet(
        x_over_c=x_over_c,
        planform=planform,
        chord_fun=chord_fun,
        base_twist_fun=base_twist_fun,
        leading_edge_x_fun=leading_edge_x_fun,
        base_leading_edge_z_fun=base_leading_edge_z_fun,
        upper_coeff_interpolants=upper_coeff_interpolants,
        lower_coeff_interpolants=lower_coeff_interpolants,
        trailing_edge_thickness_absolute_fun=trailing_edge_thickness_absolute_fun,
        cst_n1=float(anchor_sections[0].airfoil.n1),
        cst_n2=float(anchor_sections[0].airfoil.n2),
        outer_thickness_fun=outer_thickness_fun,
        outer_start_y_m=float(anchor_y[CTA_OUTER_LINEAR_START_INDEX]),
    )


def _section_id(section: Section) -> str:
    raw = section.metadata.get("cta_section_id", section.metadata.get("cta_label", section.name))
    return str(raw).lower().replace("cta_", "")


def _cta_planform_arrays_from_sections(
    anchor_sections: tuple[Section, ...],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    section_by_id = {_section_id(section): section for section in anchor_sections}
    if not all(section_id in section_by_id for section_id in CTA_SECTION_ORDER):
        return (
            CTA_PLANFORM_SECTION_Y.copy(),
            CTA_PLANFORM_SECTION_LE_X.copy(),
            CTA_PLANFORM_SECTION_CHORD.copy(),
        )
    return (
        np.asarray([float(section_by_id[name].spanwise_y_m) for name in CTA_SECTION_ORDER], dtype=float),
        np.asarray([float(section_by_id[name].leading_edge_x_m) for name in CTA_SECTION_ORDER], dtype=float),
        np.asarray([float(section_by_id[name].chord) for name in CTA_SECTION_ORDER], dtype=float),
    )


def build_cta_planform(
    *,
    section_y: np.ndarray | None = None,
    section_le_x: np.ndarray | None = None,
    section_chord: np.ndarray | None = None,
):
    section_y = CTA_PLANFORM_SECTION_Y if section_y is None else np.asarray(section_y, dtype=float)
    section_le_x = CTA_PLANFORM_SECTION_LE_X if section_le_x is None else np.asarray(section_le_x, dtype=float)
    section_chord = CTA_PLANFORM_SECTION_CHORD if section_chord is None else np.asarray(section_chord, dtype=float)
    leading_edge_points = np.column_stack((section_le_x, section_y))
    trailing_edge_points = np.column_stack((section_le_x + section_chord, section_y))

    return build_control_point_planform(
        leading_edge_points=leading_edge_points,
        trailing_edge_points=trailing_edge_points,
        root_le_x=float(section_le_x[0]),
        root_te_x=float(section_le_x[0] + section_chord[0]),
        continuity_order=CTA_PLANFORM_CONTINUITY_ORDER,
        blend_fraction=CTA_PLANFORM_BLEND_FRACTION,
        min_linear_core_fraction=CTA_PLANFORM_MIN_LINEAR_CORE_FRACTION,
        te_blend_fraction=CTA_PLANFORM_TE_BLEND_FRACTION,
        te_min_linear_core_fraction=CTA_PLANFORM_TE_MIN_LINEAR_CORE_FRACTION,
        le_linear_start_index=CTA_OUTER_LINEAR_START_INDEX,
        te_linear_start_index=CTA_OUTER_LINEAR_START_INDEX,
        le_exact_segments=(),
        te_exact_segments=(CTA_SECTION_ORDER.index("s3"),),
        le_spline_bridge=(1, len(CTA_SECTION_ORDER) - 2),
        te_spline_bridge=None,
        symmetry_blend_y=CTA_PLANFORM_SYMMETRY_BLEND_Y,
        body_le_fixed_points=(),
    )


def _outer_thickness_target_fun(
    component: Wing,
    anchor_y: np.ndarray,
) -> Callable[[float], float]:
    y_outer = np.asarray(
        [
            anchor_y[CTA_SECTION_ORDER.index("s4")],
            anchor_y[CTA_SECTION_ORDER.index("s4a")],
            anchor_y[CTA_SECTION_ORDER.index("s4b")],
            anchor_y[CTA_SECTION_ORDER.index("s5")],
        ],
        dtype=float,
    )
    thickness_s4 = float(getattr(component, "cta_thickness_s4", 0.1027))
    thickness_s5 = float(getattr(component, "cta_thickness_s5", 0.095))
    values = np.asarray(
        [
            thickness_s4,
            (2.0 * thickness_s4 + thickness_s5) / 3.0,
            (thickness_s4 + 2.0 * thickness_s5) / 3.0,
            thickness_s5,
        ],
        dtype=float,
    )
    return lambda yy: float(np.interp(float(yy), y_outer, values))


def _apply_outer_thickness_target(
    *,
    y_value: float,
    x_over_c: np.ndarray,
    upper_z_over_c: np.ndarray,
    lower_z_over_c: np.ndarray,
    law_set: CTAResolvedLawSet,
) -> tuple[np.ndarray, np.ndarray]:
    if float(y_value) < law_set.outer_start_y_m - 1.0e-12:
        return upper_z_over_c, lower_z_over_c

    x_arr = np.asarray(x_over_c, dtype=float)
    upper_arr = np.asarray(upper_z_over_c, dtype=float)
    lower_arr = np.asarray(lower_z_over_c, dtype=float)
    camber = 0.5 * (upper_arr + lower_arr)
    thickness = upper_arr - lower_arr
    te_thickness = float(thickness[-1])
    shape_thickness = thickness - x_arr * te_thickness

    target_tc = float(law_set.outer_thickness_fun(float(y_value)))

    def max_tc(scale: float) -> float:
        candidate = scale * shape_thickness + x_arr * te_thickness
        return float(np.max(candidate))

    current_tc = max_tc(1.0)
    if current_tc <= 1.0e-12 or abs(current_tc - target_tc) <= 1.0e-10:
        return upper_arr, lower_arr

    low = 0.0
    high = max(2.0, 2.0 * target_tc / max(current_tc, 1.0e-12))
    for _ in range(32):
        if max_tc(high) >= target_tc:
            break
        high *= 2.0
    for _ in range(60):
        mid = 0.5 * (low + high)
        if max_tc(mid) < target_tc:
            low = mid
        else:
            high = mid

    scale = 0.5 * (low + high)
    new_thickness = scale * shape_thickness + x_arr * te_thickness
    new_thickness = np.maximum(new_thickness, 0.0)
    return camber + 0.5 * new_thickness, camber - 0.5 * new_thickness


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
            decimals=9,
        )
    )
    return span_stations.astype(float)


def build_cta_resolved_station_factory(
    *,
    component: Wing,
    anchor_sections: tuple[Section, ...],
    config: WingGeometryConfig,
) -> Callable[[np.ndarray], tuple[ResolvedStation, ...]]:
    """Return a station resolver based on the official CTA interpolation laws."""

    law_set = _build_cta_law_set(component, anchor_sections, config)
    anchor_name_map = {
        round(float(section.spanwise_y_m), 10): section.name
        for section in anchor_sections
    }

    def resolver(sample_y: np.ndarray) -> tuple[ResolvedStation, ...]:
        stations: list[ResolvedStation] = []
        for idx, y_value in enumerate(np.asarray(sample_y, dtype=float)):
            chord_m = float(law_set.chord_fun(float(y_value)))
            upper_coeffs = tuple(_evaluate_profile_interpolants(law_set.upper_coeff_interpolants, float(y_value)))
            lower_coeffs = tuple(_evaluate_profile_interpolants(law_set.lower_coeff_interpolants, float(y_value)))
            te_thickness_m = float(law_set.trailing_edge_thickness_absolute_fun(float(y_value)))
            te_thickness = te_thickness_m / max(chord_m, 1.0e-12)
            upper_z_over_c = evaluate_cst_surface(
                law_set.x_over_c,
                upper_coeffs,
                n1=law_set.cst_n1,
                n2=law_set.cst_n2,
                trailing_edge_thickness_over_c=te_thickness,
                sign=1.0,
            )
            lower_z_over_c = evaluate_cst_surface(
                law_set.x_over_c,
                lower_coeffs,
                n1=law_set.cst_n1,
                n2=law_set.cst_n2,
                trailing_edge_thickness_over_c=te_thickness,
                sign=-1.0,
            )
            upper_z_over_c, lower_z_over_c = _apply_outer_thickness_target(
                y_value=float(y_value),
                x_over_c=law_set.x_over_c,
                upper_z_over_c=upper_z_over_c,
                lower_z_over_c=lower_z_over_c,
                law_set=law_set,
            )
            twist_deg = float(law_set.base_twist_fun(float(y_value)))
            leading_edge_x_m = float(law_set.leading_edge_x_fun(float(y_value)))
            leading_edge_z_m = float(law_set.base_leading_edge_z_fun(float(y_value)))
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
                    metadata={
                        "source": "cta_official_nine_section_laws",
                        "pygeo_te_height_scaled": float(upper_z_over_c[-1] - lower_z_over_c[-1]),
                        "trailing_edge_thickness_m": te_thickness_m,
                    },
                )
            )
        return tuple(stations)

    return resolver
