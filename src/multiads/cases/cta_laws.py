"""CTA-specific interpolation laws implemented natively in MADS."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from scipy.interpolate import PchipInterpolator
from scipy.optimize import brentq

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
CTA_ROOT_FRONT_BLEND_FRACTION = 0.74
CTA_ROOT_TE_FLATTEN_X_START = 0.92
CTA_LOWER_FRONT_XC_SIGMA = 0.22
CTA_UPPER_FRONT_XC_SIGMA = 0.24
CTA_UPPER_FRONT_LE_XC_SIGMA = 0.10
CTA_NOSE_HELPER_1_X_M = 5.418099
CTA_NOSE_HELPER_1_Y_M = 1.9
CTA_C1_LE_HELPER_X_M = 14.300099
CTA_C1_Y_M = 5.694
CTA_TE_INBOARD_BLEND_DX_M = 6.7
CTA_TE_INBOARD_BLEND_Y_M = 7.5
CTA_TE_INBOARD_RADIUS_FACTOR = 1.5
CTA_MED_3_TE_HELPER_FRACTION = 0.78
CTA_MED_3_TE_SWEEP_DEG = 0.0
CTA_NUM_BASE_STATIONS = 41
CTA_NUM_ROOT_BLEND_STATIONS = 25
CTA_PLANFORM_CONTINUITY_ORDER = 2
CTA_PLANFORM_BLEND_FRACTION = 0.24
CTA_PLANFORM_MIN_LINEAR_CORE_FRACTION = 0.58
CTA_PLANFORM_TE_BLEND_FRACTION = 0.44
CTA_PLANFORM_TE_MIN_LINEAR_CORE_FRACTION = 0.08
CTA_PLANFORM_LE_LINEAR_START_INDEX = 4
CTA_PLANFORM_TE_LINEAR_START_INDEX = 4
CTA_PLANFORM_TE_EXACT_SEGMENTS = (0, 4)
CTA_PLANFORM_TE_SPLINE_BRIDGE = (1, 3)
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

CTA_ROOT_FRONT_TARGET_Y = np.asarray([0.0, 0.95, 1.9], dtype=float)
CTA_ROOT_FRONT_TARGET_UPPER_Z_M = np.asarray(
    [3.464835489580566, 3.460764218235223, 3.37393430087873],
    dtype=float,
)
CTA_ROOT_FRONT_TARGET_LOWER_Z_M = np.asarray(
    [-2.368597705825217, -2.4001434250887357, -2.313970660484542],
    dtype=float,
)
CTA_ROOT_TE_CENTER_TARGET_Y = np.asarray([0.0, 0.95, 1.9], dtype=float)
CTA_ROOT_TE_CENTER_TARGET_Z_M = np.asarray(
    [0.8177956204143926, 0.8049039758261936, 0.8003107430812731],
    dtype=float,
)
CTA_LOWER_FRONT_TARGET_Y = np.asarray(
    [0.0, 0.95, 1.9, 5.694, 8.041, 12.5081007083],
    dtype=float,
)
CTA_LOWER_FRONT_TARGET_Z_M = np.asarray(
    [
        -2.368597705825217,
        -2.4001434250887357,
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
CTA_UPPER_FRONT_TARGET_Y = np.asarray([1.9, 5.694, 8.041, 12.5081007083], dtype=float)
CTA_UPPER_FRONT_TARGET_Z_M = np.asarray(
    [3.37393430087873, 2.684934294708757, 2.03570975256528, 1.2389898634240581],
    dtype=float,
)
CTA_UPPER_FRONT_XC = np.asarray(
    [0.22768048249248646, 0.26719273983744424, 0.27885565489049924, 0.38327731807204724],
    dtype=float,
)
CTA_UPPER_LE_TARGET_Z_M = np.asarray(
    [1.03474, 0.48145, 0.67693, 0.7299199999999999],
    dtype=float,
)


def cosine_spacing(n: int) -> np.ndarray:
    beta = np.linspace(0.0, np.pi, int(n))
    return 0.5 * (1.0 - np.cos(beta))


def quintic_c2_transition(
    y: float,
    y0: float,
    y1: float,
    x0: float,
    x1: float,
    dx0: float,
    dx1: float,
    ddx0: float = 0.0,
    ddx1: float = 0.0,
) -> float:
    length = max(y1 - y0, 1.0e-12)
    t = np.clip((y - y0) / length, 0.0, 1.0)
    rhs = np.array(
        [x0, dx0 * length, ddx0 * length * length, x1, dx1 * length, ddx1 * length * length],
        dtype=float,
    )
    matrix = np.array(
        [
            [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 2.0, 0.0, 0.0, 0.0],
            [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            [0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
            [0.0, 0.0, 2.0, 6.0, 12.0, 20.0],
        ],
        dtype=float,
    )
    coeffs = np.linalg.solve(matrix, rhs)
    powers = np.array([1.0, t, t * t, t**3, t**4, t**5], dtype=float)
    return float(coeffs @ powers)


def with_root_blend(
    interpolant: Callable[[float], float],
    root_value: float,
    root_blend_y: float,
) -> Callable[[float], float]:
    root_blend_y = float(max(root_blend_y, 0.0))
    if root_blend_y <= 1.0e-12:
        return interpolant

    x1 = float(interpolant(root_blend_y))
    eps = max(root_blend_y * 1.0e-4, 1.0e-6)
    y_left = max(root_blend_y - eps, 0.0)
    y_right = root_blend_y + eps
    if y_right <= y_left:
        dx1 = 0.0
    else:
        dx1 = (float(interpolant(y_right)) - float(interpolant(y_left))) / (y_right - y_left)

    def wrapped(yy: float) -> float:
        y_val = float(yy)
        if y_val <= 0.0:
            return float(root_value)
        if y_val >= root_blend_y:
            return float(interpolant(y_val))
        return quintic_c2_transition(
            y=y_val,
            y0=0.0,
            y1=root_blend_y,
            x0=float(root_value),
            x1=x1,
            dx0=0.0,
            dx1=float(dx1),
        )

    return wrapped


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


def _front_local_envelope(
    x_over_c: np.ndarray,
    upper_z_over_c: np.ndarray,
    lower_z_over_c: np.ndarray,
    chord_m: float,
    twist_deg: float,
) -> tuple[float, float]:
    theta = np.radians(float(twist_deg))
    x_local = chord_m * np.asarray(x_over_c, dtype=float)
    upper_local = chord_m * np.asarray(upper_z_over_c, dtype=float)
    lower_local = chord_m * np.asarray(lower_z_over_c, dtype=float)
    z_upper = x_local * np.sin(theta) + upper_local * np.cos(theta)
    z_lower = x_local * np.sin(theta) + lower_local * np.cos(theta)
    return float(np.max(z_upper)), float(np.min(z_lower))


def _solve_twist_for_target_front_thickness(
    *,
    x_over_c: np.ndarray,
    upper_z_over_c: np.ndarray,
    lower_z_over_c: np.ndarray,
    chord_m: float,
    current_twist_deg: float,
    target_thickness_m: float,
) -> float:
    current_twist = float(current_twist_deg)

    def residual(twist_deg: float) -> float:
        local_upper, local_lower = _front_local_envelope(
            x_over_c=x_over_c,
            upper_z_over_c=upper_z_over_c,
            lower_z_over_c=lower_z_over_c,
            chord_m=chord_m,
            twist_deg=float(twist_deg),
        )
        return float((local_upper - local_lower) - target_thickness_m)

    grid = np.linspace(current_twist - 6.0, current_twist + 6.0, 121)
    residuals = np.asarray([residual(float(value)) for value in grid], dtype=float)
    candidate_intervals: list[tuple[float, float, float]] = []
    for left, right, res_left, res_right in zip(
        grid[:-1],
        grid[1:],
        residuals[:-1],
        residuals[1:],
        strict=True,
    ):
        if res_left == 0.0:
            return float(left)
        if res_left * res_right <= 0.0:
            center = 0.5 * (float(left) + float(right))
            candidate_intervals.append((abs(center - current_twist), float(left), float(right)))
    if candidate_intervals:
        _, left, right = min(candidate_intervals, key=lambda item: item[0])
        return float(brentq(residual, left, right))
    best_idx = int(np.argmin(np.abs(residuals)))
    return float(grid[best_idx])


def _apply_root_te_flatten(
    *,
    y_value: float,
    x_over_c: np.ndarray,
    upper_z_over_c: np.ndarray,
    lower_z_over_c: np.ndarray,
    chord_m: float,
    twist_deg: float,
    leading_edge_z_m: float,
    target_te_center_fun: Callable[[float], float],
) -> tuple[np.ndarray, np.ndarray]:
    if y_value <= CTA_ROOT_TE_CENTER_TARGET_Y[0] + 1.0e-12 or y_value >= CTA_ROOT_TE_CENTER_TARGET_Y[-1] - 1.0e-12:
        return upper_z_over_c, lower_z_over_c
    theta = np.radians(float(twist_deg))
    cos_theta = float(np.cos(theta))
    if abs(chord_m * cos_theta) <= 1.0e-12:
        return upper_z_over_c, lower_z_over_c
    x_te = float(np.asarray(x_over_c, dtype=float)[-1]) * chord_m
    current_te_center = (
        float(leading_edge_z_m)
        + x_te * np.sin(theta)
        + 0.5 * (float(upper_z_over_c[-1]) + float(lower_z_over_c[-1])) * chord_m * cos_theta
    )
    target_te_center = float(target_te_center_fun(float(y_value)))
    delta_local = float((target_te_center - current_te_center) / (chord_m * cos_theta))
    if abs(delta_local) <= 1.0e-12:
        return upper_z_over_c, lower_z_over_c
    t = np.clip(
        (np.asarray(x_over_c, dtype=float) - CTA_ROOT_TE_FLATTEN_X_START)
        / max(1.0 - CTA_ROOT_TE_FLATTEN_X_START, 1.0e-12),
        0.0,
        1.0,
    )
    aft_weight = t * t * (3.0 - 2.0 * t)
    return upper_z_over_c + delta_local * aft_weight, lower_z_over_c + delta_local * aft_weight


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
    x_local = x_air_arr * chord_m
    z_world = float(leading_edge_z_m) + x_local * np.sin(theta) + lower_arr * chord_m * cos_theta
    current_lower = float(np.min(z_world))
    target_lower = float(target_lower_fun(float(y_value)))
    delta_local = float((target_lower - current_lower) / (chord_m * cos_theta))
    if abs(delta_local) <= 1.0e-12:
        return lower_arr
    x_c_center = float(np.clip(x_c_min_fun(float(y_value)), 0.0, 1.0))
    sigma = float(max(CTA_LOWER_FRONT_XC_SIGMA, 1.0e-3))
    weight = np.exp(-0.5 * ((x_air_arr - x_c_center) / sigma) ** 2)
    return lower_arr + delta_local * weight


def _apply_upper_front_guidance(
    *,
    y_value: float,
    x_over_c: np.ndarray,
    upper_z_over_c: np.ndarray,
    chord_m: float,
    twist_deg: float,
    leading_edge_z_m: float,
    target_upper_fun: Callable[[float], float],
    x_c_max_fun: Callable[[float], float],
) -> np.ndarray:
    if y_value < CTA_UPPER_FRONT_TARGET_Y[0] - 1.0e-12 or y_value > CTA_UPPER_FRONT_TARGET_Y[2] + 1.0e-12:
        return upper_z_over_c
    theta = np.radians(float(twist_deg))
    cos_theta = float(np.cos(theta))
    if abs(chord_m * cos_theta) <= 1.0e-12:
        return upper_z_over_c
    x_air_arr = np.asarray(x_over_c, dtype=float)
    upper_arr = np.asarray(upper_z_over_c, dtype=float)
    x_local = x_air_arr * chord_m
    z_world = float(leading_edge_z_m) + x_local * np.sin(theta) + upper_arr * chord_m * cos_theta
    current_upper = float(np.max(z_world))
    target_upper = float(target_upper_fun(float(y_value)))
    delta_local = float((target_upper - current_upper) / (chord_m * cos_theta))
    if abs(delta_local) <= 1.0e-12:
        return upper_arr
    x_c_center = float(np.clip(x_c_max_fun(float(y_value)), 0.0, 1.0))
    sigma = float(max(CTA_UPPER_FRONT_XC_SIGMA, 1.0e-3))
    weight = np.exp(-0.5 * ((x_air_arr - x_c_center) / sigma) ** 2)
    return upper_arr + delta_local * weight


def _apply_upper_le_guidance(
    *,
    y_value: float,
    x_over_c: np.ndarray,
    upper_z_over_c: np.ndarray,
    chord_m: float,
    twist_deg: float,
    leading_edge_z_m: float,
    target_le_fun: Callable[[float], float],
) -> np.ndarray:
    if y_value < CTA_UPPER_FRONT_TARGET_Y[0] - 1.0e-12 or y_value > CTA_UPPER_FRONT_TARGET_Y[-1] + 1.0e-12:
        return upper_z_over_c
    theta = np.radians(float(twist_deg))
    cos_theta = float(np.cos(theta))
    if abs(chord_m * cos_theta) <= 1.0e-12:
        return upper_z_over_c
    x_air_arr = np.asarray(x_over_c, dtype=float)
    upper_arr = np.asarray(upper_z_over_c, dtype=float)
    current_le = float(leading_edge_z_m) + float(upper_arr[0]) * chord_m * cos_theta
    target_le = float(target_le_fun(float(y_value)))
    delta_local = float((target_le - current_le) / (chord_m * cos_theta))
    if abs(delta_local) <= 1.0e-12:
        return upper_arr
    sigma = float(max(CTA_UPPER_FRONT_LE_XC_SIGMA, 1.0e-3))
    weight = np.exp(-0.5 * (x_air_arr / sigma) ** 2)
    return upper_arr + delta_local * weight


def _build_cta_law_set(
    anchor_sections: tuple[Section, ...],
    config: WingGeometryConfig,
) -> CTAResolvedLawSet:
    anchor_stations = tuple(resolve_anchor_section(section, config) for section in anchor_sections)
    anchor_y = np.asarray([station.spanwise_y_m for station in anchor_stations], dtype=float)
    x_over_c = np.asarray(anchor_stations[0].x_over_c, dtype=float)
    root_blend_y = float(CTA_ROOT_FRONT_BLEND_FRACTION * anchor_y[1])
    planform = build_cta_planform()
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


def build_cta_planform():
    leading_edge_points = np.asarray(
        [
            (CTA_PLANFORM_SECTION_LE_X[0], CTA_PLANFORM_SECTION_Y[0]),
            (CTA_NOSE_HELPER_1_X_M, CTA_NOSE_HELPER_1_Y_M),
            (CTA_C1_LE_HELPER_X_M, CTA_C1_Y_M),
            (CTA_PLANFORM_SECTION_LE_X[1], CTA_PLANFORM_SECTION_Y[1]),
            (CTA_PLANFORM_SECTION_LE_X[2], CTA_PLANFORM_SECTION_Y[2]),
            (CTA_PLANFORM_SECTION_LE_X[3], CTA_PLANFORM_SECTION_Y[3]),
        ],
        dtype=float,
    )
    te_root = float(CTA_PLANFORM_SECTION_LE_X[0] + CTA_PLANFORM_SECTION_CHORD[0])
    te_c3 = float(CTA_PLANFORM_SECTION_LE_X[1] + CTA_PLANFORM_SECTION_CHORD[1])
    te_c4 = float(CTA_PLANFORM_SECTION_LE_X[2] + CTA_PLANFORM_SECTION_CHORD[2])
    te_c5 = float(CTA_PLANFORM_SECTION_LE_X[3] + CTA_PLANFORM_SECTION_CHORD[3])
    te_inboard_blend = float(np.clip(te_c3 + CTA_TE_INBOARD_BLEND_DX_M, te_c3, te_root))
    trailing_edge_points = np.asarray(
        [
            (te_root, 0.0),
            (te_root, CTA_C1_Y_M),
            (te_inboard_blend, CTA_TE_INBOARD_BLEND_Y_M),
            (te_c3, CTA_PLANFORM_SECTION_Y[1]),
            (te_c4, CTA_PLANFORM_SECTION_Y[2]),
            (te_c5, CTA_PLANFORM_SECTION_Y[3]),
        ],
        dtype=float,
    )
    if abs(CTA_MED_3_TE_SWEEP_DEG) > 1.0e-12:
        y_med3 = float(
            CTA_PLANFORM_SECTION_Y[1]
            + CTA_MED_3_TE_HELPER_FRACTION * (CTA_PLANFORM_SECTION_Y[2] - CTA_PLANFORM_SECTION_Y[1])
        )
        te_med3 = float(
            te_c3 + np.tan(np.radians(CTA_MED_3_TE_SWEEP_DEG)) * (y_med3 - CTA_PLANFORM_SECTION_Y[1])
        )
        trailing_edge_points = np.insert(
            trailing_edge_points,
            4,
            np.asarray([[te_med3, y_med3]], dtype=float),
            axis=0,
        )

    return build_control_point_planform(
        leading_edge_points=leading_edge_points,
        trailing_edge_points=trailing_edge_points,
        root_le_x=float(CTA_PLANFORM_SECTION_LE_X[0]),
        root_te_x=te_root,
        continuity_order=CTA_PLANFORM_CONTINUITY_ORDER,
        blend_fraction=CTA_PLANFORM_BLEND_FRACTION,
        min_linear_core_fraction=CTA_PLANFORM_MIN_LINEAR_CORE_FRACTION,
        te_blend_fraction=CTA_PLANFORM_TE_BLEND_FRACTION,
        te_min_linear_core_fraction=CTA_PLANFORM_TE_MIN_LINEAR_CORE_FRACTION,
        le_linear_start_index=CTA_PLANFORM_LE_LINEAR_START_INDEX,
        te_linear_start_index=CTA_PLANFORM_TE_LINEAR_START_INDEX,
        le_exact_segments=(),
        te_exact_segments=CTA_PLANFORM_TE_EXACT_SEGMENTS,
        le_spline_bridge=None,
        te_spline_bridge=CTA_PLANFORM_TE_SPLINE_BRIDGE,
        symmetry_blend_y=CTA_PLANFORM_SYMMETRY_BLEND_Y,
        body_le_fixed_points=((CTA_NOSE_HELPER_1_X_M, CTA_NOSE_HELPER_1_Y_M), (CTA_C1_LE_HELPER_X_M, CTA_C1_Y_M)),
        te_inboard_radius_factor=CTA_TE_INBOARD_RADIUS_FACTOR,
    )


def build_cta_span_station_array(
    *,
    component: Wing,
    anchor_sections: tuple[Section, ...],
    config: WingGeometryConfig,
) -> np.ndarray:
    del component
    del config
    anchor_y = np.asarray([float(section.spanwise_y_m) for section in anchor_sections], dtype=float)
    half_span = float(anchor_y[-1])
    base_stations = np.linspace(0.0, half_span, CTA_NUM_BASE_STATIONS, dtype=float)
    root_blend_stations = np.array([], dtype=float)
    if CTA_PLANFORM_SYMMETRY_BLEND_Y > 1.0e-12:
        root_blend_stations = float(CTA_PLANFORM_SYMMETRY_BLEND_Y) * cosine_spacing(CTA_NUM_ROOT_BLEND_STATIONS)
    planform = build_cta_planform()
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

    del component
    law_set = _build_cta_law_set(anchor_sections, config)
    anchor_name_map = {
        round(float(section.spanwise_y_m), 10): section.name
        for section in anchor_sections
    }
    target_root_upper_fun = build_scalar_interpolant(
        CTA_ROOT_FRONT_TARGET_Y,
        CTA_ROOT_FRONT_TARGET_UPPER_Z_M,
        "pchip",
    )
    target_root_lower_fun = build_scalar_interpolant(
        CTA_ROOT_FRONT_TARGET_Y,
        CTA_ROOT_FRONT_TARGET_LOWER_Z_M,
        "pchip",
    )
    target_te_center_fun = build_scalar_interpolant(
        CTA_ROOT_TE_CENTER_TARGET_Y,
        CTA_ROOT_TE_CENTER_TARGET_Z_M,
        "pchip",
    )
    target_lower_fun = build_scalar_interpolant(
        CTA_LOWER_FRONT_TARGET_Y,
        CTA_LOWER_FRONT_TARGET_Z_M,
        "pchip",
    )
    lower_x_c_fun = build_scalar_interpolant(
        CTA_LOWER_FRONT_TARGET_Y,
        CTA_LOWER_FRONT_XC,
        "pchip",
    )
    target_upper_fun = build_scalar_interpolant(
        CTA_UPPER_FRONT_TARGET_Y,
        CTA_UPPER_FRONT_TARGET_Z_M,
        "pchip",
    )
    upper_x_c_fun = build_scalar_interpolant(
        CTA_UPPER_FRONT_TARGET_Y,
        CTA_UPPER_FRONT_XC,
        "pchip",
    )
    target_upper_le_fun = build_scalar_interpolant(
        CTA_UPPER_FRONT_TARGET_Y,
        CTA_UPPER_LE_TARGET_Z_M,
        "pchip",
    )

    def resolver(sample_y: np.ndarray) -> tuple[ResolvedStation, ...]:
        stations: list[ResolvedStation] = []
        for idx, y_value in enumerate(np.asarray(sample_y, dtype=float)):
            upper_z_over_c = _evaluate_profile_interpolants(law_set.upper_profile_interpolants, float(y_value))
            lower_z_over_c = _evaluate_profile_interpolants(law_set.lower_profile_interpolants, float(y_value))
            chord_m = float(law_set.chord_fun(float(y_value)))
            base_twist_deg = float(law_set.base_twist_fun(float(y_value)))
            leading_edge_x_m = float(law_set.leading_edge_x_fun(float(y_value)))
            base_leading_edge_z_m = float(law_set.base_leading_edge_z_fun(float(y_value)))

            if CTA_ROOT_FRONT_TARGET_Y[0] + 1.0e-12 < float(y_value) < CTA_ROOT_FRONT_TARGET_Y[-1] - 1.0e-12:
                target_upper = float(target_root_upper_fun(float(y_value)))
                target_lower = float(target_root_lower_fun(float(y_value)))
                target_thickness = float(target_upper - target_lower)
                twist_deg = _solve_twist_for_target_front_thickness(
                    x_over_c=law_set.x_over_c,
                    upper_z_over_c=upper_z_over_c,
                    lower_z_over_c=lower_z_over_c,
                    chord_m=chord_m,
                    current_twist_deg=base_twist_deg,
                    target_thickness_m=target_thickness,
                )
                local_upper, local_lower = _front_local_envelope(
                    law_set.x_over_c,
                    upper_z_over_c,
                    lower_z_over_c,
                    chord_m,
                    twist_deg,
                )
                leading_edge_z_m = 0.5 * (
                    (target_upper + target_lower)
                    - (float(local_upper) + float(local_lower))
                )
            else:
                twist_deg = base_twist_deg
                leading_edge_z_m = base_leading_edge_z_m

            upper_z_over_c, lower_z_over_c = _apply_root_te_flatten(
                y_value=float(y_value),
                x_over_c=law_set.x_over_c,
                upper_z_over_c=upper_z_over_c,
                lower_z_over_c=lower_z_over_c,
                chord_m=chord_m,
                twist_deg=twist_deg,
                leading_edge_z_m=leading_edge_z_m,
                target_te_center_fun=target_te_center_fun,
            )
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
            upper_z_over_c = _apply_upper_front_guidance(
                y_value=float(y_value),
                x_over_c=law_set.x_over_c,
                upper_z_over_c=upper_z_over_c,
                chord_m=chord_m,
                twist_deg=twist_deg,
                leading_edge_z_m=leading_edge_z_m,
                target_upper_fun=target_upper_fun,
                x_c_max_fun=upper_x_c_fun,
            )
            upper_z_over_c = _apply_upper_le_guidance(
                y_value=float(y_value),
                x_over_c=law_set.x_over_c,
                upper_z_over_c=upper_z_over_c,
                chord_m=chord_m,
                twist_deg=twist_deg,
                leading_edge_z_m=leading_edge_z_m,
                target_le_fun=target_upper_le_fun,
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
