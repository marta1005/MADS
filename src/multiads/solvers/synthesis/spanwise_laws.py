"""Reusable spanwise laws for section-based parametric geometries.

The functions in this module are solver-independent.  They turn station data
into callable spanwise laws that can be used by any geometry case before the
result is packed into ``ResolvedStation`` objects.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from scipy.interpolate import PchipInterpolator

_log = logging.getLogger(__name__)

from multiads.solvers.synthesis.geometry_lib import (
    build_control_point_planform,
    quintic_c2_transition,
)


@dataclass(frozen=True, slots=True)
class ControlPointPlanformLawConfig:
    """Options for a planform law defined by LE/TE control points."""

    continuity_order: int = 2
    blend_fraction: float = 0.20
    min_linear_core_fraction: float = 0.50
    te_blend_fraction: float | None = None
    te_min_linear_core_fraction: float | None = None
    le_linear_start_index: int | None = None
    te_linear_start_index: int | None = None
    le_exact_segments: tuple[int, ...] = ()
    te_exact_segments: tuple[int, ...] = ()
    le_spline_bridge: tuple[int, int] | None = None
    te_spline_bridge: tuple[int, int] | None = None
    symmetry_blend_y: float = 0.0
    body_le_fixed_points: tuple[int, ...] = ()
    te_inboard_radius_factor: float = 1.0


def cosine_spacing(n: int) -> np.ndarray:
    """Return ``n`` cosine-spaced values between 0 and 1."""

    beta = np.linspace(0.0, np.pi, int(n))
    return 0.5 * (1.0 - np.cos(beta))


def build_scalar_interpolant(
    y_sections: np.ndarray,
    values: np.ndarray,
    interpolation: str,
    linear_start_index: int | None = None,
    root_blend_y: float | None = None,
    *,
    label: str = "field",
) -> Callable[[float], float]:
    """Build a callable scalar law along span.

    Parameters
    ----------
    y_sections:
        Spanwise anchor locations.
    values:
        Scalar values at the anchor locations.
    interpolation:
        ``"linear"`` or ``"pchip"``.
    linear_start_index:
        Optional anchor index from which the outboard part is forced linear.
    root_blend_y:
        Optional inboard blend distance that imposes zero slope at symmetry.
    label:
        Human-readable name used in validation errors.
    """

    y_arr = np.asarray(y_sections, dtype=float)
    values_arr = np.asarray(values, dtype=float)
    if y_arr.ndim != 1 or values_arr.ndim != 1 or y_arr.size != values_arr.size:
        msg = f"{label} interpolation requires matching 1D y/value arrays."
        raise ValueError(msg)
    if np.any(np.diff(y_arr) <= 0.0):
        msg = f"{label} interpolation anchors must be strictly increasing in y."
        raise ValueError(msg)

    if linear_start_index is not None and values_arr.size > 2:
        linear_start_index = int(linear_start_index)
        if not (1 <= linear_start_index < values_arr.size - 1):
            msg = (
                "linear_start_index must lie inside [1, point_count - 2], "
                f"got {linear_start_index} for {values_arr.size} points"
            )
            raise ValueError(msg)
        prefix_interpolant = build_scalar_interpolant(
            y_arr[: linear_start_index + 1],
            values_arr[: linear_start_index + 1],
            interpolation,
            linear_start_index=None,
            label=label,
        )
        y_linear = y_arr[linear_start_index:]
        values_linear = values_arr[linear_start_index:]

        def hybrid_interpolant(yy: float) -> float:
            y_val = float(yy)
            if y_val <= float(y_arr[linear_start_index]):
                return float(prefix_interpolant(y_val))
            return float(np.interp(y_val, y_linear, values_linear))

        interpolant: Callable[[float], float] = hybrid_interpolant
    else:
        interpolation_name = interpolation.lower()
        if interpolation_name == "linear" or values_arr.size == 2:
            interpolant = lambda yy: float(np.interp(float(yy), y_arr, values_arr))
        elif interpolation_name == "pchip":
            pchip = PchipInterpolator(y_arr, values_arr)
            interpolant = lambda yy: float(pchip(float(yy)))
        else:
            msg = f"Unsupported interpolation '{interpolation}' for {label}."
            raise ValueError(msg)

    if root_blend_y is None or float(root_blend_y) <= 1.0e-12:
        return interpolant

    blend_y = float(root_blend_y)
    root_value = float(interpolant(0.0))
    join_value = float(interpolant(blend_y))
    eps = max(blend_y * 1.0e-4, 1.0e-6)
    y_left = max(blend_y - eps, 0.0)
    y_right = blend_y + eps
    join_slope = float(
        (interpolant(y_right) - interpolant(y_left))
        / max(y_right - y_left, 1.0e-12),
    )

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
    *,
    label: str = "positive field",
) -> Callable[[float], float]:
    """Build a positive spanwise law by interpolating the logarithm."""

    positive_values = np.maximum(np.asarray(values, dtype=float), float(floor))
    log_interpolant = build_scalar_interpolant(
        y_sections,
        np.log(positive_values),
        interpolation,
        linear_start_index=linear_start_index,
        label=label,
    )
    return lambda yy: float(np.exp(log_interpolant(float(yy))))


def build_profile_interpolants(
    anchor_y: np.ndarray,
    anchor_profiles: np.ndarray,
    *,
    interpolation: str,
    linear_start_index: int | None = None,
    label: str = "profile",
) -> list[Callable[[float], float]]:
    """Build one scalar law per profile coefficient."""

    profiles = np.asarray(anchor_profiles, dtype=float)
    if profiles.ndim != 2:
        msg = f"{label} coefficients must be a 2D array."
        raise ValueError(msg)
    return [
        build_scalar_interpolant(
            anchor_y,
            profiles[:, idx],
            interpolation,
            linear_start_index=linear_start_index,
            label=f"{label}_{idx}",
        )
        for idx in range(profiles.shape[1])
    ]


def evaluate_profile_interpolants(
    interpolants: list[Callable[[float], float]],
    y_value: float,
) -> np.ndarray:
    """Evaluate a list of coefficient laws at one spanwise location."""

    return np.asarray(
        [interpolant(float(y_value)) for interpolant in interpolants],
        dtype=float,
    )


def build_le_te_planform_law(
    *,
    section_y: np.ndarray,
    section_le_x: np.ndarray,
    section_chord: np.ndarray,
    config: ControlPointPlanformLawConfig | None = None,
):
    """Build a planform law from section ``y``, LE ``x`` and chord arrays."""

    cfg = config or ControlPointPlanformLawConfig()
    y_arr = np.asarray(section_y, dtype=float)
    le_x_arr = np.asarray(section_le_x, dtype=float)
    chord_arr = np.asarray(section_chord, dtype=float)
    if y_arr.ndim != 1 or le_x_arr.ndim != 1 or chord_arr.ndim != 1:
        msg = "Planform law expects 1D section_y, section_le_x and section_chord arrays."
        raise ValueError(msg)
    if not (y_arr.size == le_x_arr.size == chord_arr.size):
        msg = "Planform law section_y, section_le_x and section_chord must match."
        raise ValueError(msg)

    leading_edge_points = np.column_stack((le_x_arr, y_arr))
    trailing_edge_points = np.column_stack((le_x_arr + chord_arr, y_arr))
    return build_control_point_planform(
        leading_edge_points=leading_edge_points,
        trailing_edge_points=trailing_edge_points,
        root_le_x=float(le_x_arr[0]),
        root_te_x=float(le_x_arr[0] + chord_arr[0]),
        continuity_order=cfg.continuity_order,
        blend_fraction=cfg.blend_fraction,
        min_linear_core_fraction=cfg.min_linear_core_fraction,
        te_blend_fraction=cfg.te_blend_fraction,
        te_min_linear_core_fraction=cfg.te_min_linear_core_fraction,
        le_linear_start_index=cfg.le_linear_start_index,
        te_linear_start_index=cfg.te_linear_start_index,
        le_exact_segments=cfg.le_exact_segments,
        te_exact_segments=cfg.te_exact_segments,
        le_spline_bridge=cfg.le_spline_bridge,
        te_spline_bridge=cfg.te_spline_bridge,
        symmetry_blend_y=cfg.symmetry_blend_y,
        body_le_fixed_points=cfg.body_le_fixed_points,
        te_inboard_radius_factor=cfg.te_inboard_radius_factor,
    )


def scale_thickness_preserving_trailing_edge(
    *,
    x_over_c: np.ndarray,
    upper_z_over_c: np.ndarray,
    lower_z_over_c: np.ndarray,
    target_thickness_over_chord: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Scale profile thickness while preserving the trailing-edge thickness.

    The camber line is preserved.  The part of the thickness distribution that
    is not the linear trailing-edge wedge is scaled until the maximum thickness
    reaches ``target_thickness_over_chord``.
    """

    x_arr = np.asarray(x_over_c, dtype=float)
    upper_arr = np.asarray(upper_z_over_c, dtype=float)
    lower_arr = np.asarray(lower_z_over_c, dtype=float)
    if not (x_arr.shape == upper_arr.shape == lower_arr.shape):
        msg = "x_over_c, upper_z_over_c and lower_z_over_c must have matching shapes."
        raise ValueError(msg)

    camber = 0.5 * (upper_arr + lower_arr)
    thickness = upper_arr - lower_arr
    te_thickness = float(thickness[-1])
    shape_thickness = thickness - x_arr * te_thickness
    target_tc = float(target_thickness_over_chord)

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
    achieved_tc = float(np.max(new_thickness))
    if abs(achieved_tc - target_tc) > 1.0e-4:
        _log.warning(
            "scale_thickness_preserving_trailing_edge did not converge: "
            "target t/c = %.6f, achieved t/c = %.6f (error = %.2e). "
            "The profile shape may be incorrect for this sample.",
            target_tc,
            achieved_tc,
            abs(achieved_tc - target_tc),
        )
    return camber + 0.5 * new_thickness, camber - 0.5 * new_thickness


__all__ = [
    "build_le_te_planform_law",
    "build_positive_interpolant",
    "build_profile_interpolants",
    "build_scalar_interpolant",
    "ControlPointPlanformLawConfig",
    "cosine_spacing",
    "evaluate_profile_interpolants",
    "scale_thickness_preserving_trailing_edge",
]
