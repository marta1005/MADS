"""Private planform axis utilities for synthesis geometry workflows."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.interpolate import PchipInterpolator

from multiads.utilities.pyspline_loader import load_pyspline_curve


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


def cubic_c1_transition(
    y: float,
    y0: float,
    y1: float,
    x0: float,
    x1: float,
    dx0: float,
    dx1: float,
) -> float:
    length = max(y1 - y0, 1.0e-12)
    t = np.clip((y - y0) / length, 0.0, 1.0)
    rhs = np.array([x0, dx0 * length, x1, dx1 * length], dtype=float)
    matrix = np.array(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [1.0, 1.0, 1.0, 1.0],
            [0.0, 1.0, 2.0, 3.0],
        ],
        dtype=float,
    )
    coeffs = np.linalg.solve(matrix, rhs)
    powers = np.array([1.0, t, t * t, t**3], dtype=float)
    return float(coeffs @ powers)


def smooth_transition(
    y: float,
    y0: float,
    y1: float,
    x0: float,
    x1: float,
    dx0: float,
    dx1: float,
    continuity_order: int,
) -> float:
    if continuity_order == 1:
        return cubic_c1_transition(y, y0, y1, x0, x1, dx0, dx1)
    return quintic_c2_transition(y, y0, y1, x0, x1, dx0, dx1)


def spline_transition(
    y0: float,
    y1: float,
    x0: float,
    x1: float,
    dx0: float,
    dx1: float,
    continuity_order: int,
):
    length = max(y1 - y0, 1.0e-12)
    pad = 0.5 * length
    y_mid = 0.5 * (y0 + y1)
    x_mid = smooth_transition(
        y=y_mid,
        y0=y0,
        y1=y1,
        x0=x0,
        x1=x1,
        dx0=dx0,
        dx1=dx1,
        continuity_order=continuity_order,
    )
    support_y = np.asarray((y0 - pad, y0, y_mid, y1, y1 + pad), dtype=float)
    support_x = np.asarray((x0 - dx0 * pad, x0, x_mid, x1, x1 + dx1 * pad), dtype=float)
    if np.allclose(support_x, support_x[0], atol=1.0e-12, rtol=1.0e-12):
        return float(support_x[0])
    curve_class = load_pyspline_curve()
    return curve_class(x=support_x, s=support_y, k=min(4, support_x.size))


def span_transition(y: float, curve: object) -> float:
    if isinstance(curve, (float, np.floating)):
        return float(curve)
    return float(np.asarray(curve(float(y))).reshape(-1)[0])


class PiecewiseLinearAxis:
    def __init__(self, points: np.ndarray):
        pts = np.asarray(points, dtype=float)
        order = np.argsort(pts[:, 1])
        self.points = pts[order]
        if not np.all(np.diff(self.points[:, 1]) > 0.0):
            msg = "planform points must be strictly increasing in spanwise location"
            raise ValueError(msg)

    def __call__(self, y: float) -> float:
        return float(np.interp(float(y), self.points[:, 1], self.points[:, 0]))


class HybridTailAxis:
    def __init__(self, prefix_axis: object, points: np.ndarray, linear_start_index: int):
        pts = PiecewiseLinearAxis(points).points
        if linear_start_index < 1 or linear_start_index >= pts.shape[0] - 1:
            msg = (
                "HybridTailAxis requires 1 <= linear_start_index < point_count - 1, "
                f"got {linear_start_index} for {pts.shape[0]} points"
            )
            raise ValueError(msg)
        self.points = pts
        self._prefix_axis = prefix_axis
        self._linear_axis = PiecewiseLinearAxis(pts[int(linear_start_index) :])
        self._linear_start_y = float(pts[int(linear_start_index), 1])

    def __call__(self, y: float) -> float:
        y_val = float(y)
        if y_val <= self._linear_start_y:
            return float(self._prefix_axis(y_val))
        return float(self._linear_axis(y_val))


class SplineBridgeAxis:
    def __init__(
        self,
        points: np.ndarray,
        start_index: int,
        end_index: int,
        inboard_radius_factor: float = 1.0,
    ):
        pts = PiecewiseLinearAxis(points).points
        if not (0 < start_index < end_index < pts.shape[0] - 1):
            msg = (
                "SplineBridgeAxis requires 0 < start_index < end_index < point_count - 1, "
                f"got {(start_index, end_index)} for {pts.shape[0]} points"
            )
            raise ValueError(msg)

        self.points = pts
        self.start_index = int(start_index)
        self.end_index = int(end_index)
        self.y_start = float(pts[self.start_index, 1])
        self.y_end = float(pts[self.end_index, 1])
        self.inboard_radius_factor = float(max(inboard_radius_factor, 1.0e-6))
        self.left_axis = PiecewiseLinearAxis(pts[: self.start_index + 1])
        self.right_axis = PiecewiseLinearAxis(pts[self.end_index :])
        bridge_pts = pts[self.start_index : self.end_index + 1]
        self._curve = PchipInterpolator(
            bridge_pts[:, 1].astype(float),
            bridge_pts[:, 0].astype(float),
            extrapolate=True,
        )
        self.y_helper = float(bridge_pts[1, 1]) if bridge_pts.shape[0] >= 2 else self.y_start

    def __call__(self, y: float) -> float:
        y_val = float(y)
        if y_val <= self.y_start:
            return float(self.left_axis(y_val))
        if y_val >= self.y_end:
            return float(self.right_axis(y_val))
        y_eval = y_val
        if self.y_start < y_val < self.y_helper and abs(self.inboard_radius_factor - 1.0) > 1.0e-12:
            span = max(self.y_helper - self.y_start, 1.0e-12)
            t = (y_val - self.y_start) / span
            t_warp = np.power(np.clip(t, 0.0, 1.0), self.inboard_radius_factor)
            y_eval = self.y_start + t_warp * span
        return float(np.asarray(self._curve(y_eval)).reshape(-1)[0])


class SegmentedSpanAxis:
    def __init__(
        self,
        points: np.ndarray,
        continuity_order: int,
        blend_fraction: float,
        min_linear_core_fraction: float,
        exact_segment_indices=(),
    ):
        pts = PiecewiseLinearAxis(points).points
        if pts.shape[0] < 2:
            msg = "SegmentedSpanAxis expects at least 2 points"
            raise ValueError(msg)

        self.points = pts
        self.continuity_order = int(continuity_order)
        self.blend_fraction = float(np.clip(blend_fraction, 0.0, 0.45))
        self.min_linear_core_fraction = float(np.clip(min_linear_core_fraction, 0.0, 0.95))
        self.exact_segments = set(int(index) for index in exact_segment_indices)
        self.x = pts[:, 0].astype(float)
        self.y = pts[:, 1].astype(float)
        self.slopes = np.diff(self.x) / np.maximum(np.diff(self.y), 1.0e-12)
        self.interior_ids = np.arange(1, self.y.size - 1, dtype=int)
        self.widths_left = np.zeros_like(self.y)
        self.widths_right = np.zeros_like(self.y)
        self._transition_cache: dict[tuple[float, ...], object] = {}

        for idx in self.interior_ids:
            dy_left = self.y[idx] - self.y[idx - 1]
            dy_right = self.y[idx + 1] - self.y[idx]
            left_exact = (idx - 1) in self.exact_segments
            right_exact = idx in self.exact_segments
            if left_exact and right_exact:
                continue
            if right_exact and not left_exact:
                self.widths_left[idx] = self.blend_fraction * dy_left
                continue
            if left_exact and not right_exact:
                self.widths_right[idx] = self.blend_fraction * dy_right
                continue
            width = self.blend_fraction * min(dy_left, dy_right)
            self.widths_left[idx] = width
            self.widths_right[idx] = width

        for idx in range(1, self.y.size - 2):
            overlap = self.widths_right[idx] + self.widths_left[idx + 1]
            gap = self.y[idx + 1] - self.y[idx]
            max_overlap = max(0.0, (1.0 - self.min_linear_core_fraction) * gap)
            if overlap > max_overlap and overlap > 1.0e-12:
                scale = max_overlap / overlap
                self.widths_right[idx] *= scale
                self.widths_left[idx + 1] *= scale

    def _line(self, seg_idx: int, y: float) -> float:
        return float(self.x[seg_idx] + self.slopes[seg_idx] * (y - self.y[seg_idx]))

    def _smooth_at_node(self, node_idx: int, y: float) -> float:
        left_width = float(self.widths_left[node_idx])
        right_width = float(self.widths_right[node_idx])
        if left_width + right_width <= 1.0e-12:
            return float(self.x[node_idx])
        y0 = float(self.y[node_idx] - left_width)
        y1 = float(self.y[node_idx] + right_width)
        cache_key = (
            node_idx,
            y0,
            y1,
            float(self._line(node_idx - 1, y0)),
            float(self._line(node_idx, y1)),
            float(self.slopes[node_idx - 1]),
            float(self.slopes[node_idx]),
            self.continuity_order,
        )
        if cache_key not in self._transition_cache:
            self._transition_cache[cache_key] = spline_transition(
                y0=y0,
                y1=y1,
                x0=float(self._line(node_idx - 1, y0)),
                x1=float(self._line(node_idx, y1)),
                dx0=float(self.slopes[node_idx - 1]),
                dx1=float(self.slopes[node_idx]),
                continuity_order=self.continuity_order,
            )
        return span_transition(y=y, curve=self._transition_cache[cache_key])

    def __call__(self, y: float) -> float:
        y_val = float(y)
        if y_val <= self.y[0]:
            return float(self.x[0])
        if y_val >= self.y[-1]:
            return float(self.x[-1])
        for node_idx in self.interior_ids:
            y_left = float(self.y[node_idx] - self.widths_left[node_idx])
            y_right = float(self.y[node_idx] + self.widths_right[node_idx])
            if y_val < y_left:
                return self._line(node_idx - 1, y_val)
            if y_val <= y_right:
                return self._smooth_at_node(node_idx, y_val)
        return self._line(self.y.size - 2, y_val)


class SymmetryRootBlendAxis:
    def __init__(
        self,
        base_axis: object,
        root_x: float,
        target_slope: float,
        blend_y: float,
        continuity_order: int,
        post_axis: object | None = None,
    ):
        self.base_axis = base_axis
        self.post_axis = base_axis if post_axis is None else post_axis
        self.root_x = float(root_x)
        self.target_slope = float(target_slope)
        self.blend_y = float(max(blend_y, 0.0))
        self.continuity_order = int(continuity_order)

    def __call__(self, y: float) -> float:
        y_abs = abs(float(y))
        if y_abs <= 1.0e-12:
            return self.root_x
        if self.blend_y <= 1.0e-12 or y_abs >= self.blend_y:
            return float(self.post_axis(y_abs))
        x1 = float(self.post_axis(self.blend_y))
        return smooth_transition(
            y=y_abs,
            y0=0.0,
            y1=self.blend_y,
            x0=self.root_x,
            x1=x1,
            dx0=0.0,
            dx1=self.target_slope,
            continuity_order=self.continuity_order,
        )


@dataclass(frozen=True, slots=True)
class SectionedPlanform:
    le_axis: object
    te_axis: object
    leading_edge_points: np.ndarray
    trailing_edge_points: np.ndarray

    def le_x(self, y: float) -> float:
        return float(self.le_axis(float(y)))

    def te_x(self, y: float) -> float:
        return float(self.te_axis(float(y)))


def _build_span_axis(
    points: np.ndarray,
    continuity_order: int,
    blend_fraction: float,
    min_linear_core_fraction: float,
    exact_segment_indices=(),
    spline_bridge=None,
    linear_start_index=None,
    inboard_radius_factor: float = 1.0,
):
    if linear_start_index is not None:
        pts = PiecewiseLinearAxis(points).points
        if spline_bridge is not None:
            prefix_axis = SplineBridgeAxis(
                pts,
                start_index=int(spline_bridge[0]),
                end_index=int(spline_bridge[1]),
                inboard_radius_factor=inboard_radius_factor,
            )
        else:
            prefix_axis = SegmentedSpanAxis(
                pts,
                continuity_order=continuity_order,
                blend_fraction=blend_fraction,
                min_linear_core_fraction=min_linear_core_fraction,
                exact_segment_indices=exact_segment_indices,
            )
        return HybridTailAxis(prefix_axis, pts, linear_start_index=int(linear_start_index))

    if spline_bridge is not None:
        return SplineBridgeAxis(
            points,
            start_index=int(spline_bridge[0]),
            end_index=int(spline_bridge[1]),
            inboard_radius_factor=inboard_radius_factor,
        )

    return SegmentedSpanAxis(
        points,
        continuity_order=continuity_order,
        blend_fraction=blend_fraction,
        min_linear_core_fraction=min_linear_core_fraction,
        exact_segment_indices=exact_segment_indices,
    )


def build_control_point_planform(
    *,
    leading_edge_points: np.ndarray,
    trailing_edge_points: np.ndarray,
    root_le_x: float,
    root_te_x: float,
    continuity_order: int,
    blend_fraction: float,
    min_linear_core_fraction: float,
    te_blend_fraction: float | None,
    te_min_linear_core_fraction: float | None,
    le_linear_start_index: int | None,
    te_linear_start_index: int | None,
    le_exact_segments=(),
    te_exact_segments=(),
    le_spline_bridge=None,
    te_spline_bridge=None,
    symmetry_blend_y: float = 0.0,
    body_le_fixed_points=(),
    te_inboard_radius_factor: float = 1.0,
) -> SectionedPlanform:
    le_points = np.asarray(leading_edge_points, dtype=float)
    te_points = np.asarray(trailing_edge_points, dtype=float)
    le_axis = _build_span_axis(
        le_points,
        continuity_order=continuity_order,
        blend_fraction=blend_fraction,
        min_linear_core_fraction=min_linear_core_fraction,
        exact_segment_indices=le_exact_segments,
        spline_bridge=le_spline_bridge,
        linear_start_index=le_linear_start_index,
    )
    te_axis = _build_span_axis(
        te_points,
        continuity_order=continuity_order,
        blend_fraction=blend_fraction if te_blend_fraction is None else te_blend_fraction,
        min_linear_core_fraction=(
            min_linear_core_fraction
            if te_min_linear_core_fraction is None
            else te_min_linear_core_fraction
        ),
        exact_segment_indices=te_exact_segments,
        spline_bridge=te_spline_bridge,
        linear_start_index=te_linear_start_index,
        inboard_radius_factor=te_inboard_radius_factor,
    )

    if symmetry_blend_y > 1.0e-12:
        le_post_axis = le_axis
        if body_le_fixed_points and le_points.shape[0] >= 3:
            le_slope0 = float((le_points[2, 0] - le_points[1, 0]) / max(le_points[2, 1] - le_points[1, 1], 1.0e-12))
            le_linear_start_shifted = None
            if le_linear_start_index is not None:
                le_linear_start_shifted = max(int(le_linear_start_index) - 1, 1)
            le_exact_shifted = tuple(max(int(index) - 1, 0) for index in le_exact_segments if int(index) >= 1)
            le_spline_bridge_shifted = None
            if le_spline_bridge is not None:
                start_idx, end_idx = le_spline_bridge
                if int(start_idx) >= 1 and int(end_idx) >= 1:
                    le_spline_bridge_shifted = (int(start_idx) - 1, int(end_idx) - 1)
            le_post_axis = _build_span_axis(
                le_points[1:],
                continuity_order=continuity_order,
                blend_fraction=blend_fraction,
                min_linear_core_fraction=min_linear_core_fraction,
                exact_segment_indices=le_exact_shifted,
                spline_bridge=le_spline_bridge_shifted,
                linear_start_index=le_linear_start_shifted,
            )
        else:
            le_slope0 = float((le_points[1, 0] - le_points[0, 0]) / max(le_points[1, 1] - le_points[0, 1], 1.0e-12))

        if body_le_fixed_points:
            blend_y = float(min(symmetry_blend_y, le_points[1, 1]))
        else:
            blend_y = float(min(symmetry_blend_y, 0.95 * min(le_points[1, 1], te_points[1, 1])))

        le_axis = SymmetryRootBlendAxis(
            base_axis=le_axis,
            root_x=float(root_le_x),
            target_slope=le_slope0,
            blend_y=blend_y,
            continuity_order=continuity_order,
            post_axis=le_post_axis,
        )
        if not body_le_fixed_points and 0 not in set(int(index) for index in te_exact_segments):
            te_slope0 = float((te_points[1, 0] - te_points[0, 0]) / max(te_points[1, 1] - te_points[0, 1], 1.0e-12))
            te_axis = SymmetryRootBlendAxis(
                base_axis=te_axis,
                root_x=float(root_te_x),
                target_slope=te_slope0,
                blend_y=blend_y,
                continuity_order=continuity_order,
            )

    return SectionedPlanform(
        le_axis=le_axis,
        te_axis=te_axis,
        leading_edge_points=le_points,
        trailing_edge_points=te_points,
    )
