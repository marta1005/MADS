"""Geometry backend library for synthesis-driven wing resolution in MADS."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np
from numpy.typing import NDArray
from scipy.interpolate import PchipInterpolator

from multiads.assembly import AirfoilCST, Section, Span, Wing, flatten_components
from multiads.solvers import SolverOptions
from multiads.utilities.pygeo_export import export_prepared_geometry_to_pygeo
from multiads.utilities.pyspline_loader import load_pyspline_curve


SpanwiseLawFactory = Callable[..., Any]


@dataclass(slots=True)
class SamplingConfig:
    """Controls how geometry is sampled once it is resolved."""

    chordwise_points: int = 201
    spanwise_stations: int = 51
    include_anchor_stations: bool = True
    station_distribution: str = "eq"
    airfoil_distribution_mode: str = "anchors"


@dataclass(slots=True)
class InterpolationConfig:
    """Controls spanwise and section interpolation behavior."""

    spanwise_law: str = "pchip"
    section_law: str = "pchip"
    blend_curve: str = "linear"
    field_laws: dict[str, str] = field(default_factory=dict)
    field_scopes: dict[str, str] = field(default_factory=dict)
    spanwise_law_factory: SpanwiseLawFactory | None = None
    section_law_factory: SpanwiseLawFactory | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class ExportConfig:
    """Controls how the resolved geometry should be exported."""

    pygeo_mode: str = "stations"
    export_all_resolved_stations: bool = False
    blunt_trailing_edge: bool = True
    trailing_edge_height_m: float = 0.0
    out_dir: str | None = None
    iges_path: str | None = None
    meshing_iges_path: str | None = None
    frame_only_iges_path: str | None = None
    symmetric: bool = False
    tip_style: str = "rounded"
    section_curve_n_ctl: int = 18
    k_span: int = 4
    include_xy_symmetry_frame: bool = False
    write_frame_only: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class ConstraintEvaluationConfig:
    """Controls how internal packaging constraints are evaluated."""

    reference_frame_name: str = "model"
    triangle_resolution: int = 18
    mirror_about_symmetry_plane: bool = True
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class WingGeometryConfig:
    """Global configuration required to resolve a geometry-driven wing."""

    case_name: str | None = None
    component_name: str | None = None
    symmetry: bool = True
    mirror: bool = False
    sampling: SamplingConfig = field(default_factory=SamplingConfig)
    interpolation: InterpolationConfig = field(default_factory=InterpolationConfig)
    export: ExportConfig = field(default_factory=ExportConfig)
    constraints: ConstraintEvaluationConfig = field(default_factory=ConstraintEvaluationConfig)
    metadata: dict[str, Any] = field(default_factory=dict)


def _as_float_array_1d(values: NDArray[np.float64] | list[float] | tuple[float, ...]) -> NDArray[np.float64]:
    arr = np.asarray(values, dtype=float)
    if arr.ndim != 1:
        raise ValueError(f"Expected a 1D array, got shape {arr.shape}")
    return arr


def _as_float_array_2d(
    values: NDArray[np.float64] | list[list[float]] | tuple[tuple[float, ...], ...],
) -> NDArray[np.float64]:
    arr = np.asarray(values, dtype=float)
    if arr.ndim != 2:
        raise ValueError(f"Expected a 2D array, got shape {arr.shape}")
    return arr


@dataclass(slots=True)
class ResolvedStation:
    """One fully resolved wing station."""

    name: str
    spanwise_y_m: float
    chord_m: float
    twist_deg: float
    leading_edge_x_m: float
    leading_edge_z_m: float
    x_over_c: NDArray[np.float64]
    upper_z_over_c: NDArray[np.float64]
    lower_z_over_c: NDArray[np.float64]
    upper_surface_xyz_m: NDArray[np.float64]
    lower_surface_xyz_m: NDArray[np.float64]
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.x_over_c = _as_float_array_1d(self.x_over_c)
        self.upper_z_over_c = _as_float_array_1d(self.upper_z_over_c)
        self.lower_z_over_c = _as_float_array_1d(self.lower_z_over_c)
        self.upper_surface_xyz_m = _as_float_array_2d(self.upper_surface_xyz_m)
        self.lower_surface_xyz_m = _as_float_array_2d(self.lower_surface_xyz_m)

        n_points = self.x_over_c.size
        if self.upper_z_over_c.size != n_points or self.lower_z_over_c.size != n_points:
            raise ValueError("x_over_c, upper_z_over_c and lower_z_over_c must have the same length")
        if self.upper_surface_xyz_m.shape != (n_points, 3):
            raise ValueError("upper_surface_xyz_m must have shape (N, 3) matching x_over_c")
        if self.lower_surface_xyz_m.shape != (n_points, 3):
            raise ValueError("lower_surface_xyz_m must have shape (N, 3) matching x_over_c")


@dataclass(slots=True)
class GeometryEnvelope:
    """Front / side style envelopes extracted from resolved stations."""

    spanwise_y_m: NDArray[np.float64]
    upper_z_m: NDArray[np.float64]
    lower_z_m: NDArray[np.float64]
    leading_edge_x_m: NDArray[np.float64] | None = None
    trailing_edge_x_m: NDArray[np.float64] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.spanwise_y_m = _as_float_array_1d(self.spanwise_y_m)
        self.upper_z_m = _as_float_array_1d(self.upper_z_m)
        self.lower_z_m = _as_float_array_1d(self.lower_z_m)
        if self.upper_z_m.size != self.spanwise_y_m.size or self.lower_z_m.size != self.spanwise_y_m.size:
            raise ValueError("Envelope arrays must have consistent length")
        if self.leading_edge_x_m is not None:
            self.leading_edge_x_m = _as_float_array_1d(self.leading_edge_x_m)
        if self.trailing_edge_x_m is not None:
            self.trailing_edge_x_m = _as_float_array_1d(self.trailing_edge_x_m)


@dataclass(slots=True)
class PreparedGeometry:
    """Resolved geometry state shared by geometry, export and packaging solvers."""

    component_name: str
    case_name: str | None
    config: WingGeometryConfig
    anchor_stations: tuple[ResolvedStation, ...]
    resolved_stations: tuple[ResolvedStation, ...]
    envelope: GeometryEnvelope | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def station_at_y(self, spanwise_y_m: float, atol: float = 1.0e-9) -> ResolvedStation:
        """Return the station exactly matching the requested spanwise location."""

        for station in self.resolved_stations:
            if abs(float(station.spanwise_y_m) - float(spanwise_y_m)) <= atol:
                return station
        raise KeyError(f"No resolved station found at y={spanwise_y_m}")


@dataclass(slots=True)
class GeometryMetricSet:
    """Scalar metrics extracted from a resolved geometry state."""

    span_m: float
    planform_area_m2: float
    enclosed_volume_m3: float
    root_chord_m: float
    tip_chord_m: float
    mean_aerodynamic_chord_m: float | None = None
    anchor_station_count: int = 0
    resolved_station_count: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class PyGeoExportResult:
    """Artifacts and in-memory objects produced by a pyGeo export run."""

    station_names: tuple[str, ...]
    station_y_m: NDArray[np.float64]
    airfoil_dat_paths: tuple[str | None, ...]
    surface: Any | None = None
    meshing_surface: Any | None = None
    frame_only_surface: Any | None = None
    profiles_dir: str | None = None
    iges_path: str | None = None
    meshing_iges_path: str | None = None
    frame_only_iges_path: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.station_y_m = _as_float_array_1d(self.station_y_m)
        if len(self.station_names) != self.station_y_m.size:
            raise ValueError("station_names and station_y_m must have the same length")
        if len(self.airfoil_dat_paths) != self.station_y_m.size:
            raise ValueError("airfoil_dat_paths and station_y_m must have the same length")

    @property
    def station_count(self) -> int:
        return int(self.station_y_m.size)

    @property
    def written_profile_count(self) -> int:
        return int(sum(path is not None for path in self.airfoil_dat_paths))


def resolve_anchor_section(
    section: Section,
    config: WingGeometryConfig,
) -> ResolvedStation:
    """Resolve one anchor section into explicit normalized and dimensional points."""

    if section.airfoil is None:
        msg = f"Anchor section '{section.name}' does not define an airfoil."
        raise ValueError(msg)
    if section.spanwise_y_m is None:
        msg = f"Anchor section '{section.name}' does not define spanwise_y_m."
        raise ValueError(msg)

    chord_m = float(section.chord)
    twist_deg = float(section.twist)
    leading_edge_x_m = float(section.leading_edge_x_m or 0.0)
    leading_edge_z_m = float(section.leading_edge_z_m or 0.0)

    thickness_dist = section.airfoil.thickness_distribution(
        npoints=config.sampling.chordwise_points,
        distribution=config.sampling.station_distribution,
        chord=1.0,
    )
    camber_dist = section.airfoil.camber_distribution(
        npoints=config.sampling.chordwise_points,
        distribution=config.sampling.station_distribution,
        chord=1.0,
    )

    x_over_c = np.asarray(camber_dist[:, 0], dtype=float)
    thickness_over_c = np.asarray(thickness_dist[:, 1], dtype=float)
    camber_over_c = np.asarray(camber_dist[:, 1], dtype=float)
    upper_z_over_c = camber_over_c + 0.5 * thickness_over_c
    lower_z_over_c = camber_over_c - 0.5 * thickness_over_c

    upper_surface_xyz_m = _build_surface_xyz(
        x_over_c,
        upper_z_over_c,
        chord_m,
        section.spanwise_y_m,
        twist_deg,
        leading_edge_x_m,
        leading_edge_z_m,
    )
    lower_surface_xyz_m = _build_surface_xyz(
        x_over_c,
        lower_z_over_c,
        chord_m,
        section.spanwise_y_m,
        twist_deg,
        leading_edge_x_m,
        leading_edge_z_m,
    )

    return ResolvedStation(
        name=section.name,
        spanwise_y_m=float(section.spanwise_y_m),
        chord_m=chord_m,
        twist_deg=twist_deg,
        leading_edge_x_m=leading_edge_x_m,
        leading_edge_z_m=leading_edge_z_m,
        x_over_c=x_over_c,
        upper_z_over_c=upper_z_over_c,
        lower_z_over_c=lower_z_over_c,
        upper_surface_xyz_m=upper_surface_xyz_m,
        lower_surface_xyz_m=lower_surface_xyz_m,
        metadata=dict(section.metadata),
    )


def _build_surface_xyz(
    x_over_c: np.ndarray,
    z_over_c: np.ndarray,
    chord_m: float,
    spanwise_y_m: float,
    twist_deg: float,
    leading_edge_x_m: float,
    leading_edge_z_m: float,
    *,
    twist_reference_x_over_c: float = 0.25,
) -> np.ndarray:
    x_local = chord_m * np.asarray(x_over_c, dtype=float)
    z_local = chord_m * np.asarray(z_over_c, dtype=float)

    if twist_deg != 0.0:
        x_ref = twist_reference_x_over_c * chord_m
        theta = np.radians(twist_deg)
        c = np.cos(theta)
        s = np.sin(theta)
        x_shift = x_local - x_ref
        x_rot = c * x_shift + s * z_local
        z_rot = -s * x_shift + c * z_local
        x_local = x_rot + x_ref
        z_local = z_rot

    xyz = np.zeros((x_local.size, 3), dtype=float)
    xyz[:, 0] = leading_edge_x_m + x_local
    xyz[:, 1] = spanwise_y_m
    xyz[:, 2] = leading_edge_z_m + z_local
    return xyz


def _sorted_pairs(
    anchor_y: np.ndarray,
    values: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    order = np.argsort(anchor_y)
    return anchor_y[order], values[order]


def interpolate_scalar_field(
    anchor_y: np.ndarray,
    values: np.ndarray,
    sample_y: np.ndarray,
    *,
    law: str = "pchip",
) -> np.ndarray:
    """Interpolate a scalar field along span."""

    y, v = _sorted_pairs(np.asarray(anchor_y, dtype=float), np.asarray(values, dtype=float))
    if y.size == 1:
        return np.full_like(np.asarray(sample_y, dtype=float), float(v[0]), dtype=float)

    law_name = law.lower()
    if law_name == "linear":
        return np.interp(sample_y, y, v)
    if law_name == "pchip":
        return np.asarray(PchipInterpolator(y, v)(sample_y), dtype=float)

    msg = f"Unsupported spanwise law '{law}'."
    raise ValueError(msg)


def interpolate_vector_field(
    anchor_y: np.ndarray,
    values: np.ndarray,
    sample_y: np.ndarray,
    *,
    law: str = "pchip",
) -> np.ndarray:
    """Interpolate a vector-valued field component-wise along span."""

    anchor_y = np.asarray(anchor_y, dtype=float)
    values = np.asarray(values, dtype=float)
    sample_y = np.asarray(sample_y, dtype=float)

    if values.ndim != 2:
        msg = f"Expected a 2D array for vector interpolation, got shape {values.shape}."
        raise ValueError(msg)

    return np.column_stack(
        [
            interpolate_scalar_field(anchor_y, values[:, idx], sample_y, law=law)
            for idx in range(values.shape[1])
        ]
    )


def merge_sampled_stations(
    anchor_y: np.ndarray,
    requested_count: int,
    *,
    include_anchors: bool = True,
) -> np.ndarray:
    """Create the spanwise station sampling while optionally preserving anchors."""

    y = np.asarray(anchor_y, dtype=float)
    if y.ndim != 1 or y.size == 0:
        msg = "Anchor spanwise coordinates must be a non-empty 1D array."
        raise ValueError(msg)

    y = np.unique(np.sort(y))
    if requested_count <= y.size:
        return y

    sampled = np.linspace(float(y[0]), float(y[-1]), int(requested_count), dtype=float)
    if include_anchors:
        sampled = np.unique(np.concatenate((sampled, y)))
    return sampled


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


class Options(SolverOptions):
    """Options specific to the CST geometry solver."""

    def __init__(
        self,
        *,
        write_metrics_back_to_component: bool = True,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.write_metrics_back_to_component = write_metrics_back_to_component


class PyGeoExportOptions(SolverOptions):
    """Options specific to the pyGeo export solver."""

    def __init__(
        self,
        *,
        write_export_back_to_component: bool = True,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.write_export_back_to_component = write_export_back_to_component


def build_geometry_config(component: Wing) -> WingGeometryConfig:
    """Build the global geometry configuration of a geometry-driven ``Wing``."""

    config = WingGeometryConfig(
        case_name=component.case_name,
        component_name=component.name,
        symmetry=bool(component.symmetry),
        mirror=bool(component.mirror),
    )
    sampling_data = component.metadata.get("sampling", {})
    if isinstance(sampling_data, dict):
        if "chordwise_points" in sampling_data:
            config.sampling.chordwise_points = int(sampling_data["chordwise_points"])
        if "spanwise_stations" in sampling_data:
            config.sampling.spanwise_stations = int(sampling_data["spanwise_stations"])
        if "include_anchor_stations" in sampling_data:
            config.sampling.include_anchor_stations = bool(sampling_data["include_anchor_stations"])
        if "station_distribution" in sampling_data:
            config.sampling.station_distribution = str(sampling_data["station_distribution"])
        if "airfoil_distribution_mode" in sampling_data:
            config.sampling.airfoil_distribution_mode = str(sampling_data["airfoil_distribution_mode"])
    interpolation_data = component.metadata.get("interpolation", {})
    if isinstance(interpolation_data, dict):
        if "spanwise_law" in interpolation_data:
            config.interpolation.spanwise_law = str(interpolation_data["spanwise_law"])
        if "section_law" in interpolation_data:
            config.interpolation.section_law = str(interpolation_data["section_law"])
        if "blend_curve" in interpolation_data:
            config.interpolation.blend_curve = str(interpolation_data["blend_curve"])
        config.interpolation.field_laws.update(
            {
                str(key): str(value)
                for key, value in dict(interpolation_data.get("field_laws", {})).items()
            }
        )
        config.interpolation.field_scopes.update(
            {
                str(key): str(value)
                for key, value in dict(interpolation_data.get("field_scopes", {})).items()
            }
        )
        config.interpolation.metadata.update(
            {
                str(key): value
                for key, value in dict(interpolation_data.get("metadata", {})).items()
            }
        )
    export_data = component.metadata.get("export", {})
    if isinstance(export_data, dict):
        if "pygeo_mode" in export_data:
            config.export.pygeo_mode = str(export_data["pygeo_mode"])
        if "export_all_resolved_stations" in export_data:
            config.export.export_all_resolved_stations = bool(export_data["export_all_resolved_stations"])
        if "blunt_trailing_edge" in export_data:
            config.export.blunt_trailing_edge = bool(export_data["blunt_trailing_edge"])
        if "trailing_edge_height_m" in export_data:
            config.export.trailing_edge_height_m = float(export_data["trailing_edge_height_m"])
        if "out_dir" in export_data:
            config.export.out_dir = str(export_data["out_dir"])
        if "iges_path" in export_data:
            config.export.iges_path = str(export_data["iges_path"])
        if "meshing_iges_path" in export_data:
            config.export.meshing_iges_path = str(export_data["meshing_iges_path"])
        if "frame_only_iges_path" in export_data:
            config.export.frame_only_iges_path = str(export_data["frame_only_iges_path"])
        if "symmetric" in export_data:
            config.export.symmetric = bool(export_data["symmetric"])
        if "tip_style" in export_data:
            config.export.tip_style = str(export_data["tip_style"])
        if "section_curve_n_ctl" in export_data:
            config.export.section_curve_n_ctl = int(export_data["section_curve_n_ctl"])
        if "k_span" in export_data:
            config.export.k_span = int(export_data["k_span"])
        if "include_xy_symmetry_frame" in export_data:
            config.export.include_xy_symmetry_frame = bool(export_data["include_xy_symmetry_frame"])
        if "write_frame_only" in export_data:
            config.export.write_frame_only = bool(export_data["write_frame_only"])
        config.export.metadata.update(
            {
                str(key): value
                for key, value in dict(export_data.get("metadata", {})).items()
            }
        )
    return config


def resolve_geometry(
    component: Wing,
    config: WingGeometryConfig,
) -> PreparedGeometry:
    """Resolve a wing into explicit stations and envelopes."""

    if not component.sections:
        msg = "At least one anchor section is required to resolve the geometry."
        raise ValueError(msg)

    anchor_sections = _sorted_sections(component.sections)
    span_segments = _sorted_spans(component.spans)
    anchor_y = np.array([float(section.spanwise_y_m) for section in anchor_sections], dtype=float)
    custom_station_resolver = _build_custom_station_resolver(
        component,
        anchor_sections,
        config,
    )

    if custom_station_resolver is None:
        anchor_stations = tuple(
            resolve_anchor_section(section, config) for section in anchor_sections
        )
    else:
        anchor_stations = tuple(custom_station_resolver(anchor_y))

    sample_y = _build_sample_y(
        component,
        anchor_sections,
        anchor_y,
        config,
    )
    if custom_station_resolver is None:
        resolved_stations = _resolve_spanwise_stations(
            anchor_sections,
            span_segments,
            sample_y,
            config,
        )
    else:
        resolved_stations = tuple(custom_station_resolver(sample_y))
    envelope = build_envelope(resolved_stations)

    return PreparedGeometry(
        component_name=config.component_name or "wing",
        case_name=config.case_name,
        config=config,
        anchor_stations=anchor_stations,
        resolved_stations=resolved_stations,
        envelope=envelope,
    )


def _build_custom_station_resolver(
    component: Wing,
    anchor_sections: tuple[Section, ...],
    config: WingGeometryConfig,
):
    resolver_factory = config.interpolation.metadata.get("resolved_station_factory")
    if resolver_factory is None:
        return None
    if not callable(resolver_factory):
        msg = "resolved_station_factory must be callable when provided."
        raise TypeError(msg)
    resolver = resolver_factory(
        component=component,
        anchor_sections=anchor_sections,
        config=config,
    )
    if not callable(resolver):
        msg = "resolved_station_factory must return a callable station resolver."
        raise TypeError(msg)
    return resolver


def _build_sample_y(
    component: Wing,
    anchor_sections: tuple[Section, ...],
    anchor_y: np.ndarray,
    config: WingGeometryConfig,
) -> np.ndarray:
    span_station_factory = config.interpolation.metadata.get("span_station_factory")
    if span_station_factory is None:
        return merge_sampled_stations(
            anchor_y,
            config.sampling.spanwise_stations,
            include_anchors=config.sampling.include_anchor_stations,
        )
    if not callable(span_station_factory):
        msg = "span_station_factory must be callable when provided."
        raise TypeError(msg)

    sample_y = np.asarray(
        span_station_factory(
            component=component,
            anchor_sections=anchor_sections,
            config=config,
        ),
        dtype=float,
    )
    if sample_y.ndim != 1 or sample_y.size == 0:
        msg = "span_station_factory must return a non-empty 1D array."
        raise ValueError(msg)
    sample_y = np.unique(np.sort(sample_y))
    if config.sampling.include_anchor_stations:
        sample_y = np.unique(np.concatenate((sample_y, anchor_y)))
    return sample_y.astype(float)


def resolve_component_geometry(
    component: Wing,
) -> tuple[WingGeometryConfig, PreparedGeometry, GeometryMetricSet]:
    """Resolve a component all the way to configuration, geometry and metrics."""

    _validate_geometry_wing(component)
    config = build_geometry_config(component)
    geometry = resolve_geometry(component, config)
    metrics = compute_geometry_metrics(geometry)
    return config, geometry, metrics


def export_geometry_to_pygeo(
    geometry: PreparedGeometry,
) -> PyGeoExportResult:
    """Export one resolved geometry state to pyGeo/IGES artifacts."""

    return export_prepared_geometry_to_pygeo(geometry)


def export_component_geometry_to_pygeo(
    component: Wing,
) -> PyGeoExportResult:
    """Resolve and export one ``Wing`` component using the native pyGeo workflow."""

    geometry = component.geometry_state
    if geometry is None:
        _, geometry, metrics = resolve_component_geometry(component)
        component.geometry_state = geometry
        component.geometry_metrics = metrics
    else:
        geometry.config = build_geometry_config(component)

    export_result = export_geometry_to_pygeo(geometry)
    component.export_state = export_result
    return export_result


def compute_geometry_metrics(geometry: PreparedGeometry) -> GeometryMetricSet:
    """Compute scalar metrics from the prepared geometry state."""

    stations = geometry.resolved_stations
    y = np.array([station.spanwise_y_m for station in stations], dtype=float)
    chord = np.array([station.chord_m for station in stations], dtype=float)
    section_area = np.array([_section_area_m2(station) for station in stations], dtype=float)

    if y.size == 1:
        area_half = 0.0
        volume_half = 0.0
        mac = float(chord[0])
        half_span = 0.0
    else:
        area_half = float(np.trapz(chord, y))
        volume_half = float(np.trapz(section_area, y))
        half_span = float(y[-1] - y[0])
        mac = float(np.trapz(chord**2, y) / area_half) if area_half > 0.0 else float(chord[0])

    multiplier = 2.0 if geometry.config.symmetry else 1.0
    span_m = multiplier * half_span if y.size > 1 else 0.0

    return GeometryMetricSet(
        span_m=span_m,
        planform_area_m2=multiplier * area_half,
        enclosed_volume_m3=multiplier * volume_half,
        root_chord_m=float(stations[0].chord_m),
        tip_chord_m=float(stations[-1].chord_m),
        mean_aerodynamic_chord_m=mac,
        anchor_station_count=len(geometry.anchor_stations),
        resolved_station_count=len(geometry.resolved_stations),
    )


def build_envelope(stations: tuple[ResolvedStation, ...]) -> GeometryEnvelope:
    """Construct a simple geometry envelope from resolved stations."""

    spanwise_y = np.array([station.spanwise_y_m for station in stations], dtype=float)
    upper_z = np.array([np.max(station.upper_surface_xyz_m[:, 2]) for station in stations], dtype=float)
    lower_z = np.array([np.min(station.lower_surface_xyz_m[:, 2]) for station in stations], dtype=float)
    leading_edge_x = np.array([np.min(station.upper_surface_xyz_m[:, 0]) for station in stations], dtype=float)
    trailing_edge_x = np.array([np.max(station.upper_surface_xyz_m[:, 0]) for station in stations], dtype=float)
    return GeometryEnvelope(
        spanwise_y_m=spanwise_y,
        upper_z_m=upper_z,
        lower_z_m=lower_z,
        leading_edge_x_m=leading_edge_x,
        trailing_edge_x_m=trailing_edge_x,
    )


def _resolve_spanwise_stations(
    anchor_sections: tuple[Section, ...],
    span_segments: tuple[Span, ...],
    sample_y: np.ndarray,
    config: WingGeometryConfig,
) -> tuple[ResolvedStation, ...]:
    if len(anchor_sections) <= 1:
        return tuple(resolve_anchor_section(section, config) for section in anchor_sections)

    stations: list[ResolvedStation] = []
    for idx, span in enumerate(span_segments):
        sec_in = anchor_sections[idx]
        sec_out = anchor_sections[idx + 1]
        y0 = float(sec_in.spanwise_y_m)
        y1 = float(sec_out.spanwise_y_m)

        mask = (sample_y >= y0 - 1.0e-12) & (sample_y <= y1 + 1.0e-12)
        local_y = sample_y[mask]
        if local_y.size == 0:
            continue

        local_stations = _resolve_segment_stations(
            anchor_sections,
            sec_in,
            sec_out,
            span,
            local_y,
            config,
        )

        if stations:
            local_stations = local_stations[1:]
        stations.extend(local_stations)

    return tuple(stations)


def _resolve_segment_stations(
    anchor_sections: tuple[Section, ...],
    sec_in: Section,
    sec_out: Section,
    span: Span,
    sample_y: np.ndarray,
    config: WingGeometryConfig,
) -> list[ResolvedStation]:
    chord = _interpolate_section_scalar_field(
        anchor_sections,
        sec_in,
        sec_out,
        span,
        sample_y,
        config,
        field_name="chord",
        getter=lambda section: float(section.chord),
        default_law=config.interpolation.spanwise_law,
    )
    twist = _interpolate_section_scalar_field(
        anchor_sections,
        sec_in,
        sec_out,
        span,
        sample_y,
        config,
        field_name="twist",
        getter=lambda section: float(section.twist),
        default_law=config.interpolation.spanwise_law,
    )
    leading_edge_x, leading_edge_z = _resolve_segment_leading_edge_path(
        anchor_sections,
        sec_in,
        sec_out,
        span,
        sample_y,
        config,
    )
    thickness_factor = _interpolate_airfoil_scalar_field(
        anchor_sections,
        sec_in,
        sec_out,
        span,
        sample_y,
        config,
        field_name="airfoil_thickness_factor",
        getter=lambda airfoil: float(airfoil.thickness_factor),
        default_law=config.interpolation.section_law,
    )
    camber_factor = _interpolate_airfoil_scalar_field(
        anchor_sections,
        sec_in,
        sec_out,
        span,
        sample_y,
        config,
        field_name="airfoil_camber_factor",
        getter=lambda airfoil: float(airfoil.camber_factor),
        default_law=config.interpolation.section_law,
    )
    n1 = _interpolate_airfoil_scalar_field(
        anchor_sections,
        sec_in,
        sec_out,
        span,
        sample_y,
        config,
        field_name="airfoil_n1",
        getter=lambda airfoil: float(airfoil.n1),
        default_law=config.interpolation.section_law,
    )
    n2 = _interpolate_airfoil_scalar_field(
        anchor_sections,
        sec_in,
        sec_out,
        span,
        sample_y,
        config,
        field_name="airfoil_n2",
        getter=lambda airfoil: float(airfoil.n2),
        default_law=config.interpolation.section_law,
    )
    te_thickness = _interpolate_airfoil_scalar_field(
        anchor_sections,
        sec_in,
        sec_out,
        span,
        sample_y,
        config,
        field_name="airfoil_trailing_edge_thickness",
        getter=lambda airfoil: float(airfoil.trailing_edge_thickness),
        default_law=config.interpolation.section_law,
    )
    upper_coefficients = _interpolate_airfoil_coefficients(
        anchor_sections,
        sec_in,
        sec_out,
        span,
        sample_y,
        config,
        field_name="airfoil_upper_coefficients",
        getter=lambda airfoil: airfoil.upper_coefficients,
        default_law=config.interpolation.section_law,
    )
    lower_coefficients = _interpolate_airfoil_coefficients(
        anchor_sections,
        sec_in,
        sec_out,
        span,
        sample_y,
        config,
        field_name="airfoil_lower_coefficients",
        getter=lambda airfoil: airfoil.lower_coefficients,
        default_law=config.interpolation.section_law,
    )

    stations: list[ResolvedStation] = []
    for idx, y_value in enumerate(sample_y):
        if abs(float(y_value) - float(sec_in.spanwise_y_m)) <= 1.0e-12:
            name = sec_in.name
        elif abs(float(y_value) - float(sec_out.spanwise_y_m)) <= 1.0e-12:
            name = sec_out.name
        else:
            name = f"{span.name}_station_{idx:03d}"

        airfoil = AirfoilCST(
            name=f"{name}_airfoil",
            upper_coefficients=tuple(upper_coefficients[idx, :]),
            lower_coefficients=tuple(lower_coefficients[idx, :]),
            n1=float(n1[idx]),
            n2=float(n2[idx]),
            trailing_edge_thickness=float(te_thickness[idx]),
            thickness_factor=float(thickness_factor[idx]),
            camber_factor=float(camber_factor[idx]),
        )
        section = Section(
            name=name,
            airfoil=airfoil,
            chord=float(chord[idx]),
            twist=float(twist[idx]),
            spanwise_y_m=float(y_value),
            leading_edge_x_m=float(leading_edge_x[idx]),
            leading_edge_z_m=float(leading_edge_z[idx]),
            metadata=dict(sec_in.metadata),
        )
        stations.append(resolve_anchor_section(section, config))

    return stations


def _resolve_segment_leading_edge_path(
    anchor_sections: tuple[Section, ...],
    sec_in: Section,
    sec_out: Section,
    span: Span,
    sample_y: np.ndarray,
    config: WingGeometryConfig,
) -> tuple[np.ndarray, np.ndarray]:
    """Resolve the leading-edge path inside one span.

    By default we preserve the classic MADS-like definition given by the section
    anchor positions. A span can optionally switch to an angle-driven law when
    the geometry case explicitly asks for it.
    """

    mode = _leading_edge_mode(span, config)
    if mode == "span_angles":
        y0 = float(sec_in.spanwise_y_m)
        x0 = float(sec_in.leading_edge_x_m or 0.0)
        z0 = float(sec_in.leading_edge_z_m or 0.0)
        delta_y = np.asarray(sample_y, dtype=float) - y0

        sweep_rad = np.radians(float(span.sweep))
        dihed_rad = np.radians(float(span.dihed))
        leading_edge_x = x0 + delta_y * np.tan(sweep_rad)
        leading_edge_z = z0 + delta_y * np.tan(dihed_rad)
        return leading_edge_x, leading_edge_z

    leading_edge_x = _interpolate_section_scalar_field(
        anchor_sections,
        sec_in,
        sec_out,
        span,
        sample_y,
        config,
        field_name="leading_edge_x",
        getter=lambda section: float(section.leading_edge_x_m or 0.0),
        default_law=config.interpolation.spanwise_law,
    )
    leading_edge_z = _interpolate_section_scalar_field(
        anchor_sections,
        sec_in,
        sec_out,
        span,
        sample_y,
        config,
        field_name="leading_edge_z",
        getter=lambda section: float(section.leading_edge_z_m or 0.0),
        default_law=config.interpolation.spanwise_law,
    )
    return leading_edge_x, leading_edge_z


def _leading_edge_mode(
    span: Span,
    config: WingGeometryConfig,
) -> str:
    mode = span.metadata.get("leading_edge_mode")
    if mode is None:
        mode = config.interpolation.metadata.get("leading_edge_mode", "section_positions")
    mode = str(mode).lower()
    if mode not in {"section_positions", "span_angles"}:
        msg = (
            f"Unsupported leading-edge mode '{mode}' for span '{span.name}'. "
            "Use 'section_positions' or 'span_angles'."
        )
        raise ValueError(msg)
    return mode


def _interpolate_coefficients(
    anchor_y: np.ndarray,
    coefficients: list[tuple[float, ...]],
    sample_y: np.ndarray,
    law: str,
) -> np.ndarray:
    lengths = {len(coeff) for coeff in coefficients}
    if not lengths:
        return np.zeros((sample_y.size, 0), dtype=float)
    if len(lengths) != 1:
        msg = "All CST coefficient vectors must have the same length for interpolation."
        raise ValueError(msg)

    coeff_len = next(iter(lengths))
    if coeff_len == 0:
        return np.zeros((sample_y.size, 0), dtype=float)

    values = np.array(coefficients, dtype=float)
    return interpolate_vector_field(anchor_y, values, sample_y, law=law)


def _interpolate_section_scalar_field(
    anchor_sections: tuple[Section, ...],
    sec_in: Section,
    sec_out: Section,
    span: Span,
    sample_y: np.ndarray,
    config: WingGeometryConfig,
    *,
    field_name: str,
    getter,
    default_law: str,
) -> np.ndarray:
    scope = _field_scope(field_name, span, config, default="segment")
    law = _field_law(field_name, span, config, default=default_law)
    sections = anchor_sections if scope == "global" else (sec_in, sec_out)
    anchor_y = np.array([float(section.spanwise_y_m) for section in sections], dtype=float)
    values = np.array([float(getter(section)) for section in sections], dtype=float)
    return interpolate_scalar_field(anchor_y, values, sample_y, law=law)


def _interpolate_airfoil_scalar_field(
    anchor_sections: tuple[Section, ...],
    sec_in: Section,
    sec_out: Section,
    span: Span,
    sample_y: np.ndarray,
    config: WingGeometryConfig,
    *,
    field_name: str,
    getter,
    default_law: str,
) -> np.ndarray:
    scope = _field_scope(field_name, span, config, default="segment")
    law = _field_law(field_name, span, config, default=default_law)
    sections = anchor_sections if scope == "global" else (sec_in, sec_out)
    anchor_y = np.array([float(section.spanwise_y_m) for section in sections], dtype=float)
    values = np.array([float(getter(_require_cst_airfoil(section))) for section in sections], dtype=float)
    return interpolate_scalar_field(anchor_y, values, sample_y, law=law)


def _interpolate_airfoil_coefficients(
    anchor_sections: tuple[Section, ...],
    sec_in: Section,
    sec_out: Section,
    span: Span,
    sample_y: np.ndarray,
    config: WingGeometryConfig,
    *,
    field_name: str,
    getter,
    default_law: str,
) -> np.ndarray:
    scope = _field_scope(field_name, span, config, default="segment")
    law = _field_law(field_name, span, config, default=default_law)
    sections = anchor_sections if scope == "global" else (sec_in, sec_out)
    anchor_y = np.array([float(section.spanwise_y_m) for section in sections], dtype=float)
    coefficients = [tuple(getter(_require_cst_airfoil(section))) for section in sections]
    return _interpolate_coefficients(anchor_y, coefficients, sample_y, law)


def _field_law(
    field_name: str,
    span: Span,
    config: WingGeometryConfig,
    *,
    default: str,
) -> str:
    interpolation_data = span.metadata.get("interpolation", {})
    if isinstance(interpolation_data, dict):
        field_laws = interpolation_data.get("field_laws", {})
        if isinstance(field_laws, dict):
            if field_name in field_laws:
                return str(field_laws[field_name]).lower()
            prefix = _field_prefix(field_name)
            if prefix in field_laws:
                return str(field_laws[prefix]).lower()
    if field_name in config.interpolation.field_laws:
        return str(config.interpolation.field_laws[field_name]).lower()
    prefix = _field_prefix(field_name)
    if prefix in config.interpolation.field_laws:
        return str(config.interpolation.field_laws[prefix]).lower()
    return str(default).lower()


def _field_scope(
    field_name: str,
    span: Span,
    config: WingGeometryConfig,
    *,
    default: str,
) -> str:
    interpolation_data = span.metadata.get("interpolation", {})
    if isinstance(interpolation_data, dict):
        field_scopes = interpolation_data.get("field_scopes", {})
        if isinstance(field_scopes, dict):
            if field_name in field_scopes:
                return _normalize_field_scope(field_scopes[field_name], field_name, span.name)
            prefix = _field_prefix(field_name)
            if prefix in field_scopes:
                return _normalize_field_scope(field_scopes[prefix], field_name, span.name)
    if field_name in config.interpolation.field_scopes:
        return _normalize_field_scope(config.interpolation.field_scopes[field_name], field_name, span.name)
    prefix = _field_prefix(field_name)
    if prefix in config.interpolation.field_scopes:
        return _normalize_field_scope(config.interpolation.field_scopes[prefix], field_name, span.name)
    return _normalize_field_scope(default, field_name, span.name)


def _field_prefix(field_name: str) -> str:
    if field_name.startswith("airfoil_"):
        return "airfoil"
    return field_name


def _normalize_field_scope(
    scope: object,
    field_name: str,
    span_name: str,
) -> str:
    normalized = str(scope).lower()
    if normalized not in {"segment", "global"}:
        msg = (
            f"Unsupported interpolation scope '{scope}' for field '{field_name}' "
            f"in span '{span_name}'. Use 'segment' or 'global'."
        )
        raise ValueError(msg)
    return normalized


def _require_cst_airfoil(
    section: Section,
) -> AirfoilCST:
    airfoil = section.airfoil
    if not isinstance(airfoil, AirfoilCST):
        typename = type(airfoil).__name__ if airfoil is not None else "None"
        msg = (
            f"Anchor section '{section.name}' must use AirfoilCST for the current "
            f"CST geometry solver, got '{typename}'."
        )
        raise TypeError(msg)
    return airfoil


def _validate_geometry_wing(component: Wing) -> None:
    if not component.sections:
        msg = f"Wing '{component.name}' does not define any anchor sections."
        raise ValueError(msg)

    flattened = flatten_components([component])
    _ = flattened  # Explicitly trigger the recursive structure walk during validation.
    config = build_geometry_config(component)
    uses_custom_resolver = config.interpolation.metadata.get("resolved_station_factory") is not None

    for section in component.sections:
        if not isinstance(section, Section):
            msg = (
                f"Wing '{component.name}' contains a non-Section "
                f"anchor entry of type '{type(section).__name__}'."
            )
            raise TypeError(msg)
        if section.spanwise_y_m is None:
            msg = (
                f"Section '{section.name}' of wing '{component.name}' "
                "must define spanwise_y_m."
            )
            raise ValueError(msg)
        if section.airfoil is None:
            msg = (
                f"Anchor section '{section.name}' of wing '{component.name}' "
                "must define an airfoil."
            )
            raise ValueError(msg)
        if (not uses_custom_resolver) and (not isinstance(section.airfoil, AirfoilCST)):
            typename = type(section.airfoil).__name__
            msg = (
                f"Anchor section '{section.name}' of wing '{component.name}' "
                f"must use AirfoilCST, got '{typename}'."
            )
            raise TypeError(msg)

    for segment in component.spans:
        if not isinstance(segment, Span):
            msg = (
                f"Wing '{component.name}' contains a non-Span "
                f"segment entry of type '{type(segment).__name__}'."
            )
            raise TypeError(msg)
        if segment.start_y_m is None or segment.end_y_m is None:
            msg = (
                f"Span '{segment.name}' of wing '{component.name}' "
                "must define start_y_m and end_y_m."
            )
            raise ValueError(msg)
        if abs(float(segment.length) - (float(segment.end_y_m) - float(segment.start_y_m))) > 1.0e-9:
            msg = (
                f"Span '{segment.name}' of wing '{component.name}' has "
                "an inconsistent length with respect to start_y_m and end_y_m."
            )
            raise ValueError(msg)

    sections_sorted = _sorted_sections(component.sections)
    spans_sorted = _sorted_spans(component.spans)
    if len(spans_sorted) != max(len(sections_sorted) - 1, 0):
        msg = (
            f"Wing '{component.name}' must define exactly len(sections)-1 spans. "
            f"Got {len(sections_sorted)} sections and {len(spans_sorted)} spans."
        )
        raise ValueError(msg)

    for idx, span in enumerate(spans_sorted):
        sec_in = sections_sorted[idx]
        sec_out = sections_sorted[idx + 1]
        if abs(float(span.start_y_m) - float(sec_in.spanwise_y_m)) > 1.0e-9:
            msg = (
                f"Span '{span.name}' start_y_m does not match section '{sec_in.name}' "
                f"spanwise_y_m in wing '{component.name}'."
            )
            raise ValueError(msg)
        if abs(float(span.end_y_m) - float(sec_out.spanwise_y_m)) > 1.0e-9:
            msg = (
                f"Span '{span.name}' end_y_m does not match section '{sec_out.name}' "
                f"spanwise_y_m in wing '{component.name}'."
            )
            raise ValueError(msg)
        if _leading_edge_mode(span, config) == "span_angles":
            expected_x_end, expected_z_end = _resolve_segment_leading_edge_path(
                sections_sorted,
                sec_in,
                sec_out,
                span,
                np.array([float(span.end_y_m)], dtype=float),
                config,
            )
            if abs(float(sec_out.leading_edge_x_m or 0.0) - float(expected_x_end[0])) > 1.0e-6:
                msg = (
                    f"Span '{span.name}' sweep is inconsistent with the leading-edge x "
                    f"position of section '{sec_out.name}' in wing "
                    f"'{component.name}'."
                )
                raise ValueError(msg)
            if abs(float(sec_out.leading_edge_z_m or 0.0) - float(expected_z_end[0])) > 1.0e-6:
                msg = (
                    f"Span '{span.name}' dihed is inconsistent with the leading-edge z "
                    f"position of section '{sec_out.name}' in wing "
                    f"'{component.name}'."
                )
                raise ValueError(msg)


def _section_area_m2(station: ResolvedStation) -> float:
    thickness = station.upper_z_over_c - station.lower_z_over_c
    normalized_area = float(np.trapz(thickness, station.x_over_c))
    return normalized_area * station.chord_m**2


def _sorted_sections(sections: list[Section] | tuple[Section, ...]) -> tuple[Section, ...]:
    return tuple(sorted(sections, key=lambda sec: float(sec.spanwise_y_m)))


def _sorted_spans(spans: list[Span] | tuple[Span, ...]) -> tuple[Span, ...]:
    return tuple(sorted(spans, key=lambda span: float(span.start_y_m)))


__all__ = [
    "build_control_point_planform",
    "build_envelope",
    "build_geometry_config",
    "compute_geometry_metrics",
    "export_component_geometry_to_pygeo",
    "export_geometry_to_pygeo",
    "GeometryEnvelope",
    "GeometryMetricSet",
    "Options",
    "PreparedGeometry",
    "PyGeoExportOptions",
    "PyGeoExportResult",
    "ResolvedStation",
    "resolve_anchor_section",
    "resolve_component_geometry",
    "resolve_geometry",
    "WingGeometryConfig",
]
