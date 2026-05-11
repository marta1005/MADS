"""Private geometry types and configuration containers for synthesis geometry."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np
from numpy.typing import NDArray


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
