"""Internal volume indicator-surface checks for resolved MADS geometries."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from multiads.solvers.synthesis.geometry import PreparedGeometry, ResolvedStation


@dataclass(frozen=True, slots=True)
class CadReferenceFrame:
    """Mapping from a CAD-like packaging frame to the MADS model frame."""

    name: str = "model"
    offset_x_m: float = 0.0
    offset_y_m: float = 0.0
    offset_z_m: float = 0.0
    mirror_about_symmetry_plane: bool = True

    def xy_to_model(self, x_m, y_m) -> tuple[np.ndarray, np.ndarray]:  # noqa: ANN001
        x_model = np.asarray(x_m, dtype=float) - float(self.offset_x_m)
        y_model = np.asarray(y_m, dtype=float) - float(self.offset_y_m)
        if self.mirror_about_symmetry_plane:
            y_model = np.abs(y_model)
        return x_model, y_model

    def z_to_cad(self, z_model_m) -> np.ndarray:  # noqa: ANN001
        return np.asarray(z_model_m, dtype=float) + float(self.offset_z_m)


@dataclass(frozen=True, slots=True)
class IndicatorSurfaceSpec:
    """One required upper or lower indicator surface for an internal volume."""

    category: str
    sub_category: str
    sense: str
    vertices_xyz_m: np.ndarray
    minimum_clearance_m: np.ndarray

    def __post_init__(self) -> None:
        vertices = np.asarray(self.vertices_xyz_m, dtype=float)
        clearance = np.asarray(self.minimum_clearance_m, dtype=float)
        if vertices.ndim != 2 or vertices.shape[1] != 3:
            msg = f"vertices_xyz_m must have shape (N, 3), got {vertices.shape}"
            raise ValueError(msg)
        if vertices.shape[0] < 3:
            msg = "indicator surfaces must define at least 3 vertices"
            raise ValueError(msg)
        if clearance.shape != (vertices.shape[0],):
            msg = (
                "minimum_clearance_m must have one value per vertex, "
                f"got shape {clearance.shape} for {vertices.shape[0]} vertices"
            )
            raise ValueError(msg)
        if self.sense not in {"upper", "lower"}:
            msg = f"sense must be 'upper' or 'lower', got {self.sense!r}"
            raise ValueError(msg)
        object.__setattr__(self, "vertices_xyz_m", vertices)
        object.__setattr__(self, "minimum_clearance_m", clearance)

    @property
    def label(self) -> str:
        return f"{self.category}:{self.sub_category}"

    @property
    def polygon_xy_m(self) -> np.ndarray:
        return np.asarray(self.vertices_xyz_m[:, :2], dtype=float)


@dataclass(frozen=True, slots=True)
class InternalVolumeConstraintSet:
    """Collection of internal volume indicator surfaces."""

    name: str
    source_path: Path
    reference_frame: CadReferenceFrame
    surfaces: tuple[IndicatorSurfaceSpec, ...]


@dataclass(frozen=True, slots=True)
class IndicatorSurfaceResult:
    """Fit result for one indicator surface."""

    label: str
    category: str
    sub_category: str
    sense: str
    satisfied: bool
    sample_count: int
    invalid_sample_count: int
    footprint_area_m2: float
    clearance_volume_m3: float
    worst_margin_m: float
    mean_margin_m: float
    critical_x_m: float
    critical_y_m: float
    critical_target_z_m: float
    critical_geometry_z_m: float
    critical_clearance_m: float
    vertex_margins_m: tuple[float, ...]


@dataclass(frozen=True, slots=True)
class InternalVolumeConstraintResult:
    """Fit result for all internal volume indicator surfaces."""

    name: str
    source_path: Path
    reference_frame: CadReferenceFrame
    satisfied: bool
    minimum_margin_m: float
    surface_results: tuple[IndicatorSurfaceResult, ...]


def load_internal_volume_constraint_set(
    csv_path: str | Path,
    *,
    reference_frame: CadReferenceFrame | None = None,
    name: str | None = None,
) -> InternalVolumeConstraintSet:
    """Load internal volume indicator surfaces from the legacy CSV format."""

    path = Path(csv_path)
    rows = list(csv.reader(path.open(encoding="utf-8")))
    surfaces: list[IndicatorSurfaceSpec] = []
    idx = 1

    while idx < len(rows):
        row = rows[idx]
        if len(row) >= 3 and row[0] and row[1] and row[2] in {"upper", "lower"}:
            category = str(row[0]).strip()
            sub_category = str(row[1]).strip()
            sense = str(row[2]).strip()
            idx += 1
            if idx >= len(rows):
                break
            header = rows[idx]
            if len(header) < 4 or header[0] != "X" or header[1] != "Y" or header[2] != "Z":
                msg = f"Unexpected indicator-surface header at row {idx + 1}: {header}"
                raise ValueError(msg)
            idx += 1
            vertices: list[tuple[float, float, float]] = []
            clearance: list[float] = []
            while idx < len(rows):
                point_row = rows[idx]
                if not any(str(value).strip() for value in point_row):
                    break
                vertices.append((float(point_row[0]), float(point_row[1]), float(point_row[2])))
                clearance.append(
                    0.0 if len(point_row) < 4 or point_row[3] == "" else float(point_row[3])
                )
                idx += 1

            surfaces.append(
                IndicatorSurfaceSpec(
                    category=category,
                    sub_category=sub_category,
                    sense=sense,
                    vertices_xyz_m=np.asarray(vertices, dtype=float),
                    minimum_clearance_m=np.asarray(clearance, dtype=float),
                )
            )
        idx += 1

    return InternalVolumeConstraintSet(
        name=path.stem if name is None else str(name),
        source_path=path,
        reference_frame=reference_frame or CadReferenceFrame(),
        surfaces=tuple(surfaces),
    )


def evaluate_internal_volume_constraints(
    geometry: PreparedGeometry,
    constraint_set: InternalVolumeConstraintSet,
    *,
    triangle_resolution: int = 18,
) -> InternalVolumeConstraintResult:
    """Evaluate whether all indicator surfaces fit inside a resolved geometry."""

    evaluator = GeometryEnvelopeEvaluator(geometry, constraint_set.reference_frame)
    surface_results = tuple(
        _evaluate_indicator_surface(surface, evaluator, triangle_resolution=triangle_resolution)
        for surface in constraint_set.surfaces
    )
    minimum_margin = min((result.worst_margin_m for result in surface_results), default=float("nan"))
    return InternalVolumeConstraintResult(
        name=constraint_set.name,
        source_path=constraint_set.source_path,
        reference_frame=constraint_set.reference_frame,
        satisfied=all(result.satisfied for result in surface_results),
        minimum_margin_m=float(minimum_margin),
        surface_results=surface_results,
    )


class GeometryEnvelopeEvaluator:
    """Interpolate upper/lower geometry envelopes from resolved CTA stations."""

    def __init__(
        self,
        geometry: PreparedGeometry,
        reference_frame: CadReferenceFrame | None = None,
    ) -> None:
        self.geometry = geometry
        self.reference_frame = reference_frame or CadReferenceFrame()
        self._stations = tuple(sorted(geometry.resolved_stations, key=lambda station: station.spanwise_y_m))
        if not self._stations:
            msg = "Resolved geometry must contain at least one station."
            raise ValueError(msg)
        self._station_y = np.asarray([station.spanwise_y_m for station in self._stations], dtype=float)
        self._curve_cache: dict[tuple[str, float], tuple[np.ndarray, np.ndarray]] = {}

    def evaluate_points(
        self,
        x_cad_m: np.ndarray,
        y_cad_m: np.ndarray,
        sense: str,
    ) -> np.ndarray:
        x_model, y_model = self.reference_frame.xy_to_model(x_cad_m, y_cad_m)
        output = np.empty_like(np.asarray(x_model, dtype=float), dtype=float)
        flat_x = np.ravel(x_model)
        flat_y = np.ravel(y_model)
        flat_output = np.empty_like(flat_x, dtype=float)
        for idx, (xx, yy) in enumerate(zip(flat_x, flat_y, strict=True)):
            z_model = self._surface_z_at_xy(float(xx), float(yy), sense)
            flat_output[idx] = float(self.reference_frame.z_to_cad(z_model))
        output[:] = flat_output.reshape(output.shape)
        return output

    def _surface_z_at_xy(self, x_m: float, y_m: float, sense: str) -> float:
        if sense not in {"upper", "lower"}:
            msg = f"Unknown surface sense {sense!r}."
            raise ValueError(msg)
        y_min = float(self._station_y[0])
        y_max = float(self._station_y[-1])
        if y_m < y_min - 1.0e-12 or y_m > y_max + 1.0e-12:
            return float("nan")

        idx = int(np.searchsorted(self._station_y, y_m))
        if idx == 0:
            return self._station_surface_z(self._stations[0], x_m, sense)
        if idx >= self._station_y.size:
            return self._station_surface_z(self._stations[-1], x_m, sense)

        y0 = float(self._station_y[idx - 1])
        y1 = float(self._station_y[idx])
        z0 = self._station_surface_z(self._stations[idx - 1], x_m, sense)
        z1 = self._station_surface_z(self._stations[idx], x_m, sense)
        if not np.isfinite(z0) or not np.isfinite(z1):
            return float("nan")
        weight = (float(y_m) - y0) / max(y1 - y0, 1.0e-12)
        return float((1.0 - weight) * z0 + weight * z1)

    def _station_surface_z(self, station: ResolvedStation, x_m: float, sense: str) -> float:
        cache_key = (sense, round(float(station.spanwise_y_m), 9))
        if cache_key not in self._curve_cache:
            surface = station.upper_surface_xyz_m if sense == "upper" else station.lower_surface_xyz_m
            self._curve_cache[cache_key] = _sorted_unique_curve(surface[:, 0], surface[:, 2])
        x_sorted, z_sorted = self._curve_cache[cache_key]
        return float(np.interp(float(x_m), x_sorted, z_sorted, left=np.nan, right=np.nan))


def _sorted_unique_curve(x_m: np.ndarray, z_m: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    order = np.argsort(np.asarray(x_m, dtype=float))
    x_sorted = np.asarray(x_m, dtype=float)[order]
    z_sorted = np.asarray(z_m, dtype=float)[order]
    unique_x, inverse = np.unique(x_sorted, return_inverse=True)
    if unique_x.size == x_sorted.size:
        return x_sorted, z_sorted
    z_unique = np.zeros_like(unique_x, dtype=float)
    counts = np.zeros_like(unique_x, dtype=float)
    np.add.at(z_unique, inverse, z_sorted)
    np.add.at(counts, inverse, 1.0)
    return unique_x, z_unique / np.maximum(counts, 1.0)


def _polygon_area_xy(vertices_xy_m: np.ndarray) -> float:
    xy = np.asarray(vertices_xy_m, dtype=float)
    x = xy[:, 0]
    y = xy[:, 1]
    return float(
        0.5
        * abs(
            np.dot(x, np.roll(y, -1))
            - np.dot(y, np.roll(x, -1)),
        )
    )


def _evaluate_indicator_surface(
    surface: IndicatorSurfaceSpec,
    evaluator: GeometryEnvelopeEvaluator,
    *,
    triangle_resolution: int,
) -> IndicatorSurfaceResult:
    samples_xyzc, _ = _sample_indicator_surface(surface, triangle_resolution)
    x_samples = samples_xyzc[:, 0]
    y_samples = samples_xyzc[:, 1]
    z_target = samples_xyzc[:, 2]
    clearance = samples_xyzc[:, 3]
    z_geometry = evaluator.evaluate_points(x_samples, y_samples, surface.sense)

    if surface.sense == "upper":
        required_z = z_target + clearance
        margin = z_geometry - required_z
    else:
        required_z = z_target - clearance
        margin = required_z - z_geometry

    invalid_mask = ~np.isfinite(z_geometry)
    if np.any(invalid_mask):
        margin = margin.copy()
        margin[invalid_mask] = -np.inf

    critical_idx = int(np.argmin(margin))
    valid_mask = np.isfinite(margin)
    footprint_area = _polygon_area_xy(surface.polygon_xy_m)
    if np.any(valid_mask):
        worst_margin = float(np.min(margin[valid_mask]))
        mean_margin = float(np.mean(margin[valid_mask]))
        clearance_volume = mean_margin * footprint_area
        satisfied = bool(np.all(margin[valid_mask] >= 0.0) and not np.any(invalid_mask))
    else:
        worst_margin = float("-inf")
        mean_margin = float("nan")
        clearance_volume = float("-inf")
        satisfied = False

    verts = surface.vertices_xyz_m
    z_verts = evaluator.evaluate_points(verts[:, 0], verts[:, 1], surface.sense)
    if surface.sense == "upper":
        v_margin = z_verts - (verts[:, 2] + surface.minimum_clearance_m)
    else:
        v_margin = (verts[:, 2] - surface.minimum_clearance_m) - z_verts
    vertex_margins_m = tuple(float(m) for m in v_margin)

    return IndicatorSurfaceResult(
        label=surface.label,
        category=surface.category,
        sub_category=surface.sub_category,
        sense=surface.sense,
        satisfied=satisfied,
        sample_count=int(margin.size),
        invalid_sample_count=int(np.count_nonzero(invalid_mask)),
        footprint_area_m2=float(footprint_area),
        clearance_volume_m3=float(clearance_volume),
        worst_margin_m=worst_margin,
        mean_margin_m=mean_margin,
        critical_x_m=float(x_samples[critical_idx]),
        critical_y_m=float(y_samples[critical_idx]),
        critical_target_z_m=float(required_z[critical_idx]),
        critical_geometry_z_m=float(z_geometry[critical_idx]),
        critical_clearance_m=float(clearance[critical_idx]),
        vertex_margins_m=vertex_margins_m,
    )


def _sample_indicator_surface(
    surface: IndicatorSurfaceSpec,
    resolution: int,
) -> tuple[np.ndarray, np.ndarray]:
    vertices = np.asarray(surface.vertices_xyz_m, dtype=float)
    clearance = np.asarray(surface.minimum_clearance_m, dtype=float)
    sample_xyz = []
    sample_weights = []
    for idx in range(1, vertices.shape[0] - 1):
        tri_vertices = np.vstack([vertices[0], vertices[idx], vertices[idx + 1]])
        tri_clearance = np.asarray([clearance[0], clearance[idx], clearance[idx + 1]], dtype=float)
        tri_samples, tri_clearance_samples, tri_weights = _triangle_sample_points(
            tri_vertices,
            tri_clearance,
            resolution,
        )
        sample_xyz.append(
            np.column_stack(
                [
                    tri_samples[:, 0],
                    tri_samples[:, 1],
                    tri_samples[:, 2],
                    tri_clearance_samples,
                ]
            )
        )
        sample_weights.append(tri_weights)
    if not sample_xyz:
        msg = f"Could not triangulate indicator surface {surface.label}."
        raise ValueError(msg)
    return np.vstack(sample_xyz), np.concatenate(sample_weights)


def _triangle_sample_points(
    vertices_xyz_m: np.ndarray,
    clearance_m: np.ndarray,
    resolution: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    resolution = max(int(resolution), 1)
    xyz_samples = []
    clearance_samples = []
    weights = []
    for i in range(resolution + 1):
        for j in range(resolution + 1 - i):
            w0 = float(i) / float(resolution)
            w1 = float(j) / float(resolution)
            w2 = 1.0 - w0 - w1
            bary = np.asarray([w0, w1, w2], dtype=float)
            xyz_samples.append(bary @ np.asarray(vertices_xyz_m, dtype=float))
            clearance_samples.append(float(bary @ np.asarray(clearance_m, dtype=float)))
            weights.append(1.0)
    return (
        np.asarray(xyz_samples, dtype=float),
        np.asarray(clearance_samples, dtype=float),
        np.asarray(weights, dtype=float),
    )


def _triangle_barycentric_coordinates_xy(
    x_m: float,
    y_m: float,
    triangle_xy_m: np.ndarray,
) -> tuple[float, float, float] | None:
    tri = np.asarray(triangle_xy_m, dtype=float)
    x0, y0 = tri[0]
    x1, y1 = tri[1]
    x2, y2 = tri[2]
    det = (y1 - y2) * (x0 - x2) + (x2 - x1) * (y0 - y2)
    if abs(float(det)) <= 1.0e-14:
        return None
    l0 = ((y1 - y2) * (float(x_m) - x2) + (x2 - x1) * (float(y_m) - y2)) / det
    l1 = ((y2 - y0) * (float(x_m) - x2) + (x0 - x2) * (float(y_m) - y2)) / det
    l2 = 1.0 - l0 - l1
    tol = 1.0e-10
    if l0 < -tol or l1 < -tol or l2 < -tol:
        return None
    return float(l0), float(l1), float(l2)


def _surface_required_z_at_points(
    surface: IndicatorSurfaceSpec,
    x_m: np.ndarray,
    y_m: np.ndarray,
) -> np.ndarray:
    x_arr = np.asarray(x_m, dtype=float)
    y_arr = np.asarray(y_m, dtype=float)
    required = np.full(x_arr.shape, np.nan, dtype=float)
    vertices = np.asarray(surface.vertices_xyz_m, dtype=float)
    clearance = np.asarray(surface.minimum_clearance_m, dtype=float)

    for tri_idx in range(1, vertices.shape[0] - 1):
        tri_vertices = np.vstack([vertices[0], vertices[tri_idx], vertices[tri_idx + 1]])
        tri_xy = tri_vertices[:, :2]
        tri_z = tri_vertices[:, 2]
        tri_clearance = np.asarray(
            [clearance[0], clearance[tri_idx], clearance[tri_idx + 1]],
            dtype=float,
        )
        for idx, (xx, yy) in enumerate(zip(np.ravel(x_arr), np.ravel(y_arr), strict=True)):
            flat_idx = np.unravel_index(idx, x_arr.shape)
            if np.isfinite(required[flat_idx]):
                continue
            bary = _triangle_barycentric_coordinates_xy(float(xx), float(yy), tri_xy)
            if bary is None:
                continue
            weights = np.asarray(bary, dtype=float)
            z_target = float(weights @ tri_z)
            local_clearance = float(weights @ tri_clearance)
            if surface.sense == "upper":
                required[flat_idx] = z_target + local_clearance
            else:
                required[flat_idx] = z_target - local_clearance
    return required


def required_constraint_bounds_at_points(
    surfaces: tuple[IndicatorSurfaceSpec, ...] | list[IndicatorSurfaceSpec],
    x_m: np.ndarray,
    y_m: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return required upper/lower CAD-frame bounds at arbitrary XY points."""

    x_arr = np.asarray(x_m, dtype=float)
    y_arr = np.asarray(y_m, dtype=float)
    upper_required = np.full(x_arr.shape, -np.inf, dtype=float)
    lower_required = np.full(x_arr.shape, np.inf, dtype=float)
    has_upper = np.zeros(x_arr.shape, dtype=bool)
    has_lower = np.zeros(x_arr.shape, dtype=bool)

    for surface in tuple(surfaces):
        required = _surface_required_z_at_points(surface, x_arr, y_arr)
        valid = np.isfinite(required)
        if not np.any(valid):
            continue
        if surface.sense == "upper":
            upper_required[valid] = np.maximum(upper_required[valid], required[valid])
            has_upper |= valid
        else:
            lower_required[valid] = np.minimum(lower_required[valid], required[valid])
            has_lower |= valid

    upper_required[~has_upper] = np.nan
    lower_required[~has_lower] = np.nan
    return upper_required, lower_required
