"""pyGeo export helpers for resolved wing geometry workflows."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from multiads.utilities.pyspline_loader import ensure_local_dependency_paths, load_pygeo_class, patch_pyspline_for_pygeo

if TYPE_CHECKING:
    from multiads.solvers.synthesis.geometry_lib import PreparedGeometry, PyGeoExportResult, ResolvedStation


def export_prepared_geometry_to_pygeo(geometry: PreparedGeometry) -> PyGeoExportResult:
    """Export a resolved geometry state to pyGeo/IGES artifacts."""
    from multiads.solvers.synthesis.geometry_lib import PyGeoExportResult

    export_cfg = geometry.config.export
    profiles_dir, iges_path, meshing_iges_path, frame_only_iges_path = _resolve_export_paths(geometry)
    profiles_dir.mkdir(parents=True, exist_ok=True)

    mode = _airfoil_distribution_mode(geometry)
    airfoil_paths = _write_station_airfoils(profiles_dir, geometry, mode=mode)
    te_height_scaled = _build_te_height_scaled(geometry.resolved_stations)

    surface = _build_pygeo_surface(geometry, airfoil_paths, te_height_scaled=te_height_scaled)
    if iges_path is not None:
        iges_path.parent.mkdir(parents=True, exist_ok=True)
        surface.writeIGES(str(iges_path))

    frame_kwargs = dict(export_cfg.metadata.get("xy_frame", {}))

    frame_surfaces = (
        build_xy_symmetry_frame_surfaces(geometry, **frame_kwargs)
        if export_cfg.include_xy_symmetry_frame
        or meshing_iges_path is not None
        or export_cfg.write_frame_only
        or frame_only_iges_path is not None
        else None
    )

    meshing_surface = None
    if export_cfg.include_xy_symmetry_frame or meshing_iges_path is not None:
        meshing_surface = _build_pygeo_surface(geometry, airfoil_paths, te_height_scaled=te_height_scaled)
        append_xy_symmetry_frame_surfaces(meshing_surface, frame_surfaces=frame_surfaces)
        if meshing_iges_path is not None:
            meshing_iges_path.parent.mkdir(parents=True, exist_ok=True)
            meshing_surface.writeIGES(str(meshing_iges_path))

    frame_only_surface = None
    if export_cfg.write_frame_only or frame_only_iges_path is not None:
        frame_only_surface = _build_pygeo_surface(geometry, airfoil_paths, te_height_scaled=te_height_scaled)
        frame_only_surface.surfs = list(frame_surfaces or ())
        frame_only_surface.nSurf = len(frame_only_surface.surfs)
        if frame_only_iges_path is not None:
            frame_only_iges_path.parent.mkdir(parents=True, exist_ok=True)
            frame_only_surface.writeIGES(str(frame_only_iges_path))

    stations = geometry.resolved_stations
    return PyGeoExportResult(
        station_names=tuple(station.name for station in stations),
        station_y_m=np.array([station.spanwise_y_m for station in stations], dtype=float),
        airfoil_dat_paths=tuple(airfoil_paths),
        surface=surface,
        meshing_surface=meshing_surface,
        frame_only_surface=frame_only_surface,
        profiles_dir=str(profiles_dir),
        iges_path=None if iges_path is None else str(iges_path),
        meshing_iges_path=None if meshing_iges_path is None else str(meshing_iges_path),
        frame_only_iges_path=None if frame_only_iges_path is None else str(frame_only_iges_path),
        metadata={
            "airfoil_distribution_mode": mode,
            "pygeo_mode": str(export_cfg.pygeo_mode),
            "symmetric": bool(export_cfg.symmetric),
            "include_xy_symmetry_frame": bool(export_cfg.include_xy_symmetry_frame),
            "write_frame_only": bool(export_cfg.write_frame_only),
        },
    )


def _resolve_export_paths(
    geometry: PreparedGeometry,
) -> tuple[Path, Path | None, Path | None, Path | None]:
    export_cfg = geometry.config.export
    component_name = geometry.component_name or "wing"

    default_base_dir = Path.cwd() / "outputs" / component_name
    iges_path = _path_or_none(export_cfg.iges_path)
    if iges_path is not None:
        default_base_dir = iges_path.parent

    profiles_dir = _path_or_none(export_cfg.out_dir)
    if profiles_dir is None:
        profiles_dir = default_base_dir / "station_airfoils"

    meshing_iges_path = _path_or_none(export_cfg.meshing_iges_path)
    frame_only_iges_path = _path_or_none(export_cfg.frame_only_iges_path)

    if iges_path is None:
        iges_path = default_base_dir / f"{component_name}.igs"
    if export_cfg.include_xy_symmetry_frame and meshing_iges_path is None:
        meshing_iges_path = default_base_dir / f"{component_name}_meshing_xy_frame.igs"
    if export_cfg.write_frame_only and frame_only_iges_path is None:
        frame_only_iges_path = default_base_dir / f"{component_name}_xy_frame_only.igs"

    return profiles_dir, iges_path, meshing_iges_path, frame_only_iges_path


def _path_or_none(value: str | Path | None) -> Path | None:
    if value in (None, ""):
        return None
    return Path(value).expanduser().resolve()


def _airfoil_distribution_mode(geometry: PreparedGeometry) -> str:
    if geometry.config.export.export_all_resolved_stations:
        return "all"
    return str(geometry.config.sampling.airfoil_distribution_mode).lower()


def _write_station_airfoils(
    profiles_dir: Path,
    geometry: PreparedGeometry,
    *,
    mode: str,
) -> list[str | None]:
    _clean_generated_airfoil_files(profiles_dir)

    if mode == "all":
        airfoil_paths: list[str | None] = []
        for idx, station in enumerate(geometry.resolved_stations):
            path = profiles_dir / f"station_{idx:03d}_y{station.spanwise_y_m:.3f}.dat"
            _write_airfoil_dat(
                path,
                station.x_over_c,
                station.upper_z_over_c,
                station.lower_z_over_c,
                name=f"CST_STATION_{idx:03d}",
            )
            airfoil_paths.append(str(path))
        return airfoil_paths

    if mode != "anchors":
        msg = f"Unsupported airfoil_distribution_mode '{mode}'. Use 'anchors' or 'all'."
        raise ValueError(msg)

    airfoil_paths = [None] * len(geometry.resolved_stations)
    for idx, anchor_station in enumerate(geometry.anchor_stations):
        station_idx = _station_index_at_y(geometry.resolved_stations, anchor_station.spanwise_y_m)
        path = profiles_dir / f"anchor_{idx:02d}_y{anchor_station.spanwise_y_m:.3f}.dat"
        _write_airfoil_dat(
            path,
            anchor_station.x_over_c,
            anchor_station.upper_z_over_c,
            anchor_station.lower_z_over_c,
            name=f"CST_ANCHOR_{idx:02d}",
        )
        airfoil_paths[station_idx] = str(path)
    return airfoil_paths


def _clean_generated_airfoil_files(profiles_dir: Path) -> None:
    for pattern in ("station_*.dat", "anchor_*.dat"):
        for path in profiles_dir.glob(pattern):
            if path.is_file():
                path.unlink()


def _write_airfoil_dat(
    path: Path,
    x_over_c: np.ndarray,
    upper_z_over_c: np.ndarray,
    lower_z_over_c: np.ndarray,
    *,
    name: str,
) -> None:
    x = np.asarray(x_over_c, dtype=float)
    yu = np.asarray(upper_z_over_c, dtype=float)
    yl = np.asarray(lower_z_over_c, dtype=float)

    xu = x[::-1]
    zu = yu[::-1]
    xl = x[1:]
    zl = yl[1:]
    points = np.vstack((np.column_stack((xu, zu)), np.column_stack((xl, zl))))

    with path.open("w", encoding="utf-8") as stream:
        stream.write(f"{name}\n")
        for xx, zz in points:
            stream.write(f"{xx:.10f} {zz:.10f}\n")


def _build_te_height_scaled(stations: tuple[ResolvedStation, ...]) -> np.ndarray:
    return np.array(
        [
            float(
                station.metadata.get(
                    "pygeo_te_height_scaled",
                    station.upper_z_over_c[-1] - station.lower_z_over_c[-1],
                )
            )
            for station in stations
        ],
        dtype=float,
    )


def _build_pygeo_surface(
    geometry: PreparedGeometry,
    airfoil_paths: list[str | None],
    *,
    te_height_scaled: np.ndarray | None = None,
):
    export_cfg = geometry.config.export
    try:
        ensure_local_dependency_paths()
        pygeo_class = load_pygeo_class()
    except Exception as exc:
        detail = ""
        exc_text = str(exc)
        if "invalid ELF header" in exc_text or "Mach-O" in exc_text:
            detail = (
                " Detected platform mismatch in local binaries (for example macOS binaries used in Linux)."
            )
        raise RuntimeError(
            "pyGeo export is unavailable because the local pygeo/pyspline stack could not be imported. "
            "The geometry is resolved, but IGES export needs a working pyGeo installation."
            f"{detail}"
        ) from exc

    stations = geometry.resolved_stations
    xsections = list(airfoil_paths)
    scale = np.array([station.chord_m for station in stations], dtype=float)
    offset = np.zeros((len(stations), 2), dtype=float)
    x = np.array([station.leading_edge_x_m for station in stations], dtype=float)
    y = np.array([station.leading_edge_z_m for station in stations], dtype=float)
    z = np.array([station.spanwise_y_m for station in stations], dtype=float)
    rot_z = np.array([station.twist_deg for station in stations], dtype=float)
    te_height_local = None if te_height_scaled is None else np.asarray(te_height_scaled, dtype=float)

    if export_cfg.symmetric:
        mirror_slice = slice(1, None)
        xsections = list(xsections[mirror_slice][::-1]) + list(xsections)
        scale = np.concatenate((scale[mirror_slice][::-1], scale), axis=0)
        offset = np.vstack((offset[mirror_slice][::-1], offset))
        x = np.concatenate((x[mirror_slice][::-1], x), axis=0)
        y = np.concatenate((y[mirror_slice][::-1], y), axis=0)
        z = np.concatenate((-z[mirror_slice][::-1], z), axis=0)
        rot_z = np.concatenate((rot_z[mirror_slice][::-1], rot_z), axis=0)
        if te_height_local is not None:
            te_height_local = np.concatenate(
                (te_height_local[mirror_slice][::-1], te_height_local),
                axis=0,
            )

    kwargs = {}
    if export_cfg.blunt_trailing_edge:
        if te_height_local is None:
            msg = "blunt_trailing_edge=True requires trailing-edge heights for pyGeo export"
            raise ValueError(msg)
        kwargs["teHeightScaled"] = te_height_local.tolist()

    return pygeo_class(
        "liftingSurface",
        xsections=xsections,
        scale=scale,
        offset=offset,
        x=x,
        y=y,
        z=z,
        rotZ=rot_z,
        nCtl=int(export_cfg.section_curve_n_ctl),
        kSpan=int(export_cfg.k_span),
        bluntTe=bool(export_cfg.blunt_trailing_edge),
        tip=str(export_cfg.tip_style),
        **kwargs,
    )


def _load_pyspline_surface_class():
    ensure_local_dependency_paths()
    patch_pyspline_for_pygeo()
    from pyspline import Surface

    return Surface


def _build_planar_surface_patch(
    x_outer: np.ndarray,
    y_outer: np.ndarray,
    x_inner: np.ndarray,
    y_inner: np.ndarray,
):
    surface_class = _load_pyspline_surface_class()
    x_outer = np.asarray(x_outer, dtype=float)
    y_outer = np.asarray(y_outer, dtype=float)
    x_inner = np.asarray(x_inner, dtype=float)
    y_inner = np.asarray(y_inner, dtype=float)
    if x_outer.shape != x_inner.shape or y_outer.shape != y_inner.shape:
        msg = "planar surface patch boundaries must share the same discretization"
        raise ValueError(msg)

    nu = int(x_outer.size)
    coef = np.zeros((nu, 2, 3), dtype=float)
    coef[:, 0, 0] = x_outer
    coef[:, 0, 1] = y_outer
    coef[:, 1, 0] = x_inner
    coef[:, 1, 1] = y_inner

    interior = np.arange(1, nu - 1, dtype=float) / max(nu - 1, 1)
    tu = np.concatenate(([0.0, 0.0], interior, [1.0, 1.0])).astype(float)
    tv = np.asarray([0.0, 0.0, 1.0, 1.0], dtype=float)
    return surface_class(
        coef=coef,
        ku=2,
        kv=2,
        tu=tu,
        tv=tv,
    )


def build_xy_symmetry_frame_surfaces(
    geometry: PreparedGeometry,
    x_margin_factor: float = 0.08,
    y_margin_factor: float = 0.10,
    min_x_margin_m: float = 0.50,
    min_y_margin_m: float = 0.25,
    side_point_count: int = 121,
    x_min_m: float | None = None,
    x_max_m: float | None = None,
    y_min_m: float | None = None,
    y_max_m: float | None = None,
):
    root_station = _root_station(geometry.resolved_stations)
    x_air = np.asarray(root_station.x_over_c, dtype=float)
    chord = max(float(root_station.chord_m), 1.0e-6)
    x_local = x_air * chord
    y_upper_local = np.asarray(root_station.upper_z_over_c, dtype=float) * chord
    y_lower_local = np.asarray(root_station.lower_z_over_c, dtype=float) * chord

    twist_rad = np.deg2rad(float(root_station.twist_deg))
    cos_twist = float(np.cos(twist_rad))
    sin_twist = float(np.sin(twist_rad))
    leading_edge_x = float(root_station.leading_edge_x_m)
    leading_edge_y = float(root_station.leading_edge_z_m)

    # Build the frame in the same x-y plane used by the pyGeo liftingSurface
    # export. Do not reuse the already-resolved xyz station coordinates here:
    # those are in the MADS model frame and may use a different twist pivot.
    x_upper = leading_edge_x + x_local * cos_twist - y_upper_local * sin_twist
    y_upper = leading_edge_y + x_local * sin_twist + y_upper_local * cos_twist
    x_lower = leading_edge_x + x_local * cos_twist - y_lower_local * sin_twist
    y_lower = leading_edge_y + x_local * sin_twist + y_lower_local * cos_twist

    x_min_hole = float(min(np.min(x_upper), np.min(x_lower)))
    x_max_hole = float(max(np.max(x_upper), np.max(x_lower)))
    y_min_hole = float(min(np.min(y_upper), np.min(y_lower)))
    y_max_hole = float(max(np.max(y_upper), np.max(y_lower)))
    root_height = max(y_max_hole - y_min_hole, 1.0e-6)

    x_margin = max(float(x_margin_factor) * chord, float(min_x_margin_m))
    y_margin = max(float(y_margin_factor) * root_height, float(min_y_margin_m))
    x_min = x_min_hole - x_margin if x_min_m is None else float(x_min_m)
    x_max = x_max_hole + x_margin if x_max_m is None else float(x_max_m)
    y_min = y_min_hole - y_margin if y_min_m is None else float(y_min_m)
    y_max = y_max_hole + y_margin if y_max_m is None else float(y_max_m)

    outer_top_x = np.linspace(x_min, x_max, x_upper.size, dtype=float)
    outer_top_y = np.full_like(outer_top_x, y_max)
    outer_bottom_x = np.linspace(x_min, x_max, x_lower.size, dtype=float)
    outer_bottom_y = np.full_like(outer_bottom_x, y_min)

    outer_side_y = np.linspace(y_max, y_min, int(side_point_count), dtype=float)
    outer_left_x = np.full_like(outer_side_y, x_min)
    outer_right_x = np.full_like(outer_side_y, x_max)

    inner_left_x = np.linspace(float(x_upper[0]), float(x_lower[0]), int(side_point_count), dtype=float)
    inner_left_y = np.linspace(float(y_upper[0]), float(y_lower[0]), int(side_point_count), dtype=float)
    inner_right_x = np.linspace(float(x_upper[-1]), float(x_lower[-1]), int(side_point_count), dtype=float)
    inner_right_y = np.linspace(float(y_upper[-1]), float(y_lower[-1]), int(side_point_count), dtype=float)

    return [
        _build_planar_surface_patch(outer_top_x, outer_top_y, x_upper, y_upper),
        _build_planar_surface_patch(outer_bottom_x, outer_bottom_y, x_lower, y_lower),
        _build_planar_surface_patch(outer_left_x, outer_side_y, inner_left_x, inner_left_y),
        _build_planar_surface_patch(outer_right_x, outer_side_y, inner_right_x, inner_right_y),
    ]


def append_xy_symmetry_frame_surfaces(
    geometry_surface,
    geometry: PreparedGeometry | None = None,
    *,
    frame_surfaces=None,
    **kwargs,
):
    if frame_surfaces is None:
        if geometry is None:
            msg = "geometry is required when frame_surfaces are not provided"
            raise ValueError(msg)
        frame_surfaces = build_xy_symmetry_frame_surfaces(geometry, **kwargs)
    geometry_surface.surfs.extend(frame_surfaces)
    geometry_surface.nSurf = len(geometry_surface.surfs)
    return geometry_surface


def _station_index_at_y(
    stations: tuple[ResolvedStation, ...],
    spanwise_y_m: float,
    *,
    atol: float = 1.0e-9,
) -> int:
    for idx, station in enumerate(stations):
        if abs(float(station.spanwise_y_m) - float(spanwise_y_m)) <= atol:
            return idx
    raise KeyError(f"No resolved station found at y={spanwise_y_m}")


def _root_station(stations: tuple[ResolvedStation, ...]) -> ResolvedStation:
    return min(stations, key=lambda station: abs(float(station.spanwise_y_m)))
