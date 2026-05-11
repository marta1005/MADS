"""Public geometry entrypoint for synthesis-driven wing resolution in MADS."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

from multiads.assembly import AirfoilCST, Section, Span, Wing, copy_components, flatten_components
from multiads.scenario import InnerVariableFloat
from multiads.solvers import BaseSolver, SolverOptions
from multiads.solvers.synthesis._geometry_planform import build_control_point_planform
from multiads.solvers.synthesis._geometry_pygeo import export_prepared_geometry_to_pygeo
from multiads.solvers.synthesis._geometry_sections import resolve_anchor_section
from multiads.solvers.synthesis._geometry_spanwise import (
    interpolate_scalar_field,
    interpolate_vector_field,
    merge_sampled_stations,
)
from multiads.solvers.synthesis._geometry_types import (
    GeometryEnvelope,
    GeometryMetricSet,
    PyGeoExportResult,
    PreparedGeometry,
    ResolvedStation,
    WingGeometryConfig,
)

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from numpy.typing import NDArray

    from multiads.assembly import MADSComponent
    from multiads.scenario import BaseVariable


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


@dataclass(slots=True)
class GeometryRunResult:
    """One resolved geometry run attached to one geometry-driven ``Wing`` component."""

    config: WingGeometryConfig
    geometry: PreparedGeometry
    metrics: GeometryMetricSet


@dataclass(slots=True)
class GeometryExportRunResult:
    """One pyGeo export run attached to one geometry-driven ``Wing`` component."""

    geometry: PreparedGeometry
    export: PyGeoExportResult


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


class CSTGeometrySolver(BaseSolver):
    """Resolve CST-driven ``Wing`` components into explicit geometry states."""

    def __init__(self, options: Options | None = None) -> None:
        super().__init__()
        self.options = options or Options()
        self.wings: list[Wing] = []
        self.results: dict[str, GeometryRunResult] = {}
        self.metric_outputs: dict[str, InnerVariableFloat] = {}

    def parse_variables(
        self,
        components: Sequence[MADSComponent],
        *argv: Any,  # noqa: ANN401, ARG002
    ) -> Sequence[MADSComponent]:
        copied_components = copy_components(components)
        components_flat = flatten_components(copied_components)
        self.wings = [
            component
            for component in components_flat
            if isinstance(component, Wing)
        ]
        if not self.wings:
            msg = "CSTGeometrySolver requires at least one Wing component."
            raise ValueError(msg)

        self.inputs = []
        self.outputs = []
        self.metric_outputs = {}
        self.results = {}

        for wing in self.wings:
            for component in flatten_components([wing]):
                self.inputs.extend(component.variables.values())

            for suffix in (
                "span_m",
                "planform_area_m2",
                "enclosed_volume_m3",
                "root_chord_m",
                "tip_chord_m",
                "mean_aerodynamic_chord_m",
            ):
                output = InnerVariableFloat(f"{wing.name}.{suffix}", 0.0)
                self.metric_outputs[output.name] = output
                self.outputs.append(output)

        return self.wings

    def _run(self) -> None:
        self.results = {}
        for wing in self.wings:
            config, geometry, metrics = resolve_component_geometry(wing)
            wing.geometry_state = geometry

            if self.options.write_metrics_back_to_component:
                wing.geometry_metrics = metrics

            self.results[wing.name] = GeometryRunResult(
                config=config,
                geometry=geometry,
                metrics=metrics,
            )

    def compute_output(self) -> None:
        for wing_name, result in self.results.items():
            metrics = result.metrics
            self.metric_outputs[f"{wing_name}.span_m"].value = metrics.span_m
            self.metric_outputs[f"{wing_name}.planform_area_m2"].value = metrics.planform_area_m2
            self.metric_outputs[f"{wing_name}.enclosed_volume_m3"].value = metrics.enclosed_volume_m3
            self.metric_outputs[f"{wing_name}.root_chord_m"].value = metrics.root_chord_m
            self.metric_outputs[f"{wing_name}.tip_chord_m"].value = metrics.tip_chord_m
            self.metric_outputs[f"{wing_name}.mean_aerodynamic_chord_m"].value = (
                metrics.mean_aerodynamic_chord_m
                if metrics.mean_aerodynamic_chord_m is not None
                else 0.0
            )

    def compute_sensitivities(
        self,
        input_names: Sequence[str],  # noqa: ARG002
        inputs: Sequence[BaseVariable],  # noqa: ARG002
        output_names: Sequence[str],  # noqa: ARG002
        outputs: Sequence[BaseVariable],  # noqa: ARG002
    ) -> Mapping[str, NDArray]:
        return {}


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


class PyGeoExportSolver(BaseSolver):
    """Export resolved ``Wing`` components to pyGeo/IGES artifacts."""

    def __init__(self, options: PyGeoExportOptions | None = None) -> None:
        super().__init__()
        self.options = options or PyGeoExportOptions()
        self.wings: list[Wing] = []
        self.results: dict[str, GeometryExportRunResult] = {}
        self.metric_outputs: dict[str, InnerVariableFloat] = {}

    def parse_variables(
        self,
        components: Sequence[MADSComponent],
        *argv: Any,  # noqa: ANN401, ARG002
    ) -> Sequence[MADSComponent]:
        copied_components = copy_components(components)
        components_flat = flatten_components(copied_components)
        self.wings = [
            component
            for component in components_flat
            if isinstance(component, Wing)
        ]
        if not self.wings:
            msg = "PyGeoExportSolver requires at least one Wing component."
            raise ValueError(msg)

        self.inputs = []
        self.outputs = []
        self.metric_outputs = {}
        self.results = {}

        for wing in self.wings:
            for component in flatten_components([wing]):
                self.inputs.extend(component.variables.values())

            for suffix in (
                "pygeo_station_count",
                "pygeo_profile_count",
                "pygeo_surface_count",
                "pygeo_iges_written",
            ):
                output = InnerVariableFloat(f"{wing.name}.{suffix}", 0.0)
                self.metric_outputs[output.name] = output
                self.outputs.append(output)

        return self.wings

    def _run(self) -> None:
        self.results = {}
        for wing in self.wings:
            export_result = export_component_geometry_to_pygeo(wing)
            geometry = wing.geometry_state
            if geometry is None:
                msg = f"Geometry export for wing '{wing.name}' did not produce a geometry_state."
                raise RuntimeError(msg)

            if self.options.write_export_back_to_component:
                wing.export_state = export_result

            self.results[wing.name] = GeometryExportRunResult(
                geometry=geometry,
                export=export_result,
            )

    def compute_output(self) -> None:
        for wing_name, result in self.results.items():
            export = result.export
            surface_count = 0.0
            if export.surface is not None and hasattr(export.surface, "nSurf"):
                surface_count = float(export.surface.nSurf)
            self.metric_outputs[f"{wing_name}.pygeo_station_count"].value = float(export.station_count)
            self.metric_outputs[f"{wing_name}.pygeo_profile_count"].value = float(export.written_profile_count)
            self.metric_outputs[f"{wing_name}.pygeo_surface_count"].value = surface_count
            self.metric_outputs[f"{wing_name}.pygeo_iges_written"].value = (
                1.0 if export.iges_path is not None else 0.0
            )

    def compute_sensitivities(
        self,
        input_names: Sequence[str],  # noqa: ARG002
        inputs: Sequence[BaseVariable],  # noqa: ARG002
        output_names: Sequence[str],  # noqa: ARG002
        outputs: Sequence[BaseVariable],  # noqa: ARG002
    ) -> Mapping[str, NDArray]:
        return {}


__all__ = [
    "build_control_point_planform",
    "build_envelope",
    "build_geometry_config",
    "compute_geometry_metrics",
    "export_component_geometry_to_pygeo",
    "export_geometry_to_pygeo",
    "CSTGeometrySolver",
    "GeometryEnvelope",
    "GeometryExportRunResult",
    "GeometryMetricSet",
    "Options",
    "PreparedGeometry",
    "PyGeoExportOptions",
    "PyGeoExportResult",
    "PyGeoExportSolver",
    "ResolvedStation",
    "resolve_anchor_section",
    "resolve_component_geometry",
    "resolve_geometry",
    "WingGeometryConfig",
]
