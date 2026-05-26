"""Geometry solver interface for synthesis-driven wing workflows in MADS."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from multiads.assembly import Wing, copy_components, flatten_components
from multiads.scenario import InnerVariableFloat
from multiads.solvers import BaseSolver
from multiads.solvers.synthesis.geometry_lib import (
    GeometryEnvelope,
    GeometryMetricSet,
    GeometryValidationResult,
    Options,
    PreparedGeometry,
    PyGeoExportOptions,
    PyGeoExportResult,
    ResolvedStation,
    WingGeometryConfig,
    build_control_point_planform,
    build_envelope,
    build_geometry_config,
    build_resolved_surface_mesh,
    compute_geometry_metrics,
    export_component_geometry_to_pygeo,
    export_geometry_to_pygeo,
    quintic_c2_transition,
    resolve_anchor_section,
    resolve_component_geometry,
    resolve_geometry,
    ResolvedSurfaceMesh,
    write_resolved_surface_mesh_npz,
)

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from numpy.typing import NDArray

    from multiads.assembly import MADSComponent
    from multiads.scenario import BaseVariable


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
    "build_resolved_surface_mesh",
    "compute_geometry_metrics",
    "CSTGeometrySolver",
    "GeometryEnvelope",
    "GeometryExportRunResult",
    "GeometryMetricSet",
    "GeometryValidationResult",
    "Options",
    "PreparedGeometry",
    "PyGeoExportOptions",
    "PyGeoExportResult",
    "PyGeoExportSolver",
    "quintic_c2_transition",
    "ResolvedStation",
    "ResolvedSurfaceMesh",
    "export_component_geometry_to_pygeo",
    "export_geometry_to_pygeo",
    "resolve_anchor_section",
    "resolve_component_geometry",
    "resolve_geometry",
    "write_resolved_surface_mesh_npz",
    "WingGeometryConfig",
]
