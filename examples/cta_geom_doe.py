from __future__ import annotations

import copy
import csv
import json
import logging
import re
import shutil
from pathlib import Path

import gemseo
import numpy as np

import cta_geometry as cta
from multiads.ambiance import Atmosphere
from multiads.assembly import AirfoilCST, Environment, Section, Span, Wing
from multiads.disciplines import UserDefined
from multiads.scenario import MADSScenario, VariableFloat
from multiads.solvers.aerodynamics import dust, dust_lib
from multiads.utilities.internal_volume_constraints import (
    CadReferenceFrame,
    evaluate_internal_volume_constraints,
    load_internal_volume_constraint_set,
)


gemseo.configure_logger(
    level=logging.INFO,
    filename="gemseo.log",
    filemode="w",
)


N_SAMPLES = 1
DOE_ALGO = "LHS"
TRIANGLE_RESOLUTION = 8
REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = REPO_ROOT / "outputs" / "cta_geom_doe"
DUST_WORK_DIR = OUTPUT_DIR / "dust_work"
DUST_LATEST_RUN_DIR = DUST_WORK_DIR / "latest"
DUST_OVERWRITE_PREVIOUS_RUNS = True
DUST_BIN_DIR = Path(
    "/Users/martaarnabatmartin/Desktop/pruebas/dust_runs/dust-install/bin",
)
DUST_MESH_PREFIX = OUTPUT_DIR / "dust_geometry" / "cta_resolved_"
DUST_MESH_VTK_PATH = OUTPUT_DIR / "dust_geometry" / "cta_resolved_geometry.vtk"
DUST_MESH_SPANWISE_STATIONS = 25
PARAVIEW_MESH_SPANWISE_STATIONS = None
DUST_MESH_CHORDWISE_POINTS = 65
DUST_TRAILING_EDGE_INNER_PRODUCT = 0.5
DUST_ENFORCE_SYMMETRY_TANGENCY = False
DUST_SYMMETRY_TANGENCY_BLEND_M = 1.9
DUST_FLIGHT_MACH = 0.8
DUST_FLIGHT_ALTITUDE_FT = 40000.0
DUST_FLIGHT_ALTITUDE_M = DUST_FLIGHT_ALTITUDE_FT * 0.3048
DUST_ISA_DELTA_T_C = 0.0
DUST_FLIGHT_SPEED_MPS = float(
    DUST_FLIGHT_MACH * Atmosphere(DUST_FLIGHT_ALTITUDE_M).speed_of_sound[0]
)
DUST_FLIGHT_DENSITY_KG_M3 = float(Atmosphere(DUST_FLIGHT_ALTITUDE_M).density[0])
DUST_DYNAMIC_PRESSURE_PA = (
    0.5 * DUST_FLIGHT_DENSITY_KG_M3 * DUST_FLIGHT_SPEED_MPS**2
)
DUST_ALPHA_DEG = 3.0
DUST_T_END_S = 0.7
DUST_DT_S = 0.01
DUST_OUTPUT_DT_S = DUST_T_END_S
DUST_LOADS_START_STEP = 1
DUST_LOADS_END_STEP = 1
DUST_N_WAKE_PANELS = 1
DUST_N_WAKE_PARTICLES = 50_000
DUST_PARTICLES_BOX_MIN = np.array([0.0, -80.0, -15.0])
DUST_PARTICLES_BOX_MAX = np.array([120.0, 80.0, 15.0])
DUST_USE_FMM = True
DUST_FMM_BOX_LENGTH = 10.0
DUST_FMM_N_BOX = np.array([12, 16, 3], dtype=np.int32)
DUST_FMM_OCTREE_ORIGIN = np.array([0.0, -80.0, -15.0])
DUST_FMM_N_OCTREE_LEVELS = 5
DUST_FMM_MIN_OCTREE_PART = 10
DUST_FMM_MULTIPOLE_DEGREE = 2
ENABLE_INTERNAL_BOX_EVALUATION = False
ENABLE_DUST_SMOKE_RUN = True
CTA_INTERNAL_VOLUME_CONSTRAINTS_PATH = (
    REPO_ROOT / "assets" / "cta" / "internal_volume_constraints_set1.csv"
)

span_outer = VariableFloat(
    "cta_dv_span_outer_m",
    cta.CTA_SPAN_OUTER_BASE_M,
    lb=cta.CTA_SPAN_OUTER_BOUNDS_M[0],
    ub=cta.CTA_SPAN_OUTER_BOUNDS_M[1],
)
b2_span_ratio = VariableFloat(
    "cta_dv_b2_span_ratio",
    cta.CTA_B2_SPAN_RATIO_BASE,
    lb=cta.CTA_B2_SPAN_RATIO_BOUNDS[0],
    ub=cta.CTA_B2_SPAN_RATIO_BOUNDS[1],
)
dv_chord_c0 = VariableFloat(
    "cta_dv_chord_c0_m",
    cta.CTA_CHORD_C0_BASE_M,
    lb=cta.CTA_CHORD_C0_BOUNDS_M[0],
    ub=cta.CTA_CHORD_C0_BOUNDS_M[1],
)
dv_chord_c3 = VariableFloat(
    "cta_dv_chord_c3_m",
    cta.CTA_CHORD_C3_BASE_M,
    lb=cta.CTA_CHORD_C3_BOUNDS_M[0],
    ub=cta.CTA_CHORD_C3_BOUNDS_M[1],
)
c4_c3_ratio = VariableFloat(
    "cta_dv_c4_c3_ratio",
    cta.CTA_CHORD_C4_C3_RATIO_BASE,
    lb=cta.CTA_CHORD_C4_C3_RATIO_BOUNDS[0],
    ub=cta.CTA_CHORD_C4_C3_RATIO_BOUNDS[1],
)
dv_chord_c5 = VariableFloat(
    "cta_dv_chord_c5_m",
    cta.CTA_CHORD_C5_BASE_M,
    lb=cta.CTA_CHORD_C5_BOUNDS_M[0],
    ub=cta.CTA_CHORD_C5_BOUNDS_M[1],
)
sweep_s2 = VariableFloat(
    "cta_dv_s2_deg",
    cta.CTA_SWEEP_S2_BASE_DEG,
    lb=cta.CTA_SWEEP_S2_BOUNDS_DEG[0],
    ub=cta.CTA_SWEEP_S2_BOUNDS_DEG[1],
)
sweep_s3 = VariableFloat(
    "cta_dv_s3_deg",
    cta.CTA_SWEEP_S3_BASE_DEG,
    lb=cta.CTA_SWEEP_S3_BOUNDS_DEG[0],
    ub=cta.CTA_SWEEP_S3_BOUNDS_DEG[1],
)
planform_design_variables = [
    span_outer,
    b2_span_ratio,
    dv_chord_c0,
    dv_chord_c3,
    c4_c3_ratio,
    dv_chord_c5,
    sweep_s2,
    sweep_s3,
]
camber_mode_variables = {
    key: VariableFloat(
        f"cta_camber_mode_{key[0]}_c{key[1]}",
        0.0,
        lb=cta.CTA_CAMBER_MODE_BOUNDS[0],
        ub=cta.CTA_CAMBER_MODE_BOUNDS[1],
    )
    for key in cta.CTA_CAMBER_MODE_KEYS
}
camber_design_variables = list(camber_mode_variables.values())
CTA_DOE_DESIGN_VARIABLES = [
    *planform_design_variables,
    *camber_design_variables,
]

cta_planform_mapping_outputs = [
    cta.chord_c0,
    cta.chord_body_helper_01,
    cta.chord_body_helper_02,
    cta.chord_c3,
    cta.chord_c4,
    cta.chord_c5,
    cta.y_c4,
    cta.y_c5,
    cta.le_x_c3,
    cta.le_x_c4,
    cta.le_x_c5,
]

dust_mesh_export_trigger = VariableFloat("cta_dust_mesh_export_trigger", 0.0)


def _cta_planform_design_mapping(  # noqa: PLR0913
    span_outer_m,
    b2_span_ratio_value,
    chord_c0_m,
    chord_c3_m,
    c4_c3_ratio_value,
    chord_c5_m,
    sweep_s2_deg,
    sweep_s3_deg,
):  # noqa: ANN001, ANN201
    span_outer_m = float(np.ravel(span_outer_m)[0])
    b2_span_ratio_value = float(np.ravel(b2_span_ratio_value)[0])
    chord_c0_m = float(np.ravel(chord_c0_m)[0])
    chord_c3_m = float(np.ravel(chord_c3_m)[0])
    c4_c3_ratio_value = float(np.ravel(c4_c3_ratio_value)[0])
    chord_c5_m = float(np.ravel(chord_c5_m)[0])
    sweep_s2_deg = float(np.ravel(sweep_s2_deg)[0])
    sweep_s3_deg = float(np.ravel(sweep_s3_deg)[0])

    chord_c4_m = chord_c3_m * c4_c3_ratio_value
    y_c4_m = cta.CTA_B1_FIXED_M + b2_span_ratio_value * span_outer_m
    y_c5_m = cta.CTA_B1_FIXED_M + span_outer_m
    root_te_x_m = cta.CTA_LE_X_C0_BASE_M + chord_c0_m
    chord_body_helper_01_m = root_te_x_m - cta.CTA_LE_X_BODY_HELPER_01_BASE_M
    chord_body_helper_02_m = root_te_x_m - cta.CTA_LE_X_BODY_HELPER_02_BASE_M

    le_x_c4_m = (
        cta.CTA_LE_X_C3_BASE_M
        + 0.5 * chord_c3_m
        + np.tan(np.radians(sweep_s2_deg)) * (y_c4_m - cta.CTA_B1_FIXED_M)
        - 0.5 * chord_c4_m
    )
    le_x_c5_m = (
        le_x_c4_m
        + 0.25 * chord_c4_m
        + np.tan(np.radians(sweep_s3_deg)) * (y_c5_m - y_c4_m)
        - 0.25 * chord_c5_m
    )

    return (
        chord_c0_m,
        chord_body_helper_01_m,
        chord_body_helper_02_m,
        chord_c3_m,
        chord_c4_m,
        chord_c5_m,
        y_c4_m,
        y_c5_m,
        cta.CTA_LE_X_C3_BASE_M,
        le_x_c4_m,
        le_x_c5_m,
    )


CTA_INTERNAL_VOLUME_CONSTRAINTS = load_internal_volume_constraint_set(
    CTA_INTERNAL_VOLUME_CONSTRAINTS_PATH,
    reference_frame=CadReferenceFrame(
        name="CTA CAD reference",
        offset_x_m=0.0,
        offset_y_m=0.0,
        offset_z_m=0.0,
        mirror_about_symmetry_plane=True,
    ),
    name="CTA B359-V0 internal volume boxes",
)

all_boxes_fit = VariableFloat("cta_all_boxes_fit", 0.0)
internal_boxes_min_margin = VariableFloat("cta_internal_boxes_min_margin_m", 0.0)
box_result_variables: list[VariableFloat] = [
    all_boxes_fit,
    internal_boxes_min_margin,
]
for idx, surface in enumerate(CTA_INTERNAL_VOLUME_CONSTRAINTS.surfaces, start=1):
    safe_label = re.sub(r"[^0-9a-zA-Z]+", "_", surface.label).strip("_").lower()
    box_result_variables.extend(
        [
            VariableFloat(f"cta_box_{idx:02d}_{safe_label}_fits", 0.0),
            VariableFloat(f"cta_box_{idx:02d}_{safe_label}_margin_m", 0.0),
        ]
    )


def _evaluate_internal_boxes_from_last_geometry(*_geometry_metrics):  # noqa: ANN002, ANN201
    resolved_wing = cta.disc_geometry.components[0]
    geometry = getattr(resolved_wing, "geometry_state", None)
    if geometry is None:
        msg = "CTA geometry must be resolved before evaluating internal volume boxes."
        raise RuntimeError(msg)

    result = evaluate_internal_volume_constraints(
        geometry,
        CTA_INTERNAL_VOLUME_CONSTRAINTS,
        triangle_resolution=TRIANGLE_RESOLUTION,
    )
    outputs = [
        1.0 if result.satisfied else 0.0,
        result.minimum_margin_m,
    ]
    for surface_result in result.surface_results:
        outputs.extend(
            [
                1.0 if surface_result.satisfied else 0.0,
                surface_result.worst_margin_m,
            ]
        )
    return tuple(outputs)


def _linspace_indices(size: int, requested_count: int) -> np.ndarray:
    count = min(int(requested_count), int(size))
    return np.unique(np.round(np.linspace(0, size - 1, count)).astype(int))


def _mesh_indices(size: int, requested_count: int | None) -> np.ndarray:
    if requested_count is None:
        return np.arange(size, dtype=int)
    return _linspace_indices(size, requested_count)


def _write_legacy_vtk_quad_mesh(
    points: np.ndarray,
    elements: np.ndarray,
    path: Path,
) -> None:
    with path.open("w", encoding="utf-8") as stream:
        stream.write("# vtk DataFile Version 3.0\n")
        stream.write("CTA resolved DUST mesh\n")
        stream.write("ASCII\n")
        stream.write("DATASET UNSTRUCTURED_GRID\n")
        stream.write(f"POINTS {points.shape[0]} float\n")
        for x_m, y_m, z_m in points:
            stream.write(f"{x_m:.10f} {y_m:.10f} {z_m:.10f}\n")

        stream.write(f"CELLS {elements.shape[0]} {elements.shape[0] * 5}\n")
        for element in elements:
            node_ids = element - 1
            stream.write(
                f"4 {node_ids[0]} {node_ids[1]} {node_ids[2]} {node_ids[3]}\n",
            )

        stream.write(f"CELL_TYPES {elements.shape[0]}\n")
        for _ in elements:
            stream.write("9\n")


def _build_closed_dust_surface_mesh(
    upper: np.ndarray,
    lower: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    section_loops = [
        np.vstack((upper[i_span], lower[i_span, -2:0:-1]))
        for i_span in range(upper.shape[0])
    ]
    ring_size = section_loops[0].shape[0]

    points: list[np.ndarray] = []
    point_ids = np.empty((len(section_loops), ring_size), dtype=int)
    for i_span, loop in enumerate(section_loops):
        for i_point, point in enumerate(loop):
            points.append(point)
            point_ids[i_span, i_point] = len(points)

    elements: list[tuple[int, int, int, int]] = []
    for i_span in range(point_ids.shape[0] - 1):
        for i_point in range(ring_size):
            next_point = (i_point + 1) % ring_size
            elements.append(
                (
                    point_ids[i_span, i_point],
                    point_ids[i_span, next_point],
                    point_ids[i_span + 1, next_point],
                    point_ids[i_span + 1, i_point],
                ),
            )

    return np.asarray(points, dtype=float), np.asarray(elements, dtype=int)


def _build_closed_profile_visualization_mesh(
    upper: np.ndarray,
    lower: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    section_loops = [
        np.vstack((upper[i_span], lower[i_span][::-1]))
        for i_span in range(upper.shape[0])
    ]
    ring_size = section_loops[0].shape[0]

    points: list[np.ndarray] = []
    point_ids = np.empty((len(section_loops), ring_size), dtype=int)

    for i_span, loop in enumerate(section_loops):
        for i_point in range(ring_size):
            points.append(loop[i_point])
            point_ids[i_span, i_point] = len(points)

    elements: list[tuple[int, int, int, int]] = []
    for i_span in range(point_ids.shape[0] - 1):
        for i_point in range(ring_size):
            next_point = (i_point + 1) % ring_size
            elements.append(
                (
                    point_ids[i_span, i_point],
                    point_ids[i_span, next_point],
                    point_ids[i_span + 1, next_point],
                    point_ids[i_span + 1, i_point],
                ),
            )

    return np.asarray(points, dtype=float), np.asarray(elements, dtype=int)


def _collapse_profile_edges_for_surface_mesh(
    upper: np.ndarray,
    lower: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    upper_mesh = np.array(upper, dtype=float, copy=True)
    lower_mesh = np.array(lower, dtype=float, copy=True)
    leading_edge_midpoint = 0.5 * (upper_mesh[:, 0, :] + lower_mesh[:, 0, :])
    trailing_edge_midpoint = 0.5 * (upper_mesh[:, -1, :] + lower_mesh[:, -1, :])
    upper_mesh[:, 0, :] = leading_edge_midpoint
    lower_mesh[:, 0, :] = leading_edge_midpoint
    upper_mesh[:, -1, :] = trailing_edge_midpoint
    lower_mesh[:, -1, :] = trailing_edge_midpoint
    return upper_mesh, lower_mesh


def _enforce_symmetry_plane_tangency(
    surface: np.ndarray,
    blend_length_m: float,
) -> np.ndarray:
    mesh = np.array(surface, dtype=float, copy=True)
    if mesh.shape[0] < 2 or blend_length_m <= 0.0:
        return mesh

    spanwise_y = mesh[:, 0, 1]
    symmetry_y = spanwise_y[0]
    distance_from_symmetry = np.abs(spanwise_y - symmetry_y)
    root_x = mesh[0, :, 0].copy()
    root_z = mesh[0, :, 2].copy()

    for i_span, distance in enumerate(distance_from_symmetry):
        if i_span == 0 or distance > blend_length_m:
            continue
        normalized_distance = np.clip(distance / blend_length_m, 0.0, 1.0)
        smooth_weight = normalized_distance**3 * (
            normalized_distance * (6.0 * normalized_distance - 15.0) + 10.0
        )
        mesh[i_span, :, 0] = root_x + smooth_weight * (mesh[i_span, :, 0] - root_x)
        mesh[i_span, :, 2] = root_z + smooth_weight * (mesh[i_span, :, 2] - root_z)

    return mesh


def _enforce_half_wing_symmetry_tangency(
    upper: np.ndarray,
    lower: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    if not DUST_ENFORCE_SYMMETRY_TANGENCY:
        return upper, lower
    return (
        _enforce_symmetry_plane_tangency(
            upper,
            DUST_SYMMETRY_TANGENCY_BLEND_M,
        ),
        _enforce_symmetry_plane_tangency(
            lower,
            DUST_SYMMETRY_TANGENCY_BLEND_M,
        ),
    )


def _write_dust_basic_mesh_from_resolved_geometry(geometry) -> None:  # noqa: ANN001
    stations = tuple(geometry.resolved_stations)
    if not stations:
        msg = "CTA geometry has no resolved stations for DUST mesh export."
        raise RuntimeError(msg)

    span_indices = _mesh_indices(len(stations), DUST_MESH_SPANWISE_STATIONS)
    chord_count = stations[0].upper_surface_xyz_m.shape[0]
    chord_indices = _mesh_indices(chord_count, DUST_MESH_CHORDWISE_POINTS)
    upper = np.stack(
        [stations[index].upper_surface_xyz_m[chord_indices] for index in span_indices],
        axis=0,
    )
    lower = np.stack(
        [stations[index].lower_surface_xyz_m[chord_indices] for index in span_indices],
        axis=0,
    )
    upper, lower = _collapse_profile_edges_for_surface_mesh(upper, lower)
    upper, lower = _enforce_half_wing_symmetry_tangency(upper, lower)

    DUST_MESH_PREFIX.parent.mkdir(parents=True, exist_ok=True)
    points_array, elements_array = _build_closed_dust_surface_mesh(upper, lower)
    np.savetxt(
        f"{DUST_MESH_PREFIX}rr.dat",
        points_array,
        fmt="%.10f",
    )
    np.savetxt(
        f"{DUST_MESH_PREFIX}ee.dat",
        elements_array,
        fmt="%d",
    )
    vtk_span_indices = _mesh_indices(len(stations), PARAVIEW_MESH_SPANWISE_STATIONS)
    vtk_upper = np.stack(
        [stations[index].upper_surface_xyz_m[chord_indices] for index in vtk_span_indices],
        axis=0,
    )
    vtk_lower = np.stack(
        [stations[index].lower_surface_xyz_m[chord_indices] for index in vtk_span_indices],
        axis=0,
    )
    vtk_upper, vtk_lower = _collapse_profile_edges_for_surface_mesh(vtk_upper, vtk_lower)
    vtk_upper, vtk_lower = _enforce_half_wing_symmetry_tangency(vtk_upper, vtk_lower)
    vtk_points, vtk_elements = _build_closed_profile_visualization_mesh(vtk_upper, vtk_lower)
    _write_legacy_vtk_quad_mesh(vtk_points, vtk_elements, DUST_MESH_VTK_PATH)


def _export_dust_mesh_from_last_geometry(*_geometry_metrics):  # noqa: ANN002, ANN201
    resolved_wing = cta.disc_geometry.components[0]
    geometry = getattr(resolved_wing, "geometry_state", None)
    if geometry is None:
        msg = "CTA geometry must be resolved before exporting the DUST mesh."
        raise RuntimeError(msg)
    _write_dust_basic_mesh_from_resolved_geometry(geometry)
    return 0.0


disc_internal_boxes = UserDefined(
    name="CTAInternalBoxFit",
    inputs=[cta.disc_geometry.solver.metric_outputs["cta_wing.enclosed_volume_m3"]],
    outputs=box_result_variables,
    expression=_evaluate_internal_boxes_from_last_geometry,
)

disc_planform_mapping = UserDefined(
    name="CTAPlanformDesignMapping",
    inputs=planform_design_variables,
    outputs=cta_planform_mapping_outputs,
    expression=_cta_planform_design_mapping,
)

disc_camber_modes = UserDefined(
    name="CamberModeCoefficientMapping",
    inputs=list(camber_mode_variables.values()),
    outputs=list(cta.camber_coefficient_variables.values()),
    expression=cta.map_cta_camber_modes_to_coefficients,
)

# DUST still needs a MADS Wing/Section/Span object to write its component input.
# The actual geometry is not generated from these interface sections: mesh_file
# below forces DUST to read our resolved CTA basic mesh (*_rr.dat, *_ee.dat).
disc_dust_mesh_export = UserDefined(
    name="CTADUSTResolvedMeshExport",
    inputs=[cta.disc_geometry.solver.metric_outputs["cta_wing.enclosed_volume_m3"]],
    outputs=[dust_mesh_export_trigger],
    expression=_export_dust_mesh_from_last_geometry,
)

dust_root_interface_airfoil = AirfoilCST(
    name="cta_dust_root_interface_airfoil",
    upper_coefficients=cta.CTA_C0_UPPER_CST,
    lower_coefficients=cta.CTA_C0_LOWER_CST,
    n1=cta.CTA_CST_N1,
    n2=cta.CTA_CST_N2,
    trailing_edge_thickness=float(cta.te_c0.value),
)
dust_tip_interface_airfoil = AirfoilCST(
    name="cta_dust_tip_interface_airfoil",
    upper_coefficients=cta.CTA_C5_UPPER_CST,
    lower_coefficients=cta.CTA_C5_LOWER_CST,
    n1=cta.CTA_CST_N1,
    n2=cta.CTA_CST_N2,
    trailing_edge_thickness=float(cta.te_c5.value),
)

dust_section_root = Section(
    name="cta_dust_mesh_root",
    airfoil=dust_root_interface_airfoil,
    chord=1.0,
    twist=0.0,
    options=[dust_lib.SectionOptions(polar=False)],
)
dust_section_tip = Section(
    name="cta_dust_mesh_tip",
    airfoil=dust_tip_interface_airfoil,
    chord=1.0,
    twist=0.0,
    options=[dust_lib.SectionOptions(polar=False)],
)

dust_mesh_interface_span = Span(
    name="cta_dust_mesh_interface_span",
    length=1.0,
    sweep=0.0,
    dihed=0.0,
    options=[
        dust_lib.SpanOptions(
            panel_type=dust_lib.SpanPanelType.UNIFORM,
            num_panels=1,
        ),
    ],
)

dust_resolved_mesh_wing = Wing(
    name="cta_dust_wing",
    sections=[dust_section_root, dust_section_tip],
    spans=[dust_mesh_interface_span],
    symmetry=True,
    xc_ref=0.25,
    options=[
        dust_lib.WingOptions(
            discretization_method=dust_lib.WingMethod.PANELS,
            panel_type=dust_lib.WingPanelType.UNIFORM,
            num_panels=DUST_MESH_CHORDWISE_POINTS - 1,
            # This is the important switch: DUST writes mesh_file_type=basic and
            # uses our exported resolved CTA mesh, not a parametric DUST mesh.
            mesh_file=DUST_MESH_PREFIX,
            inner_product_te=DUST_TRAILING_EDGE_INNER_PRODUCT,
            output_options=dust_lib.OutputOptions(
                compute_loads=True,
                loads_start=DUST_LOADS_START_STEP,
                loads_end=DUST_LOADS_END_STEP,
                loads_avg=True,
            ),
        ),
    ],
)

def _make_dust_options(
    *,
    keep_run_directory: bool,
    visualization: bool,
) -> dust_lib.Options:
    return dust_lib.Options(
        name="cta_dust_smoke_",
        dust_pre=DUST_BIN_DIR / "dust_pre",
        dust=DUST_BIN_DIR / "dust",
        dust_post=DUST_BIN_DIR / "dust_post",
        work_dir=DUST_WORK_DIR,
        keep_run_directory=keep_run_directory,
        t_end=DUST_T_END_S,
        dt=DUST_DT_S,
        dt_out=DUST_OUTPUT_DT_S,
        n_threads=1,
        n_wake_panels=DUST_N_WAKE_PANELS,
        n_wake_particles=DUST_N_WAKE_PARTICLES,
        particles_box_min=DUST_PARTICLES_BOX_MIN,
        particles_box_max=DUST_PARTICLES_BOX_MAX,
        fmm=DUST_USE_FMM,
        box_length=DUST_FMM_BOX_LENGTH,
        n_box=DUST_FMM_N_BOX,
        octree_origin=DUST_FMM_OCTREE_ORIGIN,
        n_octree_levels=DUST_FMM_N_OCTREE_LEVELS,
        min_octree_part=DUST_FMM_MIN_OCTREE_PART,
        multipole_degree=DUST_FMM_MULTIPOLE_DEGREE,
        output_options=dust_lib.OutputOptions(
            compute_loads=True,
            loads_start=DUST_LOADS_START_STEP,
            loads_end=DUST_LOADS_END_STEP,
            loads_avg=True,
            visualization=visualization,
            viz_start=DUST_LOADS_END_STEP,
            viz_end=DUST_LOADS_END_STEP,
            viz_wake=True,
            viz_separate_wake=True,
            viz_variables=["cp", "vorticity"],
        ),
    )

dust_alpha = VariableFloat("cta_dust_alpha_deg", DUST_ALPHA_DEG)
dust_lift = VariableFloat("lift", 0.0)
dust_drag = VariableFloat("drag", 0.0)
dust_efficiency = VariableFloat("efficiency", 0.0)
dust_my = VariableFloat("my", 0.0)
dust_wing_lift = VariableFloat("cta_dust_wing.lift", 0.0)
dust_wing_drag = VariableFloat("cta_dust_wing.drag", 0.0)
dust_wing_efficiency = VariableFloat("cta_dust_wing.efficiency", 0.0)
dust_wing_my = VariableFloat("cta_dust_wing.my", 0.0)
dust_cl = VariableFloat("cta_dust_cl", 0.0)
dust_cd = VariableFloat("cta_dust_cd", 0.0)
dust_cm = VariableFloat("cta_dust_cm", 0.0)


def _dust_coefficients(
    lift: float,
    drag: float,
    my: float,
    planform_area_m2: float,
    mean_aerodynamic_chord_m: float,
) -> tuple[float, float, float]:
    force_denominator = DUST_DYNAMIC_PRESSURE_PA * planform_area_m2
    moment_denominator = force_denominator * mean_aerodynamic_chord_m
    return (
        lift / force_denominator,
        drag / force_denominator,
        my / moment_denominator,
    )


def _run_dust_at_alpha(alpha_deg: float, *, keep_run_directory: bool) -> dict[str, float]:
    environment = Environment(
        name="cta_dust_env",
        height=DUST_FLIGHT_ALTITUDE_M,
        speed=DUST_FLIGHT_SPEED_MPS,
        alpha=float(alpha_deg),
    )
    wing = copy.deepcopy(dust_resolved_mesh_wing)
    solver = dust.DUST(
        options=_make_dust_options(
            keep_run_directory=keep_run_directory,
            visualization=keep_run_directory,
        )
    )
    components = solver.parse_variables([environment, wing])
    solver.run(components)
    solver.compute_output()

    outputs = {
        variable.name: float(np.ravel(variable.value_np)[0])
        for variable in solver.outputs or []
        if np.ravel(variable.value_np).size == 1
    }
    outputs["cta_dust_run_directory"] = (
        "" if solver.run_directory is None else str(solver.run_directory)
    )
    return outputs


def _run_dust_from_last_geometry(  # noqa: ANN001, ANN201
    _mesh_export_trigger,
    planform_area_m2,
    mean_aerodynamic_chord_m,
):
    planform_area_m2 = float(np.ravel(planform_area_m2)[0])
    mean_aerodynamic_chord_m = float(np.ravel(mean_aerodynamic_chord_m)[0])
    outputs = _run_dust_at_alpha(DUST_ALPHA_DEG, keep_run_directory=True)
    cl, cd, cm = _dust_coefficients(
        outputs["lift"],
        outputs["drag"],
        outputs["my"],
        planform_area_m2,
        mean_aerodynamic_chord_m,
    )
    return (
        DUST_ALPHA_DEG,
        outputs["lift"],
        outputs["drag"],
        outputs["efficiency"],
        outputs["my"],
        outputs["cta_dust_wing.lift"],
        outputs["cta_dust_wing.drag"],
        outputs["cta_dust_wing.efficiency"],
        outputs["cta_dust_wing.my"],
        cl,
        cd,
        cm,
    )


disc_dust_alpha_run = UserDefined(
    name="CTADUSTAlphaRun",
    inputs=[
        dust_mesh_export_trigger,
        cta.disc_geometry.solver.metric_outputs["cta_wing.planform_area_m2"],
        cta.disc_geometry.solver.metric_outputs["cta_wing.mean_aerodynamic_chord_m"],
    ],
    outputs=[
        dust_alpha,
        dust_lift,
        dust_drag,
        dust_efficiency,
        dust_my,
        dust_wing_lift,
        dust_wing_drag,
        dust_wing_efficiency,
        dust_wing_my,
        dust_cl,
        dust_cd,
        dust_cm,
    ],
    expression=_run_dust_from_last_geometry,
)

DUST_OBSERVABLES = (
    "cta_dust_alpha_deg",
    "lift",
    "drag",
    "efficiency",
    "my",
    "cta_dust_wing.lift",
    "cta_dust_wing.drag",
    "cta_dust_wing.efficiency",
    "cta_dust_wing.my",
    "cta_dust_cl",
    "cta_dust_cd",
    "cta_dust_cm",
)


def build_cta_geometry_doe_scenario() -> MADSScenario:
    """Create the GEMSEO DOE scenario for CTA geometry-space exploration."""

    mads_scenario = MADSScenario()
    mads_scenario.fill_parameter_space(CTA_DOE_DESIGN_VARIABLES)

    disciplines = [
        disc_planform_mapping,
        disc_camber_modes,
        cta.disc_geometry,
    ]
    if ENABLE_INTERNAL_BOX_EVALUATION:
        disciplines.append(disc_internal_boxes)
    if ENABLE_DUST_SMOKE_RUN:
        disciplines.extend([disc_dust_mesh_export, disc_dust_alpha_run])

    mads_scenario.create_scenario(
        disciplines=disciplines,
        formulation="DisciplinaryOpt",
        objective_name="cta_wing.enclosed_volume_m3",
        scenario_type="DOE",
        name="cta_geometry_doe",
    )

    observables = [
        "cta_wing.span_m",
        "cta_wing.planform_area_m2",
        "cta_wing.root_chord_m",
        "cta_wing.tip_chord_m",
        "cta_wing.mean_aerodynamic_chord_m",
    ]
    if ENABLE_INTERNAL_BOX_EVALUATION:
        observables.extend(variable.name for variable in box_result_variables)
    if ENABLE_DUST_SMOKE_RUN:
        observables.extend(DUST_OBSERVABLES)

    for observable in observables:
        mads_scenario.scenario.add_observable(observable)

    return mads_scenario


def _write_design_space(path: Path) -> None:
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=["name", "baseline", "lower_bound", "upper_bound"],
        )
        writer.writeheader()
        for variable in CTA_DOE_DESIGN_VARIABLES:
            writer.writerow(
                {
                    "name": variable.name,
                    "baseline": float(variable.value),
                    "lower_bound": float(variable.lb),
                    "upper_bound": float(variable.ub),
                }
            )


def _write_dust_manifest(
    path: Path,
    dataset_path: Path,
    flat_dataset_path: Path,
    design_space_path: Path,
) -> None:
    manifest = {
        "purpose": "CTA geometry DOE outputs with a one-sample MADS-DUST smoke run.",
        "dataset_csv": str(dataset_path),
        "flat_dataset_csv": str(flat_dataset_path),
        "design_space_csv": str(design_space_path),
        "iges_exported_by_this_script": False,
        "dust_smoke_run_enabled": ENABLE_DUST_SMOKE_RUN,
        "dust_executables": {
            "dust_pre": str(DUST_BIN_DIR / "dust_pre"),
            "dust": str(DUST_BIN_DIR / "dust"),
            "dust_post": str(DUST_BIN_DIR / "dust_post"),
        },
        "dust_flight_condition": {
            "mach": DUST_FLIGHT_MACH,
            "altitude_ft": DUST_FLIGHT_ALTITUDE_FT,
            "altitude_m": DUST_FLIGHT_ALTITUDE_M,
            "isa_delta_t_c": DUST_ISA_DELTA_T_C,
            "speed_mps": DUST_FLIGHT_SPEED_MPS,
            "density_kg_m3": DUST_FLIGHT_DENSITY_KG_M3,
            "dynamic_pressure_pa": DUST_DYNAMIC_PRESSURE_PA,
        },
        "dust_run": {
            "mode": "fixed alpha",
            "alpha_deg": DUST_ALPHA_DEG,
            "t_end_s": DUST_T_END_S,
            "dt_s": DUST_DT_S,
            "dt_out_s": DUST_OUTPUT_DT_S,
            "loads_average_steps": [
                DUST_LOADS_START_STEP,
                DUST_LOADS_END_STEP,
            ],
            "n_wake_panels": DUST_N_WAKE_PANELS,
            "n_wake_particles": DUST_N_WAKE_PARTICLES,
            "particles_box_min": DUST_PARTICLES_BOX_MIN.tolist(),
            "particles_box_max": DUST_PARTICLES_BOX_MAX.tolist(),
            "fmm": DUST_USE_FMM,
            "fmm_box_length": DUST_FMM_BOX_LENGTH,
            "fmm_n_box": DUST_FMM_N_BOX.tolist(),
            "fmm_octree_origin": DUST_FMM_OCTREE_ORIGIN.tolist(),
            "fmm_n_octree_levels": DUST_FMM_N_OCTREE_LEVELS,
            "fmm_min_octree_part": DUST_FMM_MIN_OCTREE_PART,
            "fmm_multipole_degree": DUST_FMM_MULTIPOLE_DEGREE,
        },
        "dust_smoke_model": {
            "status": (
                "The CTA geometry discipline is executed first, a DUST basic mesh "
                "is exported from its resolved geometry_state, and the mesh is then "
                "run through the MADS DUST solver."
            ),
            "geometry_used_by_dust": (
                "Resolved CTA geometry surface mesh written as DUST basic mesh "
                "files. The Section/Span objects in the DUST Wing are only MADS "
                "interface carriers because the DUST input points to mesh_file. The exported "
                "mesh collapses the leading and trailing edge endpoints to their local "
                "midpoints to avoid artificial blunt-edge walls in DUST/ParaView. The "
                "exported mesh is the right half-wing and DUST mirrors it with "
                "mesh_symmetry=T so loads are consistent with full-wing reference area "
                "metrics."
            ),
            "mesh_file_prefix": str(DUST_MESH_PREFIX),
            "mesh_files": [
                f"{DUST_MESH_PREFIX}rr.dat",
                f"{DUST_MESH_PREFIX}ee.dat",
            ],
            "dust_mesh_spanwise_stations": DUST_MESH_SPANWISE_STATIONS,
            "dust_mesh_file_type": "basic",
            "dust_element_type": (
                "p: panel elements applied to the exported CTA mesh. This is not "
                "a parametric DUST geometry source."
            ),
            "paraview_mesh_spanwise_stations": (
                "all_resolved_stations"
                if PARAVIEW_MESH_SPANWISE_STATIONS is None
                else PARAVIEW_MESH_SPANWISE_STATIONS
            ),
            "paraview_vtk": str(DUST_MESH_VTK_PATH),
            "trailing_edge_inner_product": DUST_TRAILING_EDGE_INNER_PRODUCT,
            "symmetry_tangency": {
                "enabled": DUST_ENFORCE_SYMMETRY_TANGENCY,
                "blend_length_m": DUST_SYMMETRY_TANGENCY_BLEND_M,
                "reason": (
                    "The half-wing mesh must enter the y=0 symmetry plane with "
                    "zero lateral slope before DUST mirrors it."
                ),
            },
            "outputs_added_to_dataset": list(DUST_OBSERVABLES),
            "run_directory_retained": True,
            "run_directory_mode": (
                "The latest run is moved to dust_work/latest and previous "
                "temporary DUST runs are deleted."
                if DUST_OVERWRITE_PREVIOUS_RUNS
                else "Every retained DUST run keeps its unique temporary directory."
            ),
            "latest_run_directory": str(DUST_LATEST_RUN_DIR),
            "visualization": (
                "DUST writes a single visualization request with variables cp and "
                "vorticity. Use post/cta_dust_smoke__visualization-<loads_end>.vtu "
                "for the wing surface Cp and "
                "post/cta_dust_smoke__visualization_wpan-<loads_end>.vtu / "
                "post/cta_dust_smoke__visualization_wpart-<loads_end>.vtu for "
                "panel and particle wake data. A particle-only helper VTP is also "
                "written from the DUST HDF5 with Vorticity vectors, "
                "Vorticity_Magnitude, VortexRad and AbsY to avoid plotting the "
                "panel wake when inspecting tip vortices."
            ),
            "coefficient_outputs": [
                "cta_dust_cl = lift / (q * Sref)",
                "cta_dust_cd = drag / (q * Sref)",
                "cta_dust_cm = my / (q * Sref * MAC)",
            ],
            "work_dir": str(DUST_WORK_DIR),
        },
        "dust_export_strategy": {
            "current_mads_dust_path": (
                "DUST currently consumes MADS Wing/Section/Span objects and writes "
                "parametric DUST geometry input files before dust_pre generates geo_input.h5."
            ),
            "recommended_next_step": (
                "Promote the resolved-geometry-to-DUST-basic-mesh exporter to a "
                "utility if the mesh route is accepted as the standard CTA-DUST "
                "adapter."
            ),
            "recommended_case_outputs": [
                "design_variables.json",
                "geometry_metrics.json",
                "internal_box_fit.json if packaging checks are enabled",
                "resolved_geometry.npz",
                "DUST basic mesh files: *_rr.dat and *_ee.dat",
                "ParaView legacy VTK file: cta_resolved_geometry.vtk",
                "dust_case_manifest.json linking variables, geometry, and aerodynamic results",
            ],
            "traceability_key": (
                "Use the DOE row index as case_id and preserve it in all generated DUST "
                "folders, logs, geometry files, and aerodynamic result tables."
            ),
        },
    }
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def _write_flat_dataset(dataset, path: Path) -> None:  # noqa: ANN001
    flat_dataset = dataset.copy()
    flat_columns = []
    for column in flat_dataset.columns:
        if isinstance(column, tuple):
            group, variable, component = column
            suffix = "" if str(component) in {"", "0"} else f".{component}"
            flat_columns.append(f"{group}.{variable}{suffix}")
        else:
            flat_columns.append(str(column))
    flat_dataset.columns = flat_columns
    flat_dataset.to_csv(path_or_buf=str(path), index=False)


def _validate_dust_executables() -> None:
    required_executables = (
        DUST_BIN_DIR / "dust_pre",
        DUST_BIN_DIR / "dust",
        DUST_BIN_DIR / "dust_post",
    )
    missing = [
        executable
        for executable in required_executables
        if not executable.exists()
    ]
    if missing:
        missing_text = ", ".join(str(path) for path in missing)
        msg = f"Missing DUST executables: {missing_text}"
        raise FileNotFoundError(msg)


def _remove_path(path: Path) -> None:
    if path.is_dir():
        shutil.rmtree(path)
    elif path.exists():
        path.unlink()


def _cleanup_previous_dust_runs() -> None:
    """Keep the DUST work directory small when iterating on CTA geometry."""

    if not (ENABLE_DUST_SMOKE_RUN and DUST_OVERWRITE_PREVIOUS_RUNS):
        return

    DUST_WORK_DIR.mkdir(parents=True, exist_ok=True)
    _remove_path(DUST_LATEST_RUN_DIR)
    for run_dir in DUST_WORK_DIR.glob("cta_dust_smoke_*"):
        _remove_path(run_dir)


def _promote_latest_dust_run() -> Path | None:
    """Move the newest temporary DUST run to a stable ParaView-friendly path."""

    if not (ENABLE_DUST_SMOKE_RUN and DUST_OVERWRITE_PREVIOUS_RUNS):
        return None

    run_dirs = [path for path in DUST_WORK_DIR.glob("cta_dust_smoke_*") if path.is_dir()]
    if not run_dirs:
        return None

    latest_run_dir = max(run_dirs, key=lambda path: path.stat().st_mtime)
    _remove_path(DUST_LATEST_RUN_DIR)
    latest_run_dir.rename(DUST_LATEST_RUN_DIR)
    for run_dir in DUST_WORK_DIR.glob("cta_dust_smoke_*"):
        _remove_path(run_dir)
    return DUST_LATEST_RUN_DIR


def _format_vtp_scalar(values: np.ndarray) -> str:
    return " ".join(f"{float(value):.9g}" for value in values)


def _format_vtp_vector(values: np.ndarray) -> str:
    return " ".join(
        f"{float(x):.9g} {float(y):.9g} {float(z):.9g}"
        for x, y, z in values
    )


def _write_particle_wake_vtp(
    path: Path,
    points: np.ndarray,
    vorticity: np.ndarray,
    vortex_radius: np.ndarray,
) -> None:
    n_points = points.shape[0]
    vorticity_magnitude = np.linalg.norm(vorticity, axis=1)
    abs_y = np.abs(points[:, 1])
    connectivity = np.arange(n_points, dtype=int)
    offsets = np.arange(1, n_points + 1, dtype=int)

    path.write_text(
        "\n".join(
            [
                '<?xml version="1.0"?>',
                '<VTKFile type="PolyData" version="0.1" byte_order="LittleEndian">',
                "  <PolyData>",
                (
                    f'    <Piece NumberOfPoints="{n_points}" NumberOfVerts="{n_points}" '
                    'NumberOfLines="0" NumberOfStrips="0" NumberOfPolys="0">'
                ),
                '      <PointData Vectors="Vorticity" Scalars="Vorticity_Magnitude">',
                (
                    '        <DataArray type="Float64" Name="Vorticity" '
                    'NumberOfComponents="3" format="ascii">'
                ),
                f"          {_format_vtp_vector(vorticity)}",
                "        </DataArray>",
                (
                    '        <DataArray type="Float64" Name="Vorticity_Magnitude" '
                    'format="ascii">'
                ),
                f"          {_format_vtp_scalar(vorticity_magnitude)}",
                "        </DataArray>",
                '        <DataArray type="Float64" Name="VortexRad" format="ascii">',
                f"          {_format_vtp_scalar(vortex_radius)}",
                "        </DataArray>",
                '        <DataArray type="Float64" Name="AbsY" format="ascii">',
                f"          {_format_vtp_scalar(abs_y)}",
                "        </DataArray>",
                "      </PointData>",
                "      <Points>",
                (
                    '        <DataArray type="Float64" NumberOfComponents="3" '
                    'format="ascii">'
                ),
                f"          {_format_vtp_vector(points)}",
                "        </DataArray>",
                "      </Points>",
                "      <Verts>",
                (
                    '        <DataArray type="Int32" Name="connectivity" '
                    'format="ascii">'
                ),
                f"          {' '.join(str(int(value)) for value in connectivity)}",
                "        </DataArray>",
                '        <DataArray type="Int32" Name="offsets" format="ascii">',
                f"          {' '.join(str(int(value)) for value in offsets)}",
                "        </DataArray>",
                "      </Verts>",
                "    </Piece>",
                "  </PolyData>",
                "</VTKFile>",
                "",
            ],
        ),
    )


def _export_dust_particle_wake_vtp(run_dir: Path | None) -> Path | None:
    if run_dir is None:
        return None

    try:
        import h5py
    except ImportError:
        return None

    output_dir = run_dir / "output"
    post_dir = run_dir / "post"
    result_files = sorted(output_dir.glob("cta_dust_smoke__res_*.h5"))
    if not result_files:
        return None

    result_file = result_files[-1]
    result_id = result_file.stem.rsplit("_res_", maxsplit=1)[-1]
    with h5py.File(result_file, "r") as h5_file:
        particle_wake = h5_file["ParticleWake"]
        points = np.asarray(particle_wake["WakePoints"], dtype=float)
        vorticity = np.asarray(particle_wake["WakeVort"], dtype=float)
        vortex_radius = np.asarray(particle_wake["VortexRad"], dtype=float)

    if points.size == 0:
        return None

    post_dir.mkdir(parents=True, exist_ok=True)
    particle_vtp = post_dir / f"cta_dust_smoke__wake_particles_vector-{result_id}.vtp"
    _write_particle_wake_vtp(particle_vtp, points, vorticity, vortex_radius)
    return particle_vtp


def main() -> None:
    if ENABLE_DUST_SMOKE_RUN:
        _validate_dust_executables()
        _cleanup_previous_dust_runs()

    scenario = build_cta_geometry_doe_scenario()
    scenario.scenario.execute(
        algo_name=DOE_ALGO,
        n_samples=N_SAMPLES,
    )
    latest_dust_run_dir = _promote_latest_dust_run()
    _export_dust_particle_wake_vtp(latest_dust_run_dir)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    dataset = scenario.scenario.to_dataset(opt_naming=False)
    dataset_path = OUTPUT_DIR / "cta_geom_doe_dataset.csv"
    flat_dataset_path = OUTPUT_DIR / "cta_geom_doe_dataset_flat.csv"
    design_space_path = OUTPUT_DIR / "cta_geom_doe_design_space.csv"
    dust_manifest_path = OUTPUT_DIR / "cta_geom_doe_dust_manifest.json"
    dataset.to_csv(path_or_buf=str(dataset_path))
    _write_flat_dataset(dataset, flat_dataset_path)
    _write_design_space(design_space_path)
    _write_dust_manifest(
        dust_manifest_path,
        dataset_path,
        flat_dataset_path,
        design_space_path,
    )

    print("CTA geometry DOE completed")
    print(f"  algo = {DOE_ALGO}")
    print(f"  n_samples = {N_SAMPLES}")
    print(f"  design_variables = {len(CTA_DOE_DESIGN_VARIABLES)}")
    print(f"  internal_boxes_enabled = {ENABLE_INTERNAL_BOX_EVALUATION}")
    print(f"  dust_smoke_enabled = {ENABLE_DUST_SMOKE_RUN}")
    print(f"  dataset = {dataset_path}")
    print(f"  flat_dataset = {flat_dataset_path}")
    print(f"  design_space = {design_space_path}")
    print(f"  dust_manifest = {dust_manifest_path}")
    print(f"  paraview_vtk = {DUST_MESH_VTK_PATH}")
    if latest_dust_run_dir is not None:
        print(f"  latest_dust_run = {latest_dust_run_dir}")


if __name__ == "__main__":
    main()
