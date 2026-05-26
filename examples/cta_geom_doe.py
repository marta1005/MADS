from __future__ import annotations

import csv
import json
import logging
import re
from pathlib import Path

import gemseo
import numpy as np

import cta_geometry as cta
from multiads.disciplines import UserDefined
from multiads.scenario import MADSScenario, VariableFloat
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
RUN_BASELINE_SAMPLE = True
TRIANGLE_RESOLUTION = 8
ENABLE_INTERNAL_BOX_EVALUATION = False

REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = REPO_ROOT / "outputs" / "cta_geom_doe"
CTA_INTERNAL_VOLUME_CONSTRAINTS_PATH = (
    REPO_ROOT / "assets" / "cta" / "internal_volume_constraints_set1.csv"
)

CTA_DOE_DESIGN_VARIABLES = list(cta.CTA_CFD_JUNE_14_DESIGN_VARIABLES)

GEOMETRY_METRIC_VARIABLES = [
    cta.disc_geometry.solver.metric_outputs["cta_wing.span_m"],
    cta.disc_geometry.solver.metric_outputs["cta_wing.planform_area_m2"],
    cta.disc_geometry.solver.metric_outputs["cta_wing.enclosed_volume_m3"],
    cta.disc_geometry.solver.metric_outputs["cta_wing.root_chord_m"],
    cta.disc_geometry.solver.metric_outputs["cta_wing.tip_chord_m"],
    cta.disc_geometry.solver.metric_outputs["cta_wing.mean_aerodynamic_chord_m"],
]

cta_planform_mapping_outputs = [
    cta.chord_c0,
    cta.chord_s1,
    cta.chord_s1a,
    cta.chord_s2,
    cta.chord_c3,
    cta.chord_c4,
    cta.chord_s4a,
    cta.chord_s4b,
    cta.chord_c5,
    cta.y_s4,
    cta.y_s4a,
    cta.y_s4b,
    cta.y_s5,
    cta.le_x_s1a,
    cta.le_x_s3,
    cta.le_x_s4,
    cta.le_x_s4a,
    cta.le_x_s4b,
    cta.le_x_s5,
    cta.le_z_s0,
    cta.le_z_s1,
    cta.le_z_s1a,
    cta.le_z_s2,
    cta.le_z_s3,
    cta.le_z_s4,
    cta.le_z_s4a,
    cta.le_z_s4b,
    cta.le_z_s5,
    cta.twist_s0,
    cta.twist_s1,
    cta.twist_s1a,
    cta.twist_s2,
    cta.twist_s3,
    cta.te_s0,
    cta.te_s1,
    cta.te_s1a,
    cta.te_s2,
    cta.te_s3,
    cta.te_s4,
    cta.te_s4a,
    cta.te_s4b,
    cta.te_s5,
]

geometry_valid = VariableFloat("cta_geometry_valid", 1.0)
geometry_failure_code = VariableFloat("cta_geometry_failure_code", 0.0)
packaging_valid = VariableFloat("cta_packaging_valid", 1.0)
packaging_min_margin = VariableFloat("cta_packaging_min_margin_m", 0.0)
validation_result_variables = [
    geometry_valid,
    geometry_failure_code,
    packaging_valid,
    packaging_min_margin,
]


def _to_scalar(value) -> float:  # noqa: ANN001
    return float(np.ravel(value)[0])


def _cta_planform_design_mapping(  # noqa: PLR0913
    delta_c0_m,
    delta_c3_m,
    delta_c5_m,
    taper_ratio_midwing,
    rspan_midwing,
    span_wing_m,
    sweep_midwing_deg,
    sweep_outwing_deg,
    twist_s4_deg,
    twist_s4a_deg,
    twist_s4b_deg,
    twist_s5_deg,
    thickness_s4,
    thickness_s5,
):  # noqa: ANN001, ANN201
    derived = cta.derive_cfd_geometry_values(
        {
            "delta_c0_m": _to_scalar(delta_c0_m),
            "delta_c3_m": _to_scalar(delta_c3_m),
            "delta_c5_m": _to_scalar(delta_c5_m),
            "taper_ratio_midwing": _to_scalar(taper_ratio_midwing),
            "rspan_midwing": _to_scalar(rspan_midwing),
            "span_wing_m": _to_scalar(span_wing_m),
            "sweep_midwing_deg": _to_scalar(sweep_midwing_deg),
            "sweep_outwing_deg": _to_scalar(sweep_outwing_deg),
            "twist_s4_deg": _to_scalar(twist_s4_deg),
            "twist_s4a_deg": _to_scalar(twist_s4a_deg),
            "twist_s4b_deg": _to_scalar(twist_s4b_deg),
            "twist_s5_deg": _to_scalar(twist_s5_deg),
            "thickness_s4": _to_scalar(thickness_s4),
            "thickness_s5": _to_scalar(thickness_s5),
        }
    )
    chords = derived["chords_m"]
    y = derived["spanwise_y_m"]
    le_x = derived["leading_edge_x_m"]
    le_z = derived["leading_edge_z_m"]
    twist = derived["twist_deg"]
    te = derived["trailing_edge_thickness"]

    return (
        chords["s0"],
        chords["s1"],
        chords["s1a"],
        chords["s2"],
        chords["s3"],
        chords["s4"],
        chords["s4a"],
        chords["s4b"],
        chords["s5"],
        y["s4"],
        y["s4a"],
        y["s4b"],
        y["s5"],
        le_x["s1a"],
        le_x["s3"],
        le_x["s4"],
        le_x["s4a"],
        le_x["s4b"],
        le_x["s5"],
        le_z["s0"],
        le_z["s1"],
        le_z["s1a"],
        le_z["s2"],
        le_z["s3"],
        le_z["s4"],
        le_z["s4a"],
        le_z["s4b"],
        le_z["s5"],
        twist["s0"],
        twist["s1"],
        twist["s1a"],
        twist["s2"],
        twist["s3"],
        te["s0"],
        te["s1"],
        te["s1a"],
        te["s2"],
        te["s3"],
        te["s4"],
        te["s4a"],
        te["s4b"],
        te["s5"],
    )


def _load_cta_internal_volume_constraints():
    return load_internal_volume_constraint_set(
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


CTA_INTERNAL_VOLUME_CONSTRAINTS = (
    _load_cta_internal_volume_constraints()
    if ENABLE_INTERNAL_BOX_EVALUATION
    else None
)

all_boxes_fit = VariableFloat("cta_all_boxes_fit", 1.0)
internal_boxes_min_margin = VariableFloat("cta_internal_boxes_min_margin_m", 0.0)
box_result_variables: list[VariableFloat] = []
if CTA_INTERNAL_VOLUME_CONSTRAINTS is not None:
    box_result_variables.extend([all_boxes_fit, internal_boxes_min_margin])
    for idx, surface in enumerate(CTA_INTERNAL_VOLUME_CONSTRAINTS.surfaces, start=1):
        safe_label = re.sub(r"[^0-9a-zA-Z]+", "_", surface.label).strip("_").lower()
        box_result_variables.extend(
            [
                VariableFloat(f"cta_box_{idx:02d}_{safe_label}_fits", 0.0),
                VariableFloat(f"cta_box_{idx:02d}_{safe_label}_margin_m", 0.0),
            ]
        )


def _evaluate_internal_boxes_from_last_geometry(*_geometry_metrics):  # noqa: ANN002, ANN201
    if CTA_INTERNAL_VOLUME_CONSTRAINTS is None:
        msg = "CTA internal volume constraints are disabled."
        raise RuntimeError(msg)

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


def _validate_cta_geometry(*values):  # noqa: ANN002, ANN201
    span_m, area_m2, volume_m3, root_chord_m, tip_chord_m, mac_m = [
        _to_scalar(value)
        for value in values[:6]
    ]
    metrics = np.asarray(
        [span_m, area_m2, volume_m3, root_chord_m, tip_chord_m, mac_m],
        dtype=float,
    )
    positive_metrics = metrics[[0, 1, 3, 4, 5]]
    if not np.all(np.isfinite(metrics)) or np.any(positive_metrics <= 0.0):
        return 0.0, 1.0, 0.0, 0.0

    if ENABLE_INTERNAL_BOX_EVALUATION:
        boxes_fit = _to_scalar(values[6])
        min_margin_m = _to_scalar(values[7])
        if boxes_fit < 0.5:
            return 0.0, 2.0, 0.0, min_margin_m
        return 1.0, 0.0, 1.0, min_margin_m

    return 1.0, 0.0, 1.0, 0.0


disc_planform_mapping = UserDefined(
    name="CTAPlanformDesignMapping",
    inputs=CTA_DOE_DESIGN_VARIABLES,
    outputs=cta_planform_mapping_outputs,
    expression=_cta_planform_design_mapping,
)

disc_internal_boxes = (
    UserDefined(
        name="CTAInternalBoxFit",
        inputs=[cta.disc_geometry.solver.metric_outputs["cta_wing.enclosed_volume_m3"]],
        outputs=box_result_variables,
        expression=_evaluate_internal_boxes_from_last_geometry,
    )
    if CTA_INTERNAL_VOLUME_CONSTRAINTS is not None
    else None
)

validation_inputs = list(GEOMETRY_METRIC_VARIABLES)
if CTA_INTERNAL_VOLUME_CONSTRAINTS is not None:
    validation_inputs.extend([all_boxes_fit, internal_boxes_min_margin])

disc_geometry_validation = UserDefined(
    name="CTAGeometryValidation",
    inputs=validation_inputs,
    outputs=validation_result_variables,
    expression=_validate_cta_geometry,
)


def build_cta_geometry_doe_scenario() -> MADSScenario:
    """Create a solver-independent GEMSEO DOE scenario for CTA geometry."""

    mads_scenario = MADSScenario()
    mads_scenario.fill_parameter_space(CTA_DOE_DESIGN_VARIABLES)

    disciplines = [
        disc_planform_mapping,
        cta.disc_geometry,
    ]
    if disc_internal_boxes is not None:
        disciplines.append(disc_internal_boxes)
    disciplines.append(disc_geometry_validation)

    mads_scenario.create_scenario(
        disciplines=disciplines,
        formulation="DisciplinaryOpt",
        objective_name=geometry_valid.name,
        scenario_type="DOE",
        name="cta_geometry_doe",
        maximize_objective=True,
    )

    observables = [
        *(variable.name for variable in GEOMETRY_METRIC_VARIABLES),
        *(
            variable.name
            for variable in validation_result_variables
            if variable.name != geometry_valid.name
        ),
    ]
    if box_result_variables:
        observables.extend(variable.name for variable in box_result_variables)

    for observable in observables:
        mads_scenario.scenario.add_observable(observable)

    return mads_scenario


def _execute_cta_geometry_doe_scenario(scenario: MADSScenario) -> None:
    if RUN_BASELINE_SAMPLE:
        baseline_sample = np.asarray(
            [[float(variable.value) for variable in CTA_DOE_DESIGN_VARIABLES]],
            dtype=float,
        )
        scenario.scenario.execute(
            algo_name="CustomDOE",
            samples=baseline_sample,
        )
        return

    scenario.scenario.execute(
        algo_name=DOE_ALGO,
        n_samples=N_SAMPLES,
    )


def _write_design_space(path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
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


def _value_for_variable(row: dict[str, str], variable_name: str) -> float:
    for column in row:
        if column == variable_name or column.endswith(f".{variable_name}"):
            return float(row[column])
        negated_name = f"-{variable_name}"
        if column == negated_name or column.endswith(f".{negated_name}"):
            return -float(row[column])
    msg = f"Could not find variable '{variable_name}' in flattened DOE dataset."
    raise KeyError(msg)


def _write_validation_summary(flat_dataset_path: Path, path: Path) -> None:
    with flat_dataset_path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))

    valid_count = 0
    invalid_count = 0
    failure_counts: dict[str, int] = {}
    for row in rows:
        valid = _value_for_variable(row, geometry_valid.name) >= 0.5
        failure_code = str(int(round(_value_for_variable(row, geometry_failure_code.name))))
        if valid:
            valid_count += 1
        else:
            invalid_count += 1
            failure_counts[failure_code] = failure_counts.get(failure_code, 0) + 1

    summary = {
        "sample_count": len(rows),
        "valid_count": valid_count,
        "invalid_count": invalid_count,
        "failure_code_counts": failure_counts,
        "failure_code_map": {
            "0": "valid",
            "1": "invalid geometry metrics",
            "2": "internal packaging boxes do not fit",
        },
    }
    path.write_text(json.dumps(summary, indent=2), encoding="utf-8")


def _write_manifest(
    path: Path,
    dataset_path: Path,
    flat_dataset_path: Path,
    design_space_path: Path,
    validation_summary_path: Path,
) -> None:
    manifest = {
        "purpose": "CTA geometry DOE / baseline check without solver-specific logic.",
        "architecture": {
            "geometry_source": "examples/cta_geometry.py",
            "geometry_core": "multiads.solvers.synthesis.geometry_lib",
            "case_laws": "multiads.cases.cta_laws",
            "solver_specific_logic": "excluded from this script",
            "dust_next_home": "multiads.solvers.aerodynamics.dust* or a generic DUST exporter/adapter",
        },
        "dataset_csv": str(dataset_path),
        "flat_dataset_csv": str(flat_dataset_path),
        "design_space_csv": str(design_space_path),
        "validation_summary_json": str(validation_summary_path),
        "baseline_sample_enabled": RUN_BASELINE_SAMPLE,
        "doe_algorithm": "CustomDOE" if RUN_BASELINE_SAMPLE else DOE_ALGO,
        "n_samples": 1 if RUN_BASELINE_SAMPLE else N_SAMPLES,
        "objective_note": (
            "GEMSEO requires an objective name even for DOE scenarios. "
            "This script uses cta_geometry_valid as a technical DOE output, "
            "not as an optimization target. It is maximized only so invalid "
            "samples are never ranked above valid samples by GEMSEO summaries."
        ),
        "volume_policy": (
            "cta_wing.enclosed_volume_m3 is an observable/check quantity. "
            "It is not optimized in this script."
        ),
        "internal_box_evaluation_enabled": ENABLE_INTERNAL_BOX_EVALUATION,
        "internal_box_constraints_csv": (
            str(CTA_INTERNAL_VOLUME_CONSTRAINTS_PATH)
            if ENABLE_INTERNAL_BOX_EVALUATION
            else None
        ),
        "design_variables": [variable.name for variable in CTA_DOE_DESIGN_VARIABLES],
        "geometry_observables": [variable.name for variable in GEOMETRY_METRIC_VARIABLES],
        "validation_observables": [variable.name for variable in validation_result_variables],
        "box_observables": [variable.name for variable in box_result_variables],
        "failure_code_map": {
            "0": "valid",
            "1": "invalid geometry metrics",
            "2": "internal packaging boxes do not fit",
        },
        "outputs_not_generated_here": [
            "DUST input files",
            "DUST launcher files",
            "DUST wake or panel postprocessing",
            "DUST polar plots",
            "IGES export",
        ],
        "related_scripts": {
            "baseline_geometry_and_iges": "examples/cta_geometry.py",
            "future_single_dust_case": "to be added after the DUST adapter is separated",
        },
    }
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def main() -> None:
    scenario = build_cta_geometry_doe_scenario()
    _execute_cta_geometry_doe_scenario(scenario)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    dataset = scenario.scenario.to_dataset(opt_naming=False)
    dataset_path = OUTPUT_DIR / "cta_geom_doe_dataset.csv"
    flat_dataset_path = OUTPUT_DIR / "cta_geom_doe_dataset_flat.csv"
    design_space_path = OUTPUT_DIR / "cta_geom_doe_design_space.csv"
    manifest_path = OUTPUT_DIR / "cta_geom_doe_manifest.json"
    validation_summary_path = OUTPUT_DIR / "cta_geom_doe_validation_summary.json"

    dataset.to_csv(path_or_buf=str(dataset_path))
    _write_flat_dataset(dataset, flat_dataset_path)
    _write_design_space(design_space_path)
    _write_validation_summary(flat_dataset_path, validation_summary_path)
    _write_manifest(
        manifest_path,
        dataset_path,
        flat_dataset_path,
        design_space_path,
        validation_summary_path,
    )

    print("CTA geometry DOE completed")
    print(f"  algo = {'CustomDOE' if RUN_BASELINE_SAMPLE else DOE_ALGO}")
    print(f"  baseline_sample = {RUN_BASELINE_SAMPLE}")
    print(f"  n_samples = {1 if RUN_BASELINE_SAMPLE else N_SAMPLES}")
    print(f"  design_variables = {len(CTA_DOE_DESIGN_VARIABLES)}")
    print(f"  internal_boxes_enabled = {ENABLE_INTERNAL_BOX_EVALUATION}")
    print("  dust_enabled = False")
    print(f"  dataset = {dataset_path}")
    print(f"  flat_dataset = {flat_dataset_path}")
    print(f"  design_space = {design_space_path}")
    print(f"  validation_summary = {validation_summary_path}")
    print(f"  manifest = {manifest_path}")
    print("  iges_export = outputs/cta_geometry_export (generated by cta_geometry.py)")


if __name__ == "__main__":
    main()
