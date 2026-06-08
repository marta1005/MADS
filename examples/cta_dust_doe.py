from __future__ import annotations

import argparse
import logging
import shutil
from pathlib import Path

import gemseo
import numpy as np

import cta_geom_doe as geometry_doe
import cta_geometry as cta
from multiads.assembly import Environment
from multiads.scenario import MADSScenario
from multiads.utilities.campaign_export import (
    CampaignExportPaths,
    campaign_export_paths,
    write_campaign_results,
)
from multiads.solvers.aerodynamics.dust_lib import (
    DustMeshSettings,
    Options,
    OutputOptions,
    ResolvedGeometryDustDiscipline,
    WingMethod,
    WingOptions,
    WingPanelType,
    dust_case_tag,
    dust_executable,
)


gemseo.configure_logger(
    level=logging.INFO,
    filename="gemseo.log",
    filemode="w",
)


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = REPO_ROOT / "outputs" / "CTA_case" / "doe_dust"


def _geometry_provider():
    resolved_wing = cta.disc_geometry.components[0]
    return getattr(resolved_wing, "geometry_state", None)


def _build_environment(args: argparse.Namespace) -> Environment:
    altitude_m = args.altitude_ft * 0.3048
    probe = Environment(name="env", height=altitude_m, speed=1.0)
    environment = Environment(
        name="env",
        height=altitude_m,
        speed=args.mach * float(probe.sound_speed),
        alpha=args.alpha_deg,
    )
    environment.disa_k = float(args.disa_k)
    return environment


def _build_dust_options(args: argparse.Namespace, environment: Environment) -> Options:
    n_steps = int(args.n_steps)
    return Options(
        name=f"cta_{dust_case_tag(environment.alpha)}",
        dust_pre=dust_executable("dust_pre", args.dust_bin_dir),
        dust=dust_executable("dust", args.dust_bin_dir),
        dust_post=dust_executable("dust_post", args.dust_bin_dir),
        output_dir=Path("Output"),
        post_dir=Path("post"),
        keep_run_directory=True,
        t_start=0.0,
        t_end=n_steps * float(args.dt),
        dt=float(args.dt),
        dt_out=float(args.dt),
        output_start=True,
        n_threads=int(args.n_threads),
        n_wake_panels=n_steps,
        n_wake_particles=int(args.n_wake_particles),
        particles_box_min=np.asarray((-80.0, -90.0, -80.0), dtype=float),
        particles_box_max=np.asarray((380.0, 90.0, 80.0), dtype=float),
        penetration_avoidance=False,
        output_options=OutputOptions(
            visualization=not args.no_vtk,
            viz_start=n_steps,
            viz_end=n_steps,
            viz_step=1,
            viz_fmt="vtk",
            viz_wake=True,
            viz_separate_wake=True,
            viz_variables=["cp", "vorticity_vector", "velocity"],
        ),
    )


def _build_wing_options(args: argparse.Namespace, environment: Environment) -> WingOptions:
    n_steps = int(args.n_steps)
    velocity = np.asarray(environment.velocity, dtype=float)
    speed = max(float(np.linalg.norm(velocity)), 1.0e-14)
    return WingOptions(
        discretization_method=WingMethod.PANELS,
        panel_type=WingPanelType.UNIFORM,
        num_panels=1,
        mesh_file=Path("geometry") / "cta_basic_",
        mesh_file_type="basic",
        inner_product_te=0.5,
        tol_se_wing=1.0e-3,
        proj_te=True,
        proj_te_dir="parallel",
        proj_te_vector=velocity / speed,
        output_options=OutputOptions(
            compute_loads=True,
            loads_start=max(1, n_steps - 20),
            loads_end=n_steps,
            loads_step=1,
            loads_avg=True,
            loads_reference="0",
        ),
    )


def _build_dust_discipline(args: argparse.Namespace) -> ResolvedGeometryDustDiscipline:
    environment = _build_environment(args)
    options = _build_dust_options(args, environment)
    wing_options = _build_wing_options(args, environment)
    mesh_settings = DustMeshSettings(
        n_span_stations=args.mesh_span_stations,
        n_chord_stations=args.mesh_chord_stations,
        span_spacing=args.span_spacing,
        chord_spacing=args.chord_spacing,
        span_curvature_weight=args.span_curvature_weight,
        chord_curvature_weight=args.chord_curvature_weight,
        leading_edge_opening_m=args.leading_edge_opening_m,
        mirror_span=True,
    )
    return ResolvedGeometryDustDiscipline(
        name="CTADustFromResolvedGeometry",
        geometry_provider=_geometry_provider,
        metric_inputs=geometry_doe.GEOMETRY_METRIC_VARIABLES,
        output_dir=args.output_dir / "cases",
        environment=environment,
        options=options,
        wing_options=wing_options,
        mesh_settings=mesh_settings,
        fail_fast=args.fail_fast,
        reuse_run_directory=not args.store_case_directories,
        run_directory_name=args.run_directory_name,
    )


def build_cta_dust_doe_scenario(
    args: argparse.Namespace,
) -> tuple[MADSScenario, ResolvedGeometryDustDiscipline]:
    """Create the GEMSEO DOE scenario for CTA geometry plus DUST."""

    dust_discipline = _build_dust_discipline(args)
    mads_scenario = MADSScenario()
    mads_scenario.fill_parameter_space(cta.CTA_CFD_JUNE_14_DESIGN_VARIABLES)
    disciplines = [
        geometry_doe.disc_planform_mapping,
        cta.disc_geometry,
    ]
    if geometry_doe.disc_internal_boxes is not None:
        disciplines.append(geometry_doe.disc_internal_boxes)
    disciplines.extend(
        [
            geometry_doe.disc_geometry_validation,
            dust_discipline,
        ],
    )

    mads_scenario.create_scenario(
        disciplines=disciplines,
        formulation="DisciplinaryOpt",
        objective_name="cta_dust_failure_code",
        scenario_type="DOE",
        name="cta_dust_doe",
        maximize_objective=False,
    )

    observables = [
        "cta_dust_success",
        "cta_dust_cl",
        "cta_dust_cd",
        "cta_dust_cm",
        "cta_dust_cy",
        "cta_dust_ld",
        "cta_dust_lift_n",
        "cta_dust_drag_n",
        "cta_dust_side_n",
        "cta_dust_fx_reference_n",
        "cta_dust_fy_reference_n",
        "cta_dust_fz_reference_n",
        "cta_dust_mx_reference_nm",
        "cta_dust_my_reference_nm",
        "cta_dust_mz_reference_nm",
        *(variable.name for variable in geometry_doe.GEOMETRY_METRIC_VARIABLES),
        *(
            variable.name
            for variable in geometry_doe.validation_result_variables
            if variable.name != geometry_doe.geometry_valid.name
        ),
        *(variable.name for variable in geometry_doe.box_result_variables),
    ]
    for observable in observables:
        mads_scenario.scenario.add_observable(observable)

    return mads_scenario, dust_discipline


def _execute_scenario(scenario: MADSScenario, args: argparse.Namespace) -> None:
    if args.baseline_only:
        baseline_sample = np.asarray(
            [[float(variable.value) for variable in cta.CTA_CFD_JUNE_14_DESIGN_VARIABLES]],
            dtype=float,
        )
        scenario.scenario.execute(
            algo_name="CustomDOE",
            samples=baseline_sample,
        )
        return

    scenario.scenario.execute(
        algo_name=args.algo,
        n_samples=args.n_samples,
    )


def _design_space_rows() -> list[dict[str, float | str]]:
    rows: list[dict[str, float | str]] = []
    for variable in cta.CTA_CFD_JUNE_14_DESIGN_VARIABLES:
        rows.append(
            {
                "name": variable.name,
                "baseline": float(variable.value),
                "lower_bound": float(variable.lb),
                "upper_bound": float(variable.ub),
            },
        )
    return rows


def _analysis_settings_rows(args: argparse.Namespace) -> list[list[object]]:
    return [
        ["key", "value"],
        ["n_steps", int(args.n_steps)],
        ["alpha_deg", float(args.alpha_deg)],
        ["mach", float(args.mach)],
        ["altitude_ft", float(args.altitude_ft)],
        ["disa_k", float(args.disa_k)],
        ["store_case_directories", bool(args.store_case_directories)],
        ["run_directory", str(args.output_dir / "cases" / args.run_directory_name)],
        [
            "note",
            (
                "DUST run files are overwritten by default; campaign history is "
                "stored in this workbook and CSV files."
            ),
        ],
    ]


def _build_manifest(
    args: argparse.Namespace,
    paths: CampaignExportPaths,
) -> dict[str, object]:
    return {
        "purpose": "CTA 14-variable geometry DOE connected to DUST.",
        "architecture": {
            "doe_script": "examples/cta_dust_doe.py",
            "geometry_case": "examples/cta_geometry.py",
            "geometry_mapping": "examples/cta_geom_doe.py::disc_planform_mapping",
            "geometry_core": "multiads.solvers.synthesis.geometry_lib",
            "dust_adapter": "multiads.solvers.aerodynamics.dust_lib.ResolvedGeometryDustDiscipline",
            "dust_mesh_writer": "multiads.solvers.aerodynamics.dust_lib.write_basic_two_skin_mesh_from_resolved_npz",
        },
        "algorithm": "CustomDOE" if args.baseline_only else args.algo,
        "n_samples": 1 if args.baseline_only else int(args.n_samples),
        "alpha_deg": float(args.alpha_deg),
        "mach": float(args.mach),
        "altitude_ft": float(args.altitude_ft),
        "disa_k": float(args.disa_k),
        "mesh": {
            "span_stations": int(args.mesh_span_stations),
            "chord_stations": int(args.mesh_chord_stations),
            "span_spacing": args.span_spacing,
            "chord_spacing": args.chord_spacing,
            "span_curvature_weight": float(args.span_curvature_weight),
            "chord_curvature_weight": float(args.chord_curvature_weight),
            "leading_edge_opening_m": float(args.leading_edge_opening_m),
            "mirror_span": True,
        },
        "dust": {
            "n_steps": int(args.n_steps),
            "dt": float(args.dt),
            "n_threads": int(args.n_threads),
            "n_wake_particles": int(args.n_wake_particles),
            "write_visualization": not args.no_vtk,
            "dust_bin_dir": args.dust_bin_dir,
            "force_axes": "reference",
            "store_case_directories": bool(args.store_case_directories),
            "run_directory_name": args.run_directory_name,
        },
        "objective_note": (
            "GEMSEO requires an objective for DOE scenarios. "
            "The script minimizes cta_dust_failure_code only as a technical "
            "execution status; this is not an aerodynamic optimization target."
        ),
        "design_variables": [
            variable.name for variable in cta.CTA_CFD_JUNE_14_DESIGN_VARIABLES
        ],
        "outputs": {
            "dataset_csv": str(paths.dataset_csv),
            "flat_dataset_csv": str(paths.flat_dataset_csv),
            "campaign_workbook_xlsx": str(paths.workbook_xlsx),
            "design_space_csv": str(paths.design_space_csv),
            "case_directory_policy": (
                "one directory per sample"
                if args.store_case_directories
                else f"single overwritten directory: {args.output_dir / 'cases' / args.run_directory_name}"
            ),
        },
        "notes": [
            (
                "DUST receives a derived panel mesh from the resolved geometry, "
                "not only section definitions."
            ),
            "CL, CD and CM are normalized using the resolved sample area and MAC.",
            "For this debugging/campaign setup, drag is read from the reference axes without additional wind-axis projection.",
        ],
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a GEMSEO DOE over the CTA 14-variable design space and evaluate each sample with DUST.",
    )
    parser.add_argument("--n-samples", type=int, default=1, help="Number of LHS/DOE samples.")
    parser.add_argument("--algo", default="LHS", help="GEMSEO DOE algorithm, e.g. LHS.")
    parser.add_argument(
        "--baseline-only",
        action="store_true",
        help="Run one CustomDOE sample at the CTA baseline instead of an LHS campaign.",
    )
    parser.add_argument("--alpha-deg", type=float, default=3.0, help="DUST angle of attack in degrees.")
    parser.add_argument("--mach", type=float, default=0.8, help="Freestream Mach number.")
    parser.add_argument("--altitude-ft", type=float, default=40000.0, help="Altitude in feet.")
    parser.add_argument("--disa-k", type=float, default=0.0, help="DISA temperature offset, recorded in outputs.")
    parser.add_argument("--mesh-span-stations", type=int, default=49, help="Half-span DUST mesh stations before mirroring.")
    parser.add_argument("--mesh-chord-stations", type=int, default=45, help="DUST chordwise stations per skin.")
    parser.add_argument("--span-spacing", choices=["uniform", "curvature"], default="curvature")
    parser.add_argument("--chord-spacing", choices=["uniform", "curvature"], default="curvature")
    parser.add_argument("--span-curvature-weight", type=float, default=5.0)
    parser.add_argument("--chord-curvature-weight", type=float, default=4.0)
    parser.add_argument("--leading-edge-opening-m", type=float, default=0.0)
    parser.add_argument("--n-steps", type=int, default=80)
    parser.add_argument("--dt", type=float, default=0.005)
    parser.add_argument("--n-threads", type=int, default=6)
    parser.add_argument("--n-wake-particles", type=int, default=100000)
    parser.add_argument("--dust-bin-dir", default=None, help="Directory containing dust_pre, dust and dust_post.")
    parser.add_argument("--no-vtk", action="store_true", help="Skip VTK wake/solution export.")
    parser.add_argument("--fail-fast", action="store_true", help="Stop the DOE if one DUST case fails.")
    parser.add_argument(
        "--store-case-directories",
        action="store_true",
        help="Keep one DUST run directory per sample instead of overwriting a single run directory.",
    )
    parser.add_argument(
        "--run-directory-name",
        default="run",
        help="Name of the overwritten DUST run directory when --store-case-directories is not used.",
    )
    parser.add_argument("--keep-existing", action="store_true", help="Do not delete the previous output directory.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Output directory for the DOE dataset and DUST cases.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    args.output_dir = args.output_dir.resolve()
    if args.output_dir.exists() and not args.keep_existing:
        shutil.rmtree(args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    scenario, _dust_discipline = build_cta_dust_doe_scenario(args)
    _execute_scenario(scenario, args)

    dataset = scenario.scenario.to_dataset(opt_naming=False)
    export_paths = campaign_export_paths(args.output_dir, "cta_dust_doe")
    manifest = _build_manifest(args, export_paths)
    export_paths = write_campaign_results(
        output_dir=args.output_dir,
        file_prefix="cta_dust_doe",
        dataset=dataset,
        design_space_rows=_design_space_rows(),
        analysis_settings_rows=_analysis_settings_rows(args),
        manifest=manifest,
        paths=export_paths,
    )

    print("CTA DUST DOE completed")
    print(f"  algo = {'CustomDOE' if args.baseline_only else args.algo}")
    print(f"  n_samples = {1 if args.baseline_only else args.n_samples}")
    print(f"  alpha_deg = {args.alpha_deg:g}")
    print(f"  output_dir = {args.output_dir}")
    print(f"  flat_dataset = {export_paths.flat_dataset_csv}")
    print(f"  workbook = {export_paths.workbook_xlsx}")
    print(f"  design_space = {export_paths.design_space_csv}")
    print(f"  manifest = {export_paths.manifest_json}")


if __name__ == "__main__":
    main()
