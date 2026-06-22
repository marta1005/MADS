"""GEMSEO DOE over the CTA 14-variable design space evaluated with DUST lifting-line.

Usage examples
--------------
Generate 10 000 Sobol samples and run them with NeuralFoil polars::

    python examples/cta_dust_lifting_line_doe.py --n-samples 10000 --algo OT_SOBOL

Run only the baseline geometry::

    python examples/cta_dust_lifting_line_doe.py --baseline-only

Generate a sample CSV without running DUST (for HPC pre-generation)::

    python examples/cta_dust_lifting_line_doe.py --generate-samples-only --n-samples 10000

Run a specific shard from a pre-generated CSV::

    python examples/cta_dust_lifting_line_doe.py \\
        --samples-csv /path/to/samples.csv --sample-start 0 --sample-count 500
"""

from __future__ import annotations

import argparse
import logging
import shutil
from pathlib import Path

import gemseo
import numpy as np

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
    build_neuralfoil_polar_provider,
    dust_case_tag,
    dust_executable,
    lifting_line_solver_preset,
)


gemseo.configure_logger(
    level=logging.INFO,
    filename="gemseo_ll_doe.log",
    filemode="w",
)


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = REPO_ROOT / "outputs" / "CTA_case" / "doe_dust_ll"
DEFAULT_SAMPLES_CSV = (
    REPO_ROOT
    / "outputs"
    / "CTA_case"
    / "datasets"
    / "campaign_ll_exploration"
    / "samples"
    / "cta_dust_ll_samples.csv"
)
SUPPORTED_SAMPLE_METHODS = ("sobol", "lhs", "halton", "random")
SUPPORTED_POLAR_SOURCES = ("neuralfoil", "synthetic")
SUPPORTED_LL_PRESETS = ("conservative", "nominal", "fast")


def _geometry_provider():
    resolved_wing = cta.disc_geometry.components[0]
    return getattr(resolved_wing, "geometry_state", None)


def _build_environment(args: argparse.Namespace) -> Environment:
    altitude_m = args.altitude_ft * 0.3048
    probe = Environment(name="env", height=altitude_m, speed=1.0)
    return Environment(
        name="env",
        height=altitude_m,
        speed=args.mach * float(probe.sound_speed),
        alpha=args.alpha_deg,
    )


def _build_dust_options(args: argparse.Namespace, environment: Environment) -> Options:
    n_steps = int(args.n_steps)
    preset = lifting_line_solver_preset(args.ll_preset)
    return Options(
        name=f"cta_ll_{dust_case_tag(environment.alpha)}",
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
        ll_solver=preset.get("ll_solver", "GammaMethod"),
        ll_damp=preset.get("ll_damp", 5.0),
        ll_reynolds_corrections=preset.get("ll_reynolds_corrections", False),
        ll_stall_regularisation=preset.get("ll_stall_regularisation", True),
        ll_artificial_viscosity=preset.get("ll_artificial_viscosity", 0.0),
        ll_loads_avl=preset.get("ll_loads_avl", False),
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


def _build_wing_options(args: argparse.Namespace) -> WingOptions:
    n_steps = int(args.n_steps)
    return WingOptions(
        discretization_method=WingMethod.LIFTING_LINE,
        output_options=OutputOptions(
            compute_loads=True,
            loads_start=max(1, n_steps - 20),
            loads_end=n_steps,
            loads_step=1,
            loads_avg=True,
            loads_reference="0",
        ),
    )


def _build_mesh_settings(args: argparse.Namespace) -> DustMeshSettings:
    return DustMeshSettings(
        n_span_stations=args.mesh_span_stations,
        n_chord_stations=1,
        lifting_line_geometry_mode=args.ll_geometry_mode,
        lifting_line_effective_start_y_m=args.ll_effective_start_y_m,
        lifting_line_outer_start_y_m=args.ll_outer_start_y_m,
        lifting_line_blend_root_to_start=not args.no_ll_root_blend,
        lifting_line_anchor_min_spacing_m=args.ll_anchor_min_spacing_m,
        mirror_span=True,
    )


def _build_dust_discipline(args: argparse.Namespace) -> ResolvedGeometryDustDiscipline:
    environment = _build_environment(args)
    options = _build_dust_options(args, environment)
    wing_options = _build_wing_options(args)
    mesh_settings = _build_mesh_settings(args)
    polar_provider = build_neuralfoil_polar_provider(
        source=args.polar_source,
        model=args.neuralfoil_model,
        n_crit=float(args.neuralfoil_n_crit),
        sanitize=True,
        cd_min=float(args.polar_cd_min),
    )
    return ResolvedGeometryDustDiscipline(
        name="CTADustLiftingLineFromResolvedGeometry",
        geometry_provider=_geometry_provider,
        metric_inputs=cta.GEOMETRY_METRIC_VARIABLES,
        output_dir=args.output_dir / "cases",
        environment=environment,
        options=options,
        wing_options=wing_options,
        mesh_settings=mesh_settings,
        polar_provider=polar_provider,
        fail_fast=args.fail_fast,
        reuse_run_directory=not args.store_case_directories,
        run_directory_name=args.run_directory_name,
    )


def build_cta_dust_ll_doe_scenario(
    args: argparse.Namespace,
) -> tuple[MADSScenario, ResolvedGeometryDustDiscipline]:
    """Create the GEMSEO DOE scenario for CTA geometry + DUST lifting-line."""

    dust_discipline = _build_dust_discipline(args)
    mads_scenario = MADSScenario()
    mads_scenario.fill_parameter_space(cta.CTA_DOE_DESIGN_VARIABLES)
    disciplines = [
        cta.disc_planform_mapping,
        cta.disc_geometry,
    ]
    if cta.disc_internal_boxes is not None:
        disciplines.append(cta.disc_internal_boxes)
    disciplines.extend(
        [
            cta.disc_geometry_validation,
            dust_discipline,
        ],
    )

    mads_scenario.create_scenario(
        disciplines=disciplines,
        formulation="DisciplinaryOpt",
        objective_name="cta_dust_failure_code",
        scenario_type="DOE",
        name="cta_dust_ll_doe",
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
        *(variable.name for variable in cta.GEOMETRY_METRIC_VARIABLES),
        *(
            variable.name
            for variable in cta.validation_result_variables
            if variable.name != cta.geometry_valid.name
        ),
        *(variable.name for variable in cta.box_result_variables),
    ]
    for observable in observables:
        mads_scenario.scenario.add_observable(observable)

    return mads_scenario, dust_discipline


def _execute_scenario(scenario: MADSScenario, args: argparse.Namespace) -> int:
    if args.samples_csv is not None and args.baseline_only:
        msg = "--samples-csv and --baseline-only are mutually exclusive."
        raise ValueError(msg)

    if args.samples_csv is not None:
        samples = cta.load_design_sample_slice(
            args.samples_csv,
            sample_start=args.sample_start,
            sample_count=args.sample_count,
        )
        scenario.scenario.execute(algo_name="CustomDOE", samples=samples)
        return int(samples.shape[0])

    if args.baseline_only:
        scenario.scenario.execute(
            algo_name="CustomDOE",
            samples=cta.baseline_sample(),
        )
        return 1

    scenario.scenario.execute(algo_name=args.algo, n_samples=args.n_samples)
    return int(args.n_samples)


def _analysis_settings_rows(args: argparse.Namespace) -> list[list[object]]:
    return [
        ["key", "value"],
        ["dust_method", "lifting_line"],
        ["ll_preset", args.ll_preset],
        ["ll_geometry_mode", args.ll_geometry_mode],
        ["polar_source", args.polar_source],
        ["neuralfoil_model", args.neuralfoil_model],
        ["n_steps", int(args.n_steps)],
        ["executed_samples", int(getattr(args, "executed_n_samples", 1 if args.baseline_only else args.n_samples))],
        ["samples_csv", "" if args.samples_csv is None else str(Path(args.samples_csv).resolve())],
        ["sample_start", int(args.sample_start)],
        ["sample_count", "" if args.sample_count is None else int(args.sample_count)],
        ["alpha_deg", float(args.alpha_deg)],
        ["mach", float(args.mach)],
        ["altitude_ft", float(args.altitude_ft)],
    ]


def _build_manifest(
    args: argparse.Namespace,
    paths: CampaignExportPaths,
) -> dict[str, object]:
    return {
        "purpose": "CTA 14-variable geometry DOE connected to DUST lifting-line.",
        "architecture": {
            "doe_script": "examples/cta_dust_lifting_line_doe.py",
            "geometry_case": "examples/cta_geometry.py",
            "dust_adapter": "multiads.solvers.aerodynamics.dust_lib.ResolvedGeometryDustDiscipline",
            "polar_provider": "multiads.solvers.aerodynamics.dust_lib.build_neuralfoil_polar_provider",
        },
        "algorithm": "CustomDOE" if args.baseline_only or args.samples_csv else args.algo,
        "n_samples": int(getattr(args, "executed_n_samples", 1 if args.baseline_only else args.n_samples)),
        "sample_source": {
            "samples_csv": None if args.samples_csv is None else str(Path(args.samples_csv).resolve()),
            "sample_start": int(args.sample_start),
            "sample_count": None if args.sample_count is None else int(args.sample_count),
        },
        "alpha_deg": float(args.alpha_deg),
        "mach": float(args.mach),
        "altitude_ft": float(args.altitude_ft),
        "lifting_line": {
            "ll_preset": args.ll_preset,
            "ll_geometry_mode": args.ll_geometry_mode,
            "ll_effective_start_y_m": float(args.ll_effective_start_y_m),
            "ll_outer_start_y_m": float(args.ll_outer_start_y_m),
            "no_ll_root_blend": bool(args.no_ll_root_blend),
            "ll_anchor_min_spacing_m": float(args.ll_anchor_min_spacing_m),
            "mesh_span_stations": int(args.mesh_span_stations),
        },
        "polars": {
            "source": args.polar_source,
            "neuralfoil_model": args.neuralfoil_model,
            "neuralfoil_n_crit": float(args.neuralfoil_n_crit),
            "polar_cd_min": float(args.polar_cd_min),
        },
        "dust": {
            "n_steps": int(args.n_steps),
            "dt": float(args.dt),
            "n_threads": int(args.n_threads),
            "n_wake_particles": int(args.n_wake_particles),
            "write_visualization": not args.no_vtk,
            "store_case_directories": bool(args.store_case_directories),
            "run_directory_name": args.run_directory_name,
        },
        "design_variables": [variable.name for variable in cta.CTA_DOE_DESIGN_VARIABLES],
        "outputs": {
            "dataset_csv": str(paths.dataset_csv),
            "flat_dataset_csv": str(paths.flat_dataset_csv),
            "campaign_workbook_xlsx": str(paths.workbook_xlsx),
            "design_space_csv": str(paths.design_space_csv),
        },
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run a GEMSEO DOE over the CTA 14-variable design space and evaluate "
            "each sample with DUST lifting-line aerodynamics."
        ),
    )

    # --- Sampling ---
    parser.add_argument("--n-samples", type=int, default=1, help="Number of DOE samples.")
    parser.add_argument("--algo", default="OT_SOBOL", help="GEMSEO DOE algorithm, e.g. OT_SOBOL, LHS.")
    parser.add_argument("--samples-csv", type=Path, default=None, help="Run a fixed DOE sample CSV.")
    parser.add_argument("--sample-start", type=int, default=0, help="First row from --samples-csv (HPC sharding).")
    parser.add_argument("--sample-count", type=int, default=None, help="Number of rows from --samples-csv.")
    parser.add_argument("--baseline-only", action="store_true", help="Run one sample at the CTA baseline.")
    parser.add_argument("--generate-samples-only", action="store_true", help="Generate sample CSV and exit.")
    parser.add_argument("--sample-method", choices=SUPPORTED_SAMPLE_METHODS, default="sobol")
    parser.add_argument("--sample-seed", type=int, default=17)

    # --- Flight condition ---
    parser.add_argument("--alpha-deg", type=float, default=3.0, help="Angle of attack in degrees.")
    parser.add_argument("--mach", type=float, default=0.8, help="Freestream Mach number.")
    parser.add_argument("--altitude-ft", type=float, default=40000.0, help="Altitude in feet.")

    # --- Lifting-line geometry ---
    parser.add_argument("--ll-preset", choices=SUPPORTED_LL_PRESETS, default="nominal", help="LL solver preset.")
    parser.add_argument(
        "--ll-geometry-mode",
        default="resolved",
        help="LL station selection mode (resolved, equivalent, outer_only, ...).",
    )
    parser.add_argument("--ll-effective-start-y-m", type=float, default=12.5, help="LL section extraction start Y (m).")
    parser.add_argument("--ll-outer-start-y-m", type=float, default=22.5, help="LL outer section start Y (m).")
    parser.add_argument("--no-ll-root-blend", action="store_true", help="Disable synthetic root station blending.")
    parser.add_argument("--ll-anchor-min-spacing-m", type=float, default=1.5, help="Minimum anchor station spacing (m).")
    parser.add_argument("--mesh-span-stations", type=int, default=49, help="Number of LL span stations.")

    # --- Polars ---
    parser.add_argument("--polar-source", choices=SUPPORTED_POLAR_SOURCES, default="neuralfoil")
    parser.add_argument("--neuralfoil-model", default="large", help="NeuralFoil model size.")
    parser.add_argument("--neuralfoil-n-crit", type=float, default=9.0, help="NeuralFoil n-crit.")
    parser.add_argument("--polar-cd-min", type=float, default=0.006, help="Minimum CD floor for section polars.")

    # --- DUST execution ---
    parser.add_argument("--n-steps", type=int, default=80, help="Number of DUST time steps.")
    parser.add_argument("--dt", type=float, default=0.005, help="DUST time step size.")
    parser.add_argument("--n-threads", type=int, default=1)
    parser.add_argument("--n-wake-particles", type=int, default=100000)
    parser.add_argument("--dust-bin-dir", default=None, help="Directory containing DUST executables.")
    parser.add_argument("--no-vtk", action="store_true", help="Skip VTK output.")
    parser.add_argument("--fail-fast", action="store_true", help="Abort DOE if one DUST case fails.")
    parser.add_argument(
        "--store-case-directories",
        action="store_true",
        help="Keep one DUST run directory per sample instead of overwriting.",
    )
    parser.add_argument("--run-directory-name", default="run")

    # --- Output ---
    parser.add_argument("--keep-existing", action="store_true", help="Do not delete previous output directory.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)

    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    args.output_dir = args.output_dir.resolve()

    if args.generate_samples_only:
        samples_csv = (
            Path(args.samples_csv).expanduser()
            if args.samples_csv is not None
            else DEFAULT_SAMPLES_CSV
        )
        samples_csv = samples_csv.resolve()
        cta.write_design_samples_csv(
            samples_csv,
            method=args.sample_method,
            n_samples=args.n_samples,
            seed=args.sample_seed,
        )
        print("CTA DUST LL DOE samples generated")
        print(f"  method = {args.sample_method}")
        print(f"  n_samples = {args.n_samples}")
        print(f"  seed = {args.sample_seed}")
        print(f"  samples_csv = {samples_csv}")
        return

    if args.output_dir.exists() and not args.keep_existing:
        shutil.rmtree(args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    scenario, _dust_discipline = build_cta_dust_ll_doe_scenario(args)
    args.executed_n_samples = _execute_scenario(scenario, args)

    dataset = scenario.scenario.to_dataset(opt_naming=False)
    export_paths = campaign_export_paths(args.output_dir, "cta_dust_ll_doe")
    manifest = _build_manifest(args, export_paths)
    export_paths = write_campaign_results(
        output_dir=args.output_dir,
        file_prefix="cta_dust_ll_doe",
        dataset=dataset,
        design_space_rows=cta.design_space_rows(),
        analysis_settings_rows=_analysis_settings_rows(args),
        manifest=manifest,
        paths=export_paths,
    )

    print("CTA DUST Lifting-Line DOE completed")
    print(f"  algo = {'CustomDOE' if args.baseline_only or args.samples_csv else args.algo}")
    print(f"  ll_preset = {args.ll_preset}")
    print(f"  ll_geometry_mode = {args.ll_geometry_mode}")
    print(f"  polar_source = {args.polar_source}")
    print(f"  n_steps = {args.n_steps}")
    print(f"  executed_samples = {args.executed_n_samples}")
    if args.samples_csv is not None:
        print(f"  samples_csv = {Path(args.samples_csv).resolve()}")
        print(f"  sample_slice = [{args.sample_start}, {args.sample_start + args.executed_n_samples})")
    print(f"  alpha_deg = {args.alpha_deg:g}")
    print(f"  output_dir = {args.output_dir}")
    print(f"  flat_dataset = {export_paths.flat_dataset_csv}")
    print(f"  workbook = {export_paths.workbook_xlsx}")
    print(f"  manifest = {export_paths.manifest_json}")


if __name__ == "__main__":
    main()
