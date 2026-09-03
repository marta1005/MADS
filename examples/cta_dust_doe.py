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
    WingPanelType,
    dust_case_tag,
    dust_executable,
)


gemseo.configure_logger(
    level=logging.INFO,
    filename="gemseo.log",
    filemode="w",
)

_log = logging.getLogger(__name__)


def _make_fault_tolerant(discipline, label: str):
    """Wrap discipline._run so any non-ValueError exception becomes ValueError.

    GEMSEO 6.x CustomDOE only catches ValueError to skip failed samples and
    continue the DOE.  Any other exception (RuntimeError, OSError, PyGeo crash,
    …) would kill the entire shard.  This wrapper converts those to ValueError
    so the sample is skipped and the next one proceeds.

    The DUST discipline handles its own exceptions internally (fail_fast=False),
    so it does not need this wrapper — only the upstream disciplines do.
    """
    original_run = discipline._run

    def _safe_run(input_data):
        try:
            return original_run(input_data)
        except ValueError:
            raise
        except Exception as exc:
            _log.exception(
                "[%s] evaluation failed; sample will be skipped: %s", label, exc
            )
            raise ValueError(f"[{label}] evaluation failed: {exc}") from exc

    discipline._run = _safe_run
    return discipline


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = REPO_ROOT / "outputs" / "CTA_case" / "doe_dust"
DEFAULT_SAMPLES_CSV = (
    REPO_ROOT
    / "outputs"
    / "CTA_case"
    / "datasets"
    / "campaign_001_exploration"
    / "samples"
    / "cta_dust_vlm_samples.csv"
)
SUPPORTED_DUST_METHODS = ("vlm", "panels")
SUPPORTED_SAMPLE_METHODS = ("sobol", "lhs", "halton", "random")


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
            visualization=True,
            viz_start=n_steps,
            viz_end=n_steps,
            viz_step=1,
            viz_fmt="vtk",
            viz_wake=False,
            viz_separate_wake=False,
            viz_variables=["cp"],
        ),
    )


def _build_wing_options(args: argparse.Namespace, environment: Environment) -> WingOptions:
    n_steps = int(args.n_steps)
    velocity = np.asarray(environment.velocity, dtype=float)
    speed = max(float(np.linalg.norm(velocity)), 1.0e-14)
    output_options = OutputOptions(
        compute_loads=True,
        loads_start=max(1, n_steps - 20),
        loads_end=n_steps,
        loads_step=1,
        loads_avg=True,
        loads_reference="0",
    )
    if args.dust_method == "vlm":
        return WingOptions(
            discretization_method=WingMethod.VORTEX_LATTICE,
            panel_type=WingPanelType.UNIFORM,
            num_panels=max(1, int(args.mesh_chord_stations) - 1),
            inner_product_te=0.5,
            tol_se_wing=1.0e-3,
            proj_te=True,
            proj_te_dir="parallel",
            proj_te_vector=velocity / speed,
            output_options=output_options,
        )

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
        output_options=output_options,
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
    polar_mach = float(args.polar_mach) if float(args.polar_mach) > 0.0 else None
    return ResolvedGeometryDustDiscipline(
        name="CTADustFromResolvedGeometry",
        geometry_provider=_geometry_provider,
        metric_inputs=cta.GEOMETRY_METRIC_VARIABLES,
        output_dir=args.output_dir / "cases",
        environment=environment,
        options=options,
        wing_options=wing_options,
        mesh_settings=mesh_settings,
        fail_fast=args.fail_fast,
        reuse_run_directory=not args.store_case_directories,
        run_directory_name=args.run_directory_name,
        neuralfoil_mach_polar=polar_mach,
        neuralfoil_cd_min_friction=float(args.polar_cd_min),
        neuralfoil_model=args.polar_model,
        output_prefix="bwb_dust",
    )


def build_cta_dust_doe_scenario(
    args: argparse.Namespace,
) -> tuple[MADSScenario, ResolvedGeometryDustDiscipline]:
    """Create the GEMSEO DOE scenario for CTA geometry plus DUST."""

    dust_discipline = _build_dust_discipline(args)
    mads_scenario = MADSScenario()
    mads_scenario.fill_parameter_space(cta.CTA_DOE_DESIGN_VARIABLES)
    disciplines = [
        _make_fault_tolerant(cta.disc_planform_mapping, "planform_mapping"),
        _make_fault_tolerant(cta.disc_geometry, "geometry"),
    ]
    if cta.disc_internal_boxes is not None:
        disciplines.append(_make_fault_tolerant(cta.disc_internal_boxes, "internal_boxes"))
    disciplines.extend(
        [
            _make_fault_tolerant(cta.disc_geometry_validation, "geometry_validation"),
            dust_discipline,  # already fault-tolerant via fail_fast=False
        ],
    )

    mads_scenario.create_scenario(
        disciplines=disciplines,
        formulation="DisciplinaryOpt",
        objective_name="bwb_dust_failure_code",
        scenario_type="DOE",
        name="cta_dust_doe",
        maximize_objective=False,
    )

    polar_mach = float(args.polar_mach) if float(args.polar_mach) > 0.0 else None
    nf_observables = (
        ["bwb_dust_neuralfoil_full_profile_cd", "bwb_dust_cd_induced_full_aircraft", "bwb_dust_cd_total_full_aircraft", "bwb_dust_ld_full_aircraft"]
        if polar_mach is not None
        else []
    )
    observables = [
        "bwb_dust_success",
        "bwb_dust_cl",
        "bwb_dust_cd",
        "bwb_dust_cm",
        "bwb_dust_cy",
        "bwb_dust_ld",
        "bwb_dust_cl_wind",
        "bwb_dust_cd_wind",
        "bwb_dust_ld_wind",
        *nf_observables,
        "bwb_dust_lift_n",
        "bwb_dust_drag_n",
        "bwb_dust_side_n",
        "bwb_dust_fx_reference_n",
        "bwb_dust_fy_reference_n",
        "bwb_dust_fz_reference_n",
        "bwb_dust_mx_reference_nm",
        "bwb_dust_my_reference_nm",
        "bwb_dust_mz_reference_nm",
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
        scenario.scenario.execute(
            algo_name="CustomDOE",
            samples=samples,
        )
        return int(samples.shape[0])

    if args.baseline_only:
        scenario.scenario.execute(
            algo_name="CustomDOE",
            samples=cta.baseline_sample(),
        )
        return 1

    scenario.scenario.execute(
        algo_name=args.algo,
        n_samples=args.n_samples,
    )
    return int(args.n_samples)


def _design_space_rows() -> list[dict[str, float | str]]:
    return cta.design_space_rows()


def _analysis_settings_rows(args: argparse.Namespace) -> list[list[object]]:
    return [
        ["key", "value"],
        ["dust_method", args.dust_method],
        ["n_steps", int(args.n_steps)],
        ["executed_samples", int(getattr(args, "executed_n_samples", 1 if args.baseline_only else args.n_samples))],
        ["samples_csv", "" if args.samples_csv is None else str(Path(args.samples_csv).resolve())],
        ["sample_start", int(args.sample_start)],
        ["sample_count", "" if args.sample_count is None else int(args.sample_count)],
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
            "case_geometry_definition": "examples/cta_geometry.py",
            "geometry_core": "multiads.solvers.synthesis.geometry_lib",
            "dust_adapter": "multiads.solvers.aerodynamics.dust_lib.ResolvedGeometryDustDiscipline",
            "dust_geometry_bridge": "multiads.solvers.aerodynamics.dust_lib resolved-geometry runners",
        },
        "algorithm": "CustomDOE" if args.baseline_only or args.samples_csv else args.algo,
        "n_samples": int(
            getattr(args, "executed_n_samples", 1 if args.baseline_only else args.n_samples),
        ),
        "sample_source": {
            "samples_csv": None if args.samples_csv is None else str(Path(args.samples_csv).resolve()),
            "sample_start": int(args.sample_start),
            "sample_count": None if args.sample_count is None else int(args.sample_count),
        },
        "alpha_deg": float(args.alpha_deg),
        "mach": float(args.mach),
        "altitude_ft": float(args.altitude_ft),
        "disa_k": float(args.disa_k),
        "mesh": {
            "dust_method": args.dust_method,
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
            variable.name for variable in cta.CTA_DOE_DESIGN_VARIABLES
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
                "DUST receives a mesh derived from the resolved geometry, not only "
                "section definitions. For VLM campaigns this mesh is converted into "
                "a parametric DUST wing inside dust_lib.py."
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
        "--samples-csv",
        type=Path,
        default=None,
        help=(
            "Run a fixed DOE sample table instead of generating samples inside "
            "GEMSEO. With --generate-samples-only, this is the output CSV."
        ),
    )
    parser.add_argument(
        "--sample-start",
        type=int,
        default=0,
        help="First row to run from --samples-csv. Used for HPC shards.",
    )
    parser.add_argument(
        "--sample-count",
        type=int,
        default=None,
        help="Number of rows to run from --samples-csv. Used for HPC shards.",
    )
    parser.add_argument(
        "--baseline-only",
        action="store_true",
        help="Run one CustomDOE sample at the CTA baseline instead of an LHS campaign.",
    )
    parser.add_argument(
        "--generate-samples-only",
        action="store_true",
        help="Only generate a fixed 14-variable DOE sample CSV and exit.",
    )
    parser.add_argument(
        "--sample-method",
        choices=SUPPORTED_SAMPLE_METHODS,
        default="sobol",
        help="Sampling method used by --generate-samples-only.",
    )
    parser.add_argument(
        "--sample-seed",
        type=int,
        default=17,
        help="Random seed used by --generate-samples-only.",
    )
    parser.add_argument(
        "--dust-method",
        choices=SUPPORTED_DUST_METHODS,
        default="vlm",
        help="DUST aerodynamic discretization used by the DOE campaign.",
    )
    parser.add_argument("--alpha-deg", type=float, default=3.0, help="DUST angle of attack in degrees.")
    parser.add_argument("--mach", type=float, default=0.8, help="Freestream Mach number.")
    parser.add_argument("--altitude-ft", type=float, default=40000.0, help="Altitude in feet.")
    parser.add_argument("--disa-k", type=float, default=0.0, help="DISA temperature offset, recorded in outputs.")
    parser.add_argument("--mesh-span-stations", type=int, default=21, help="Half-span DUST mesh stations before mirroring.")
    parser.add_argument("--mesh-chord-stations", type=int, default=21, help="DUST chordwise stations per skin.")
    parser.add_argument("--span-spacing", choices=["uniform", "curvature"], default="curvature")
    parser.add_argument("--chord-spacing", choices=["uniform", "curvature"], default="curvature")
    parser.add_argument("--span-curvature-weight", type=float, default=5.0)
    parser.add_argument("--chord-curvature-weight", type=float, default=4.0)
    parser.add_argument("--leading-edge-opening-m", type=float, default=0.0)
    parser.add_argument("--n-steps", type=int, default=150)
    parser.add_argument("--dt", type=float, default=0.005)
    parser.add_argument("--n-threads", type=int, default=1)
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
    parser.add_argument(
        "--polar-mach",
        type=float,
        default=0.4,
        help=(
            "Mach number used for NeuralFoil 2D polars with Prandtl-Glauert correction back "
            "to flight Mach. Set to 0 to disable NeuralFoil profile drag. Default: 0.4."
        ),
    )
    parser.add_argument(
        "--polar-cd-min",
        type=float,
        default=0.006,
        help="Minimum friction CD floor for Prandtl-Glauert PG correction. Default: 0.006.",
    )
    parser.add_argument(
        "--polar-model",
        default="large",
        help="NeuralFoil model size (small/medium/large/xlarge/xxxlarge). Default: large.",
    )
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
        print("CTA DUST DOE samples generated")
        print(f"  method = {args.sample_method}")
        print(f"  n_samples = {args.n_samples}")
        print(f"  seed = {args.sample_seed}")
        print(f"  samples_csv = {samples_csv}")
        return

    if args.output_dir.exists() and not args.keep_existing:
        shutil.rmtree(args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    scenario, _dust_discipline = build_cta_dust_doe_scenario(args)
    args.executed_n_samples = _execute_scenario(scenario, args)

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
    print(f"  algo = {'CustomDOE' if args.baseline_only or args.samples_csv else args.algo}")
    print(f"  dust_method = {args.dust_method}")
    print(f"  n_steps = {args.n_steps}")
    print(f"  executed_samples = {args.executed_n_samples}")
    if args.samples_csv is not None:
        print(f"  samples_csv = {Path(args.samples_csv).resolve()}")
        print(f"  sample_slice = [{args.sample_start}, {args.sample_start + args.executed_n_samples})")
    print(f"  alpha_deg = {args.alpha_deg:g}")
    print(f"  output_dir = {args.output_dir}")
    print(f"  flat_dataset = {export_paths.flat_dataset_csv}")
    print(f"  workbook = {export_paths.workbook_xlsx}")
    print(f"  design_space = {export_paths.design_space_csv}")
    print(f"  manifest = {export_paths.manifest_json}")


if __name__ == "__main__":
    main()
