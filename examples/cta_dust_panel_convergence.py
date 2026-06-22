from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path
from typing import Any

import numpy as np

import cta_convergence_common as cta_common
from multiads.assembly import Environment
from multiads.solvers.aerodynamics.dust_lib import (
    DustMeshSettings,
    Options,
    OutputOptions,
    WingMethod,
    WingOptions,
    WingPanelType,
    dust_executable,
    run_dust_case_from_resolved_npz,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = REPO_ROOT / "outputs" / "CTA_case" / "convergence" / "panel"
COMPONENT_NAME = "cta_wing"


def _dust_options(
    args: argparse.Namespace,
    *,
    n_steps: int,
    run_dir: Path,
    write_vtk: bool,
) -> Options:
    return Options(
        name=f"cta_panel_steps_{n_steps:04d}",
        dust_pre=dust_executable("dust_pre", args.dust_bin_dir),
        dust=dust_executable("dust", args.dust_bin_dir),
        dust_post=dust_executable("dust_post", args.dust_bin_dir),
        run_directory=run_dir,
        output_dir=Path("Output"),
        post_dir=Path("post"),
        keep_run_directory=True,
        t_start=0.0,
        t_end=float(n_steps) * float(args.dt),
        dt=float(args.dt),
        dt_out=float(args.dt),
        output_start=True,
        n_threads=int(args.n_threads),
        n_wake_panels=int(n_steps),
        n_wake_particles=int(args.n_wake_particles),
        particles_box_min=np.asarray(tuple(args.particles_box_min), dtype=float),
        particles_box_max=np.asarray(tuple(args.particles_box_max), dtype=float),
        penetration_avoidance=False,
        output_options=OutputOptions(
            visualization=write_vtk,
            viz_start=int(n_steps),
            viz_end=int(n_steps),
            viz_step=1,
            viz_fmt="vtk",
            viz_wake=True,
            viz_separate_wake=True,
            viz_variables=["cp", "vorticity_vector", "velocity"],
        ),
    )


def _wing_options(args: argparse.Namespace, env: Environment, *, n_steps: int) -> WingOptions:
    velocity = np.asarray(env.velocity, dtype=float)
    speed = max(float(np.linalg.norm(velocity)), 1.0e-14)
    if args.save_force_history:
        loads_start = 1
        loads_avg = False
    else:
        loads_start = max(1, int(n_steps) - int(args.loads_average_window))
        loads_avg = True
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
            loads_start=loads_start,
            loads_end=int(n_steps),
            loads_step=1,
            loads_avg=loads_avg,
            loads_reference="0",
        ),
    )


def _write_manifest(args: argparse.Namespace, path: Path, cases: list[dict[str, int]]) -> None:
    path.write_text(
        json.dumps(
            {
                "purpose": "CTA DUST panel-method convergence study on the fixed baseline geometry.",
                "case_policy": "DUST run directory is overwritten; metrics are accumulated in CSV/XLSX outputs.",
                "method": "DUST panels",
                "cases": cases,
                "alpha_deg": float(args.alpha_deg),
                "mach": float(args.mach),
                "altitude_ft": float(args.altitude_ft),
                "disa_k": float(args.disa_k),
                "dt": float(args.dt),
                "loads_average_window": int(args.loads_average_window),
                "force_history_saved": bool(args.save_force_history),
                "outputs": {
                    "results_csv": "cta_dust_panel_convergence_results.csv",
                    "force_history_csv": "cta_dust_panel_convergence_force_history.csv",
                    "workbook": "cta_dust_panel_convergence_results.xlsx",
                },
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run CTA baseline DUST panel convergence cases.")
    parser.add_argument("--steps", default="50,60,70,80,100,120")
    parser.add_argument("--meshes", default="49x45")
    parser.add_argument("--alpha-deg", type=float, default=3.0)
    parser.add_argument("--mach", type=float, default=0.8)
    parser.add_argument("--altitude-ft", type=float, default=40000.0)
    parser.add_argument("--disa-k", type=float, default=0.0)
    parser.add_argument("--dt", type=float, default=0.005)
    parser.add_argument("--n-threads", type=int, default=6)
    parser.add_argument("--n-wake-particles", type=int, default=100000)
    parser.add_argument("--span-spacing", choices=["uniform", "curvature"], default="curvature")
    parser.add_argument("--chord-spacing", choices=["uniform", "curvature"], default="curvature")
    parser.add_argument("--span-curvature-weight", type=float, default=5.0)
    parser.add_argument("--chord-curvature-weight", type=float, default=4.0)
    parser.add_argument("--chord-endpoint-weight", type=float, default=0.60)
    parser.add_argument("--loads-average-window", type=int, default=20)
    parser.add_argument(
        "--no-force-history",
        action="store_false",
        dest="save_force_history",
        help="Do not write the per-time-step force/coefficient history CSV.",
    )
    parser.add_argument("--write-vtk", action="store_true")
    parser.add_argument("--dust-bin-dir", default=None)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--run-directory-name", default="run")
    parser.add_argument("--keep-existing", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--max-cases", type=int, default=None)
    parser.add_argument("--particles-box-min", nargs=3, type=float, default=(-80.0, -90.0, -80.0))
    parser.add_argument("--particles-box-max", nargs=3, type=float, default=(380.0, 90.0, 80.0))
    parser.set_defaults(save_force_history=True)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.dust_bin_dir is not None:
        args.dust_bin_dir = str(Path(args.dust_bin_dir).expanduser().resolve())
    output_dir = args.output_dir.resolve()
    if output_dir.exists() and not args.keep_existing:
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    steps = cta_common.parse_int_list(args.steps)
    meshes = cta_common.parse_mesh_list(args.meshes)
    cases = [
        {"n_steps": n_steps, "mesh_span_stations": span, "mesh_chord_stations": chord}
        for span, chord in meshes
        for n_steps in steps
    ]
    if args.max_cases is not None:
        cases = cases[: int(args.max_cases)]

    mesh_path, geometry_metrics, geometry_state = cta_common.baseline_geometry(output_dir)
    design_row = cta_common.design_variable_row()
    box_metrics = cta_common.evaluate_boxes(geometry_state, cta_common.DEFAULT_CONSTRAINTS)
    env = cta_common.build_environment(args)
    s_ref_m2 = geometry_metrics[f"{COMPONENT_NAME}.planform_area_m2"]
    c_ref_m = geometry_metrics[f"{COMPONENT_NAME}.mean_aerodynamic_chord_m"]

    _write_manifest(args, output_dir / "cta_dust_panel_convergence_manifest.json", cases)

    results_csv = output_dir / "cta_dust_panel_convergence_results.csv"
    history_csv = output_dir / "cta_dust_panel_convergence_force_history.csv"
    result_rows: list[dict[str, Any]] = []
    history_fieldnames: list[str] | None = None

    if args.dry_run:
        print("Dry run cases:")
        for case in cases:
            print(case)
        return

    for index, case in enumerate(cases, start=1):
        run_dir = output_dir / "cases" / args.run_directory_name
        n_steps = int(case["n_steps"])
        span_count = int(case["mesh_span_stations"])
        chord_count = int(case["mesh_chord_stations"])
        case_tag = f"case_{index:04d}_steps_{n_steps:04d}_mesh_{span_count}x{chord_count}"
        print(f"[{index}/{len(cases)}] {case_tag}")

        options = _dust_options(args, n_steps=n_steps, run_dir=run_dir, write_vtk=bool(args.write_vtk))
        wing_options = _wing_options(args, env, n_steps=n_steps)
        mesh_settings = DustMeshSettings(
            n_span_stations=span_count,
            n_chord_stations=chord_count,
            span_spacing=args.span_spacing,
            chord_spacing=args.chord_spacing,
            span_curvature_weight=float(args.span_curvature_weight),
            chord_curvature_weight=float(args.chord_curvature_weight),
            chord_endpoint_weight=float(args.chord_endpoint_weight),
            leading_edge_opening_m=0.0,
            collapse_trailing_edge=True,
            mirror_span=True,
        )

        start = time.perf_counter()
        try:
            result = run_dust_case_from_resolved_npz(
                mesh_path,
                environment=env,
                options=options,
                s_ref_m2=s_ref_m2,
                c_ref_m=c_ref_m,
                mesh_settings=mesh_settings,
                wing_options=wing_options,
                clean_run_dir=True,
                component_name=COMPONENT_NAME,
            )
            elapsed_s = time.perf_counter() - start
            history = (
                cta_common.read_force_history(
                    run_dir,
                    options.name,
                    q_pa=result.q_pa,
                    s_ref_m2=s_ref_m2,
                    c_ref_m=c_ref_m,
                    environment=env,
                )
                if args.save_force_history
                else []
            )
            row: dict[str, Any] = {
                "case_index": index,
                "case_tag": case_tag,
                "success": 1.0,
                "failure_message": "",
                "elapsed_s": elapsed_s,
                "solver_method": "panels",
                "n_steps": n_steps,
                "dt_s": float(args.dt),
                "mesh_span_stations": span_count,
                "mesh_chord_stations": chord_count,
                "span_spacing": args.span_spacing,
                "chord_spacing": args.chord_spacing,
                "loads_average_window": int(args.loads_average_window),
                **design_row,
                **geometry_metrics,
                **box_metrics,
                **result.to_flat_dict(),
                **cta_common.final_window_stats(history, int(args.loads_average_window)),
            }
            for step_index, history_row in enumerate(history, start=1):
                flat_history = {
                    "case_index": index,
                    "case_tag": case_tag,
                    "step_index": step_index,
                    "n_steps": n_steps,
                    "mesh_span_stations": span_count,
                    "mesh_chord_stations": chord_count,
                    **history_row,
                }
                if history_fieldnames is None:
                    history_fieldnames = list(flat_history)
                cta_common.append_csv(history_csv, flat_history, history_fieldnames)
        except Exception as exc:  # noqa: BLE001
            elapsed_s = time.perf_counter() - start
            row = {
                "case_index": index,
                "case_tag": case_tag,
                "success": 0.0,
                "failure_message": str(exc),
                "elapsed_s": elapsed_s,
                "solver_method": "panels",
                "n_steps": n_steps,
                "dt_s": float(args.dt),
                "mesh_span_stations": span_count,
                "mesh_chord_stations": chord_count,
                "span_spacing": args.span_spacing,
                "chord_spacing": args.chord_spacing,
                "loads_average_window": int(args.loads_average_window),
                **design_row,
                **geometry_metrics,
                **box_metrics,
            }

        result_rows.append(row)
        cta_common.write_rows_csv(results_csv, result_rows)

    cta_common.write_workbook(output_dir, "cta_dust_panel_convergence_results.xlsx", result_rows, design_row)
    print("CTA panel convergence completed")
    print(f"  results = {results_csv}")
    if args.save_force_history:
        print(f"  force_history = {history_csv}")
    print(f"  workbook = {output_dir / 'cta_dust_panel_convergence_results.xlsx'}")


if __name__ == "__main__":
    main()
