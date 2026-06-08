from __future__ import annotations

import argparse
import json
import math
import shutil
import time
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import cta_dust_common as cta_common
from multiads.assembly import Environment
from multiads.solvers.aerodynamics.dust_lib import (
    DustMeshSettings,
    Options,
    OutputOptions,
    WingMethod,
    WingOptions,
    dust_executable,
    run_dust_lifting_line_case_from_prepared_geometry,
)
from multiads.solvers.aerodynamics.neuralfoil import Neuralfoil
from multiads.scenario.polars import POLAR_DEFAULT_AOA, PolarVariable
from multiads.utilities.campaign_export import write_xlsx_workbook


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = (
    REPO_ROOT / "outputs" / "CTA_case" / "convergence" / "lifting_line"
)
COMPONENT_NAME = "cta_wing"
SOLVER_LABEL = "lifting_line"


def _dust_options(
    args: argparse.Namespace,
    *,
    n_steps: int,
    run_dir: Path,
    write_vtk: bool,
) -> Options:
    return Options(
        name=f"cta_lifting_line_steps_{n_steps:04d}",
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
            viz_variables=["vorticity_vector", "velocity"],
        ),
    )


def _wing_options(
    args: argparse.Namespace,
    *,
    n_steps: int,
) -> WingOptions:
    if args.save_force_history:
        loads_start = 1
        loads_avg = False
    else:
        loads_start = max(1, int(n_steps) - int(args.loads_average_window))
        loads_avg = True
    return WingOptions(
        discretization_method=WingMethod.LIFTING_LINE,
        output_options=OutputOptions(
            compute_loads=True,
            loads_start=loads_start,
            loads_end=int(n_steps),
            loads_step=1,
            loads_avg=loads_avg,
            loads_reference="0",
        ),
    )


def _profile_metrics(
    args: argparse.Namespace,
    *,
    geometry_state: Any,
    env: Environment,
    s_ref_m2: float,
) -> dict[str, float]:
    if args.profile_drag == "neuralfoil":
        metrics: dict[str, float] = {}
        metrics.update(
            Neuralfoil.estimate_profile_drag_from_resolved_stations(
                geometry_state.resolved_stations,
                env,
                s_ref_m2=s_ref_m2,
                alpha_mode=args.profile_alpha_mode,
                model=args.neuralfoil_model,
                n_crit=float(args.neuralfoil_n_crit),
                station_stride=int(args.profile_station_stride),
                y_min_m=float(args.outer_start_y_m),
                metric_prefix="neuralfoil_outer",
            ),
        )
        metrics.update(
            Neuralfoil.estimate_profile_drag_from_resolved_stations(
                geometry_state.resolved_stations,
                env,
                s_ref_m2=s_ref_m2,
                alpha_mode=args.profile_alpha_mode,
                model=args.neuralfoil_model,
                n_crit=float(args.neuralfoil_n_crit),
                station_stride=int(args.profile_station_stride),
                y_min_m=float(args.transition_start_y_m),
                metric_prefix="neuralfoil_transition_outer",
            ),
        )
        return metrics

    metrics = {}
    metrics.update(
        Neuralfoil.zero_profile_drag_metrics(
            "neuralfoil_outer",
            float(args.outer_start_y_m),
        ),
    )
    metrics.update(
        Neuralfoil.zero_profile_drag_metrics(
            "neuralfoil_transition_outer",
            float(args.transition_start_y_m),
        ),
    )
    return metrics


def _populate_lifting_line_polars(
    env: Environment,
    wing: Any,
    polars: dict[str, PolarVariable],
    *,
    model: str,
    n_crit: float,
) -> None:
    alphas = np.asarray(POLAR_DEFAULT_AOA, dtype=float)
    mach = float(env.mach)
    speed = float(env.speed)
    kin_viscosity = max(float(env.kin_viscosity), 1.0e-14)
    for section in wing.sections:
        polar = polars[f"{section.name}.polar"]
        airfoil = Neuralfoil.make_airfoil(section.airfoil)
        reynolds = speed * float(section.chord) / kin_viscosity
        out = Neuralfoil.compute_aero_from_airfoil(
            airfoil,
            alphas=alphas,
            reynolds=reynolds,
            mach=mach,
            n_crit=float(n_crit),
            model=model,
            include_360_deg_effects=True,
        )
        polar.mach = np.array([mach])
        polar.reynolds = np.array([reynolds])
        polar.aoa = alphas
        polar.cl = np.asarray(out["CL"], dtype=float)
        polar.cd = np.asarray(out["CD"], dtype=float)
        polar.cm = np.asarray(out["CM"], dtype=float)


def _write_workbook(
    output_dir: Path,
    result_rows: list[dict[str, Any]],
    summary_rows: list[dict[str, Any]],
    design_row: dict[str, float],
) -> None:
    if not result_rows:
        return
    result_keys = list(result_rows[0])
    summary_keys = list(summary_rows[0]) if summary_rows else []
    workbook = output_dir / "cta_dust_lifting_line_convergence_results.xlsx"
    write_xlsx_workbook(
        workbook,
        {
            "results": [
                result_keys,
                *[[row.get(key) for key in result_keys] for row in result_rows],
            ],
            "summary": [
                summary_keys,
                *[[row.get(key) for key in summary_keys] for row in summary_rows],
            ],
            "baseline_design": [
                ["name", "value"],
                *[[key, value] for key, value in design_row.items()],
            ],
        },
    )


def _float(row: dict[str, Any], key: str, default: float = math.nan) -> float:
    try:
        value = row.get(key, default)
        if value in ("", None):
            return default
        return float(value)
    except Exception:
        return default


def _int(row: dict[str, Any], key: str) -> int:
    return int(round(float(row[key])))


def _mmss(seconds: float) -> str:
    total = int(round(seconds))
    return f"{total // 60}:{total % 60:02d}"


def _summary_rows(result_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    valid = [row for row in result_rows if _float(row, "success", 0.0) == 1.0]
    if not valid:
        return []
    ref = valid[-1]
    ref_step = _int(ref, "n_steps")
    metric_map = {
        "cl": "cl",
        "cd": "cd",
        "cm": "cm",
        "ld": "ld",
        "cl_mean": "history_final_window_cl_mean",
        "cd_mean": "history_final_window_cd_mean",
        "ld_mean": "history_final_window_ld_mean",
        "ld_total_outer_profile": (
            f"ld_total_{SOLVER_LABEL}_plus_outer_profile"
        ),
        "ld_total_transition_outer_profile": (
            f"ld_total_{SOLVER_LABEL}_plus_transition_outer_profile"
        ),
    }
    rows: list[dict[str, Any]] = []
    for row in result_rows:
        item: dict[str, Any] = {
            "n_steps": _int(row, "n_steps"),
            "success": int(_float(row, "success", 0.0)),
            "elapsed_s": _float(row, "elapsed_s"),
            "elapsed_mmss": _mmss(_float(row, "elapsed_s")),
            "cl": _float(row, "cl"),
            "cd": _float(row, "cd"),
            "cm": _float(row, "cm"),
            "ld": _float(row, "ld"),
            "cl_mean": _float(row, "history_final_window_cl_mean"),
            "cd_mean": _float(row, "history_final_window_cd_mean"),
            "ld_mean": _float(row, "history_final_window_ld_mean"),
            "cd_total_outer_profile": _float(
                row,
                f"cd_total_{SOLVER_LABEL}_plus_outer_profile",
            ),
            "ld_total_outer_profile": _float(
                row,
                f"ld_total_{SOLVER_LABEL}_plus_outer_profile",
            ),
            "cd_total_transition_outer_profile": _float(
                row,
                f"cd_total_{SOLVER_LABEL}_plus_transition_outer_profile",
            ),
            "ld_total_transition_outer_profile": _float(
                row,
                f"ld_total_{SOLVER_LABEL}_plus_transition_outer_profile",
            ),
        }
        for label, source_key in metric_map.items():
            value = _float(row, source_key)
            ref_value = _float(ref, source_key)
            if math.isfinite(value) and math.isfinite(ref_value) and abs(ref_value) > 1.0e-12:
                item[f"{label}_rel_error_vs_{ref_step}_%"] = (
                    100.0 * abs(value - ref_value) / abs(ref_value)
                )
        rows.append(item)
    return rows


def _plot_results(
    output_dir: Path,
    result_rows: list[dict[str, Any]],
    history_rows: list[dict[str, Any]],
) -> None:
    valid = [row for row in result_rows if _float(row, "success", 0.0) == 1.0]
    if not valid:
        return

    steps = [_int(row, "n_steps") for row in valid]
    elapsed = [_float(row, "elapsed_s") for row in valid]
    cl = [_float(row, "cl") for row in valid]
    cd = [_float(row, "cd") for row in valid]
    cm = [_float(row, "cm") for row in valid]
    ld = [_float(row, "ld") for row in valid]
    cl_mean = [_float(row, "history_final_window_cl_mean") for row in valid]
    cl_min = [_float(row, "history_final_window_cl_min") for row in valid]
    cl_max = [_float(row, "history_final_window_cl_max") for row in valid]
    cd_outer = [
        _float(row, f"cd_total_{SOLVER_LABEL}_plus_outer_profile")
        for row in valid
    ]
    cd_transition = [
        _float(row, f"cd_total_{SOLVER_LABEL}_plus_transition_outer_profile")
        for row in valid
    ]
    ld_outer = [
        _float(row, f"ld_total_{SOLVER_LABEL}_plus_outer_profile")
        for row in valid
    ]
    ld_transition = [
        _float(row, f"ld_total_{SOLVER_LABEL}_plus_transition_outer_profile")
        for row in valid
    ]
    err_low = [max(0.0, mean - lo) for mean, lo in zip(cl_mean, cl_min, strict=True)]
    err_high = [max(0.0, hi - mean) for mean, hi in zip(cl_mean, cl_max, strict=True)]
    colors = plt.cm.viridis([i / max(1, len(steps) - 1) for i in range(len(steps))])

    fig, ax1 = plt.subplots(figsize=(11.3, 5.4))
    ax1.bar(steps, elapsed, width=8.0, color=colors, alpha=0.65, label="Runtime")
    ax1.set_xlabel("Number of time steps")
    ax1.set_ylabel("Runtime [s]", color="#33691e")
    ax1.tick_params(axis="y", labelcolor="#33691e")
    ax1.grid(True, axis="y", alpha=0.25)
    for x, y in zip(steps, elapsed, strict=True):
        ax1.annotate(
            _mmss(y),
            (x, y),
            textcoords="offset points",
            xytext=(0, 5),
            ha="center",
            fontsize=8,
            color="#263238",
        )
    ax2 = ax1.twinx()
    ax2.errorbar(
        steps,
        cl_mean,
        yerr=[err_low, err_high],
        fmt="o-",
        color="#0d47a1",
        ecolor="#ef6c00",
        elinewidth=1.8,
        capsize=5,
        capthick=1.6,
        linewidth=2.0,
        label="CL mean min/max",
    )
    ax2.set_ylabel("CL mean and min/max final-window range [-]", color="#0d47a1")
    ax2.tick_params(axis="y", labelcolor="#0d47a1")
    ax1.set_title("CTA DUST lifting-line convergence: runtime and CL min/max")
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left")
    fig.tight_layout()
    fig.savefig(output_dir / "cta_dust_lifting_line_runtime_vs_cl_minmax.png", dpi=220)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10.8, 5.4))
    ax.fill_between(steps, cl_min, cl_max, color="#ffcc80", alpha=0.26)
    ax.errorbar(
        steps,
        cl_mean,
        yerr=[err_low, err_high],
        fmt="o-",
        color="#0d47a1",
        ecolor="#ef6c00",
        elinewidth=1.8,
        capsize=5,
        capthick=1.6,
        linewidth=2.0,
        label="CL mean with min/max range",
    )
    ax.set_xlabel("Number of time steps")
    ax.set_ylabel("CL [-]")
    ax.set_title("CTA DUST lifting-line convergence: CL mean with min/max bars")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(output_dir / "cta_dust_lifting_line_cl_mean_minmax.png", dpi=220)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10.8, 5.3))
    ax.plot(steps, cl, marker="o", label="CL")
    ax.plot(steps, cd, marker="o", label="CD lifting line")
    ax.plot(steps, cd_outer, marker="o", label="CD + outer profile")
    ax.plot(steps, cd_transition, marker="o", label="CD + transition/outer profile")
    ax.plot(steps, cm, marker="o", label="CM")
    ax.set_xlabel("Number of time steps")
    ax.set_ylabel("Coefficient [-]")
    ax.set_title("CTA DUST lifting-line coefficient convergence")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(output_dir / "cta_dust_lifting_line_convergence_coefficients.png", dpi=220)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10.8, 5.3))
    ax.plot(steps, ld, marker="o", label="L/D lifting line only")
    ax.plot(steps, ld_outer, marker="o", label="L/D + outer profile drag")
    ax.plot(
        steps,
        ld_transition,
        marker="o",
        label="L/D + transition/outer profile drag",
    )
    ax.set_xlabel("Number of time steps")
    ax.set_ylabel("L/D [-]")
    ax.set_title("CTA DUST lifting-line efficiency convergence")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(output_dir / "cta_dust_lifting_line_convergence_efficiency.png", dpi=220)
    plt.close(fig)

    if history_rows:
        fig, (ax_cl, ax_cd) = plt.subplots(2, 1, figsize=(11.5, 7.2), sharex=False)
        for step_count in steps:
            case_hist = [
                row for row in history_rows if int(float(row.get("n_steps", 0))) == step_count
            ]
            if not case_hist:
                continue
            xs = [int(float(row["step_index"])) for row in case_hist]
            ax_cl.plot(
                xs,
                [_float(row, "cl") for row in case_hist],
                linewidth=1.25,
                label=f"{step_count} steps",
            )
            ax_cd.plot(
                xs,
                [_float(row, "cd") for row in case_hist],
                linewidth=1.25,
                label=f"{step_count} steps",
            )
        ax_cl.set_ylabel("CL [-]")
        ax_cl.set_title("CTA DUST lifting-line force-history convergence")
        ax_cl.grid(True, alpha=0.25)
        ax_cd.set_xlabel("Time step")
        ax_cd.set_ylabel("CD [-]")
        ax_cd.grid(True, alpha=0.25)
        ax_cl.legend(ncol=3, fontsize=8)
        fig.tight_layout()
        fig.savefig(output_dir / "cta_dust_lifting_line_force_history.png", dpi=220)
        plt.close(fig)


def _write_summary(
    output_dir: Path,
    result_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    rows = _summary_rows(result_rows)
    if not rows:
        return rows
    cta_common.write_rows_csv(output_dir / "cta_dust_lifting_line_convergence_summary.csv", rows)
    ref_step = int(rows[-1]["n_steps"])
    (output_dir / "cta_dust_lifting_line_convergence_summary.json").write_text(
        json.dumps(
            {
                "reference_n_steps": ref_step,
                "solver_method": SOLVER_LABEL,
                "rows": rows,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return rows


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run CTA baseline DUST lifting-line convergence cases.",
    )
    parser.add_argument("--steps", default="20,30,40,50,60,80,100,120,160,200,220,250")
    parser.add_argument("--meshes", default="21x1")
    parser.add_argument("--alpha-deg", type=float, default=3.0)
    parser.add_argument("--mach", type=float, default=0.8)
    parser.add_argument("--altitude-ft", type=float, default=40000.0)
    parser.add_argument("--disa-k", type=float, default=0.0)
    parser.add_argument("--dt", type=float, default=0.005)
    parser.add_argument("--n-threads", type=int, default=6)
    parser.add_argument("--n-wake-particles", type=int, default=100000)
    parser.add_argument("--span-spacing", choices=["uniform", "curvature"], default="curvature")
    parser.add_argument("--chord-spacing", choices=["uniform", "curvature"], default="uniform")
    parser.add_argument("--span-curvature-weight", type=float, default=5.0)
    parser.add_argument("--chord-curvature-weight", type=float, default=3.0)
    parser.add_argument("--chord-endpoint-weight", type=float, default=0.45)
    parser.add_argument(
        "--lifting-line-start-y-m",
        type=float,
        default=0.0,
        help=(
            "Minimum half-span station used by the DUST lifting-line adapter. "
            "Use a positive value to exclude the BWB centerbody from LL."
        ),
    )
    parser.add_argument("--loads-average-window", type=int, default=20)
    parser.add_argument("--profile-drag", choices=["none", "neuralfoil"], default="neuralfoil")
    parser.add_argument("--neuralfoil-model", default="large")
    parser.add_argument("--neuralfoil-n-crit", type=float, default=9.0)
    parser.add_argument(
        "--profile-alpha-mode",
        choices=["global", "global-plus-twist", "global-minus-twist"],
        default="global-plus-twist",
    )
    parser.add_argument("--profile-station-stride", type=int, default=1)
    parser.add_argument("--outer-start-y-m", type=float, default=22.5)
    parser.add_argument("--transition-start-y-m", type=float, default=12.5)
    parser.add_argument(
        "--no-force-history",
        action="store_false",
        dest="save_force_history",
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

    _mesh_path, geometry_metrics, geometry_state = cta_common.baseline_geometry(output_dir)
    design_row = cta_common.design_variable_row()
    box_metrics = cta_common.evaluate_boxes(geometry_state, cta_common.DEFAULT_CONSTRAINTS)
    env = cta_common.build_environment(args)
    s_ref_m2 = geometry_metrics[f"{COMPONENT_NAME}.planform_area_m2"]
    c_ref_m = geometry_metrics[f"{COMPONENT_NAME}.mean_aerodynamic_chord_m"]
    profile_metrics = _profile_metrics(
        args,
        geometry_state=geometry_state,
        env=env,
        s_ref_m2=s_ref_m2,
    )

    results_csv = output_dir / "cta_dust_lifting_line_convergence_results.csv"
    history_csv = output_dir / "cta_dust_lifting_line_convergence_force_history.csv"
    result_rows: list[dict[str, Any]] = []
    history_rows: list[dict[str, Any]] = []
    history_fieldnames: list[str] | None = None

    if args.dry_run:
        print("Dry run cases:")
        for case in cases:
            print(case)
        print("profile_drag:", args.profile_drag)
        print("neuralfoil_outer_profile_cd:", profile_metrics["neuralfoil_outer_profile_cd"])
        print(
            "neuralfoil_transition_outer_profile_cd:",
            profile_metrics["neuralfoil_transition_outer_profile_cd"],
        )
        return

    for index, case in enumerate(cases, start=1):
        run_dir = output_dir / "cases" / args.run_directory_name
        n_steps = int(case["n_steps"])
        span_count = int(case["mesh_span_stations"])
        chord_count = int(case["mesh_chord_stations"])
        case_tag = f"case_{index:04d}_steps_{n_steps:04d}_mesh_{span_count}x{chord_count}"
        print(f"[{index}/{len(cases)}] {case_tag}")

        options = _dust_options(
            args,
            n_steps=n_steps,
            run_dir=run_dir,
            write_vtk=bool(args.write_vtk),
        )
        wing_options = _wing_options(args, n_steps=n_steps)
        mesh_settings = DustMeshSettings(
            n_span_stations=span_count,
            n_chord_stations=chord_count,
            span_spacing=args.span_spacing,
            chord_spacing=args.chord_spacing,
            span_curvature_weight=float(args.span_curvature_weight),
            chord_curvature_weight=float(args.chord_curvature_weight),
            chord_endpoint_weight=float(args.chord_endpoint_weight),
            mirror_span=True,
            span_min_y_m=(
                None
                if float(args.lifting_line_start_y_m) <= 0.0
                else float(args.lifting_line_start_y_m)
            ),
        )

        start = time.perf_counter()
        try:
            result = run_dust_lifting_line_case_from_prepared_geometry(
                geometry_state,
                environment=env,
                options=options,
                s_ref_m2=s_ref_m2,
                c_ref_m=c_ref_m,
                mesh_settings=mesh_settings,
                wing_options=wing_options,
                clean_run_dir=True,
                component_name=COMPONENT_NAME,
                polar_provider=lambda polar_env, polar_wing, polar_map: (
                    _populate_lifting_line_polars(
                        polar_env,
                        polar_wing,
                        polar_map,
                        model=args.neuralfoil_model,
                        n_crit=float(args.neuralfoil_n_crit),
                    )
                ),
            )
            elapsed_s = time.perf_counter() - start
            history = (
                cta_common.read_force_history(
                    run_dir,
                    options.name,
                    q_pa=result.q_pa,
                    s_ref_m2=s_ref_m2,
                    c_ref_m=c_ref_m,
                )
                if args.save_force_history
                else []
            )
            history = cta_common.augment_history_with_profile_drag(
                history,
                profiles=profile_metrics,
                solver_label=SOLVER_LABEL,
            )
            row: dict[str, Any] = {
                "case_index": index,
                "case_tag": case_tag,
                "success": 1.0,
                "failure_message": "",
                "elapsed_s": elapsed_s,
                "solver_method": SOLVER_LABEL,
                "profile_drag_model": args.profile_drag,
                "neuralfoil_model": args.neuralfoil_model,
                "profile_alpha_mode": args.profile_alpha_mode,
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
                **cta_common.final_profile_drag_window_stats(
                    history,
                    solver_label=SOLVER_LABEL,
                    window=int(args.loads_average_window),
                ),
            }
            row = cta_common.augment_row_with_profile_drag(
                row,
                profiles=profile_metrics,
                solver_label=SOLVER_LABEL,
                q_pa=result.q_pa,
                s_ref_m2=s_ref_m2,
            )
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
                history_rows.append(flat_history)
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
                "solver_method": SOLVER_LABEL,
                "profile_drag_model": args.profile_drag,
                "neuralfoil_model": args.neuralfoil_model,
                "profile_alpha_mode": args.profile_alpha_mode,
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
                **profile_metrics,
            }

        result_rows.append(row)
        cta_common.write_rows_csv(results_csv, result_rows)

    summary = _write_summary(output_dir, result_rows)
    _write_workbook(output_dir, result_rows, summary, design_row)
    _plot_results(output_dir, result_rows, history_rows)
    print("CTA lifting-line convergence completed")
    print(f"  results = {results_csv}")
    if args.save_force_history:
        print(f"  force_history = {history_csv}")
    print(f"  workbook = {output_dir / 'cta_dust_lifting_line_convergence_results.xlsx'}")


if __name__ == "__main__":
    main()
