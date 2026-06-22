from __future__ import annotations

import argparse
import json
import math
import shutil
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import cta_convergence_common as cta_common
from multiads.assembly import Environment
from multiads.solvers.aerodynamics.dust_lib import (
    DustMeshSettings,
    Options,
    OutputOptions,
    WingMethod,
    WingOptions,
    build_neuralfoil_polar_provider,
    compute_reconstructed_drag_coefficients,
    dust_executable,
    lifting_line_solver_preset,
    run_dust_lifting_line_case_from_prepared_geometry,
    validate_lifting_line_case,
)
from multiads.solvers.aerodynamics.neuralfoil import Neuralfoil


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
    ll_preset_name: str,
    overrides: dict[str, Any] | None = None,
) -> Options:
    preset = lifting_line_solver_preset(ll_preset_name)
    overrides = overrides or {}

    def value(name: str) -> Any:
        if name in overrides:
            return overrides[name]
        arg_value = getattr(args, name, None)
        return preset[name] if arg_value is None else arg_value

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
        max_iter=int(value("max_iter")),
        tol=float(value("tol")),
        ll_solver=str(value("ll_solver")),
        ll_reynolds_corrections=bool(value("ll_reynolds_corrections")),
        ll_reynolds_corrections_nfact=float(value("ll_reynolds_corrections_nfact")),
        ll_damp=float(value("ll_damp")),
        ll_stall_regularisation=bool(value("ll_stall_regularisation")),
        ll_stall_regularisation_nelems=int(value("ll_stall_regularisation_nelems")),
        ll_stall_regularizations_niters=int(value("ll_stall_regularizations_niters")),
        ll_stall_regularization_alpha_stall=float(value("ll_stall_regularization_alpha_stall")),
        ll_artificial_viscosity=float(value("ll_artificial_viscosity")),
        ll_artificial_viscosity_adaptive=bool(value("ll_artificial_viscosity_adaptive")),
        ll_artificial_viscosity_adaptive_alpha=float(value("ll_artificial_viscosity_adaptive_alpha")),
        ll_artificial_viscosity_adaptive_dalpha=float(value("ll_artificial_viscosity_adaptive_dalpha")),
        ll_loads_avl=bool(value("ll_loads_avl")),
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
    mesh_flat: bool,
) -> WingOptions:
    if args.save_force_history:
        loads_start = 1
        loads_avg = False
    else:
        loads_start = max(1, int(n_steps) - int(args.loads_average_window))
        loads_avg = True
    return WingOptions(
        discretization_method=WingMethod.LIFTING_LINE,
        mesh_flat=bool(mesh_flat),
        output_options=OutputOptions(
            compute_loads=True,
            loads_start=loads_start,
            loads_end=int(n_steps),
            loads_step=1,
            loads_avg=loads_avg,
            loads_reference="0",
            compute_spanwise=True,
            spanwise_start=loads_start,
            spanwise_end=int(n_steps),
            spanwise_step=1,
            spanwise_avg=loads_avg,
            spanwise_size=int(args.spanwise_output_size),
            spanwise_lifting_line_data=True,
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
                y_min_m=0.0,
                metric_prefix="neuralfoil_full",
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
            "neuralfoil_full",
            0.0,
        ),
    )
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


def _profile_cd_for_geometry_variant(
    row: dict[str, Any],
    geometry_variant: str,
) -> float:
    normalized = str(geometry_variant).strip().lower()
    if normalized in {
        "full_bwb_equivalent",
        "full_bwb_anchor_equivalent",
        "full_bwb_smooth_equivalent",
        "full_equivalent",
        "full_anchor_equivalent",
        "resolved",
    }:
        return float(row.get("neuralfoil_full_profile_cd", 0.0))
    if normalized in {"outer_only", "calibrated_outer", "rectangular"}:
        return float(row.get("neuralfoil_outer_profile_cd", 0.0))
    return float(row.get("neuralfoil_transition_outer_profile_cd", 0.0))


def _add_reconstructed_drag(
    row: dict[str, Any],
    *,
    geometry_variant: str,
    e_eff: float,
) -> dict[str, Any]:
    profile_cd = _profile_cd_for_geometry_variant(row, geometry_variant)
    ar_eff = float(row.get("mesh_lifting_line_ar_eff", float("nan")))
    cl_for_drag = float(row.get("cl_wind", row.get("cl", 0.0)))
    row.update(
        compute_reconstructed_drag_coefficients(
            cl=cl_for_drag,
            cd_profile=profile_cd,
            ar_eff=ar_eff,
            e_eff=float(e_eff),
        ),
    )
    return row


def _add_alpha_slope_checks(result_rows: list[dict[str, Any]]) -> None:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for row in result_rows:
        key = (
            row.get("geometry_variant"),
            row.get("mesh_flat"),
            row.get("ll_preset"),
            row.get("span_panel_type"),
            row.get("ll_damp"),
            row.get("ll_loads_avl"),
            row.get("polar_source"),
        )
        groups.setdefault(key, []).append(row)
    for rows in groups.values():
        by_alpha = {round(float(row.get("alpha_deg", 0.0)), 6): row for row in rows}
        if -3.0 in by_alpha and 3.0 in by_alpha:
            cl_minus = float(
                by_alpha[-3.0].get(
                    "history_final_window_cl_wind_mean",
                    by_alpha[-3.0].get("cl_wind", by_alpha[-3.0].get("cl", 0.0)),
                ),
            )
            cl_plus = float(
                by_alpha[3.0].get(
                    "history_final_window_cl_wind_mean",
                    by_alpha[3.0].get("cl_wind", by_alpha[3.0].get("cl", 0.0)),
                ),
            )
            slope = (cl_plus - cl_minus) / 6.0
            for row in rows:
                row["dcl_dalpha_wind_per_deg_m3_p3"] = slope
                row["dcl_dalpha_positive_flag"] = 1.0 if slope > 0.0 else 0.0
        if 0.0 in by_alpha and 3.0 in by_alpha:
            cd_0 = float(
                by_alpha[0.0].get(
                    "cd_reconstructed",
                    by_alpha[0.0].get(
                        "history_final_window_cd_wind_mean",
                        by_alpha[0.0].get("cd_wind", 0.0),
                    ),
                ),
            )
            cd_3 = float(
                by_alpha[3.0].get(
                    "cd_reconstructed",
                    by_alpha[3.0].get(
                        "history_final_window_cd_wind_mean",
                        by_alpha[3.0].get("cd_wind", 0.0),
                    ),
                ),
            )
            for row in rows:
                row["cd_increases_0_to_3_flag"] = 1.0 if cd_3 >= cd_0 else 0.0


def _naca0012_station_coordinates(n: int = 80) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = np.linspace(0.0, 1.0, int(n))
    thickness = 0.12
    yt = 5.0 * thickness * (
        0.2969 * np.sqrt(np.maximum(x, 0.0))
        - 0.1260 * x
        - 0.3516 * x**2
        + 0.2843 * x**3
        - 0.1015 * x**4
    )
    return x, yt, -yt


def _rectangular_lifting_line_geometry(
    *,
    half_span_m: float = 30.0,
    chord_m: float = 3.0,
    n_stations: int = 21,
) -> tuple[Any, dict[str, float]]:
    x, upper, lower = _naca0012_station_coordinates()
    stations = []
    for idx, y in enumerate(np.linspace(0.0, float(half_span_m), int(n_stations))):
        stations.append(
            SimpleNamespace(
                name=f"rect_{idx:03d}",
                spanwise_y_m=float(y),
                leading_edge_x_m=0.0,
                leading_edge_z_m=0.0,
                chord_m=float(chord_m),
                twist_deg=0.0,
                x_over_c=x,
                upper_z_over_c=upper,
                lower_z_over_c=lower,
            ),
        )
    area = 2.0 * float(half_span_m) * float(chord_m)
    return SimpleNamespace(resolved_stations=stations), {
        f"{COMPONENT_NAME}.planform_area_m2": area,
        f"{COMPONENT_NAME}.mean_aerodynamic_chord_m": float(chord_m),
        f"{COMPONENT_NAME}.rectangular_half_span_m": float(half_span_m),
    }


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


def _parse_string_list(raw: str) -> list[str]:
    return [item.strip() for item in str(raw).split(",") if item.strip()]


def _parse_float_list(raw: str) -> list[float]:
    return [float(item.strip()) for item in str(raw).split(",") if item.strip()]


def _parse_bool_list(raw: str) -> list[bool]:
    out: list[bool] = []
    for item in _parse_string_list(raw):
        normalized = item.lower()
        if normalized in {"1", "true", "t", "yes", "y"}:
            out.append(True)
        elif normalized in {"0", "false", "f", "no", "n"}:
            out.append(False)
        else:
            msg = f"Cannot parse boolean value {item!r}"
            raise ValueError(msg)
    return out


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
        "cl_wind": "cl_wind",
        "cd_wind": "cd_wind",
        "cd_reconstructed": "cd_reconstructed",
        "cm": "cm",
        "ld": "ld",
        "cl_mean": "history_final_window_cl_mean",
        "cd_mean": "history_final_window_cd_mean",
        "ld_mean": "history_final_window_ld_mean",
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
            "cl_wind": _float(row, "cl_wind"),
            "cd_wind": _float(row, "cd_wind"),
            "cd_reconstructed": _float(row, "cd_reconstructed"),
            "drag_source": row.get("drag_source", ""),
            "ll_case_accepted": _float(row, "ll_case_accepted"),
            "cl_mean": _float(row, "history_final_window_cl_mean"),
            "cd_mean": _float(row, "history_final_window_cd_mean"),
            "ld_mean": _float(row, "history_final_window_ld_mean"),
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

    if history_rows:
        rolling_window = 10
        history_start_step = 5
        steps = sorted({int(float(row.get("n_steps", 0))) for row in history_rows})

        def _rolling_band(
            arr: "np.ndarray", w: int
        ) -> "tuple[np.ndarray, np.ndarray]":
            lo = np.empty_like(arr)
            hi = np.empty_like(arr)
            for i in range(len(arr)):
                sl = arr[max(0, i - w + 1) : i + 1]
                lo[i] = sl.min()
                hi[i] = sl.max()
            return lo, hi

        hist_colors = plt.cm.tab10(
            np.linspace(0, 0.9, max(1, len(steps)))
        )
        fig, (ax_cl, ax_cd, ax_ld) = plt.subplots(3, 1, figsize=(13.0, 10.5), sharex=True)
        for color, step_count in zip(hist_colors, steps):
            case_hist = [
                row for row in history_rows
                if int(float(row.get("n_steps", 0))) == step_count
                and int(float(row["step_index"])) >= 1
            ]
            if not case_hist:
                continue
            xs = np.array([int(float(row["step_index"])) for row in case_hist])
            cls = np.array([_float(row, "cl_wind") for row in case_hist])
            cds = np.array([_float(row, "cd_wind") for row in case_hist])
            lds = np.array([_float(row, "ld_wind") for row in case_hist])
            cl_lo, cl_hi = _rolling_band(cls, rolling_window)
            cd_lo, cd_hi = _rolling_band(cds, rolling_window)
            ld_lo, ld_hi = _rolling_band(lds, rolling_window)
            cl_mean = float(np.mean(cls))
            cd_mean = float(np.mean(cds))
            ld_mean = float(np.mean(lds))
            label = (
                f"{step_count} steps  |  "
                f"CL={cl_mean:.4f}  CD={cd_mean:.4f}  L/D={ld_mean:.2f}"
            )
            ax_cl.plot(xs, cls, linewidth=1.2, color=color, alpha=0.75, label=label)
            ax_cl.fill_between(xs, cl_lo, cl_hi, color=color, alpha=0.13)
            ax_cl.axhline(cl_mean, color=color, linewidth=1.5, linestyle="--", alpha=0.9)
            ax_cd.plot(xs, cds, linewidth=1.2, color=color, alpha=0.75)
            ax_cd.fill_between(xs, cd_lo, cd_hi, color=color, alpha=0.13)
            ax_cd.axhline(cd_mean, color=color, linewidth=1.5, linestyle="--", alpha=0.9)
            ax_ld.plot(xs, lds, linewidth=1.2, color=color, alpha=0.75)
            ax_ld.fill_between(xs, ld_lo, ld_hi, color=color, alpha=0.13)
            ax_ld.axhline(ld_mean, color=color, linewidth=1.5, linestyle="--", alpha=0.9)
        ax_cl.set_ylabel("CL (wind axes) [-]")
        ax_cl.set_title(
            f"CTA DUST LL — convergencia fuerzas (ejes viento, a partir de paso {history_start_step})\n"
            f"banda min/max ventana {rolling_window} pasos · línea discontinua = media global"
        )
        ax_cl.grid(True, alpha=0.25)
        ax_cl.legend(ncol=2, fontsize=8)
        ax_cd.set_ylabel("CD (wind axes) [-]")
        ax_cd.grid(True, alpha=0.25)
        ax_ld.set_xlabel("Time step")
        ax_ld.set_ylabel("L/D (wind axes) [-]")
        ax_ld.grid(True, alpha=0.25)
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
        description="Run CTA baseline DUST lifting-line diagnostic cases.",
    )
    parser.add_argument("--steps", default="70")
    parser.add_argument("--meshes", default="21x1,41x1")
    parser.add_argument(
        "--geometry-modes",
        default="transition_outer",
        help=(
            "Comma-separated LL geometry variants: outer_only, transition_outer, "
            "full_bwb_equivalent, full_bwb_anchor_equivalent, calibrated_outer, "
            "equivalent, resolved, rectangular."
        ),
    )
    parser.add_argument("--mesh-flat-options", default="T,F")
    parser.add_argument("--ll-presets", default="conservative,nominal")
    parser.add_argument("--span-types", default="uniform,cosine")
    parser.add_argument(
        "--alphas",
        default=None,
        help="Comma-separated AoA list. Overrides --alpha-deg when set.",
    )
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
        default=None,
        help=(
            "Deprecated alias for --lifting-line-effective-start-y-m."
        ),
    )
    parser.add_argument("--lifting-line-effective-start-y-m", type=float, default=12.5)
    parser.add_argument("--lifting-line-root-chord-scale", type=float, default=1.0)
    parser.add_argument("--lifting-line-root-twist-deg", type=float, default=None)
    parser.add_argument("--no-lifting-line-root-blend", action="store_true")
    parser.add_argument("--lifting-line-anchor-min-spacing-m", type=float, default=1.5)
    parser.add_argument("--loads-average-window", type=int, default=20)
    parser.add_argument("--spanwise-output-size", type=int, default=80)
    parser.add_argument("--profile-drag", choices=["none", "neuralfoil"], default="neuralfoil")
    parser.add_argument("--polar-sources", default="neuralfoil")
    parser.add_argument("--sanitize-polars", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--polar-cd-min", type=float, default=0.006)
    parser.add_argument("--neuralfoil-model", default="large")
    parser.add_argument("--neuralfoil-n-crit", type=float, default=9.0)
    parser.add_argument(
        "--polar-mach",
        type=float,
        default=0.4,
        help=(
            "Mach at which NeuralFoil computes section polars. "
            "A Prandtl-Glauert correction scales the result to flight Mach. "
            "Use 0.4 for transonic cruise (M=0.8) where NeuralFoil is unreliable at flight Mach. "
            "Set to 0 to disable correction and run NeuralFoil at flight Mach."
        ),
    )
    parser.add_argument(
        "--profile-alpha-mode",
        choices=["global", "global-plus-twist", "global-minus-twist"],
        default="global-plus-twist",
    )
    parser.add_argument("--profile-station-stride", type=int, default=1)
    parser.add_argument("--outer-start-y-m", type=float, default=22.5)
    parser.add_argument("--transition-start-y-m", type=float, default=12.5)
    parser.add_argument("--ll-damps", default=None)
    parser.add_argument("--ll-loads-avl-options", default=None)
    parser.add_argument("--reconstructed-drag-e-eff", type=float, default=0.75)
    parser.add_argument(
        "--diagnostic-suite",
        action="store_true",
        help=(
            "Run a compact baseline diagnostic matrix: alpha -3/0/3, "
            "outer and transition LL variants, synthetic and NeuralFoil polars."
        ),
    )
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
    parser.add_argument("--ll-solver", default=None)
    parser.add_argument(
        "--ll-reynolds-corrections",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument("--ll-reynolds-corrections-nfact", type=float, default=None)
    parser.add_argument("--ll-max-iter", dest="max_iter", type=int, default=None)
    parser.add_argument("--ll-tol", dest="tol", type=float, default=None)
    parser.add_argument("--ll-damp", type=float, default=None)
    parser.add_argument(
        "--ll-stall-regularisation",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument("--ll-stall-regularisation-nelems", type=int, default=None)
    parser.add_argument("--ll-stall-regularizations-niters", type=int, default=None)
    parser.add_argument("--ll-stall-regularization-alpha-stall", type=float, default=None)
    parser.add_argument("--ll-artificial-viscosity", type=float, default=None)
    parser.add_argument(
        "--ll-artificial-viscosity-adaptive",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument("--ll-artificial-viscosity-adaptive-alpha", type=float, default=None)
    parser.add_argument("--ll-artificial-viscosity-adaptive-dalpha", type=float, default=None)
    parser.add_argument(
        "--ll-loads-avl",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
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
    if args.diagnostic_suite:
        geometry_modes = [
            "full_bwb_anchor_equivalent",
            "full_bwb_equivalent",
            "outer_only",
            "transition_outer",
            "rectangular",
        ]
        mesh_flat_options = [True]
        ll_presets = ["conservative"]
        span_types = ["cosine"]
        alphas = [-3.0, 0.0, 3.0]
        polar_sources = ["neuralfoil", "synthetic"]
        ll_damps: list[float | None] = [0.10]
        ll_loads_avl_options: list[bool | None] = [False]
    else:
        geometry_modes = _parse_string_list(args.geometry_modes)
        mesh_flat_options = _parse_bool_list(args.mesh_flat_options)
        ll_presets = _parse_string_list(args.ll_presets)
        span_types = _parse_string_list(args.span_types)
        alphas = _parse_float_list(args.alphas) if args.alphas else [float(args.alpha_deg)]
        polar_sources = _parse_string_list(args.polar_sources)
        ll_damps = _parse_float_list(args.ll_damps) if args.ll_damps else [None]
        ll_loads_avl_options = (
            _parse_bool_list(args.ll_loads_avl_options)
            if args.ll_loads_avl_options
            else [None]
        )
    cases = []
    for alpha in alphas:
        for n_steps in steps:
            for span, chord in meshes:
                for geometry_mode in geometry_modes:
                    for mesh_flat in mesh_flat_options:
                        for ll_preset in ll_presets:
                            for span_type in span_types:
                                for polar_source in polar_sources:
                                    for ll_damp in ll_damps:
                                        for ll_loads_avl in ll_loads_avl_options:
                                            cases.append(
                                                {
                                                    "alpha_deg": alpha,
                                                    "n_steps": n_steps,
                                                    "mesh_span_stations": span,
                                                    "mesh_chord_stations": chord,
                                                    "geometry_mode": geometry_mode,
                                                    "mesh_flat": mesh_flat,
                                                    "ll_preset": ll_preset,
                                                    "span_type": span_type,
                                                    "polar_source": polar_source,
                                                    "ll_damp": ll_damp,
                                                    "ll_loads_avl": ll_loads_avl,
                                                },
                                            )
    if args.max_cases is not None:
        cases = cases[: int(args.max_cases)]

    _mesh_path, geometry_metrics, geometry_state = cta_common.baseline_geometry(output_dir)
    rectangular_geometry_state, rectangular_geometry_metrics = _rectangular_lifting_line_geometry()
    design_row = cta_common.design_variable_row()
    box_metrics = cta_common.evaluate_boxes(geometry_state, cta_common.DEFAULT_CONSTRAINTS)

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
        print("polar_sources:", polar_sources)
        return

    for index, case in enumerate(cases, start=1):
        run_dir = output_dir / "cases" / args.run_directory_name
        n_steps = int(case["n_steps"])
        span_count = int(case["mesh_span_stations"])
        chord_count = int(case["mesh_chord_stations"])
        geometry_variant = str(case["geometry_mode"])
        geometry_mode = "resolved" if geometry_variant == "rectangular" else geometry_variant
        mesh_flat = bool(case["mesh_flat"])
        ll_preset = str(case["ll_preset"])
        span_type = str(case["span_type"])
        alpha_deg = float(case["alpha_deg"])
        polar_source = str(case["polar_source"])
        ll_damp = case["ll_damp"]
        ll_loads_avl = case["ll_loads_avl"]
        if args.lifting_line_start_y_m is not None:
            import warnings
            warnings.warn(
                "--lifting-line-start-y-m is deprecated; use --lifting-line-effective-start-y-m instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            effective_start_y = float(args.lifting_line_start_y_m)
        else:
            effective_start_y = float(args.lifting_line_effective_start_y_m)
        env = cta_common.build_environment(
            SimpleNamespace(
                altitude_ft=float(args.altitude_ft),
                mach=float(args.mach),
                alpha_deg=alpha_deg,
                disa_k=float(args.disa_k),
            ),
        )
        if geometry_variant == "rectangular":
            active_geometry_state = rectangular_geometry_state
            active_geometry_metrics = rectangular_geometry_metrics
            active_box_metrics = {"packaging_metrics_applicable": 0.0}
            s_ref_m2 = rectangular_geometry_metrics[f"{COMPONENT_NAME}.planform_area_m2"]
            c_ref_m = rectangular_geometry_metrics[
                f"{COMPONENT_NAME}.mean_aerodynamic_chord_m"
            ]
            profile_metrics = {
                **Neuralfoil.estimate_profile_drag_from_resolved_stations(
                    rectangular_geometry_state.resolved_stations,
                    env,
                    s_ref_m2=s_ref_m2,
                    y_min_m=0.0,
                    alpha_mode=args.profile_alpha_mode,
                    model=args.neuralfoil_model,
                    n_crit=float(args.neuralfoil_n_crit),
                    metric_prefix="neuralfoil_outer",
                ),
                **Neuralfoil.estimate_profile_drag_from_resolved_stations(
                    rectangular_geometry_state.resolved_stations,
                    env,
                    s_ref_m2=s_ref_m2,
                    y_min_m=0.0,
                    alpha_mode=args.profile_alpha_mode,
                    model=args.neuralfoil_model,
                    n_crit=float(args.neuralfoil_n_crit),
                    metric_prefix="neuralfoil_transition_outer",
                ),
            }
        else:
            active_geometry_state = geometry_state
            active_geometry_metrics = geometry_metrics
            active_box_metrics = {"packaging_metrics_applicable": 1.0, **box_metrics}
            s_ref_m2 = geometry_metrics[f"{COMPONENT_NAME}.planform_area_m2"]
            c_ref_m = geometry_metrics[f"{COMPONENT_NAME}.mean_aerodynamic_chord_m"]
            profile_metrics = _profile_metrics(
                args,
                geometry_state=geometry_state,
                env=env,
                s_ref_m2=s_ref_m2,
            )
        case_tag = (
            f"case_{index:04d}_steps_{n_steps:04d}_mesh_{span_count}x{chord_count}"
            f"_aoa_{alpha_deg:+05.1f}_{geometry_variant}_flat_{'T' if mesh_flat else 'F'}"
            f"_{ll_preset}_{span_type}_{polar_source}"
        )
        print(f"[{index}/{len(cases)}] {case_tag}")

        option_overrides: dict[str, Any] = {}
        if ll_damp is not None:
            option_overrides["ll_damp"] = float(ll_damp)
        if ll_loads_avl is not None:
            option_overrides["ll_loads_avl"] = bool(ll_loads_avl)
        options = _dust_options(
            args,
            n_steps=n_steps,
            run_dir=run_dir,
            write_vtk=bool(args.write_vtk),
            ll_preset_name=ll_preset,
            overrides=option_overrides,
        )
        wing_options = _wing_options(args, n_steps=n_steps, mesh_flat=mesh_flat)
        mesh_settings = DustMeshSettings(
            n_span_stations=span_count,
            n_chord_stations=chord_count,
            span_spacing=args.span_spacing,
            chord_spacing=args.chord_spacing,
            span_curvature_weight=float(args.span_curvature_weight),
            chord_curvature_weight=float(args.chord_curvature_weight),
            chord_endpoint_weight=float(args.chord_endpoint_weight),
            mirror_span=True,
            span_panel_type=span_type,
            lifting_line_geometry_mode=geometry_mode,
            lifting_line_geometry_variant=None if geometry_variant == "rectangular" else geometry_variant,
            lifting_line_effective_start_y_m=effective_start_y,
            lifting_line_outer_start_y_m=float(args.outer_start_y_m),
            lifting_line_transition_start_y_m=float(args.transition_start_y_m),
            lifting_line_root_chord_scale=float(args.lifting_line_root_chord_scale),
            lifting_line_root_twist_deg=args.lifting_line_root_twist_deg,
            lifting_line_blend_root_to_start=not bool(args.no_lifting_line_root_blend),
            lifting_line_anchor_min_spacing_m=float(args.lifting_line_anchor_min_spacing_m),
        )

        start = time.perf_counter()
        polar_diagnostics: dict[str, float] = {}
        try:
            result = run_dust_lifting_line_case_from_prepared_geometry(
                active_geometry_state,
                environment=env,
                options=options,
                s_ref_m2=s_ref_m2,
                c_ref_m=c_ref_m,
                mesh_settings=mesh_settings,
                wing_options=wing_options,
                clean_run_dir=True,
                component_name=COMPONENT_NAME,
                polar_provider=build_neuralfoil_polar_provider(
                    source=polar_source,
                    model=args.neuralfoil_model,
                    n_crit=float(args.neuralfoil_n_crit),
                    sanitize=bool(args.sanitize_polars),
                    cd_min=float(args.polar_cd_min),
                    mach_polar=float(args.polar_mach) if float(args.polar_mach) > 0.0 else None,
                    diagnostics=polar_diagnostics,
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
                "solver_method": SOLVER_LABEL,
                "profile_drag_model": args.profile_drag,
                "polar_source": polar_source,
                "polar_sanitization_enabled": 1.0 if args.sanitize_polars else 0.0,
                "neuralfoil_model": args.neuralfoil_model,
                "profile_alpha_mode": args.profile_alpha_mode,
                "alpha_deg": alpha_deg,
                "n_steps": n_steps,
                "dt_s": float(args.dt),
                "mesh_span_stations": span_count,
                "mesh_chord_stations": chord_count,
                "geometry_variant": geometry_variant,
                "lifting_line_geometry_mode": geometry_mode,
                "lifting_line_effective_start_y_m": effective_start_y,
                "lifting_line_root_chord_scale": float(args.lifting_line_root_chord_scale),
                "lifting_line_anchor_min_spacing_m": float(
                    args.lifting_line_anchor_min_spacing_m,
                ),
                "mesh_flat": 1.0 if mesh_flat else 0.0,
                "ll_preset": ll_preset,
                "ll_damp": float(options.ll_damp),
                "ll_loads_avl": 1.0 if bool(options.ll_loads_avl) else 0.0,
                "span_panel_type": span_type,
                "span_spacing": args.span_spacing,
                "chord_spacing": args.chord_spacing,
                "loads_average_window": int(args.loads_average_window),
                **design_row,
                **active_geometry_metrics,
                **active_box_metrics,
                **polar_diagnostics,
                **profile_metrics,
                **result.to_flat_dict(),
                **cta_common.final_window_stats(history, int(args.loads_average_window)),
            }
            _add_reconstructed_drag(
                row,
                geometry_variant=geometry_variant,
                e_eff=float(args.reconstructed_drag_e_eff),
            )
            row.update(validate_lifting_line_case(row))
            for step_index, history_row in enumerate(history, start=1):
                flat_history = {
                    "case_index": index,
                    "case_tag": case_tag,
                    "step_index": step_index,
                    "n_steps": n_steps,
                    "mesh_span_stations": span_count,
                    "mesh_chord_stations": chord_count,
                    "geometry_variant": geometry_variant,
                    "lifting_line_geometry_mode": geometry_mode,
                    "mesh_flat": 1.0 if mesh_flat else 0.0,
                    "ll_preset": ll_preset,
                    "ll_damp": float(options.ll_damp),
                    "ll_loads_avl": 1.0 if bool(options.ll_loads_avl) else 0.0,
                    "span_panel_type": span_type,
                    "polar_source": polar_source,
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
                "polar_source": polar_source,
                "polar_sanitization_enabled": 1.0 if args.sanitize_polars else 0.0,
                "neuralfoil_model": args.neuralfoil_model,
                "profile_alpha_mode": args.profile_alpha_mode,
                "alpha_deg": alpha_deg,
                "n_steps": n_steps,
                "dt_s": float(args.dt),
                "mesh_span_stations": span_count,
                "mesh_chord_stations": chord_count,
                "geometry_variant": geometry_variant,
                "lifting_line_geometry_mode": geometry_mode,
                "lifting_line_effective_start_y_m": effective_start_y,
                "lifting_line_root_chord_scale": float(args.lifting_line_root_chord_scale),
                "lifting_line_anchor_min_spacing_m": float(
                    args.lifting_line_anchor_min_spacing_m,
                ),
                "mesh_flat": 1.0 if mesh_flat else 0.0,
                "ll_preset": ll_preset,
                "ll_damp": float(
                    option_overrides.get(
                        "ll_damp",
                        args.ll_damp
                        if args.ll_damp is not None
                        else lifting_line_solver_preset(ll_preset)["ll_damp"],
                    ),
                ),
                "ll_loads_avl": (
                    1.0
                    if bool(
                        option_overrides.get(
                            "ll_loads_avl",
                            args.ll_loads_avl
                            if args.ll_loads_avl is not None
                            else lifting_line_solver_preset(ll_preset)["ll_loads_avl"],
                        ),
                    )
                    else 0.0
                ),
                "span_panel_type": span_type,
                "span_spacing": args.span_spacing,
                "chord_spacing": args.chord_spacing,
                "loads_average_window": int(args.loads_average_window),
                **design_row,
                **active_geometry_metrics,
                **active_box_metrics,
                **profile_metrics,
                **polar_diagnostics,
            }

        result_rows.append(row)
        cta_common.write_rows_csv(results_csv, result_rows)

    _add_alpha_slope_checks(result_rows)
    cta_common.write_rows_csv(results_csv, result_rows)
    summary = _write_summary(output_dir, result_rows)
    cta_common.write_workbook(output_dir, "cta_dust_lifting_line_convergence_results.xlsx", result_rows, design_row, summary_rows=summary)
    _plot_results(output_dir, result_rows, history_rows)
    print("CTA lifting-line convergence completed")
    print(f"  results = {results_csv}")
    if args.save_force_history:
        print(f"  force_history = {history_csv}")
    print(f"  workbook = {output_dir / 'cta_dust_lifting_line_convergence_results.xlsx'}")


if __name__ == "__main__":
    main()
