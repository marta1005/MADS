"""Análisis completo del dataset DOE CTA — inputs + outputs + correlaciones.

Genera 4 figuras en el mismo directorio que el CSV de entrada:
  fig1_inputs_coverage.png   — scatter strip de las 14 variables de diseño
  fig2_geometry_outputs.png  — distribuciones de outputs geométricos
  fig3_aero_outputs.png      — distribuciones de outputs aerodinámicos
  fig4_correlations.png      — heatmap Spearman inputs → outputs clave

Uso:
    python examples/cta_analyse_doe.py \
        --input outputs/CTA_case/datasets/campaign_panel_10k/cta_dust_doe_dataset_flat_nf.csv
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from scipy import stats

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "examples"))
sys.path.insert(0, str(REPO_ROOT / "src"))

import cta_geometry as cta  # noqa: E402

VARS = cta.CTA_DOE_DESIGN_VARIABLES
INPUT_COLS = [f"inputs.cta_{v.name}" for v in VARS]

KEY_AERO = [
    ("outputs.cta_dust_cl_wind",                 "CL (wind axis)"),
    ("outputs.cta_dust_cd_total_full_aircraft",   "CD total"),
    ("outputs.cta_dust_neuralfoil_full_profile_cd","CD profile (NF)"),
    ("outputs.cta_dust_cd_induced_full_aircraft", "CD induced"),
    ("outputs.cta_dust_ld_full_aircraft",         "L/D full"),
    ("outputs.cta_dust_cm",                       "CM (body)"),
]

KEY_GEOM = [
    ("outputs.cta_wing.span_m",                  "Span [m]"),
    ("outputs.cta_wing.planform_area_m2",         "Área [m²]"),
    ("outputs.cta_wing.enclosed_volume_m3",       "Volumen [m³]"),
    ("outputs.cta_wing.root_chord_m",             "Cuerda raíz [m]"),
    ("outputs.cta_wing.mean_aerodynamic_chord_m", "MAC [m]"),
    ("outputs.cta_wing.tip_chord_m",              "Cuerda punta [m]"),
]


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _save(fig: plt.Figure, path: Path, title: str) -> None:
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Guardado: {path.name}")


def _filter_valid(df: pd.DataFrame, col: str, lo: float = -1e6, hi: float = 1e6) -> pd.Series:
    s = df[col] if col in df.columns else pd.Series(dtype=float)
    return s[(s > lo) & (s < hi)].dropna()


# ---------------------------------------------------------------------------
# Fig 1 — input scatter coverage
# ---------------------------------------------------------------------------

def fig_inputs(df: pd.DataFrame, out_dir: Path, max_pts: int = 5000) -> None:
    df_plot = df.sample(min(max_pts, len(df)), random_state=42) if len(df) > max_pts else df
    rng = np.random.default_rng(0)

    ncols, n_vars = 4, len(VARS)
    nrows = (n_vars + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 3.8, nrows * 2.6))
    axes = axes.flatten()

    for i, var in enumerate(VARS):
        ax = axes[i]
        col = f"inputs.cta_{var.name}"
        lb, ub, baseline = float(var.lb), float(var.ub), float(var.value)

        if col in df_plot.columns:
            vals = df_plot[col].dropna().values
            jitter = rng.uniform(-0.3, 0.3, size=len(vals))
            ax.scatter(vals, jitter, s=2, alpha=0.2, color="#4C72B0",
                       linewidths=0, rasterized=True)

        ax.axvline(lb,       color="#888", linewidth=0.8, linestyle="--", zorder=3)
        ax.axvline(ub,       color="#888", linewidth=0.8, linestyle="--", zorder=3)
        ax.axvline(baseline, color="red",  linewidth=1.5, zorder=4)
        ax.set_xlim(lb - 0.05 * (ub - lb), ub + 0.05 * (ub - lb))
        ax.set_ylim(-0.6, 0.6)
        ax.set_yticks([])
        ax.set_title(var.name, fontsize=8, pad=3)
        ax.tick_params(axis="x", labelsize=7)
        ax.spines[["left", "right", "top"]].set_visible(False)
        ax.text(lb,       -0.55, f"{lb:.3g}",       ha="left",   va="bottom", fontsize=6, color="#555")
        ax.text(ub,       -0.55, f"{ub:.3g}",       ha="right",  va="bottom", fontsize=6, color="#555")
        ax.text(baseline,  0.52, f"{baseline:.3g}",  ha="center", va="bottom", fontsize=6, color="red")

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    n_ok = int(df.get("outputs.cta_dust_success", pd.Series(dtype=float)).eq(1.0).sum()) \
        if "outputs.cta_dust_success" in df.columns else "?"
    fig.suptitle(
        f"Cobertura inputs DOE — {len(df)} muestras ({n_ok} éxito)  |  "
        "azul=Sobol  rojo=baseline  gris=bounds",
        fontsize=9,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    _save(fig, out_dir / "fig1_inputs_coverage.png", "inputs")


# ---------------------------------------------------------------------------
# Fig 2 — geometric outputs
# ---------------------------------------------------------------------------

def fig_geometry(df: pd.DataFrame, out_dir: Path) -> None:
    ncols = 3
    nrows = (len(KEY_GEOM) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 4, nrows * 3))
    axes = axes.flatten()

    for i, (col, label) in enumerate(KEY_GEOM):
        ax = axes[i]
        vals = _filter_valid(df, col)
        if len(vals) == 0:
            ax.set_visible(False)
            continue
        ax.hist(vals, bins=50, color="#55A868", alpha=0.85, edgecolor="none")
        ax.axvline(vals.median(), color="red", linewidth=1.4, linestyle="--", label=f"mediana={vals.median():.3g}")
        ax.set_title(label, fontsize=9)
        ax.tick_params(labelsize=8)
        ax.legend(fontsize=7)
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("Distribución de outputs geométricos (casos con éxito)", fontsize=10)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    _save(fig, out_dir / "fig2_geometry_outputs.png", "geometry")


# ---------------------------------------------------------------------------
# Fig 3 — aero outputs + success breakdown
# ---------------------------------------------------------------------------

def fig_aero(df: pd.DataFrame, out_dir: Path) -> None:
    success_col = "outputs.cta_dust_success"
    df_ok = df[df[success_col] == 1.0] if success_col in df.columns else df

    # clip outliers for L/D
    if "outputs.cta_dust_ld_full_aircraft" in df_ok.columns:
        df_ok = df_ok[df_ok["outputs.cta_dust_ld_full_aircraft"].abs() < 100]

    ncols = 3
    n_aero = len(KEY_AERO)
    nrows_aero = (n_aero + ncols - 1) // ncols
    total_rows = nrows_aero + 1   # +1 for success row

    fig = plt.figure(figsize=(ncols * 4, total_rows * 3))

    # --- success/failure pie + failure codes ---
    ax_pie = fig.add_subplot(total_rows, ncols, 1)
    ax_fc  = fig.add_subplot(total_rows, ncols, 2)
    ax_geo = fig.add_subplot(total_rows, ncols, 3)

    if success_col in df.columns:
        n_ok  = int((df[success_col] == 1.0).sum())
        n_fail = len(df) - n_ok
        ax_pie.pie(
            [n_ok, n_fail],
            labels=[f"Éxito\n{n_ok}", f"Fallo\n{n_fail}"],
            colors=["#55A868", "#C44E52"],
            autopct="%1.1f%%", startangle=90,
            textprops={"fontsize": 8},
        )
        ax_pie.set_title("Tasa de éxito DUST", fontsize=9)

        fc_col = "outputs.cta_dust_failure_code"
        if fc_col in df.columns:
            fail_df = df[df[success_col] != 1.0]
            fc_counts = fail_df[fc_col].value_counts()
            ax_fc.bar(fc_counts.index.astype(str), fc_counts.values, color="#C44E52", alpha=0.8)
            ax_fc.set_title("Códigos de fallo DUST", fontsize=9)
            ax_fc.set_xlabel("failure_code", fontsize=8)
            ax_fc.tick_params(labelsize=8)

        gc_col = "outputs.cta_geometry_failure_code"
        if gc_col in df.columns:
            gc_counts = df[gc_col].value_counts()
            ax_geo.bar(gc_counts.index.astype(str), gc_counts.values, color="#8172B2", alpha=0.8)
            ax_geo.set_title("Códigos de fallo geometría", fontsize=9)
            ax_geo.set_xlabel("geom_failure_code", fontsize=8)
            ax_geo.tick_params(labelsize=8)
    else:
        ax_pie.set_visible(False)
        ax_fc.set_visible(False)
        ax_geo.set_visible(False)

    # --- aero histograms ---
    for i, (col, label) in enumerate(KEY_AERO):
        ax = fig.add_subplot(total_rows, ncols, ncols + 1 + i)
        vals = _filter_valid(df_ok, col)
        if len(vals) == 0:
            ax.set_visible(False)
            continue
        ax.hist(vals, bins=60, color="#4C72B0", alpha=0.85, edgecolor="none")
        med = vals.median()
        p5, p95 = vals.quantile(0.05), vals.quantile(0.95)
        ax.axvline(med,  color="red",    linewidth=1.4, linestyle="--", label=f"p50={med:.3g}")
        ax.axvline(p5,   color="orange", linewidth=0.9, linestyle=":",  label=f"p5={p5:.3g}")
        ax.axvline(p95,  color="orange", linewidth=0.9, linestyle=":",  label=f"p95={p95:.3g}")
        ax.set_title(label, fontsize=9)
        ax.tick_params(labelsize=8)
        ax.legend(fontsize=6.5)
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))

    fig.suptitle(
        f"Distribución outputs aerodinámicos — {len(df_ok)} casos válidos de {len(df)} totales",
        fontsize=10,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    _save(fig, out_dir / "fig3_aero_outputs.png", "aero")


# ---------------------------------------------------------------------------
# Fig 4 — Spearman correlation heatmap inputs → key outputs
# ---------------------------------------------------------------------------

def fig_correlations(df: pd.DataFrame, out_dir: Path) -> None:
    success_col = "outputs.cta_dust_success"
    df_ok = df[df[success_col] == 1.0].copy() if success_col in df.columns else df.copy()

    # clip L/D outliers
    ld_col = "outputs.cta_dust_ld_full_aircraft"
    if ld_col in df_ok.columns:
        df_ok = df_ok[df_ok[ld_col].abs() < 100]

    output_targets = [(c, lbl) for c, lbl in KEY_AERO if c in df_ok.columns]
    input_present  = [(col, var.name) for col, var in zip(INPUT_COLS, VARS) if col in df_ok.columns]

    if not input_present or not output_targets:
        print("  (sin datos suficientes para correlaciones)")
        return

    in_cols  = [c for c, _ in input_present]
    in_names = [n for _, n in input_present]
    out_cols = [c for c, _ in output_targets]
    out_names = [n for _, n in output_targets]

    corr_matrix = np.zeros((len(in_cols), len(out_cols)))
    for j, oc in enumerate(out_cols):
        for i, ic in enumerate(in_cols):
            sub = df_ok[[ic, oc]].dropna()
            if len(sub) > 10:
                corr_matrix[i, j] = stats.spearmanr(sub[ic], sub[oc]).statistic
            else:
                corr_matrix[i, j] = np.nan

    fig, ax = plt.subplots(figsize=(len(out_cols) * 1.4 + 3, len(in_cols) * 0.55 + 2))
    im = ax.imshow(corr_matrix, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    fig.colorbar(im, ax=ax, label="Spearman ρ")

    ax.set_xticks(range(len(out_names)))
    ax.set_xticklabels(out_names, rotation=35, ha="right", fontsize=9)
    ax.set_yticks(range(len(in_names)))
    ax.set_yticklabels(in_names, fontsize=8)

    for i in range(len(in_cols)):
        for j in range(len(out_cols)):
            v = corr_matrix[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                        fontsize=7, color="white" if abs(v) > 0.5 else "black")

    ax.set_title(
        f"Correlación Spearman — inputs vs outputs  ({len(df_ok)} casos válidos)",
        fontsize=10, pad=10,
    )
    plt.tight_layout()
    _save(fig, out_dir / "fig4_correlations.png", "correlations")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", required=True, help="Flat DOE CSV (NF post-procesado o raw)")
    parser.add_argument("--max-scatter-pts", type=int, default=5000,
                        help="Máx. puntos en scatter inputs (default 5000)")
    args = parser.parse_args()

    csv_path = Path(args.input)
    out_dir  = csv_path.parent
    print(f"Leyendo {csv_path}")
    df = pd.read_csv(csv_path)
    print(f"  {len(df)} filas × {len(df.columns)} columnas")

    print("Fig 1 — cobertura inputs …")
    fig_inputs(df, out_dir, args.max_scatter_pts)

    print("Fig 2 — outputs geométricos …")
    fig_geometry(df, out_dir)

    print("Fig 3 — outputs aerodinámicos …")
    fig_aero(df, out_dir)

    print("Fig 4 — correlaciones Spearman …")
    fig_correlations(df, out_dir)

    print("Listo.")


if __name__ == "__main__":
    main()
