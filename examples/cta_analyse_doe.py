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
# var.name already contains the cta_ prefix  (e.g. "cta_delta_c0_m")
INPUT_COLS = [f"inputs.{v.name}" for v in VARS]
INPUT_NAMES = [v.name for v in VARS]

# (column, label, lo_clip, hi_clip)
# NF-corrected cols preferred; fallback to raw cols handled in _resolve_aero_cols()
KEY_AERO_NF = [
    ("outputs.cta_dust_cl_wind",                  "CL (wind)",        -3.0,  3.0),
    ("outputs.cta_dust_cd_total_full_aircraft",    "CD total (NF)",     0.0,  0.3),
    ("outputs.cta_dust_neuralfoil_full_profile_cd","CD profile (NF)",   0.0,  0.3),
    ("outputs.cta_dust_cd_induced_full_aircraft",  "CD induced",        0.0,  0.3),
    ("outputs.cta_dust_ld_full_aircraft",          "L/D full (NF)",   -80.0, 80.0),
    ("outputs.cta_dust_cm",                        "CM (body)",        -3.0,  3.0),
]
KEY_AERO_RAW = [
    ("outputs.cta_dust_cl_wind",  "CL (wind)",  -3.0,  3.0),
    ("outputs.cta_dust_cd",       "CD (body)",   0.0,  0.3),
    ("outputs.cta_dust_ld",       "L/D (body)", -80.0, 80.0),
    ("outputs.cta_dust_cm",       "CM (body)",  -3.0,  3.0),
    ("outputs.cta_dust_cd_wind",  "CD (wind)",   0.0,  0.3),
    ("outputs.cta_dust_ld_wind",  "L/D (wind)", -80.0, 80.0),
]

KEY_GEOM = [
    ("outputs.cta_wing.span_m",                   "Span [m]",      60.0, 100.0),
    ("outputs.cta_wing.planform_area_m2",          "Área [m²]",    400.0, 1400.0),
    ("outputs.cta_wing.enclosed_volume_m3",        "Volumen [m³]", 500.0, 4000.0),
    ("outputs.cta_wing.root_chord_m",              "Cuerda raíz [m]", 20.0, 70.0),
    ("outputs.cta_wing.mean_aerodynamic_chord_m",  "MAC [m]",       10.0, 40.0),
    ("outputs.cta_wing.tip_chord_m",               "Cuerda punta [m]", 0.0, 5.0),
]


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _save(fig: plt.Figure, path: Path) -> None:
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Guardado: {path.name}")


def _clipped(df: pd.DataFrame, col: str, lo: float, hi: float) -> pd.Series:
    if col not in df.columns:
        return pd.Series(dtype=float)
    s = pd.to_numeric(df[col], errors="coerce")
    return s[(s >= lo) & (s <= hi)].dropna()


def _resolve_aero_cols(df: pd.DataFrame) -> list[tuple[str, str, float, float]]:
    """Use NF-corrected cols if they have non-zero data, else fall back to raw."""
    nf_cd = "outputs.cta_dust_cd_total_full_aircraft"
    if nf_cd in df.columns and df[nf_cd].abs().max() > 1e-9:
        return [(c, l, lo, hi) for c, l, lo, hi in KEY_AERO_NF if c in df.columns]
    return [(c, l, lo, hi) for c, l, lo, hi in KEY_AERO_RAW if c in df.columns]


def _df_success(df: pd.DataFrame) -> pd.DataFrame:
    sc = "outputs.cta_dust_success"
    if sc in df.columns:
        return df[pd.to_numeric(df[sc], errors="coerce") == 1.0]
    return df


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
        col = f"inputs.{var.name}"
        lb, ub, baseline = float(var.lb), float(var.ub), float(var.value)

        if col in df_plot.columns:
            vals = pd.to_numeric(df_plot[col], errors="coerce").dropna().values
            jitter = rng.uniform(-0.3, 0.3, size=len(vals))
            ax.scatter(vals, jitter, s=2, alpha=0.2, color="#4C72B0",
                       linewidths=0, rasterized=True)

        ax.axvline(lb,       color="#888", linewidth=0.8, linestyle="--", zorder=3)
        ax.axvline(ub,       color="#888", linewidth=0.8, linestyle="--", zorder=3)
        ax.axvline(baseline, color="red",  linewidth=1.5, zorder=4)
        pad = 0.05 * (ub - lb)
        ax.set_xlim(lb - pad, ub + pad)
        ax.set_ylim(-0.6, 0.6)
        ax.set_yticks([])
        ax.set_title(var.name, fontsize=7, pad=3)
        ax.tick_params(axis="x", labelsize=6)
        ax.spines[["left", "right", "top"]].set_visible(False)
        ax.text(lb,       -0.55, f"{lb:.3g}",      ha="left",   va="bottom", fontsize=5.5, color="#555")
        ax.text(ub,       -0.55, f"{ub:.3g}",      ha="right",  va="bottom", fontsize=5.5, color="#555")
        ax.text(baseline,  0.52, f"{baseline:.3g}", ha="center", va="bottom", fontsize=5.5, color="red")

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    n_ok = int((_df_success(df)).shape[0])
    fig.suptitle(
        f"Cobertura inputs DOE — {len(df)} muestras ({n_ok} éxito)  |  "
        "azul=Sobol  rojo=baseline  gris=bounds",
        fontsize=9,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    _save(fig, out_dir / "fig1_inputs_coverage.png")


# ---------------------------------------------------------------------------
# Fig 2 — geometric outputs
# ---------------------------------------------------------------------------

def fig_geometry(df: pd.DataFrame, out_dir: Path) -> None:
    df_ok = _df_success(df)
    ncols = 3
    nrows = (len(KEY_GEOM) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 4, nrows * 3))
    axes = axes.flatten()

    for i, (col, label, lo, hi) in enumerate(KEY_GEOM):
        ax = axes[i]
        vals = _clipped(df_ok, col, lo, hi)
        if len(vals) < 2:
            ax.set_visible(False)
            continue
        ax.hist(vals, bins=50, color="#55A868", alpha=0.85, edgecolor="none")
        med = vals.median()
        ax.axvline(med, color="red", linewidth=1.4, linestyle="--", label=f"p50 = {med:.4g}")
        ax.set_title(label, fontsize=9)
        ax.tick_params(labelsize=8)
        ax.legend(fontsize=7)
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle(f"Distribución de outputs geométricos  ({len(df_ok)} casos válidos)", fontsize=10)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    _save(fig, out_dir / "fig2_geometry_outputs.png")


# ---------------------------------------------------------------------------
# Fig 3 — aero outputs + success breakdown
# ---------------------------------------------------------------------------

def fig_aero(df: pd.DataFrame, out_dir: Path) -> None:
    df_ok = _df_success(df)
    aero_cols = _resolve_aero_cols(df_ok)
    print(f"  usando cols aero: {[l for _, l, *_ in aero_cols]}")

    ncols = 3
    n_aero = len(aero_cols)
    nrows_aero = (n_aero + ncols - 1) // ncols
    total_rows = nrows_aero + 1

    fig = plt.figure(figsize=(ncols * 4, total_rows * 3))

    # --- fila 0: éxito / failure codes ---
    ax_pie = fig.add_subplot(total_rows, ncols, 1)
    ax_fc  = fig.add_subplot(total_rows, ncols, 2)
    ax_geo = fig.add_subplot(total_rows, ncols, 3)

    sc = "outputs.cta_dust_success"
    if sc in df.columns:
        n_ok   = len(df_ok)
        n_fail = len(df) - n_ok
        if n_fail > 0:
            ax_pie.pie(
                [n_ok, n_fail],
                labels=[f"Éxito\n{n_ok}", f"Fallo\n{n_fail}"],
                colors=["#55A868", "#C44E52"],
                autopct="%1.1f%%", startangle=90,
                textprops={"fontsize": 8},
            )
        else:
            ax_pie.pie([1], labels=[f"100% éxito\n{n_ok} casos"],
                       colors=["#55A868"], textprops={"fontsize": 9})
        ax_pie.set_title("Tasa de éxito DUST", fontsize=9)

        fc_col = "outputs.cta_dust_failure_code"
        if fc_col in df.columns:
            fc = pd.to_numeric(df[fc_col], errors="coerce").dropna().astype(int)
            fc_counts = fc.value_counts().sort_index()
            ax_fc.bar(fc_counts.index.astype(str), fc_counts.values, color="#C44E52", alpha=0.8)
            ax_fc.set_title("Failure code DUST", fontsize=9)
            ax_fc.tick_params(labelsize=8)

        gc_col = "outputs.cta_geometry_failure_code"
        if gc_col in df.columns:
            gc = pd.to_numeric(df[gc_col], errors="coerce").dropna().astype(int)
            gc_counts = gc.value_counts().sort_index()
            ax_geo.bar(gc_counts.index.astype(str), gc_counts.values, color="#8172B2", alpha=0.8)
            ax_geo.set_title("Failure code geometría", fontsize=9)
            ax_geo.tick_params(labelsize=8)

    # --- filas siguientes: histogramas aero ---
    nf_drag_cols = {
        "outputs.cta_dust_cd_total_full_aircraft",
        "outputs.cta_dust_neuralfoil_full_profile_cd",
        "outputs.cta_dust_cd_induced_full_aircraft",
        "outputs.cta_dust_ld_full_aircraft",
    }
    for i, (col, label, lo, hi) in enumerate(aero_cols):
        ax = fig.add_subplot(total_rows, ncols, ncols + 1 + i)
        vals = _clipped(df_ok, col, lo, hi)
        # excluir ceros solo en columnas de drag NF (casos donde NF falló)
        if col in nf_drag_cols:
            vals = vals[vals > 1e-9]
        if len(vals) < 2:
            ax.set_visible(False)
            continue
        # clip adicional p1–p99 para quitar outliers extremos
        p1, p99 = vals.quantile(0.01), vals.quantile(0.99)
        n_total_ok = len(vals)
        vals = vals[(vals >= p1) & (vals <= p99)]
        n_out = len(df_ok) - len(vals)
        ax.hist(vals, bins=60, color="#4C72B0", alpha=0.85, edgecolor="none")
        med  = vals.median()
        p5   = vals.quantile(0.05)
        p95  = vals.quantile(0.95)
        ax.axvline(med, color="red",    linewidth=1.4, linestyle="--", label=f"p50={med:.3g}")
        ax.axvline(p5,  color="orange", linewidth=0.9, linestyle=":",  label=f"p5={p5:.3g}")
        ax.axvline(p95, color="orange", linewidth=0.9, linestyle=":",  label=f"p95={p95:.3g}")
        ax.set_title(f"{label}" + (f"  [{n_out} outliers]" if n_out else ""), fontsize=8)
        ax.tick_params(labelsize=7)
        ax.legend(fontsize=6)
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))

    fig.suptitle(
        f"Outputs aerodinámicos — {len(df_ok)} válidos de {len(df)} totales",
        fontsize=10,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    _save(fig, out_dir / "fig3_aero_outputs.png")


# ---------------------------------------------------------------------------
# Fig 4 — Spearman correlation heatmap
# ---------------------------------------------------------------------------

def fig_correlations(df: pd.DataFrame, out_dir: Path) -> None:
    df_ok = _df_success(df)
    aero_cols = _resolve_aero_cols(df_ok)

    in_present  = [(col, name) for col, name in zip(INPUT_COLS, INPUT_NAMES)
                   if col in df_ok.columns]
    out_present = [(col, label, lo, hi) for col, label, lo, hi in aero_cols]

    if not in_present or not out_present:
        print("  (sin datos suficientes para correlaciones)")
        return

    in_cols   = [c for c, _ in in_present]
    in_names  = [n for _, n in in_present]
    out_cols  = [c for c, l, *_ in out_present]
    out_names = [l for _, l, *_ in out_present]
    out_clips = {c: (lo, hi) for c, _, lo, hi in out_present}

    corr = np.zeros((len(in_cols), len(out_cols)))
    for j, oc in enumerate(out_cols):
        lo, hi = out_clips[oc]
        y = _clipped(df_ok, oc, lo, hi)
        common_idx = y.index
        for i, ic in enumerate(in_cols):
            x = pd.to_numeric(df_ok.loc[common_idx, ic], errors="coerce").dropna()
            idx = x.index.intersection(y.index)
            if len(idx) > 30:
                corr[i, j] = stats.spearmanr(x.loc[idx], y.loc[idx]).statistic
            else:
                corr[i, j] = np.nan

    fig, ax = plt.subplots(figsize=(max(6, len(out_cols) * 1.5 + 2), len(in_cols) * 0.6 + 2))
    im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    fig.colorbar(im, ax=ax, label="Spearman ρ", shrink=0.8)

    ax.set_xticks(range(len(out_names)))
    ax.set_xticklabels(out_names, rotation=35, ha="right", fontsize=9)
    ax.set_yticks(range(len(in_names)))
    ax.set_yticklabels(in_names, fontsize=7)

    for i in range(len(in_cols)):
        for j in range(len(out_cols)):
            v = corr[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                        fontsize=7.5, color="white" if abs(v) > 0.55 else "black")

    ax.set_title(
        f"Correlación Spearman — inputs vs outputs  ({len(df_ok)} casos válidos)",
        fontsize=10, pad=10,
    )
    plt.tight_layout()
    _save(fig, out_dir / "fig4_correlations.png")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", required=True)
    parser.add_argument("--max-scatter-pts", type=int, default=5000)
    args = parser.parse_args()

    csv_path = Path(args.input)
    out_dir  = csv_path.parent
    print(f"Leyendo {csv_path.name}  →  {csv_path}")
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
