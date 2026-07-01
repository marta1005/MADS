"""Plot CTA 14-variable design space coverage from a DOE flat CSV.

Usage:
    python examples/cta_plot_design_space.py \
        --input outputs/CTA_case/datasets/campaign_panel_10k/cta_dust_doe_dataset_flat_nf.csv \
        --output outputs/CTA_case/datasets/campaign_panel_10k/design_space_coverage.png
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "examples"))
sys.path.insert(0, str(REPO_ROOT / "src"))

import cta_geometry as cta  # noqa: E402

VARS = cta.CTA_DOE_DESIGN_VARIABLES


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", required=True, help="Flat DOE CSV (post-processed or raw)")
    parser.add_argument("--output", default="design_space_coverage.png")
    parser.add_argument("--max-points", type=int, default=5000,
                        help="Max samples to plot (random subsample if larger, default 5000)")
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    n_total = len(df)

    if n_total > args.max_points:
        df = df.sample(args.max_points, random_state=42)

    n_vars = len(VARS)
    ncols = 4
    nrows = (n_vars + ncols - 1) // ncols

    rng = np.random.default_rng(0)
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 3.8, nrows * 2.8))
    axes = axes.flatten()

    for i, var in enumerate(VARS):
        ax = axes[i]
        col = f"inputs.{var.name}"
        lb, ub, baseline = float(var.lb), float(var.ub), float(var.value)

        if col in df.columns:
            samples = df[col].dropna().values
            jitter = rng.uniform(-0.3, 0.3, size=len(samples))
            ax.scatter(
                samples, jitter,
                s=2, alpha=0.25, color="#4C72B0", linewidths=0,
                rasterized=True,
            )

        # bounds
        ax.axvline(lb, color="#888888", linewidth=0.9, linestyle="--", zorder=3)
        ax.axvline(ub, color="#888888", linewidth=0.9, linestyle="--", zorder=3)
        # baseline
        ax.axvline(baseline, color="red", linewidth=1.5, zorder=4, label="baseline")

        ax.set_xlim(lb - 0.05 * (ub - lb), ub + 0.05 * (ub - lb))
        ax.set_ylim(-0.6, 0.6)
        ax.set_yticks([])
        ax.set_title(var.name, fontsize=8, pad=3)
        ax.tick_params(axis="x", labelsize=7)
        ax.spines[["left", "right", "top"]].set_visible(False)

        # label bounds
        ax.text(lb, -0.55, f"{lb:.3g}", ha="left", va="bottom", fontsize=6, color="#555")
        ax.text(ub, -0.55, f"{ub:.3g}", ha="right", va="bottom", fontsize=6, color="#555")
        ax.text(baseline, 0.52, f"{baseline:.3g}", ha="center", va="bottom",
                fontsize=6, color="red")

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    n_success = (
        int(df.get("outputs.cta_dust_success", pd.Series(dtype=float)).eq(1.0).sum())
        if "outputs.cta_dust_success" in df.columns else "?"
    )
    fig.suptitle(
        f"CTA design space — {n_total} samples ({n_success} éxito)  |  "
        "azul=Sobol  rojo=baseline  gris=bounds",
        fontsize=9,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
