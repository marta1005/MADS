"""Plot CTA 14-variable design space coverage via normalised ECDF.

Each variable is normalised to [0, 1] using its design bounds and the
empirical CDF is compared to the ideal uniform CDF (diagonal).
A perfect Sobol sequence would lie exactly on the diagonal.

Usage:
    python examples/cta_plot_design_space.py \
        --input outputs/CTA_case/datasets/campaign_panel_10k/cta_dust_doe_dataset_flat.txt \
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
    parser.add_argument("--input", required=True, help="Flat DOE CSV (raw o post-procesado)")
    parser.add_argument("--output", default="design_space_coverage.png")
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    n_total = len(df)

    ncols = 4
    nrows = (len(VARS) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 3.2, nrows * 3.0))
    axes = axes.flatten()

    max_ks = 0.0

    for i, var in enumerate(VARS):
        ax = axes[i]
        col = f"inputs.{var.name}"
        lb, ub = float(var.lb), float(var.ub)

        if col in df.columns:
            raw = pd.to_numeric(df[col], errors="coerce").dropna().values
            # normalise to [0, 1]
            x_norm = np.clip((raw - lb) / (ub - lb), 0.0, 1.0)
            x_sorted = np.sort(x_norm)
            n = len(x_sorted)
            ecdf_y = np.arange(1, n + 1) / n

            ax.plot(x_sorted, ecdf_y, color="#4C72B0", linewidth=1.2, label="Sobol")
            ax.fill_between(x_sorted, ecdf_y, x_sorted,
                            where=(ecdf_y >= x_sorted),
                            alpha=0.15, color="#4C72B0")
            ax.fill_between(x_sorted, ecdf_y, x_sorted,
                            where=(ecdf_y < x_sorted),
                            alpha=0.15, color="#C44E52")

            ks = float(np.max(np.abs(ecdf_y - x_sorted)))
            max_ks = max(max_ks, ks)
            ax.text(0.97, 0.05, f"KS={ks:.3f}", transform=ax.transAxes,
                    ha="right", va="bottom", fontsize=6.5, color="#333")

        # ideal uniform diagonal
        ax.plot([0, 1], [0, 1], color="#999", linewidth=0.8,
                linestyle="--", label="uniforme ideal")

        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_title(var.name, fontsize=7.5, pad=3)
        ax.tick_params(labelsize=6.5)
        ax.set_xlabel("valor norm.", fontsize=6)
        ax.set_ylabel("CDF", fontsize=6)
        ax.spines[["right", "top"]].set_visible(False)

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    # shared legend
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower right", fontsize=8,
               bbox_to_anchor=(0.98, 0.01), framealpha=0.9)

    fig.suptitle(
        f"Cobertura Sobol — ECDF normalizada por variable  ({n_total} muestras)\n"
        "azul=Sobol  gris=uniforme ideal  |  KS máx = "
        f"{max_ks:.3f}  (0 = perfecto)",
        fontsize=9,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.94])

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
