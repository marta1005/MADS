"""Plot CTA wing geometry: planform, front view, section profiles and distributions."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt

import cta_geometry as cta
from multiads.utilities.geometry_plots import plot_geometry

REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_PATH = REPO_ROOT / "outputs" / "CTA_case" / "export" / "geometry" / "cta_geometry_plot.png"


def main() -> None:
    input_data = {v.name: v.value_np.copy() for v in cta.variables}
    cta.disc_geometry.execute(input_data=input_data)

    resolved_wing = cta.disc_geometry.components[0]
    geometry_state = resolved_wing.geometry_state

    fig = plot_geometry(geometry_state, symmetric=True, figsize=(16, 12))

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_PATH, dpi=150, bbox_inches="tight")
    print(f"Plot saved: {OUTPUT_PATH}")
    plt.show()


if __name__ == "__main__":
    main()
