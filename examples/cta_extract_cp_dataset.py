"""Extract Cp distributions from campaign VTK files into a single compressed dataset.

For each case, reads the DUST surface VTK and saves:
  - cp         : pressure coefficient per panel
  - x_over_c   : chordwise position x/c (0 = LE, 1 = TE), computed per spanwise strip
  - y_over_b   : spanwise position y/b (0 = root, 1 = tip), half-span normalised
  - sample_index: DOE sample index, for joining with the scalar dataset CSV

Usage (after campaign finishes, with pyvista installed):

    python examples/cta_extract_cp_dataset.py \
        --campaign-root outputs/CTA_case/datasets/campaign_panel_10k_cp \
        --output outputs/CTA_case/datasets/campaign_panel_10k_cp/cp_dataset.npz
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np


def _find_vtu_files(campaign_root: Path) -> list[tuple[int, Path]]:
    pattern = re.compile(r"sample_(\d+)_")
    results = []
    for vtu_path in campaign_root.glob("shards/*/cases/*/post/*_visualization-*.vtu"):
        m = pattern.search(vtu_path.parent.parent.name)
        if m:
            results.append((int(m.group(1)), vtu_path))
    return sorted(results)


def _normalized_coords(
    x: np.ndarray,
    y: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute x/c and y/b from panel centroid coordinates."""
    y_max = float(y.max())
    y_over_b = y / y_max if y_max > 1e-10 else np.zeros_like(y)

    x_over_c = np.zeros_like(x)
    y_unique = np.unique(y.round(4))
    for y_val in y_unique:
        mask = np.abs(y - y_val) < 1e-3
        x_strip = x[mask]
        chord = float(x_strip.max() - x_strip.min())
        if chord > 1e-10:
            x_over_c[mask] = (x_strip - x_strip.min()) / chord

    return x_over_c.astype(np.float32), y_over_b.astype(np.float32)


def _extract_case(vtk_path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    import pyvista as pv

    mesh = pv.read(str(vtk_path))
    if "cp" not in mesh.cell_data:
        return None

    cp = np.asarray(mesh.cell_data["cp"], dtype=np.float32)
    centers = np.asarray(mesh.cell_centers().points, dtype=np.float64)
    x_over_c, y_over_b = _normalized_coords(centers[:, 0], centers[:, 1])
    return cp, x_over_c, y_over_b


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--campaign-root", type=Path, required=True, help="Campaign root directory.")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output .npz path. Default: <campaign-root>/cp_dataset.npz",
    )
    args = parser.parse_args()

    output = args.output or args.campaign_root / "cp_dataset.npz"
    output.parent.mkdir(parents=True, exist_ok=True)

    vtk_files = _find_vtu_files(args.campaign_root)
    if not vtk_files:
        print("No .vtu files found. Check --campaign-root and that --store-case-directories was used.")
        return

    print(f"Found {len(vtk_files)} .vtu files. Extracting Cp...")

    cp_list: list[np.ndarray] = []
    xc_list: list[np.ndarray] = []
    yb_list: list[np.ndarray] = []
    indices: list[int] = []

    for i, (sample_idx, vtk_path) in enumerate(vtk_files):
        result = _extract_case(vtk_path)
        if result is None:
            print(f"  [{i+1}/{len(vtk_files)}] skip — no cp field: {vtk_path.name}")
            continue
        cp, xc, yb = result
        cp_list.append(cp)
        xc_list.append(xc)
        yb_list.append(yb)
        indices.append(sample_idx)
        if (i + 1) % 500 == 0 or (i + 1) == len(vtk_files):
            print(f"  {i+1}/{len(vtk_files)} cases processed")

    if not cp_list:
        print("No valid cases extracted.")
        return

    np.savez_compressed(
        output,
        cp=np.array(cp_list, dtype=np.float32),
        x_over_c=np.array(xc_list, dtype=np.float32),
        y_over_b=np.array(yb_list, dtype=np.float32),
        sample_index=np.array(indices, dtype=np.int32),
    )
    shape = np.array(cp_list).shape
    print(f"\nSaved to {output}")
    print(f"  cases     : {shape[0]}")
    print(f"  panels/case: {shape[1]}  (n_span x n_chord, flattened)")
    print(f"\nTo load in Python:")
    print(f"  data = np.load('{output}')")
    print(f"  cp   = data['cp']       # ({shape[0]}, {shape[1]})")
    print(f"  xc   = data['x_over_c'] # ({shape[0]}, {shape[1]})")
    print(f"  yb   = data['y_over_b'] # ({shape[0]}, {shape[1]})")
    print(f"  idx  = data['sample_index']  # ({shape[0]},) — join with scalar CSV")


if __name__ == "__main__":
    main()
