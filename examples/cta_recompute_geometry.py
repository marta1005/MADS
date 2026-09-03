"""Re-evaluate geometry and packaging constraints from an existing DOE flat dataset.

Reads design-variable columns from a GEMSEO flat dataset CSV (produced by a prior
campaign run), re-runs only the geometry and packaging disciplines (no DUST), and
writes a new CSV with the geometry/packaging outputs using the current bwb_ names.

The output can be joined to the original flat dataset on `sample_index` to enrich
the existing dataset with updated or new packaging columns (e.g. MLG1 vertex margins).

Usage:
    python examples/cta_recompute_geometry.py \\
        --dataset datasets/campaign_panel_10k_cp/bwb_dust_panel_dataset_flat.csv \\
        --output  datasets/campaign_panel_10k_cp/bwb_geometry_dataset.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Make sure the MADS source and examples/ are importable when run from anywhere
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).resolve().parents[1]
_EXAMPLES_DIR = Path(__file__).resolve().parent
for _p in (str(_REPO_ROOT / "src"), str(_EXAMPLES_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import cta_geometry as cta  # noqa: E402  (both files live in examples/)

# mlg1_vertex_variables only exists when CTA_INTERNAL_VOLUME_CONSTRAINTS is not None
# AND the constraints file contains an MLG_1 surface — use getattr to be safe.
_MLG1_VARS: list = getattr(cta, "mlg1_vertex_variables", [])


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _find_design_var_columns(columns: list[str]) -> dict[str, str]:
    """Return {variable_name: csv_column} for every DOE design variable."""
    col_set = set(columns)
    mapping: dict[str, str] = {}

    for var in cta.CTA_DOE_DESIGN_VARIABLES:
        found: str | None = None
        # Try each accepted csv key, with and without common group prefixes
        for key in cta.variable_csv_keys(var.name):
            for prefix in ("", "inputs.", "design_parameters.", "parameters."):
                candidate = f"{prefix}{key}"
                if candidate in col_set:
                    found = candidate
                    break
            if found:
                break
        if found is None:
            raise ValueError(
                f"Design variable '{var.name}' not found in dataset columns. "
                f"Tried keys: {cta.variable_csv_keys(var.name)}"
            )
        mapping[var.name] = found

    return mapping


def _evaluate_row(row: "pd.Series", col_map: dict[str, str]) -> dict[str, float]:
    """Run geometry + packaging disciplines for one design point."""
    input_data = {
        var_name: np.array([float(row[col])])
        for var_name, col in col_map.items()
    }

    # 1. Resolve geometry
    try:
        geometry_outputs = cta.disc_geometry.execute(input_data=input_data)
    except Exception:
        return _nan_row()

    geo_metric_values = tuple(
        float(geometry_outputs[var.name][0])
        for var in cta.GEOMETRY_METRIC_VARIABLES
    )

    # 2. Internal-box packaging
    if cta.CTA_INTERNAL_VOLUME_CONSTRAINTS is not None:
        try:
            boxes_result = cta._evaluate_internal_boxes_from_last_geometry(*geo_metric_values)
        except Exception:
            return _nan_row()
        all_boxes_fit = float(boxes_result[0])
        min_margin = float(boxes_result[1])
        vertex_margins = tuple(float(v) for v in boxes_result[2:])
    else:
        all_boxes_fit = float("nan")
        min_margin = float("nan")
        vertex_margins = ()

    # 3. Geometry validation
    if cta.CTA_INTERNAL_VOLUME_CONSTRAINTS is not None:
        val_inputs = geo_metric_values + (all_boxes_fit, min_margin)
    else:
        val_inputs = geo_metric_values
    try:
        geom_valid, geom_failure_code, pkg_valid, pkg_min_margin = cta._validate_cta_geometry(*val_inputs)
    except Exception:
        return _nan_row()

    # Assemble output dict
    result: dict[str, float] = {}
    for var, val in zip(cta.GEOMETRY_METRIC_VARIABLES, geo_metric_values):
        result[var.name] = val
    if cta.CTA_INTERNAL_VOLUME_CONSTRAINTS is not None:
        result[cta.all_boxes_fit.name] = all_boxes_fit
        result[cta.internal_boxes_min_margin.name] = min_margin
        for var, val in zip(_MLG1_VARS, vertex_margins):
            result[var.name] = val
    result[cta.geometry_valid.name] = float(geom_valid)
    result[cta.geometry_failure_code.name] = float(geom_failure_code)
    result[cta.packaging_valid.name] = float(pkg_valid)
    result[cta.packaging_min_margin.name] = float(pkg_min_margin)
    return result


def _nan_row() -> dict[str, float]:
    result: dict[str, float] = {}
    for var in cta.GEOMETRY_METRIC_VARIABLES:
        result[var.name] = float("nan")
    if cta.CTA_INTERNAL_VOLUME_CONSTRAINTS is not None:
        result[cta.all_boxes_fit.name] = float("nan")
        result[cta.internal_boxes_min_margin.name] = float("nan")
        for var in _MLG1_VARS:
            result[var.name] = float("nan")
    result[cta.geometry_valid.name] = float("nan")
    result[cta.geometry_failure_code.name] = float("nan")
    result[cta.packaging_valid.name] = float("nan")
    result[cta.packaging_min_margin.name] = float("nan")
    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        required=True,
        help="Path to the flat dataset CSV from the prior campaign.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output CSV path. Default: <dataset_dir>/bwb_geometry_dataset.csv",
    )
    parser.add_argument(
        "--sample-index-col",
        default=None,
        help=(
            "Column to use as sample index in the output (e.g. 'sample_index', 'case_id'). "
            "If omitted, the row number is used."
        ),
    )
    args = parser.parse_args()

    dataset_path = args.dataset.expanduser().resolve()
    if not dataset_path.exists():
        print(f"ERROR: dataset not found: {dataset_path}", file=sys.stderr)
        sys.exit(1)

    output_path = args.output or dataset_path.parent / "bwb_geometry_dataset.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Loading dataset: {dataset_path}")
    df = pd.read_csv(dataset_path, low_memory=False)
    n_rows = len(df)
    print(f"  {n_rows} rows, {len(df.columns)} columns")

    # Find design-variable columns
    col_map = _find_design_var_columns(list(df.columns))
    print(f"  Mapped {len(col_map)} design variables to CSV columns")

    # Determine index column
    index_col: str | None = args.sample_index_col
    if index_col is not None and index_col not in df.columns:
        print(f"WARNING: --sample-index-col '{index_col}' not found; using row number.", file=sys.stderr)
        index_col = None

    # Evaluate row by row
    records: list[dict[str, object]] = []
    n_errors = 0
    log_every = max(1, n_rows // 20)

    for i, (_, row) in enumerate(df.iterrows()):
        result = _evaluate_row(row, col_map)
        record: dict[str, object] = {"sample_index": int(row[index_col]) if index_col else i}
        record.update(result)
        records.append(record)
        if (i + 1) % log_every == 0 or (i + 1) == n_rows:
            n_nan = sum(1 for r in records if any(np.isnan(v) for v in r.values() if isinstance(v, float)))
            print(f"  {i + 1}/{n_rows} rows evaluated  ({n_nan} with errors so far)")

    out_df = pd.DataFrame(records)
    out_df.to_csv(output_path, index=False)

    n_ok = int((out_df["bwb_geometry_valid"] == 1.0).sum()) if "bwb_geometry_valid" in out_df.columns else -1
    print(f"\nSaved: {output_path}")
    print(f"  Total rows   : {len(out_df)}")
    if n_ok >= 0:
        print(f"  Geometry OK  : {n_ok} / {len(out_df)}")
    print(f"\nTo join with the aero dataset:")
    print(f"  import pandas as pd")
    print(f"  aero = pd.read_csv('bwb_dust_panel_dataset_flat.csv')")
    print(f"  geo  = pd.read_csv('{output_path.name}')")
    print(f"  merged = aero.merge(geo, on='sample_index', suffixes=('', '_geo'))")


if __name__ == "__main__":
    main()
