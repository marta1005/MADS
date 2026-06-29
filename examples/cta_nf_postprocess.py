"""Post-process a CTA DOE flat CSV to add NeuralFoil drag columns.

Usage (after the DOE campaign has finished and shards have been merged):

    python examples/cta_nf_postprocess.py \
        --input  outputs/CTA_case/datasets/campaign_panel_10k/cta_dust_doe_dataset_flat.csv \
        --output outputs/CTA_case/datasets/campaign_panel_10k/cta_dust_doe_dataset_flat_nf.csv \
        --polar-mach 0.4 \
        --polar-cd-min 0.006

The script re-generates the CTA planform geometry for each sample row (no DUST)
and calls NeuralFoil to fill in:
    outputs.cta_dust_neuralfoil_full_profile_cd
    outputs.cta_dust_cd_induced_full_aircraft
    outputs.cta_dust_cd_total_full_aircraft
    outputs.cta_dust_ld_full_aircraft

Rows where DUST failed (success != 1) or where |cl_wind| > 50 are skipped and
their NF columns are left as 0.0.

Progress is printed every --log-every rows (default 50).
"""
from __future__ import annotations

import argparse
import logging
import math
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "examples"))
sys.path.insert(0, str(REPO_ROOT / "src"))

import cta_geometry as cta  # noqa: E402
from multiads.assembly import Environment  # noqa: E402
from multiads.solvers.aerodynamics.neuralfoil import Neuralfoil  # noqa: E402

logging.basicConfig(level=logging.WARNING, format="%(levelname)s - %(message)s")
_log = logging.getLogger(__name__)

INPUT_COLS = [f"inputs.{v.name}" for v in cta.CTA_DOE_DESIGN_VARIABLES]
DESIGN_VAR_NAMES = [v.name for v in cta.CTA_DOE_DESIGN_VARIABLES]

NF_COLS = [
    "outputs.cta_dust_neuralfoil_full_profile_cd",
    "outputs.cta_dust_cd_induced_full_aircraft",
    "outputs.cta_dust_cd_total_full_aircraft",
    "outputs.cta_dust_ld_full_aircraft",
]


def _build_environment(alpha_deg: float, mach: float, altitude_ft: float) -> Environment:
    altitude_m = altitude_ft * 0.3048
    probe = Environment(name="env", height=altitude_m, speed=1.0)
    return Environment(
        name="env",
        height=altitude_m,
        speed=mach * float(probe.sound_speed),
        alpha=alpha_deg,
    )


def _run_geometry(row: pd.Series) -> object | None:
    """Run planform mapping + geometry discipline for one CSV row.

    Returns geometry_state (with .resolved_stations) or None on failure.
    """
    # Build input dict for planform mapping
    input_data = {name: np.atleast_1d(float(row[f"inputs.{name}"]))
                  for name in DESIGN_VAR_NAMES}

    try:
        mapped = cta.disc_planform_mapping.execute(input_data=input_data)
    except Exception as exc:
        _log.debug("Planform mapping failed: %s", exc)
        return None

    try:
        cta.disc_geometry.execute(input_data=mapped)
    except Exception as exc:
        _log.debug("Geometry discipline failed: %s", exc)
        return None

    resolved_wing = cta.disc_geometry.components[0]
    return getattr(resolved_wing, "geometry_state", None)


def _nf_for_row(
    row: pd.Series,
    environment: Environment,
    mach_polar: float,
    cd_min_friction: float,
    model: str,
) -> dict:
    """Return NF drag dict for one CSV row, or zeros on failure."""
    zeros = {col: 0.0 for col in NF_COLS}

    cl_wind_col = "outputs.cta_dust_cl_wind"
    success_col = "outputs.cta_dust_success"

    if row.get(success_col, 0.0) != 1.0:
        return zeros

    cl_wind = float(row.get(cl_wind_col, 0.0))
    if not math.isfinite(cl_wind) or abs(cl_wind) > 50.0:
        return zeros

    geometry_state = _run_geometry(row)
    if geometry_state is None or not geometry_state.resolved_stations:
        return zeros

    try:
        s_ref = float(row.get("outputs.cta_wing.planform_area_m2",
                               row.get("inputs.s_ref_m2", 845.26)))
    except Exception:
        s_ref = 845.26

    try:
        nf_metrics = Neuralfoil.estimate_profile_drag_from_resolved_stations(
            geometry_state.resolved_stations,
            environment,
            s_ref_m2=s_ref,
            y_min_m=0.0,
            mach_polar=mach_polar,
            cd_min_friction=cd_min_friction,
            model=model,
            metric_prefix="neuralfoil_full",
        )
        cd_profile = float(nf_metrics.get("neuralfoil_full_profile_cd", 0.0))
    except Exception as exc:
        _log.debug("NeuralFoil failed: %s", exc)
        return zeros

    stations = geometry_state.resolved_stations
    span_m = 2.0 * max(float(st.spanwise_y_m) for st in stations) if stations else float("nan")
    ar_full = span_m ** 2 / s_ref if s_ref > 0.0 else float("nan")
    cd_induced = cl_wind ** 2 / (math.pi * ar_full) if math.isfinite(ar_full) else 0.0
    cd_total = cd_profile + cd_induced
    ld_full = cl_wind / cd_total if cd_total > 1.0e-14 else float("nan")

    return {
        "outputs.cta_dust_neuralfoil_full_profile_cd": cd_profile,
        "outputs.cta_dust_cd_induced_full_aircraft": cd_induced,
        "outputs.cta_dust_cd_total_full_aircraft": cd_total,
        "outputs.cta_dust_ld_full_aircraft": ld_full,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input",  required=True, help="Flat CSV from the DOE campaign")
    parser.add_argument("--output", required=True, help="Output CSV with NF columns added")
    parser.add_argument("--polar-mach",    type=float, default=0.4)
    parser.add_argument("--polar-cd-min",  type=float, default=0.006)
    parser.add_argument("--polar-model",   default="large")
    parser.add_argument("--alpha-deg",     type=float, default=3.0)
    parser.add_argument("--mach",          type=float, default=0.8)
    parser.add_argument("--altitude-ft",   type=float, default=39370.0)
    parser.add_argument("--log-every",     type=int,   default=50)
    args = parser.parse_args()

    print(f"Reading {args.input}")
    df = pd.read_csv(args.input)
    n = len(df)
    print(f"  {n} rows")

    # Ensure NF columns exist (initialised to 0)
    for col in NF_COLS:
        if col not in df.columns:
            df[col] = 0.0

    # Only re-process rows where NF outputs are zero (cluster run failed)
    needs_nf = (
        (df.get("outputs.cta_dust_success", pd.Series(0.0, index=df.index)) == 1.0)
        & (df[NF_COLS].abs().sum(axis=1) < 1.0e-12)
    )
    rows_to_process = df.index[needs_nf].tolist()
    print(f"  {len(rows_to_process)} rows need NeuralFoil (success=1 but NF=0)")

    environment = _build_environment(args.alpha_deg, args.mach, args.altitude_ft)

    t0 = time.perf_counter()
    n_done = 0
    n_ok = 0

    for idx in rows_to_process:
        row = df.loc[idx]
        result = _nf_for_row(row, environment, args.polar_mach, args.polar_cd_min, args.polar_model)
        for col, val in result.items():
            df.at[idx, col] = val

        if any(v != 0.0 for v in result.values()):
            n_ok += 1

        n_done += 1
        if n_done % args.log_every == 0:
            elapsed = time.perf_counter() - t0
            rate = n_done / elapsed
            remaining = (len(rows_to_process) - n_done) / rate if rate > 0 else float("nan")
            print(f"  [{n_done}/{len(rows_to_process)}]  "
                  f"{rate:.1f} casos/s  ~{remaining/60:.1f} min restantes  "
                  f"NF OK: {n_ok}/{n_done}")

    elapsed = time.perf_counter() - t0
    print(f"\nDone: {n_done} procesados, {n_ok} con NF OK en {elapsed:.0f}s")

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"Guardado: {out}")


if __name__ == "__main__":
    main()
