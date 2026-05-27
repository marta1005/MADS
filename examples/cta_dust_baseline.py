from __future__ import annotations

import csv
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Any

import numpy as np

from multiads.assembly import Environment
from multiads.solvers.aerodynamics import dust_lib as dl
from multiads.solvers.aerodynamics.dust_mesh import (
    write_basic_two_skin_mesh_from_resolved_npz,
)


MADS_ROOT = Path(__file__).resolve().parents[1]
MESH_NPZ = MADS_ROOT / "outputs" / "cta_geometry_export" / "cta_resolved_mesh.npz"
SUMMARY_JSON = MADS_ROOT / "outputs" / "cta_geometry_export" / "summary.json"
OUT_ROOT = MADS_ROOT / "outputs" / "cta_dust_aoa_sweep"

FT_TO_M = 0.3048
AOA_DEG = (0.0, 5.0, 10.0)
MACH = 0.8
ALTITUDE_M = 40000.0 * FT_TO_M
DISA_K = 0.0


def _dust_executable(name: str) -> Path:
    if dust_bin_dir := os.environ.get("CTA_DUST_BIN_DIR"):
        return Path(dust_bin_dir) / name
    return Path(name)


def run_case(alpha_deg: float, summary: dict[str, Any]) -> dict[str, Any]:
    case_tag = f"aoa_{int(round(alpha_deg)):02d}"
    run_dir = OUT_ROOT / case_tag
    if run_dir.exists():
        shutil.rmtree(run_dir)
    (run_dir / "geometry").mkdir(parents=True, exist_ok=True)
    (run_dir / "Output").mkdir(parents=True, exist_ok=True)
    (run_dir / "Postpro").mkdir(parents=True, exist_ok=True)

    mesh_prefix = Path("geometry") / "cta_basic_"
    mesh_info = write_basic_two_skin_mesh_from_resolved_npz(
        MESH_NPZ,
        run_dir / mesh_prefix,
        n_span_stations=33,
        n_chord_stations=33,
        leading_edge_opening_m=0.05,
        leading_edge_opening_extent=0.12,
    )

    env_probe = Environment(name="env", height=ALTITUDE_M, speed=1.0)
    speed = MACH * float(env_probe.sound_speed)
    env = Environment(name="env", height=ALTITUDE_M, speed=speed, alpha=alpha_deg)

    wing = dl.Wing(
        name="cta_wing",
        sections=[],
        spans=[],
        method=dl.WingMethod.PANELS,
        panel_type=dl.WingPanelType.UNIFORM,
        num_panels=1,
        mesh_file=mesh_prefix,
        mesh_file_type="basic",
        symmetry=True,
        inner_product_te=0.5,
        tol_se_wing=1.0e-3,
        proj_te=True,
        proj_te_dir="parallel",
        proj_te_vector=np.array([1.0, 0.0, 0.0]),
        options=dl.OutputOptions(compute_loads=True, loads_reference="0"),
    )

    n_steps = 40
    options = dl.Options(
        name=f"cta_{case_tag}",
        dust_pre=_dust_executable("dust_pre"),
        dust=_dust_executable("dust"),
        dust_post=_dust_executable("dust_post"),
        output_dir=Path("Output"),
        post_dir=Path("Postpro"),
        t_start=0.0,
        t_end=0.20,
        dt=0.005,
        dt_out=0.005,
        output_start=True,
        n_threads=6,
        n_wake_panels=60,
        n_wake_particles=50000,
        particles_box_min=np.array([-80.0, -90.0, -80.0]),
        particles_box_max=np.array([380.0, 90.0, 80.0]),
        penetration_avoidance=False,
    )
    driver = dl.Driver(environment=env, options=options, wings=[wing])
    loads = dl.PostLoads(
        name="cta_wing_loads",
        start_res=n_steps - 10,
        end_res=n_steps,
        step_res=1,
        average=True,
        reference="0",
        components=["cta_wing"],
    )

    old_cwd = Path.cwd()
    os.chdir(run_dir)
    try:
        driver.preprocess()
        driver.run()
        driver.postprocess([loads])
    finally:
        os.chdir(old_cwd)

    force = np.asarray(loads.f[-1], dtype=float)
    moment = np.asarray(loads.m[-1], dtype=float)
    if not np.all(np.isfinite(force)) or not np.all(np.isfinite(moment)):
        msg = f"DUST returned non-finite loads for alpha={alpha_deg:g} deg."
        raise RuntimeError(msg)

    q_inf = 0.5 * float(env.density) * speed**2
    geom = summary["geometry_outputs"]
    s_ref = float(geom["cta_wing.planform_area_m2"])
    c_ref = float(geom["cta_wing.mean_aerodynamic_chord_m"])

    drag_n = -float(force[0])
    side_n = float(force[1])
    lift_n = float(force[2])

    return {
        "alpha_deg": float(alpha_deg),
        "mach": MACH,
        "altitude_ft": 40000.0,
        "disa_k": DISA_K,
        "speed_mps": speed,
        "rho_kg_m3": float(env.density),
        "q_pa": q_inf,
        "s_ref_m2": s_ref,
        "c_ref_m": c_ref,
        "fx_wind_n": float(force[0]),
        "fy_wind_n": float(force[1]),
        "fz_wind_n": float(force[2]),
        "mx_wind_nm": float(moment[0]),
        "my_wind_nm": float(moment[1]),
        "mz_wind_nm": float(moment[2]),
        "lift_n": lift_n,
        "drag_n": drag_n,
        "side_n": side_n,
        "cl": lift_n / (q_inf * s_ref),
        "cd": drag_n / (q_inf * s_ref),
        "cy": side_n / (q_inf * s_ref),
        "cm": float(moment[1]) / (q_inf * s_ref * c_ref),
        "run_dir": str(run_dir),
        **{f"mesh_{key}": value for key, value in mesh_info.items()},
    }


def main() -> None:
    if not MESH_NPZ.exists() or not SUMMARY_JSON.exists():
        msg = "Run examples/cta_geometry.py before launching the CTA DUST baseline."
        raise FileNotFoundError(msg)

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    summary = json.loads(SUMMARY_JSON.read_text())
    results = [run_case(alpha, summary) for alpha in AOA_DEG]

    csv_path = OUT_ROOT / "cta_dust_aoa_sweep_results.csv"
    json_path = OUT_ROOT / "cta_dust_aoa_sweep_results.json"
    fieldnames = list(results[0].keys())
    with csv_path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
    json_path.write_text(json.dumps(results, indent=2, sort_keys=True))

    print(f"Wrote {csv_path}")
    for row in results:
        print(
            "AoA={alpha_deg:.1f} deg  CL={cl:.8f}  CD={cd:.8f}  CM={cm:.8f}  "
            "L={lift_n:.3f} N  D={drag_n:.3f} N  My={my_wind_nm:.3f} Nm".format(
                **row,
            ),
        )


if __name__ == "__main__":
    sys.exit(main())
