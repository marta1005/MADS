"""Plot CTA main section airfoil shapes and NeuralFoil polars."""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import cta_geometry as cta
from multiads.assembly import Environment
from multiads.solvers.aerodynamics.neuralfoil import Neuralfoil
from multiads.utilities.cst import evaluate_cst_surface, normalized_chordwise_points


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
ALPHAS = np.linspace(-10, 20, 61)
MACHS_TO_PLOT = [0.4, 0.5]
ALTITUDE_M = 12192  # 40 000 ft baseline
N_CRIT = 9.0

MAIN_SECTIONS = ("s0", "s1", "s2", "s3", "s4", "s5")
SECTION_COLORS = plt.cm.viridis(np.linspace(0.0, 1.0, len(MAIN_SECTIONS)))
MACH_LINESTYLES = ["-", "--"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _airfoil_coords(section_id: str, n: int = 241) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    profile = cta.CTA_PROFILE_DATA[section_id]
    x = normalized_chordwise_points(n, "cosine")
    upper = evaluate_cst_surface(x, profile["upper"], n1=cta.CTA_CST_N1, n2=cta.CTA_CST_N2,
                                  trailing_edge_thickness_over_c=profile["te"], sign=1.0)
    lower = evaluate_cst_surface(x, profile["lower"], n1=cta.CTA_CST_N1, n2=cta.CTA_CST_N2,
                                  trailing_edge_thickness_over_c=profile["te"], sign=-1.0)
    return x, upper, lower


def _nf_coords(section_id: str, n: int = 241) -> np.ndarray:
    x, upper, lower = _airfoil_coords(section_id, n)
    return np.vstack((np.column_stack((x[::-1], upper[::-1])), np.column_stack((x[1:], lower[1:]))))


def _polar(section_id: str, chord_m: float, *, mach: float) -> dict:
    env = Environment(name="polar_env", height=ALTITUDE_M, speed=1.0)
    speed = mach * env.sound_speed
    reynolds = env.density * speed * chord_m / env.dyn_viscosity
    coords = _nf_coords(section_id)
    return Neuralfoil.compute_aero_from_coordinates(
        name=f"cta_{section_id}",
        coordinates=coords,
        alphas=ALPHAS,
        mach=mach,
        reynolds=float(reynolds),
        model="large",
        n_crit=N_CRIT,
    )


def _chord(section_id: str) -> float:
    chords = cta.CTA_BASELINE_CHORDS_M
    taper = cta.CTA_CFD_JUNE_14_VARIABLE_INFO["taper_ratio_midwing"][0]
    return {"s0": chords["s0"], "s1": chords["s1"], "s2": chords["s2"],
            "s3": chords["s3"], "s4": chords["s3"] * taper, "s5": chords["s5"]}[section_id]


# ---------------------------------------------------------------------------
# Build polars for each Mach
# ---------------------------------------------------------------------------
# polars_by_mach[mach][section_id] = {"CL": ..., "CD": ..., "CM": ...}
polars_by_mach: dict[float, dict[str, dict]] = {}
for mach in MACHS_TO_PLOT:
    print(f"Computing NeuralFoil polars at M={mach}...")
    polars_by_mach[mach] = {}
    for sid in MAIN_SECTIONS:
        chord = _chord(sid)
        print(f"  {sid:4s}  chord={chord:.2f} m", end="  ", flush=True)
        polars_by_mach[mach][sid] = _polar(sid, chord, mach=mach)
        print("done")

# ---------------------------------------------------------------------------
# Figure 1 — Airfoil shapes (Mach-independent)
# ---------------------------------------------------------------------------
fig_shape, ax_shape = plt.subplots(figsize=(12, 5))
ax_shape.set_aspect("equal")
ax_shape.set_xlabel("x/c")
ax_shape.set_ylabel("z/c")
ax_shape.set_title("CTA main section profiles (normalized, baseline CST)")
ax_shape.axhline(0, color="0.8", lw=0.5, zorder=0)

for color, sid in zip(SECTION_COLORS, MAIN_SECTIONS):
    x, upper, lower = _airfoil_coords(sid)
    idx30 = np.argmin(np.abs(x - 0.30))
    tc = float(upper[idx30] - lower[idx30]) * 100
    label = f"{sid.upper()}  c={_chord(sid):.1f} m  t/c≈{tc:.1f}%"
    ax_shape.plot(x, upper, color=color, lw=1.5, label=label)
    ax_shape.plot(x, lower, color=color, lw=1.5)
    ax_shape.fill_between(x, lower, upper, alpha=0.08, color=color)

ax_shape.legend(fontsize=9, loc="upper right")
fig_shape.tight_layout()

# ---------------------------------------------------------------------------
# Figures 2a/2b — CL, CD, polar  (one figure per Mach)
# ---------------------------------------------------------------------------
for mach, ls in zip(MACHS_TO_PLOT, MACH_LINESTYLES):
    fig_pol, axes = plt.subplots(1, 3, figsize=(16, 5))
    ax_cl, ax_cd, ax_drag = axes

    ax_cl.set_xlabel("α [deg]");  ax_cl.set_ylabel("CL");  ax_cl.set_title("Lift curve")
    ax_cd.set_xlabel("α [deg]");  ax_cd.set_ylabel("CD");  ax_cd.set_title("Drag curve")
    ax_drag.set_xlabel("CD");     ax_drag.set_ylabel("CL"); ax_drag.set_title("Drag polar")

    for color, sid in zip(SECTION_COLORS, MAIN_SECTIONS):
        p = polars_by_mach[mach][sid]
        cl = np.asarray(p["CL"], dtype=float)
        cd = np.asarray(p["CD"], dtype=float)
        label = sid.upper()
        ax_cl.plot(ALPHAS, cl, color=color, lw=1.5, label=label)
        ax_cd.plot(ALPHAS, cd, color=color, lw=1.5, label=label)
        ax_drag.plot(cd, cl, color=color, lw=1.5, label=label)

    for ax in axes:
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    fig_pol.suptitle(
        f"CTA section polars — NeuralFoil large  M={mach}  h={ALTITUDE_M/0.3048:.0f} ft  Ncrit={N_CRIT}",
        fontsize=11,
    )
    fig_pol.tight_layout()

# ---------------------------------------------------------------------------
# Figure 3 — CL/CD vs α, both Machs overlaid
# ---------------------------------------------------------------------------
fig_ld, ax_ld = plt.subplots(figsize=(9, 5))
ax_ld.set_xlabel("α [deg]")
ax_ld.set_ylabel("CL / CD")
ax_ld.set_title("Section efficiency  CL/CD vs α")
ax_ld.axhline(0, color="0.7", lw=0.5)

for mach, ls in zip(MACHS_TO_PLOT, MACH_LINESTYLES):
    for color, sid in zip(SECTION_COLORS, MAIN_SECTIONS):
        p = polars_by_mach[mach][sid]
        cl = np.asarray(p["CL"], dtype=float)
        cd = np.asarray(p["CD"], dtype=float)
        with np.errstate(divide="ignore", invalid="ignore"):
            ld = np.where(cd > 1e-6, cl / cd, np.nan)
        label = f"{sid.upper()} M={mach}" if sid == MAIN_SECTIONS[0] else f"{sid.upper()}"
        ax_ld.plot(ALPHAS, ld, color=color, lw=1.5, ls=ls,
                   label=f"{sid.upper()} M={mach}")

ax_ld.legend(fontsize=8, ncol=2)
ax_ld.grid(True, alpha=0.3)
fig_ld.tight_layout()

plt.show()
print("\nDone.")
