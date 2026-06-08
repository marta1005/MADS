from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

import numpy as np

import cta_geometry as cta
from multiads.assembly import Environment
from multiads.solvers.aerodynamics.dust_lib import normalize_reference_loads
from multiads.solvers.synthesis import build_resolved_surface_mesh, write_resolved_surface_mesh_npz
from multiads.utilities.internal_volume_constraints import (
    CadReferenceFrame,
    evaluate_internal_volume_constraints,
    load_internal_volume_constraint_set,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONSTRAINTS = REPO_ROOT / "assets" / "cta" / "internal_volume_constraints_set1.csv"
COMPONENT_NAME = "cta_wing"


def parse_int_list(raw: str) -> list[int]:
    return [int(item.strip()) for item in raw.split(",") if item.strip()]


def parse_mesh_list(raw: str) -> list[tuple[int, int]]:
    meshes: list[tuple[int, int]] = []
    for item in raw.split(","):
        item = item.strip().lower()
        if not item:
            continue
        span, chord = item.split("x", maxsplit=1)
        meshes.append((int(span), int(chord)))
    return meshes


def build_environment(args: argparse.Namespace) -> Environment:
    altitude_m = float(args.altitude_ft) * 0.3048
    probe = Environment(name="env", height=altitude_m, speed=1.0)
    env = Environment(
        name="env",
        height=altitude_m,
        speed=float(args.mach) * float(probe.sound_speed),
        alpha=float(args.alpha_deg),
    )
    env.disa_k = float(args.disa_k)
    return env


def baseline_geometry(output_dir: Path) -> tuple[Path, dict[str, float], Any]:
    input_data = {variable.name: variable.value_np.copy() for variable in cta.variables}
    geometry_outputs = cta.disc_geometry.execute(input_data=input_data)
    wing = cta.disc_geometry.components[0]
    if getattr(wing, "geometry_state", None) is None:
        msg = "CTA geometry discipline did not write a resolved geometry_state."
        raise RuntimeError(msg)

    geometry_dir = output_dir / "geometry"
    geometry_dir.mkdir(parents=True, exist_ok=True)
    mesh_path = geometry_dir / "cta_baseline_resolved_mesh.npz"
    mesh = build_resolved_surface_mesh(wing.geometry_state)
    write_resolved_surface_mesh_npz(mesh_path, mesh)

    metrics = {
        key: float(np.ravel(value)[0])
        for key, value in geometry_outputs.items()
        if key.startswith(f"{COMPONENT_NAME}.")
    }
    return mesh_path, metrics, wing.geometry_state


def design_variable_row() -> dict[str, float]:
    return {
        variable.name: float(variable.value)
        for variable in cta.CTA_CFD_JUNE_14_DESIGN_VARIABLES
    }


def evaluate_boxes(
    geometry_state: Any,
    constraints_path: Path = DEFAULT_CONSTRAINTS,
) -> dict[str, float]:
    constraint_set = load_internal_volume_constraint_set(
        constraints_path,
        reference_frame=CadReferenceFrame(
            name="CTA CAD reference",
            offset_x_m=0.0,
            offset_y_m=0.0,
            offset_z_m=0.0,
            mirror_about_symmetry_plane=True,
        ),
        name="CTA internal volume boxes",
    )
    result = evaluate_internal_volume_constraints(
        geometry_state,
        constraint_set,
        triangle_resolution=8,
    )
    out: dict[str, float] = {
        "cta_packaging_valid": 1.0 if result.satisfied else 0.0,
        "cta_packaging_min_margin_m": float(result.minimum_margin_m),
    }
    for idx, surface in enumerate(result.surface_results, start=1):
        label = surface.label.lower().replace(":", "_").replace(" ", "_")
        prefix = f"box_{idx:02d}_{label}"
        out[f"{prefix}_fits"] = 1.0 if surface.satisfied else 0.0
        out[f"{prefix}_margin_m"] = float(surface.worst_margin_m)
        out[f"{prefix}_footprint_area_m2"] = float(surface.footprint_area_m2)
        out[f"{prefix}_clearance_volume_m3"] = float(surface.clearance_volume_m3)
        out[f"{prefix}_critical_x_m"] = float(surface.critical_x_m)
        out[f"{prefix}_critical_y_m"] = float(surface.critical_y_m)
        out[f"{prefix}_critical_target_z_m"] = float(surface.critical_target_z_m)
        out[f"{prefix}_critical_geometry_z_m"] = float(surface.critical_geometry_z_m)
    return out


def read_force_history(
    run_dir: Path,
    case_name: str,
    *,
    q_pa: float,
    s_ref_m2: float,
    c_ref_m: float,
    component_name: str = COMPONENT_NAME,
) -> list[dict[str, float]]:
    path = run_dir / "post" / f"{case_name}_{component_name}_loads.dat"
    if not path.exists():
        return []

    rows: list[dict[str, float]] = []
    with path.open(encoding="utf-8") as stream:
        for _ in range(4):
            stream.readline()
        for line in stream:
            parts = line.split()
            if len(parts) < 7:
                continue
            t = float(parts[0])
            force = np.asarray([float(value) for value in parts[1:4]], dtype=float)
            moment = np.asarray([float(value) for value in parts[4:7]], dtype=float)
            normalized = normalize_reference_loads(force, moment, q_pa, s_ref_m2, c_ref_m)
            rows.append(
                {
                    "time_s": t,
                    "fx_reference_n": float(force[0]),
                    "fy_reference_n": float(force[1]),
                    "fz_reference_n": float(force[2]),
                    "mx_reference_nm": float(moment[0]),
                    "my_reference_nm": float(moment[1]),
                    "mz_reference_nm": float(moment[2]),
                    **normalized,
                },
            )
    return rows


def final_window_stats(history: list[dict[str, float]], window: int) -> dict[str, float]:
    if not history:
        return {}
    tail = history[-max(1, int(window)) :]
    keys = [
        "fx_reference_n",
        "fy_reference_n",
        "fz_reference_n",
        "mx_reference_nm",
        "my_reference_nm",
        "mz_reference_nm",
        "cl",
        "cd",
        "cm",
        "cy",
        "ld",
        "lift_n",
        "drag_n",
        "side_n",
    ]
    out: dict[str, float] = {}
    for key in keys:
        values = np.asarray([row[key] for row in tail], dtype=float)
        out[f"history_final_window_{key}_mean"] = float(np.mean(values))
        out[f"history_final_window_{key}_std"] = float(np.std(values))
        out[f"history_final_window_{key}_min"] = float(np.min(values))
        out[f"history_final_window_{key}_max"] = float(np.max(values))
    return out


def augment_row_with_profile_drag(
    row: dict[str, Any],
    *,
    profiles: dict[str, float],
    solver_label: str,
    q_pa: float,
    s_ref_m2: float,
) -> dict[str, Any]:
    outer_cd = float(profiles["neuralfoil_outer_profile_cd"])
    outer_drag = float(profiles["neuralfoil_outer_profile_drag_n"])
    transition_outer_cd = float(profiles["neuralfoil_transition_outer_profile_cd"])
    transition_outer_drag = float(
        profiles["neuralfoil_transition_outer_profile_drag_n"],
    )
    lift = float(row["lift_n"])
    drag = float(row["drag_n"])
    outer_drag_total = drag + outer_drag
    transition_outer_drag_total = drag + transition_outer_drag
    row.update(profiles)
    row[f"cd_total_{solver_label}_plus_outer_profile"] = float(row["cd"]) + outer_cd
    row[f"drag_total_{solver_label}_plus_outer_profile_n"] = outer_drag_total
    row[f"ld_total_{solver_label}_plus_outer_profile"] = (
        lift / outer_drag_total if abs(outer_drag_total) > 1.0e-14 else float("nan")
    )
    row[f"cd_total_{solver_label}_plus_transition_outer_profile"] = (
        float(row["cd"]) + transition_outer_cd
    )
    row[f"drag_total_{solver_label}_plus_transition_outer_profile_n"] = (
        transition_outer_drag_total
    )
    row[f"ld_total_{solver_label}_plus_transition_outer_profile"] = (
        lift / transition_outer_drag_total
        if abs(transition_outer_drag_total) > 1.0e-14
        else float("nan")
    )
    row["outer_profile_drag_qs_check_cd"] = outer_drag / (float(q_pa) * float(s_ref_m2))
    row["transition_outer_profile_drag_qs_check_cd"] = transition_outer_drag / (
        float(q_pa) * float(s_ref_m2)
    )
    return row


def augment_history_with_profile_drag(
    history: list[dict[str, float]],
    *,
    profiles: dict[str, float],
    solver_label: str,
) -> list[dict[str, float]]:
    outer_cd = float(profiles["neuralfoil_outer_profile_cd"])
    outer_drag = float(profiles["neuralfoil_outer_profile_drag_n"])
    transition_outer_cd = float(profiles["neuralfoil_transition_outer_profile_cd"])
    transition_outer_drag = float(
        profiles["neuralfoil_transition_outer_profile_drag_n"],
    )
    out: list[dict[str, float]] = []
    for row in history:
        enriched = dict(row)
        enriched["neuralfoil_outer_profile_cd"] = outer_cd
        enriched["neuralfoil_outer_profile_drag_n"] = outer_drag
        enriched["neuralfoil_transition_outer_profile_cd"] = transition_outer_cd
        enriched["neuralfoil_transition_outer_profile_drag_n"] = transition_outer_drag
        enriched[f"cd_total_{solver_label}_plus_outer_profile"] = float(row["cd"]) + outer_cd
        enriched[f"drag_total_{solver_label}_plus_outer_profile_n"] = (
            float(row["drag_n"]) + outer_drag
        )
        enriched[f"ld_total_{solver_label}_plus_outer_profile"] = (
            float(row["lift_n"])
            / enriched[f"drag_total_{solver_label}_plus_outer_profile_n"]
            if abs(enriched[f"drag_total_{solver_label}_plus_outer_profile_n"]) > 1.0e-14
            else float("nan")
        )
        enriched[f"cd_total_{solver_label}_plus_transition_outer_profile"] = (
            float(row["cd"]) + transition_outer_cd
        )
        enriched[f"drag_total_{solver_label}_plus_transition_outer_profile_n"] = (
            float(row["drag_n"]) + transition_outer_drag
        )
        enriched[f"ld_total_{solver_label}_plus_transition_outer_profile"] = (
            float(row["lift_n"])
            / enriched[f"drag_total_{solver_label}_plus_transition_outer_profile_n"]
            if abs(enriched[f"drag_total_{solver_label}_plus_transition_outer_profile_n"])
            > 1.0e-14
            else float("nan")
        )
        out.append(enriched)
    return out


def final_profile_drag_window_stats(
    history: list[dict[str, float]],
    *,
    solver_label: str,
    window: int,
) -> dict[str, float]:
    if not history:
        return {}
    tail = history[-max(1, int(window)) :]
    keys = [
        f"cd_total_{solver_label}_plus_outer_profile",
        f"drag_total_{solver_label}_plus_outer_profile_n",
        f"ld_total_{solver_label}_plus_outer_profile",
        f"cd_total_{solver_label}_plus_transition_outer_profile",
        f"drag_total_{solver_label}_plus_transition_outer_profile_n",
        f"ld_total_{solver_label}_plus_transition_outer_profile",
    ]
    out: dict[str, float] = {}
    for key in keys:
        values = np.asarray([row[key] for row in tail], dtype=float)
        out[f"history_final_window_{key}_mean"] = float(np.mean(values))
        out[f"history_final_window_{key}_std"] = float(np.std(values))
        out[f"history_final_window_{key}_min"] = float(np.min(values))
        out[f"history_final_window_{key}_max"] = float(np.max(values))
    return out


def append_csv(path: Path, row: dict[str, Any], fieldnames: list[str]) -> None:
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, extrasaction="ignore")
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def write_rows_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
