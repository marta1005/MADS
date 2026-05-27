from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from scipy.interpolate import RegularGridInterpolator


def _resample_surface(
    surface: np.ndarray,
    span: np.ndarray,
    chord: np.ndarray,
    span_new: np.ndarray,
    chord_new: np.ndarray,
) -> np.ndarray:
    grid_span, grid_chord = np.meshgrid(span_new, chord_new, indexing="ij")
    sample_points = np.column_stack((grid_span.ravel(), grid_chord.ravel()))
    out = np.empty((len(span_new), len(chord_new), 3), dtype=float)
    for axis in range(3):
        interpolator = RegularGridInterpolator(
            (span, chord),
            surface[:, :, axis],
            bounds_error=False,
            fill_value=None,
        )
        out[:, :, axis] = interpolator(sample_points).reshape(
            len(span_new),
            len(chord_new),
        )
    return out


def _surface_mesh_diagnostics(rr: np.ndarray, ee: np.ndarray) -> dict[str, float]:
    quads = rr[ee - 1]
    area = 0.5 * np.linalg.norm(
        np.cross(quads[:, 1] - quads[:, 0], quads[:, 2] - quads[:, 0]),
        axis=1,
    ) + 0.5 * np.linalg.norm(
        np.cross(quads[:, 2] - quads[:, 0], quads[:, 3] - quads[:, 0]),
        axis=1,
    )
    edges = np.stack(
        (
            np.linalg.norm(quads[:, 1] - quads[:, 0], axis=1),
            np.linalg.norm(quads[:, 2] - quads[:, 1], axis=1),
            np.linalg.norm(quads[:, 3] - quads[:, 2], axis=1),
            np.linalg.norm(quads[:, 0] - quads[:, 3], axis=1),
        ),
    )
    return {
        "min_panel_area_m2": float(area.min()),
        "max_panel_area_m2": float(area.max()),
        "min_edge_length_m": float(edges.min()),
        "max_edge_length_m": float(edges.max()),
    }


def write_basic_two_skin_mesh_from_resolved_npz(
    mesh_npz: Path,
    prefix: Path,
    *,
    n_span_stations: int = 33,
    n_chord_stations: int = 33,
    leading_edge_opening_m: float = 0.05,
    leading_edge_opening_extent: float = 0.12,
    collapse_trailing_edge: bool = True,
) -> dict[str, Any]:
    """Write a DUST ``basic`` mesh from a resolved upper/lower surface mesh.

    The geometry framework may resolve high-fidelity skins for CAD/IGES and
    mesh-deformation workflows. DUST panel wakes are more robust with a solver
    mesh that has controlled spacing, a single trailing-edge line, and no
    coincident upper/lower leading-edge boundary. This adapter builds that
    aerodynamic mesh without modifying the source geometry.
    """

    data = np.load(mesh_npz)
    span = np.asarray(data["span_stations"], dtype=float)
    chord = np.asarray(data["x_airfoil"], dtype=float)
    upper = np.asarray(data["upper_vertices"], dtype=float)
    lower = np.asarray(data["lower_vertices"], dtype=float)

    span_new = np.linspace(float(span[0]), float(span[-1]), n_span_stations)
    chord_new = np.linspace(0.0, 1.0, n_chord_stations)
    upper_solver = _resample_surface(upper, span, chord, span_new, chord_new)
    lower_solver = _resample_surface(lower, span, chord, span_new, chord_new)

    if collapse_trailing_edge:
        te_mid = 0.5 * (upper_solver[:, -1, :] + lower_solver[:, -1, :])
        upper_solver[:, -1, :] = te_mid
        lower_solver[:, -1, :] = te_mid

    le_shape = np.clip(1.0 - chord_new / leading_edge_opening_extent, 0.0, 1.0) ** 2
    upper_solver[:, :, 2] += 0.5 * leading_edge_opening_m * le_shape[None, :]
    lower_solver[:, :, 2] -= 0.5 * leading_edge_opening_m * le_shape[None, :]

    sections = [
        np.vstack((upper_solver[i_span], lower_solver[i_span]))
        for i_span in range(n_span_stations)
    ]
    rr = np.vstack(sections)
    n_per_section = int(sections[0].shape[0])

    quads: list[tuple[int, int, int, int]] = []
    for i_span in range(n_span_stations - 1):
        cur = i_span * n_per_section
        nxt = (i_span + 1) * n_per_section
        for i_chord in range(n_chord_stations - 1):
            quads.append(
                (
                    nxt + i_chord + 1,
                    cur + i_chord + 1,
                    cur + i_chord + 2,
                    nxt + i_chord + 2,
                ),
            )
            lower_i = n_chord_stations + i_chord
            quads.append(
                (
                    cur + lower_i + 1,
                    nxt + lower_i + 1,
                    nxt + lower_i + 2,
                    cur + lower_i + 2,
                ),
            )

    ee = np.asarray(quads, dtype=int)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(prefix.with_name(prefix.name + "rr.dat"), rr, fmt="%.10e")
    np.savetxt(prefix.with_name(prefix.name + "ee.dat"), ee, fmt="%d")

    return {
        "n_sections": int(n_span_stations),
        "n_chord_stations": int(n_chord_stations),
        "n_points_per_section": int(n_per_section),
        "n_points": int(rr.shape[0]),
        "n_elements": int(ee.shape[0]),
        "leading_edge_opening_m": float(leading_edge_opening_m),
        "leading_edge_opening_extent": float(leading_edge_opening_extent),
        "trailing_edge_collapsed": bool(collapse_trailing_edge),
        **_surface_mesh_diagnostics(rr, ee),
    }
