"""Private section-level resolution utilities for synthesis geometry workflows."""


from __future__ import annotations

import numpy as np

from multiads.assembly import Section
from multiads.solvers.synthesis._geometry_types import ResolvedStation, WingGeometryConfig


def resolve_anchor_section(
    section: Section,
    config: WingGeometryConfig,
) -> ResolvedStation:
    """Resolve one anchor section into explicit normalized and dimensional points."""

    if section.airfoil is None:
        msg = f"Anchor section '{section.name}' does not define an airfoil."
        raise ValueError(msg)
    if section.spanwise_y_m is None:
        msg = f"Anchor section '{section.name}' does not define spanwise_y_m."
        raise ValueError(msg)

    chord_m = float(section.chord)
    twist_deg = float(section.twist)
    leading_edge_x_m = float(section.leading_edge_x_m or 0.0)
    leading_edge_z_m = float(section.leading_edge_z_m or 0.0)

    thickness_dist = section.airfoil.thickness_distribution(
        npoints=config.sampling.chordwise_points,
        distribution=config.sampling.station_distribution,
        chord=1.0,
    )
    camber_dist = section.airfoil.camber_distribution(
        npoints=config.sampling.chordwise_points,
        distribution=config.sampling.station_distribution,
        chord=1.0,
    )

    x_over_c = np.asarray(camber_dist[:, 0], dtype=float)
    thickness_over_c = np.asarray(thickness_dist[:, 1], dtype=float)
    camber_over_c = np.asarray(camber_dist[:, 1], dtype=float)
    upper_z_over_c = camber_over_c + 0.5 * thickness_over_c
    lower_z_over_c = camber_over_c - 0.5 * thickness_over_c

    upper_surface_xyz_m = _build_surface_xyz(
        x_over_c,
        upper_z_over_c,
        chord_m,
        section.spanwise_y_m,
        twist_deg,
        leading_edge_x_m,
        leading_edge_z_m,
    )
    lower_surface_xyz_m = _build_surface_xyz(
        x_over_c,
        lower_z_over_c,
        chord_m,
        section.spanwise_y_m,
        twist_deg,
        leading_edge_x_m,
        leading_edge_z_m,
    )

    return ResolvedStation(
        name=section.name,
        spanwise_y_m=float(section.spanwise_y_m),
        chord_m=chord_m,
        twist_deg=twist_deg,
        leading_edge_x_m=leading_edge_x_m,
        leading_edge_z_m=leading_edge_z_m,
        x_over_c=x_over_c,
        upper_z_over_c=upper_z_over_c,
        lower_z_over_c=lower_z_over_c,
        upper_surface_xyz_m=upper_surface_xyz_m,
        lower_surface_xyz_m=lower_surface_xyz_m,
        metadata=dict(section.metadata),
    )


def _build_surface_xyz(
    x_over_c: np.ndarray,
    z_over_c: np.ndarray,
    chord_m: float,
    spanwise_y_m: float,
    twist_deg: float,
    leading_edge_x_m: float,
    leading_edge_z_m: float,
    *,
    twist_reference_x_over_c: float = 0.25,
) -> np.ndarray:
    x_local = chord_m * np.asarray(x_over_c, dtype=float)
    z_local = chord_m * np.asarray(z_over_c, dtype=float)

    if twist_deg != 0.0:
        x_ref = twist_reference_x_over_c * chord_m
        theta = np.radians(twist_deg)
        c = np.cos(theta)
        s = np.sin(theta)
        x_shift = x_local - x_ref
        x_rot = c * x_shift + s * z_local
        z_rot = -s * x_shift + c * z_local
        x_local = x_rot + x_ref
        z_local = z_rot

    xyz = np.zeros((x_local.size, 3), dtype=float)
    xyz[:, 0] = leading_edge_x_m + x_local
    xyz[:, 1] = spanwise_y_m
    xyz[:, 2] = leading_edge_z_m + z_local
    return xyz
