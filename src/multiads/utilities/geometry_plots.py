"""Matplotlib-based geometry visualisation for resolved wing geometry.

No pyGeo or pyspline required — only numpy and matplotlib.

Public functions
----------------
plot_planform      -- top-down view (chord, sweep, taper)
plot_front_view    -- front view (span vs thickness envelope)
plot_sections      -- airfoil profiles at selected spanwise stations
plot_geometry      -- all three views in one figure
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Sequence

import numpy as np

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

    from multiads.solvers.synthesis.geometry_lib import PreparedGeometry, ResolvedStation


def plot_planform(
    geometry: PreparedGeometry,
    *,
    ax: Axes | None = None,
    symmetric: bool = True,
    color: str = "k",
    lw: float = 1.2,
) -> Axes:
    """Top-down planform view (y spanwise, x streamwise, nose at top).

    Parameters
    ----------
    geometry:
        Resolved geometry state returned by ``build_geometry``.
    ax:
        Existing axes to draw into. A new figure/axes is created if None.
    symmetric:
        Mirror the half-wing to show the full span.
    """
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(figsize=(10, 5))

    stations = geometry.resolved_stations
    y = np.array([s.spanwise_y_m for s in stations])
    le_x = np.array([s.leading_edge_x_m for s in stations])
    te_x = np.array([s.leading_edge_x_m + s.chord_m for s in stations])

    ax.plot(y, le_x, color=color, lw=lw)
    ax.plot(y, te_x, color=color, lw=lw)
    # tip close-off
    ax.plot([y[-1], y[-1]], [le_x[-1], te_x[-1]], color=color, lw=lw)
    # root close-off
    ax.plot([y[0], y[0]], [le_x[0], te_x[0]], color=color, lw=lw)

    if symmetric:
        ax.plot(-y, le_x, color=color, lw=lw)
        ax.plot(-y, te_x, color=color, lw=lw)
        ax.plot([-y[-1], -y[-1]], [le_x[-1], te_x[-1]], color=color, lw=lw)
        ax.plot([-y[0], -y[0]], [le_x[0], te_x[0]], color=color, lw=lw)

    ax.set_xlabel("y (m)")
    ax.set_ylabel("x (m)")
    ax.set_title("Planform (top view)")
    ax.invert_yaxis()
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)
    return ax


def plot_front_view(
    geometry: PreparedGeometry,
    *,
    ax: Axes | None = None,
    symmetric: bool = True,
    color: str = "k",
    lw: float = 1.2,
) -> Axes:
    """Front view: span (y) vs vertical extent of the cross-section.

    Shows the upper and lower surface z-envelope across the span.
    """
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(figsize=(10, 4))

    stations = geometry.resolved_stations
    y = np.array([s.spanwise_y_m for s in stations])
    z_upper = np.array([np.max(s.upper_surface_xyz_m[:, 2]) for s in stations])
    z_lower = np.array([np.min(s.lower_surface_xyz_m[:, 2]) for s in stations])

    ax.plot(y, z_upper, color=color, lw=lw, label="upper")
    ax.plot(y, z_lower, color=color, lw=lw, linestyle="--", label="lower")
    ax.fill_between(y, z_lower, z_upper, alpha=0.08, color=color)

    if symmetric:
        ax.plot(-y, z_upper, color=color, lw=lw)
        ax.plot(-y, z_lower, color=color, lw=lw, linestyle="--")
        ax.fill_between(-y, z_lower, z_upper, alpha=0.08, color=color)

    ax.set_xlabel("y (m)")
    ax.set_ylabel("z (m)")
    ax.set_title("Front view")
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)
    return ax


def plot_sections(
    geometry: PreparedGeometry,
    *,
    stations: Sequence[int] | int | None = None,
    ax: Axes | None = None,
    cmap_name: str = "viridis",
    lw: float = 1.0,
    label_stations: bool = True,
) -> Axes:
    """Airfoil profile overlay for selected spanwise stations.

    Parameters
    ----------
    stations:
        Indices into ``geometry.resolved_stations`` to plot.
        Pass an integer N to plot N evenly spaced stations.
        Defaults to 7 evenly spaced stations.
    """
    import matplotlib.pyplot as plt
    from matplotlib import colormaps

    if ax is None:
        _, ax = plt.subplots(figsize=(10, 4))

    all_stations = geometry.resolved_stations
    n_total = len(all_stations)

    if stations is None:
        idx = np.linspace(0, n_total - 1, min(7, n_total), dtype=int)
    elif isinstance(stations, int):
        idx = np.linspace(0, n_total - 1, stations, dtype=int)
    else:
        idx = np.asarray(stations, dtype=int)

    cmap = colormaps[cmap_name]
    colors = [cmap(i / max(len(idx) - 1, 1)) for i in range(len(idx))]

    for color, i in zip(colors, idx):
        s: ResolvedStation = all_stations[int(i)]
        ax.plot(s.x_over_c, s.upper_z_over_c, color=color, lw=lw)
        ax.plot(s.x_over_c, s.lower_z_over_c, color=color, lw=lw)
        if label_stations:
            ax.annotate(
                f"y={s.spanwise_y_m:.1f}m",
                xy=(s.x_over_c[-1], (s.upper_z_over_c[-1] + s.lower_z_over_c[-1]) / 2),
                fontsize=7,
                color=color,
                va="center",
            )

    ax.set_xlabel("x/c")
    ax.set_ylabel("z/c")
    ax.set_title("Section profiles")
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)
    return ax


def plot_geometry(
    geometry: PreparedGeometry,
    *,
    symmetric: bool = True,
    stations: Sequence[int] | int | None = None,
    figsize: tuple[float, float] = (14, 11),
) -> Figure:
    """Composite figure with planform, front view and section profiles.

    Returns the :class:`matplotlib.figure.Figure` so the caller can
    save or further customise it.
    """
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=figsize)
    fig.suptitle(
        f"{geometry.case_name or geometry.component_name} — resolved geometry",
        fontsize=12,
    )

    plot_planform(geometry, ax=axes[0, 0], symmetric=symmetric)
    plot_front_view(geometry, ax=axes[0, 1], symmetric=symmetric)
    plot_sections(geometry, ax=axes[1, :].reshape(-1)[0], stations=stations)

    # distribution plots in the last cell
    ax_dist = axes[1, 1]
    _plot_distributions(geometry, ax=ax_dist)

    fig.tight_layout()
    return fig


def _plot_distributions(geometry: PreparedGeometry, ax: Axes) -> None:
    stations = geometry.resolved_stations
    y = np.array([s.spanwise_y_m for s in stations])
    chord = np.array([s.chord_m for s in stations])
    twist = np.array([s.twist_deg for s in stations])
    tc = np.array([
        np.max(s.upper_z_over_c) - np.min(s.lower_z_over_c)
        for s in stations
    ])

    ax2 = ax.twinx()
    l1, = ax.plot(y, chord, "b-", lw=1.2, label="chord (m)")
    l2, = ax.plot(y, tc * 100, "g--", lw=1.2, label="t/c (%)")
    l3, = ax2.plot(y, twist, "r:", lw=1.2, label="twist (deg)")

    ax.set_xlabel("y (m)")
    ax.set_ylabel("chord (m) / t/c (%)")
    ax2.set_ylabel("twist (deg)")
    ax.set_title("Distributions")
    ax.legend(handles=[l1, l2, l3], fontsize=8)
    ax.grid(True, alpha=0.3)
