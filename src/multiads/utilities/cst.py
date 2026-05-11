"""Shared CST utilities used by airfoil and geometry modules."""

from __future__ import annotations

from math import comb

import numpy as np
from numpy.typing import NDArray


def normalized_chordwise_points(
    npoints: int,
    distribution: str = "uniform",
) -> NDArray[np.float64]:
    """Return chordwise points in ``[0, 1]``."""

    if npoints < 2:
        msg = "At least two chordwise points are required."
        raise ValueError(msg)

    dist = distribution.lower()
    if dist in {"uniform", "eq"}:
        return np.linspace(0.0, 1.0, npoints, dtype=float)
    if dist in {"cosine", "le_te", "te_le"}:
        return 0.5 * (1.0 - np.cos(np.linspace(0.0, np.pi, npoints, dtype=float)))
    if dist == "leading_edge":
        return 1.0 - np.cos(np.linspace(0.0, np.pi / 2.0, npoints, dtype=float))
    if dist == "trailing_edge":
        return np.sin(np.linspace(0.0, np.pi / 2.0, npoints, dtype=float))

    msg = f"Unsupported chordwise distribution '{distribution}'."
    raise ValueError(msg)


def class_function(
    x_over_c: NDArray[np.float64],
    n1: float,
    n2: float,
) -> NDArray[np.float64]:
    """Evaluate the CST class function."""

    x = np.asarray(x_over_c, dtype=float)
    return np.power(x, n1) * np.power(1.0 - x, n2)


def bernstein_shape_function(
    x_over_c: NDArray[np.float64],
    coefficients: tuple[float, ...],
) -> NDArray[np.float64]:
    """Evaluate the Bernstein polynomial shape function."""

    x = np.asarray(x_over_c, dtype=float)
    if not coefficients:
        return np.zeros_like(x)

    order = len(coefficients) - 1
    out = np.zeros_like(x)
    for idx, coeff in enumerate(coefficients):
        out += coeff * comb(order, idx) * x**idx * (1.0 - x) ** (order - idx)
    return out


def evaluate_cst_surface(
    x_over_c: NDArray[np.float64],
    coefficients: tuple[float, ...],
    *,
    n1: float,
    n2: float,
    trailing_edge_thickness_over_c: float = 0.0,
    sign: float = 1.0,
) -> NDArray[np.float64]:
    """Evaluate one CST surface in normalized ``z/c`` coordinates."""

    x = np.asarray(x_over_c, dtype=float)
    cls = class_function(x, n1, n2)
    shp = bernstein_shape_function(x, coefficients)
    te_term = 0.5 * trailing_edge_thickness_over_c * x
    return sign * (cls * shp + te_term)
