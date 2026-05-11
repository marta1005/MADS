"""Private spanwise interpolation helpers for synthesis geometry workflows."""

from __future__ import annotations

import numpy as np
from scipy.interpolate import PchipInterpolator


def _sorted_pairs(
    anchor_y: np.ndarray,
    values: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    order = np.argsort(anchor_y)
    return anchor_y[order], values[order]


def interpolate_scalar_field(
    anchor_y: np.ndarray,
    values: np.ndarray,
    sample_y: np.ndarray,
    *,
    law: str = "pchip",
) -> np.ndarray:
    """Interpolate a scalar field along span."""

    y, v = _sorted_pairs(np.asarray(anchor_y, dtype=float), np.asarray(values, dtype=float))
    if y.size == 1:
        return np.full_like(np.asarray(sample_y, dtype=float), float(v[0]), dtype=float)

    law_name = law.lower()
    if law_name == "linear":
        return np.interp(sample_y, y, v)
    if law_name == "pchip":
        return np.asarray(PchipInterpolator(y, v)(sample_y), dtype=float)

    msg = f"Unsupported spanwise law '{law}'."
    raise ValueError(msg)


def interpolate_vector_field(
    anchor_y: np.ndarray,
    values: np.ndarray,
    sample_y: np.ndarray,
    *,
    law: str = "pchip",
) -> np.ndarray:
    """Interpolate a vector-valued field component-wise along span."""

    anchor_y = np.asarray(anchor_y, dtype=float)
    values = np.asarray(values, dtype=float)
    sample_y = np.asarray(sample_y, dtype=float)

    if values.ndim != 2:
        msg = f"Expected a 2D array for vector interpolation, got shape {values.shape}."
        raise ValueError(msg)

    return np.column_stack(
        [
            interpolate_scalar_field(anchor_y, values[:, idx], sample_y, law=law)
            for idx in range(values.shape[1])
        ]
    )


def merge_sampled_stations(
    anchor_y: np.ndarray,
    requested_count: int,
    *,
    include_anchors: bool = True,
) -> np.ndarray:
    """Create the spanwise station sampling while optionally preserving anchors."""

    y = np.asarray(anchor_y, dtype=float)
    if y.ndim != 1 or y.size == 0:
        msg = "Anchor spanwise coordinates must be a non-empty 1D array."
        raise ValueError(msg)

    y = np.unique(np.sort(y))
    if requested_count <= y.size:
        return y

    sampled = np.linspace(float(y[0]), float(y[-1]), int(requested_count), dtype=float)
    if include_anchors:
        sampled = np.unique(np.concatenate((sampled, y)))
    return sampled
