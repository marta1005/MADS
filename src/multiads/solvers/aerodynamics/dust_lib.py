from __future__ import annotations

import os
import subprocess as sp
from abc import ABC, abstractmethod
from collections.abc import Callable
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any, TextIO

import numpy as np
from scipy.interpolate import RegularGridInterpolator
from typing_extensions import Self

from multiads import assembly
from multiads.scenario.span_loads import SPANLOAD_DEFAULT_NUM_STATIONS
from multiads.solvers import SolverOptions
from multiads.solvers.aerodynamics import SectionOptions

if TYPE_CHECKING:
    from collections.abc import Sequence

    from numpy.typing import NDArray

    from multiads.scenario.polars import PolarVariable


def _log_cmd_output(
    file_name: Path,
    cmd: sp.CompletedProcess,
    name: str,
    mode: str = "w",
) -> None:
    with file_name.open(mode) as f:
        for line in cmd.stdout.splitlines():
            f.write(f"INFO: {line}\n")
        for line in cmd.stderr.splitlines():
            f.write(f"ERROR: {line}\n")
        if cmd.returncode != 0:
            msg = f"Could not run '{name}'."
            raise RuntimeError(msg)


def write_polar_block(
    file: TextIO,
    mach: NDArray[np.float64],
    aoa: NDArray[np.float64],
    data: NDArray[np.float64],
) -> None:
    file.write(12 * " " + "    ".join([f"{m:<7.3f}" for m in mach]) + "\n")
    for i, ia in enumerate(aoa):
        file.write(f"{ia:>8.1f}   ")
        file.write("   ".join(f"{cl:< 8.5f}" for cl in data[i, :]) + "\n")


def write_polar(polar_data: PolarVariable, file: Path) -> None:
    with file.open("w") as f:
        # File header
        f.write(f"{polar_data.num_polars} 0 0\n")
        f.write("0 0\n")
        f.write("0.158 0.158\n")

        for i in range(polar_data.num_polars):
            # Extract polar data
            mach = np.array([0.0, 2.0])
            re = polar_data.reynolds[i]
            aoa = polar_data.aoa[i, :]
            cl = polar_data.cl[i, :]
            cd = polar_data.cd[i, :]
            cm = polar_data.cm[i, :]

            # Polar header
            f.write("COMMENT#1\n")
            f.write(f"{re:12.1f}    0.200\n")
            f.write("DUMMY" + 25 * " " + 3 * f"{len(mach):02d}{len(aoa):02d}" + "\n")

            # Polar data
            write_polar_block(f, mach, aoa, np.stack([cl, cl], axis=1))
            write_polar_block(f, mach, aoa, np.stack([cd, cd], axis=1))
            write_polar_block(f, mach, aoa, np.stack([cm, cm], axis=1))


def _cumulative_trapezoid(values: np.ndarray, coordinates: np.ndarray) -> np.ndarray:
    area = 0.5 * (values[1:] + values[:-1]) * np.diff(coordinates)
    return np.concatenate(([0.0], np.cumsum(area)))


def _normalize_metric(metric: np.ndarray) -> np.ndarray:
    clean = np.asarray(metric, dtype=float)
    clean = np.where(np.isfinite(clean), np.maximum(clean, 0.0), 0.0)
    if clean.size == 0 or float(np.max(clean)) <= 1.0e-15:
        return np.zeros_like(clean)
    cap = float(np.percentile(clean, 95.0))
    if cap <= 1.0e-15:
        cap = float(np.max(clean))
    clean = np.clip(clean, 0.0, cap)
    return clean / max(float(np.max(clean)), 1.0e-15)


def _curve_curvature_metric(axis: np.ndarray, coordinates: np.ndarray) -> np.ndarray:
    axis = np.asarray(axis, dtype=float)
    coordinates = np.asarray(coordinates, dtype=float)
    if axis.size < 3:
        return np.zeros_like(axis)
    first = np.gradient(coordinates, axis, axis=0, edge_order=1)
    second = np.gradient(first, axis, axis=0, edge_order=1)
    scale = np.linalg.norm(first, axis=1)
    return np.linalg.norm(second, axis=1) / np.maximum(scale * scale, 1.0e-12)


def _adaptive_axis_from_density(
    axis: np.ndarray,
    density: np.ndarray,
    count: int,
) -> np.ndarray:
    axis = np.asarray(axis, dtype=float)
    density = np.asarray(density, dtype=float)
    if int(count) < 2:
        raise ValueError(f"adaptive axis requires at least 2 stations, got {count}")
    if axis.ndim != 1 or density.shape != axis.shape:
        raise ValueError("axis and density must be 1D arrays with the same shape")
    if np.any(np.diff(axis) <= 0.0):
        raise ValueError("adaptive axis source coordinates must be strictly increasing")

    density = np.maximum(density, 1.0e-6)
    cumulative = _cumulative_trapezoid(density, axis)
    if cumulative[-1] <= 1.0e-15:
        return np.linspace(float(axis[0]), float(axis[-1]), int(count))
    targets = np.linspace(0.0, float(cumulative[-1]), int(count))
    out = np.interp(targets, cumulative, axis)
    out[0] = float(axis[0])
    out[-1] = float(axis[-1])
    return out


def _span_curvature_density(
    upper: np.ndarray,
    lower: np.ndarray,
    span: np.ndarray,
    weight: float,
) -> np.ndarray:
    le = 0.5 * (upper[:, 0, :] + lower[:, 0, :])
    te = 0.5 * (upper[:, -1, :] + lower[:, -1, :])
    mid = 0.5 * (
        upper[:, upper.shape[1] // 2, :] + lower[:, lower.shape[1] // 2, :]
    )
    upper_env = np.column_stack((span, np.max(upper[:, :, 2], axis=1)))
    lower_env = np.column_stack((span, np.min(lower[:, :, 2], axis=1)))
    metric = (
        _curve_curvature_metric(span, le[:, (0, 2)])
        + _curve_curvature_metric(span, te[:, (0, 2)])
        + 0.5 * _curve_curvature_metric(span, mid[:, (0, 2)])
        + 0.5 * _curve_curvature_metric(span, upper_env)
        + 0.5 * _curve_curvature_metric(span, lower_env)
    )
    return 1.0 + float(weight) * _normalize_metric(metric)


def _chord_curvature_density(
    upper: np.ndarray,
    lower: np.ndarray,
    chord: np.ndarray,
    weight: float,
    endpoint_weight: float,
) -> np.ndarray:
    metric = np.zeros_like(chord, dtype=float)
    for i_span in range(upper.shape[0]):
        metric += _curve_curvature_metric(chord, upper[i_span])
        metric += _curve_curvature_metric(chord, lower[i_span])
    metric /= max(2 * upper.shape[0], 1)

    endpoint = (
        np.exp(-0.5 * (chord / 0.055) ** 2)
        + 0.55 * np.exp(-0.5 * ((1.0 - chord) / 0.075) ** 2)
    )
    metric = _normalize_metric(metric) + float(endpoint_weight) * _normalize_metric(endpoint)
    return 1.0 + float(weight) * _normalize_metric(metric)


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
    span_spacing: str = "uniform",
    chord_spacing: str = "uniform",
    span_curvature_weight: float = 4.0,
    chord_curvature_weight: float = 3.0,
    chord_endpoint_weight: float = 0.60,
    leading_edge_opening_m: float = 0.0,
    leading_edge_opening_chord_fraction: float | None = None,
    leading_edge_opening_extent: float = 0.12,
    collapse_trailing_edge: bool = True,
    mirror_span: bool = False,
) -> dict[str, Any]:
    """Write a DUST ``basic`` mesh from a resolved upper/lower surface mesh.

    The synthesis geometry framework resolves CAD-ready upper/lower skins. DUST
    panel wakes are more robust with a dedicated solver mesh that controls
    spacing and has a single trailing-edge line. The default keeps the leading
    edge closed; a small numerical opening can be requested explicitly.
    """

    data = np.load(mesh_npz)
    span = np.asarray(data["span_stations"], dtype=float)
    chord = np.asarray(data["x_airfoil"], dtype=float)
    upper = np.asarray(data["upper_vertices"], dtype=float)
    lower = np.asarray(data["lower_vertices"], dtype=float)

    if span_spacing == "uniform":
        span_new = np.linspace(float(span[0]), float(span[-1]), n_span_stations)
    elif span_spacing == "curvature":
        span_density = _span_curvature_density(
            upper,
            lower,
            span,
            weight=span_curvature_weight,
        )
        span_new = _adaptive_axis_from_density(span, span_density, n_span_stations)
    else:
        raise ValueError("span_spacing must be 'uniform' or 'curvature'")

    if chord_spacing == "uniform":
        chord_new = np.linspace(0.0, 1.0, n_chord_stations)
    elif chord_spacing == "curvature":
        chord_density = _chord_curvature_density(
            upper,
            lower,
            chord,
            weight=chord_curvature_weight,
            endpoint_weight=chord_endpoint_weight,
        )
        chord_new = _adaptive_axis_from_density(chord, chord_density, n_chord_stations)
    else:
        raise ValueError("chord_spacing must be 'uniform' or 'curvature'")

    upper_solver = _resample_surface(upper, span, chord, span_new, chord_new)
    lower_solver = _resample_surface(lower, span, chord, span_new, chord_new)

    if collapse_trailing_edge:
        te_mid = 0.5 * (upper_solver[:, -1, :] + lower_solver[:, -1, :])
        upper_solver[:, -1, :] = te_mid
        lower_solver[:, -1, :] = te_mid

    le_shape = np.clip(1.0 - chord_new / leading_edge_opening_extent, 0.0, 1.0) ** 2
    if leading_edge_opening_chord_fraction is None:
        opening = np.full(n_span_stations, float(leading_edge_opening_m), dtype=float)
    else:
        le_mid = 0.5 * (upper_solver[:, 0, :] + lower_solver[:, 0, :])
        te_mid = 0.5 * (upper_solver[:, -1, :] + lower_solver[:, -1, :])
        local_chord = np.linalg.norm(te_mid - le_mid, axis=1)
        opening = np.minimum(
            float(leading_edge_opening_m),
            float(leading_edge_opening_chord_fraction) * local_chord,
        )
    upper_solver[:, :, 2] += 0.5 * opening[:, None] * le_shape[None, :]
    lower_solver[:, :, 2] -= 0.5 * opening[:, None] * le_shape[None, :]

    if mirror_span:
        mirror_slice = slice(1, None)
        upper_mirror = upper_solver[mirror_slice][::-1].copy()
        lower_mirror = lower_solver[mirror_slice][::-1].copy()
        upper_mirror[:, :, 1] *= -1.0
        lower_mirror[:, :, 1] *= -1.0
        upper_solver = np.concatenate((upper_mirror, upper_solver), axis=0)
        lower_solver = np.concatenate((lower_mirror, lower_solver), axis=0)
        span_new = np.concatenate((-span_new[mirror_slice][::-1], span_new))
        opening = np.concatenate((opening[mirror_slice][::-1], opening))

    n_solver_span = int(upper_solver.shape[0])
    sections = [
        np.vstack((upper_solver[i_span], lower_solver[i_span]))
        for i_span in range(n_solver_span)
    ]
    rr = np.vstack(sections)
    n_per_section = int(sections[0].shape[0])

    quads: list[tuple[int, int, int, int]] = []
    for i_span in range(n_solver_span - 1):
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
        "n_sections": int(n_solver_span),
        "n_half_span_sections": int(n_span_stations),
        "n_chord_stations": int(n_chord_stations),
        "n_points_per_section": int(n_per_section),
        "n_points": int(rr.shape[0]),
        "n_elements": int(ee.shape[0]),
        "span_spacing": span_spacing,
        "chord_spacing": chord_spacing,
        "min_span_spacing_m": float(np.min(np.diff(span_new))),
        "max_span_spacing_m": float(np.max(np.diff(span_new))),
        "min_chord_spacing": float(np.min(np.diff(chord_new))),
        "max_chord_spacing": float(np.max(np.diff(chord_new))),
        "span_curvature_weight": float(span_curvature_weight),
        "chord_curvature_weight": float(chord_curvature_weight),
        "leading_edge_opening_m": float(leading_edge_opening_m),
        "leading_edge_opening_chord_fraction": (
            None
            if leading_edge_opening_chord_fraction is None
            else float(leading_edge_opening_chord_fraction)
        ),
        "min_effective_leading_edge_opening_m": float(np.min(opening)),
        "max_effective_leading_edge_opening_m": float(np.max(opening)),
        "leading_edge_opening_extent": float(leading_edge_opening_extent),
        "trailing_edge_collapsed": bool(collapse_trailing_edge),
        "span_mirrored": bool(mirror_span),
        **_surface_mesh_diagnostics(rr, ee),
    }


class Options(SolverOptions):
    def __init__(
        self,
        *,
        name: str = "dust",
        # DUST commands
        dust_pre: str | Path = "dust_pre",
        dust: str | Path = "dust",
        dust_post: str | Path = "dust_post",
        n_threads: int = 1,
        work_dir: str | Path = ".",
        run_directory: str | Path | None = None,
        output_dir: str | Path = "output",
        post_dir: str | Path = "post",
        keep_run_directory: bool = False,
        # Global postprocess options
        output_options: OutputOptions | None = None,
        # Solver options
        t_start: float = 0.0,
        t_end: float | None = None,
        dt: float | None = None,
        dt_out: float | None = None,
        output_start: bool = False,
        post_preprocess_hook: Callable[[Path], None] | None = None,
        n_turns: float = 4,
        steps_per_turn: int = 40,
        n_wake_panels: int = 1,
        n_wake_particles: int = 10000,
        particles_box_min: NDArray[np.float64] | None = None,
        particles_box_max: NDArray[np.float64] | None = None,
        penetration_avoidance: bool = False,
        fmm: bool = True,
        box_length: float | None = None,
        n_box: NDArray[np.int32] | None = None,
        octree_origin: NDArray[np.float64] | None = None,
        min_octree_divisions: int = 2,
        n_octree_levels: int = 6,
        min_octree_part: int = 7,
        multipole_degree: int = 2,
        max_iter: int = 100,
        tol: float = 1e-6,
        # Moving root reference frame
        moving: bool = False,
        rotation_dir: NDArray[np.float64] | None = None,
        rotation_ampl: float = 1.0,
        rotation_omega: float = 1.0,
        rotation_phase: float = 0.0,
        pole_ampl: float = 1.0,
        pole_weights: NDArray[np.float64] | None = None,
        pole_omega: NDArray[np.float64] | None = None,
        pole_phase: NDArray[np.float64] | None = None,
        **kwargs: Any,  # noqa: ANN401
    ) -> None:
        super().__init__(**kwargs)
        self.name = name
        self.dust_pre = dust_pre
        self.dust = dust
        self.dust_post = dust_post
        self.n_threads = n_threads

        self.work_dir = Path(work_dir)
        self.run_directory = None if run_directory is None else Path(run_directory)
        self.output_dir = Path(output_dir)
        self.post_dir = Path(post_dir)
        self.keep_run_directory = keep_run_directory

        self.output_options = output_options or OutputOptions()

        self.t_start = t_start
        self.t_end = t_end
        self.dt = dt
        self.dt_out = dt_out
        self.output_start = output_start
        self.post_preprocess_hook = post_preprocess_hook
        self.n_turns = n_turns
        self.steps_per_turn = steps_per_turn
        self.n_wake_panels = n_wake_panels
        self.n_wake_particles = n_wake_particles
        self.penetration_avoidance = penetration_avoidance
        self.fmm = fmm
        self.min_octree_divisions = min_octree_divisions
        self.n_octree_levels = n_octree_levels
        self.min_octree_part = min_octree_part
        self.multipole_degree = multipole_degree
        self.max_iter = max_iter
        self.tol = tol

        self.particles_box_min = (
            particles_box_min if particles_box_min is not None else np.zeros(3)
        )
        self.particles_box_max = (
            particles_box_max if particles_box_max is not None else np.ones(3)
        )

        if box_length is None or n_box is None or octree_origin is None:
            self.box_length, self.octree_origin, self.n_box = (
                self._set_octree_parameters(
                    self.fmm,
                    self.min_octree_divisions,
                    self.particles_box_min,
                    self.particles_box_max,
                )
            )
        else:
            self.box_length = box_length
            self.n_box = n_box if n_box is not None else np.ones(3, dtype=np.int32)
            self.octree_origin = (
                octree_origin if octree_origin is not None else np.zeros(3)
            )

        self.moving = moving
        self.pole_ampl = pole_ampl
        self.rotation_ampl = rotation_ampl
        self.rotation_omega = rotation_omega
        self.rotation_phase = rotation_phase

        self.pole_weights = pole_weights if pole_weights is not None else np.ones(3)
        self.pole_omega = pole_omega if pole_omega is not None else np.ones(3)
        self.pole_phase = pole_phase if pole_phase is not None else np.zeros(3)
        self.rotation_dir = (
            rotation_dir if rotation_dir is not None else np.array([1.0, 0.0, 0.0])
        )

    def _set_octree_parameters(
        self,
        fmm: bool,
        min_n: int,
        box_min: NDArray[np.float64],
        box_max: NDArray[np.float64],
    ) -> tuple[float, NDArray[np.float64], NDArray[np.int_]]:
        # Nothing to do if FMM is not used
        if not fmm:
            return 0.0, np.zeros(3), np.zeros(3, np.int_)

        # Sort box dimensions by length
        dbox = np.array(
            [bmax - bmin for bmin, bmax in zip(box_min, box_max, strict=True)],
        )
        inds = np.argsort(dbox)
        dbox_sorted = dbox[inds]

        # Use at least `min_n` divisions in each axis
        n = [min_n, 0, 0]
        octree_margins = 0.1

        def ceil_th(x: float, th: float) -> int:
            ix = int(x)
            return ix if (x % ix < th) else ix + 1

        dx0, dx1, dx2 = dbox_sorted
        box_length = dx0 / n[0]

        n[1] = ceil_th(dx1 / box_length, octree_margins)
        n[2] = ceil_th(dx2 / box_length, octree_margins)

        # Add margins if needed
        for ni, dxi in zip(n, dbox_sorted, strict=True):
            margin = max(octree_margins, ni * box_length - dxi)
            length = (dxi + margin) / ni
            box_length = max(box_length, length)

        # Final octree values
        n_box = np.empty(3, dtype=int)
        n_box[inds] = n
        octree_origin = np.array(
            [
                bmin - (ni * box_length - dxi) / 2.0
                for bmin, ni, dxi in zip(box_min, n_box, dbox, strict=True)
            ],
        )

        return box_length, octree_origin, n_box


class OutputOptions:
    def __init__(
        self,
        *,
        compute_loads: bool = False,
        compute_spanwise: bool = False,
        visualization: bool = False,
        loads_start: int = 0,
        loads_step: int = 1,
        loads_end: int = 0,
        loads_avg: bool = False,
        loads_reference: str = "0",
        spanwise_start: int = 0,
        spanwise_step: int = 1,
        spanwise_end: int = 0,
        spanwise_avg: bool = False,
        spanwise_axis_node: NDArray[np.float64] | None = None,
        spanwise_axis_dir: NDArray[np.float64] | None = None,
        spanwise_size: int = SPANLOAD_DEFAULT_NUM_STATIONS,
        viz_start: int = 0,
        viz_step: int = 1,
        viz_end: int = 0,
        viz_fmt: str = "vtk",
        viz_wake: bool = True,
        viz_separate_wake: bool = True,
        viz_avg: bool = False,
        viz_variables: Sequence[str] | None = None,
    ) -> None:
        axis_node = np.zeros(3) if spanwise_axis_node is None else spanwise_axis_node
        axis_dir = (
            np.array([0.0, 1.0, 0.0])
            if spanwise_axis_dir is None
            else spanwise_axis_dir
        )

        self.compute_loads = compute_loads
        self.compute_spanwise = compute_spanwise
        self.visualization = visualization
        self.loads_start = loads_start
        self.loads_step = loads_step
        self.loads_end = loads_end
        self.loads_avg = loads_avg
        self.loads_reference = loads_reference
        self.spanwise_start = spanwise_start
        self.spanwise_step = spanwise_step
        self.spanwise_end = spanwise_end
        self.spanwise_avg = spanwise_avg
        self.spanwise_axis_node = axis_node
        self.spanwise_axis_dir = axis_dir
        self.spanwise_size = spanwise_size
        self.viz_start = viz_start
        self.viz_step = viz_step
        self.viz_end = viz_end
        self.viz_fmt = viz_fmt
        self.viz_wake = viz_wake
        self.viz_separate_wake = viz_separate_wake
        self.viz_avg = viz_avg
        self.viz_variables = [] if viz_variables is None else viz_variables


class Section:
    def __init__(
        self,
        name: str,
        airfoil: str | Path,
        chord: float,
        twist: float,
        polar: bool,
        polar_length: int,
    ) -> None:
        self.name = name
        self.airfoil = airfoil
        self.chord = chord
        self.twist = twist
        self.polar = polar
        self.polar_length = polar_length

        base = Path(airfoil).stem
        ext = Path(airfoil).suffix
        if ext:
            self.airfoil = Path(airfoil).resolve()
        elif base.lower()[:4] == "naca":
            self.airfoil = base.upper()

    @classmethod
    def from_component(cls, comp: assembly.Section) -> Self:
        opts = next(
            (o for o in comp.options if type(o) is SectionOptions),
            SectionOptions(),
        )
        airfoil = comp.airfoil.airfoil_name
        return cls(
            name=comp.name,
            airfoil=airfoil,
            chord=comp.chord,
            twist=comp.twist,
            polar=opts.polar,
            polar_length=opts.polar_length,
        )

    def update(self, comp: assembly.Section) -> None:
        self.airfoil = comp.airfoil.airfoil_name
        self.chord = comp.chord
        self.twist = comp.twist


class SpanPanelType(Enum):
    UNIFORM = "uniform"
    COSINE = "cosine"
    COSINE_IB = "cosineIB"
    COSINE_OB = "cosineOB"
    EQUALAREA = "equalarea"


class SpanOptions(assembly.ComponentOptions):
    def __init__(
        self,
        panel_type: SpanPanelType,
        num_panels: int | None = None,
        panel_density: int | None = None,
    ) -> None:
        self.panel_type = panel_type
        self.num_panels = num_panels
        self.panel_density = panel_density


class Span:
    def __init__(
        self,
        length: float,
        sweep: float,
        dihed: float,
        panel_type: SpanPanelType,
        num_panels: int,
    ) -> None:
        self.length = length
        self.sweep = sweep
        self.dihed = dihed
        self.panel_type = panel_type
        self.num_panels = num_panels

    @classmethod
    def from_component(cls, comp: assembly.Span) -> Self:
        if opts := next((o for o in comp.options if type(o) is SpanOptions), None):
            opts.num_panels = cls._num_panels_from_density(opts, comp.length)
        else:
            msg = f"No DUST options in component '{comp.name}'."
            raise ValueError(msg)

        return cls(
            comp.length,
            comp.sweep,
            comp.dihed,
            opts.panel_type,
            opts.num_panels,
        )

    def update(self, comp: assembly.Span) -> None:
        if opts := next((o for o in comp.options if type(o) is SpanOptions), None):
            self.num_panels = self._num_panels_from_density(opts, comp.length)

        self.length = comp.length
        self.sweep = comp.sweep
        self.dihed = comp.dihed

    @classmethod
    def _num_panels_from_density(cls, opts: SpanOptions, length: float) -> int:
        if num_panels := opts.num_panels:
            return num_panels
        if density := opts.panel_density:
            return int(max(1, np.ceil(length * density)))
        msg = f"No 'num_panels' nor 'panel_density' in {cls.__name__}'."
        raise ValueError(msg)
        
class MovableSurface:
    def __init__(
        self,
        length: float,
        pos_start: list[float],
        pos_end: list[float],
        ampl: float,
        derivative: bool = False,
        dDelta: float = 0.0,
    ) -> None:
        self.length = length
        self.pos_start = pos_start
        self.pos_end = pos_end
        self.ampl = ampl
        self.derivative = derivative
        self.dDelta = dDelta


    @classmethod
    def from_component(cls, comp: assembly.MovableSurface) -> Self:
        try:
            return cls(
                length=comp.length,
                pos_start=comp.pos_start,
                pos_end=comp.pos_end,
                ampl=comp.ampl,
                derivative=comp.derivative,
                dDelta=comp.dDelta,
            )

        except StopIteration:
            msg = f"No DUST options in component '{comp.name}'."
            raise ValueError(msg) from None

    def _write_hinge(
        self,
        out: list[str],
        wing_name: str,
        section_twist: float,
    ) -> None:
        lam = np.radians(self.sweep)
        phi = np.radians(self.dihed)
        x1, y1, z1 = self.pos

        if y1 < 0:
            lam = -lam
            phi = -phi
            y2 = y1 - self.length
            x2 = x1 - self.length * (np.sin(lam) / np.cos(lam))
            z2 = z1 - self.length * (np.sin(phi) / (np.cos(phi) * np.cos(lam)))
        else:
            y2 = y1 + self.length
            x2 = x1 + self.length * (np.sin(lam) / np.cos(lam))
            z2 = z1 + self.length * (np.sin(phi) / (np.cos(phi) * np.cos(lam)))

        out.extend(
            [
                "hinge = {\n",
                f"    hinge_tag = {wing_name}_movable_surface\n",
                "    hinge_nodes_input = parametric\n",
            ],
        )

        if y1 < 0:
            out.extend(
                [
                    f"    node_2 = (/ {x1:.6f}, {y1:.6f}, {z1:.6f} /)\n",
                    f"    node_1 = (/ {x2:.6f}, {y2:.6f}, {z2:.6f} /)\n",
                ],
            )
        else:
            out.extend(
                [
                    f"    node_1 = (/ {x1:.6f}, {y1:.6f}, {z1:.6f} /)\n",
                    f"    node_2 = (/ {x2:.6f}, {y2:.6f}, {z2:.6f} /)\n",
                ],
            )

        out.extend(
            [
                "    n_nodes = 2\n",
                "    hinge_ref_dir = (/ 1.0, 0.0, 0.0 /)\n",
                "    hinge_offset = 0.1\n",
                "    hinge_spanwise_blending = 0.01\n",
                "    hinge_adaptive_mesh = F\n",
                "    hinge_rotation_input = function:const\n",
                "    hinge_rotation_function = {\n",
                f"    amplitude = {self.ampl:.6f}\n",
                "    omega     = 0.0\n",
                "    phase     = 0.0\n",
                "    }\n",
                "}\n",
            ],
        )


class WingPanelType(Enum):
    UNIFORM = "uniform"
    COSINE = "cosine"
    COSINE_LE = "cosineLE"
    COSINE_TE = "cosineTE"


class WingMethod(Enum):
    LIFTING_LINE = "l"
    VORTEX_LATTICE = "v"
    PANELS = "p"


class WingOptions(assembly.ComponentOptions):
    def __init__(
        self,
        discretization_method: WingMethod,
        panel_type: WingPanelType | None = None,
        num_panels: int = 0,
        mesh_file: Path | None = None,
        mesh_file_type: str | None = None,
        mesh_definition: Sequence[str] | None = None,
        inner_product_te: float | None = None,
        tol_se_wing: float | None = None,
        proj_te: bool | None = None,
        proj_te_dir: str | None = None,
        proj_te_vector: Sequence[float] | None = None,
        output_options: OutputOptions | None = None,
    ) -> None:
        self.method = discretization_method
        self.panel_type = panel_type
        self.num_panels = num_panels
        self.mesh_file = mesh_file
        self.mesh_file_type = mesh_file_type
        self.mesh_definition = tuple(mesh_definition or ())
        self.inner_product_te = inner_product_te
        self.tol_se_wing = tol_se_wing
        self.proj_te = proj_te
        self.proj_te_dir = proj_te_dir
        self.proj_te_vector = (
            None if proj_te_vector is None else np.asarray(proj_te_vector, dtype=float)
        )
        self.output_opts = output_options or OutputOptions()
        self._check_args()

    def _check_args(self) -> None:
        if self.method is not WingMethod.LIFTING_LINE and (
            self.panel_type is None or self.num_panels <= 0
        ):
            msg = (
                f"'{type(self).__name__}' cannot have empty 'panely_type' and "
                "'num_panels' if 'discretization_method' is not lifting line."
            )
            raise ValueError(msg)


class Wing:
    def __init__(
        self,
        name: str,
        sections: Sequence[Section],
        spans: Sequence[Span],
        movable_surfaces: Sequence[MovableSurface] | None = None,
        method: WingMethod | None = None,
        panel_type: WingPanelType | None = None,
        num_panels: int = 100.0,
        mesh_file: Path | None = None,
        mesh_file_type: str | None = None,
        mesh_definition: Sequence[str] | None = None,
        pos: NDArray[np.float64] | None = None,
        offset: NDArray[np.float64] | None = None,
        scaling: float = 1.0,
        alpha: float = 0.0,
        beta: float = 0.0,
        roll: float = 0.0,
        xc_ref: float = 0.25,
        inner_product_te: float | None = None,
        tol_se_wing: float | None = None,
        proj_te: bool | None = None,
        proj_te_dir: str | None = None,
        proj_te_vector: Sequence[float] | None = None,
        symmetry: bool = False,
        mirror: bool = False,
        options: OutputOptions | None = None,
    ) -> None:
        self.name = name
        self.sections = sections
        self.spans = spans
        self.movable_surfaces = movable_surfaces
        self.method = method
        self.panel_type = panel_type
        self.num_panels = num_panels
        self.mesh_file = mesh_file
        self.mesh_file_type = mesh_file_type
        self.mesh_definition = tuple(mesh_definition or ())
        self.scaling = scaling
        self.alpha = alpha
        self.beta = beta
        self.roll = roll
        self.xc_ref = xc_ref
        self.inner_product_te = inner_product_te
        self.tol_se_wing = tol_se_wing
        self.proj_te = proj_te
        self.proj_te_dir = proj_te_dir
        self.proj_te_vector = (
            None if proj_te_vector is None else np.asarray(proj_te_vector, dtype=float)
        )
        self.symmetry = symmetry
        self.mirror = mirror
        self.options = options or OutputOptions()

        self.pos = pos if pos is not None else np.zeros(3)
        self.offset = offset if offset is not None else np.zeros(3)

    @classmethod
    def from_component(cls, comp: assembly.Wing) -> Self:
        try:
            opts: WingOptions = next(o for o in comp.options if type(o) is WingOptions)
            sections = [Section.from_component(s) for s in comp.sections]
            spans = [Span.from_component(s) for s in comp.spans]
            movable_surfaces = [MovableSurface.from_component(s) for s in comp.movable_surfaces]

            return cls(
                name=comp.name,
                sections=sections,
                spans=spans,
                movable_surfaces = movable_surfaces,
                method=opts.method,
                panel_type=opts.panel_type,
                num_panels=opts.num_panels,
                mesh_file=opts.mesh_file,
                mesh_file_type=opts.mesh_file_type,
                mesh_definition=opts.mesh_definition,
                pos=comp.global_pos,
                offset=comp.offset,
                scaling=comp.scaling,
                alpha=comp.alpha,
                beta=comp.beta,
                roll=comp.roll,
                xc_ref=comp.xc_ref,
                inner_product_te=opts.inner_product_te,
                tol_se_wing=opts.tol_se_wing,
                proj_te=opts.proj_te,
                proj_te_dir=opts.proj_te_dir,
                proj_te_vector=opts.proj_te_vector,
                symmetry=comp.symmetry,
                mirror=comp.mirror,
                options=opts.output_opts,
            )

        except StopIteration:
            msg = f"No DUST options in component '{comp.name}'."
            raise ValueError(msg) from None

    def update(self, comp: assembly.Wing) -> None:
        for sec, assembly_sec in zip(self.sections, comp.sections, strict=True):
            sec.update(assembly_sec)
        for span, assembly_span in zip(self.spans, comp.spans, strict=True):
            span.update(assembly_span)

        if opts := next((o for o in comp.options if type(o) is WingOptions), None):
            self.num_panels = opts.num_panels
            self.mesh_file = opts.mesh_file
            self.mesh_file_type = opts.mesh_file_type
            self.mesh_definition = opts.mesh_definition
            self.inner_product_te = opts.inner_product_te
            self.tol_se_wing = opts.tol_se_wing
            self.proj_te = opts.proj_te
            self.proj_te_dir = opts.proj_te_dir
            self.proj_te_vector = opts.proj_te_vector

        self.pos = comp.global_pos
        self.offset = comp.offset
        self.scaling = comp.scaling
        self.alpha = comp.alpha
        self.beta = comp.beta
        self.roll = comp.roll
        self.xc_ref = comp.xc_ref
        self.symmetry = comp.symmetry
        self.mirror = comp.mirror

    def write_file(self, wing_file: Path) -> None:
        with wing_file.open("w") as f:
            f.writelines(self._make_header())
            f.writelines(self._make_global_props())
            f.writelines(self._make_sections())

    def _make_header(self) -> Sequence[str]:
        if self.mesh_file_type:
            out = []
            if self.mesh_file:
                out.append(f"mesh_file = {self.mesh_file}\n")
            out.append(f"mesh_file_type = {self.mesh_file_type}\n")
            return out
        if self.mesh_file:
            ext = self.mesh_file.suffix
            file_type = "cgns" if (ext == ".cgns") else "basic"
            return [
                f"mesh_file = {self.mesh_file}\n",
                f"mesh_file_type = {file_type}\n",
            ]
        return ["mesh_file_type = parametric\n"]

    def _make_global_props(self) -> Sequence[str]:
        out = [
            f"el_type = {self.method.value}\n",
            "\n",
            "offset = (/ 0.0, 0.0, 0.0 /)\n",
            f"scaling_factor = {self.scaling:.6f}\n",
            "\n",
            f"mesh_symmetry = {'T' if self.symmetry else 'F'}\n",
            "symmetry_point = (/ 0.0, 0.0, 0.0 /)\n",
            "symmetry_normal = (/ 0.0, 1.0, 0.0 /)\n",
            "\n",
            f"mesh_mirror = {'T' if self.mirror else 'F'}\n",
            "mirror_point = (/ {:.6f}, {:.6f}, {:.6f} /)\n".format(*self.offset),
            "mirror_normal = (/ 0.0, 0.0, 1.0 /)\n",
        ]
        if self.inner_product_te is not None:
            out += [f"inner_product_te = {self.inner_product_te:.6f}\n"]
        if self.tol_se_wing is not None:
            out += [f"tol_se_wing = {self.tol_se_wing:.6e}\n"]
        if self.proj_te is not None:
            out += [f"proj_te = {'T' if self.proj_te else 'F'}\n"]
            if self.proj_te and self.proj_te_dir is not None:
                out += [f"proj_te_dir = {self.proj_te_dir}\n"]
            if self.proj_te and self.proj_te_vector is not None:
                out += [
                    "proj_te_vector = (/ {:.6f}, {:.6f}, {:.6f} /)\n".format(
                        *self.proj_te_vector,
                    ),
                ]

        # Modify offsets when using lifting lines
        if self.method == WingMethod.LIFTING_LINE:
            chord = self.sections[0].chord
            twist = np.radians(self.sections[0].twist)
            off = (self.xc_ref - 0.25) * chord
            start = [
                self.offset[0] - off * np.cos(twist),
                self.offset[1],
                self.offset[2] + off * np.sin(twist),
            ]
            out += [
                "\n",
                "starting_point = (/ {:.6f}, {:.6f}, {:.6f} /)\n".format(*start),
                "# reference_chord_fraction is null with lifting lines\n",
            ]
        else:
            if self.panel_type is None:
                msg = (
                    f"'Panel_type' in '{type(self).__name__}' cannot be None if "
                    "'method' is lifting line."
                )
                raise ValueError(msg)

            out += [
                "\n",
                "starting_point = (/ {:.6f}, {:.6f}, {:.6f} /)\n".format(*self.offset),
                f"reference_chord_fraction = {self.xc_ref:.6f}\n",
                "\n",
                f"nelem_chord = {self.num_panels}\n",
                f"type_chord = {self.panel_type.value}\n",
            ]

        # Handle non-linear VLM  # TODO @Andres: Set to False now
        if self.method == WingMethod.VORTEX_LATTICE and any(
            sec.polar for sec in self.sections
        ):
            out += ["airfoil_table_correction = F\n"]

        return out

    def _make_sections(self) -> Sequence[str]:
        if self.mesh_definition:
            return self.mesh_definition

        # No sections if using an external mesh file
        if self.mesh_file:
            return []

        # First section
        section = self.sections[0]
        out = [
            "\n",
            f"chord = {section.chord:.6f}\n",
            f"twist = {section.twist:.6f}\n",
            f"airfoil = {section.airfoil}\n",
        ]
        if section.polar:
            out += [f"airfoil_table = {section.name}.c81\n"]

        # Spans and sections
        for i, span in enumerate(self.spans):
            if self.method == WingMethod.LIFTING_LINE:
                span_len, sweep, dihed = self._compute_ll_geometry(
                    self.sections[i],
                    self.sections[i + 1],
                    span,
                )
            else:
                span_len = span.length
                sweep = span.sweep
                dihed = span.dihed

            section = self.sections[i + 1]
            out += [
                "\n",
                f"span = {span_len:.6f}\n",
                f"sweep = {sweep:.6f}\n",
                f"dihed = {dihed:.6f}\n",
                f"nelem_span = {span.num_panels}\n",
                f"type_span = {span.panel_type.value}\n",
                "\n",
                f"chord = {section.chord:.6f}\n",
                f"twist = {section.twist:.6f}\n",
                f"airfoil = {section.airfoil}\n",
            ]
            if section.polar:
                out += [f"airfoil_table = {section.name}.c81\n"]

        return out

    def _compute_ll_geometry(
        self,
        sec_0: Section,
        sec_1: Section,
        span: Span,
    ) -> tuple[float, float, float]:
        cr = sec_0.chord
        ct = sec_1.chord
        ar = np.radians(sec_0.twist)
        at = np.radians(sec_1.twist)
        lam = np.radians(span.sweep)
        phi = np.radians(span.dihed)
        b = span.length
        delta_xc = self.xc_ref - 0.25

        sweep = np.arctan(
            np.tan(lam)
            - delta_xc
            / (b * np.cos(phi) * np.cos(lam))
            * (ct * np.cos(at) - cr * np.cos(ar)),
        )
        dihed = np.arctan(
            np.cos(sweep)
            / np.cos(lam)
            * (
                np.tan(phi)
                + delta_xc / (b * np.cos(phi)) * (ct * np.sin(at) - cr * np.sin(ar))
            ),
        )
        if abs(dihed) < 1e-4:  # noqa: PLR2004
            length = b
        else:
            length = (
                1.0
                / np.sin(dihed)
                * (b * np.sin(phi) + delta_xc * (ct * np.sin(at) - cr * np.sin(ar)))
            )

        sweep = np.degrees(sweep)
        dihed = np.degrees(dihed)

        return length, sweep, dihed


class PropellerOptions(assembly.ComponentOptions):
    def __init__(
        self,
        parent: str | None = None,
        output_options: OutputOptions | None = None,
    ) -> None:
        self.parent = parent
        self.output_opts = output_options or OutputOptions()


class Propeller:
    def __init__(
        self,
        name: str,
        blade: Wing,
        hub_offset: float,
        n_blades: int,
        pitch: float,
        rpm: float,
        reverse: bool,
        pos: NDArray[np.float64],
        alpha: float,
        beta: float,
        options: OutputOptions,
        parent: str | None = None,
    ) -> None:
        self.name = name
        self.blade = blade
        self.hub_offset = hub_offset
        self.n_blades = n_blades
        self.pitch = pitch
        self.rpm = rpm
        self.reverse = reverse
        self.pos = pos
        self.alpha = alpha
        self.beta = beta
        self.options = options
        self.parent = parent

    @classmethod
    def from_component(cls, comp: assembly.Propeller) -> Self:
        opts_gen = (o for o in comp.options if type(o) is PropellerOptions)
        opts: PropellerOptions = next(opts_gen, PropellerOptions())
        blade = Wing.from_component(comp.blade)

        return cls(
            name=comp.name,
            blade=blade,
            hub_offset=comp.hub_offset,
            n_blades=comp.n_blades,
            pitch=comp.pitch,
            rpm=comp.rpm,
            reverse=comp.reverse,
            pos=comp.global_pos,
            alpha=comp.alpha,
            beta=comp.beta,
            options=opts.output_opts,
            parent=opts.parent,
        )

    def update(self, comp: assembly.Propeller) -> None:
        self.blade.update(comp.blade)
        self.hub_offset = comp.hub_offset
        self.n_blades = comp.n_blades
        self.pitch = comp.pitch
        self.rpm = comp.rpm
        self.reverse = comp.reverse
        self.pos = comp.global_pos
        self.alpha = comp.alpha
        self.beta = comp.beta

    def write_file(self, blade_file: Path) -> None:
        self.blade.write_file(blade_file)


class Fuselage:
    def __init__(
        self,
        name: str,
        maximum_width: float,
        maximum_height: float,
        length: float,
        pos: NDArray[np.float64] | None = None,
    ) -> None:
        self.name = name
        self.maximum_width = maximum_width
        self.maximum_height = maximum_height
        self.length = length

        self.pos = pos if pos is not None else np.zeros(3)

    @classmethod
    def from_component(cls, comp: assembly.Fuselage) -> Self:
        return cls(
            name=comp.name,
            maximum_width=comp.maximum_width,
            maximum_height=comp.maximum_height,
            length=comp.length,
            pos=comp.global_pos,
        )

    def update(self, comp: assembly.Fuselage) -> None:
        self.maximum_width = comp.maximum_width
        self.maximum_height = comp.maximum_height
        self.length = comp.length
        self.pos = comp.global_pos

    def write_file(self, fuselage_file: Path) -> None:
        # Write wing file
        with fuselage_file.open("w") as f:
            f.writelines(
                [
                    "mesh_file = ../geometry/fuselage_parametric/\n",
                    "mesh_file_type = basic\n",
                    "el_type = p\n",
                    "scaling_factor = 1\n",
                    "inner_product_te = -0.99999999999\n",
                    "mesh_symmetry = F\n",
                    "symmetry_point = (/ 0.0, 0.0, 0.0 /)\n",
                    "symmetry_normal = (/ 0.0, 1.0, 0.0 /)\n",
                ],
            )


class Post(ABC):
    def __init__(
        self,
        name: str,
        start_res: int = 1,
        end_res: int = 2,
        step_res: int = 1,
        components: Sequence[str] | None = None,
    ) -> None:
        self.name = name
        self.start_res = start_res
        self.end_res = end_res
        self.step_res = step_res
        self.components = components or ["all"]

    @abstractmethod
    def get_dust_strings(self) -> Sequence[str]:
        pass

    @abstractmethod
    def parse_file(self, file_name: Path) -> None:
        pass


class PostLoads(Post):
    def __init__(
        self,
        average: bool = False,
        reference: str = "0",
        rotation: NDArray[np.float64] | None = None,
        **kwargs: Any,  # noqa: ANN401
    ) -> None:
        super().__init__(**kwargs)
        self.average = average
        self.reference = reference
        self.rotation = rotation

        self.t: NDArray[np.float64]
        self.f: NDArray[np.float64]
        self.m: NDArray[np.float64]

    def get_dust_strings(self) -> Sequence[str]:
        # Parameters of the analysis
        strings = [
            "analysis = {\n",
            "    type = integral_loads\n",
            f"    name = {self.name}\n",
            f"    start_res = {self.start_res}\n",
            f"    end_res = {self.end_res}\n",
            f"    step_res = {self.step_res}\n",
            "    format = dat\n",
            f"    average = {'T' if self.average else 'F'}\n",
            f"    reference_tag = {self.reference}\n",
        ]

        # Add list of components
        for comp in self.components:
            strings += [f"    component = {comp}\n"]

        # Close the block
        return [*strings, "}\n"]

    def parse_file(self, file_name: Path) -> None:
        with file_name.open() as fh:
            # Skip header
            for _ in range(4):
                fh.readline()

            time: list[float] = []
            force: list[list[float]] = []
            moment: list[list[float]] = []

            # Average case only has one line
            if self.average:
                line = fh.readline().split()
                f = np.array([float(x) for x in line[0:3]])
                m = np.array([float(x) for x in line[3:6]])

                if self.rotation is not None:
                    rot = np.array([float(x) for x in line[6:15]])
                    rot = np.reshape(rot, (3, 3), order="F")
                    rot = np.dot(self.rotation, rot)
                    off = np.array([float(x) for x in line[15:18]])
                    f = np.dot(rot, f)
                    m = np.dot(rot, m) + np.cross(off, f)

                time.append(0.0)
                force.append(f.tolist())
                moment.append(m.tolist())

            # Read tabulated data otherwise
            else:
                data = fh.readlines()
                for _line in data:
                    line = _line.strip().split()
                    if not line:
                        continue

                    f = np.array([float(x) for x in line[1:4]])
                    m = np.array([float(x) for x in line[4:7]])

                    if self.rotation is not None:
                        rot = np.array([float(x) for x in line[7:16]])
                        rot = np.reshape(rot, (3, 3), order="F")
                        rot = np.dot(self.rotation, rot)
                        off = np.array([float(x) for x in line[16:19]])
                        f = np.dot(rot, f)
                        m = np.dot(rot, m) + np.cross(off, f)

                    time.append(float(line[0]))
                    force.append(f.tolist())
                    moment.append(m.tolist())

            self.t = np.array(time)
            self.f = np.array(force)
            self.m = np.array(moment)


class PostSpanwiseLoads(Post):
    def __init__(
        self,
        average: bool = False,
        resolution: int = SPANLOAD_DEFAULT_NUM_STATIONS,
        axis_nod: NDArray[np.float64] | None = None,
        axis_dir: NDArray[np.float64] | None = None,
        symmetric_geo: bool = False,
        **kwargs: Any,  # noqa: ANN401
    ) -> None:
        super().__init__(**kwargs)
        self.average = average
        self.resolution = resolution
        self.axis_nod = np.zeros(3) if axis_nod is None else axis_nod
        self.axis_dir = np.array([0.0, 1.0, 0.0]) if axis_dir is None else axis_dir
        self.symmetric_geo = symmetric_geo

        self.y_cen: NDArray[np.float64]
        self.y_span: NDArray[np.float64]
        self.chord: NDArray[np.float64]
        self.t: NDArray[np.float64]
        self.fx: NDArray[np.float64]
        self.fy: NDArray[np.float64]
        self.fz: NDArray[np.float64]
        self.mo: NDArray[np.float64]

    def get_dust_strings(self) -> Sequence[str]:
        fvec_temp = "(/ {:.6f}, {:.6f}, {:.6f} /)\n"
        return [
            "analysis = {\n",
            "    type = sectional_loads\n",
            f"    name = {self.name}\n",
            f"    start_res = {self.start_res}\n",
            f"    end_res = {self.end_res}\n",
            f"    step_res = {self.step_res}\n",
            "    format = dat\n",
            f"    average = {'T' if self.average else 'F'}\n",
            f"    component = {self.components[0]}\n",
            "    axis_nod = " + fvec_temp.format(*self.axis_nod),
            "    axis_dir = " + fvec_temp.format(*self.axis_dir),
            "    lifting_line_data = F\n",
            "    vortex_lattice_data = F\n",
            "}\n",
        ]

    def parse_file(self, file_name: Path) -> None:
        # Construct the actual file name
        stem = file_name.stem
        suffix = file_name.suffix
        if self.average:
            suffix = "_ave" + suffix

        path = file_name.with_name(stem + "_Fx" + suffix)

        # Get geometry information
        y_0, self.y_cen, self.y_span, self.chord = self._parse_geometry(path)

        # Read forces and moments
        for comp, att in zip(
            ("_Fx", "_Fy", "_Fz", "_Mo"),
            ("fx", "fy", "fz", "mo"),
            strict=True,
        ):
            path = file_name.with_name(stem + comp + suffix)
            self.t, f = self._parse_force(path, y_0, self.y_cen)
            setattr(self, att, f)

    def _parse_geometry(
        self,
        file_name: Path,
    ) -> tuple[
        NDArray[np.float64],
        NDArray[np.float64],
        NDArray[np.float64],
        NDArray[np.float64],
    ]:
        with file_name.open("r") as f:
            # Skip header
            for _ in range(2):
                f.readline()

            # Read geometry
            y_cen_0 = np.array([float(x) for x in f.readline().split()])
            y_span_0 = np.array([float(x) for x in f.readline().split()])
            chord_0 = np.array([float(x) for x in f.readline().split()])

            # If the geometry is symmetric, the ordering of the chords seems to be wrong
            if self.symmetric_geo:
                n2 = len(chord_0) // 2
                chord_0 = np.concat((np.flip(chord_0[:n2]), chord_0[n2:]))

        y_0 = y_cen_0[0] - y_span_0[0] * 0.5
        y_1 = y_cen_0[-1] + y_span_0[-1] * 0.5
        y_sec = np.linspace(y_0, y_1, self.resolution + 1)

        y_cen = (y_sec[1:] + y_sec[:-1]) * 0.5
        y_span = y_sec[1:] - y_sec[:-1]
        chord = np.interp(y_cen, y_cen_0, chord_0)

        return y_cen_0, y_cen, y_span, chord

    def _parse_force(
        self,
        file_name: Path,
        y_in: NDArray[np.float64],
        y_out: NDArray[np.float64],
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        with file_name.open("r") as fh:
            # Skip header and geometry
            for _ in range(6):
                fh.readline()

            time: list[float] = []
            data: list[list[float]] = []

            # Average case only has one line
            if self.average:
                line = fh.readline().split()
                f = np.array([float(x) for x in line])
                f = np.interp(y_out, y_in, f)
                time.append(0.0)
                data.append(f.tolist())

            # Read tabulated data otherwise
            else:
                for _line in fh.readlines():
                    # Parse line and skip if empty
                    line = _line.strip().split()
                    if not line:
                        continue

                    # Store data
                    f = np.array([float(x) for x in line[1:-12]])
                    f = np.interp(y_out, y_in, f)
                    time.append(float(line[0]))
                    data.append(f.tolist())

        return np.array(time), np.array(data)


class PostViz(Post):
    def __init__(
        self,
        fmt: str = "vtk",
        wake: bool = True,
        separate_wake: bool = False,
        average: bool = False,
        variables: Sequence[str] | None = None,
        **kwargs: Any,  # noqa: ANN401
    ) -> None:
        super().__init__(**kwargs)
        self.fmt = fmt
        self.wake = wake
        self.separate_wake = separate_wake
        self.average = average
        self.variables = (
            variables if variables is not None else ["vorticity_vector", "cp"]
        )

    def get_dust_strings(self) -> Sequence[str]:
        # Parameters of the analysis
        strings = [
            "analysis = {\n",
            "    type = viz\n",
            f"    name = {self.name}\n",
            f"    start_res = {self.start_res}\n",
            f"    end_res = {self.end_res}\n",
            f"    step_res = {self.step_res}\n",
            f"    format = {self.fmt}\n",
            f"    wake = {'T' if self.wake else 'F'}\n",
            f"    separate_wake = {'T' if self.separate_wake else 'F'}\n",
            f"    average = {'T' if self.average else 'F'}\n",
        ]

        # Add list of variables
        for var in self.variables:
            strings += [f"    variable = {var}\n"]

        # Add list of components
        for comp in self.components:
            strings += [f"    component = {comp}\n"]

        # Close the block
        return [*strings, "}\n"]

    def parse_file(self, file_name: Path) -> None:
        pass


class Driver:
    def __init__(
        self,
        environment: assembly.Environment,
        options: Options,
        wings: Sequence[Wing] | None = None,
        propellers: Sequence[Propeller] | None = None,
        fuselages: Sequence[Fuselage] | None = None,
    ) -> None:
        self.environment = environment
        self.wings = wings if wings is not None else []
        self.propellers = propellers if propellers is not None else []
        self.fuselages = fuselages if fuselages is not None else []
        self.run_directory: str | None = None
        self.options = options

    def preprocess(self) -> None:
        # Create output folders
        self.options.output_dir.mkdir(parents=True, exist_ok=True)
        self.options.post_dir.mkdir(parents=True, exist_ok=True)

        # Create preprocess files
        for prop in self.propellers:
            prop.write_file(Path(prop.blade.name + ".in"))

        for wing in self.wings:
            wing.write_file(Path(wing.name + ".in"))

        for fuselage in self.fuselages:
            fuselage.write_file(Path(fuselage.name + ".in"))

        self.write_dust_pre()

        # Run `dust_pre`
        out = sp.run(
            self.options.dust_pre,
            capture_output=True,
            text=True,
            check=False,
        )
        _log_cmd_output(
            Path(self.options.name + ".log"),
            out,
            "dust_pre",
            mode="w",
        )

        # Create simulation files
        self.write_references()
        self.write_dust()

    def run(self) -> None:
        # Run `dust`
        run_env = os.environ.copy()
        run_env["OMP_NUM_THREADS"] = str(self.options.n_threads)

        out = sp.run(
            self.options.dust,
            capture_output=True,
            text=True,
            env=run_env,
            check=False,
        )
        _log_cmd_output(
            Path(self.options.name + ".log"),
            out,
            "dust",
            mode="a",
        )

    def postprocess(self, analyses: Sequence[Post] | None = None) -> None:
        # Create postprocess file
        analyses = analyses or []
        self.options.post_dir.mkdir(parents=True, exist_ok=True)
        self.write_dust_post(analyses)

        # Run `dust_post`
        out = sp.run(
            self.options.dust_post,
            capture_output=True,
            text=True,
            check=False,
        )
        _log_cmd_output(
            Path(self.options.name + ".log"),
            out,
            "dust_post",
            mode="a",
        )

        # Parse postprocessing files
        self.parse_dust_post(analyses)

    def write_dust_pre(self) -> None:
        with Path("dust_pre.in").open("w") as f:
            # Connect components, files and tags for each propeller and wing
            for prop in self.propellers:
                f.writelines(
                    [
                        f"comp_name = {prop.name}\n",
                        f"geo_file = {prop.blade.name}.in\n",
                        f"ref_tag = {prop.name}_ref\n",
                        "\n",
                    ],
                )

            for wing in self.wings:
                f.writelines(
                    [
                        f"comp_name = {wing.name}\n",
                        f"geo_file = {wing.name}.in\n",
                        f"ref_tag = {wing.name}_ref\n",
                        "\n",
                    ],
                )

            for fuselage in self.fuselages:
                f.writelines(
                    [
                        f"comp_name = {fuselage.name}\n",
                        f"geo_file = {fuselage.name}.in\n",
                        f"ref_tag = {fuselage.name}_ref\n",
                        "\n",
                    ],
                )

            # Set output name
            f.write("file_name = geo_input.h5\n")

    def write_references(self) -> None:
        with Path("references.in").open("w") as f:
            # Body axes definition
            f.writelines(
                [
                    "reference_tag = root\n",
                    "parent_tag = 0\n",
                    "origin = (/ 0.0, 0.0, 0.0 /)\n",
                    "orientation = (/ 1, 0, 0, 0, 1, 0, 0, 0, 1 /)\n",
                    "multiple = F\n",
                ],
            )

            # Moving frame
            f.write(f"moving = {'T' if self.options.moving else 'F'}\n")

            if not self.options.moving:
                f.write("\n")
            else:
                fvec_temp = "(/ {:.6f}, {:.6f}, {:.6f} /)"
                pole_function_ = np.ceil(self.options.pole_weights).astype(int)
                pole_function = fvec_temp.format(*pole_function_)
                pole_weights = fvec_temp.format(*self.options.pole_weights)
                pole_omega = fvec_temp.format(*self.options.pole_omega)
                pole_phase = fvec_temp.format(*self.options.pole_phase)
                rotation_dir = fvec_temp.format(*self.options.rotation_dir)

                f.writelines(
                    [
                        "motion = {\n",
                        "    pole = {\n",
                        "        input = position\n",
                        "        input_type = simple_function\n",
                        f"        function = {pole_function}\n",
                        f"        amplitude = {self.options.pole_ampl}\n",
                        f"        vector = {pole_weights}\n",
                        f"        omega = {pole_omega}\n",
                        f"        phase = {pole_phase}\n",
                        "        offset = (/ 0.0 , 0.0 , 0.0 /)\n",
                        "    }\n",
                        "    rotation = {\n",
                        "        input = position\n",
                        "        input_type = simple_function\n",
                        "        function = 1\n",
                        f"        Axis = {rotation_dir}\n",
                        f"        amplitude = {self.options.rotation_ampl}\n",
                        f"        omega = {self.options.rotation_omega}\n",
                        f"        phase = {self.options.rotation_phase}\n",
                        "        offset = 0.0\n",
                        "    }\n",
                        "}\n\n",
                    ],
                )

            fvec_temp = "(/ {:.6f}, {:.6f}, {:.6f} /)\n"
            fmat_temp = "(/ " + ", ".join(9 * ["{:.6f}"]) + " /)\n"

            # Define the local frame of each component of the aircraft
            for fuselage in self.fuselages:
                f.writelines(
                    [
                        f"reference_tag = {fuselage.name}_ref\n",
                        "parent_tag = root\n",
                        "origin = " + fvec_temp.format(*fuselage.pos),
                        "orientation = (/ 1, 0, 0, 0, 1, 0, 0, 0, 1 /)\n",
                        "multiple = F\n",
                        "moving = F\n",
                        "\n",
                    ],
                )

            for wing in self.wings:
                alpha = np.radians(wing.alpha)
                beta = np.radians(wing.beta)
                roll = np.radians(wing.roll)
                rotation = (
                    np.array(
                        [
                            [1.0, 0.0, 0.0],
                            [0.0, np.cos(roll), -np.sin(roll)],
                            [0.0, np.sin(roll), np.cos(roll)],
                        ],
                    )
                    @ np.array(
                        [
                            [np.cos(alpha), 0.0, np.sin(alpha)],
                            [0.0, 1.0, 0.0],
                            [-np.sin(alpha), 0.0, np.cos(alpha)],
                        ],
                    )
                    @ np.array(
                        [
                            [np.cos(beta), -np.sin(beta), 0.0],
                            [np.sin(beta), np.cos(beta), 0.0],
                            [0.0, 0.0, 1.0],
                        ],
                    )
                )
                f.writelines(
                    [
                        f"reference_tag = {wing.name}_ref\n",
                        "parent_tag = root\n",
                        "origin = " + fvec_temp.format(*wing.pos),
                        "orientation = " + fmat_temp.format(*rotation.flatten()),
                        "multiple = F\n",
                        "moving = F\n",
                        "\n",
                    ],
                )

            for prop in self.propellers:
                alpha = np.radians(prop.alpha)
                beta = np.radians(prop.beta + 180.0)
                rotation = np.array(
                    [
                        [np.cos(alpha), 0.0, np.sin(alpha)],
                        [0.0, 1.0, 0.0],
                        [-np.sin(alpha), 0.0, np.cos(alpha)],
                    ],
                ) @ np.array(
                    [
                        [np.cos(beta), -np.sin(beta), 0.0],
                        [np.sin(beta), np.cos(beta), 0.0],
                        [0.0, 0.0, 1.0],
                    ],
                )
                f.writelines(
                    [
                        f"reference_tag = {prop.name}_ref\n",
                        "parent_tag = root\n",
                        "origin = " + fvec_temp.format(*prop.pos),
                        "orientation = " + fmat_temp.format(*rotation.flatten()),
                        "multiple = T\n",
                        "moving = F\n",
                    ],
                )

                if prop.reverse:
                    rot_axis = -1.0
                    pitch = -prop.pitch
                else:
                    rot_axis = 1.0
                    pitch = prop.pitch

                f.writelines(
                    [
                        "",
                        "multiplicity = {\n",
                        "    mult_type = rotor\n",
                        f"    n_blades = {prop.n_blades}\n",
                        f"    rot_axis = (/ {rot_axis:.1f}, 0.0, 0.0 /)\n",
                        f"    rot_rate = {prop.rpm / 30.0 * np.pi:.6f}\n",
                        "    psi_0 = 0.0\n",
                        f"    hub_offset = {prop.hub_offset:.6f}\n",
                        "    n_dofs = 1\n",
                        "    dof = {\n",
                        "        hinge_type = Pitch\n",
                        "        hinge_offset = (/ 0.0, 0.0, 0.0 /)\n",
                        f"        collective = {pitch:.6f}\n",
                        "    }\n",
                        "}\n",
                        "\n",
                    ],
                )

    def write_dust(self) -> None:
        # Compute final time and time step
        if self.options.dt is not None and self.options.t_end is not None:
            dt = self.options.dt
            tend = self.options.t_end
        elif self.propellers:
            max_rpm = max(prop.rpm for prop in self.propellers)
            dt_per_turn = 60.0 / max_rpm
            dt = dt_per_turn / self.options.steps_per_turn
            tend = self.options.n_turns * dt_per_turn
        else:
            msg = (
                f"Solver '{type(self).__name__}' needs total simulation time and step "
                "size if there are no propellers"
            )
            raise ValueError(msg)

        dt_out = dt if self.options.dt_out is None else self.options.dt_out
        fvec_temp = "(/ {:.6f}, {:.6f}, {:.6f} /)\n"
        ivec_temp = "(/ {}, {}, {} /)\n"

        with Path("dust.in").open("w") as f:
            f.writelines(
                [
                    f"basename = {self.options.output_dir / self.options.name}\n",
                    "debug_level = 0\n",
                    "\n",
                    "u_inf = " + fvec_temp.format(*self.environment.velocity),
                    f"rho_inf = {self.environment.density}\n",
                    f"a_inf = {self.environment.sound_speed}\n",
                    f"p_inf = {self.environment.pressure}\n",
                    f"mu_inf = {self.environment.dyn_viscosity}\n",
                    "\n",
                    f"t_start = {self.options.t_start}\n",
                    f"tend = {tend:.6f}\n",
                    f"dt = {dt:.6e}\n",
                    f"dt_out = {dt_out:.6e}\n",
                    "output_start = {}\n".format(
                        "T" if self.options.output_start else "F",
                    ),
                    "\n",
                    "geometry_file = geo_input.h5\n",
                    "\n",
                    "reference_file = references.in\n",
                    "\n",
                    f"n_wake_panels = {self.options.n_wake_panels}\n",
                    f"n_wake_particles = {self.options.n_wake_particles}\n",
                    "particles_box_min = "
                    + fvec_temp.format(*self.options.particles_box_min),
                    "particles_box_max = "
                    + fvec_temp.format(*self.options.particles_box_max),
                    "\n",
                    "penetration_avoidance = {}\n".format(
                        "T" if self.options.penetration_avoidance else "F",
                    ),
                    "\n",
                    f"fmm = {'T' if self.options.fmm else 'F'}\n",
                    f"box_length = {self.options.box_length}\n",
                    "n_box = " + ivec_temp.format(*self.options.n_box),
                    "octree_origin = " + fvec_temp.format(*self.options.octree_origin),
                    f"n_octree_levels = {self.options.n_octree_levels}\n",
                    f"min_octree_part = {self.options.min_octree_part}\n",
                    f"multipole_degree = {self.options.multipole_degree}\n",
                    "\n",
                    f"ll_max_iter = {self.options.max_iter}\n",
                    f"ll_tol = {self.options.tol:.6e}\n",
                    f"vl_maxiter = {self.options.max_iter}\n",
                    f"vl_tol = {self.options.tol:.6e}\n",
                ],
            )

    def write_dust_post(self, analyses: Sequence[Post]) -> None:
        with Path("dust_post.in").open("w") as f:
            # Header
            f.writelines(
                [
                    f"data_basename = {self.options.output_dir / self.options.name}\n",
                    f"basename = {self.options.post_dir / self.options.name}\n",
                ],
            )

            # Write analyses
            for analysis in analyses:
                f.writelines(["\n", *analysis.get_dust_strings()])

    def parse_dust_post(self, analyses: Sequence[Post]) -> None:
        for analysis in analyses:
            file = self.options.post_dir / f"{self.options.name}_{analysis.name}.dat"
            try:
                analysis.parse_file(file)
            except FileNotFoundError:
                if isinstance(analysis, PostSpanwiseLoads):
                    continue
                raise
