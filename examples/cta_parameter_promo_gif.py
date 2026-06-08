from __future__ import annotations

import argparse
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from scipy.interpolate import PchipInterpolator

import cta_geometry as cta
from multiads.utilities.cst import evaluate_cst_surface


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = REPO_ROOT / "outputs" / "CTA_case" / "videos" / "parameter_promo"

SECTION_ORDER = tuple(cta.CTA_SECTION_ORDER)
OUTER_SECTIONS = {"s4", "s4a", "s4b", "s5"}


@dataclass(frozen=True, slots=True)
class ParameterVisual:
    key: str
    label: str
    group: str
    description: str


PARAMETERS = [
    ParameterVisual("delta_c0_m", "dC0", "Planform", "Centerbody chord: C0, C1 and C2 move together."),
    ParameterVisual("delta_c3_m", "dC3", "Planform", "Transition chord: controls the chord at section S3."),
    ParameterVisual("delta_c5_m", "dC5", "Planform", "Tip chord: controls the outer-wing tip chord S5."),
    ParameterVisual("taper_ratio_midwing", "TRw", "Planform", "Mid-wing taper: C4 = TRw * C3."),
    ParameterVisual("rspan_midwing", "RBw", "Planform", "Relative span position of S4 measured from S3."),
    ParameterVisual("span_wing_m", "Bw", "Planform", "Outer wing span: position of the S5 tip station."),
    ParameterVisual("sweep_midwing_deg", "S1", "Planform", "Mid-wing sweep at 50% chord, setting LE position of S4."),
    ParameterVisual("sweep_outwing_deg", "S2", "Planform", "Outer-wing sweep at 25% chord, setting S4a/S4b/S5 LE positions."),
    ParameterVisual("twist_s4_deg", "T4", "Outer twist", "Outer-wing twist at section S4 around the 25% chord line."),
    ParameterVisual("twist_s4a_deg", "T4a", "Outer twist", "Outer-wing twist at derived section S4a."),
    ParameterVisual("twist_s4b_deg", "T4b", "Outer twist", "Outer-wing twist at derived section S4b."),
    ParameterVisual("twist_s5_deg", "T5", "Outer twist", "Outer-wing twist at the tip section S5."),
    ParameterVisual("thickness_s4", "t/c S4", "Outer thickness", "Maximum thickness scaling of the S4 profile; TE thickness stays absolute."),
    ParameterVisual("thickness_s5", "t/c S5", "Outer thickness", "Maximum thickness scaling of the S5 profile; TE thickness stays absolute."),
]

GROUP_COLORS = {
    "Planform": "#0f6b8f",
    "Outer twist": "#e08d00",
    "Outer thickness": "#bc1f49",
}


def smooth_cycle_value(phase: float, baseline: float, lower: float, upper: float) -> float:
    """Move baseline -> upper -> baseline -> lower -> baseline."""

    phase = float(phase % 1.0)
    if phase <= 0.25:
        return baseline + (upper - baseline) * phase / 0.25
    if phase <= 0.50:
        return upper + (baseline - upper) * (phase - 0.25) / 0.25
    if phase <= 0.75:
        return baseline + (lower - baseline) * (phase - 0.50) / 0.25
    return lower + (baseline - lower) * (phase - 0.75) / 0.25


def design_baseline() -> dict[str, float]:
    return {
        name: float(info[0])
        for name, info in cta.CTA_CFD_JUNE_14_VARIABLE_INFO.items()
    }


def design_with_parameter(parameter: ParameterVisual, phase: float) -> tuple[dict[str, float], float]:
    values = design_baseline()
    baseline, lower, upper = cta.CTA_CFD_JUNE_14_VARIABLE_INFO[parameter.key]
    value = smooth_cycle_value(phase, float(baseline), float(lower), float(upper))
    values[parameter.key] = value
    return values, value


def _outer_thickness_scaled_profile(
    x_over_c: np.ndarray,
    upper: np.ndarray,
    lower: np.ndarray,
    target_tc: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Scale profile thickness while keeping the normalized TE wedge shape.

    This mirrors the MADS CTA law used for the outer wing: the thickness away
    from the trailing edge is scaled, while the TE thickness term is preserved.
    The absolute TE requirement is enforced upstream by recomputing TE/c from
    the fixed TE thickness and the current chord.
    """

    camber = 0.5 * (upper + lower)
    thickness = np.maximum(upper - lower, 0.0)
    te_thickness = float(thickness[-1])
    shape_thickness = thickness - x_over_c * te_thickness

    target = float(target_tc)
    if target <= 0.0:
        return upper, lower

    def max_thickness(scale: float) -> float:
        candidate = np.maximum(scale * shape_thickness + x_over_c * te_thickness, 0.0)
        return float(np.max(candidate))

    lo, hi = 0.0, 1.0
    while max_thickness(hi) < target and hi < 1.0e4:
        hi *= 2.0

    for _ in range(40):
        mid = 0.5 * (lo + hi)
        if max_thickness(mid) < target:
            lo = mid
        else:
            hi = mid

    new_thickness = np.maximum(hi * shape_thickness + x_over_c * te_thickness, 0.0)
    return camber + 0.5 * new_thickness, camber - 0.5 * new_thickness


def _section_profile(
    section_id: str,
    x_over_c: np.ndarray,
    derived: dict[str, dict[str, float]],
) -> tuple[np.ndarray, np.ndarray]:
    profile = cta.CTA_PROFILE_DATA[section_id]
    te = float(derived["trailing_edge_thickness"][section_id])
    upper = evaluate_cst_surface(
        x_over_c,
        tuple(profile["upper"]),
        n1=cta.CTA_CST_N1,
        n2=cta.CTA_CST_N2,
        trailing_edge_thickness_over_c=te,
        sign=1.0,
    )
    lower = evaluate_cst_surface(
        x_over_c,
        tuple(profile["lower"]),
        n1=cta.CTA_CST_N1,
        n2=cta.CTA_CST_N2,
        trailing_edge_thickness_over_c=te,
        sign=-1.0,
    )
    if section_id in OUTER_SECTIONS:
        upper, lower = _outer_thickness_scaled_profile(
            x_over_c,
            upper,
            lower,
            float(derived["outer_thickness"][section_id]),
        )
    return upper, lower


def _section_arrays(derived: dict[str, dict[str, float]]) -> tuple[np.ndarray, ...]:
    y = np.asarray([derived["spanwise_y_m"][section] for section in SECTION_ORDER], dtype=float)
    le_x = np.asarray([derived["leading_edge_x_m"][section] for section in SECTION_ORDER], dtype=float)
    le_z = np.asarray([derived["leading_edge_z_m"][section] for section in SECTION_ORDER], dtype=float)
    chord = np.asarray([derived["chords_m"][section] for section in SECTION_ORDER], dtype=float)
    twist = np.asarray([derived["twist_deg"][section] for section in SECTION_ORDER], dtype=float)
    return y, le_x, le_z, chord, twist


def build_visual_surface(
    design_values: dict[str, float],
    *,
    n_span: int,
    n_chord: int,
) -> dict[str, np.ndarray]:
    """Build a lightweight 3D CTA surface for visual communication."""

    derived = cta.derive_cfd_geometry_values(design_values)
    x_over_c = 0.5 * (1.0 - np.cos(np.linspace(0.0, np.pi, n_chord)))
    y_anchor, le_x_anchor, le_z_anchor, chord_anchor, twist_anchor = _section_arrays(derived)

    upper_anchor = []
    lower_anchor = []
    for section in SECTION_ORDER:
        upper, lower = _section_profile(section, x_over_c, derived)
        upper_anchor.append(upper)
        lower_anchor.append(lower)
    upper_anchor = np.asarray(upper_anchor, dtype=float)
    lower_anchor = np.asarray(lower_anchor, dtype=float)

    y_pos = np.unique(
        np.concatenate(
            [
                np.linspace(float(y_anchor[0]), float(y_anchor[-1]), n_span),
                y_anchor,
            ],
        ),
    )

    le_x_fun = PchipInterpolator(y_anchor, le_x_anchor)
    le_z_fun = PchipInterpolator(y_anchor, le_z_anchor)
    chord_fun = PchipInterpolator(y_anchor, chord_anchor)
    twist_fun = PchipInterpolator(y_anchor, twist_anchor)
    upper_fun = [PchipInterpolator(y_anchor, upper_anchor[:, i]) for i in range(n_chord)]
    lower_fun = [PchipInterpolator(y_anchor, lower_anchor[:, i]) for i in range(n_chord)]

    x_upper = np.zeros((y_pos.size, n_chord), dtype=float)
    y_upper = np.zeros_like(x_upper)
    z_upper = np.zeros_like(x_upper)
    x_lower = np.zeros_like(x_upper)
    y_lower = np.zeros_like(x_upper)
    z_lower = np.zeros_like(x_upper)

    for row, yy in enumerate(y_pos):
        chord = float(chord_fun(yy))
        le_x = float(le_x_fun(yy))
        le_z = float(le_z_fun(yy))
        theta = np.radians(float(twist_fun(yy)))
        cos_t = np.cos(theta)
        sin_t = np.sin(theta)
        x_local = chord * x_over_c
        upper_z_local = chord * np.asarray([fun(yy) for fun in upper_fun], dtype=float)
        lower_z_local = chord * np.asarray([fun(yy) for fun in lower_fun], dtype=float)

        x_upper[row, :] = le_x + x_local * cos_t - upper_z_local * sin_t
        y_upper[row, :] = yy
        z_upper[row, :] = le_z + x_local * sin_t + upper_z_local * cos_t
        x_lower[row, :] = le_x + x_local * cos_t - lower_z_local * sin_t
        y_lower[row, :] = yy
        z_lower[row, :] = le_z + x_local * sin_t + lower_z_local * cos_t

    mirror = slice(1, None)
    full = {
        "upper_x": np.vstack([x_upper[:0:-1, :], x_upper]),
        "upper_y": np.vstack([-y_upper[:0:-1, :], y_upper]),
        "upper_z": np.vstack([z_upper[:0:-1, :], z_upper]),
        "lower_x": np.vstack([x_lower[:0:-1, :], x_lower]),
        "lower_y": np.vstack([-y_lower[:0:-1, :], y_lower]),
        "lower_z": np.vstack([z_lower[:0:-1, :], z_lower]),
        "y_pos": y_pos,
        "le_x": np.asarray(le_x_fun(y_pos), dtype=float),
        "te_x": np.asarray(le_x_fun(y_pos) + chord_fun(y_pos), dtype=float),
        "section_y": y_anchor,
        "section_le_x": le_x_anchor,
        "section_te_x": le_x_anchor + chord_anchor,
        "section_chord": chord_anchor,
    }
    full["outer_mask"] = np.abs(full["upper_y"][:, 0]) >= float(derived["spanwise_y_m"]["s4"])
    return full


def _set_axes(ax) -> None:  # noqa: ANN001
    ax.set_xlim(0.0, 50.0)
    ax.set_ylim(-42.0, 42.0)
    ax.set_zlim(-4.0, 5.5)
    ax.set_box_aspect((50.0, 84.0, 12.0))
    ax.set_xlabel("x [m]", labelpad=8)
    ax.set_ylabel("span y [m]", labelpad=8)
    ax.set_zlabel("z [m]", labelpad=6)
    ax.grid(False)
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.fill = False
        axis.pane.set_edgecolor("#d7dde6")


def _plot_planform_lines(ax, surface: dict[str, np.ndarray], color: str, active: bool) -> None:  # noqa: ANN001
    z_floor = -3.35
    lw = 2.8 if active else 1.3
    alpha = 0.95 if active else 0.35
    for sign in (-1.0, 1.0):
        yy = sign * surface["y_pos"]
        ax.plot(surface["le_x"], yy, np.full_like(yy, z_floor), color="#15708d", lw=lw, alpha=alpha)
        ax.plot(surface["te_x"], yy, np.full_like(yy, z_floor), color="#c4502d", lw=lw, alpha=alpha)
    if active:
        for sign in (-1.0, 1.0):
            ax.scatter(
                surface["section_le_x"],
                sign * surface["section_y"],
                np.full_like(surface["section_y"], z_floor),
                color=color,
                s=28,
                depthshade=False,
            )
            ax.scatter(
                surface["section_te_x"],
                sign * surface["section_y"],
                np.full_like(surface["section_y"], z_floor),
                color=color,
                s=18,
                marker="s",
                depthshade=False,
            )


def _plot_highlight(ax, surface: dict[str, np.ndarray], parameter: ParameterVisual) -> None:  # noqa: ANN001
    color = GROUP_COLORS[parameter.group]
    if parameter.group == "Planform":
        _plot_planform_lines(ax, surface, color=color, active=True)
        return

    _plot_planform_lines(ax, surface, color=color, active=False)
    mask = surface["outer_mask"]
    if np.any(mask):
        ax.plot_surface(
            surface["upper_x"][mask, :],
            surface["upper_y"][mask, :],
            surface["upper_z"][mask, :],
            color=color,
            alpha=0.42,
            linewidth=0.0,
            antialiased=True,
            shade=True,
        )
        ax.plot_surface(
            surface["lower_x"][mask, :],
            surface["lower_y"][mask, :],
            surface["lower_z"][mask, :],
            color=color,
            alpha=0.30,
            linewidth=0.0,
            antialiased=True,
            shade=True,
        )

    if parameter.group == "Outer twist":
        for sign in (-1.0, 1.0):
            quarter_x = surface["le_x"] + 0.25 * (surface["te_x"] - surface["le_x"])
            y_line = sign * surface["y_pos"]
            outer = np.abs(y_line) >= float(surface["section_y"][5])
            ax.plot(
                quarter_x[outer],
                y_line[outer],
                np.full(np.count_nonzero(outer), 4.65),
                color=color,
                lw=3.0,
                alpha=0.95,
            )

    if parameter.group == "Outer thickness":
        section_lookup = {"thickness_s4": "s4", "thickness_s5": "s5"}
        section = section_lookup.get(parameter.key)
        if section is not None:
            idx = SECTION_ORDER.index(section)
            yy = float(surface["section_y"][idx])
            for sign in (-1.0, 1.0):
                row = int(np.argmin(np.abs(surface["upper_y"][:, 0] - sign * yy)))
                ax.plot(
                    surface["upper_x"][row, :],
                    surface["upper_y"][row, :],
                    surface["upper_z"][row, :],
                    color=color,
                    lw=3.2,
                )
                ax.plot(
                    surface["lower_x"][row, :],
                    surface["lower_y"][row, :],
                    surface["lower_z"][row, :],
                    color=color,
                    lw=3.2,
                )


def render_frame(
    parameter: ParameterVisual,
    phase: float,
    frame_index: int,
    total_frames: int,
    *,
    n_span: int,
    n_chord: int,
    dpi: int,
) -> Image.Image:
    design, value = design_with_parameter(parameter, phase)
    surface = build_visual_surface(design, n_span=n_span, n_chord=n_chord)
    baseline, lower, upper = cta.CTA_CFD_JUNE_14_VARIABLE_INFO[parameter.key]
    color = GROUP_COLORS[parameter.group]

    fig = plt.figure(figsize=(12.8, 7.2), dpi=dpi, facecolor="#f7f8fa")
    ax = fig.add_subplot(111, projection="3d")
    fig.subplots_adjust(left=0.02, right=0.98, bottom=0.02, top=0.92)

    ax.plot_surface(
        surface["upper_x"],
        surface["upper_y"],
        surface["upper_z"],
        color="#dbe2ea",
        alpha=0.88,
        linewidth=0.0,
        antialiased=True,
        shade=True,
    )
    ax.plot_surface(
        surface["lower_x"],
        surface["lower_y"],
        surface["lower_z"],
        color="#9aa4af",
        alpha=0.82,
        linewidth=0.0,
        antialiased=True,
        shade=True,
    )
    _plot_highlight(ax, surface, parameter)
    _set_axes(ax)
    ax.view_init(elev=23.0, azim=-58.0 + 0.28 * frame_index)

    fig.text(0.035, 0.925, "CTA/BWB 14-parameter CFD campaign", fontsize=22, weight="bold", color="#111827")
    fig.text(0.035, 0.885, f"{parameter.group}  |  {parameter.label}", fontsize=16, weight="bold", color=color)
    fig.text(0.035, 0.852, parameter.description, fontsize=12, color="#374151")
    fig.text(
        0.035,
        0.815,
        f"{parameter.key}: value {value:.4g}   baseline {baseline:.4g}   bounds [{lower:.4g}, {upper:.4g}]",
        fontsize=11,
        color="#111827",
    )
    fig.text(
        0.735,
        0.055,
        f"{frame_index + 1:03d}/{total_frames:03d}",
        fontsize=10,
        color="#6b7280",
    )

    buffer = BytesIO()
    fig.savefig(buffer, format="png", dpi=dpi)
    plt.close(fig)
    buffer.seek(0)
    return Image.open(buffer).convert("P", palette=Image.Palette.ADAPTIVE)


def build_gif(
    output_path: Path,
    *,
    frames_per_parameter: int,
    n_span: int,
    n_chord: int,
    dpi: int,
    duration_ms: int,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    phases = np.linspace(0.0, 1.0, frames_per_parameter, endpoint=False)
    total_frames = frames_per_parameter * len(PARAMETERS)
    frames: list[Image.Image] = []

    frame_index = 0
    for parameter in PARAMETERS:
        for phase in phases:
            frames.append(
                render_frame(
                    parameter,
                    float(phase),
                    frame_index,
                    total_frames,
                    n_span=n_span,
                    n_chord=n_chord,
                    dpi=dpi,
                ),
            )
            frame_index += 1

    frames[0].save(
        output_path,
        save_all=True,
        append_images=frames[1:],
        optimize=True,
        duration=duration_ms,
        loop=0,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a 3D promotional GIF showing the CTA 14-variable CFD design space.",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--frames-per-parameter", type=int, default=9)
    parser.add_argument("--span-stations", type=int, default=46)
    parser.add_argument("--chord-stations", type=int, default=64)
    parser.add_argument("--dpi", type=int, default=110)
    parser.add_argument("--duration-ms", type=int, default=95)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_path = args.output_dir / "cta_june14_parameters_3d.gif"
    build_gif(
        output_path,
        frames_per_parameter=args.frames_per_parameter,
        n_span=args.span_stations,
        n_chord=args.chord_stations,
        dpi=args.dpi,
        duration_ms=args.duration_ms,
    )
    print(output_path)


if __name__ == "__main__":
    main()
