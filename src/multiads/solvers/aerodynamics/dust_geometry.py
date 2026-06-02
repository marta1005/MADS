"""DUST adapters for solver-independent resolved geometry meshes."""

from __future__ import annotations

import json
import os
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
from gemseo.core.discipline import Discipline

from multiads.assembly import Environment, Wing as MadsWing
from multiads.scenario import InnerVariableFloat
from multiads.solvers.aerodynamics.dust import DUST
from multiads.solvers.aerodynamics import dust_lib as dl
from multiads.solvers.synthesis.geometry_lib import (
    PreparedGeometry,
    build_resolved_surface_mesh,
    write_resolved_surface_mesh_npz,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence

    from numpy.typing import NDArray

    from multiads.scenario import BaseVariable


@dataclass(slots=True)
class DustMeshSettings:
    """Settings for the DUST aerodynamic mesh derived from a resolved surface."""

    n_span_stations: int = 49
    n_chord_stations: int = 45
    span_spacing: str = "curvature"
    chord_spacing: str = "curvature"
    span_curvature_weight: float = 5.0
    chord_curvature_weight: float = 4.0
    chord_endpoint_weight: float = 0.60
    leading_edge_opening_m: float = 0.0
    leading_edge_opening_chord_fraction: float | None = None
    leading_edge_opening_extent: float = 0.12
    collapse_trailing_edge: bool = True
    mirror_span: bool = True


@dataclass(slots=True)
class DustRunSettings:
    """Settings for one steady CTA/BWB DUST run."""

    mach: float = 0.8
    altitude_m: float = 40000.0 * 0.3048
    disa_k: float = 0.0
    alpha_deg: float = 3.0
    n_steps: int = 80
    dt: float = 0.005
    n_threads: int = 6
    n_wake_particles: int = 100000
    particles_box_min: tuple[float, float, float] = (-80.0, -90.0, -80.0)
    particles_box_max: tuple[float, float, float] = (380.0, 90.0, 80.0)
    loads_average_window: int = 20
    write_visualization: bool = True
    force_axes: str = "reference"
    dust_bin_dir: str | None = None


@dataclass(slots=True)
class DustCaseResult:
    """Scalar result and artifact metadata from one DUST run."""

    alpha_deg: float
    mach: float
    altitude_ft: float
    disa_k: float
    speed_mps: float
    rho_kg_m3: float
    q_pa: float
    s_ref_m2: float
    c_ref_m: float
    fx_reference_n: float
    fy_reference_n: float
    fz_reference_n: float
    mx_reference_nm: float
    my_reference_nm: float
    mz_reference_nm: float
    lift_n: float
    drag_n: float
    side_n: float
    cl: float
    cd: float
    cy: float
    cm: float
    ld: float
    run_dir: str
    mesh_info: dict[str, Any]

    def to_flat_dict(self) -> dict[str, Any]:
        data = asdict(self)
        mesh_info = data.pop("mesh_info")
        data.update({f"mesh_{key}": value for key, value in mesh_info.items()})
        return data


def _dust_executable(name: str, dust_bin_dir: str | None = None) -> Path:
    if dust_bin_dir:
        return Path(dust_bin_dir) / name
    if env_bin_dir := os.environ.get("CTA_DUST_BIN_DIR"):
        return Path(env_bin_dir) / name
    return Path(name)


def _case_tag(alpha_deg: float) -> str:
    alpha_int = int(round(float(alpha_deg)))
    return f"aoa_{alpha_int:02d}" if alpha_int >= 0 else f"aoa_m{abs(alpha_int):02d}"


def _normalization_loads(
    force_reference: NDArray[np.float64],
    moment_reference: NDArray[np.float64],
    q_inf: float,
    s_ref_m2: float,
    c_ref_m: float,
) -> dict[str, float]:
    drag_n = -float(force_reference[0])
    side_n = float(force_reference[1])
    lift_n = float(force_reference[2])
    return {
        "lift_n": lift_n,
        "drag_n": drag_n,
        "side_n": side_n,
        "cl": lift_n / (q_inf * s_ref_m2),
        "cd": drag_n / (q_inf * s_ref_m2),
        "cy": side_n / (q_inf * s_ref_m2),
        "cm": float(moment_reference[1]) / (q_inf * s_ref_m2 * c_ref_m),
        "ld": lift_n / drag_n if abs(drag_n) > 1.0e-14 else float("nan"),
    }


def run_dust_case_from_resolved_npz(
    mesh_npz: str | Path,
    run_dir: str | Path,
    *,
    s_ref_m2: float,
    c_ref_m: float,
    mesh_settings: DustMeshSettings | None = None,
    run_settings: DustRunSettings | None = None,
    clean_run_dir: bool = True,
) -> DustCaseResult:
    """Run one DUST panel case from a solver-independent resolved mesh file."""

    mesh_settings = mesh_settings or DustMeshSettings()
    run_settings = run_settings or DustRunSettings()
    run_path = Path(run_dir)
    if clean_run_dir and run_path.exists():
        shutil.rmtree(run_path)
    (run_path / "geometry").mkdir(parents=True, exist_ok=True)
    (run_path / "Output").mkdir(parents=True, exist_ok=True)
    (run_path / "post").mkdir(parents=True, exist_ok=True)

    mesh_prefix = Path("geometry") / "cta_basic_"
    mesh_info = dl.write_basic_two_skin_mesh_from_resolved_npz(
        Path(mesh_npz),
        run_path / mesh_prefix,
        n_span_stations=mesh_settings.n_span_stations,
        n_chord_stations=mesh_settings.n_chord_stations,
        span_spacing=mesh_settings.span_spacing,
        chord_spacing=mesh_settings.chord_spacing,
        span_curvature_weight=mesh_settings.span_curvature_weight,
        chord_curvature_weight=mesh_settings.chord_curvature_weight,
        chord_endpoint_weight=mesh_settings.chord_endpoint_weight,
        leading_edge_opening_m=mesh_settings.leading_edge_opening_m,
        leading_edge_opening_chord_fraction=mesh_settings.leading_edge_opening_chord_fraction,
        leading_edge_opening_extent=mesh_settings.leading_edge_opening_extent,
        collapse_trailing_edge=mesh_settings.collapse_trailing_edge,
        mirror_span=mesh_settings.mirror_span,
    )

    env_probe = Environment(name="env", height=run_settings.altitude_m, speed=1.0)
    speed = run_settings.mach * float(env_probe.sound_speed)
    env = Environment(
        name="env",
        height=run_settings.altitude_m,
        speed=speed,
        alpha=run_settings.alpha_deg,
    )

    n_steps = int(run_settings.n_steps)
    loads_start = max(1, n_steps - int(run_settings.loads_average_window))
    wing = MadsWing(
        name="cta_wing",
        sections=[],
        spans=[],
        symmetry=not mesh_settings.mirror_span,
        options=[
            dl.WingOptions(
                discretization_method=dl.WingMethod.PANELS,
                panel_type=dl.WingPanelType.UNIFORM,
                num_panels=1,
                mesh_file=mesh_prefix,
                mesh_file_type="basic",
                inner_product_te=0.5,
                tol_se_wing=1.0e-3,
                proj_te=True,
                proj_te_dir="parallel",
                proj_te_vector=env.velocity / np.linalg.norm(env.velocity),
                output_options=dl.OutputOptions(
                    compute_loads=True,
                    loads_start=loads_start,
                    loads_end=n_steps,
                    loads_step=1,
                    loads_avg=True,
                    loads_reference="0",
                ),
            ),
        ],
    )

    options = dl.Options(
        name=f"cta_{_case_tag(run_settings.alpha_deg)}",
        dust_pre=_dust_executable("dust_pre", run_settings.dust_bin_dir),
        dust=_dust_executable("dust", run_settings.dust_bin_dir),
        dust_post=_dust_executable("dust_post", run_settings.dust_bin_dir),
        run_directory=run_path,
        output_dir=Path("Output"),
        post_dir=Path("post"),
        keep_run_directory=True,
        t_start=0.0,
        t_end=n_steps * float(run_settings.dt),
        dt=float(run_settings.dt),
        dt_out=float(run_settings.dt),
        output_start=True,
        n_threads=int(run_settings.n_threads),
        n_wake_panels=n_steps,
        n_wake_particles=int(run_settings.n_wake_particles),
        particles_box_min=np.asarray(run_settings.particles_box_min, dtype=float),
        particles_box_max=np.asarray(run_settings.particles_box_max, dtype=float),
        penetration_avoidance=False,
        output_options=dl.OutputOptions(
            visualization=run_settings.write_visualization,
            viz_start=n_steps,
            viz_end=n_steps,
            viz_step=1,
            viz_fmt="vtk",
            viz_wake=True,
            viz_separate_wake=True,
            viz_variables=["cp", "vorticity_vector", "velocity"],
        ),
    )
    dust_solver = DUST(options=options)
    components = dust_solver.parse_variables([env, wing])
    dust_solver.run(components)
    dust_solver.compute_output()
    if dust_solver.outputs_map is None:
        msg = "DUST did not expose output variables."
        raise RuntimeError(msg)

    force_reference = np.asarray(
        dust_solver.outputs_map["cta_wing.force"].value,
        dtype=float,
    )
    moment_reference = np.asarray(
        dust_solver.outputs_map["cta_wing.moment"].value,
        dtype=float,
    )
    if not np.all(np.isfinite(force_reference)) or not np.all(np.isfinite(moment_reference)):
        msg = f"DUST returned non-finite loads for alpha={run_settings.alpha_deg:g} deg."
        raise RuntimeError(msg)

    q_inf = 0.5 * float(env.density) * speed**2
    loads_norm = _normalization_loads(
        force_reference,
        moment_reference,
        q_inf,
        float(s_ref_m2),
        float(c_ref_m),
    )
    result = DustCaseResult(
        alpha_deg=float(run_settings.alpha_deg),
        mach=float(run_settings.mach),
        altitude_ft=float(run_settings.altitude_m / 0.3048),
        disa_k=float(run_settings.disa_k),
        speed_mps=float(speed),
        rho_kg_m3=float(env.density),
        q_pa=float(q_inf),
        s_ref_m2=float(s_ref_m2),
        c_ref_m=float(c_ref_m),
        fx_reference_n=float(force_reference[0]),
        fy_reference_n=float(force_reference[1]),
        fz_reference_n=float(force_reference[2]),
        mx_reference_nm=float(moment_reference[0]),
        my_reference_nm=float(moment_reference[1]),
        mz_reference_nm=float(moment_reference[2]),
        run_dir=str(run_path),
        mesh_info=mesh_info,
        **loads_norm,
    )
    (run_path / "cta_dust_result.json").write_text(
        json.dumps(result.to_flat_dict(), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return result


def run_dust_case_from_prepared_geometry(
    geometry: PreparedGeometry,
    run_dir: str | Path,
    *,
    s_ref_m2: float,
    c_ref_m: float,
    mesh_settings: DustMeshSettings | None = None,
    run_settings: DustRunSettings | None = None,
) -> DustCaseResult:
    """Build the resolved surface mesh from geometry and run one DUST case."""

    run_path = Path(run_dir)
    if run_path.exists():
        shutil.rmtree(run_path)
    (run_path / "geometry").mkdir(parents=True, exist_ok=True)
    resolved_mesh_path = run_path / "geometry" / "cta_resolved_mesh.npz"
    mesh = build_resolved_surface_mesh(geometry)
    write_resolved_surface_mesh_npz(resolved_mesh_path, mesh)
    return run_dust_case_from_resolved_npz(
        resolved_mesh_path,
        run_path,
        s_ref_m2=s_ref_m2,
        c_ref_m=c_ref_m,
        mesh_settings=mesh_settings,
        run_settings=run_settings,
        clean_run_dir=False,
    )


class ResolvedGeometryDustDiscipline(Discipline):
    """GEMSEO discipline running DUST from an already resolved geometry state.

    The discipline depends on geometry metric outputs only to enforce execution
    order in GEMSEO. The actual surface comes from ``geometry_provider`` so the
    DoE script remains a high-level orchestration layer.
    """

    def __init__(
        self,
        *,
        name: str,
        geometry_provider: Callable[[], PreparedGeometry | None],
        metric_inputs: Sequence[BaseVariable],
        output_dir: str | Path,
        run_settings: DustRunSettings | None = None,
        mesh_settings: DustMeshSettings | None = None,
        reference_area_name: str = "cta_wing.planform_area_m2",
        reference_chord_name: str = "cta_wing.mean_aerodynamic_chord_m",
        fail_fast: bool = False,
        reuse_run_directory: bool = False,
        run_directory_name: str = "run",
    ) -> None:
        super().__init__(name)
        self.geometry_provider = geometry_provider
        self.output_dir = Path(output_dir)
        self.run_settings = run_settings or DustRunSettings()
        self.mesh_settings = mesh_settings or DustMeshSettings()
        self.reference_area_name = reference_area_name
        self.reference_chord_name = reference_chord_name
        self.fail_fast = bool(fail_fast)
        self.reuse_run_directory = bool(reuse_run_directory)
        self.run_directory_name = str(run_directory_name)
        self.case_index = 0

        self.input_grammar.update_from_data({var.name: var.value_np for var in metric_inputs})
        self.output_variables = [
            InnerVariableFloat("cta_dust_success", 1.0),
            InnerVariableFloat("cta_dust_failure_code", 0.0),
            InnerVariableFloat("cta_dust_cl", 0.0),
            InnerVariableFloat("cta_dust_cd", 0.0),
            InnerVariableFloat("cta_dust_cm", 0.0),
            InnerVariableFloat("cta_dust_cy", 0.0),
            InnerVariableFloat("cta_dust_ld", 0.0),
            InnerVariableFloat("cta_dust_lift_n", 0.0),
            InnerVariableFloat("cta_dust_drag_n", 0.0),
            InnerVariableFloat("cta_dust_side_n", 0.0),
            InnerVariableFloat("cta_dust_fx_reference_n", 0.0),
            InnerVariableFloat("cta_dust_fy_reference_n", 0.0),
            InnerVariableFloat("cta_dust_fz_reference_n", 0.0),
            InnerVariableFloat("cta_dust_mx_reference_nm", 0.0),
            InnerVariableFloat("cta_dust_my_reference_nm", 0.0),
            InnerVariableFloat("cta_dust_mz_reference_nm", 0.0),
        ]
        self.output_grammar.update_from_data(
            {var.name: var.value_np for var in self.output_variables},
        )

    def _run(self, input_data: Mapping[str, NDArray[np.float64]]) -> dict[str, NDArray[np.float64]]:
        if self.reuse_run_directory:
            case_dir = self.output_dir / self.run_directory_name
            if case_dir.exists():
                shutil.rmtree(case_dir)
        else:
            case_dir = self.output_dir / f"sample_{self.case_index:04d}_{_case_tag(self.run_settings.alpha_deg)}"
        self.case_index += 1

        try:
            geometry = self.geometry_provider()
            if geometry is None:
                raise RuntimeError("No resolved geometry state is available for DUST.")

            s_ref = float(np.ravel(input_data[self.reference_area_name])[0])
            c_ref = float(np.ravel(input_data[self.reference_chord_name])[0])
            result = run_dust_case_from_prepared_geometry(
                geometry,
                case_dir,
                s_ref_m2=s_ref,
                c_ref_m=c_ref,
                mesh_settings=self.mesh_settings,
                run_settings=self.run_settings,
            )
            values = {
                "cta_dust_success": 1.0,
                "cta_dust_failure_code": 0.0,
                "cta_dust_cl": result.cl,
                "cta_dust_cd": result.cd,
                "cta_dust_cm": result.cm,
                "cta_dust_cy": result.cy,
                "cta_dust_ld": result.ld,
                "cta_dust_lift_n": result.lift_n,
                "cta_dust_drag_n": result.drag_n,
                "cta_dust_side_n": result.side_n,
                "cta_dust_fx_reference_n": result.fx_reference_n,
                "cta_dust_fy_reference_n": result.fy_reference_n,
                "cta_dust_fz_reference_n": result.fz_reference_n,
                "cta_dust_mx_reference_nm": result.mx_reference_nm,
                "cta_dust_my_reference_nm": result.my_reference_nm,
                "cta_dust_mz_reference_nm": result.mz_reference_nm,
            }
        except Exception as exc:  # noqa: BLE001
            if self.fail_fast:
                raise
            case_dir.mkdir(parents=True, exist_ok=True)
            (case_dir / "failure.json").write_text(
                json.dumps({"error": str(exc)}, indent=2, sort_keys=True),
                encoding="utf-8",
            )
            values = {
                "cta_dust_success": 0.0,
                "cta_dust_failure_code": 1.0,
                "cta_dust_cl": 0.0,
                "cta_dust_cd": 0.0,
                "cta_dust_cm": 0.0,
                "cta_dust_cy": 0.0,
                "cta_dust_ld": 0.0,
                "cta_dust_lift_n": 0.0,
                "cta_dust_drag_n": 0.0,
                "cta_dust_side_n": 0.0,
                "cta_dust_fx_reference_n": 0.0,
                "cta_dust_fy_reference_n": 0.0,
                "cta_dust_fz_reference_n": 0.0,
                "cta_dust_mx_reference_nm": 0.0,
                "cta_dust_my_reference_nm": 0.0,
                "cta_dust_mz_reference_nm": 0.0,
            }

        return {name: np.atleast_1d(value) for name, value in values.items()}
