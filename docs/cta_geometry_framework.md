# CTA/BWB Geometry Framework In MADS

This document describes the active CTA/BWB parametrization in MADS after synchronizing it with the corrected geometry in `BWB/parametrization/CTA`.

The objective is a reusable global BWB geometry framework, not a CTA-specific shortcut. CTA is only one concrete baseline case built from the generic MADS geometry core and connected to solver adapters such as pyGeo/IGES and DUST.

## Active Status

The active CTA geometry in MADS is built from the official 9-section contract:

```text
S0, S1, S1a, S2, S3, S4, S4a, S4b, S5
```

The responsibility chain is:

```text
CTA design variables
-> derived and fixed geometry parameters
-> generic MADS Wing / Section / AirfoilCST objects
-> reusable spanwise laws plus CTA case configuration
-> PreparedGeometry / resolved mesh
-> exporters/adapters: IGES, pyGeo, DUST
-> analysis results and campaign tables
```

There is no DUST-specific logic in `cta_geometry.py`, `cta_geom_doe.py` or `cta_laws.py`. DUST integration lives in the aerodynamics solver layer.

## Main Files

| File | Responsibility |
| --- | --- |
| `examples/cta_geometry.py` | Official CTA baseline, 14-variable CFD design space, MADS sections, resolved mesh and IGES export. |
| `examples/cta_geom_doe.py` | Solver-independent GEMSEO geometry DoE/baseline check. Maps design variables to derived geometry parameters. |
| `examples/cta_dust_doe.py` | GEMSEO 14-variable campaign connected to DUST through the aerodynamic adapter. |
| `src/multiads/solvers/synthesis/spanwise_laws.py` | Generic reusable laws: spanwise interpolation, LE/TE planform, CST profile interpolation and thickness scaling. |
| `src/multiads/cases/cta_laws.py` | CTA-specific configuration built on top of the generic geometry laws. |
| `src/multiads/solvers/synthesis/geometry_lib.py` | Generic geometry core: `PreparedGeometry`, section resolution, planform, resolved mesh and metrics. |
| `src/multiads/utilities/pygeo_export.py` | pyGeo/IGES adapter from resolved geometry. |
| `src/multiads/solvers/aerodynamics/dust_lib.py` | DUST primitives, basic mesh writer, resolved-geometry runner and GEMSEO DUST discipline. |
| `src/multiads/utilities/campaign_export.py` | Generic CSV/XLSX/JSON campaign export utilities. It contains no CTA or DUST-specific logic. |

## Active Design Variables

The current CFD campaign design space has 14 variables. In GEMSEO/MADS they use the `cta_` prefix.

| Variable | Baseline | Bounds | Meaning |
| --- | ---: | ---: | --- |
| `delta_c0_m` | 0.0 | [-2.203, 1.797] | Chord increment applied to S0, S1 and S2. |
| `delta_c3_m` | 0.0 | [-0.927, 2.073] | Chord increment applied to S3. |
| `delta_c5_m` | 0.0 | [0.0, 1.0] | Chord increment applied to S5. |
| `taper_ratio_midwing` | 0.5578 | [0.45, 0.60] | Defines `C4 = TRw * C3`. |
| `rspan_midwing` | 0.142 | [0.13, 0.21] | Relative S4 span position from S3 over `Bw`. |
| `span_wing_m` | 31.4585 | [28.0, 35.0] | Spanwise distance from S3 to S5. |
| `sweep_midwing_deg` | 34.6 | [15.0, 45.0] | S1 sweep measured on the 50% chord line from S3 to S4. |
| `sweep_outwing_deg` | 24.7 | [22.0, 40.0] | S2 sweep measured on the 25% chord line from S4 to S5. |
| `twist_s4_deg` | 0.483 | [-3.517, 4.483] | S4 twist about the 25% chord point. |
| `twist_s4a_deg` | 1.381 | [-2.619, 5.381] | S4a twist about the 25% chord point. |
| `twist_s4b_deg` | 2.279 | [-1.721, 6.279] | S4b twist about the 25% chord point. |
| `twist_s5_deg` | 3.177 | [-0.823, 7.177] | S5 twist about the 25% chord point. |
| `thickness_s4` | 0.1027 | [0.08, 0.16] | Target maximum relative thickness `t/c` for S4. |
| `thickness_s5` | 0.095 | [0.08, 0.13] | Target maximum relative thickness `t/c` for S5. |

The outer-wing twist bounds are currently defined as `baseline +/- 4 deg`. The baseline values are:

```text
S4  = 0.483 deg
S5  = 3.177 deg
S4a = (2*S4 + S5)/3 = 1.381 deg
S4b = (S4 + 2*S5)/3 = 2.279 deg
```

Therefore:

```text
twist_s4_deg  = [0.483 - 4, 0.483 + 4]
twist_s4a_deg = [1.381 - 4, 1.381 + 4]
twist_s4b_deg = [2.279 - 4, 2.279 + 4]
twist_s5_deg  = [3.177 - 4, 3.177 + 4]
```

Geometry variables are design variables, derived parameters or fixed parameters. They are not inner variables. Inner variables are quantities produced by the analysis, for example loads, coefficients, packaging checks and failure flags.

Volume and packaging are feasibility checks or constraints. They are not optimization objectives.

## Section Contract

| Section | Planform | Z / twist | Airfoil |
| --- | --- | --- | --- |
| `S0` | `C0 = 41.203 + delta_c0_m` | fixed `zle0`, `zte0` in the CFD design space | independent 6U/6L CST |
| `S1` | `C1 = 39.246 + delta_c0_m` | smooth slave between S0 and S3 | slave of S0 |
| `S1a` | derived chord between S1 and S2 | smooth slave between S0 and S3 | independent 6U/6L CST in the full geometry contract |
| `S2` | `C2 = 30.139 + delta_c0_m` | smooth slave between S0 and S3 | independent 6U/6L CST |
| `S3` | `C3 = 13.927 + delta_c3_m` | fixed `zle3`, `zte3` in the CFD design space | independent 6U/6L CST |
| `S4` | `C4 = taper_ratio_midwing * C3` | twist about 25% chord | camber CST plus target thickness |
| `S4a` | `(2*C4 + C5)/3` | twist about 25% chord | derived from S4/S5 |
| `S4b` | `(C4 + 2*C5)/3` | twist about 25% chord | derived from S4/S5 |
| `S5` | `C5 = 0.8 + delta_c5_m` | twist about 25% chord | camber CST plus target thickness |

The absolute trailing-edge thickness is preserved when the chord or outer-wing thickness variables change. This is a manufacturing constraint.

## Generic Geometry Laws

Reusable primitives live in `multiads.solvers.synthesis.spanwise_laws`:

```text
ControlPointPlanformLawConfig
build_le_te_planform_law
build_scalar_interpolant
build_positive_interpolant
build_profile_interpolants
evaluate_profile_interpolants
scale_thickness_preserving_trailing_edge
cosine_spacing
```

These functions do not know about CTA. Any BWB case with a set of sections can reuse them to build `field(y)` laws and then package each resolved station as a `ResolvedStation`.

`cta_laws.py` only fixes the CTA configuration:

```text
CTA_SECTION_ORDER
CTA_OUTER_LINEAR_START_INDEX
CTA_PLANFORM_LAW_CONFIG
CTA_NUM_BASE_STATIONS
CTA_NUM_ROOT_BLEND_STATIONS
```

### Planform

`cta_laws.py` calls `build_le_te_planform_law()` with the official section points and `CTA_PLANFORM_LAW_CONFIG`. The entry into the symmetry plane is forced to have zero slope by using `symmetry_blend_y = 1.9`.

Key points:

- LE uses `le_spline_bridge=(1, 7)` and a linear outer segment from S4.
- TE keeps the S3-S4 transition segment straight with `te_exact_segments=(4,)`.
- The outer TE is linear from S4 to S5 through S4a and S4b.
- There is no artificial frontal lower-surface guide and no plot-only post-transform.

### Airfoils

Profiles are interpolated through CST coefficients, as in BWB, not by directly interpolating already resolved ordinates. This prevents intermediate sections from creating undesired local bulges.

The outer-wing thickness transform preserves:

- the camberline,
- the absolute trailing-edge thickness,
- the requested maximum `t/c` measured over the full resolved profile.

### Z / Twist

S0-S3 use a quintic C2 transition of `zle` and `zte`; central-body twist is derived from `(zte - zle) / chord`. S4-S5 are governed by twist about the 25% chord point.

### Discretization

The resolved baseline geometry uses:

```text
n_span = 71
n_chord = 241
quads per surface = 16800
```

The resolved mesh is the solver-independent geometry authority:

```text
outputs/CTA_case/export/geometry/cta_resolved_mesh.npz
```

The IGES is an inspection/export adapter:

```text
outputs/CTA_case/export/geometry/cta.igs
```

## DUST Adapter

DUST does not receive the fine CAD-like `cta_resolved_mesh.npz` directly. For the panel method, MADS derives a solver-ready aerodynamic `basic` mesh in:

```text
multiads/solvers/aerodynamics/dust_lib.py
```

The generic resampler `write_basic_two_skin_mesh_from_resolved_npz()`:

- reads `upper_vertices`, `lower_vertices`, `span_stations` and `x_airfoil`;
- resamples the geometry onto a uniform or curvature-adaptive solver mesh;
- concentrates panels near the nose, LE/TE and high-curvature regions;
- uses fewer panels where the geometry is almost linear;
- keeps upper and lower skins as separate surfaces;
- collapses the trailing edge to a common line so DUST can detect the wake;
- keeps the leading edge closed by default with `leading_edge_opening_m = 0.0`;
- can numerically open the leading edge only if explicitly requested for a numerical experiment.

The adapter `multiads.solvers.aerodynamics.dust_lib.ResolvedGeometryDustDiscipline`:

- receives an already resolved `PreparedGeometry`;
- writes a copy of the resolved mesh used by the DUST case;
- generates the `basic` mesh with adaptive spacing;
- builds a MADS `Wing` with `WingOptions(mesh_file=..., mesh_file_type="basic")`;
- delegates execution to the generic `multiads.solvers.aerodynamics.dust.DUST` solver;
- uses the existing `dust_lib.py` classes to configure `dust_pre`, `dust`, `dust_post`, `PostLoads` and `PostViz`;
- normalizes `CL`, `CD` and `CM` with the resolved sample area and MAC;
- writes `cta_dust_result.json` inside each case directory.

There is no second DUST geometry module. The bridge between resolved geometry and the existing DUST solver lives in `dust_lib.py`.

DUST executable resolution is centralized in `multiads.solvers.aerodynamics.dust_lib.dust_executable()` and follows this priority:

```text
1. explicit script option: --dust-bin-dir <DUST_INSTALL_DIR>/bin
2. environment variable: MADS_DUST_BIN_DIR
3. legacy environment variable: CTA_DUST_BIN_DIR
4. system PATH: dust_pre, dust, dust_post
```

For reproducible runs on the office/HPC machine, prefer either:

```bash
export MADS_DUST_BIN_DIR=<DUST_INSTALL_DIR>/bin
```

or pass the path explicitly in each command:

```bash
--dust-bin-dir <DUST_INSTALL_DIR>/bin
```

For the current CTA baseline, the DUST mesh uses:

```text
n_span_stations = 49
n_chord_stations = 45
span_spacing = curvature
chord_spacing = curvature
leading_edge_opening_m = 0.0
mirror_span = True
```

This does not modify the source geometry or the IGES. It only creates a derived aerodynamic discretization for DUST and wake generation.

Integral DUST loads are read in the case reference axes. In the current campaign setup, drag is stored and normalized in these reference axes without an additional wind-axis projection, because the immediate goal is to stabilize the wake and compare geometry trends consistently.

## Usage

The commands below assume the working directory is `BWB/` and that the shared environment is available as `./.venv`. DUST can be found either through `MADS_DUST_BIN_DIR`, through `--dust-bin-dir`, or through the system `PATH`.

Recommended shell setup:

```bash
export MADS_DUST_BIN_DIR=<DUST_INSTALL_DIR>/bin
```

For convenience, the same runtime variables can be configured from the MADS
repository root with:

```bash
source scripts/setup_cta_env.sh <DUST_INSTALL_DIR>/bin
```

This sets `MADS_ROOT`, `PYTHONDONTWRITEBYTECODE`, `MPLCONFIGDIR`,
`PYTHONPATH`, `MADS_DUST_BIN_DIR`, and `I_MPI_FABRICS=shm` for local/serial
geometry and DUST checks.

Generate the baseline CTA geometry and IGES:

```bash
PYTHONDONTWRITEBYTECODE=1 MPLCONFIGDIR=/private/tmp/mads_mpl PYTHONPATH=../MADS/src:../MADS/examples ./.venv/bin/python ../MADS/examples/cta_geometry.py
```

Main outputs:

```text
MADS/outputs/CTA_case/export/geometry/cta_resolved_mesh.npz
MADS/outputs/CTA_case/export/geometry/cta.igs
MADS/outputs/CTA_case/export/geometry/station_airfoils/
MADS/outputs/CTA_case/export/geometry/summary.json
```

Run the geometry-only baseline DoE/check:

```bash
PYTHONDONTWRITEBYTECODE=1 MPLCONFIGDIR=/private/tmp/mads_mpl PYTHONPATH=../MADS/src:../MADS/examples ./.venv/bin/python ../MADS/examples/cta_geom_doe.py
```

Main outputs:

```text
MADS/outputs/CTA_case/doe_geometry/cta_geom_doe_dataset.csv
MADS/outputs/CTA_case/doe_geometry/cta_geom_doe_dataset_flat.csv
MADS/outputs/CTA_case/doe_geometry/cta_geom_doe_design_space.csv
MADS/outputs/CTA_case/doe_geometry/cta_geom_doe_validation_summary.json
MADS/outputs/CTA_case/doe_geometry/cta_geom_doe_manifest.json
```

Run the current CTA DUST baseline at `AoA = 3 deg` with 70 time steps:

```bash
PYTHONDONTWRITEBYTECODE=1 MPLCONFIGDIR=/private/tmp/mads_mpl PYTHONPATH=../MADS/src:../MADS/examples ./.venv/bin/python ../MADS/examples/cta_dust_doe.py   --baseline-only   --alpha-deg 3.0   --n-steps 70   --dust-bin-dir "$MADS_DUST_BIN_DIR"   --output-dir ../MADS/outputs/CTA_case/doe_dust_70_baseline_aoa03
```

The latest verified 70-step baseline produced:

```text
CL  = 0.243677
CD  = 0.018102
CM  = -0.253814
L/D = 13.461696
```

Main outputs:

```text
MADS/outputs/CTA_case/doe_dust_70_baseline_aoa03/cta_dust_doe_dataset_flat.csv
MADS/outputs/CTA_case/doe_dust_70_baseline_aoa03/cta_dust_doe_results.xlsx
MADS/outputs/CTA_case/doe_dust_70_baseline_aoa03/cases/run/
```

ParaView outputs from the last time step:

```text
MADS/outputs/CTA_case/doe_dust_70_baseline_aoa03/cases/run/post/cta_aoa_03_visualization-0070.vtu
MADS/outputs/CTA_case/doe_dust_70_baseline_aoa03/cases/run/post/cta_aoa_03_visualization_wpan-0070.vtu
MADS/outputs/CTA_case/doe_dust_70_baseline_aoa03/cases/run/post/cta_aoa_03_visualization_wpart-0070.vtu
```

Open `visualization-0070.vtu` for the surface and optionally load `visualization_wpan-0070.vtu` and `visualization_wpart-0070.vtu` for wake inspection.

## GEMSEO + DUST CFD Campaign

The first CFD campaign uses the 14-variable design space:

```text
8 planform variables:
  cta_delta_c0_m, cta_delta_c3_m, cta_delta_c5_m,
  cta_taper_ratio_midwing, cta_rspan_midwing, cta_span_wing_m,
  cta_sweep_midwing_deg, cta_sweep_outwing_deg

4 outer-wing twist variables:
  cta_twist_s4_deg, cta_twist_s4a_deg,
  cta_twist_s4b_deg, cta_twist_s5_deg

2 outer-wing thickness variables:
  cta_thickness_s4, cta_thickness_s5
```

Run a 20-sample LHS campaign at `AoA = 3 deg`:

```bash
PYTHONDONTWRITEBYTECODE=1 MPLCONFIGDIR=/private/tmp/mads_mpl PYTHONPATH=../MADS/src:../MADS/examples ./.venv/bin/python ../MADS/examples/cta_dust_doe.py   --n-samples 20   --alpha-deg 3.0   --dust-bin-dir "$MADS_DUST_BIN_DIR"   --output-dir ../MADS/outputs/CTA_case/doe_dust_20
```

By default, DUST uses `n_steps = 80` in `cta_dust_doe.py`. The physical DUST run directory is overwritten:

```text
MADS/outputs/CTA_case/doe_dust/cases/run/
```

This avoids storing a full DUST run directory for every sample. The campaign history is kept in CSV/XLSX tables. Use `--store-case-directories` only when one directory per sample is needed for debugging.

Run only the baseline within the same GEMSEO/DUST workflow:

```bash
PYTHONDONTWRITEBYTECODE=1 MPLCONFIGDIR=/private/tmp/mads_mpl PYTHONPATH=../MADS/src:../MADS/examples ./.venv/bin/python ../MADS/examples/cta_dust_doe.py   --baseline-only   --alpha-deg 3.0   --dust-bin-dir "$MADS_DUST_BIN_DIR"
```

Main command-line options:

```text
--n-samples N                 number of DOE samples
--algo LHS                    GEMSEO DOE algorithm
--alpha-deg 3.0               DUST angle of attack
--mesh-span-stations 49       half-span stations before mirroring
--mesh-chord-stations 45      chordwise points per skin
--span-spacing curvature      adaptive spanwise mesh
--chord-spacing curvature     adaptive chordwise mesh
--leading-edge-opening-m 0.0  closed leading edge
--n-steps 80                  DUST time steps per sample
--store-case-directories      keep sample_XXXX instead of overwriting cases/run
--no-vtk                      skip ParaView output to reduce I/O
--fail-fast                   stop if a sample fails
```

Campaign outputs:

```text
MADS/outputs/CTA_case/doe_dust/cta_dust_doe_dataset.csv
MADS/outputs/CTA_case/doe_dust/cta_dust_doe_dataset_flat.csv
MADS/outputs/CTA_case/doe_dust/cta_dust_doe_results.xlsx
MADS/outputs/CTA_case/doe_dust/cta_dust_doe_design_space.csv
MADS/outputs/CTA_case/doe_dust/cta_dust_doe_manifest.json
MADS/outputs/CTA_case/doe_dust/cases/run/
```

The Excel workbook contains:

```text
results       inputs, geometry checks, packaging checks, forces, coefficients and L/D
design_space  baseline, lower_bound and upper_bound for the 14 variables
analysis_settings summary of Environment, Options, OutputOptions and storage policy
```

The DUST connection does not create a parallel configuration class. The example `cta_dust_doe.py` directly builds:

```text
Environment    altitude, Mach/speed and alpha
Options        executables, time, wake, particles, folders and threads
OutputOptions  loads and visualization/post-processing
WingOptions    panel method, basic mesh, TE projection and load averaging window
DustMeshSettings only controls remeshing of the resolved geometry
```

In `results`, the columns `outputs.cta_box_*` are packaging indicator-surface checks:

```text
*_fits                  1 if the local packaging surface fits, 0 otherwise
*_margin_m              minimum local margin
*_footprint_area_m2     indicator-surface footprint area
*_clearance_volume_m3   approximate integrated margin = mean_margin * footprint_area
```

The current packaging file defines upper/lower indicator surfaces, not closed 3D boxes with their own nominal volume. Therefore the recorded box volume is a clearance volume, not an optimization target volume.

The folder `cases/run` contains only the latest DUST simulation:

```text
geometry/cta_resolved_mesh.npz
geometry/cta_basic_rr.dat
geometry/cta_basic_ee.dat
Output/
post/
cta_dust_result.json
```

GEMSEO requires a technical objective even for DOE scenarios. Here `cta_dust_failure_code` is minimized only to register execution status (`0 = OK`, `1 = DUST/geometry failure`). It is not an aerodynamic objective.

## Outer-Wing Thickness

The variables `cta_thickness_s4` and `cta_thickness_s5` are target maximum relative thicknesses `t/c` for the outer wing:

```text
S4  = cta_thickness_s4
S4a = (2*S4 + S5)/3
S4b = (S4 + 2*S5)/3
S5  = cta_thickness_s5
```

During profile resolution, MADS computes the local camberline, separates the thickness distribution and rescales the full section profile so that the maximum `t/c` over the whole profile matches the target. CTA does not apply an additional `x/c` window for measuring outer-wing thickness.

The absolute trailing-edge height is preserved from the baseline contract. The law interpolates trailing-edge thickness in meters and converts it locally to a relative CST value:

```text
te_over_c(y) = te_abs_m(y) / chord_m(y)
```

Therefore, changing chord or target outer-wing `t/c` does not freely scale the trailing edge. This is a manufacturing requirement, not a design variable.

## Consistency With BWB

The MADS implementation has been compared against:

```text
BWB/outputs/cta_geometry_official_9/export/cta_resolved_mesh.npz
```

Latest comparison status:

```text
span_stations: same size, max difference < 5e-10 m
x_airfoil: identical
upper/lower/camber: same shape (71, 241, 3)
maximum upper/lower surface difference: about 1.5e-3 m
maximum camber difference: about 1e-9 m
```

The residual surface difference is localized near the interpolated trailing edge of one interior station. It does not change topology, section count, airfoil discretization or the aerodynamic mesh contract.

## Creating Another BWB Case In MADS

For a new BWB case:

1. Define the official section table.
2. Separate design variables, derived parameters and fixed parameters.
3. Create MADS `Section` and `AirfoilCST` objects for each section.
4. Reuse the generic laws from `multiads.solvers.synthesis.spanwise_laws`.
5. Resolve to `PreparedGeometry`.
6. Export through generic adapters such as resolved NPZ, pyGeo/IGES or DUST.

Do not place aerodynamic solver logic inside the geometry parametrization.

Recommended pattern:

```python
from multiads.solvers.synthesis.spanwise_laws import (
    ControlPointPlanformLawConfig,
    build_le_te_planform_law,
    build_profile_interpolants,
    build_scalar_interpolant,
    evaluate_profile_interpolants,
    scale_thickness_preserving_trailing_edge,
)

MY_PLANFORM_CONFIG = ControlPointPlanformLawConfig(
    continuity_order=2,
    symmetry_blend_y=2.0,
    le_linear_start_index=3,
    te_linear_start_index=3,
)

def build_my_case_planform(section_y, section_le_x, section_chord):
    return build_le_te_planform_law(
        section_y=section_y,
        section_le_x=section_le_x,
        section_chord=section_chord,
        config=MY_PLANFORM_CONFIG,
    )

def build_my_case_resolved_station_factory(*, component, anchor_sections, config):
    # Build field(y) laws here: chord, LE, twist, CST, thickness...
    # Return a resolver function: sample_y -> tuple[ResolvedStation, ...].
    ...
```

The concrete case should only decide which sections exist, which variables control them and which configuration is passed to the generic laws.
