# CTA/BWB DUST Lifting-Line Baseline

This note documents the DUST lifting-line setup prepared for the CTA/BWB
baseline. The goal is not to run a DOE yet, but to obtain a traceable baseline
configuration that can be diagnosed and iterated.

## Why an Equivalent LL Geometry Is Needed

DUST lifting line uses `el_type = l` and represents a one-dimensional lifting
line, not the full 3D BWB volume. The CTA centerbody has nose volume, thickness,
packaging constraints, and a broad body-like planform; sending the full BWB
directly to lifting line is numerically fragile and physically inappropriate.

The implemented adapter therefore supports:

- `lifting_line_geometry_mode` / `lifting_line_geometry_variant`: builds a
  solver-specific equivalent lifting line from the effective lifting region.
- `full_bwb_anchor_equivalent`: includes the whole semispan from `S0` to `S5`,
  but selects named/anchor stations and well-spaced intermediate stations. This
  avoids passing the dense CAD helper stations near the centerbody nose directly
  to LL.
- `full_bwb_equivalent`: includes the full semispan by selecting from the
  resolved station list. This is useful as a failure diagnostic, but it can
  select very small centerbody span segments and should not be used for DOE.
- `outer_only`: starts from the outer wing. This is useful as a numerical
  control case, but it is not considered representative of the full CTA/BWB.
- `transition_outer`: starts near the transition wing and keeps more CTA/BWB
  lifting extent for diagnostics.
- `calibrated_outer`: same geometry idea as `outer_only`, but intended for a
  later CD calibration against panel-method references.
- `rectangular`: a high-aspect-ratio control wing used to check that DUST LL,
  `.c81` polars and force signs can produce positive drag on a simple case.
- `lifting_line_effective_start_y_m`: first real CTA station used as lifting
  geometry, defaulting to the transition wing region.
- synthetic root station at `y = 0`: used only for partial-span variants. The
  full-BWB variants already start from `S0`.
- `mesh_flat = T/F`: written explicitly to the DUST component input for LL.
- per-section `airfoil_table = *.c81`: generated from the polar provider.
- explicit quarter-chord placement: DUST ignores `reference_chord_fraction` for
  `el_type = l`, so the adapter shifts `starting_point` and the span geometry to
  the 25% chord line before writing the DUST component.

The original CTA geometry is not modified. The simplification only exists in
the DUST LL adapter.

## Solver Presets

The solver library exposes three LL presets:

| Preset | Purpose | Main Settings |
|---|---|---|
| `conservative` | Robust debugging | `ll_max_iter = 500`, `ll_tol = 1e-6`, `ll_damp = 0.10`, stall regularisation, small artificial viscosity |
| `nominal` | Balanced run | standard tolerance, default damping, stall regularisation |
| `fast` | Later campaigns after validation | fewer iterations, looser tolerance |

All individual `ll_*` parameters can still be overridden from the command line.

## Diagnostic Command

Run the compact diagnostic suite:

```bash
python examples/cta_dust_lifting_line_convergence.py \
  --dust-bin-dir "$MADS_DUST_BIN_DIR" \
  --diagnostic-suite \
  --steps 70 \
  --meshes 21x1 \
  --write-vtk
```

This runs:

- AoA `-3/0/+3 deg`,
- `full_bwb_anchor_equivalent`, `full_bwb_equivalent`, `outer_only`,
  `transition_outer`, and `rectangular`,
- sanitized NeuralFoil polars and synthetic positive-CD polars,
- `mesh_flat = T`,
- `type_span = cosine`,
- conservative LL settings.

Run a sensitivity sweep manually:

```bash
python examples/cta_dust_lifting_line_convergence.py \
  --dust-bin-dir "$MADS_DUST_BIN_DIR" \
  --steps 70 \
  --meshes 21x1 \
  --geometry-modes outer_only,transition_outer \
  --mesh-flat-options T,F \
  --ll-presets conservative \
  --span-types uniform,cosine \
  --ll-damps 0.05,0.10,0.20 \
  --ll-loads-avl-options F,T \
  --polar-sources neuralfoil,synthetic \
  --sanitize-polars \
  --write-vtk
```

For a single robust debug case:

```bash
python examples/cta_dust_lifting_line_convergence.py \
  --dust-bin-dir "$MADS_DUST_BIN_DIR" \
  --steps 70 \
  --meshes 21x1 \
  --geometry-modes outer_only \
  --mesh-flat-options T \
  --ll-presets conservative \
  --span-types cosine \
  --polar-sources synthetic \
  --write-vtk
```

Outputs are written under:

```text
outputs/CTA_case/convergence/lifting_line/
```

The script writes:

- scalar results CSV/XLSX,
- force-history CSV,
- optional VTU visualization files,
- DUST input files in the case `run` directory,
- spanwise LL postprocess files when available.
- reference-axis and wind-axis force coefficients:
  - `CL_reference`, `CD_reference`,
  - `CL_wind`, `CD_wind`,
  - raw `Fx`, `Fy`, `Fz`.
- reconstructed drag:
  - `CD_profile`,
  - `CD_induced = CL^2 / (pi * AR_eff * e_eff)`,
  - `CD_reconstructed`,
  - `drag_source`.
- acceptance flags:
  - `ll_case_accepted`,
  - `ll_case_rejection_reason`,
  - `dcl_dalpha_positive_flag` when `-3/+3 deg` are available.

## Force Axes and Drag Diagnostics

DUST postprocessed loads are treated as already expressed in wind axes for the
CTA workflow. No additional AoA projection is applied in MADS.

The convention used in the postprocessor is:

```text
D = -Fx
Y = Fy
L = Fz
```

For backward compatibility, the result table still contains both
`CL_reference/CD_reference` and `CL_wind/CD_wind`, but in this CTA DUST workflow
they are aliases of the same direct DUST loads. They are not double-projected.

## Polar Sanitization

The LL workflow can sanitize the `.c81` input polars:

- `Cd >= Cd_min`, default `0.006`,
- non-finite values are replaced,
- `Cl` and `Cm` are clipped to broad physical limits,
- the CSV records clipping counts and min/max values.

Use:

```bash
--sanitize-polars --polar-cd-min 0.006
```

or disable it with:

```bash
--no-sanitize-polars
```

## Diagnostics Written to DUST

The generated DUST component file includes:

- `mesh_file_type = parametric`
- `el_type = l`
- `mesh_flat = T/F`
- `airfoil_table = <section>.c81`
- span discretization through `nelem_span` and `type_span`

The generated `dust_post.in` requests:

- integral loads,
- optional visualization,
- sectional loads with `lifting_line_data = T`.

## Current Baseline Observation

The equivalent LL baseline now has two full-geometry diagnostic variants:

| Variant | Meaning | Current Status |
|---|---|---|
| `full_bwb_equivalent` | Full semispan using resolved stations | Runs can become non-physical because the centerbody has many tiny `dy` stations with chord near 40 m |
| `full_bwb_anchor_equivalent` | Full semispan using anchor stations and minimum spacing | Runs without the tiny-span singularity, but direct DUST LL drag remains negative and the force history is still not stable enough for DOE |
| `outer_only` | Outer wing numerical control | More stable, but not accepted as the full-aircraft CTA/BWB representation |
| `rectangular` | High-aspect-ratio sanity check | Confirms the DUST LL executable, `.c81` format and force-axis projection can produce positive drag on a LL-appropriate geometry |

For `full_bwb_anchor_equivalent` at AoA = 3 deg with sanitized NeuralFoil
polars, `mesh_flat = T`, `type_span = cosine`, conservative LL preset and
70 time steps:

| Quantity | Value |
|---|---:|
| selected half-span stations | 19 |
| minimum span segment | 1.755 m |
| full equivalent area | 843.99 m2 |
| effective AR | 7.39 |
| `CL_wind_mean` | 0.1568 |
| `CD_wind_mean` | -0.5134 |
| `CL_wind_std` | 1.6536 |
| `CD_wind_std` | 0.1563 |
| `CD_reconstructed` | 0.1795 |

The direct DUST LL drag is therefore not accepted. `CD_reconstructed` is kept
only as a diagnostic quantity; it must not hide a negative `CD_wind` result.

This means the negative drag is not only a generic DUST executable or `.c81`
format failure. It is linked to the CTA equivalent LL representation and/or the
way DUST LL computes/reports direct drag for that geometry.

This means:

- LL numerical execution is working for full and partial CTA diagnostic cases.
- The `full_bwb_anchor_equivalent` geometry avoids the previous tiny-span
  centerbody singularity.
- The setup is diagnostic enough to inspect spanwise `Cl`, `Cd`, `Cm`, local
  alpha and velocities.
- The LL direct drag is not yet accepted for full CTA/BWB DOE because all tested
  full-geometry sensitivity cases still have negative `CD_wind`.

Recommended next step: use this diagnostic matrix to decide whether the DOE LL
database can use an explicitly calibrated full-geometry LL equivalent for
`CL/CM` with separately reconstructed drag, or whether the campaign must return
to panel/VLM-like surface methods for trustworthy direct drag.

## Recommended LL DOE Gate

Before any large DOE with LL, require:

- `geometry_variant = full_bwb_anchor_equivalent`,
- `polar_source = neuralfoil`,
- `--sanitize-polars`,
- direct `CD_wind > 0`; reconstructed drag is diagnostic only until approved,
- final-window `CL_wind_mean > 0` at positive AoA,
- sufficiently small final-window force oscillation,
- positive `dCL/dAoA` over the `-3/0/+3 deg` diagnostic sweep,
- failed cases kept in the dataset with the DUST error message.
