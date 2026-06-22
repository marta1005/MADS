# Wing Geometry System

How to define a wing, specify interpolation behaviour, and export the resolved mesh and IGES file.

---

## Core concepts

| Concept | Class | Role |
|---|---|---|
| Cross-section shape | `Section` | One airfoil at a spanwise position |
| Span segment | `Span` | Transition rules between two neighbouring sections |
| Wing | `Wing` | Ordered list of sections + spans |
| Geometry spec | `WingGeometrySpec` | Sampling, interpolation and export options |

The solver (`geometry_lib.build_geometry`) ingests a `Wing` and produces a fully resolved mesh — a dense array of interpolated cross-sections — plus an IGES surface suitable for CFD meshing.

---

## 1. Sections

A `Section` defines one anchor cross-section.

```python
from multiads.assembly import Section, Airfoil

root = Section(
    name="root",
    eta=0.0,                    # normalised semi-span position [0, 1]
    chord_m=10.0,
    twist_deg=0.0,
    leading_edge_x_m=0.0,
    leading_edge_z_m=0.0,
    airfoil=Airfoil.from_file("naca0012.dat"),
)
```

Any number of sections can be defined. The solver interpolates between consecutive pairs.

---

## 2. Spans

A `Span` specifies the interpolation behaviour between two consecutive sections.

```python
from multiads.assembly import Span

inner_span = Span(
    name="inner",
    leading_edge_mode="section_positions",  # how the leading-edge x/z are treated
    field_laws={                             # per-field interpolation scheme
        "chord":          "pchip",
        "twist":          "pchip",
        "leading_edge_x": "pchip",
        "leading_edge_z": "pchip",
        "airfoil":        "pchip",
    },
    field_scopes={                           # "global" = use all sections as knots
        "chord":          "global",          # "local"  = use only the two bounding sections
        "twist":          "global",
        "leading_edge_x": "global",
        "leading_edge_z": "global",
        "airfoil":        "global",
    },
)
```

### `leading_edge_mode`

Controls how the leading-edge x/z coordinates are resolved when placing a station:

| Value | Behaviour |
|---|---|
| `"section_positions"` (default) | Leading-edge x/z are read directly from each resolved station's interpolated values |
| `"span_angles"` | Leading-edge position is derived from a prescribed sweep angle law |

### `field_laws`

Interpolation scheme for each field. Supported values: `"pchip"`, `"linear"`, `"akima"`.

If a field is not listed, the wing-level default (`WingGeometrySpec.spanwise_law`) is used.

### `field_scopes`

Which sections act as knots for a given field:

- `"global"` — all sections in the wing are knot points (smooth, globally-consistent curve)
- `"local"` — only the two sections bounding the current span are knots (piecewise, segment-local)

---

## 3. Wing

Assemble sections and spans into a wing. The span list must have exactly `len(sections) - 1` entries, ordered root→tip.

```python
from multiads.assembly import Wing, WingGeometrySpec

wing = Wing(
    name="my_wing",
    sections=[root, mid, tip],
    spans=[inner_span, outer_span],
    case_name="MY_CASE",
    symmetry=True,      # mirror to produce a full-span wing in the solver output
    mirror=False,
    geometry=WingGeometrySpec(
        chordwise_points=201,
        spanwise_stations=51,
        station_distribution="le_te",
        include_anchor_stations=True,
        spanwise_law="pchip",
        section_law="pchip",
        # per-field overrides (merged with span-level field_laws)
        field_laws={
            "chord":          "pchip",
            "twist":          "pchip",
            "leading_edge_x": "pchip",
            "leading_edge_z": "pchip",
            "airfoil":        "pchip",
        },
        field_scopes={
            "chord":          "global",
            "twist":          "global",
            "leading_edge_x": "global",
            "leading_edge_z": "global",
            "airfoil":        "global",
        },
    ),
)
```

### `WingGeometrySpec` reference

**Sampling**

| Field | Default | Description |
|---|---|---|
| `chordwise_points` | `201` | Number of points per cross-section |
| `spanwise_stations` | `51` | Number of resolved spanwise stations |
| `station_distribution` | `"le_te"` | Spacing law along the span (`"le_te"`, `"cosine"`, `"uniform"`) |
| `include_anchor_stations` | `True` | Force the exact section positions into the station list |

**Interpolation**

| Field | Default | Description |
|---|---|---|
| `spanwise_law` | `"pchip"` | Default scheme for spanwise field interpolation |
| `section_law` | `"pchip"` | Scheme used to interpolate inside a resolved cross-section |
| `field_laws` | `{}` | Per-field overrides (merged with span-level settings) |
| `field_scopes` | `{}` | Per-field scope overrides (`"global"` / `"local"`) |
| `resolved_station_factory` | `None` | Optional callable — replaces the built-in station resolver (advanced use) |
| `span_station_factory` | `None` | Optional callable — replaces the built-in station array builder (advanced use) |

**Export**

| Field | Default | Description |
|---|---|---|
| `out_dir` | `None` | Directory for per-station airfoil `.dat` files |
| `iges_path` | `None` | Output path for the IGES file |
| `meshing_iges_path` | `None` | Secondary IGES path for a meshing-specific variant |
| `frame_only_iges_path` | `None` | Secondary IGES path with structural frame only |
| `export_all_resolved_stations` | `False` | Write a `.dat` file for every resolved station (not just anchors) |
| `blunt_trailing_edge` | `True` | Close the trailing edge with a blunt cap |
| `symmetric` | `False` | Write symmetric (full-span) stations to the IGES |
| `tip_style` | `"rounded"` | Tip closure style (`"rounded"`, `"flat"`) |
| `section_curve_n_ctl` | `18` | Number of B-spline control points per section curve |
| `k_span` | `4` | Spanwise B-spline degree |

---

## 4. Solving

Resolution is driven by the `Geometry` discipline, which wraps the internal `resolve_geometry` call and writes results back onto the `Wing` object.

```python
from multiads.disciplines.geometry import Geometry, CSTGeometrySolver

disc = Geometry(components=[wing], solver=CSTGeometrySolver())
input_data = {v.name: v.value_np.copy() for v in disc.variables}
disc.execute(input_data=input_data)

resolved_wing = disc.components[0]

print(f"span  = {resolved_wing.span_m:.3f} m")
print(f"area  = {resolved_wing.planform_area_m2:.3f} m²")
print(f"MAC   = {resolved_wing.mean_aerodynamic_chord_m:.3f} m")
```

After execution the `Wing` object is populated with:
- `wing.geometry_state` — `PreparedGeometry` with all resolved stations
- `wing.geometry_metrics` — `GeometryMetricSet` (span, area, MAC, …)

---

## 5. IGES export (requires pyGeo)

The IGES export is **optional** — if pyGeo/pyspline are not installed the rest of the geometry pipeline (resolution, mesh, plots) still works. A `RuntimeError` is raised only when `export_geometry_to_pygeo` is actually called.

Set the export options in `WingGeometrySpec`, then run the `Geometry` discipline:

```python
from multiads.solvers.synthesis import build_resolved_surface_mesh, write_resolved_surface_mesh_npz
from multiads.disciplines.geometry import Geometry, CSTGeometrySolver

# 1. attach export options to the wing spec
wing.geometry.iges_path            = "outputs/wing.igs"
wing.geometry.blunt_trailing_edge  = True
wing.geometry.tip_style            = "rounded"
wing.geometry.section_curve_n_ctl  = 18
wing.geometry.k_span               = 4

# 2. resolve + export via the Geometry discipline
disc = Geometry(components=[wing], solver=CSTGeometrySolver())
input_data = {v.name: v.value_np.copy() for v in disc.variables}
disc.execute(input_data=input_data)

resolved_wing = disc.components[0]

# 3. optionally save the structured mesh (.npz) for DUST or post-processing
mesh = build_resolved_surface_mesh(resolved_wing.geometry_state)
write_resolved_surface_mesh_npz("outputs/wing_mesh.npz", mesh)

print(f"IGES: {resolved_wing.export_state.iges_path}")
```

`WingGeometrySpec` IGES-relevant fields:

| Field | Default | Effect |
|---|---|---|
| `iges_path` | `None` | Output path for the main IGES file |
| `blunt_trailing_edge` | `True` | Close TE with a blunt cap |
| `symmetric` | `False` | Mirror stations to produce a full-span IGES |
| `tip_style` | `"rounded"` | Tip closure (`"rounded"` / `"flat"`) |
| `section_curve_n_ctl` | `18` | B-spline control points per section curve |
| `k_span` | `4` | Spanwise B-spline degree |
| `export_all_resolved_stations` | `False` | Write `.dat` for every station, not just anchors |
| `out_dir` | `None` | Directory for per-station airfoil `.dat` files |

---

## 6. Plotting the geometry

`multiads.utilities.geometry_plots` provides ready-made plot functions that only need numpy and matplotlib — no pyGeo required.

```python
from multiads.utilities.geometry_plots import (
    plot_planform,    # top-down view
    plot_front_view,  # front view (span vs z-envelope)
    plot_sections,    # airfoil profiles at selected stations
    plot_geometry,    # all three views + distributions in one figure
)
```

### All views at once

```python
import matplotlib.pyplot as plt

fig = plot_geometry(resolved_wing.geometry_state, symmetric=True)
plt.savefig("outputs/wing_geometry.png", dpi=150)
plt.show()
```

### Individual views

```python
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 3, figsize=(16, 5))

plot_planform(geometry_state, ax=axes[0], symmetric=True)
plot_front_view(geometry_state, ax=axes[1], symmetric=True)
plot_sections(geometry_state, ax=axes[2], stations=9)   # 9 evenly-spaced sections

plt.tight_layout()
plt.show()
```

### Section selection

```python
# 7 evenly spaced (default)
plot_sections(geometry_state)

# specific count
plot_sections(geometry_state, stations=5)

# specific indices
plot_sections(geometry_state, stations=[0, 10, 25, 40, -1])
```

---

## 7. Full minimal example

```python
from multiads.assembly import Section, Airfoil, Span, Wing, WingGeometrySpec
from multiads.disciplines.geometry import Geometry, CSTGeometrySolver

root = Section("root", eta=0.0, chord_m=12.0, twist_deg=0.0,
               leading_edge_x_m=0.0, leading_edge_z_m=0.0,
               airfoil=Airfoil.naca("0012"))

tip = Section("tip", eta=1.0, chord_m=2.0, twist_deg=-3.0,
              leading_edge_x_m=8.0, leading_edge_z_m=0.0,
              airfoil=Airfoil.naca("0009"))

span = Span("main", leading_edge_mode="section_positions",
            field_laws={"chord": "pchip", "twist": "pchip",
                        "leading_edge_x": "pchip", "airfoil": "pchip"},
            field_scopes={"chord": "global", "twist": "global",
                          "leading_edge_x": "global", "airfoil": "global"})

wing = Wing(
    name="simple_wing",
    sections=[root, tip],
    spans=[span],
    case_name="DEMO",
    symmetry=True,
    geometry=WingGeometrySpec(
        chordwise_points=101,
        spanwise_stations=31,
        iges_path="outputs/simple_wing.igs",
    ),
)

disc = Geometry(components=[wing], solver=CSTGeometrySolver())
disc.execute(input_data={v.name: v.value_np.copy() for v in disc.variables})
resolved = disc.components[0]
```

---

## 8. Advanced: custom station resolver (CTA pattern)

For cases that require non-standard spanwise sampling or section construction (e.g. thickness constraints, multi-body coupling), the station resolver and station array can be replaced via `WingGeometrySpec.resolved_station_factory` and `span_station_factory`:

```python
from multiads.cases.cta_laws import (
    build_cta_resolved_station_factory,
    build_cta_span_station_array,
)

wing.geometry = WingGeometrySpec(
    ...
    resolved_station_factory=build_cta_resolved_station_factory,
    span_station_factory=build_cta_span_station_array,
)
```

These callables receive the resolved interpolation config and return, respectively, a factory that builds one cross-section at a given span position, and a function that produces the full array of span positions to resolve.

This is an escape hatch for specialised geometry rules. For standard wings, leave both as `None`.
