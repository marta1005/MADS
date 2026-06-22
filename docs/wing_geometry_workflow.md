# Wing Geometry — Workflow

```mermaid
flowchart TD
    %% ── USER INPUT ──────────────────────────────────────────────
    subgraph INPUT["User input"]
        S["Section\n(name, eta, chord, twist,\nLE x/z, airfoil CST)"]
        SP["Span\n(leading_edge_mode,\nfield_laws, field_scopes)"]
        GS["WingGeometrySpec\n(chordwise_points, spanwise_stations,\nspanwise_law, field_laws/scopes,\niges_path, …)"]
        W["Wing\n(sections, spans,\ncase_name, symmetry, mirror\ngeometry = WingGeometrySpec)"]
    end

    S -->|"1..N\n(root → tip)"| W
    SP -->|"N-1 spans"| W
    GS -->|"wing.geometry"| W

    %% ── DISCIPLINE ──────────────────────────────────────────────
    subgraph DISC["Geometry discipline  —  multiads.disciplines.geometry"]
        direction TB
        DEX["Geometry(components=[wing], solver=CSTGeometrySolver())\n.execute(input_data)"]

        subgraph INTERNALS["internal steps"]
            direction TB
            CFG["build_geometry_config\n────────────────\nWingGeometryConfig\n  SamplingConfig\n  InterpolationConfig\n  ExportConfig"]
            RG["resolve_geometry\n────────────────\n① build span axis\n② sample spanwise y positions\n③ interpolate fields per segment\n   chord · twist · LE x/z · airfoil\n④ evaluate CST → ResolvedStation\n   x/c · z/c · xyz_m"]
            PG["PreparedGeometry\n────────────────\nanchor_stations  (N)\nresolved_stations (M)\nenvelope"]
            CFG --> RG --> PG
        end

        DEX --> INTERNALS
    end

    W -->|"components=[wing]"| DISC

    %% ── CUSTOM FACTORY (advanced) ───────────────────────────────
    CF["Custom factories\n(WingGeometrySpec)\n────────────────\nresolved_station_factory\nspan_station_factory\n\ne.g. build_cta_resolved_station_factory"] -.->|"override default\nstation builder"| RG

    %% ── OUTPUTS ─────────────────────────────────────────────────
    PG --> MET["Metrics\ncompute_geometry_metrics\n────────────────\nspan · area · volume\nroot/tip chord · MAC\n→ wing.geometry_metrics"]

    PG --> MESH["Surface mesh\nbuild_resolved_surface_mesh\n────────────────\nupper · lower · camber vertices\nquad connectivity\n→ .npz  (DUST / post-processing)"]

    PG --> PLT["Plots\ngeometry_plots.*\n────────────────\nplot_planform\nplot_front_view\nplot_sections\nplot_geometry\n★ no pyGeo required"]

    PG --> IGS["IGES export\nexport_geometry_to_pygeo\n────────────────\n① write station .dat files\n② build pyGeo liftingSurface\n③ writeIGES → .igs\n→ wing.export_state\n⚠ requires pyGeo + pyspline"]

    %% ── STYLES ──────────────────────────────────────────────────
    classDef user   fill:#dbeafe,stroke:#3b82f6,color:#1e3a5f
    classDef core   fill:#dcfce7,stroke:#16a34a,color:#14532d
    classDef out    fill:#fef9c3,stroke:#ca8a04,color:#713f12
    classDef iges   fill:#fce7f3,stroke:#db2777,color:#831843
    classDef disc   fill:#ede9fe,stroke:#7c3aed,color:#3b0764
    classDef adv    fill:#ffedd5,stroke:#ea580c,color:#7c2d12

    class S,SP,GS,W user
    class CFG,RG,PG core
    class MET,MESH,PLT out
    class IGS iges
    class DISC,DEX,INTERNALS disc
    class CF adv
```

---

## Step-by-step summary

| Step | What happens | Key object |
|---|---|---|
| **1. Define sections** | Each `Section` fixes one spanwise anchor: chord, twist, LE position, airfoil shape (CST) | `Section` |
| **2. Define spans** | Each `Span` says how to interpolate between the two sections it connects (law, scope, LE mode) | `Span` |
| **3. Attach spec** | `WingGeometrySpec` on `wing.geometry` controls sampling density, default laws, and export paths | `WingGeometrySpec` |
| **4. Execute discipline** | `Geometry(components=[wing], solver=CSTGeometrySolver()).execute(input_data)` — this is the single public entry point | `Geometry` |
| **4a. (internal) Build config** | `build_geometry_config` translates the spec into internal sub-configs | `WingGeometryConfig` |
| **4b. (internal) Resolve** | `resolve_geometry` samples the span, interpolates every field per segment, evaluates each CST section in 3-D | `PreparedGeometry` |
| **6a. Metrics** | `compute_geometry_metrics` integrates span, area, MAC, volume from the resolved stations | `GeometryMetricSet` |
| **6b. Mesh** | `build_resolved_surface_mesh` assembles structured upper/lower/camber grids + quad connectivity | `ResolvedSurfaceMesh` |
| **6c. Plots** | `geometry_plots.*` draws planform, front view, sections — matplotlib only, no pyGeo | PNG / interactive |
| **6d. IGES** | `export_geometry_to_pygeo` feeds the stations into pyGeo and calls `writeIGES` | `.igs` file |

Steps 6a–6d are all independent — you can use any combination.

The **Geometry discipline** (`multiads.disciplines.geometry.Geometry`) is the single public entry point — it wraps config build, resolution and export into a GEMSEO-compatible block so the same wing can be driven by an optimiser or a DOE loop without changing the geometry code. Steps 6a–6d are all independent — you can use any combination after execution.
