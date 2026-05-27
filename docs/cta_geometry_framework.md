# CTA/BWB Geometry Framework En MADS

Este documento describe el estado activo de la parametrizacion CTA/BWB en
MADS despues de sincronizarla con la geometria corregida en
`BWB/parametrization/CTA`.

## Estado Activo

La geometria CTA activa en MADS ya no es la version antigua de 6 secciones con
helpers. Ahora se construye desde la tabla oficial de 9 secciones:

```text
S0, S1, S1a, S2, S3, S4, S4a, S4b, S5
```

La cadena de responsabilidades es:

```text
CTA design variables
-> parametros derivados/fijos de geometria
-> Wing/Section/AirfoilCST genericos de MADS
-> cta_laws.py para leyes CTA de interpolacion
-> PreparedGeometry / resolved mesh
-> exporters/adapters: IGES, pyGeo, DUST futuro
```

No hay logica DUST dentro de `cta_geometry.py`, `cta_geom_doe.py` ni
`cta_laws.py`.

## Archivos

| Archivo | Responsabilidad |
| --- | --- |
| `examples/cta_geometry.py` | Baseline CTA oficial, 14 variables CFD, secciones MADS, export mesh/IGES. |
| `examples/cta_geom_doe.py` | DoE/baseline GEMSEO sin DUST. Mapea variables de diseno a parametros derivados. |
| `src/multiads/cases/cta_laws.py` | Leyes CTA: planform, CST spanwise, Z/twist, sampling. |
| `src/multiads/solvers/synthesis/geometry_lib.py` | Core generico: `PreparedGeometry`, planform, mesh resuelto, metricas. |
| `src/multiads/utilities/pygeo_export.py` | Adaptador pyGeo/IGES desde geometria resuelta. |
| `src/multiads/solvers/aerodynamics/dust*` | Lugar previsto para DUST. |

## Variables De Diseno Activas

El espacio activo inicial es el control CFD de 14 variables. En MADS los
nombres de variables llevan prefijo `cta_` en GEMSEO.

| Variable | Baseline | Bounds | Descripcion |
| --- | ---: | ---: | --- |
| `delta_c0_m` | 0.0 | [-2.203, 1.797] | Incremento de cuerda en S0, S1 y S2. |
| `delta_c3_m` | 0.0 | [-0.927, 2.073] | Incremento de cuerda en S3. |
| `delta_c5_m` | 0.0 | [0.0, 1.0] | Incremento de cuerda en S5. |
| `taper_ratio_midwing` | 0.5578 | [0.45, 0.60] | Define `C4 = TRw * C3`. |
| `rspan_midwing` | 0.142 | [0.13, 0.21] | Posicion relativa de S4 desde S3 sobre `Bw`. |
| `span_wing_m` | 31.4585 | [28.0, 35.0] | Distancia spanwise desde S3 hasta S5. |
| `sweep_midwing_deg` | 34.6 | [15.0, 45.0] | Sweep S1 medido en la linea de 50% cuerda S3-S4. |
| `sweep_outwing_deg` | 24.7 | [22.0, 33.0] | Sweep S2 medido en la linea de 25% cuerda S4-S5. |
| `twist_s4_deg` | 0.483 | [-3.517, 4.483] | Twist de S4 alrededor del 25% de cuerda. |
| `twist_s4a_deg` | 1.381 | [-2.619, 5.381] | Twist de S4a alrededor del 25% de cuerda. |
| `twist_s4b_deg` | 2.279 | [-1.721, 6.279] | Twist de S4b alrededor del 25% de cuerda. |
| `twist_s5_deg` | 3.177 | [-0.823, 7.177] | Twist de S5 alrededor del 25% de cuerda. |
| `thickness_s4` | 0.1027 | [0.08, 0.16] | Espesor maximo objetivo de S4. |
| `thickness_s5` | 0.095 | [0.08, 0.13] | Espesor maximo objetivo de S5. |

El volumen y packaging no son objetivos de optimizacion. Son checks/constraints
de factibilidad.

## Reglas De Seccion

| Seccion | Planform | Z / twist | Perfil |
| --- | --- | --- | --- |
| `S0` | `C0 = 41.203 + delta_c0_m` | `zle0`, `zte0` fijos en el espacio CFD | 6U/6L CST independiente |
| `S1` | `C1 = 39.246 + delta_c0_m` | derivado suave entre S0 y S3 | slave de S0 |
| `S1a` | cuerda derivada entre S1 y S2 | derivado suave entre S0 y S3 | 6U/6L CST independiente en el contrato completo |
| `S2` | `C2 = 30.139 + delta_c0_m` | derivado suave entre S0 y S3 | 6U/6L CST independiente |
| `S3` | `C3 = 13.927 + delta_c3_m` | `zle3`, `zte3` fijos en el espacio CFD | 6U/6L CST independiente |
| `S4` | `C4 = taper_ratio_midwing * C3` | twist a 25% cuerda | camber CST + espesor objetivo |
| `S4a` | `(2*C4 + C5)/3` | twist a 25% cuerda | derivado S4/S5 |
| `S4b` | `(C4 + 2*C5)/3` | twist a 25% cuerda | derivado S4/S5 |
| `S5` | `C5 = 0.8 + delta_c5_m` | twist a 25% cuerda | camber CST + espesor objetivo |

El espesor absoluto de borde de salida se conserva al cambiar la cuerda.

## Leyes Geometricas

### Planform

`cta_laws.py` construye el planform desde todos los puntos oficiales de
seccion. La entrada al plano de simetria se fuerza con pendiente nula mediante
`symmetry_blend_y = 1.9`.

Puntos clave:

- LE: `le_spline_bridge=(1, 7)` y tramo exterior lineal desde S4.
- TE: tramo S3-S4 recto con `te_exact_segments=(4,)`.
- TE exterior: lineal desde S4 hasta S5, pasando por S4a y S4b.
- No hay guia artificial del intrados frontal ni post-transform frontal.

### Perfiles

Los perfiles se interpolan por coeficientes CST, igual que en BWB, no por
ordenadas ya resueltas. Esto evita que la seccion intermedia cree "barrigas"
no deseadas.

El transform de espesor exterior conserva:

- camberline,
- espesor absoluto del TE,
- objetivo de `t/c` en la ventana `x/c=[0.15, 0.65]`.

### Z / Twist

S0-S3 usan una transicion quintica C2 de `zle` y `zte`; el twist central se
deriva de `(zte - zle) / chord`. S4-S5 se gobiernan por twist a 25% de cuerda.

### Discretizacion

La geometria resuelta baseline usa:

```text
n_span = 71
n_chord = 241
quads por superficie = 16800
```

El mesh resuelto es la autoridad para solvers:

```text
outputs/cta_geometry_export/cta_resolved_mesh.npz
```

El IGES es un adaptador de inspeccion/exportacion:

```text
outputs/cta_geometry_export/cta.igs
```

### Adapter DUST

DUST no recibe directamente la malla CAD fina de `cta_resolved_mesh.npz`. Para
el metodo de paneles se genera una malla aerodinamica `basic` acondicionada en:

```text
multiads/solvers/aerodynamics/dust_mesh.py
```

El adapter generico `write_basic_two_skin_mesh_from_resolved_npz()`:

- lee `upper_vertices`, `lower_vertices`, `span_stations` y `x_airfoil`;
- remuestrea la geometria a una malla regular de solver;
- mantiene dos pieles upper/lower separadas;
- colapsa el borde de salida a una linea comun para que DUST detecte wake;
- abre numericamente el borde de ataque solo en la malla DUST para evitar
  aristas upper/lower coincidentes y singularidades del sistema de paneles.

Para el baseline CTA actual se usa:

```text
n_span_stations = 33
n_chord_stations = 33
leading_edge_opening_m = 0.05
leading_edge_opening_extent = 0.12
mesh_symmetry = T
```

Esto no modifica la geometria fuente ni el IGES. Es una discretizacion
aerodinamica derivada para DUST/wake.

## Uso

Desde `BWB/`, con el entorno que se esta usando en esta integracion:

```bash
PYTHONDONTWRITEBYTECODE=1 MPLCONFIGDIR=/private/tmp/mads_mpl \
PYTHONPATH=../MADS/src:../MADS/examples \
./.venv/bin/python ../MADS/examples/cta_geometry.py
```

Salidas principales:

```text
MADS/outputs/cta_geometry_export/cta_resolved_mesh.npz
MADS/outputs/cta_geometry_export/cta.igs
MADS/outputs/cta_geometry_export/station_airfoils/
MADS/outputs/cta_geometry_export/summary.json
```

Ejecutar el baseline DoE/check:

```bash
PYTHONDONTWRITEBYTECODE=1 MPLCONFIGDIR=/private/tmp/mads_mpl \
PYTHONPATH=../MADS/src:../MADS/examples \
./.venv/bin/python ../MADS/examples/cta_geom_doe.py
```

Salidas:

```text
MADS/outputs/cta_geom_doe/cta_geom_doe_dataset.csv
MADS/outputs/cta_geom_doe/cta_geom_doe_dataset_flat.csv
MADS/outputs/cta_geom_doe/cta_geom_doe_design_space.csv
MADS/outputs/cta_geom_doe/cta_geom_doe_validation_summary.json
MADS/outputs/cta_geom_doe/cta_geom_doe_manifest.json
```

Ejecutar el baseline aerodinamico DUST CTA para `AoA=[0, 5, 10]`:

```bash
PYTHONDONTWRITEBYTECODE=1 MPLCONFIGDIR=/private/tmp/mads_mpl \
PYTHONPATH=../MADS/src CTA_DUST_BIN_DIR=/path/to/dust-install/bin \
./.venv/bin/python ../MADS/examples/cta_dust_baseline.py
```

Salidas:

```text
MADS/outputs/cta_dust_aoa_sweep/aoa_00/
MADS/outputs/cta_dust_aoa_sweep/aoa_05/
MADS/outputs/cta_dust_aoa_sweep/aoa_10/
MADS/outputs/cta_dust_aoa_sweep/cta_dust_aoa_sweep_results.csv
MADS/outputs/cta_dust_aoa_sweep/cta_dust_aoa_sweep_results.json
```

Este script no es un DoE. Lanza tres casos sueltos del baseline fijo, uno por
angulo de ataque.

## Coherencia Con BWB

La implementacion MADS se ha comparado contra:

```text
BWB/outputs/cta_geometry_official_9/export/cta_resolved_mesh.npz
```

Estado de comparacion:

```text
span_stations: mismo tamano, diferencia max < 5e-10 m
x_airfoil: identico
quads: identicos
upper/lower/camber: misma forma (71, 241, 3)
diferencia maxima de superficie: < 8e-4 m
```

La diferencia residual esta localizada en el borde de salida interpolado entre
secciones, por debajo de 1 mm, y no cambia la topologia ni la discretizacion.

## Como Crear Otro Caso BWB En MADS

Para un caso nuevo:

1. Definir una tabla de secciones oficiales.
2. Separar variables de diseno, parametros derivados y parametros fijos.
3. Crear `Section` + `AirfoilCST` MADS para cada seccion.
4. Declarar las leyes spanwise en un modulo `multiads.cases.<case>_laws`.
5. Resolver a `PreparedGeometry`.
6. Exportar con adaptadores genericos (`cta_resolved_mesh.npz`, pyGeo, IGES,
   DUST futuro).

No colocar logica del solver aerodinamico dentro de la parametrizacion.
