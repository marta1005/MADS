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
| `examples/cta_dust_doe.py` | Campana GEMSEO de 14 variables conectada a DUST mediante el adapter aerodinamico. |
| `src/multiads/cases/cta_laws.py` | Leyes CTA: planform, CST spanwise, Z/twist, sampling. |
| `src/multiads/solvers/synthesis/geometry_lib.py` | Core generico: `PreparedGeometry`, planform, mesh resuelto, metricas. |
| `src/multiads/utilities/pygeo_export.py` | Adaptador pyGeo/IGES desde geometria resuelta. |
| `src/multiads/solvers/aerodynamics/dust_lib.py` | Primitivas DUST y escritura de malla `basic` desde una geometria resuelta. |
| `src/multiads/solvers/aerodynamics/dust_geometry.py` | Adapter DUST desde `PreparedGeometry` o `cta_resolved_mesh.npz`. |
| `src/multiads/utilities/campaign_export.py` | Escritura generica de CSV/XLSX/JSON para campanas; no contiene logica CTA ni DUST. |

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
| `sweep_outwing_deg` | 24.7 | [22.0, 40.0] | Sweep S2 medido en la linea de 25% cuerda S4-S5. |
| `twist_s4_deg` | 0.483 | [-3.517, 4.483] | Twist de S4 alrededor del 25% de cuerda. |
| `twist_s4a_deg` | 1.381 | [-2.619, 5.381] | Twist de S4a alrededor del 25% de cuerda. |
| `twist_s4b_deg` | 2.279 | [-1.721, 6.279] | Twist de S4b alrededor del 25% de cuerda. |
| `twist_s5_deg` | 3.177 | [-0.823, 7.177] | Twist de S5 alrededor del 25% de cuerda. |
| `thickness_s4` | 0.1027 | [0.08, 0.16] | Espesor maximo objetivo de S4. |
| `thickness_s5` | 0.095 | [0.08, 0.13] | Espesor maximo objetivo de S5. |

Los bounds de twist de la outer wing se han definido como `baseline +/- 4 deg`.
Los valores baseline son:

```text
S4  = 0.483 deg
S5  = 3.177 deg
S4a = (2*S4 + S5)/3 = 1.381 deg
S4b = (S4 + 2*S5)/3 = 2.279 deg
```

Por tanto:

```text
twist_s4_deg  = [0.483 - 4, 0.483 + 4]
twist_s4a_deg = [1.381 - 4, 1.381 + 4]
twist_s4b_deg = [2.279 - 4, 2.279 + 4]
twist_s5_deg  = [3.177 - 4, 3.177 + 4]
```

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
- objetivo de `t/c` medido sobre todo el perfil resuelto.

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
multiads/solvers/aerodynamics/dust_lib.py
```

El remuestreador generico `write_basic_two_skin_mesh_from_resolved_npz()`:

- lee `upper_vertices`, `lower_vertices`, `span_stations` y `x_airfoil`;
- remuestrea la geometria a una malla de solver uniforme o adaptativa por
  curvatura;
- en modo adaptativo concentra paneles cerca del morro, LE/TE y zonas de alta
  curvatura, y deja menos paneles donde la geometria es casi lineal;
- mantiene dos pieles upper/lower separadas;
- colapsa el borde de salida a una linea comun para que DUST detecte wake;
- por defecto mantiene el borde de ataque cerrado (`leading_edge_opening_m=0.0`);
- puede abrir numericamente el borde de ataque solo si se pide de forma
  explicita para un experimento numerico.

El adapter `multiads.solvers.aerodynamics.dust_geometry`:

- recibe una `PreparedGeometry` ya resuelta o un `cta_resolved_mesh.npz`;
- escribe una copia del mesh resuelto usada por el caso DUST;
- genera la malla `basic` con spacing adaptativo;
- construye un `Wing` MADS con `WingOptions(mesh_file=..., mesh_file_type="basic")`;
- delega la ejecucion en el solver generico `multiads.solvers.aerodynamics.dust.DUST`;
- usa `dust_lib.py` para configurar `dust_pre`, `dust`, `dust_post`,
  `PostLoads` y `PostViz`;
- normaliza `CL`, `CD` y `CM` con el area y MAC de la muestra resuelta;
- guarda `cta_dust_result.json` dentro de cada caso.

Por tanto, `dust_geometry.py` no es una segunda implementacion de DUST. Es la
capa de adaptacion entre geometria resuelta y el solver DUST existente. La
logica de bajo nivel del solver sigue viviendo en `dust.py` y `dust_lib.py`.

Para el baseline CTA actual se usa:

```text
n_span_stations = 49
n_chord_stations = 45
span_spacing = curvature
chord_spacing = curvature
leading_edge_opening_m = 0.0
mirror_span = True
```

Esto no modifica la geometria fuente ni el IGES. Es una discretizacion
aerodinamica derivada para DUST/wake.

Las cargas integrales DUST se leen en el `reference_tag=0` usado por el caso.
Para la campana actual se guarda y normaliza la carga en esos ejes de
referencia sin proyeccion adicional a ejes viento, porque el objetivo inmediato
es estabilizar la estela y comparar tendencias de geometria.

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

Ejecutar el baseline aerodinamico DUST CTA actual para `AoA=3 deg` con el
mismo workflow de campana:

```bash
PYTHONDONTWRITEBYTECODE=1 MPLCONFIGDIR=/private/tmp/mads_mpl \
PYTHONPATH=../MADS/src:../MADS/examples \
CTA_DUST_BIN_DIR=/path/to/dust-install/bin \
./.venv/bin/python ../MADS/examples/cta_dust_doe.py \
  --baseline-only \
  --alpha-deg 3.0 \
  --n-steps 80 \
  --output-dir ../MADS/outputs/cta_dust_doe_80_baseline_aoa03
```

Salidas:

```text
MADS/outputs/cta_dust_doe_80_baseline_aoa03/cta_dust_doe_dataset_flat.csv
MADS/outputs/cta_dust_doe_80_baseline_aoa03/cta_dust_doe_results.xlsx
MADS/outputs/cta_dust_doe_80_baseline_aoa03/cases/run/
```

`--baseline-only` usa `CustomDOE` con una sola muestra baseline. Asi se valida
la misma cadena que usara la campana real, pero sin crear un segundo workflow
ni otra estructura de carpetas.

El postpro DUST tambien escribe visualizacion ParaView del ultimo paso:

```text
MADS/outputs/cta_dust_doe_80_baseline_aoa03/cases/run/post/cta_aoa_03_visualization-0080.vtu
MADS/outputs/cta_dust_doe_80_baseline_aoa03/cases/run/post/cta_aoa_03_visualization_wpan-0080.vtu
MADS/outputs/cta_dust_doe_80_baseline_aoa03/cases/run/post/cta_aoa_03_visualization_wpart-0080.vtu
```

Abrir en ParaView el `visualization-0080.vtu` para la superficie y, si se quiere
ver la estela, cargar tambien `visualization_wpan-0080.vtu` y
`visualization_wpart-0080.vtu`.

### Campana CFD Con GEMSEO + DUST

La primera campana CFD usa el espacio de 14 variables:

```text
8 planform:
  cta_delta_c0_m, cta_delta_c3_m, cta_delta_c5_m,
  cta_taper_ratio_midwing, cta_rspan_midwing, cta_span_wing_m,
  cta_sweep_midwing_deg, cta_sweep_outwing_deg

4 twist outer wing:
  cta_twist_s4_deg, cta_twist_s4a_deg,
  cta_twist_s4b_deg, cta_twist_s5_deg

2 outer thickness:
  cta_thickness_s4, cta_thickness_s5
```

Ejecutar una campana LHS de, por ejemplo, 20 muestras a `AoA=3 deg`:

```bash
PYTHONDONTWRITEBYTECODE=1 MPLCONFIGDIR=/private/tmp/mads_mpl \
PYTHONPATH=../MADS/src:../MADS/examples \
CTA_DUST_BIN_DIR=/path/to/dust-install/bin \
./.venv/bin/python ../MADS/examples/cta_dust_doe.py \
  --n-samples 20 \
  --alpha-deg 3.0 \
  --output-dir ../MADS/outputs/cta_dust_doe_20
```

Por defecto esta campana usa `n_steps = 80`. La ejecucion DUST fisica se
sobrescribe en un unico directorio:

```text
MADS/outputs/cta_dust_doe/cases/run/
```

Esto evita almacenar una carpeta DUST completa por muestra. La historia de la
campana se conserva en las tablas CSV/XLSX. Si se quiere guardar una carpeta
por muestra para debug, usar `--store-case-directories`.

Ejecutar solo el baseline dentro del mismo workflow GEMSEO/DUST:

```bash
PYTHONDONTWRITEBYTECODE=1 MPLCONFIGDIR=/private/tmp/mads_mpl \
PYTHONPATH=../MADS/src:../MADS/examples \
CTA_DUST_BIN_DIR=/path/to/dust-install/bin \
./.venv/bin/python ../MADS/examples/cta_dust_doe.py \
  --baseline-only \
  --alpha-deg 3.0
```

Opciones principales:

```text
--n-samples N                 numero de muestras del DOE
--algo LHS                    algoritmo DOE GEMSEO
--alpha-deg 3.0               angulo de ataque DUST
--mesh-span-stations 49       estaciones semiala antes de espejar
--mesh-chord-stations 45      puntos chordwise por piel
--span-spacing curvature      mallado spanwise adaptativo
--chord-spacing curvature     mallado chordwise adaptativo
--leading-edge-opening-m 0.0  borde de ataque cerrado
--n-steps 80                  pasos temporales DUST por muestra
--store-case-directories      guardar sample_XXXX en vez de sobrescribir cases/run
--no-vtk                      no escribir VTK/ParaView para ahorrar I/O
--fail-fast                   parar si una muestra falla
```

Salidas:

```text
MADS/outputs/cta_dust_doe/cta_dust_doe_dataset.csv
MADS/outputs/cta_dust_doe/cta_dust_doe_dataset_flat.csv
MADS/outputs/cta_dust_doe/cta_dust_doe_results.xlsx
MADS/outputs/cta_dust_doe/cta_dust_doe_design_space.csv
MADS/outputs/cta_dust_doe/cta_dust_doe_manifest.json
MADS/outputs/cta_dust_doe/cases/run/
```

El Excel contiene:

```text
results       inputs, checks geometricos/packaging, fuerzas, coeficientes y L/D
design_space  baseline, lower_bound y upper_bound de las 14 variables
run_settings  AoA, Mach, altitud, n_steps y politica de almacenamiento
```

En `results`, las columnas `outputs.cta_box_*` son checks por superficie
indicadora de packaging:

```text
*_fits                  1 si entra, 0 si falla
*_margin_m              margen minimo local
*_footprint_area_m2     area de huella de la superficie indicadora
*_clearance_volume_m3   margen integrado aproximado = mean_margin * footprint_area
```

El fichero de packaging actual define superficies indicadoras `upper/lower`,
no cajas 3D cerradas con volumen nominal propio. Por eso el volumen registrado
por caja es un volumen de margen/clearance, no un volumen objetivo de
optimizacion.

La carpeta `cases/run` contiene solo la ultima simulacion DUST ejecutada:

```text
geometry/cta_resolved_mesh.npz
geometry/cta_basic_rr.dat
geometry/cta_basic_ee.dat
Output/
post/
cta_dust_result.json
```

Igual que en el DoE geometrico, GEMSEO necesita un objetivo tecnico. Aqui se
usa `cta_dust_failure_code` minimizado solo para registrar si la muestra ha
fallado (`0 = OK`, `1 = fallo DUST/geometria`). No es una funcion objetivo
aerodinamica.

### Espesor Outer Wing

Las variables `cta_thickness_s4` y `cta_thickness_s5` son objetivos de espesor
maximo relativo `t/c` para la outer wing:

```text
S4  = cta_thickness_s4
S4a = (2*S4 + S5)/3
S4b = (S4 + 2*S5)/3
S5  = cta_thickness_s5
```

Durante la resolucion de perfiles, MADS calcula la camberline local, separa la
distribucion de espesor y reescala el perfil completo de esa seccion para que
el maximo `t/c` de todo el perfil coincida con ese objetivo. CTA no aplica una
ventana adicional en `x/c` para medir el espesor exterior.

La altura absoluta del borde de salida se mantiene con el contrato baseline. La
ley interpola el espesor de borde de salida en metros y, en cada estacion, pasa
a CST el valor relativo:

```text
te_over_c(y) = te_abs_m(y) / chord_m(y)
```

Por tanto, cambiar la cuerda o el objetivo de `t/c` exterior no escala
libremente el borde de salida. Este contrato es de fabricacion, no una variable
de diseno.

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
