#!/usr/bin/env bash
#$ -N cta_panel_10k
#$ -pe py 6
#$ -o cta_panel_10k.out
#$ -e cta_panel_10k.err
# 10000 samples / 100 samples per task = 100 qsub array tasks.
# Change to "#$ -t 1-1" for a smoke test (runs first 100 cases only).
#$ -t 1-100
#$ -cwd
#$ -l m7j
#$ -q m7gpus

set -euo pipefail

# ---------------------------------------------------------------------------
# CTA DUST Panel Method — 10 000-case DOE worker
#
# Generate the sample table first:
#   qsub scripts/qsub_cta_generate_panel_10k_doe.sh
#
# Then launch this array job:
#   qsub scripts/qsub_cta_dust_panel_10k_shards.sh
#
# Smoke test (first 100 cases):
#   Change "#$ -t 1-100" to "#$ -t 1-1" above and resubmit.
#
# Settings:
#   method  = panels
#   mesh    = 49 × 45
#   steps   = 70  (converged from panel convergence study)
#   AoA     = 3°, M = 0.8, FL400
# ---------------------------------------------------------------------------

DEFAULT_CASES_PER_WORKER=100
DEFAULT_CAMPAIGN_ROOT="outputs/CTA_case/datasets/campaign_panel_10k"
DEFAULT_SAMPLES_CSV="${DEFAULT_CAMPAIGN_ROOT}/samples/cta_dust_panel_samples.csv"
DEFAULT_N_STEPS=70
DEFAULT_DUST_THREADS=6
DEFAULT_MESH_SPAN_STATIONS=49
DEFAULT_MESH_CHORD_STATIONS=45
DEFAULT_WRITE_VTK=1
DEFAULT_STORE_CASE_DIRECTORIES=1
DEFAULT_POLAR_MACH=0.4
DEFAULT_POLAR_CD_MIN=0.006

DEFAULT_VENV_DIR=""
DEFAULT_DUST_BIN_DIR="/home/c05279/DUST/bin"
DEFAULT_DUST_LIB_DIR="/home/c05279/DUST/lib"

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    cat <<'EOF'
Usage:
  qsub scripts/qsub_cta_dust_panel_10k_shards.sh

CTA DUST Panel Method — 10 000-case DOE (100 tasks × 100 cases).

Generate DOE samples first:
  qsub scripts/qsub_cta_generate_panel_10k_doe.sh

Campaign defaults:
  total samples       = 10 000
  qsub array          = 1-100
  cases per task      = 100
  method              = panels
  mesh                = 49 × 45
  steps               = 70
  DUST threads/task   = 6
  VTK                 = off

Smoke test — edit this file and change:
  #$ -t 1-100
to:
  #$ -t 1-1

Override example:
  qsub -v CASES_PER_WORKER=50,N_STEPS=70 scripts/qsub_cta_dust_panel_10k_shards.sh

Outputs per shard:
  <CAMPAIGN_ROOT>/shards/shard_XXXXX/cta_dust_doe_dataset_flat.csv

Merge all shards after completion:
  python examples/cta_merge_dust_shards.py \
      --campaign-root outputs/CTA_case/datasets/campaign_panel_10k
EOF
    exit 0
fi

MADS_ROOT="${MADS_ROOT:-$(pwd)}"
cd "${MADS_ROOT}"

if [[ ! -d "src/multiads" || ! -d "examples" ]]; then
    echo "Run from the MADS repository root, or export MADS_ROOT first." >&2
    exit 2
fi

RAW_TASK_ID="${SGE_TASK_ID:-${PBS_ARRAYID:-${TASK_ID:-1}}}"
TASK_ID=$((RAW_TASK_ID - 1))
[[ "${TASK_ID}" -lt 0 ]] && TASK_ID=0

CASES_PER_WORKER="${CASES_PER_WORKER:-${DEFAULT_CASES_PER_WORKER}}"
START=$((TASK_ID * CASES_PER_WORKER))
SHARD_ID="$(printf "%05d" "${TASK_ID}")"
CAMPAIGN_ROOT="${CAMPAIGN_ROOT:-${DEFAULT_CAMPAIGN_ROOT}}"
SAMPLES_CSV="${SAMPLES_CSV:-${DEFAULT_SAMPLES_CSV}}"
SHARD_OUTPUT="${CAMPAIGN_ROOT}/shards/shard_${SHARD_ID}"
VENV_DIR="${VENV_DIR:-${DEFAULT_VENV_DIR}}"
DUST_BIN_DIR="${DUST_BIN_DIR:-${DEFAULT_DUST_BIN_DIR}}"
DUST_LIB_DIR="${DUST_LIB_DIR:-${DEFAULT_DUST_LIB_DIR}}"
N_STEPS="${N_STEPS:-${DEFAULT_N_STEPS}}"
DUST_THREADS="${DUST_THREADS:-${DEFAULT_DUST_THREADS}}"
MESH_SPAN_STATIONS="${MESH_SPAN_STATIONS:-${DEFAULT_MESH_SPAN_STATIONS}}"
MESH_CHORD_STATIONS="${MESH_CHORD_STATIONS:-${DEFAULT_MESH_CHORD_STATIONS}}"
WRITE_VTK="${WRITE_VTK:-${DEFAULT_WRITE_VTK}}"
STORE_CASE_DIRECTORIES="${STORE_CASE_DIRECTORIES:-${DEFAULT_STORE_CASE_DIRECTORIES}}"
POLAR_MACH="${POLAR_MACH:-${DEFAULT_POLAR_MACH}}"
POLAR_CD_MIN="${POLAR_CD_MIN:-${DEFAULT_POLAR_CD_MIN}}"

mkdir -p "${CAMPAIGN_ROOT}/logs" "${CAMPAIGN_ROOT}/shards"
RUN_LOG="${CAMPAIGN_ROOT}/logs/shard_${SHARD_ID}.log"

log()       { echo "$*" | tee -a "${RUN_LOG}"; }
log_error() { echo "$*" | tee -a "${RUN_LOG}" >&2; }

activate_python_env() {
    local env_path="$1"
    [[ -z "${env_path}" ]] && return 0
    if [[ -f "${env_path}" ]]; then
        source "${env_path}"
    elif [[ -f "${env_path}/bin/activate" ]]; then
        source "${env_path}/bin/activate"
    else
        log_error "ERROR: VENV_DIR not valid: ${env_path}"; exit 2
    fi
}

log "CTA DUST Panel 10k — shard started"
log "  date                 = $(date)"
log "  host                 = $(hostname)"
log "  job_id               = ${JOB_ID:-unknown}"
log "  raw_task_id          = ${RAW_TASK_ID}"
log "  zero_based_task_id   = ${TASK_ID}"
log "  ns_slots             = ${NSLOTS:-1}"
log "  mads_root            = ${MADS_ROOT}"
log "  samples_csv          = ${SAMPLES_CSV}"
log "  sample_start         = ${START}"
log "  sample_count         = ${CASES_PER_WORKER}"
log "  dust_method          = panels"
log "  n_steps              = ${N_STEPS}"
log "  dust_threads         = ${DUST_THREADS}"
log "  mesh_span_stations   = ${MESH_SPAN_STATIONS}"
log "  mesh_chord_stations  = ${MESH_CHORD_STATIONS}"
log "  write_vtk            = ${WRITE_VTK}"
log "  polar_mach           = ${POLAR_MACH}"
log "  polar_cd_min         = ${POLAR_CD_MIN}"
log "  shard_output         = ${SHARD_OUTPUT}"

if [[ ! -f "${SAMPLES_CSV}" ]]; then
    log_error "ERROR: samples CSV not found: ${SAMPLES_CSV}"
    log_error "Generate it first with: qsub scripts/qsub_cta_generate_panel_10k_doe.sh"
    exit 2
fi

activate_python_env "${VENV_DIR:-}"

if [[ -n "${DUST_BIN_DIR:-}" && -n "${DUST_LIB_DIR:-}" ]]; then
    source scripts/setup_cta_env.sh "${DUST_BIN_DIR}" "${DUST_LIB_DIR}"
elif [[ -n "${DUST_BIN_DIR:-}" ]]; then
    source scripts/setup_cta_env.sh "${DUST_BIN_DIR}"
else
    source scripts/setup_cta_env.sh
fi

NO_VTK_ARG=""
[[ "${WRITE_VTK}" != "1" ]] && NO_VTK_ARG="--no-vtk"

STORE_CASES_ARG=""
[[ "${STORE_CASE_DIRECTORIES}" == "1" ]] && STORE_CASES_ARG="--store-case-directories"

python examples/cta_dust_doe.py \
    --samples-csv "${SAMPLES_CSV}" \
    --sample-start "${START}" \
    --sample-count "${CASES_PER_WORKER}" \
    --dust-method panels \
    --n-steps "${N_STEPS}" \
    --mesh-span-stations "${MESH_SPAN_STATIONS}" \
    --mesh-chord-stations "${MESH_CHORD_STATIONS}" \
    --n-threads "${DUST_THREADS}" \
    --polar-mach "${POLAR_MACH}" \
    --polar-cd-min "${POLAR_CD_MIN}" \
    ${NO_VTK_ARG} \
    ${STORE_CASES_ARG} \
    --output-dir "${SHARD_OUTPUT}" \
    --keep-existing 2>&1 | tee -a "${RUN_LOG}"

log "Shard finished: $(date)"
