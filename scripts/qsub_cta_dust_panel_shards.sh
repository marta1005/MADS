#!/usr/bin/env bash
#$ -N cta_panel
#$ -pe py 6
#$ -o cta_panel.out
#$ -e cta_panel.err
# 60000 samples / 50 samples per qsub task = 1200 qsub array tasks.
# Change to "#$ -t 1-1" for a smoke test.
#$ -t 1-1200
#$ -cwd
# Office cluster selector: this queue maps to the RHEL8-capable mcn320/mcn321 nodes.
# Keep this active because DUST/Python must run on those nodes.
#$ -l m7j
#$ -q m7gpus

set -euo pipefail

# ---------------------------------------------------------------------------
# User configuration
# ---------------------------------------------------------------------------
# Normal workflow from the MADS repository root:
#
#   qsub scripts/qsub_cta_generate_panel_doe.sh
#   qsub scripts/qsub_cta_dust_panel_shards.sh
#
# The campaign defaults below match the Panel Method convergence setup:
#   method = panels
#   mesh   = 49 x 45
#   steps  = 70
#   AoA    = 3 deg
#
DEFAULT_CASES_PER_WORKER=50
DEFAULT_CAMPAIGN_ROOT="outputs/CTA_case/datasets/campaign_002_panel_60k"
DEFAULT_SAMPLES_CSV="${DEFAULT_CAMPAIGN_ROOT}/samples/cta_dust_panel_samples.csv"
DEFAULT_N_STEPS=70
DEFAULT_DUST_THREADS=6
DEFAULT_MESH_SPAN_STATIONS=49
DEFAULT_MESH_CHORD_STATIONS=45
DEFAULT_WRITE_VTK=0
DEFAULT_STORE_CASE_DIRECTORIES=0

# Leave empty if the qsub environment already activates the right Python/DUST.
# DEFAULT_VENV_DIR accepts either the environment directory or the activate file:
#   /path/to/env
#   /path/to/env/bin/activate
DEFAULT_VENV_DIR=""
DEFAULT_DUST_BIN_DIR="/home/c05279/DUST/bin"
DEFAULT_DUST_LIB_DIR="/home/c05279/DUST/lib"

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    cat <<'EOF'
Usage:
  qsub scripts/qsub_cta_dust_panel_shards.sh

This is the CTA DUST Panel Method Grid Engine / SGE qsub worker.
Generate the DOE first with:

  qsub scripts/qsub_cta_generate_panel_doe.sh

Default campaign:
  total samples       = 60000
  qsub array          = 1-1200
  cases per task      = 50
  method              = panels
  mesh                = 49 x 45
  steps               = 70
  DUST threads/task   = 6
  VTK                 = off

Smoke test:
  Edit this file and change:
    #$ -t 1-1200
  to:
    #$ -t 1-1

Optional qsub overrides:
  qsub -v CASES_PER_WORKER=25,N_STEPS=70,DUST_THREADS=6 scripts/qsub_cta_dust_panel_shards.sh

Required files:
  <CAMPAIGN_ROOT>/samples/cta_dust_panel_samples.csv

Outputs:
  <CAMPAIGN_ROOT>/shards/shard_XXXXX/cta_dust_doe_dataset_flat.csv
  <CAMPAIGN_ROOT>/logs/shard_XXXXX.log
EOF
    exit 0
fi

MADS_ROOT="${MADS_ROOT:-$(pwd)}"
cd "${MADS_ROOT}"

if [[ ! -d "src/multiads" || ! -d "examples" ]]; then
    echo "Run this job from the MADS repository root, or export MADS_ROOT first." >&2
    exit 2
fi

RAW_TASK_ID="${SGE_TASK_ID:-${PBS_ARRAYID:-${TASK_ID:-1}}}"
TASK_ID=$((RAW_TASK_ID - 1))
if [[ "${TASK_ID}" -lt 0 ]]; then
    TASK_ID=0
fi

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

mkdir -p "${CAMPAIGN_ROOT}/logs" "${CAMPAIGN_ROOT}/shards"
RUN_LOG="${CAMPAIGN_ROOT}/logs/shard_${SHARD_ID}.log"

log() {
    echo "$*" | tee -a "${RUN_LOG}"
}

log_error() {
    echo "$*" | tee -a "${RUN_LOG}" >&2
}

activate_python_env() {
    local env_path="$1"

    if [[ -z "${env_path}" ]]; then
        return 0
    fi

    if [[ -f "${env_path}" ]]; then
        log "Activating virtual environment file=${env_path}"
        source "${env_path}"
    elif [[ -f "${env_path}/bin/activate" ]]; then
        log "Activating virtual environment dir=${env_path}"
        source "${env_path}/bin/activate"
    else
        log_error "ERROR: VENV_DIR does not point to a venv directory or activate file: ${env_path}"
        exit 2
    fi
}

log "CTA DUST Panel qsub shard started"
log "  date = $(date)"
log "  host = $(hostname)"
log "  job_id = ${JOB_ID:-unknown}"
log "  raw_task_id = ${RAW_TASK_ID}"
log "  zero_based_task_id = ${TASK_ID}"
log "  ns_slots = ${NSLOTS:-1}"
log "  mads_root = ${MADS_ROOT}"
log "  samples_csv = ${SAMPLES_CSV}"
log "  sample_start = ${START}"
log "  sample_count = ${CASES_PER_WORKER}"
log "  dust_method = panels"
log "  n_steps = ${N_STEPS}"
log "  dust_threads = ${DUST_THREADS}"
log "  mesh_span_stations = ${MESH_SPAN_STATIONS}"
log "  mesh_chord_stations = ${MESH_CHORD_STATIONS}"
log "  write_vtk = ${WRITE_VTK}"
log "  store_case_directories = ${STORE_CASE_DIRECTORIES}"
log "  shard_output = ${SHARD_OUTPUT}"
log "  run_log = ${RUN_LOG}"

if [[ ! -f "${SAMPLES_CSV}" ]]; then
    log_error "ERROR: samples CSV not found: ${SAMPLES_CSV}"
    log_error "Generate it first with scripts/qsub_cta_generate_panel_doe.sh."
    exit 2
fi

activate_python_env "${VENV_DIR:-}"

if [[ -n "${DUST_BIN_DIR:-}" && -n "${DUST_LIB_DIR:-}" ]]; then
    source scripts/setup_cta_env.sh "${DUST_BIN_DIR}" "${DUST_LIB_DIR}"
elif [[ -n "${DUST_BIN_DIR:-}" ]]; then
    source scripts/setup_cta_env.sh "${DUST_BIN_DIR:?Set DUST_BIN_DIR to the DUST bin directory}"
else
    source scripts/setup_cta_env.sh
fi

NO_VTK_ARG=""
if [[ "${WRITE_VTK}" != "1" ]]; then
    NO_VTK_ARG="--no-vtk"
fi

STORE_CASES_ARG=""
if [[ "${STORE_CASE_DIRECTORIES}" == "1" ]]; then
    STORE_CASES_ARG="--store-case-directories"
fi

python examples/cta_dust_doe.py \
    --samples-csv "${SAMPLES_CSV}" \
    --sample-start "${START}" \
    --sample-count "${CASES_PER_WORKER}" \
    --dust-method panels \
    --n-steps "${N_STEPS}" \
    --mesh-span-stations "${MESH_SPAN_STATIONS}" \
    --mesh-chord-stations "${MESH_CHORD_STATIONS}" \
    --n-threads "${DUST_THREADS}" \
    ${NO_VTK_ARG} \
    ${STORE_CASES_ARG} \
    --output-dir "${SHARD_OUTPUT}" \
    --keep-existing 2>&1 | tee -a "${RUN_LOG}"
