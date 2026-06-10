#!/usr/bin/env bash
#$ -N cta_doe
#$ -pe py 1
#$ -o cta_doe.out
#$ -e cta_doe.err
#$ -cwd
# Office cluster selector: this queue maps to the RHEL8-capable mcn320/mcn321 nodes.
# Keep this active when the Python/DUST environment is only available there.
#$ -l m7j
#$ -q m7gpus

set -euo pipefail

# ---------------------------------------------------------------------------
# User configuration
# ---------------------------------------------------------------------------
# Edit this block once for the office cluster. Every value can still be
# overridden from qsub with -v, but the normal workflow is simply:
#
#   qsub scripts/qsub_cta_generate_doe.sh
#
DEFAULT_N_SAMPLES=5000
DEFAULT_CASES_PER_WORKER=50
DEFAULT_CAMPAIGN_ROOT="outputs/CTA_case/datasets/campaign_001_exploration"
DEFAULT_SAMPLE_METHOD="sobol"
DEFAULT_SAMPLE_SEED=17

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
  qsub scripts/qsub_cta_generate_doe.sh

This Grid Engine / SGE job generates the fixed CTA 14-variable DOE sample CSV.
It does not run DUST. After this job finishes, launch the DUST-VLM array with
scripts/qsub_cta_dust_vlm_shards.sh.

Recommended workflow:

  1. Edit the "User configuration" block inside this script.
  2. Launch:

       qsub scripts/qsub_cta_generate_doe.sh

Options can still be overridden with qsub -v, for example:

  qsub -v N_SAMPLES=5000,CASES_PER_WORKER=50,VENV_DIR=<VENV_DIR> \
    scripts/qsub_cta_generate_doe.sh

Optional variables:
  N_SAMPLES           Total number of DOE samples. Default: 5000
  CASES_PER_WORKER    Samples per DUST array task. Default: 50
  CAMPAIGN_ROOT       Campaign output root.
  SAMPLE_METHOD       sobol, lhs, halton or random. Default: sobol
  SAMPLE_SEED         Sampling seed. Default: 17
  VENV_DIR            Python virtual environment directory or bin/activate file
  DUST_BIN_DIR        Optional DUST bin directory, only used to configure PATH
  DUST_LIB_DIR        Optional extra DUST/HDF5 library directory

Outputs:
  ${CAMPAIGN_ROOT}/samples/cta_dust_vlm_samples.csv
  ${CAMPAIGN_ROOT}/logs/generate_doe.log
EOF
    exit 0
fi

MADS_ROOT="${MADS_ROOT:-$(pwd)}"
cd "${MADS_ROOT}"

if [[ ! -d "src/multiads" || ! -d "examples" ]]; then
    echo "Run this job from the MADS repository root, or export MADS_ROOT first." >&2
    exit 2
fi

N_SAMPLES="${N_SAMPLES:-${DEFAULT_N_SAMPLES}}"
CASES_PER_WORKER="${CASES_PER_WORKER:-${DEFAULT_CASES_PER_WORKER}}"
CAMPAIGN_ROOT="${CAMPAIGN_ROOT:-${DEFAULT_CAMPAIGN_ROOT}}"
SAMPLE_METHOD="${SAMPLE_METHOD:-${DEFAULT_SAMPLE_METHOD}}"
SAMPLE_SEED="${SAMPLE_SEED:-${DEFAULT_SAMPLE_SEED}}"
VENV_DIR="${VENV_DIR:-${DEFAULT_VENV_DIR}}"
DUST_BIN_DIR="${DUST_BIN_DIR:-${DEFAULT_DUST_BIN_DIR}}"
DUST_LIB_DIR="${DUST_LIB_DIR:-${DEFAULT_DUST_LIB_DIR}}"
SAMPLES_CSV="${CAMPAIGN_ROOT}/samples/cta_dust_vlm_samples.csv"

mkdir -p \
    "${CAMPAIGN_ROOT}/samples" \
    "${CAMPAIGN_ROOT}/shards" \
    "${CAMPAIGN_ROOT}/logs"

RUN_LOG="${CAMPAIGN_ROOT}/logs/generate_doe.log"

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

log "CTA DOE qsub job started"
log "  date = $(date)"
log "  host = $(hostname)"
log "  job_id = ${JOB_ID:-unknown}"
log "  mads_root = ${MADS_ROOT}"
log "  n_samples = ${N_SAMPLES}"
log "  cases_per_worker = ${CASES_PER_WORKER}"
log "  campaign_root = ${CAMPAIGN_ROOT}"
log "  sample_method = ${SAMPLE_METHOD}"
log "  sample_seed = ${SAMPLE_SEED}"
log "  samples_csv = ${SAMPLES_CSV}"
log "  run_log = ${RUN_LOG}"

activate_python_env "${VENV_DIR:-}"

if [[ -n "${DUST_BIN_DIR:-}" && -n "${DUST_LIB_DIR:-}" ]]; then
    source scripts/setup_cta_env.sh "${DUST_BIN_DIR}" "${DUST_LIB_DIR}"
elif [[ -n "${DUST_BIN_DIR:-}" ]]; then
    source scripts/setup_cta_env.sh "${DUST_BIN_DIR:?Set DUST_BIN_DIR to the DUST bin directory}"
else
    source scripts/setup_cta_env.sh
fi

python examples/cta_dust_doe.py \
    --generate-samples-only \
    --sample-method "${SAMPLE_METHOD}" \
    --n-samples "${N_SAMPLES}" \
    --sample-seed "${SAMPLE_SEED}" \
    --samples-csv "${SAMPLES_CSV}" 2>&1 | tee -a "${RUN_LOG}"

N_WORKERS=$(( (N_SAMPLES + CASES_PER_WORKER - 1) / CASES_PER_WORKER ))

log "CTA DOE samples ready"
log "  n_workers = ${N_WORKERS}"
log "  qsub_array = 1-${N_WORKERS}"
log ""
log "Next qsub command:"
log "  qsub -t 1-${N_WORKERS} scripts/qsub_cta_dust_vlm_shards.sh"
