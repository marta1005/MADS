#!/usr/bin/env bash
#$ -N cta_panel_10k_doe
#$ -pe py 1
#$ -o cta_panel_10k_doe.out
#$ -e cta_panel_10k_doe.err
#$ -cwd
#$ -l m7j
#$ -q m7gpus

set -euo pipefail

# ---------------------------------------------------------------------------
# CTA Panel DOE — 10 000-sample Sobol campaign
#
# Workflow from MADS repo root:
#
#   qsub scripts/qsub_cta_generate_panel_10k_doe.sh
#   # Wait for it to finish, then:
#   qsub scripts/qsub_cta_dust_panel_10k_shards.sh
#
# ---------------------------------------------------------------------------

DEFAULT_N_SAMPLES=10000
DEFAULT_CASES_PER_WORKER=100
DEFAULT_CAMPAIGN_ROOT="outputs/CTA_case/datasets/campaign_panel_10k_cp"
DEFAULT_SAMPLE_METHOD="sobol"
DEFAULT_SAMPLE_SEED=42

DEFAULT_VENV_DIR=""
DEFAULT_DUST_BIN_DIR="/home/c05279/DUST/bin"
DEFAULT_DUST_LIB_DIR="/home/c05279/DUST/lib"

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    cat <<'EOF'
Usage:
  qsub scripts/qsub_cta_generate_panel_10k_doe.sh

Generates the CTA 14-variable Sobol sample table for a 10 000-case panel campaign.

Defaults:
  N_SAMPLES=10000        (10 000 Sobol samples)
  CASES_PER_WORKER=100   (100 qsub tasks of 100 cases each)
  CAMPAIGN_ROOT=outputs/CTA_case/datasets/campaign_panel_10k
  SAMPLE_METHOD=sobol
  SAMPLE_SEED=42

Override example:
  qsub -v N_SAMPLES=5000,SAMPLE_SEED=7 scripts/qsub_cta_generate_panel_10k_doe.sh

After completion run:
  qsub scripts/qsub_cta_dust_panel_10k_shards.sh
EOF
    exit 0
fi

MADS_ROOT="${MADS_ROOT:-$(pwd)}"
cd "${MADS_ROOT}"

if [[ ! -d "src/multiads" || ! -d "examples" ]]; then
    echo "Run from the MADS repository root, or export MADS_ROOT first." >&2
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
SAMPLES_CSV="${CAMPAIGN_ROOT}/samples/cta_dust_panel_samples.csv"

mkdir -p \
    "${CAMPAIGN_ROOT}/samples" \
    "${CAMPAIGN_ROOT}/shards" \
    "${CAMPAIGN_ROOT}/logs"

RUN_LOG="${CAMPAIGN_ROOT}/logs/generate_doe.log"

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

log "CTA Panel 10k DOE — generate samples"
log "  date            = $(date)"
log "  host            = $(hostname)"
log "  job_id          = ${JOB_ID:-unknown}"
log "  mads_root       = ${MADS_ROOT}"
log "  n_samples       = ${N_SAMPLES}"
log "  cases_per_worker= ${CASES_PER_WORKER}"
log "  campaign_root   = ${CAMPAIGN_ROOT}"
log "  sample_method   = ${SAMPLE_METHOD}"
log "  sample_seed     = ${SAMPLE_SEED}"
log "  samples_csv     = ${SAMPLES_CSV}"

activate_python_env "${VENV_DIR:-}"

if [[ -n "${DUST_BIN_DIR:-}" && -n "${DUST_LIB_DIR:-}" ]]; then
    source scripts/setup_cta_env.sh "${DUST_BIN_DIR}" "${DUST_LIB_DIR}"
elif [[ -n "${DUST_BIN_DIR:-}" ]]; then
    source scripts/setup_cta_env.sh "${DUST_BIN_DIR}"
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

log "Samples ready at ${SAMPLES_CSV}"
log "  n_workers    = ${N_WORKERS}"
log "  qsub_array   = 1-${N_WORKERS}"
log ""
log "Next step:"
log "  qsub scripts/qsub_cta_dust_panel_10k_shards.sh"
