#!/usr/bin/env bash
set -euo pipefail

# Submit with, for example:
#   export DUST_BIN_DIR=/path/to/DUST/bin
#   export DUST_LIB_DIR=/path/to/DUST/lib
#   export VENV_DIR=/path/to/venv
#   export SAMPLES_CSV=outputs/CTA_case/datasets/campaign_001_exploration/samples/cta_dust_vlm_samples.csv
#   export CAMPAIGN_ROOT=outputs/CTA_case/datasets/campaign_001_exploration
#   export CASES_PER_WORKER=50
#   sbatch --array=0-99 scripts/slurm_cta_dust_vlm_shards.sh

#SBATCH --job-name=cta_vlm
#SBATCH --output=outputs/CTA_case/datasets/slurm_logs/cta_vlm_%A_%a.out
#SBATCH --error=outputs/CTA_case/datasets/slurm_logs/cta_vlm_%A_%a.err
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G

MADS_ROOT="${MADS_ROOT:-$(pwd)}"
cd "${MADS_ROOT}"

if [[ -n "${VENV_DIR:-}" ]]; then
    source "${VENV_DIR}/bin/activate"
fi

if [[ -n "${DUST_LIB_DIR:-}" ]]; then
    source scripts/setup_cta_env.sh "${DUST_BIN_DIR:?Set DUST_BIN_DIR to the DUST bin directory}" "${DUST_LIB_DIR}"
else
    source scripts/setup_cta_env.sh "${DUST_BIN_DIR:?Set DUST_BIN_DIR to the DUST bin directory}"
fi

TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
CASES_PER_WORKER="${CASES_PER_WORKER:-50}"
START=$((TASK_ID * CASES_PER_WORKER))
SHARD_ID="$(printf "%05d" "${TASK_ID}")"
CAMPAIGN_ROOT="${CAMPAIGN_ROOT:-outputs/CTA_case/datasets/campaign_001_exploration}"
SAMPLES_CSV="${SAMPLES_CSV:-${CAMPAIGN_ROOT}/samples/cta_dust_vlm_samples.csv}"
SHARD_OUTPUT="${CAMPAIGN_ROOT}/shards/shard_${SHARD_ID}"

mkdir -p "${CAMPAIGN_ROOT}/logs" "${CAMPAIGN_ROOT}/shards"

python examples/cta_dust_doe.py \
    --samples-csv "${SAMPLES_CSV}" \
    --sample-start "${START}" \
    --sample-count "${CASES_PER_WORKER}" \
    --dust-method vlm \
    --n-steps "${N_STEPS:-150}" \
    --mesh-span-stations "${MESH_SPAN_STATIONS:-21}" \
    --mesh-chord-stations "${MESH_CHORD_STATIONS:-21}" \
    --n-threads "${DUST_THREADS:-1}" \
    --no-vtk \
    --output-dir "${SHARD_OUTPUT}" \
    --keep-existing
