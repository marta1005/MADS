#!/usr/bin/env bash
# Source this file from the MADS root to set up the dev environment:
#   source activate.sh

MADS_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export VIRTUAL_ENV="/Users/martaarnabatmartin/.venvs/bwb-parametrization"
export PATH="$VIRTUAL_ENV/bin:$PATH"
export PYTHONPATH="$MADS_ROOT/src:$MADS_ROOT/examples"
export DUST_BIN_DIR="/Users/martaarnabatmartin/Desktop/pruebas/dust_runs/dust-install/bin"

echo "MADS env active:"
echo "  python  = $(which python)"
echo "  PYTHONPATH = $PYTHONPATH"
echo "  DUST_BIN_DIR = $DUST_BIN_DIR"

source $VIRTUAL_ENV/bin/activate