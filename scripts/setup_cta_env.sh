# CTA/MADS runtime environment helper.
#
# Usage from the MADS repository root:
#
#   source scripts/setup_cta_env.sh <DUST_INSTALL_DIR>/bin
#
# The script must be sourced, not executed, so the exported variables remain
# available in the current shell.

if [ -z "${MADS_ROOT:-}" ]; then
    if [ -d "$PWD/src/multiads" ] && [ -d "$PWD/examples" ]; then
        export MADS_ROOT="$PWD"
    else
        echo "setup_cta_env.sh: run from the MADS root or export MADS_ROOT first." >&2
        return 1 2>/dev/null || exit 1
    fi
fi

export PYTHONDONTWRITEBYTECODE=1
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/mads_mpl}"
mkdir -p "$MPLCONFIGDIR" 2>/dev/null || true

case ":${PYTHONPATH:-}:" in
    *":$MADS_ROOT/src:"*) ;;
    *)
        if [ -n "${PYTHONPATH:-}" ]; then
            export PYTHONPATH="$MADS_ROOT/src:$MADS_ROOT/examples:$PYTHONPATH"
        else
            export PYTHONPATH="$MADS_ROOT/src:$MADS_ROOT/examples"
        fi
        ;;
esac

if [ -n "${1:-}" ]; then
    export MADS_DUST_BIN_DIR="$1"
fi

if [ -n "${MADS_DUST_BIN_DIR:-}" ]; then
    case ":$PATH:" in
        *":$MADS_DUST_BIN_DIR:"*) ;;
        *) export PATH="$MADS_DUST_BIN_DIR:$PATH" ;;
    esac
fi

# Good default for local/serial geometry and DUST checks on clusters where the
# default Intel MPI fabric points to an unavailable OFA provider.
export I_MPI_FABRICS="${I_MPI_FABRICS:-shm}"

if [ "${MADS_CLEAR_OFI_PROVIDERS:-1}" = "1" ]; then
    unset FI_PROVIDER
    unset I_MPI_OFI_PROVIDER
fi

echo "CTA/MADS environment configured"
echo "  MADS_ROOT=${MADS_ROOT}"
echo "  PYTHONPATH=${PYTHONPATH}"
echo "  MPLCONFIGDIR=${MPLCONFIGDIR}"
echo "  I_MPI_FABRICS=${I_MPI_FABRICS}"
if [ -n "${MADS_DUST_BIN_DIR:-}" ]; then
    echo "  MADS_DUST_BIN_DIR=${MADS_DUST_BIN_DIR}"
else
    echo "  MADS_DUST_BIN_DIR is not set; DUST will be searched in PATH or --dust-bin-dir."
fi
