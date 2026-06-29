#!/usr/bin/env bash
# Source this before running any swmf_vr script, under either runtime:
#   source slurm/env.sh
#   python3      "$SWMF_VR_ROOT/scripts/seeds.py"  ...   # plain python3
#   "$PV_BATCH"  "$SWMF_VR_ROOT/scripts/trace.py"  ...    # pvbatch
#
# It loads .env (machine config), then puts src/ on PYTHONPATH so `import swmf_vr`
# resolves under BOTH system python3 and pvbatch. Sourcing it before `sbatch` also
# exports SBATCH_ACCOUNT / SBATCH_PARTITION, which sbatch applies automatically --
# so the job scripts carry no machine specifics.

SWMF_VR_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export SWMF_VR_ROOT

# Machine config (paths, account). Falls back to the reference values if absent.
if [[ -f "$SWMF_VR_ROOT/.env" ]]; then
    source "$SWMF_VR_ROOT/.env"
fi
export SWMF_RUN_DIR="${SWMF_RUN_DIR:-/nfs/turbo/coe-tuija/shared/run_mothersday_ne}"
export CIMI_NPZ="${CIMI_NPZ:-/nfs/turbo/coe-tuija/shared/connor_austin/CRA/data/20240511_170000_e_fls.npz}"
export PV_BATCH="${PV_BATCH:-/nfs/turbo/coe-tuija/shared/connor_austin/CRA/tools/ParaView-5.12.1-osmesa-MPI-Linux-Python3.10-x86_64/bin/pvbatch}"

export PYTHONPATH="$SWMF_VR_ROOT/src:${PYTHONPATH:-}"
