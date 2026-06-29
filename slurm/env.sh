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

# Machine config (paths, account): your .env if you made one, else the committed
# .env.example template, else the reference fallbacks below.
if [[ -f "$SWMF_VR_ROOT/.env" ]]; then
    source "$SWMF_VR_ROOT/.env"
elif [[ -f "$SWMF_VR_ROOT/.env.example" ]]; then
    source "$SWMF_VR_ROOT/.env.example"   # template fallback (placeholder paths)
fi

export PYTHONPATH="$SWMF_VR_ROOT/src:${PYTHONPATH:-}"
