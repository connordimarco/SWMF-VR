#!/bin/bash
# Submit the belt-shell pipeline as a dependent chain (run on the login node):
#   1. seeds (submit_seeds.sh)  -- system python3: flux plots + seed CSVs
#   2. trace (submit_trace.sh)  -- pvbatch: field-line OBJ shells + screenshots
# trace waits for seeds (afterok). Reads .env via env.sh: account/partition come
# through SBATCH_*, chdir/mail are passed on the sbatch line.
#
#   ./slurm/submit_all.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$SCRIPT_DIR/slurm/env.sh"

common=(--chdir="$SWMF_VR_ROOT")
[[ -n "${SWMFVR_MAIL:-}" ]] && common+=(--mail-user="$SWMFVR_MAIL")

echo "Submitting seeds..."
SEEDS_JOB=$(sbatch --parsable "${common[@]}" "$SWMF_VR_ROOT/slurm/submit_seeds.sh")
echo "  seeds job: $SEEDS_JOB"

echo "Submitting trace..."
TRACE_JOB=$(sbatch --parsable "${common[@]}" --dependency=afterok:"$SEEDS_JOB" \
            "$SWMF_VR_ROOT/slurm/submit_trace.sh")
echo "  trace job: $TRACE_JOB"

echo
echo "Submitted:  seeds $SEEDS_JOB  ->  trace $TRACE_JOB"
echo "Monitor: squeue -u \$USER   Cancel: scancel $SEEDS_JOB $TRACE_JOB"
