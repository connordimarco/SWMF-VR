#!/bin/bash
#SBATCH --mail-user=cdimarco@umich.edu
#SBATCH --mail-type=END,FAIL
#
# Submit the two steps as a dependent chain:
#   1. seeds (submit_seeds.sh)  -- system python3: flux plots + seed CSVs
#   2. trace (submit_trace.sh) -- pvbatch: field-line OBJ shells + screenshots
# trace waits for seeds (afterok). Output root: $OUT_ROOT (default run/).
#
# Usage: ./submit_all.sh
set -e

if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
    SCRIPT_DIR="$SLURM_SUBMIT_DIR"
else
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
fi

echo "Submitting seeds..."
SEEDS_JOB=$(sbatch --parsable "$SCRIPT_DIR/submit_seeds.sh")
echo "  seeds job: $SEEDS_JOB"

echo "Submitting trace..."
TRACE_JOB=$(sbatch --parsable --dependency=afterok:$SEEDS_JOB "$SCRIPT_DIR/submit_trace.sh")
echo "  trace job: $TRACE_JOB"

echo ""
echo "Submitted:"
echo "  1. seeds:  Job $SEEDS_JOB"
echo "  2. trace: Job $TRACE_JOB (after $SEEDS_JOB)"
echo "Monitor: squeue -u \$USER   Cancel: scancel $SEEDS_JOB $TRACE_JOB"
